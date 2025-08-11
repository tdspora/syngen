from itertools import chain
from typing import Union, List
from lazy import lazy
from loguru import logger

import numpy as np
import pandas as pd
from scipy.stats import shapiro, kurtosis
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    OneHotEncoder
)
# Removed TensorFlow/Keras dependencies

from syngen.ml.utils import (
    slugify_parameters,
    inverse_dict,
    datetime_to_timestamp,
    timestamp_to_datetime
)

KURTOSIS_THRESHOLD = 50  # threshold for kurtosis to consider extreme outliers


class _CharTokenizer:
    """
    Minimal character-level tokenizer to replace Keras Tokenizer here.
    Provides:
      - fit_on_texts(list[str])
      - texts_to_sequences(list[str]) -> list[list[int]]
      - word_index: dict[str, int] (1-based)
      - word_counts: dict[str, int]
      - inverse_dict: dict[int, str]
    """

    def __init__(self, lower: bool = False, char_level: bool = True):
        self.lower = lower
        self.char_level = char_level
        self.word_index: dict[str, int] = {}
        self.word_counts: dict[str, int] = {}
        self.inverse_dict: dict[int, str] = {}

    def fit_on_texts(self, texts: List[str]):
        index = 1
        for t in texts:
            if t is None:
                t = ""
            if isinstance(t, bytes):
                try:
                    t = t.decode("utf-8", errors="ignore")
                except Exception:
                    t = ""
            s = str(t)
            if self.lower:
                s = s.lower()
            for ch in s:
                self.word_counts[ch] = self.word_counts.get(ch, 0) + 1
                if ch not in self.word_index:
                    self.word_index[ch] = index
                    index += 1
        self.inverse_dict = {v: k for k, v in self.word_index.items()}

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences: List[List[int]] = []
        for t in texts:
            if t is None:
                t = ""
            if isinstance(t, bytes):
                try:
                    t = t.decode("utf-8", errors="ignore")
                except Exception:
                    t = ""
            s = str(t)
            if self.lower:
                s = s.lower()
            seq = [self.word_index.get(ch, 0) for ch in s]
            sequences.append(seq)
        return sequences

def _pad_sequences(sequences: List[List[int]], maxlen: int, value: int = 0) -> np.ndarray:
    arr = np.full((len(sequences), maxlen), fill_value=value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        trunc = seq[:maxlen]
        arr[i, : len(trunc)] = trunc
    return arr

def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

class BaseFeature:
    """
    Base class for feature classes.
    Each feature class implements feature preprocessing, transformation and inverse transformation.
    What is more, each feature class contains modules for the neural network (NN), including
    corresponding input, encoder, decoder, and loss
    """

    def __init__(self, name):
        self.name: str = self._reset_name(name=name)
        self.original_name: str = name
        self.weight: float = 1.0

    @staticmethod
    @slugify_parameters(turn_on=False)
    def _reset_name(name):
        """
        Slugify the attribute 'name' of the instance
        """
        return name

    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Fit scalars, one-hot encoders, and mappers in the data
        """
        pass

    def transform(self, data: pd.DataFrame) -> List:
        """
        Transform feature to numeric format according to the fitted preprocessors
        """
        pass

    def inverse_transform(self, data: List) -> np.ndarray:
        """
        Inverse transform feature from numeric to original format to obtain an original-like table
        """
        pass

    def input(self):
        """
        Define a feature-specific input for the NN
        """
        pass

    def encoder(self):
        """
        Define a feature-specific encoder for the NN
        """
        pass

    def __decoder_layer(self):
        """
        Define an elementary layer for decoder to use in create_decoder() method
        """
        pass

    def create_decoder(self, encoder_output):
        """
        Create a feature-specific decoder combining given decoder layers and encoder outputs
        """
        pass

    def loss(self):
        """
        Define a feature-specific loss taking into account the data types
        """
        pass


class BinaryFeature(BaseFeature):
    """
    A class to process binary features, i.e. features containing only two unique values
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.feature_type = "categorical"

    def fit(self, data: pd.DataFrame, **kwargs):
        self.mapping = {k: n for n, k in enumerate(np.unique(data))}
        self.inverse_mapping = inverse_dict(self.mapping)
        self.inverse_vectorizer = np.vectorize(self.inverse_mapping.get, otypes=[object])
        self.input_dimension = data.shape[1]

    def transform(self, data: pd.DataFrame):
        data = data.astype("object").replace(self.mapping)
        data = data.replace(self.mapping)
        return data.astype("float32")

    def inverse_transform(self, data: List) -> np.ndarray:
        # Clamp to [0,1] to avoid out-of-range indices after generation
        data = np.clip(data, 0, 1)
        data = np.round(data).astype(int)
        inversed = self.inverse_vectorizer(data)
        return np.where(inversed == "?", None, inversed)

    @lazy
    def input(self):
        return (self.input_dimension,)

    @lazy
    def encoder(self):
        return self.input

    @lazy
    def __decoder_layer(self):
        return ("linear", self.input_dimension)

    def create_decoder(self, encoder_output):
        # Placeholder; PyTorch model handles decoding
        self.decoder = encoder_output
        return self.decoder

    @lazy
    def loss(self):
        return None


class ContinuousFeature(BaseFeature):
    """
    A class to process the continuous numeric features, including floats and integers
    """

    def __init__(
        self,
        name: str,
        decoder_layers: Union[None, tuple, int] = (60,),
        weight_randomizer: Union[None, bool, tuple] = None,
        column_type=float,
    ):
        super().__init__(name=name)
        if decoder_layers is None:
            decoder_layers = ()
        elif isinstance(decoder_layers, int):
            decoder_layers = (decoder_layers,)

        # TODO: teset features
        if weight_randomizer is None or (
            isinstance(weight_randomizer, bool) and not weight_randomizer
        ):
            weight_randomizer = (1, 1)
        elif isinstance(weight_randomizer, bool) and weight_randomizer:
            weight_randomizer = (0, 1)
        elif isinstance(weight_randomizer, (float, int)):
            weight_randomizer = (weight_randomizer, weight_randomizer)

        self.decoder_layers = decoder_layers
        self.weight_randomizer = weight_randomizer
        self.column_type = column_type
        self.feature_type = "numeric"

    def fit(self, data: pd.DataFrame, **kwargs):
        self.is_positive = (data >= 0).sum().item() >= len(data) * 0.99
        self.scaler = self._select_scaler(data)
        self.scaler.fit(data)
        self.input_dimension = data.shape[1]

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(data).astype("float32")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        reverse_transformed = self.scaler.inverse_transform(data)
        reverse_transformed = (
            np.abs(reverse_transformed) if self.is_positive else reverse_transformed
        )
        return (
            reverse_transformed
            if self.column_type is float
            else np.around(reverse_transformed).astype("int64")
        )

    def _select_scaler(
            self, data: pd.DataFrame,
            kurtosis_threshold=KURTOSIS_THRESHOLD
    ) -> object:
        """
        Select appropriate scaler based on data characteristics.

        Strategy:
        - Extreme outliers -> QuantileTransformer
        - Normal distribution -> StandardScaler
        - Other cases -> MinMaxScaler
        """
        data = data.iloc[:, 0]

        kurt = kurtosis(data)
        normality = shapiro(data.sample(n=min(len(data), 500))).pvalue

        if normality >= 0.05:
            return StandardScaler()

        # QuantileTransformer for extreme outliers
        if kurt > kurtosis_threshold:
            logger.debug(
                f"Column '{self.name}' has extreme outliers: "
                f"kurtosis={kurt:.2f} > kurtosis_threshold={kurtosis_threshold}. "
                f"Using QuantileTransformer."
            )
            quantile_params = self._get_quantile_transformer_params(
                n_samples=len(data),
                kurt=kurt,
                kurtosis_threshold=kurtosis_threshold
            )
            return QuantileTransformer(**quantile_params)
        else:
            return MinMaxScaler()

    def _get_quantile_transformer_params(
            self, n_samples, kurt, kurtosis_threshold
            ) -> dict:
        """
        Get optimal parameters for QuantileTransformer
        based on data characteristics
        """
        base_quantiles = min(n_samples, 100_000)

        # for distributions with very extreme outliers use more quantiles
        if kurt > 4 * kurtosis_threshold:
            quantile_factor = 1.5
        elif kurt > 2 * kurtosis_threshold:
            quantile_factor = 1.2
        else:
            quantile_factor = 1.0

        n_quantiles = min(int(base_quantiles * quantile_factor), n_samples)

        # supsample did not affect time for fitting and transforming
        subsample = None

        return {
            'n_quantiles': n_quantiles,
            'subsample': subsample,
            'output_distribution': 'normal'
        }

    @lazy
    def input(self):
        return (self.input_dimension,)

    @lazy
    def encoder(self):
        return self.input

    @lazy
    def __decoder_layer(self):
        decoder_layers = list()
        for idx, item in enumerate(self.decoder_layers):
            name = "%s_decoder_%d" % (self.name, idx)
            if isinstance(item, int):
                decoder_layers.append(("linear", item))

            # Keras Layer types removed in PyTorch port

        decoder_layers.append(("linear", self.input_dimension))

        return decoder_layers

    def create_decoder(self, encoder_output):
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        # Placeholder path: actual decoding is done in PyTorch model
        for _ in decoder_layers:
            x = x

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self):
        # Loss is computed explicitly in PyTorch, so this is a placeholder.
        return None


class CategoricalFeature(BaseFeature):
    """
    A class to process categorical values, i.e. values with 2 < unique_values < 50
    """

    def __init__(
        self,
        name: str,
        decoder_layers: Union[None, tuple, int] = (60,),
        weight_randomizer: Union[None, bool, tuple] = None,
    ):
        if decoder_layers is None:
            decoder_layers = ()
        elif isinstance(decoder_layers, int):
            decoder_layers = (decoder_layers,)

        if weight_randomizer is None or (
            isinstance(weight_randomizer, bool) and not weight_randomizer
        ):
            weight_randomizer = (1, 1)
        elif isinstance(weight_randomizer, (float, int)):
            weight_randomizer = (weight_randomizer, weight_randomizer)
        elif isinstance(weight_randomizer, bool) and weight_randomizer:
            weight_randomizer = (0, 1)

        super().__init__(name="_".join(name.split()))

        self.one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            dtype=np.float32)
        self.decoder = None
        self.decoder_layers = decoder_layers
        self.weight_randomizer = weight_randomizer
        self.feature_type = "categorical"

    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Fit the encoder and create mappings.
        """
        data = data.iloc[:, 0].astype(str).to_numpy().reshape(-1, 1)

        self.one_hot_encoder.fit(data)

        categories = self.one_hot_encoder.categories_[0]
        self.mapping = {cat: idx for idx, cat in enumerate(categories)}
        self.inverse_mapping = inverse_dict(self.mapping)
        self.inverse_vectorizer = np.vectorize(self.inverse_mapping.get)

        self.input_dimension = len(self.mapping)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data to one-hot encoding.
        """
        data = data.iloc[:, 0].astype(str).to_numpy().reshape(-1, 1)

        return self.one_hot_encoder.transform(data).astype("float32")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Convert one-hot encoded data back to original categories.
        """
        data = np.asarray(data)
        if data.ndim > 1:
            data = data.argmax(axis=1)
        inversed = self.inverse_vectorizer(data)

        return np.where(inversed == "?", None, inversed)

    @lazy
    def input(self):
        return (self.input_dimension,)

    @lazy
    def encoder(self):
        return self.input

    @lazy
    def __decoder_layer(self):
        decoder_layers = list()
        for idx, item in enumerate(self.decoder_layers):
            name = "%s_decoder_%d" % (self.name, idx)
            if isinstance(item, int):
                decoder_layers.append(("linear", item))
            # Keras Layer types removed

        decoder_layers.append(("linear", self.input_dimension))

        return decoder_layers

    def create_decoder(self, encoder_output):
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        # Placeholder; decoding handled by PyTorch model
        for _ in decoder_layers:
            x = x

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self):
        return None


class CharBasedTextFeature(BaseFeature):
    """
    A class to process the text features
    """

    def __init__(
        self,
        name: str,
        text_max_len: int,
        rnn_units: int = 128,
        dropout: int = 0,
    ):
        super().__init__(name=name)
        self.decoder = None
        self.text_max_len = text_max_len
        self.rnn_units = rnn_units
        # rnn definitions removed in PyTorch port; handled in model
        self.dropout = dropout
        self.feature_type = "text"

    def fit(self, data: pd.DataFrame, **kwargs):
        if len(data.columns) > 1:
            raise Exception("CharBasedTextFeature can work only with one text column")

        data = data[data.columns[0]]

        tokenizer = _CharTokenizer(lower=False, char_level=True)
        tokenizer.fit_on_texts(list(data))
        tokenizer.inverse_dict = inverse_dict(tokenizer.word_index)

        self.vocab_size = len(tokenizer.word_index)
        self.tokenizer = tokenizer

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if len(data.columns) > 1:
            raise Exception("CharBasedTextFeature can work only with one text column")

        data = data[data.columns[0]]

        sequences = self.tokenizer.texts_to_sequences(list(data))
        data_gen = _pad_sequences(sequences, maxlen=self.text_max_len, value=0)
        # one-hot encode to shape (batch, max_len, vocab_size)
        n, L = data_gen.shape
        one_hot = np.zeros((n, L, self.vocab_size), dtype=np.float32)
        # fill one-hot for non-zero token ids
        for i in range(n):
            idxs = data_gen[i]
            for t, token in enumerate(idxs):
                if token > 0 and token <= self.vocab_size:
                    one_hot[i, t, token - 1] = 1.0
        return one_hot

    @staticmethod
    def _top_p_filtering(
            logits: np.ndarray,
            top_p: float = 0.9
    ):
        # logits: (batch, seq_len, vocab)
        probs = _softmax_np(logits, axis=-1)
        # sort descending
        sorted_idx = np.argsort(-probs, axis=-1)
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
        cumprobs = np.cumsum(sorted_probs, axis=-1)
        # mark tokens with cumulative prob >= top_p (except keep first above threshold)
        to_remove = cumprobs >= top_p
        # shift right to keep first token above threshold
        to_remove[..., 1:] = to_remove[..., :-1]
        to_remove[..., 0] = False
        # scatter mask back to vocab order
        mask = np.zeros_like(to_remove)
        np.put_along_axis(mask, sorted_idx, to_remove, axis=-1)
        filtered = np.where(mask, 0.0, probs)
        return filtered.astype(np.float64)

    @staticmethod
    def _top_k_filtering(
            logits: np.ndarray,
            top_k: int = 0
    ):
        if top_k <= 0:
            return logits
        # keep only top_k per last axis
        kth_vals = np.partition(logits, -top_k, axis=-1)[..., -top_k][..., None]
        mask = logits < kth_vals
        out = logits.copy()
        out[mask] = 0.0
        return out

    def _process_batch(self, batch: np.ndarray) -> List[str]:
        probs = _softmax_np(batch, axis=-1).astype(float)
        probs = self._top_p_filtering(probs, top_p=0.9)
        # probs = self._top_k_filtering(probs, top_k=6)
        # TODO: select top_k based on inverse_dict length

        probs /= probs.sum(axis=2, keepdims=True)

        multinomial_samples = np.apply_along_axis(
            lambda x: np.argmax(np.random.multinomial(1, x)), -1, probs
        )

        chars_array = np.vectorize(lambda x: self.tokenizer.inverse_dict.get(x, ''))(
            multinomial_samples)

        # Convert tokens to words
        words = ["".join(sample) for sample in chars_array]

        return words

    def inverse_transform(self, data: np.ndarray, **kwargs) -> List[str]:

        batch_size = 10000
        num_batches = (len(data) + batch_size - 1) // batch_size
        feature_values = []

        for i in range(num_batches):
            batch = data[i * batch_size: (i + 1) * batch_size]
            feature_values.extend(self._process_batch(batch))

        return feature_values

    @lazy
    def input(self):
        return (self.text_max_len, self.vocab_size)

    @lazy
    def encoder(self):
        # Handled by PyTorch model
        return self.input

    @lazy
    def __decoder_layer(self):
        # Placeholder; decoding handled by PyTorch model
        return [("repeat", self.text_max_len), ("rnn", self.rnn_units), ("linear", self.vocab_size)]

    def create_decoder(self, encoder_output):
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        for _ in decoder_layers:
            x = x

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self):
        return None


class EmailFeature(CharBasedTextFeature):
    def __init__(
        self,
        name: str,
        text_max_len: int,
        rnn_units: int = 128,
        dropout: int = 0,
        domain: str = 'tdspora.ai'
    ):
        super().__init__(name=name,
                         text_max_len=text_max_len,
                         rnn_units=rnn_units,
                         dropout=dropout)
        self.domain = domain

    def fit(self, data: pd.DataFrame, **kwargs):
        super().fit(self.extract_email_name(data))

    @staticmethod
    def extract_email_name(data: pd.DataFrame) -> pd.DataFrame:
        pattern = r'^([^@]+).*$'  # returns all the string if there's no "@" in it
        return data.iloc[:, 0].str.extract(pattern).rename({0: data.columns[0]}, axis=1).fillna('')

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if len(data.columns) > 1:
            raise Exception("CharBasedTextFeature can work only with one text column")

        # TODO: add logic for emails to Dataset._preprocess_nan_cols
        return super().transform(self.extract_email_name(data))

    def inverse_transform(self, data: np.ndarray, **kwargs) -> List[str]:
        return [
            s + "@" +
            self.domain for s in super().inverse_transform(data, **kwargs)
        ]


class DateFeature(BaseFeature):
    """
    A class to process datetime features
    """

    def __init__(self, name, decoder_layers=(60,), weight_randomizer=None):
        if decoder_layers is None:
            decoder_layers = ()
        elif isinstance(decoder_layers, int):
            decoder_layers = (decoder_layers,)

        if weight_randomizer is None or (
            isinstance(weight_randomizer, bool) and not weight_randomizer
        ):
            weight_randomizer = (1, 1)
        elif isinstance(weight_randomizer, bool) and weight_randomizer:
            weight_randomizer = (0, 1)
        elif isinstance(weight_randomizer, (float, int)):
            weight_randomizer = (weight_randomizer, weight_randomizer)

        super().__init__(name=name)
        self.decoder_layers = decoder_layers
        self.weight_randomizer = weight_randomizer
        self.feature_type = "numeric"

    def fit(self, data, **kwargs):
        self.date_format = kwargs["date_mapping"][self.original_name]
        self.data = chain.from_iterable(data.values)
        self.data = pd.DataFrame(
            list(
                map(
                    lambda d: datetime_to_timestamp(d, self.date_format),
                    self.data
                )
            )
        )
        self.is_positive = (self.data >= 0).sum().item() >= len(self.data) * 0.99
        normality = shapiro(self.data.sample(n=min(len(self.data), 500))).pvalue
        self.data = np.array(self.data).reshape(-1, 1)

        self.scaler = StandardScaler() if normality >= 0.05 else MinMaxScaler()
        self.scaler.fit(self.data)
        self.input_dimension = self.data.shape[1]

    def transform(self, data):
        return self.scaler.transform(self.data)

    def inverse_transform(self, data):
        unscaled = self.scaler.inverse_transform(data)
        unscaled = chain.from_iterable(unscaled)
        return list(
            map(
                lambda t: timestamp_to_datetime(int(t)).strftime(self.date_format),
                unscaled,
            )
        )

    @lazy
    def input(self):
        return (self.input_dimension,)

    @lazy
    def encoder(self):
        return self.input

    @lazy
    def __decoder_layer(self):
        decoder_layers = list()
        for idx, item in enumerate(self.decoder_layers):
            name = "%s_decoder_%d" % (self.name, idx)
            if isinstance(item, int):
                decoder_layers.append(("linear", item))
        decoder_layers.append(("linear", self.input_dimension))
        return decoder_layers

    def create_decoder(self, encoder_output):
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        # no-op; placeholder
        for _ in decoder_layers:
            x = x

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self):
        return None
