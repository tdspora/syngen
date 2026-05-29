from collections import OrderedDict
from itertools import chain
from typing import Union, List
from loguru import logger

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import shapiro, kurtosis
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    OneHotEncoder
)

from syngen.ml.vae.models.custom_layers import TextEncoder, TextDecoder
from syngen.ml.utils import (
    slugify_parameters,
    inverse_dict,
    datetime_to_timestamp,
    convert_to_date
)

KURTOSIS_THRESHOLD = 50  # threshold for kurtosis to consider extreme outliers
_CE_EPS = 1e-7  # clip for categorical cross-entropy, mirrors keras epsilon


class CharTokenizer:
    """Keras-free char-level tokenizer.

    Replaces ``keras.preprocessing.text.Tokenizer(lower=False, char_level=True)``
    used in ``features.py`` and ``handlers.LongTextsHandler``. For ``char_level``
    Keras ignores ``filters`` and treats every character (spaces and punctuation
    included) as a token; ``word_index`` is assigned by **descending frequency**
    with ties keeping first-appearance order, starting at index 1;
    ``texts_to_sequences`` drops characters absent from ``word_index``.
    """

    def __init__(self, lower: bool = False, char_level: bool = True):
        self.lower = lower
        self.char_level = char_level
        self.word_counts: "OrderedDict[str, int]" = OrderedDict()
        self.word_index: dict = {}
        self.index_word: dict = {}

    def _split(self, text) -> List[str]:
        text = "" if text is None else str(text)
        if self.lower:
            text = text.lower()
        return list(text) if self.char_level else text.split()

    def fit_on_texts(self, texts):
        for text in texts:
            for token in self._split(text):
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
        # Stable sort by count desc; OrderedDict preserves first-appearance order
        # so ties break exactly like Keras.
        ordered = sorted(self.word_counts.items(), key=lambda kv: kv[1], reverse=True)
        self.word_index = {token: idx + 1 for idx, (token, _) in enumerate(ordered)}
        self.index_word = {idx: token for token, idx in self.word_index.items()}

    def texts_to_sequences(self, texts) -> List[List[int]]:
        return [
            [self.word_index[token] for token in self._split(text) if token in self.word_index]
            for text in texts
        ]


def pad_sequences(sequences, maxlen: int, value: float = 0.0) -> np.ndarray:
    """Post-pad/truncate to ``maxlen`` with ``value`` (matches keras
    ``pad_sequences(padding='post', truncating='post')``)."""
    out = np.full((len(sequences), maxlen), value, dtype="int64")
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        trunc = seq[:maxlen]
        out[i, : len(trunc)] = trunc
    return out


def _one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    """Match ``tf.one_hot``: out-of-range indices map to an all-zero vector.

    With ``word_index`` running 1..vocab_size and ``depth == vocab_size``, the
    pad index 0 maps to position 0 and the least-frequent char (index
    ``vocab_size``) is out of range → all zeros. This preserves the exact (quirky)
    TF encoding so generated-text statistics match the baseline.
    """
    indices = np.asarray(indices)
    out = np.zeros((indices.size, depth), dtype="float32")
    flat = indices.reshape(-1)
    valid = (flat >= 0) & (flat < depth)
    rows = np.nonzero(valid)[0]
    out[rows, flat[valid]] = 1.0
    return out.reshape(indices.shape + (depth,))


class BaseFeature:
    """
    Base class for feature classes.
    Each feature class implements feature preprocessing, transformation and inverse transformation.
    What is more, each feature class contributes a PyTorch sub-network to the CVAE
    via ``encoded_dim``/``build_encoder``/``build_decoder_head`` and a loss via
    ``compute_loss``.
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

    # --- PyTorch sub-network contract (was Keras input/encoder/decoder/loss) ---

    @property
    def encoded_dim(self) -> int:
        """Width this feature contributes to the concatenated encoder input."""
        return self.input_dimension

    def build_encoder(self) -> nn.Module:
        """Per-feature encoder piece. Identity for tabular features; the shared
        encoder (``model.py``) does the heavy lifting."""
        return nn.Identity()

    def build_decoder_head(self, in_features: int) -> nn.Module:
        """Map the shared decoder output to this feature's reconstruction."""
        raise NotImplementedError

    def compute_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Per-feature reconstruction loss (scalar)."""
        raise NotImplementedError


def _numeric_decoder_head(decoder_layers, in_features, output_dim, final_activation=None):
    """Build ``[Linear→ReLU]* → Linear(output_dim)[→activation]``.

    Mirrors the Keras ``__decoder_layer`` builders (``features.py`` Continuous/
    Categorical/Date): integer entries become ``Dense(item, relu)``; a trailing
    ``Dense(output_dim, …)`` finishes. Non-int entries are ignored (the TF code's
    ``Layer`` branch was dead — it appended the class, not an instance).
    """
    layers: List[nn.Module] = []
    prev = in_features
    for item in decoder_layers:
        if isinstance(item, int):
            layers += [nn.Linear(prev, item), nn.ReLU()]
            prev = item
    layers.append(nn.Linear(prev, output_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


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
        self.inverse_vectorizer = np.vectorize(self.inverse_mapping.get)
        self.input_dimension = data.shape[1]

    def transform(self, data: pd.DataFrame):
        data = data.astype("object").replace(self.mapping)
        data = data.replace(self.mapping)
        return data.astype("float32")

    def inverse_transform(self, data: List) -> np.ndarray:
        data = np.round(data)
        inversed = self.inverse_vectorizer(data)
        return np.where(inversed == "?", None, inversed)

    def build_decoder_head(self, in_features: int) -> nn.Module:
        # Keras: single Dense(input_dimension, sigmoid), no hidden layer
        # (features.py:142).
        return nn.Sequential(nn.Linear(in_features, self.input_dimension), nn.Sigmoid())

    def compute_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return self.weight * F.binary_cross_entropy(output, target)


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
        self.loss_weight = _sample_loss_weight(weight_randomizer)
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
        - "_null" features and other cases -> MinMaxScaler
        """
        data = data.iloc[:, 0]

        kurt = kurtosis(data)
        normality = shapiro(data.sample(n=min(len(data), 500))).pvalue

        if normality >= 0.05:
            return StandardScaler()

        # Column that ends with "_null"
        # (indicators of nan values of corresponding column)
        # is binary but stored as numeric.
        # It must not be transformed with QuantileTransformer
        if data.nunique() == 2:
            return MinMaxScaler()

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

    def build_decoder_head(self, in_features: int) -> nn.Module:
        return _numeric_decoder_head(self.decoder_layers, in_features, self.input_dimension)

    def compute_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.mse_loss(output, target)


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
        self.decoder_layers = decoder_layers
        self.weight_randomizer = weight_randomizer
        self.loss_weight = _sample_loss_weight(weight_randomizer)
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
        data = data.argmax(axis=1)
        inversed = self.inverse_vectorizer(data)

        return np.where(inversed == "?", None, inversed)

    def build_decoder_head(self, in_features: int) -> nn.Module:
        # Keras: Dense(60, relu) -> Dense(n_cat, softmax) (features.py:411-428).
        return _numeric_decoder_head(
            self.decoder_layers, in_features, self.input_dimension,
            final_activation=nn.Softmax(dim=-1),
        )

    def compute_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # categorical_crossentropy on softmax probabilities (output already softmaxed)
        output = output.clamp(_CE_EPS, 1.0 - _CE_EPS)
        return self.loss_weight * -(target * torch.log(output)).sum(dim=-1).mean()


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
        self.text_max_len = text_max_len
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.feature_type = "text"

    def fit(self, data: pd.DataFrame, **kwargs):
        if len(data.columns) > 1:
            raise Exception("CharBasedTextFeature can work only with one text column")

        data = data[data.columns[0]]

        tokenizer = CharTokenizer(lower=False, char_level=True)
        tokenizer.fit_on_texts(data)
        tokenizer.inverse_dict = inverse_dict(tokenizer.word_index)

        self.vocab_size = len(tokenizer.word_index)
        self.tokenizer = tokenizer

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if len(data.columns) > 1:
            raise Exception("CharBasedTextFeature can work only with one text column")

        data = data[data.columns[0]]

        data_gen = self.tokenizer.texts_to_sequences(data)
        data_gen = pad_sequences(data_gen, maxlen=self.text_max_len, value=0.0)
        return _one_hot(data_gen, self.vocab_size)

    @staticmethod
    def _top_p_filtering(probs: np.ndarray, top_p: float = 0.9) -> np.ndarray:
        """Nucleus filter on a probability tensor ``(…, vocab)``.

        Keeps the smallest set of highest-probability tokens whose cumulative
        mass crosses ``top_p`` (first crossing token retained); the rest are
        zeroed. Faithful numpy port of the former TF implementation.
        """
        probs = np.asarray(probs, dtype=np.float64)
        sorted_idx = np.argsort(-probs, axis=-1)
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
        cumulative = np.cumsum(sorted_probs, axis=-1)
        remove_sorted = cumulative >= top_p
        # shift right so the first token that crosses the threshold is kept
        remove_sorted[..., 1:] = remove_sorted[..., :-1]
        remove_sorted[..., 0] = False
        remove = np.empty_like(remove_sorted)
        np.put_along_axis(remove, sorted_idx, remove_sorted, axis=-1)
        return np.where(remove, 0.0, probs)

    @staticmethod
    def _top_k_filtering(logits: np.ndarray, top_k: int = 0) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if top_k <= 0:
            return logits
        kth = np.sort(logits, axis=-1)[..., -top_k][..., None]
        return np.where(logits < kth, 0.0, logits)

    def _process_batch(self, batch: np.ndarray) -> List[str]:
        # softmax over the vocab axis
        batch = np.asarray(batch, dtype=np.float64)
        shifted = batch - batch.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)

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

    @property
    def encoded_dim(self) -> int:
        return 2 * self.rnn_units  # BiLSTM concatenates both directions

    def build_encoder(self) -> nn.Module:
        return TextEncoder(self.vocab_size, self.rnn_units)

    def build_decoder_head(self, in_features: int) -> nn.Module:
        return TextDecoder(in_features, self.text_max_len, self.rnn_units, self.vocab_size)

    def compute_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # softmax cross-entropy with logits, mean over batch and sequence positions
        log_probs = F.log_softmax(output, dim=-1)
        return self.weight * -(target * log_probs).sum(dim=-1).mean()


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
        self.loss_weight = _sample_loss_weight(weight_randomizer)
        self.feature_type = "numeric"

    def fit(self, data, **kwargs):
        self.date_format = kwargs["date_mapping"].get(self.original_name)
        self.to_datetime_conversion = kwargs["to_datetime_conversion"].get(self.original_name)
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
        return self.scaler.transform(self.data).astype("float32")

    def inverse_transform(self, data):
        unscaled = self.scaler.inverse_transform(data)
        unscaled = chain.from_iterable(unscaled)
        return list(
            map(
                lambda t: convert_to_date(
                    value=t,
                    date_format=self.date_format,
                    to_datetime_conversion=self.to_datetime_conversion
                ),
                unscaled,
            )
        )

    def build_decoder_head(self, in_features: int) -> nn.Module:
        return _numeric_decoder_head(self.decoder_layers, in_features, self.input_dimension)

    def compute_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.mse_loss(output, target)


def _sample_loss_weight(weight_randomizer) -> float:
    """Per-feature loss weight. Default ``(1, 1)`` → constant ``1.0``; matches the
    Keras ``K.random_uniform_variable(low, high)`` that was sampled once at build
    (``features.py:327``). Sampled here at construction time."""
    low, high = weight_randomizer
    if low == high:
        return float(low)
    return float(np.random.uniform(low, high))
