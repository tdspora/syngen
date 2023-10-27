from collections import Counter
from itertools import chain
from typing import Union, List
import re
from lazy import lazy

import category_encoders as ce
import numpy as np
import pandas as pd
from pandas._libs.tslibs.parsing import guess_datetime_format
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import losses
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Input,
    LSTM,
    Layer,
    RepeatVector,
    TimeDistributed,
)

from syngen.ml.utils import (
    slugify_parameters,
    inverse_dict,
    datetime_to_timestamp,
    timestamp_to_datetime,
)


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
    @slugify_parameters()
    def _reset_name(name):
        """
        Slugify the attribute 'name' of the instance
        """
        return name

    def fit(self, data: pd.DataFrame):
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

    def input(self) -> tf.Tensor:
        """
        Define a feature-specific input for the NN
        """
        pass

    def encoder(self) -> tf.Tensor:
        """
        Define a feature-specific encoder for the NN
        """
        pass

    def __decoder_layer(self) -> tf.Tensor:
        """
        Define an elementary layer for decoder to use in create_decoder() method
        """
        pass

    def create_decoder(self, encoder_output: tf.Tensor):
        """
        Create a feature-specific decoder combining given decoder layers and encoder outputs
        """
        pass

    def loss(self) -> tf.Tensor:
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

    def fit(self, data: pd.DataFrame):
        self.mapping = {k: n for n, k in enumerate(np.unique(data))}
        self.inverse_mapping = inverse_dict(self.mapping)
        self.inverse_vectorizer = np.vectorize(self.inverse_mapping.get)
        self.input_dimension = data.shape[1]

    def transform(self, data: pd.DataFrame) -> List:
        data = data.replace(self.mapping)
        return data.astype("float64")

    def inverse_transform(self, data: List) -> np.ndarray:
        data = np.round(data)
        inversed = self.inverse_vectorizer(data)
        return np.where(inversed == "?", None, inversed)

    @lazy
    def input(self) -> tf.Tensor:
        return Input(shape=(self.input_dimension,), name="input_%s" % self.name)

    @lazy
    def encoder(self) -> tf.Tensor:
        return self.input

    @lazy
    def __decoder_layer(self) -> tf.Tensor:
        return Dense(self.input_dimension, activation="sigmoid")

    def create_decoder(self, encoder_output: tf.Tensor) -> tf.Tensor:
        self.decoder = self.__decoder_layer(encoder_output)
        return self.decoder

    @lazy
    def loss(self) -> tf.Tensor:
        if not hasattr(self, "decoder"):
            Exception("Decoder isn't created")

        return self.weight * losses.binary_crossentropy(self.input, self.decoder)


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

    def fit(self, data: pd.DataFrame):
        self.is_positive = (data >= 0).sum().item() >= len(data) * 0.99
        normality = shapiro(data.sample(n=min(len(data), 500))).pvalue
        self.scaler = StandardScaler() if normality >= 0.05 else MinMaxScaler()
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

    @lazy
    def input(self) -> tf.Tensor:
        return Input(shape=(self.input_dimension,), name="input_%s" % self.name)

    @lazy
    def encoder(self) -> tf.Tensor:
        return self.input

    @lazy
    def __decoder_layer(self) -> List[tf.Tensor]:
        decoder_layers = list()
        for idx, item in enumerate(self.decoder_layers):
            name = "%s_decoder_%d" % (self.name, idx)
            if isinstance(item, int):
                decoder_layers.append(Dense(item, activation="relu", name=name))

            if isinstance(item, Layer):
                item.name = name
                decoder_layers.append(Layer)

        decoder_layers.append(
            Dense(self.input_dimension, activation="linear", name="%s_linear" % self.name)
        )

        return decoder_layers

    def create_decoder(self, encoder_output: tf.Tensor) -> tf.Tensor:
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        for layer in decoder_layers:
            x = layer(x)

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self) -> tf.Tensor:
        if not hasattr(self, "decoder"):
            Exception("Decoder isn't created")

        low = self.weight_randomizer[0]
        high = self.weight_randomizer[1]
        random_weight = K.random_uniform_variable(shape=(1,), low=low, high=high)

        return random_weight * tf.keras.losses.MSE(self.input, self.decoder)


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
        self.one_hot_encoder = ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
        self.decoder = None
        self.decoder_layers = decoder_layers
        self.weight_randomizer = weight_randomizer

    def fit(self, data: pd.DataFrame):
        data = data.astype(object)
        self.one_hot_encoder.fit(data)
        self.mapping = {
            k: v for k, v in self.one_hot_encoder.category_mapping[0]["mapping"].items()
        }
        self.inverse_mapping = inverse_dict(self.mapping)
        self.inverse_vectorizer = np.vectorize(self.inverse_mapping.get)

        # because in mapping exist additional class None, input dimensionality should be less on 1
        self.input_dimension = len(self.mapping) - 1

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if isinstance(data, pd.Series):
            data = data.values
        data = np.array(self.one_hot_encoder.transform(data)).astype("float32")
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = (
            data.argmax(axis=1) + 1
        )  # because in array numbers starts from 0, in dict it starts from 1
        inversed = self.inverse_vectorizer(data)
        return np.where(inversed == "?", None, inversed)

    @lazy
    def input(self) -> tf.Tensor:
        self.idx_input = Input(shape=(self.input_dimension,), name="input_%s" % self.name)

        return self.idx_input

    @lazy
    def encoder(self) -> tf.Tensor:
        return self.idx_input

    @lazy
    def __decoder_layer(self) -> List[tf.Tensor]:
        decoder_layers = list()
        for idx, item in enumerate(self.decoder_layers):
            name = "%s_decoder_%d" % (self.name, idx)
            if isinstance(item, int):
                decoder_layers.append(Dense(item, activation="relu", name=name))

            if isinstance(item, Layer):
                item.name = name
                decoder_layers.append(Layer)

        decoder_layers.append(
            Dense(
                self.input_dimension,
                activation="softmax",
                name="%s_softmax" % self.name,
            )
        )

        return decoder_layers

    def create_decoder(self, encoder_output: tf.Tensor) -> tf.Tensor:
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        for layer in decoder_layers:
            x = layer(x)

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self) -> tf.Tensor:
        if not hasattr(self, "decoder"):
            Exception("Decoder isn't created")

        low = self.weight_randomizer[0]
        high = self.weight_randomizer[1]
        random_weight = K.random_uniform_variable(shape=(1,), low=low, high=high)

        return random_weight * tf.keras.losses.categorical_crossentropy(self.input, self.decoder)


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
        self.rnn_unit = LSTM
        self.dropout = dropout

    def fit(self, data: pd.DataFrame):
        from tensorflow.keras.preprocessing.text import Tokenizer

        if len(data.columns) > 1:
            raise Exception("CharBasedTextFeature can work only with one text column")

        data = data[data.columns[0]]

        tokenizer = Tokenizer(lower=False, char_level=True)
        tokenizer.fit_on_texts(data)
        tokenizer.inverse_dict = inverse_dict(tokenizer.word_index)

        self.vocab_size = len(tokenizer.word_index)
        self.tokenizer = tokenizer

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if len(data.columns) > 1:
            raise Exception("CharBasedTextFeature can work only with one text column")

        data = data[data.columns[0]]

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        data_gen = self.tokenizer.texts_to_sequences(data)
        data_gen = pad_sequences(
            data_gen,
            maxlen=self.text_max_len,
            padding="post",
            truncating="post",
            value=0.0,
        )
        # return data_gen
        return K.one_hot(K.cast(data_gen, "int32"), self.vocab_size)

    def top_k_top_p_filtering(
        self,
        logits: np.ndarray,
        top_k: int = 0,
        top_p: float = 0,
        filter_value: int = -1e8,
    ):
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        https://github.com/ari-holtzman/degen
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: >0 keep only top k tokens with the highest probability (top-k filtering).
            top_p: >0.0 keep the top tokens with cumulative probability >= top_p
            (nucleus filtering).
            filter_value: value to replace small values with.
        """

        def custom_scatter_(zeros, index, src):
            """Custom implementation of torch.Tensor.scatter_()
            args:
                zeros: np.ndarray placeholder for logits shape like logits
                index: np.ndarray sorted indeces for scatters
                src: np.ndarray indeces to remove
            """
            # zeros[i][index[i][j]] = src[i][j]
            for i in range(len(index)):
                for j in range(len(index[i])):
                    zeros[i][index[i][j]] = src[i][j]

            return zeros

        logits_removed = logits
        # top_k = min(top_k, logits.size(-1)) # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < tf.math.top_k(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits = tf.sort(logits, direction="DESCENDING")
            sorted_indices = tf.argsort(logits, direction="DESCENDING")
            cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs >= top_p

            # Shift the indices to the right to keep also the first token above the threshold
            zeros_for_shift = tf.zeros([tf.shape(sorted_indices_to_remove)[0], 1], dtype=tf.bool)
            sorted_indices_to_remove = tf.concat(
                (zeros_for_shift, sorted_indices_to_remove[:, :-1]), axis=1
            )

            # sorted_indices - ids of columns to replace
            # line in sorted_indices corresponds to line in sorted_indices_to_remove

            # indices are row and column coordinates like
            # [[0, 5], ... [0, 2], [1, 7], ..., [1, 6], ...]
            row_numbers = tf.reshape(
                tf.repeat(tf.range(sorted_indices.shape[0]), sorted_indices.shape[1]),
                [sorted_indices.shape[0] * sorted_indices.shape[1]],
            )
            flattened_sorted_indices = tf.reshape(
                sorted_indices, [sorted_indices.shape[0] * sorted_indices.shape[1]]
            )
            update_indices = tf.stack([row_numbers, flattened_sorted_indices], axis=1)
            # update values correspond to indices [False, False, True, .... True]
            flattened_update_values = tf.reshape(sorted_indices_to_remove, [-1])
            # replace old index values with new update values
            indices_to_remove = tf.tensor_scatter_nd_update(
                tf.zeros_like(logits, dtype=sorted_indices_to_remove.dtype),
                update_indices,
                flattened_update_values,
            )
            # fill indices with one filter_value
            logits_removed = tf.where(
                indices_to_remove,
                tf.fill(indices_to_remove.shape, filter_value),
                logits,
            )

        return logits_removed

    def inverse_transform(self, data: np.ndarray, **kwargs) -> List[str]:
        top_p = 0.9
        if len(kwargs) > 0:
            top_p = kwargs["top_p"]

        out = []
        for batch in data:
            # batch shape (self.text_max_len, self.vocab_size)
            logits = self.top_k_top_p_filtering(tf.convert_to_tensor(batch), top_p=top_p)
            probs = tf.nn.softmax(logits, axis=-1).numpy().astype(float)
            probs /= probs.sum(axis=1)[:, None]
            multinomial_samples = np.apply_along_axis(
                lambda x: np.argmax(np.random.multinomial(1, x)), 1, probs
            )
            tokens = list(multinomial_samples)
            word = "".join(
                self.tokenizer.inverse_dict[x]
                for x in tokens
                if x in self.tokenizer.inverse_dict.keys()
            )
            out.append(word)
        return out

    @lazy
    def input(self) -> tf.Tensor:
        self.index_input = Input(
            shape=(self.text_max_len, self.vocab_size), name="input_%s" % self.name
        )

        return self.index_input

    @lazy
    def encoder(self) -> tf.Tensor:
        rnn_encoder_layer = Bidirectional(self.rnn_unit(self.rnn_units, return_sequences=False))

        rnn_econder = rnn_encoder_layer(self.input)
        return rnn_econder

    @lazy
    def __decoder_layer(self) -> List[tf.Tensor]:
        decoder_layers = list()

        decoder_layers.append(RepeatVector(self.text_max_len))
        decoder_layers.append(self.rnn_unit(self.rnn_units, return_sequences=True))
        decoder_layers.append(TimeDistributed(Dense(self.vocab_size, activation="linear")))

        return decoder_layers

    def create_decoder(self, encoder_output: tf.Tensor) -> tf.Tensor:
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        for layer in decoder_layers:
            x = layer(x)

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self) -> tf.Tensor:
        if not hasattr(self, "decoder"):
            Exception("Decoder isn't created")

        return self.weight * K.mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.input, logits=self.decoder)
        )


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

    @staticmethod
    def __validate_format(date_text: pd.DataFrame):
        """
        Define the most common date format.
        Supported date formats -
        MM/DD/YYYY; MM-DD-YYYY; DD/MM/YYYY; DD-MM-YYYY;
        YYYY/MM/DD; YYYY-MM-DD; MMM DD, YYYY; MMM DD YYYY;
        DD MMM YYYY; YYYY-MM-DD HH:MM:SS

        Not supported formats -
        MM/DD/YY, DD/MM/YY, YY/MM/DD, MM-DD-YY, DD-MM-YY

        """
        pattern = (
            r"\s{0,1}\d+[-/\\:\.]\s{0,1}\d+[-/\\:\.]\s{0,1}\d+|"
            r"[A-Z][a-z]+ \d{1,2} \d{4}|"
            r"[A-Z][a-z]+ \d{1,2}, \d{4}|"
            r"\d{2} [A-Z][a-z]+ \d{4}|"
            r"\d{2}[-][A-Z][a-z]+[-]\d{2}|"
            r"\d{4}[-/\\]\d{1,2}"
        )
        types = []
        sample = date_text.dropna().sample(100, replace=len(date_text) <= 100).values
        for i in sample:
            date_format = guess_datetime_format(re.match(pattern, i[0]).group(0))
            types.append(date_format)
        if not list(filter(lambda x: bool(x), types)) or not types:
            return "%d-%m-%Y"
        if Counter(types).most_common(1)[0][0] is None:
            return Counter(types).most_common(2)[1][0]
        return Counter(types).most_common(1)[0][0]

    def fit(self, data):
        self.date_format = self.__validate_format(data)
        self.data = chain.from_iterable(data.values)
        self.data = pd.DataFrame(list(map(lambda d: datetime_to_timestamp(d), self.data)))
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
        return Input(shape=(self.input_dimension,), name="input_%s" % self.name, dtype="float64")

    @lazy
    def encoder(self):
        return self.input

    @lazy
    def __decoder_layer(self):
        decoder_layers = list()
        for idx, item in enumerate(self.decoder_layers):
            name = "%s_decoder_%d" % (self.name, idx)
            if isinstance(item, int):
                decoder_layers.append(Dense(item, activation="relu", name=name))

            if isinstance(item, Layer):
                item.name = name
                decoder_layers.append(Layer)

        decoder_layers.append(
            Dense(
                self.input_dimension,
                dtype="float32",
                activation="linear",
                name="%s_linear" % self.name,
            )
        )

        return decoder_layers

    def create_decoder(self, encoder_output):
        if not isinstance(self.__decoder_layer, list):
            decoder_layers = [self.__decoder_layer]
        else:
            decoder_layers = self.__decoder_layer

        x = encoder_output
        for layer in decoder_layers:
            x = layer(x)

        self.decoder = x
        return self.decoder

    @lazy
    def loss(self):
        if not hasattr(self, "decoder"):
            Exception("Decoder isn't created")

        low = self.weight_randomizer[0]
        high = self.weight_randomizer[1]
        random_weight = K.random_uniform_variable(shape=(1,), low=low, high=high)

        return random_weight * tf.keras.losses.MSE(self.input, self.decoder)
