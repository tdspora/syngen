from itertools import chain
from typing import Union, List
from lazy import lazy
from loguru import logger

import keras
import keras.ops as ops
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import shapiro, kurtosis
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    OneHotEncoder
)
from keras import losses
from keras.layers import (
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
    convert_to_date_string
)
from syngen.ml.vae.models.custom_layers import get_seed_generator

KURTOSIS_THRESHOLD = 50  # threshold for kurtosis to consider extreme outliers


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

    def inverse_transform(self, data: List, temperature: float = 1.0, **kwargs) -> np.ndarray:
        """
        Inverse transform feature from numeric to original format to obtain an original-like table.
        
        Args:
            data: Decoder output (numeric format)
            temperature: Sampling temperature for probabilistic features.
                        0 = deterministic (argmax), >0 = probabilistic sampling.
                        Lower values = more deterministic, higher = more random.
            **kwargs: Additional feature-specific arguments
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

    def inverse_transform(self, data: List, temperature: float = 0.0, **kwargs) -> np.ndarray:
        """
        Convert decoder output back to binary values.
        
        Args:
            data: Decoder output probabilities
            temperature: Sampling temperature (default 0 = deterministic argmax)
                        0 = deterministic (hard threshold at 0.5)
                        >0 = probabilistic sampling based on sigmoid output
                        Lower temperature = more deterministic, higher = more random
        """
        data = np.clip(data, 0, 1)  # Ensure [0, 1] range
        
        if temperature == 0:
            # Deterministic: hard threshold at 0.5
            data = np.round(data)
        else:
            # Probabilistic sampling based on sigmoid output
            # Apply temperature scaling to make sampling more/less deterministic
            # temperature < 1 = sharper (more deterministic)
            # temperature > 1 = softer (more random)
            probs = data.flatten()
            if temperature != 1.0:
                # Scale probabilities using temperature
                # For binary, we treat the output as probability of class 1
                logits = np.clip(np.log(probs / (1 - probs + 1e-10)), -10, 10)
                scaled_probs = 1 / (1 + np.exp(-logits / temperature))
            else:
                scaled_probs = probs
            
            # Sample from Bernoulli distribution
            random_vals = np.random.random(len(scaled_probs))
            data = (random_vals < scaled_probs).astype(float).reshape(data.shape)
        
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
        self.feature_type = "numeric"

    def fit(self, data: pd.DataFrame, **kwargs):
        self.is_positive = (data >= 0).sum().item() >= len(data) * 0.99
        self.scaler = self._select_scaler(data)
        self.scaler.fit(data)
        self.input_dimension = data.shape[1]
        
        # Store learned bounds for clipping during generation
        # Use percentiles to be robust to outliers
        self.learned_min = float(data.values.min())
        self.learned_max = float(data.values.max())
        # Also store percentile bounds for optional stricter clipping
        self.learned_p01 = float(np.percentile(data.values, 1))
        self.learned_p99 = float(np.percentile(data.values, 99))

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(data).astype("float32")

    def inverse_transform(self, data: np.ndarray, temperature: float = 0.0, **kwargs) -> np.ndarray:
        """
        Convert scaled data back to original values.
        
        Args:
            data: Decoder output (scaled values)
            temperature: Not used for continuous features (kept for API consistency)
        """
        reverse_transformed = self.scaler.inverse_transform(data)
        
        # Apply learned bounds if available
        if hasattr(self, 'learned_min') and hasattr(self, 'learned_max'):
            reverse_transformed = np.clip(
                reverse_transformed, 
                self.learned_min, 
                self.learned_max
            )
        
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
        random_weight = keras.random.uniform(
            shape=(1,), minval=low, maxval=high, seed=get_seed_generator()
        )

        return random_weight * keras.losses.mean_squared_error(self.input, self.decoder)


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

    def inverse_transform(self, data: np.ndarray, temperature: float = 0.0, **kwargs) -> np.ndarray:
        """
        Convert one-hot encoded data back to original categories.
        
        Args:
            data: Decoder output logits/probabilities [batch, num_categories]
            temperature: Sampling temperature (default 0 = deterministic argmax)
                        0 = deterministic (argmax - original behavior)
                        >0 = probabilistic sampling with temperature scaling
                        Lower temperature = sharper distribution, higher = flatter
        """
        if temperature == 0:
            # Deterministic: argmax (original behavior)
            indices = data.argmax(axis=1)
        else:
            # Probabilistic sampling with temperature
            # Apply softmax with temperature scaling
            # Lower temp = sharper peaks, higher temp = flatter distribution
            scaled_logits = data / temperature
            # Numerical stability: subtract max before exp
            scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            
            # Sample from categorical distribution
            indices = np.array([
                np.random.choice(len(p), p=p) for p in probs
            ])
        
        inversed = self.inverse_vectorizer(indices)
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
        random_weight = keras.random.uniform(
            shape=(1,), minval=low, maxval=high, seed=get_seed_generator()
        )

        return random_weight * keras.losses.categorical_crossentropy(self.input, self.decoder)


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
        self.feature_type = "text"

    def fit(self, data: pd.DataFrame, **kwargs):
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
        return tf.one_hot(tf.cast(data_gen, "int32"), self.vocab_size)

    @staticmethod
    def _top_p_filtering(
            logits: np.ndarray,
            top_p: float = 0.9
    ):
        # Convert logits to TensorFlow tensor
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)

        # Sort logits and get sorted indices
        sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
        sorted_indices = tf.argsort(logits, direction="DESCENDING", axis=-1)

        # Calculate cumulative probabilities
        cumulative_probs = tf.cumsum(sorted_logits, axis=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p

        # Shift the indices to the right to keep also the first token above the threshold
        zeros_for_shift = tf.zeros_like(sorted_indices_to_remove[:, :, :1], dtype=tf.bool)
        sorted_indices_to_remove = tf.concat(
            [zeros_for_shift, sorted_indices_to_remove[:, :, :-1]],
            axis=-1
        )

        # Create a mask for indices to remove
        batch_size, seq_length, vocab_size = logits.shape

        batch_indices = tf.repeat(tf.range(batch_size), seq_length * vocab_size)
        feature_length_indices = tf.tile(
            tf.repeat(tf.range(seq_length), vocab_size),
            [batch_size]
        )
        vocab_selection_indices = tf.reshape(sorted_indices, [-1])
        update_indices = tf.stack(
            [batch_indices, feature_length_indices, vocab_selection_indices],
            axis=1
        )
        flattened_update_values = tf.reshape(sorted_indices_to_remove, [-1])
        indices_to_remove = tf.tensor_scatter_nd_update(
            tf.zeros_like(logits, dtype=sorted_indices_to_remove.dtype),
            update_indices,
            flattened_update_values,
        )

        # Apply the filter value to the logits
        logits_removed = tf.where(indices_to_remove, tf.fill(indices_to_remove.shape, 0.0), logits)

        return logits_removed.numpy().astype(np.float64)

    @staticmethod
    def _top_k_filtering(
            logits: np.ndarray,
            top_k: int = 0
    ):
        indices_to_remove = logits < tf.math.top_k(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = 0.0
        return logits

    def _process_batch(self, batch: np.ndarray) -> List[str]:
        probs = tf.nn.softmax(batch, axis=-1).numpy().astype(float)
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

        return self.weight * ops.mean(
            keras.losses.categorical_crossentropy(
                y_true=self.input, y_pred=self.decoder, from_logits=True
            )
        )


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
        self.domain = domain  # fallback domain
        self.domains = None  # will be populated by fit()
        self.domain_probs = None

    def fit(self, data: pd.DataFrame, **kwargs):
        # Extract and store domain distribution from actual data
        col = data.iloc[:, 0].dropna().astype(str)
        domains = col.str.extract(r'@(.+)$')[0].dropna()
        
        if len(domains) > 0:
            domain_counts = domains.value_counts(normalize=True)
            self.domains = domain_counts.index.tolist()
            self.domain_probs = domain_counts.values.tolist()
        else:
            # Fallback to default domain if no valid emails found
            self.domains = [self.domain]
            self.domain_probs = [1.0]
        
        # Continue with original fit (local part only)
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
        local_parts = super().inverse_transform(data, **kwargs)
        
        # Sample domains from original distribution
        if self.domains and self.domain_probs:
            sampled_domains = np.random.choice(
                self.domains,
                size=len(local_parts),
                p=self.domain_probs
            )
            return [f"{local}@{domain}" for local, domain in zip(local_parts, sampled_domains)]
        else:
            # Fallback to default domain
            return [f"{s}@{self.domain}" for s in local_parts]


class SmartTextFeature(BaseFeature):
    """
    A feature for structured text data (phone numbers, IPs, dates, etc.)
    that leverages positional patterns rather than sequential dependencies.
    
    Uses character embeddings + sinusoidal positional encoding instead of 
    one-hot + LSTM, which is more efficient for text with fixed structure.
    
    Key differences from CharBasedTextFeature:
    - Character embeddings (learnable, 32-dim) instead of one-hot
    - Sinusoidal positional encoding (32-dim) captures position semantics
    - Dense encoder instead of LSTM (positions matter, not sequence)
    - Per-position weighted loss focuses on variable positions
    """

    def __init__(
        self,
        name: str,
        text_max_len: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__(name=name)
        self.decoder = None
        self.text_max_len = text_max_len
        self.embedding_dim = embedding_dim  # character embedding dimension
        self.hidden_dim = hidden_dim  # dense layer hidden dimension
        self.feature_type = "smart_text"  # distinct from "text" which uses one-hot
        self.position_weights = None  # will be computed during fit

    def fit(self, data: pd.DataFrame, **kwargs):
        """Fit tokenizer and compute position-wise entropy for weighting."""
        from tensorflow.keras.preprocessing.text import Tokenizer

        if len(data.columns) > 1:
            raise Exception("SmartTextFeature can work only with one text column")

        data = data[data.columns[0]]

        # Build character tokenizer
        tokenizer = Tokenizer(lower=False, char_level=True)
        tokenizer.fit_on_texts(data)
        tokenizer.inverse_dict = inverse_dict(tokenizer.word_index)

        self.vocab_size = len(tokenizer.word_index) + 1  # +1 for padding (0)
        self.tokenizer = tokenizer

        # Compute position-wise entropy for adaptive weighting
        # Positions with higher entropy (variable characters) get higher weight
        self._compute_position_weights(data)

    def _compute_position_weights(self, data: pd.Series):
        """
        Compute per-position weights based on character entropy.
        Positions with variable characters get higher weight during training.
        """
        # Pad strings and convert to character matrix
        padded = data.fillna('').astype(str).str.pad(self.text_max_len, side='right', fillchar='\x00')
        padded = padded.str.slice(0, self.text_max_len)
        
        char_matrix = np.array([list(s) for s in padded])
        
        weights = np.ones(self.text_max_len, dtype=np.float32)
        
        for pos in range(min(self.text_max_len, char_matrix.shape[1])):
            chars_at_pos = char_matrix[:, pos]
            unique, counts = np.unique(chars_at_pos, return_counts=True)
            probs = counts / counts.sum()
            # Shannon entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            # Higher entropy = more variable = higher weight
            # Scale entropy to range [0.5, 2.0]
            weights[pos] = 0.5 + min(entropy / 3.0, 1.5)
        
        self.position_weights = weights

    def _get_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """
        Generate sinusoidal positional encoding.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        positions = np.arange(seq_len)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        
        angles = positions / np.power(10000.0, (2 * (dimensions // 2)) / d_model)
        
        # Apply sin to even indices, cos to odd indices
        pos_encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        
        return pos_encoding

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform text data to character indices (not one-hot).
        Positional encoding will be added in the model.
        """
        if len(data.columns) > 1:
            raise Exception("SmartTextFeature can work only with one text column")

        data = data[data.columns[0]]

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        data_gen = self.tokenizer.texts_to_sequences(data)
        data_gen = pad_sequences(
            data_gen,
            maxlen=self.text_max_len,
            padding="post",
            truncating="post",
            value=0,  # 0 is padding index
        )
        
        # Return indices (shape: [batch, seq_len])
        # The model will handle embeddings and positional encoding
        return data_gen.astype(np.int32)

    @staticmethod
    def _top_p_filtering(logits: np.ndarray, top_p: float = 0.9):
        """Top-p (nucleus) sampling filter."""
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        
        sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
        sorted_indices = tf.argsort(logits, direction="DESCENDING", axis=-1)
        
        cumulative_probs = tf.cumsum(sorted_logits, axis=-1)
        
        sorted_indices_to_remove = cumulative_probs >= top_p
        
        zeros_for_shift = tf.zeros_like(sorted_indices_to_remove[:, :, :1], dtype=tf.bool)
        sorted_indices_to_remove = tf.concat(
            [zeros_for_shift, sorted_indices_to_remove[:, :, :-1]],
            axis=-1
        )
        
        batch_size, seq_length, vocab_size = logits.shape
        
        batch_indices = tf.repeat(tf.range(batch_size), seq_length * vocab_size)
        feature_length_indices = tf.tile(
            tf.repeat(tf.range(seq_length), vocab_size),
            [batch_size]
        )
        vocab_selection_indices = tf.reshape(sorted_indices, [-1])
        update_indices = tf.stack(
            [batch_indices, feature_length_indices, vocab_selection_indices],
            axis=1
        )
        flattened_update_values = tf.reshape(sorted_indices_to_remove, [-1])
        indices_to_remove = tf.tensor_scatter_nd_update(
            tf.zeros_like(logits, dtype=sorted_indices_to_remove.dtype),
            update_indices,
            flattened_update_values,
        )
        
        logits_removed = tf.where(indices_to_remove, tf.fill(indices_to_remove.shape, 0.0), logits)
        
        return logits_removed.numpy().astype(np.float64)

    def _process_batch(self, batch: np.ndarray) -> List[str]:
        """Process a batch of decoder outputs to generate strings."""
        probs = tf.nn.softmax(batch, axis=-1).numpy().astype(float)
        probs = self._top_p_filtering(probs, top_p=0.9)
        
        # Normalize
        probs_sum = probs.sum(axis=2, keepdims=True)
        probs_sum = np.where(probs_sum == 0, 1, probs_sum)  # avoid division by zero
        probs /= probs_sum
        
        # Sample from distribution
        multinomial_samples = np.apply_along_axis(
            lambda x: np.argmax(np.random.multinomial(1, x)), -1, probs
        )
        
        # Convert indices to characters
        chars_array = np.vectorize(lambda x: self.tokenizer.inverse_dict.get(x, ''))(
            multinomial_samples)
        
        # Join characters into strings
        words = ["".join(sample) for sample in chars_array]
        
        return words

    def inverse_transform(self, data: np.ndarray, **kwargs) -> List[str]:
        """Transform decoder output back to strings."""
        batch_size = 10000
        num_batches = (len(data) + batch_size - 1) // batch_size
        feature_values = []

        for i in range(num_batches):
            batch = data[i * batch_size: (i + 1) * batch_size]
            feature_values.extend(self._process_batch(batch))

        return feature_values

    @lazy
    def input(self) -> tf.Tensor:
        """
        Input: character indices [batch, seq_len].
        Unlike CharBasedTextFeature which uses one-hot [batch, seq_len, vocab_size].
        """
        self.index_input = Input(
            shape=(self.text_max_len,), dtype="int32", name="input_%s" % self.name
        )
        return self.index_input

    @lazy
    def encoder(self) -> tf.Tensor:
        """
        Encoder with character embeddings + positional encoding.
        
        1. Embed character indices -> [batch, seq_len, embedding_dim]
        2. Add positional encoding -> [batch, seq_len, embedding_dim * 2]
        3. Flatten and dense layers -> latent representation
        """
        from keras.layers import Embedding, Flatten, Dropout, Concatenate, Lambda
        
        # Character embedding layer
        char_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name=f"{self.name}_char_embedding"
        )(self.input)
        
        # Precompute positional encoding as a constant
        pos_encoding = self._get_positional_encoding(self.text_max_len, self.embedding_dim)
        
        # Use a Lambda layer to add positional encoding (Keras 3 compatible)
        def add_positional_encoding(x, pos_enc=pos_encoding):
            """Add precomputed positional encoding to the embeddings."""
            batch_size = ops.shape(x)[0]
            pos_enc_tensor = ops.convert_to_tensor(pos_enc, dtype=x.dtype)
            # Broadcast positional encoding to batch
            pos_enc_broadcast = ops.broadcast_to(
                ops.expand_dims(pos_enc_tensor, 0),
                (batch_size, pos_enc_tensor.shape[0], pos_enc_tensor.shape[1])
            )
            # Concatenate along last dimension
            return ops.concatenate([x, pos_enc_broadcast], axis=-1)
        
        # Apply positional encoding
        combined = Lambda(
            add_positional_encoding,
            name=f"{self.name}_pos_encoding"
        )(char_embedding)
        
        # Flatten and encode
        # Shape: [batch, seq_len * embedding_dim * 2]
        flattened = Flatten()(combined)
        
        # Dense encoding layers
        encoded = Dense(self.hidden_dim, activation="relu", name=f"{self.name}_enc_dense1")(flattened)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(self.hidden_dim // 2, activation="relu", name=f"{self.name}_enc_dense2")(encoded)
        
        return encoded

    @lazy
    def __decoder_layer(self) -> List[tf.Tensor]:
        """Build decoder layers for per-position character prediction."""
        decoder_layers = []
        
        # Expand from latent to full sequence representation
        decoder_layers.append(
            Dense(self.hidden_dim, activation="relu", name=f"{self.name}_dec_dense1")
        )
        decoder_layers.append(
            Dense(self.text_max_len * self.hidden_dim // 2, activation="relu", name=f"{self.name}_dec_dense2")
        )
        
        return decoder_layers

    def create_decoder(self, encoder_output: tf.Tensor) -> tf.Tensor:
        """
        Create decoder that outputs logits for each position.
        Output shape: [batch, seq_len, vocab_size]
        """
        from keras.layers import Reshape
        
        x = encoder_output
        
        # Apply dense layers
        for layer in self.__decoder_layer:
            x = layer(x)
        
        # Reshape to [batch, seq_len, hidden_dim // 2]
        x = Reshape((self.text_max_len, self.hidden_dim // 2))(x)
        
        # Per-position prediction: [batch, seq_len, vocab_size]
        x = TimeDistributed(
            Dense(self.vocab_size, activation="linear", name=f"{self.name}_char_logits"),
            name=f"{self.name}_time_dist"
        )(x)
        
        self.decoder = x
        return self.decoder

    @lazy
    def loss(self) -> tf.Tensor:
        """
        Per-position weighted categorical crossentropy loss.
        
        Positions with higher entropy (more variable characters) get higher weight.
        Uses sparse categorical crossentropy since input is indices, not one-hot.
        """
        if not hasattr(self, "decoder"):
            raise Exception("Decoder isn't created")
        
        # Convert position weights to tensor
        if self.position_weights is not None:
            weights = tf.constant(self.position_weights, dtype=tf.float32)
        else:
            weights = tf.ones(self.text_max_len, dtype=tf.float32)
        
        # Sparse categorical crossentropy per position
        # Input shape: [batch, seq_len] (indices)
        # Decoder shape: [batch, seq_len, vocab_size] (logits)
        per_position_loss = keras.losses.sparse_categorical_crossentropy(
            y_true=self.input, y_pred=self.decoder, from_logits=True
        )
        
        # Apply position weights: [batch, seq_len] * [seq_len] -> [batch, seq_len]
        weighted_loss = per_position_loss * weights
        
        # Mean over positions and batch
        return self.weight * ops.mean(weighted_loss)


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

    def inverse_transform(self, data, temperature: float = 0.0, **kwargs):
        """
        Convert scaled timestamp back to date string.
        
        Args:
            data: Decoder output (scaled timestamps)
            temperature: Not used for dates (kept for API consistency)
        """
        unscaled = self.scaler.inverse_transform(data)
        unscaled = chain.from_iterable(unscaled)
        return list(
            map(
                lambda t: convert_to_date_string(t, self.date_format),
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
        random_weight = keras.random.uniform(
            shape=(1,), minval=low, maxval=high, seed=get_seed_generator()
        )

        return random_weight * keras.losses.mean_squared_error(self.input, self.decoder)
