from typing import Optional

import keras
from keras.layers import Layer
import keras.ops as ops


# Module-level seed generator for reproducible random operations
_seed_generator: Optional[keras.random.SeedGenerator] = None


def set_seed_generator(seed: Optional[int] = None):
    """
    Set the module-level seed generator for reproducible random operations.
    Call this before building the VAE model.
    """
    global _seed_generator
    if seed is not None:
        _seed_generator = keras.random.SeedGenerator(seed)
    else:
        _seed_generator = None


def get_seed_generator() -> Optional[keras.random.SeedGenerator]:
    """Get the current seed generator."""
    return _seed_generator


class FeatureLossLayer(Layer):
    """
    Custom layer that computes and adds reconstruction loss for a feature.
    In Keras 3, Model.add_loss() doesn't work for Functional models,
    so we use this layer to add losses via Layer.add_loss() in the call method.
    
    Supports weight_randomizer for dynamic loss weighting during training.
    """

    def __init__(self, feature, loss_type='categorical', weight=1.0, 
                 weight_randomizer=None, seed_generator=None, custom_loss=None, **kwargs):
        super().__init__(**kwargs)
        self.feature = feature
        self.loss_type = loss_type
        self.weight = weight
        self.seed_generator = seed_generator
        self.custom_loss = custom_loss
        
        # Handle weight_randomizer: convert to (low, high) tuple
        if weight_randomizer is None:
            # Get from feature if available, else use fixed weight
            if hasattr(feature, 'weight_randomizer'):
                self.weight_randomizer = feature.weight_randomizer
            else:
                self.weight_randomizer = (weight, weight)
        elif isinstance(weight_randomizer, (list, tuple)) and len(weight_randomizer) == 2:
            self.weight_randomizer = tuple(weight_randomizer)
        elif isinstance(weight_randomizer, bool):
            self.weight_randomizer = (0, 1) if weight_randomizer else (weight, weight)
        elif isinstance(weight_randomizer, (int, float)):
            self.weight_randomizer = (weight_randomizer, weight_randomizer)
        else:
            self.weight_randomizer = (weight, weight)

    def call(self, inputs, training=None, **kwargs):
        """
        Compute the reconstruction loss from input and decoder output.

        Args:
            inputs: tuple of (feature_input, feature_decoder)
            training: whether the model is in training mode
        """
        feature_input, feature_decoder = inputs

        # Compute random weight for loss (weight_randomizer support)
        low, high = self.weight_randomizer
        if low == high:
            random_weight = low
        else:
            # Use random weight during training for regularization
            seed = self.seed_generator if self.seed_generator is not None else get_seed_generator()
            random_weight = keras.random.uniform(
                shape=(1,), minval=low, maxval=high, seed=seed
            )

        # Compute loss based on feature type
        if self.loss_type == 'continuous':
            loss = random_weight * ops.mean(keras.losses.mean_squared_error(feature_input, feature_decoder))
        elif self.loss_type == 'binary':
            loss = random_weight * ops.mean(keras.losses.binary_crossentropy(feature_input, feature_decoder))
        elif self.loss_type == 'sparse_categorical':
            # For SmartTextFeature: input is indices [batch, seq_len], decoder is logits [batch, seq_len, vocab]
            raw_loss = keras.losses.sparse_categorical_crossentropy(
                feature_input, feature_decoder, from_logits=True
            )
            # Cap per-position loss to prevent explosion from rare characters
            capped_loss = ops.minimum(raw_loss, 10.0)
            loss = random_weight * ops.mean(capped_loss)
        else:  # categorical
            loss = random_weight * ops.mean(keras.losses.categorical_crossentropy(feature_input, feature_decoder))

        self.add_loss(loss)
        return feature_decoder

    def get_config(self):
        config = super().get_config()
        config.update({
            "loss_type": self.loss_type,
            "weight": self.weight,
            "weight_randomizer": self.weight_randomizer,
            # Note: custom_loss is not serializable, so we omit it from config
        })
        return config


class SampleLayer(Layer):
    def __init__(self, gamma, capacity, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.max_capacity = capacity

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, layer_inputs, **kwargs):
        if len(layer_inputs) != 2:
            raise Exception("input layers must be a list: mean and stddev")
        if len(layer_inputs[0].shape) != 2 or len(layer_inputs[1].shape) != 2:
            raise Exception("input shape is not a vector [batchSize, latentSize]")

        mean = layer_inputs[0]
        log_var = layer_inputs[1]

        batch = ops.shape(mean)[0]
        dim = mean.shape[1]

        latent_loss = -0.5 * (1 + log_var - ops.square(mean) - ops.exp(log_var))
        latent_loss = ops.sum(latent_loss, axis=1, keepdims=True)
        latent_loss = ops.mean(latent_loss)
        latent_loss = self.gamma * ops.abs(latent_loss - self.max_capacity)

        latent_loss = ops.reshape(latent_loss, [1, 1])

        epsilon = keras.random.normal(
            shape=(batch, dim), mean=0.0, stddev=1.0, seed=get_seed_generator()
        )
        layer_output = mean + ops.exp(0.5 * log_var) * epsilon

        self.add_loss(latent_loss)

        return layer_output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "capacity": self.max_capacity,
        })
        return config
