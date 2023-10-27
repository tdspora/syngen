from typing import Dict

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class FeatureLossLayer(Layer):
    def __init__(self, feature, **kwargs):
        self.feature = feature
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        feature_input, feature_decoder = inputs
        self.add_loss(self.feature.loss, inputs=inputs)
        return feature_decoder


class SampleLayer(Layer):
    def __init__(self, gamma, capacity, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.max_capacity = capacity

    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)
        self.built = True

    def call(self, layer_inputs, **kwargs):
        if len(layer_inputs) != 2:
            raise Exception("input layers must be a list: mean and stddev")
        if len(K.int_shape(layer_inputs[0])) != 2 or len(K.int_shape(layer_inputs[1])) != 2:
            raise Exception("input shape is not a vector [batchSize, latentSize]")

        mean = layer_inputs[0]
        log_var = layer_inputs[1]

        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]

        latent_loss = -0.5 * (1 + log_var - K.square(mean) - K.exp(log_var))
        latent_loss = K.sum(latent_loss, axis=1, keepdims=True)
        latent_loss = K.mean(latent_loss)
        latent_loss = self.gamma * K.abs(latent_loss - self.max_capacity)

        latent_loss = K.reshape(latent_loss, [1, 1])

        epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        layer_output = mean + K.exp(0.5 * log_var) * epsilon

        self.add_loss(losses=[latent_loss], inputs=[layer_inputs])

        return layer_output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @property
    def get_config(self):
        config = {
            "gamma": self.gamma,
            "capacity": self.max_capacity,
        }
        base_config: Dict = super(SampleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
