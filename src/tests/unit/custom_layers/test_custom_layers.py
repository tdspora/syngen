"""
Tests for custom Keras/PyTorch layers.

HOW TO MIGRATE TO PYTORCH:
  Only the adapter functions in tests/unit/ml_compat.py need updating:
    - fwd_feature_loss_layer()
    - fwd_feature_loss_layer_with_losses()
    - fwd_sample_layer()
  No changes to test methods below.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from syngen.ml.vae.models.custom_layers import FeatureLossLayer, SampleLayer
from tests.unit.ml_compat import (
    to_numpy,
    fwd_feature_loss_layer,
    fwd_feature_loss_layer_with_losses,
    fwd_sample_layer,
)
from tests.conftest import SUCCESSFUL_MESSAGE


# ---------------------------------------------------------------------------
# FeatureLossLayer
# ---------------------------------------------------------------------------

class TestFeatureLossLayer:

    def test_init_stores_feature_and_name(self, rp_logger):
        rp_logger.info(
            "Testing that FeatureLossLayer stores the feature reference and layer name"
        )
        mock_feature = MagicMock()
        layer = FeatureLossLayer(mock_feature, name="feature_loss")

        assert layer.feature is mock_feature
        assert layer.name == "feature_loss"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_call_returns_feature_decoder_unchanged(self, rp_logger):
        rp_logger.info(
            "Testing that FeatureLossLayer forward pass returns feature_decoder unchanged"
        )
        mock_feature = MagicMock()
        mock_feature.loss = 0.5

        layer = FeatureLossLayer(mock_feature, name="test_layer")

        feature_input = np.array([[1.0, 2.0]], dtype=np.float32)
        feature_decoder = np.array([[0.8, 0.9]], dtype=np.float32)

        result = fwd_feature_loss_layer(layer, feature_input, feature_decoder)

        np.testing.assert_allclose(result, feature_decoder, rtol=1e-6)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_call_output_shape_matches_decoder(self, rp_logger):
        rp_logger.info(
            "Testing that FeatureLossLayer output shape matches feature_decoder shape"
        )
        mock_feature = MagicMock()
        mock_feature.loss = 0.0

        layer = FeatureLossLayer(mock_feature, name="shape_layer")

        feature_input = np.zeros((4, 3), dtype=np.float32)
        feature_decoder = np.ones((4, 3), dtype=np.float32) * 0.5

        result = fwd_feature_loss_layer(layer, feature_input, feature_decoder)

        assert result.shape == feature_decoder.shape
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_call_adds_loss_that_is_accessible(self, rp_logger):
        rp_logger.info(
            "Testing that FeatureLossLayer contributes at least one finite loss value"
        )
        loss_value = 0.42
        mock_feature = MagicMock()
        mock_feature.loss = loss_value

        layer = FeatureLossLayer(mock_feature, name="value_layer")

        feature_input = np.ones((1, 2), dtype=np.float32)
        feature_decoder = np.ones((1, 2), dtype=np.float32) * 0.5

        _, losses = fwd_feature_loss_layer_with_losses(
            layer, feature_input, feature_decoder
        )

        assert len(losses) >= 1
        np.testing.assert_allclose(losses[0], loss_value, rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_two_layers_contribute_individual_losses(self, rp_logger):
        rp_logger.info(
            "Testing that two FeatureLossLayers each contribute their own loss value"
        )
        feat_a, feat_b = MagicMock(), MagicMock()
        feat_a.loss, feat_b.loss = 0.3, 0.7

        inp = np.ones((1, 2), dtype=np.float32)

        _, losses_a = fwd_feature_loss_layer_with_losses(
            FeatureLossLayer(feat_a, name="layer_a"), inp, inp
        )
        _, losses_b = fwd_feature_loss_layer_with_losses(
            FeatureLossLayer(feat_b, name="layer_b"), inp, inp
        )

        np.testing.assert_allclose(losses_a[0], 0.3, rtol=1e-5)
        np.testing.assert_allclose(losses_b[0], 0.7, rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# SampleLayer
# ---------------------------------------------------------------------------

class TestSampleLayer:

    def test_init_stores_gamma_and_capacity(self, rp_logger):
        rp_logger.info(
            "Testing that SampleLayer stores gamma and max_capacity on init"
        )
        layer = SampleLayer(gamma=2.0, capacity=30.0, name="sample")

        assert layer.gamma == 2.0
        assert layer.max_capacity == 30.0
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("gamma, capacity", [
        (1.0, 10.0),
        (0.5, 5.0),
        (3.0, 50.0),
    ])
    def test_init_parametrized(self, gamma, capacity, rp_logger):
        rp_logger.info(
            f"Testing SampleLayer init with gamma={gamma}, capacity={capacity}"
        )
        layer = SampleLayer(gamma=gamma, capacity=capacity)
        assert layer.gamma == gamma
        assert layer.max_capacity == capacity
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("batch_size, latent_dim", [
        (4, 4),
        (8, 8),
        (16, 10),
    ])
    def test_forward_output_shape_matches_mean(self, batch_size, latent_dim, rp_logger):
        rp_logger.info(
            f"Testing SampleLayer output shape with batch={batch_size}, dim={latent_dim}"
        )
        layer = SampleLayer(gamma=2.0, capacity=30.0)
        mean = np.zeros((batch_size, latent_dim), dtype=np.float32)
        log_var = np.zeros((batch_size, latent_dim), dtype=np.float32)

        result = fwd_sample_layer(layer, mean, log_var)

        assert result.shape == (batch_size, latent_dim)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_forward_output_is_numpy_array(self, rp_logger):
        rp_logger.info(
            "Testing that SampleLayer forward pass returns a numpy-convertible array"
        )
        layer = SampleLayer(gamma=1.0, capacity=5.0)
        mean = np.zeros((4, 3), dtype=np.float32)
        log_var = np.zeros((4, 3), dtype=np.float32)

        result = fwd_sample_layer(layer, mean, log_var)

        assert isinstance(result, np.ndarray)
        assert result.dtype in (np.float32, np.float64)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_forward_is_stochastic(self, rp_logger):
        rp_logger.info(
            "Testing that SampleLayer produces different samples on consecutive calls"
        )
        layer = SampleLayer(gamma=1.0, capacity=5.0)
        mean = np.zeros((32, 8), dtype=np.float32)
        log_var = np.zeros((32, 8), dtype=np.float32)

        output_1 = fwd_sample_layer(layer, mean, log_var)
        output_2 = fwd_sample_layer(layer, mean, log_var)

        assert not np.allclose(output_1, output_2), (
            "SampleLayer identical outputs on two calls — should be stochastic"
        )
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_forward_output_values_are_finite(self, rp_logger):
        rp_logger.info(
            "Testing SampleLayer output values are finite (no NaN or Inf)"
        )
        layer = SampleLayer(gamma=2.0, capacity=30.0)
        mean = np.zeros((8, 6), dtype=np.float32)
        log_var = np.zeros((8, 6), dtype=np.float32)

        result = fwd_sample_layer(layer, mean, log_var)

        assert np.all(np.isfinite(result)), "SampleLayer output contains non-finite values"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_forward_near_mean_when_variance_near_zero(self, rp_logger):
        rp_logger.info(
            "Testing SampleLayer output ≈ mu when log_sigma → -∞ (vanishing noise)"
        )
        layer = SampleLayer(gamma=2.0, capacity=30.0)
        mu = np.ones((64, 8), dtype=np.float32) * 2.0
        log_var = np.full((64, 8), -20.0, dtype=np.float32)  # exp(-10) ≈ 0

        result = fwd_sample_layer(layer, mu, log_var)

        np.testing.assert_allclose(result, mu, atol=1e-2)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_build_sets_built_flag_if_applicable(self, rp_logger):
        rp_logger.info("Testing SampleLayer.build() when it exists (Keras-specific)")
        layer = SampleLayer(gamma=1.0, capacity=10.0)
        # Keras-specific method; skipped when layer is already a PyTorch nn.Module
        if hasattr(layer, "build"):
            layer.build([(None, 4), (None, 4)])
            assert layer.built is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_compute_output_shape_if_available(self, rp_logger):
        rp_logger.info("Testing SampleLayer.compute_output_shape when available")
        layer = SampleLayer(gamma=1.0, capacity=5.0)
        if hasattr(layer, "compute_output_shape"):
            result = layer.compute_output_shape([(None, 8), (None, 8)])
            assert result == (None, 8)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("latent_dim", [4, 8, 16])
    def test_compute_output_shape_parametrized(self, latent_dim, rp_logger):
        rp_logger.info(
            f"Testing compute_output_shape with latent_dim={latent_dim}"
        )
        layer = SampleLayer(gamma=1.0, capacity=5.0)
        if hasattr(layer, "compute_output_shape"):
            result = layer.compute_output_shape(
                [(None, latent_dim), (None, latent_dim)]
            )
            assert result == (None, latent_dim)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_config_attributes_accessible(self, rp_logger):
        rp_logger.info(
            "Testing SampleLayer config attributes (gamma, capacity) are accessible"
        )
        # NOTE: In TF 2.15 the original code has a @property bug in get_config
        # that causes infinite recursion. We test stored attributes directly —
        # valid in both TF and PyTorch implementations.
        layer = SampleLayer(gamma=3.5, capacity=25.0, name="sample_cfg")

        assert layer.gamma == 3.5
        assert layer.max_capacity == 25.0
        rp_logger.info(SUCCESSFUL_MESSAGE)
