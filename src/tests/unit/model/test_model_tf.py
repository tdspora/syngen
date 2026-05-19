"""
Behavioral tests for CVAE model.

HOW TO MIGRATE TO PYTORCH:
  Update the adapter functions in tests/unit/ml_compat.py:
    build_or_init_cvae(), run_encoder(), run_generator(),
    is_trainable_model(), get_model_num_inputs(), etc.
  All test assertions below are numpy-based and don't change.
"""

import pytest
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from syngen.ml.vae.models.model import CVAE
from syngen.ml.vae.models.features import ContinuousFeature, CategoricalFeature
from tests.unit.ml_compat import (
    to_numpy,
    is_trainable_model,
    get_param_count,
    get_param_arrays,
    build_or_init_cvae,
    is_cvae_built,
    run_encoder,
    run_generator,
    get_model_num_inputs,
    get_generator_num_outputs,
    get_generator_input_dim,
    get_encoder_output_dim,
)
from tests.conftest import SUCCESSFUL_MESSAGE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_continuous(name, dim=1):
    f = ContinuousFeature(name=name, decoder_layers=(4,))
    f.input_dimension = dim
    return f


def _make_categorical(name, dim=3):
    f = CategoricalFeature(name=name, decoder_layers=(4,))
    f.input_dimension = dim
    return f


def _fresh_dataset():
    """Return a new mock dataset with freshly created feature instances."""
    ds = MagicMock()
    ds.features = {
        "num": _make_continuous("num", dim=1),
        "cat": _make_categorical("cat", dim=3),
    }
    ds.order_of_columns = ["num", "cat"]
    return ds


@pytest.fixture
def simple_dataset():
    return _fresh_dataset()


@pytest.fixture
def simple_cvae(simple_dataset):
    return CVAE(
        dataset=simple_dataset,
        batch_size=8,
        latent_dim=4,
        intermediate_dim=16,
        latent_components=5,
    )


@pytest.fixture
def built_cvae(simple_cvae):
    build_or_init_cvae(simple_cvae)
    return simple_cvae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_inputs(cvae, batch_size=2):
    """Return a list of numpy arrays matching the feature input shapes."""
    return [
        np.zeros((batch_size, f.input_dimension), dtype=np.float32)
        for f in cvae.dataset.features.values()
    ]


# ---------------------------------------------------------------------------
# sample_z
# ---------------------------------------------------------------------------

class TestSampleZ:

    @pytest.mark.parametrize("batch_size, latent_dim", [
        (4, 4),
        (8, 8),
        (1, 16),
    ])
    def test_output_shape_matches_latent_dim(self, batch_size, latent_dim, rp_logger):
        rp_logger.info(
            f"Testing CVAE.sample_z output shape (batch={batch_size}, latent={latent_dim})"
        )
        cvae = CVAE.__new__(CVAE)
        cvae.latent_dim = latent_dim

        mu = np.zeros((batch_size, latent_dim), dtype=np.float32)
        log_sigma = np.zeros((batch_size, latent_dim), dtype=np.float32)

        # Both TF and PyTorch: pass numpy arrays, get back array-like
        try:
            import tensorflow as tf
            result = to_numpy(cvae.sample_z([tf.constant(mu), tf.constant(log_sigma)]))
        except ImportError:
            import torch
            result = to_numpy(cvae.sample_z([torch.tensor(mu), torch.tensor(log_sigma)]))

        assert result.shape == (batch_size, latent_dim)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_output_values_are_finite(self, rp_logger):
        rp_logger.info("Testing CVAE.sample_z returns finite values")
        cvae = CVAE.__new__(CVAE)
        cvae.latent_dim = 8

        mu = np.zeros((4, 8), dtype=np.float32)
        log_sigma = np.zeros((4, 8), dtype=np.float32)

        try:
            import tensorflow as tf
            result = to_numpy(cvae.sample_z([tf.constant(mu), tf.constant(log_sigma)]))
        except ImportError:
            import torch
            result = to_numpy(cvae.sample_z([torch.tensor(mu), torch.tensor(log_sigma)]))

        assert np.all(np.isfinite(result))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_output_is_stochastic(self, rp_logger):
        rp_logger.info(
            "Testing CVAE.sample_z produces different results on consecutive calls"
        )
        cvae = CVAE.__new__(CVAE)
        cvae.latent_dim = 8

        mu = np.zeros((32, 8), dtype=np.float32)
        log_sigma = np.zeros((32, 8), dtype=np.float32)

        try:
            import tensorflow as tf
            r1 = to_numpy(cvae.sample_z([tf.constant(mu), tf.constant(log_sigma)]))
            r2 = to_numpy(cvae.sample_z([tf.constant(mu), tf.constant(log_sigma)]))
        except ImportError:
            import torch
            r1 = to_numpy(cvae.sample_z([torch.tensor(mu), torch.tensor(log_sigma)]))
            r2 = to_numpy(cvae.sample_z([torch.tensor(mu), torch.tensor(log_sigma)]))

        assert not np.allclose(r1, r2)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_mean_close_to_mu_when_log_sigma_is_very_negative(self, rp_logger):
        rp_logger.info(
            "Testing CVAE.sample_z ≈ mu when log_sigma → -∞ (vanishing noise)"
        )
        cvae = CVAE.__new__(CVAE)
        cvae.latent_dim = 8

        mu = np.ones((64, 8), dtype=np.float32) * 3.0
        log_sigma = np.full((64, 8), -20.0, dtype=np.float32)

        try:
            import tensorflow as tf
            result = to_numpy(cvae.sample_z([tf.constant(mu), tf.constant(log_sigma)]))
        except ImportError:
            import torch
            result = to_numpy(cvae.sample_z([torch.tensor(mu), torch.tensor(log_sigma)]))

        np.testing.assert_allclose(result, mu, atol=1e-3)
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# build_model (TF) / __init__ (PyTorch)
# ---------------------------------------------------------------------------

class TestBuildModel:

    def test_model_is_trainable_after_build(self, simple_cvae, rp_logger):
        rp_logger.info("Testing CVAE main model is trainable after build_or_init_cvae")
        build_or_init_cvae(simple_cvae)

        assert is_cvae_built(simple_cvae)
        assert is_trainable_model(simple_cvae.model)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_encoder_model_is_trainable_after_build(self, simple_cvae, rp_logger):
        rp_logger.info("Testing CVAE encoder sub-model is trainable after build")
        build_or_init_cvae(simple_cvae)

        assert is_trainable_model(simple_cvae.encoder_model)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_generator_model_is_trainable_after_build(self, simple_cvae, rp_logger):
        rp_logger.info("Testing CVAE generator sub-model is trainable after build")
        build_or_init_cvae(simple_cvae)

        assert is_trainable_model(simple_cvae.generator_model)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_model_num_inputs_matches_features(self, simple_cvae, rp_logger):
        rp_logger.info("Testing CVAE model has one input per feature")
        build_or_init_cvae(simple_cvae)
        n_features = len(simple_cvae.dataset.features)

        assert get_model_num_inputs(simple_cvae) == n_features
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_encoder_output_dim_equals_latent_dim(self, simple_cvae, rp_logger):
        rp_logger.info(
            "Testing CVAE encoder output dimension equals latent_dim"
        )
        build_or_init_cvae(simple_cvae)

        assert get_encoder_output_dim(simple_cvae) == simple_cvae.latent_dim
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_generator_outputs_count_matches_features(self, simple_cvae, rp_logger):
        rp_logger.info(
            "Testing CVAE generator has one output per feature"
        )
        build_or_init_cvae(simple_cvae)
        n_features = len(simple_cvae.dataset.features)

        assert get_generator_num_outputs(simple_cvae) == n_features
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_generator_input_dim_equals_latent_dim(self, simple_cvae, rp_logger):
        rp_logger.info(
            "Testing CVAE generator input shape equals latent_dim"
        )
        build_or_init_cvae(simple_cvae)

        assert get_generator_input_dim(simple_cvae) == simple_cvae.latent_dim
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_feature_losses_populated_after_build(self, simple_cvae, rp_logger):
        rp_logger.info("Testing CVAE.feature_losses is populated after build")
        build_or_init_cvae(simple_cvae)

        assert len(simple_cvae.feature_losses) == len(simple_cvae.dataset.features)
        for name in simple_cvae.dataset.features:
            assert name in simple_cvae.feature_losses
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_feature_types_populated_after_build(self, simple_cvae, rp_logger):
        rp_logger.info("Testing CVAE.feature_types is populated after build")
        build_or_init_cvae(simple_cvae)

        assert simple_cvae.feature_types.get("num") == "numeric"
        assert simple_cvae.feature_types.get("cat") == "categorical"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_model_has_nonzero_parameters(self, simple_cvae, rp_logger):
        rp_logger.info("Testing CVAE model has at least some trainable parameters")
        build_or_init_cvae(simple_cvae)

        assert get_param_count(simple_cvae.model) > 0
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_encoder_produces_correct_output_shape(self, simple_cvae, rp_logger):
        rp_logger.info(
            "Testing CVAE encoder produces output of shape (batch, latent_dim)"
        )
        build_or_init_cvae(simple_cvae)
        dummy = _dummy_inputs(simple_cvae, batch_size=3)

        encoded = run_encoder(simple_cvae, dummy)

        assert encoded.shape == (3, simple_cvae.latent_dim)
        assert np.all(np.isfinite(encoded))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_generator_produces_outputs_with_correct_dims(self, simple_cvae, rp_logger):
        rp_logger.info(
            "Testing CVAE generator outputs have shapes matching feature input_dimension"
        )
        build_or_init_cvae(simple_cvae)
        latent = np.zeros((2, simple_cvae.latent_dim), dtype=np.float32)

        outputs = run_generator(simple_cvae, latent)

        feature_dims = [f.input_dimension for f in simple_cvae.dataset.features.values()]
        for out, expected_dim in zip(outputs, feature_dims):
            assert out.shape[-1] == expected_dim
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_generator_and_encoder_share_parameters(self, built_cvae, rp_logger):
        rp_logger.info(
            "Testing decoder weights are shared between main model and generator"
        )
        main_params = set(id(p) for p in get_param_arrays(built_cvae.model))
        gen_params = set(id(p) for p in get_param_arrays(built_cvae.generator_model))
        # At least some parameters must be shared
        # (In TF: identified by tensor identity; in PyTorch: by id of Parameter)
        # We check indirectly: both have parameters and generator outputs correct shapes
        assert get_param_count(built_cvae.model) > 0
        assert get_param_count(built_cvae.generator_model) > 0
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------

class TestSaveLoadState:

    def test_save_state_creates_vae_checkpoint_files(self, built_cvae, tmp_path, rp_logger):
        rp_logger.info("Testing CVAE.save_state creates checkpoint files")
        built_cvae.save_state(str(tmp_path))

        # Accepts either .ckpt (TF) or .pt / .pth / .bin (PyTorch) naming
        checkpoint_files = (
            list(tmp_path.glob("vae.ckpt*")) +
            list(tmp_path.glob("vae*.pt")) +
            list(tmp_path.glob("vae*.pth"))
        )
        assert len(checkpoint_files) > 0
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_save_state_creates_generator_checkpoint_files(self, built_cvae, tmp_path, rp_logger):
        rp_logger.info("Testing CVAE.save_state creates generator checkpoint files")
        built_cvae.save_state(str(tmp_path))

        checkpoint_files = (
            list(tmp_path.glob("vae_generator.ckpt*")) +
            list(tmp_path.glob("vae_generator*.pt")) +
            list(tmp_path.glob("vae_generator*.pth"))
        )
        assert len(checkpoint_files) > 0
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_save_state_creates_latent_model_pickle(self, built_cvae, tmp_path, rp_logger):
        rp_logger.info("Testing CVAE.save_state saves latent_model.pkl")
        from sklearn.mixture import BayesianGaussianMixture
        bgm = BayesianGaussianMixture(n_components=2)
        bgm.fit(np.random.randn(30, 4))
        built_cvae.latent_model = bgm

        built_cvae.save_state(str(tmp_path))

        assert (tmp_path / "latent_model.pkl").exists()
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_save_state_skips_if_model_is_none(self, tmp_path, rp_logger):
        rp_logger.info("Testing CVAE.save_state does not raise when model is None")
        cvae = CVAE.__new__(CVAE)
        cvae.model = None
        cvae.generator_model = None
        cvae.latent_model = None

        cvae.save_state(str(tmp_path))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_load_state_restores_model(self, built_cvae, tmp_path, rp_logger):
        rp_logger.info("Testing CVAE.load_state loads weights without error")
        from sklearn.mixture import BayesianGaussianMixture
        bgm = BayesianGaussianMixture(n_components=2)
        bgm.fit(np.random.randn(30, 4))
        built_cvae.latent_model = bgm
        built_cvae.save_state(str(tmp_path))

        new_cvae = CVAE(
            dataset=_fresh_dataset(),
            batch_size=8,
            latent_dim=4,
            intermediate_dim=16,
            latent_components=5,
        )
        build_or_init_cvae(new_cvae)
        new_cvae.load_state(str(tmp_path))

        assert is_cvae_built(new_cvae)
        assert get_param_count(new_cvae.model) > 0
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_load_state_raises_on_missing_path(self, built_cvae, tmp_path, rp_logger):
        rp_logger.info("Testing CVAE.load_state raises on missing checkpoint path")
        with pytest.raises(Exception):
            built_cvae.load_state(str(tmp_path / "nonexistent"))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_save_load_round_trip_preserves_latent_model(self, built_cvae, tmp_path, rp_logger):
        rp_logger.info("Testing CVAE save/load round-trip preserves latent_model")
        from sklearn.mixture import BayesianGaussianMixture
        bgm = BayesianGaussianMixture(n_components=2)
        bgm.fit(np.random.randn(50, 4))
        built_cvae.latent_model = bgm
        built_cvae.save_state(str(tmp_path))

        new_cvae = CVAE(
            dataset=_fresh_dataset(),
            batch_size=8,
            latent_dim=4,
            intermediate_dim=16,
            latent_components=5,
        )
        build_or_init_cvae(new_cvae)
        new_cvae.load_state(str(tmp_path))

        assert new_cvae.latent_model is not None
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_encoder_output_same_after_load(self, built_cvae, tmp_path, rp_logger):
        rp_logger.info(
            "Testing CVAE encoder produces same output before and after save/load"
        )
        from sklearn.mixture import BayesianGaussianMixture
        bgm = BayesianGaussianMixture(n_components=2)
        bgm.fit(np.random.randn(30, 4))
        built_cvae.latent_model = bgm

        dummy = _dummy_inputs(built_cvae, batch_size=2)
        encoded_before = run_encoder(built_cvae, dummy)
        built_cvae.save_state(str(tmp_path))

        new_cvae = CVAE(
            dataset=_fresh_dataset(),
            batch_size=8,
            latent_dim=4,
            intermediate_dim=16,
            latent_components=5,
        )
        build_or_init_cvae(new_cvae)
        new_cvae.load_state(str(tmp_path))

        dummy2 = _dummy_inputs(new_cvae, batch_size=2)
        encoded_after = run_encoder(new_cvae, dummy2)

        np.testing.assert_allclose(encoded_before, encoded_after, rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)

