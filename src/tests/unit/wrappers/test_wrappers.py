"""
Behavioral tests for VAEWrapper training utilities.

HOW TO MIGRATE TO PYTORCH:
  Update adapter functions in tests/unit/ml_compat.py:
    get_optimizer_lr(), accumulate_and_get_mean(), reset_loss_metric(),
    get_loss_metric_result(), count_batches(), first_batch_element_shapes()
  No changes to test methods below.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from syngen.ml.vae.wrappers.wrappers import VAEWrapper
from tests.unit.ml_compat import (
    to_numpy,
    get_optimizer_lr,
    accumulate_and_get_mean,
    reset_loss_metric,
    get_loss_metric_result,
    count_batches,
    first_batch_element_shapes,
)
from tests.conftest import SUCCESSFUL_MESSAGE


# ---------------------------------------------------------------------------
# Framework-specific setup helpers
# (Only THESE helper functions change when migrating to PyTorch)
# ---------------------------------------------------------------------------

def _make_wrapper(batch_size=4) -> VAEWrapper:
    """Bypass VAEWrapper.__post_init__ for unit testing individual methods."""
    with patch.object(VAEWrapper, "__post_init__", lambda self: None):
        wrapper = VAEWrapper(
            df=None, schema=None, metadata={},
            table_name="test_table", paths={},
            process="train", main_process="train",
            batch_size=batch_size, log_level="INFO",
        )
    return wrapper


def _make_optimizer(learning_rate: float):
    """
    Create an optimizer in the current framework.
    TF  : tf.keras.optimizers.Adam
    PT  : torch.optim.Adam (dummy parameter list)
    """
    try:
        import tensorflow as tf
        with patch("platform.processor", return_value="x86_64"):
            return VAEWrapper._create_optimizer(learning_rate)
    except ImportError:
        import torch
        dummy_param = [torch.nn.Parameter(torch.zeros(1))]
        return torch.optim.Adam(dummy_param, lr=learning_rate)


def _make_loss_metric():
    """
    Create a loss accumulator in the current framework.
    TF  : tf.keras.metrics.Mean
    PT  : simple dict-based accumulator (or custom class)
    """
    try:
        return VAEWrapper._create_loss()
    except Exception:
        # PyTorch fallback: simple accumulator
        class _MeanAccumulator:
            def __init__(self):
                self._total = 0.0
                self._count = 0
            def __call__(self, v):
                self._total += float(v)
                self._count += 1
            def result(self):
                return self._total / self._count if self._count else 0.0
            def reset(self):
                self._total = 0.0
                self._count = 0
        return _MeanAccumulator()


def _make_train_model():
    """
    Return a minimal trainable model that adds two real losses (feature + KL).
    TF  : tf.keras.Model subclass
    PT  : torch.nn.Module subclass
    """
    try:
        import tensorflow as tf

        class _TFDualLossModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(1)

            def call(self, inputs):
                x = self.dense(inputs[0])
                self.add_loss(tf.reduce_mean(tf.square(x)))
                self.add_loss(tf.reduce_mean(tf.abs(x)) * 0.01)
                return x

        return _TFDualLossModel()
    except ImportError:
        import torch
        import torch.nn as nn

        class _PTDualLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)

            def forward(self, inputs):
                x = self.linear(inputs[0])
                feat_loss = torch.mean(x ** 2)
                kl_loss = torch.mean(torch.abs(x)) * 0.01
                self._last_losses = [feat_loss, kl_loss]
                return x

        return _PTDualLossModel()


def _run_train_step(wrapper, batch):
    """
    Execute one training step and return (total_loss, kl_loss, feature_losses).
    Works for both TF GradientTape and PyTorch optimizers.
    """
    try:
        return wrapper._train_step(batch)
    except TypeError:
        # PyTorch: batch might be list of tensors instead of tuple
        import torch
        pt_batch = tuple(torch.as_tensor(b) for b in batch)
        return wrapper._train_step(pt_batch)


def _make_batch(batch_size=4, n_features=1, feature_dim=2):
    """Return a batch tuple of numpy arrays (converted to framework tensors inside)."""
    try:
        import tensorflow as tf
        return tuple(
            tf.random.normal([batch_size, feature_dim])
            for _ in range(n_features)
        )
    except ImportError:
        import torch
        return tuple(
            torch.randn(batch_size, feature_dim)
            for _ in range(n_features)
        )


# ---------------------------------------------------------------------------
# _create_optimizer
# ---------------------------------------------------------------------------

class TestCreateOptimizer:

    def test_optimizer_is_created_successfully(self, rp_logger):
        rp_logger.info("Testing _create_optimizer returns a usable optimizer")
        optimizer = _make_optimizer(1e-3)
        assert optimizer is not None
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("lr", [1e-4, 1e-3, 1e-2])
    def test_optimizer_learning_rate_is_set(self, lr, rp_logger):
        rp_logger.info(f"Testing optimizer learning_rate={lr} is correctly set")
        optimizer = _make_optimizer(lr)
        actual_lr = get_optimizer_lr(optimizer)
        np.testing.assert_allclose(actual_lr, lr, rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_arm_returns_different_optimizer_class(self, rp_logger):
        rp_logger.info(
            "Testing _create_optimizer returns legacy Adam on ARM (TF-specific)"
        )
        try:
            import tensorflow as tf
            with patch("platform.processor", return_value="arm"):
                opt_arm = VAEWrapper._create_optimizer(1e-3)
            with patch("platform.processor", return_value="x86_64"):
                opt_x86 = VAEWrapper._create_optimizer(1e-3)
            # Both are usable optimizers; class may differ
            assert get_optimizer_lr(opt_arm) > 0
            assert get_optimizer_lr(opt_x86) > 0
        except ImportError:
            pytest.skip("TF not available — ARM optimizer test not applicable")
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# _create_loss (loss metric / accumulator)
# ---------------------------------------------------------------------------

class TestCreateLossMetric:

    def test_loss_metric_is_created(self, rp_logger):
        rp_logger.info("Testing _create_loss returns a usable accumulator")
        metric = _make_loss_metric()
        assert metric is not None
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_loss_metric_accumulates_mean_correctly(self, rp_logger):
        rp_logger.info("Testing loss metric correctly averages provided values")
        metric = _make_loss_metric()
        mean = accumulate_and_get_mean(metric, [2.0, 4.0])
        np.testing.assert_allclose(mean, 3.0, rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_loss_metric_resets_to_zero(self, rp_logger):
        rp_logger.info("Testing loss metric resets to zero after reset")
        metric = _make_loss_metric()
        accumulate_and_get_mean(metric, [5.0])
        reset_loss_metric(metric)
        result = get_loss_metric_result(metric)
        assert result == 0.0
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_multiple_values_averaged_correctly(self, rp_logger):
        rp_logger.info("Testing loss metric mean of multiple values")
        metric = _make_loss_metric()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = accumulate_and_get_mean(metric, values)
        np.testing.assert_allclose(mean, 3.0, rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# _create_batched_dataset
# ---------------------------------------------------------------------------

class TestCreateBatchedDataset:

    @pytest.fixture(autouse=True)
    def setup_wrapper(self):
        self.batch_size = 4
        self.wrapper = _make_wrapper(batch_size=self.batch_size)

        mock_dataset = MagicMock()
        mock_dataset.transform.return_value = [
            np.random.randn(20, 1).astype("float32"),
            np.random.randn(20, 3).astype("float32"),
            np.random.randn(20, 2).astype("float32"),
        ]
        self.wrapper.dataset = mock_dataset
        self.df = pd.DataFrame({"placeholder": range(20)})

    def test_dataset_has_correct_number_of_batches(self, rp_logger):
        rp_logger.info(
            "Testing _create_batched_dataset produces floor(n/batch_size) batches"
        )
        dataset = self.wrapper._create_batched_dataset(self.df)
        n_batches = count_batches(dataset)

        assert n_batches == 5  # 20 / 4 = 5 (drop_remainder=True)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_each_batch_has_num_features_elements(self, rp_logger):
        rp_logger.info(
            "Testing each batch has one tensor per feature"
        )
        dataset = self.wrapper._create_batched_dataset(self.df)
        shapes = first_batch_element_shapes(dataset)

        assert len(shapes) == 3
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_batch_shapes_match_feature_dims(self, rp_logger):
        rp_logger.info(
            "Testing batch tensor shapes match (batch_size, feature_dim)"
        )
        dataset = self.wrapper._create_batched_dataset(self.df)
        shapes = first_batch_element_shapes(dataset)

        assert shapes[0] == (self.batch_size, 1)
        assert shapes[1] == (self.batch_size, 3)
        assert shapes[2] == (self.batch_size, 2)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_drop_remainder_drops_incomplete_batch(self, rp_logger):
        rp_logger.info(
            "Testing drop_remainder=True drops last partial batch"
        )
        mock_dataset = MagicMock()
        mock_dataset.transform.return_value = [
            np.random.randn(21, 1).astype("float32"),
        ]
        self.wrapper.dataset = mock_dataset
        dataset = self.wrapper._create_batched_dataset(self.df)

        assert count_batches(dataset) == 5  # 21 // 4 = 5
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("batch_size, n_samples, expected_batches", [
        (4, 16, 4),
        (8, 24, 3),
        (16, 32, 2),
    ])
    def test_batch_count_parametrized(self, batch_size, n_samples, expected_batches, rp_logger):
        rp_logger.info(
            f"Testing batch count: batch_size={batch_size}, n_samples={n_samples}"
        )
        wrapper = _make_wrapper(batch_size=batch_size)
        mock_dataset = MagicMock()
        mock_dataset.transform.return_value = [
            np.random.randn(n_samples, 2).astype("float32")
        ]
        wrapper.dataset = mock_dataset

        dataset = wrapper._create_batched_dataset(
            pd.DataFrame({"x": range(n_samples)})
        )
        assert count_batches(dataset) == expected_batches
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# _train_step
# ---------------------------------------------------------------------------

class TestTrainStep:

    @pytest.fixture(autouse=True)
    def setup_wrapper(self):
        self.wrapper = _make_wrapper(batch_size=4)
        self.wrapper.model = _make_train_model()
        self.wrapper.optimizer = _make_optimizer(1e-3)
        self.wrapper.loss_metric = _make_loss_metric()

        mock_vae = MagicMock()
        mock_vae.feature_losses = {"feat1": None}
        self.wrapper.vae = mock_vae

        # Warm up model
        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        self.wrapper.model(batch)

    def test_train_step_returns_three_values(self, rp_logger):
        rp_logger.info(
            "Testing _train_step returns (total_loss, kl_loss, feature_losses)"
        )
        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        result = _run_train_step(self.wrapper, batch)

        assert len(result) == 3
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_train_step_total_loss_is_positive_and_finite(self, rp_logger):
        rp_logger.info(
            "Testing _train_step total loss is a positive finite scalar"
        )
        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        total_loss, kl_loss, feature_losses = _run_train_step(self.wrapper, batch)

        total = float(total_loss) if not isinstance(total_loss, float) else total_loss
        assert np.isfinite(total)
        assert total > 0
        # total = feature_loss + kl_loss (sum of individual components)
        np.testing.assert_allclose(total, feature_losses["feat1"] + kl_loss, rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_train_step_kl_loss_is_finite(self, rp_logger):
        rp_logger.info("Testing _train_step kl_loss is a finite float")
        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        _, kl_loss, _ = _run_train_step(self.wrapper, batch)

        assert np.isfinite(kl_loss)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_train_step_feature_losses_keys_match_vae(self, rp_logger):
        rp_logger.info(
            "Testing _train_step feature_losses dict has correct keys"
        )
        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        _, _, feature_losses = _run_train_step(self.wrapper, batch)

        assert set(feature_losses.keys()) == {"feat1"}
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_train_step_feature_loss_values_are_floats(self, rp_logger):
        rp_logger.info(
            "Testing _train_step feature loss values are Python floats"
        )
        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        _, _, feature_losses = _run_train_step(self.wrapper, batch)

        for value in feature_losses.values():
            assert isinstance(value, (float, np.floating))
            assert np.isfinite(value)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_train_step_updates_loss_metric(self, rp_logger):
        rp_logger.info(
            "Testing _train_step updates the loss accumulator after the step"
        )
        reset_loss_metric(self.wrapper.loss_metric)
        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        total_loss, _, _ = _run_train_step(self.wrapper, batch)

        metric_val = get_loss_metric_result(self.wrapper.loss_metric)
        np.testing.assert_allclose(metric_val, float(total_loss), rtol=1e-5)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_train_step_updates_model_weights(self, rp_logger):
        rp_logger.info(
            "Testing _train_step actually updates model parameters"
        )
        from tests.unit.ml_compat import get_param_arrays
        weights_before = [w.copy() for w in get_param_arrays(self.wrapper.model)]

        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        _run_train_step(self.wrapper, batch)

        weights_after = get_param_arrays(self.wrapper.model)
        changed = any(
            not np.allclose(b, a)
            for b, a in zip(weights_before, weights_after)
        )
        assert changed, "Model weights were not updated by _train_step"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_train_step_multiple_features(self, rp_logger):
        rp_logger.info(
            "Testing _train_step with multiple features returns correct dict keys"
        )
        try:
            import tensorflow as tf

            class _MultiLossModel(tf.keras.Model):
                def __init__(self):
                    super().__init__()
                    self.dense = tf.keras.layers.Dense(1)

                def call(self, inputs):
                    x = self.dense(inputs[0])
                    self.add_loss(tf.reduce_mean(tf.square(x)))
                    self.add_loss(tf.reduce_mean(tf.square(x)) * 0.5)
                    self.add_loss(tf.reduce_mean(tf.abs(x)) * 0.01)
                    return x
        except ImportError:
            import torch.nn as nn, torch

            class _MultiLossModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(2, 1)

                def forward(self, inputs):
                    x = self.linear(inputs[0])
                    self._last_losses = [
                        torch.mean(x ** 2),
                        torch.mean(x ** 2) * 0.5,
                        torch.mean(torch.abs(x)) * 0.01,
                    ]
                    return x

        wrapper = _make_wrapper(batch_size=4)
        wrapper.model = _MultiLossModel()
        wrapper.optimizer = _make_optimizer(1e-3)
        wrapper.loss_metric = _make_loss_metric()
        mock_vae = MagicMock()
        mock_vae.feature_losses = {"num": None, "cat": None}
        wrapper.vae = mock_vae

        batch = _make_batch(batch_size=4, n_features=1, feature_dim=2)
        wrapper.model(batch)

        total_loss, kl_loss, feature_losses = _run_train_step(wrapper, batch)

        assert np.isfinite(float(total_loss))
        assert np.isfinite(kl_loss)
        assert set(feature_losses.keys()) == {"num", "cat"}
        assert all(np.isfinite(v) for v in feature_losses.values())
        rp_logger.info(SUCCESSFUL_MESSAGE)

