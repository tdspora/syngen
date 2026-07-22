from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from syngen.ml.vae.wrappers.wrappers import VAEWrapper
from tests.conftest import SUCCESSFUL_MESSAGE


def test_find_non_finite_features_detects_nan_and_inf(rp_logger):
    """EPMCTDM-7581 guardrail: features whose transformed (model-input) arrays
    contain NaN or inf are reported by name."""
    rp_logger.info("Test 'VAEWrapper._find_non_finite_features' detects NaN/inf")
    feature_names = ["clean_num", "bad_date", "clean_cat"]
    transformed_data = [
        np.array([[0.1], [0.2], [0.3]], dtype="float32"),
        np.array([[np.nan], [0.5], [0.6]], dtype="float32"),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype="float32"),
    ]
    assert VAEWrapper._find_non_finite_features(feature_names, transformed_data) == [
        "bad_date"
    ]

    transformed_data[2] = np.array([[np.inf], [0.0], [0.0]], dtype="float32")
    assert set(
        VAEWrapper._find_non_finite_features(feature_names, transformed_data)
    ) == {"bad_date", "clean_cat"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_find_non_finite_features_passes_clean_data(rp_logger):
    rp_logger.info("Test 'VAEWrapper._find_non_finite_features' passes clean data")
    feature_names = ["a", "b"]
    transformed_data = [
        np.array([[0.1], [0.2]], dtype="float32"),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
    ]
    assert VAEWrapper._find_non_finite_features(feature_names, transformed_data) == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_transformed_data_raises_on_nan(rp_logger):
    """The guardrail must abort training with a ValueError naming the offending
    feature when NaN reaches the model input."""
    rp_logger.info("Test 'VAEWrapper._validate_transformed_data' raises on NaN")

    class _StubWrapper:
        dataset = type("_Dataset", (), {"features": {"good": None, "bad": None}})()
        _find_non_finite_features = staticmethod(VAEWrapper._find_non_finite_features)
        _validate_transformed_data = VAEWrapper._validate_transformed_data

    transformed_data = [
        np.array([[0.1], [0.2]], dtype="float32"),
        np.array([[np.nan], [0.2]], dtype="float32"),
    ]
    with pytest.raises(ValueError, match="bad"):
        _StubWrapper()._validate_transformed_data(transformed_data)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_transformed_data_passes_clean_data(rp_logger):
    rp_logger.info("Test 'VAEWrapper._validate_transformed_data' passes clean data")

    class _StubWrapper:
        dataset = type("_Dataset", (), {"features": {"good": None}})()
        _find_non_finite_features = staticmethod(VAEWrapper._find_non_finite_features)
        _validate_transformed_data = VAEWrapper._validate_transformed_data

    _StubWrapper()._validate_transformed_data([np.array([[0.1], [0.2]], dtype="float32")])
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_create_batched_dataset_preserves_order_and_prefetches(rp_logger):
    first = np.arange(8, dtype=np.float32).reshape(4, 2)
    second = np.arange(4, dtype=np.int32).reshape(4, 1)
    wrapper = VAEWrapper.__new__(VAEWrapper)
    wrapper.batch_size = 2
    wrapper.dataset = SimpleNamespace(
        features={"first": None, "second": None},
        transform=lambda _: [first, second],
    )

    batched = wrapper._create_batched_dataset(pd.DataFrame(index=range(4)))
    first_batch, second_batch = next(iter(batched))

    assert "PrefetchDataset" in type(batched).__name__
    assert "BatchDataset" in type(batched._input_dataset).__name__
    np.testing.assert_array_equal(first_batch.numpy(), first[:2])
    np.testing.assert_array_equal(second_batch.numpy(), second[:2])
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_accumulate_feature_losses_supports_tensor_values(rp_logger):
    accumulated = VAEWrapper._accumulate_feature_losses(
        {"text": tf.constant(1.5)},
        {},
    )
    accumulated = VAEWrapper._accumulate_feature_losses(
        {"text": tf.constant(2.5)},
        accumulated,
    )

    assert float(np.mean(accumulated["text"] / 2)) == pytest.approx(2.0)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_step_runs_as_compiled_tensorflow_function(rp_logger):
    inputs = tf.keras.Input(shape=(1,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.add_loss(tf.reduce_mean(tf.square(outputs - inputs)))
    model.add_loss(tf.reduce_sum(outputs * 0))

    wrapper = VAEWrapper.__new__(VAEWrapper)
    wrapper.model = model
    wrapper.vae = SimpleNamespace(feature_losses={"value": None})
    wrapper.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    wrapper.loss_metric = tf.keras.metrics.Mean()
    batch = (tf.constant([[1.0], [2.0]], dtype=tf.float32),)
    train_step = wrapper._compile_train_step()

    loss, kl_loss, feature_losses = train_step(batch)
    train_step(batch)

    assert np.isfinite(float(loss.numpy()))
    assert float(kl_loss.numpy()) == 0.0
    assert np.isfinite(float(feature_losses["value"].numpy()))
    assert train_step.python_function is not None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_loop_uses_compiled_step(tmp_path, rp_logger):
    wrapper = VAEWrapper.__new__(VAEWrapper)
    compiled_step = MagicMock(
        return_value=(
            tf.constant(1.0),
            tf.constant(0.0),
            {"value": tf.constant(1.0)},
        )
    )
    wrapper._compile_train_step = MagicMock(return_value=compiled_step)
    wrapper.table_name = "table"
    wrapper.paths = {"state_path": str(tmp_path)}
    wrapper.num_batches = 1
    wrapper.model = MagicMock()
    wrapper._gather_losses_info = MagicMock()
    wrapper._VAEWrapper__save_losses = MagicMock()
    wrapper._log_losses_info_to_mlflow = MagicMock()
    batch = (tf.constant([[1.0]], dtype=tf.float32),)

    with patch(
        "syngen.ml.vae.wrappers.wrappers.ProgressBarHandler"
    ), patch("syngen.ml.vae.wrappers.wrappers.MlflowTracker"):
        wrapper._train([batch], epochs=1)

    wrapper._compile_train_step.assert_called_once_with()
    compiled_step.assert_called_once_with(batch)
    rp_logger.info(SUCCESSFUL_MESSAGE)
