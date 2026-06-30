import numpy as np
import pytest

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
