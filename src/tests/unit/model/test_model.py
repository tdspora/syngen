import gc
import pickle
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, MagicMock

from syngen.ml.vae.models.features import CharBasedTextFeature
from syngen.ml.vae.models.model import CVAE
from tests.conftest import SUCCESSFUL_MESSAGE


@pytest.fixture
def mock_cvae():
    """
    Create a CVAE instance with mocked dataset and inverse_transformed_df
    """
    dataset = MagicMock()
    dataset.long_text_columns = set()
    dataset.uuid_columns = set()
    dataset.pk_uq_keys_types = {}
    dataset.dropped_columns = set()

    with patch.object(CVAE, "__init__", lambda self, *args, **kwargs: None):
        cvae = CVAE.__new__(CVAE)
        cvae.dataset = dataset
        cvae.inverse_transformed_df = pd.DataFrame()
    return cvae


def _build_text_cvae(serialized_feature):
    feature = pickle.loads(serialized_feature)
    dataset = SimpleNamespace(
        features={"text": feature},
        order_of_columns=["text"],
    )
    cvae = CVAE(
        dataset=dataset,
        batch_size=2,
        latent_dim=2,
        intermediate_dim=8,
        latent_components=2,
    )
    cvae.build_model()
    return cvae


@pytest.mark.parametrize("architecture_version", [1, 2])
def test_text_checkpoint_round_trip_preserves_architecture(
    architecture_version,
    tmp_path,
    rp_logger,
):
    tf.keras.backend.clear_session()
    data = pd.DataFrame({"text": ["ABCDEF", "FEDCBA"]})
    feature = CharBasedTextFeature(name="text", text_max_len=6, rnn_units=32)
    feature.architecture_version = architecture_version
    feature.fit(data)
    if architecture_version == 1:
        del feature.architecture_version
    serialized_feature = pickle.dumps(feature)

    source = _build_text_cvae(serialized_feature)
    source.latent_model = {"architecture_version": architecture_version}
    latent_input = tf.zeros((2, 2), dtype=tf.float32)
    source_input = source.dataset.features["text"].transform(data)
    expected_encoder = source.encoder_model(source_input, training=False).numpy()
    expected = source.generator_model(latent_input, training=False).numpy()
    source.save_state(str(tmp_path))

    tf.keras.backend.clear_session()
    restored = _build_text_cvae(serialized_feature)
    restored.load_state(str(tmp_path))
    restored_input = restored.dataset.features["text"].transform(data)
    actual_encoder = restored.encoder_model(restored_input, training=False).numpy()
    actual = restored.generator_model(latent_input, training=False).numpy()

    np.testing.assert_allclose(actual_encoder, expected_encoder)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (2, 6, restored.dataset.features["text"].vocab_size)
    assert restored.latent_model == {"architecture_version": architecture_version}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_text_checkpoint_rejects_cross_architecture_loading(tmp_path, rp_logger):
    tf.keras.backend.clear_session()
    data = pd.DataFrame({"text": ["ABCDEF", "FEDCBA"]})
    legacy_feature = CharBasedTextFeature(
        name="text",
        text_max_len=6,
        rnn_units=32,
    )
    legacy_feature.architecture_version = 1
    legacy_feature.fit(data)
    del legacy_feature.architecture_version
    legacy = _build_text_cvae(pickle.dumps(legacy_feature))
    legacy.latent_model = {"architecture_version": 1}
    legacy.save_state(str(tmp_path))

    tf.keras.backend.clear_session()
    optimized_feature = CharBasedTextFeature(name="text", text_max_len=6)
    optimized_feature.fit(data)
    optimized = _build_text_cvae(pickle.dumps(optimized_feature))

    tensorflow_log_level = tf.get_logger().level
    tf.get_logger().setLevel("ERROR")
    try:
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            optimized.load_state(str(tmp_path))
        del optimized
        gc.collect()
    finally:
        tf.get_logger().setLevel(tensorflow_log_level)
    rp_logger.info(SUCCESSFUL_MESSAGE)


class TestMakePkUqUnique:
    def test_numeric_column_without_regex_applies_sequential_keys(self, mock_cvae, rp_logger):
        rp_logger.info(
            "Test that a numeric (float) PK column without regex gets sequential keys"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({"id": [10.0, 20.0, 30.0]})
        mock_cvae.dataset.pk_uq_keys_types = {"id": float}

        pk_uq_keys_mapping = {
            "pk_key": {"columns": ["id"]}
        }

        mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            mock_cvae.inverse_transformed_df["id"].values, expected
        )
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_string_column_with_values_cast_to_numeric_without_regex_applies_sequential_keys(
        self, mock_cvae, rp_logger
    ):
        rp_logger.info(
            "Test that a string PK column containing values cast to numeric without regex "
            "gets sequential keys"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({"id": ["100", "200", "300"]})
        mock_cvae.dataset.pk_uq_keys_types = {"id": str}
        mock_cvae.dataset.long_text_columns = set()
        mock_cvae.dataset.uuid_columns = set()

        pk_uq_keys_mapping = {
            "pk_key": {"columns": ["id"]}
        }

        mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            mock_cvae.inverse_transformed_df["id"].values, expected
        )
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_string_column_with_regex_applies_regex_keys(self, mock_cvae, rp_logger):
        rp_logger.info(
            "Test that a string PK column with a regex pattern gets regex-generated keys"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({"id": ["a", "b", "c"]})
        mock_cvae.dataset.pk_uq_keys_types = {"id": str}

        pk_uq_keys_mapping = {
            "pk_key": {
                "columns": ["id"],
                "regex_patterns": {"id": r"CUST-[0-9]{4}"}
            }
        }

        with patch(
            "syngen.ml.vae.models.model.generate_unique_values_by_regex",
            return_value=["CUST-0001", "CUST-0002", "CUST-0003"]
        ):
            mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        assert list(mock_cvae.inverse_transformed_df["id"]) == [
            "CUST-0001", "CUST-0002", "CUST-0003"
        ]
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_numeric_column_with_numeric_regex_applies_regex_keys(self, mock_cvae, rp_logger):
        rp_logger.info(
            "Test that a numeric PK column with a numeric regex pattern gets regex-generated keys"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({"id": [1.0, 2.0, 3.0]})
        mock_cvae.dataset.pk_uq_keys_types = {"id": float}

        pk_uq_keys_mapping = {
            "pk_key": {
                "columns": ["id"],
                "regex_patterns": {"id": r"[1-9][0-9]{3}"}
            }
        }

        with patch(
            "syngen.ml.vae.models.model.is_number_regex_pattern", return_value=True
        ), patch(
            "syngen.ml.vae.models.model.generate_unique_values_by_regex",
            return_value=["1234", "5678", "9012"]
        ):
            mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        assert list(mock_cvae.inverse_transformed_df["id"]) == ["1234", "5678", "9012"]
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_numeric_column_with_non_numeric_regex_falls_back_to_sequential(
        self, mock_cvae, rp_logger
    ):
        rp_logger.info(
            "Test that a numeric PK column with a non-numeric regex pattern "
            "falls back to sequential keys and logs a warning"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({"id": [1.0, 2.0, 3.0]})
        mock_cvae.dataset.pk_uq_keys_types = {"id": float}

        pk_uq_keys_mapping = {
            "pk_key": {
                "columns": ["id"],
                "regex_patterns": {"id": r"CUST-[0-9]{4}"}
            }
        }

        with patch(
            "syngen.ml.vae.models.model.is_number_regex_pattern", return_value=False
        ):
            mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            mock_cvae.inverse_transformed_df["id"].values, expected
        )
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_multiple_keys_processed(self, mock_cvae, rp_logger):
        rp_logger.info(
            "Test that multiple PK/UQ keys are all processed"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({
            "pk_col": [1.0, 2.0, 3.0],
            "uq_col": ["a", "b", "c"]
        })
        mock_cvae.dataset.pk_uq_keys_types = {"pk_col": float, "uq_col": str}

        pk_uq_keys_mapping = {
            "pk_key": {"columns": ["pk_col"]},
            "uq_key": {
                "columns": ["uq_col"],
                "regex_patterns": {"uq_col": r"UQ-[0-9]{3}"}
            }
        }

        with patch(
            "syngen.ml.vae.models.model.generate_unique_values_by_regex",
            return_value=["UQ-001", "UQ-002", "UQ-003"]
        ):
            mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        expected_pk = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            mock_cvae.inverse_transformed_df["pk_col"].values, expected_pk
        )
        assert list(mock_cvae.inverse_transformed_df["uq_col"]) == [
            "UQ-001", "UQ-002", "UQ-003"
        ]
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_string_column_in_long_text_columns_not_treated_as_numeric(
        self, mock_cvae, rp_logger
    ):
        rp_logger.info(
            "Test that a string PK column in long_text_columns is not treated as numeric"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({"id": ["100", "200", "300"]})
        mock_cvae.dataset.pk_uq_keys_types = {"id": str}
        mock_cvae.dataset.long_text_columns = {"id"}
        mock_cvae.dataset.uuid_columns = set()

        pk_uq_keys_mapping = {
            "pk_key": {"columns": ["id"]}
        }

        mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        assert list(mock_cvae.inverse_transformed_df["id"]) == ["100", "200", "300"]
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_string_non_numeric_column_without_regex_unchanged(self, mock_cvae, rp_logger):
        rp_logger.info(
            "Test that a string PK column with non-numeric values and no regex "
            "remains unchanged"
        )
        mock_cvae.inverse_transformed_df = pd.DataFrame({"id": ["abc", "def", "ghi"]})
        mock_cvae.dataset.pk_uq_keys_types = {"id": str}
        mock_cvae.dataset.long_text_columns = set()
        mock_cvae.dataset.uuid_columns = set()

        pk_uq_keys_mapping = {
            "pk_key": {"columns": ["id"]}
        }

        mock_cvae._CVAE__make_pk_uq_unique(pk_uq_keys_mapping, empty_columns=set())

        assert list(mock_cvae.inverse_transformed_df["id"]) == ["abc", "def", "ghi"]
        rp_logger.info(SUCCESSFUL_MESSAGE)
