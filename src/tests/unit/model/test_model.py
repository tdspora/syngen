import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

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
