import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from syngen.ml.vae.models.model import CVAE, _batched_encode, _batched_forward
from tests.conftest import SUCCESSFUL_MESSAGE


def _make_dataset_mock():
    dataset = MagicMock()
    dataset.order_of_columns = ["col_a", "col_b"]
    return dataset


class _FakeEncodeModel:
    """Deterministic per-row transform standing in for CVAEModule.encode:
    mu = 2*x, log_sigma = -x, so batching can be checked against a plain
    unbatched call without needing a real VAE."""

    def encode(self, tensors):
        x = tensors[0]
        return 2 * x, -x


class _FakeForwardModel:
    """Deterministic per-row transform standing in for CVAEModule.forward:
    recon = 3*x (single feature), mu = x, log_sigma = 0."""

    def __call__(self, tensors):
        x = tensors[0]
        return [3 * x], x, torch.zeros_like(x)


class TestCVAEDeviceResolution:
    """CVAE.__init__ auto-detects a device when none is explicitly passed -
    the mechanism that keeps every existing CVAE(...) call site transparent
    under the auto-detect device-selection design (no CLI/API change)."""

    def test_defaults_to_cpu_when_no_gpu_visible(self, rp_logger):
        rp_logger.info("Test 'CVAE.__init__' defaults to cpu when no GPU is visible")
        with patch("torch.cuda.is_available", return_value=False):
            cvae = CVAE(
                _make_dataset_mock(),
                batch_size=32,
                latent_dim=10,
                intermediate_dim=128,
                latent_components=5,
            )
        assert cvae.device == torch.device("cpu")
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_auto_detects_cuda_zero_when_gpu_visible(self, rp_logger):
        rp_logger.info("Test 'CVAE.__init__' auto-detects cuda:0 when a GPU is visible")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            cvae = CVAE(
                _make_dataset_mock(),
                batch_size=32,
                latent_dim=10,
                intermediate_dim=128,
                latent_components=5,
            )
        assert cvae.device == torch.device("cuda:0")
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_explicit_device_overrides_auto_detection(self, rp_logger):
        rp_logger.info("Test 'CVAE.__init__' honors an explicitly passed device")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
        ):
            cvae = CVAE(
                _make_dataset_mock(),
                batch_size=32,
                latent_dim=10,
                intermediate_dim=128,
                latent_components=5,
                device=torch.device("cuda:3"),
            )
        assert cvae.device == torch.device("cuda:3")
        rp_logger.info(SUCCESSFUL_MESSAGE)


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


class TestBatchedEncodeForward:
    """Regression: CVAE.fit_sampler/predict used to run model.encode()/model()
    over an ENTIRE table in a single unbatched call. On CPU this just uses a
    lot of RAM; on a GPU with limited VRAM it tried to allocate tens of GiB
    at once for a table with a long-text column and raised
    `torch.OutOfMemoryError` - discovered by actually running training on GPU
    hardware. `_batched_encode`/`_batched_forward` must chunk by batch_size
    while producing results identical to the unbatched call."""

    def test_batched_encode_matches_unbatched_result(self, rp_logger):
        rp_logger.info("Test '_batched_encode' matches an unbatched encode() call")
        x = torch.arange(10, dtype=torch.float32).unsqueeze(1)
        model = _FakeEncodeModel()

        mu, log_sigma = _batched_encode(model, [x], batch_size=3)
        expected_mu, expected_log_sigma = model.encode([x])

        assert torch.equal(mu, expected_mu)
        assert torch.equal(log_sigma, expected_log_sigma)
        assert mu.shape[0] == 10, "row count must be preserved across chunks"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_batched_encode_handles_batch_size_larger_than_data(self, rp_logger):
        rp_logger.info("Test '_batched_encode' handles batch_size >= row count as a single chunk")
        x = torch.arange(4, dtype=torch.float32).unsqueeze(1)
        model = _FakeEncodeModel()

        mu, log_sigma = _batched_encode(model, [x], batch_size=100)

        assert torch.equal(mu, 2 * x)
        assert torch.equal(log_sigma, -x)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_batched_forward_matches_unbatched_result(self, rp_logger):
        rp_logger.info("Test '_batched_forward' matches an unbatched forward() call")
        x = torch.arange(10, dtype=torch.float32).unsqueeze(1)
        model = _FakeForwardModel()

        recons, mu, log_sigma = _batched_forward(model, [x], batch_size=3)
        expected_recons, expected_mu, expected_log_sigma = model([x])

        assert torch.equal(recons[0], expected_recons[0])
        assert torch.equal(mu, expected_mu)
        assert torch.equal(log_sigma, expected_log_sigma)
        assert recons[0].shape[0] == 10, "row count must be preserved across chunks"
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
