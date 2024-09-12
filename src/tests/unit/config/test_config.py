import pandas as pd
import pytest
from unittest.mock import patch

from syngen.ml.config import TrainConfig
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_get_state_of_train_config(rp_logger):
    rp_logger.info("Test the method '__getstate__' of the class TrainConfig")
    train_config = TrainConfig(
        source="data/source.csv",
        epochs=10,
        drop_null=True,
        row_limit=1000,
        table_name="test_table",
        metadata_path="metadata/path.yaml",
        print_report=True,
        batch_size=32,
        loader=lambda x: pd.DataFrame()
    )
    expected_attributes = {
        "source",
        "epochs",
        "drop_null",
        "row_limit",
        "table_name",
        "metadata_path",
        "print_report",
        "batch_size"
    }
    state = train_config.__getstate__()
    for attr in expected_attributes:
        assert attr in state
    assert "loader" not in state
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("drop_null, row_limit, expected_size", [
    (False, None, 1000),
    (True, None, 801),
    (True, 100, 100)
])
@patch.object(TrainConfig, "_save_original_schema")
@patch.object(TrainConfig, "_remove_empty_columns")
@patch.object(TrainConfig, "_mark_removed_columns")
@patch.object(TrainConfig, "_save_input_data")
def test_preprocess_data(
    mock_save_input_data,
    mock_mark_removed_columns,
    mock_remove_empty_columns,
    mock_save_original_schema,
    drop_null,
    row_limit,
    expected_size,
    rp_logger
):
    rp_logger.info("Test the method '_preprocess_data' of the class TrainConfig")
    train_config = TrainConfig(
        source=f"{DIR_NAME}/unit/config/fixtures/data_types_detection_set.csv",
        epochs=10,
        drop_null=drop_null,
        row_limit=row_limit,
        table_name="test_table",
        metadata_path="metadata/path.yaml",
        print_report=True,
        batch_size=32,
        loader=None
    )
    train_config.preprocess_data()
    mock_save_input_data.assert_called_once()
    mock_save_original_schema.assert_called_once()
    mock_mark_removed_columns.assert_called_once()
    mock_remove_empty_columns.assert_called_once()
    assert train_config.row_subset == expected_size
    rp_logger.info(SUCCESSFUL_MESSAGE)
