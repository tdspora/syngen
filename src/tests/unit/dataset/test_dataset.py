import pytest
from unittest.mock import patch

import pandas as pd
import pandavro as pdx

from syngen.ml.vae.models.dataset import Dataset
from tests.conftest import SUCCESSFUL_MESSAGE


CSV_SCHEMA = {"fields": {}, "format": "CSV"}

AVRO_SCHEMA = {
    "fields": {
        "UUIDv1": "string",
        "UUIDv2": "string",
        "UUIDv3": "string",
        "UUIDv4": "string",
        "UUIDv5": "string",
        "ULID": "string"
    },
    "format": "Avro"
}


def uuid_detection(df, schema, rp_logger):
    with patch.object(Dataset, "__post_init__", lambda x: None):
        mock_dataset = Dataset(
            df=pd.DataFrame(),
            schema=schema,
            metadata={"table_name": "mock_table"},
            table_name="mock_table",
            paths={}
        )

    mock_dataset._set_uuid_columns(df)
    assert mock_dataset.uuid_columns == {"UUIDv1", "UUIDv2", "UUIDv3", "UUIDv4", "UUIDv5", "ULID"}
    assert mock_dataset.uuid_columns_types == {
        "UUIDv1": 1, "UUIDv2": 2, "UUIDv3": 3, "UUIDv4": 4, "UUIDv5": 5, "ULID": "ulid"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)

@pytest.mark.parametrize(
    "path_to_test_table", [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.csv",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.csv"
])
def test_is_valid_uuid_defined_in_csv_table_without_missing_values(path_to_test_table, rp_logger):
    rp_logger.info("Test the process of the detection of UUID columns in the table in csv format")
    df = pd.read_csv(path_to_test_table)
    uuid_detection(df, CSV_SCHEMA, rp_logger)

@pytest.mark.parametrize(
    "path_to_test_table", [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.avro",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.avro"
])
def test_is_valid_uuid_defined_in_avro_table_without_missing_values(path_to_test_table, rp_logger):
    rp_logger.info("Test the process of the detection of UUID columns in the table in avro format")
    df = pdx.from_avro(path_to_test_table)
    uuid_detection(df, AVRO_SCHEMA, rp_logger)