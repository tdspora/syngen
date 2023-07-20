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


def uuid_detection(df, schema):
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


@pytest.mark.parametrize(
    "path_to_test_table", [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.csv",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.csv"
])
def test_is_valid_uuid_defined_in_csv_table_without_missing_values(path_to_test_table, rp_logger):
    rp_logger.info("Test the process of the detection of UUID columns in the table in csv format")
    df = pd.read_csv(path_to_test_table)
    uuid_detection(df, CSV_SCHEMA)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path_to_test_table", [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.avro",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.avro"
])
def test_is_valid_uuid_defined_in_avro_table_without_missing_values(path_to_test_table, rp_logger):
    rp_logger.info("Test the process of the detection of UUID columns in the table in avro format")
    df = pdx.from_avro(path_to_test_table)
    uuid_detection(df, AVRO_SCHEMA)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_dataset(rp_logger):
    rp_logger.info("Test the process of saving the dataset")
    df = pd.read_csv("./tests/unit/dataset/fixtures/data.csv")
    with patch.object(Dataset, "__post_init__", lambda x: None):
        mock_dataset = Dataset(
            df=df,
            schema=CSV_SCHEMA,
            metadata={},
            table_name="mock_table",
            paths={}
        )
        mock_dataset._set_metadata()
        fetched_dataset = mock_dataset.__getstate__()
        assert "df" not in fetched_dataset
        assert fetched_dataset == {
            "schema": {
                "fields": {},
                "format": "CSV"
            },
            "metadata": {},
            "table_name": "mock_table",
            "paths": {},
            "primary_keys_mapping": {},
            "primary_keys_list": [],
            "primary_key_name": None,
            "pk_columns": [],
            "unique_keys_mapping": {},
            "unique_keys_mapping_list": [],
            "unique_keys_list": [],
            "uq_columns": [],
            "foreign_keys_mapping": {},
            "foreign_keys_list": [],
            "fk_columns": [],
            "empty_columns": [],
            "uuid_columns_types": {},
            "uuid_columns": set(),
            "binary_columns": set(),
            "categ_columns": {
                "last_name",
                "ip_address",
                "email",
                "gender",
                "id",
                "first_name"
            },
            "long_text_columns": set(),
            "float_columns": set(),
            "int_columns": set(),
            "str_columns": set(),
            "date_columns": set()
        }
    rp_logger.info(SUCCESSFUL_MESSAGE)
