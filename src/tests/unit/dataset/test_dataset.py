import pytest
from unittest.mock import patch
import datetime

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
        "ULID": "string",
    },
    "format": "Avro",
}


def uuid_detection(df, schema):
    with patch.object(Dataset, "__post_init__", lambda x: None):
        mock_dataset = Dataset(
            df=pd.DataFrame(),
            schema=schema,
            metadata={"table_name": "mock_table"},
            table_name="mock_table",
            paths={},
        )

    mock_dataset._set_uuid_columns(df)
    assert mock_dataset.uuid_columns == {
        "UUIDv1",
        "UUIDv2",
        "UUIDv3",
        "UUIDv4",
        "UUIDv5",
        "ULID",
    }
    assert mock_dataset.uuid_columns_types == {
        "UUIDv1": 1,
        "UUIDv2": 2,
        "UUIDv3": 3,
        "UUIDv4": 4,
        "UUIDv5": 5,
        "ULID": "ulid",
    }


@pytest.mark.parametrize(
    "path_to_test_table",
    [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.csv",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.csv",
    ],
)
def test_is_valid_uuid_defined_in_csv_table_without_missing_values(path_to_test_table, rp_logger):
    rp_logger.info("Test the process of the detection of UUID columns in the table in csv format")
    df = pd.read_csv(path_to_test_table)
    uuid_detection(df, CSV_SCHEMA)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path_to_test_table",
    [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.avro",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.avro",
    ],
)
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
            df=df, schema=CSV_SCHEMA, metadata={}, table_name="mock_table", paths={}
        )
        setattr(mock_dataset, "dropped_columns", set())
        setattr(mock_dataset, "non_existent_columns", set())
        mock_dataset._set_metadata()
        fetched_dataset = mock_dataset.__getstate__()
        assert "df" not in fetched_dataset
        assert fetched_dataset == {
            "schema": {"fields": {}, "format": "CSV"},
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
            "dropped_columns": set(),
            "uuid_columns_types": {},
            "uuid_columns": set(),
            "binary_columns": set(),
            "categ_columns": {
                "last_name",
                "ip_address",
                "email",
                "gender",
                "id",
                "first_name",
            },
            "long_text_columns": set(),
            "float_columns": set(),
            "int_columns": set(),
            "str_columns": set(),
            "date_columns": set(),
            "date_mapping": dict()
        }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_non_existent_columns(rp_logger):
    rp_logger.info("Test the process of checking non-existent columns")
    df = pd.read_csv("./tests/unit/dataset/fixtures/data.csv")
    metadata = {
        "mock_table": {
            "keys": {
                "PK": {
                    "type": "PK",
                    "columns": [
                        "id",
                        "non_existent_pk_column",
                        "non_existent_pk_column_2",
                    ],
                },
                "UQ": {
                    "type": "UQ",
                    "columns": ["first_name", "non_existent_uq_column"],
                },
            }
        }
    }
    with patch.object(Dataset, "__post_init__", lambda x: None):
        mock_dataset = Dataset(
            df=df,
            schema=CSV_SCHEMA,
            metadata=metadata,
            table_name="mock_table",
            paths={},
        )
        setattr(mock_dataset, "dropped_columns", set())
        setattr(mock_dataset, "non_existent_columns", set())
        mock_dataset._set_metadata()
        assert mock_dataset.non_existent_columns == {
            "non_existent_pk_column",
            "non_existent_uq_column",
            "non_existent_pk_column_2",
        }
        assert mock_dataset.metadata == {
            "mock_table": {
                "keys": {
                    "PK": {"type": "PK", "columns": ["id"]},
                    "UQ": {"type": "UQ", "columns": ["first_name"]},
                }
            }
        }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "initial_date_format, expected_date_format",
    [
        ("%m-%d-%Y", "%m-%d-%Y"),
        ("%d-%m-%Y", "%d-%m-%Y"),
        ("%m/%d/%Y", "%m/%d/%Y"),
        ("%d/%m/%Y", "%d/%m/%Y"),
        ("%Y/%m/%d", "%Y/%m/%d"),
        ("%Y-%m-%d", "%Y-%m-%d"),
        ("%B %d, %Y", "%B %d, %Y"),
        ("%b %d, %Y", "%b %d, %Y"),
        ("%d %B %Y", "%d %B %Y"),
        ("%b %d %Y", "%b %d %Y"),
        ("%d.%m.%Y", "%d.%m.%Y"),
        ("%m-%b-%y", "%d-%m-%Y"),
    ]
)
def test_define_date_format_with_diff_format(initial_date_format, expected_date_format, rp_logger):
    rp_logger.info(
        "Test the process of identifying the date format in the date column "
        "where the initial date format - %s and expected date format: %s",
        initial_date_format,
        expected_date_format,
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }
    data = {
        "Date": [(datetime.datetime(2020, 1, 1) + datetime.timedelta(days=x)).
                 strftime(initial_date_format) for x in range(10000)]
    }
    df = pd.DataFrame(data, columns=["Date"])
    with patch.object(Dataset, "__post_init__", lambda x: None):
        mock_dataset = Dataset(
            df=df,
            schema=CSV_SCHEMA,
            metadata=metadata,
            table_name="mock_table",
            paths={},
        )
        setattr(mock_dataset, "dropped_columns", set())
        setattr(mock_dataset, "non_existent_columns", set())
        mock_dataset._set_metadata()
        assert mock_dataset.date_mapping == {"Date": expected_date_format}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "initial_date_format, expected_date_format, extreme_values",
    [
        ("%Y-%m-%d", "%Y-%m-%d", ["4723-10-17T07:45:35Z", "9999-12-31T05:22:15Z"]),
        ("%Y/%m/%d", "%Y/%m/%d", ["3/10/17T07:45:35Z", "9/12/31T05:22:15Z"]),
        ("%m/%d/%Y", "%m/%d/%Y", ["31/11/2017T07:45:35Z", "31/02/1999T05:22:15Z"])
    ]
)
def test_define_date_format_with_extreme_values(
        initial_date_format, expected_date_format, extreme_values, rp_logger
):
    rp_logger.info(
        "Test the process of identifying the date format in the date column "
        "where the initial date format - %s and expected date format: %s",
        initial_date_format,
        expected_date_format,
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }
    data = {
        "Date": [(datetime.datetime(2020, 1, 1) + datetime.timedelta(days=x)).
                 strftime(initial_date_format) for x in range(10000)]
    }
    data.update({"Date": data["Date"] + extreme_values})
    df = pd.DataFrame(data, columns=["Date"])
    with patch.object(Dataset, "__post_init__", lambda x: None):
        mock_dataset = Dataset(
            df=df,
            schema=CSV_SCHEMA,
            metadata=metadata,
            table_name="mock_table",
            paths={},
        )
        setattr(mock_dataset, "dropped_columns", set())
        setattr(mock_dataset, "non_existent_columns", set())
        mock_dataset._set_metadata()
        assert mock_dataset.date_mapping == {"Date": expected_date_format}
    rp_logger.info(SUCCESSFUL_MESSAGE)
