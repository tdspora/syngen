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


test_data = [
    ({"Date": ["01-01-2020", "02/02/2000", "05-05-2020"]}, "%m-%d-%Y"),
    ({"Date": ["31-01-2020", "20/02/2000", "25-05-2020"]}, "%d-%m-%Y"),
    ({"Date": ["03/03/2000", "01/01/2020", "05-05-2020"]}, "%m/%d/%Y"),
    ({"Date": ["31/01/2020", "20/02/2000", "25/05/2020"]}, "%d/%m/%Y"),
    ({"Date": ["2020/01/01", "1999/01/09", "05-05-2020"]}, "%Y/%m/%d"),
    ({"Date": ["2020-01-01", "1999-01-09", "05-05-2020"]}, "%Y-%m-%d"),
    ({"Date": ["March 10, 2022", "September 11, 1900", "May 15, 1877"]}, "%B %d, %Y"),
    ({"Date": ["Jul 10, 2022", "Jan 11, 1900", "Feb 15, 1877"]}, "%b %d, %Y"),
    ({"Date": ["10 June 2022", "11 January 1900", "01 February 1877"]}, "%d %B %Y"),
    ({"Date": ["Jul 10 2022", "Jan 11 1900", "Feb 15 1877"]}, "%b %d %Y"),
    (
        {
            "Date": [
                "1989-01-01 00:00:00.000000",
                "1897-01-01 03:03:00.000000",
                "2020-01-01 03:03:03.000000",
            ]
        },
        "%Y-%m-%d",
    ),
    (
        {
            "Date": [
                "1989/01/01 00:00:00.000000",
                "1897/01/01 03:03:00.000000",
                "2020/01/01 03:03:03.000000",
            ]
        },
        "%Y/%m/%d",
    ),
    (
        {
            "Date": [
                "2010-10-23 18:25:00 BRST",
                "2012-01-19 17:21:00 BRST",
                "2002-05-09 11:31:00 BRST",
            ]
        },
        "%Y-%m-%d",
    ),
    (
        {
            "Date": [
                "2010-10-23 18:25:00 +0300",
                "2012-01-19 17:21:00 +0300",
                "2002-05-09 11:31:00 +0300",
            ]
        },
        "%Y-%m-%d",
    ),
    (
        {"Date": ["2012/01/19 17:21:00", "2012/01/19 17:21:00", "2012/01/19 17:21:00"]},
        "%Y/%m/%d",
    ),
    (
        {"Date": ["01/01/19 17:21:00", "02/02/00 15:01:10", "12/10/19 17:21:00"]},
        "%d-%m-%Y",
    ),
    (
        {"Date": ["01-01-19 17:21:00", "02-02-00 15:01:10", "12-10-19 17:21:00"]},
        "%d-%m-%Y",
    ),
    (
        {"Date": ["01-01-19 17:21:00", "02-02-00 15:01:10", "31/10/2019 17:21:00"]},
        "%d/%m/%Y",
    ),
    (
        {"Date": ["01/01/19 17:21:00", "02/02/00 15:01:10", "31-10-2019 17:21:00"]},
        "%d-%m-%Y",
    ),
    (
        {
            "Date": [
                "2021-07-07T15:16:01.795Z",
                "2022-07-07T15:16:01.795Z",
                "2023-07-07T15:16:01.795Z",
            ]
        },
        "%Y-%m-%d",
    ),
    ({"Date": ["24.1.1997", "5.6.1899", "30.01.1990"]}, "%d.%m.%Y"),
    (
        {"Date": ["4723-10-17T07:45:35Z", "9999-12-31T05:22:15Z", "1-01-01T08:59:23Z"]},
        "%Y-%m-%d",
    ),
    ({"Date": ["2022-01", "1999-02", "9999-01"]}, "%d-%m-%Y"),
    ({"Date": ["31-11-28", "1999-02", "9999-01"]}, "%d-%m-%Y"),
    (
        {"Date": ["8448-05-17 05:51:56", "9999-12-31 05:22:15", "0001-01-01 08:59:23"]},
        "%Y-%m-%d",
    ),
    ({"Date": ["14-May-59", "31-Dec-99", "05-Jun-89"]}, "%d-%m-%Y"),
]


@pytest.mark.parametrize("data, expected_date_format", test_data)
def test_define_date_format(data, expected_date_format, rp_logger):
    rp_logger.info(
        "Test the process of identifying the date format in the date column: %s and expected date format: %s",
        data,
        expected_date_format,
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }
    df = pd.DataFrame(data)
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
