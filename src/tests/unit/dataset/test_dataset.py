import numpy as np
import pytest
import datetime
import random
import string
from unittest.mock import patch, MagicMock

import pandas as pd

from syngen.ml.vae.models.dataset import Dataset
from syngen.ml.data_loaders import DataLoader
from syngen.ml.utils import generate_uuids
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME

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


@pytest.mark.parametrize(
    "path_to_test_table, expected_schema",
    [
        (
            f"{DIR_NAME}/unit/dataset/fixtures/table_with_diff_uuid_columns.csv",
            {}
        ),
        (
            f"{DIR_NAME}/unit/dataset/fixtures/"
            "table_with_diff_uuid_columns_with_missing_values.csv",
            {}
        ),
        (
            f"{DIR_NAME}/unit/dataset/fixtures/table_with_diff_uuid_columns.avro",
            {
                "UUIDv1": "string",
                "UUIDv2": "string",
                "UUIDv3": "string",
                "UUIDv4": "string",
                "UUIDv5": "string",
                "ULID": "string",
            },
        ),
        (
            f"{DIR_NAME}/unit/dataset/fixtures/"
            "table_with_diff_uuid_columns_with_missing_values.avro",
            {
                "UUIDv1": "string",
                "UUIDv2": "string",
                "UUIDv3": "string",
                "UUIDv4": "string",
                "UUIDv5": "string",
                "ULID": "string",
            },
        )
    ],
)
@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
@patch.object(Dataset, "_set_categorical_columns")
def test_is_valid_uuid_defined_in_csv_table_without_missing_values(
    mock_set_categorical_columns,
    mock_fetch_config,
    path_to_test_table,
    expected_schema,
    rp_logger
):
    rp_logger.info(
        "Test the process of the detection of UUID columns"
    )
    df, schema = DataLoader(path_to_test_table).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    assert mock_dataset.fields == expected_schema
    mock_dataset.launch_detection()

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
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_save_dataset(rp_logger):
    rp_logger.info("Test the process of saving the dataset")
    df, schema = DataLoader(f"{DIR_NAME}/unit/dataset/fixtures/data.csv").load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    fetched_dataset = mock_dataset.__getstate__()
    assert "df" not in fetched_dataset
    assert set(fetched_dataset.keys()) == {
        "fields",
        "schema_format",
        "metadata",
        "table_name",
        "paths",
        "main_process",
        "features",
        "columns",
        "is_fitted",
        "all_columns",
        "null_num_column_names",
        "zero_num_column_names",
        "nan_labels_dict",
        "uuid_columns",
        "uuid_columns_types",
        "nan_labels_in_uuid",
        "dropped_columns",
        "order_of_columns",
        "custom_categ_columns",
        "categ_columns",
        "str_columns",
        "float_columns",
        "int_columns",
        "date_columns",
        "date_mapping",
        "binary_columns",
        "email_columns",
        "long_text_columns",
        "primary_keys_mapping",
        "primary_keys_list",
        "primary_key_name",
        "pk_columns",
        "unique_keys_mapping",
        "unique_keys_mapping_list",
        "unique_keys_list",
        "uq_columns_lists",
        "uq_columns",
        "foreign_keys_mapping",
        "foreign_keys_list",
        "fk_columns",
        "format"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_is_valid_categ_defined_in_csv_table(rp_logger):
    rp_logger.info(
        "Test the process of the detection of "
        "the categorical columns in the table in '.csv' format"
    )
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/dataset/fixtures/table_with_categ_columns.csv"
    ).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.categ_columns == {
        "time",
        "ptd_dt",
        "email",
        "id",
        "timestamp",
        "ensure",
        "upd_dt"
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_set_custom_categ_columns(rp_logger):
    rp_logger.info(
        "Test the process of the detection of "
        "the categorical columns that has been set by a user"
    )
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/dataset/fixtures/data_with_emails.csv"
    ).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={
            "mock_table": {
                "train_settings": {
                    "source": "path/to/source.csv",
                    "column_types": {
                        "categorical": [
                            "DocNumber",
                            "MetadataSubject",
                            "ExtractedFrom"
                        ]
                    }
                }
                },
                "infer_settings": {}
            },
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.categ_columns == {
        "SenderPersonId",
        "DocNumber",
        "MetadataSubject",
        "ExtractedFrom"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_is_valid_binary_defined_in_csv_table(rp_logger):
    rp_logger.info(
        "Test the process of the detection of the binary columns in the table in '.csv' format"
    )
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/dataset/fixtures/table_with_binary_columns.csv"
    ).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.binary_columns == {
        "time",
        "upd_dt",
        "ptd_dt",
        "email",
        "id",
        "timestamp"
    }
    assert mock_dataset.categ_columns == {"ensure"}


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_check_non_existent_columns(rp_logger):
    rp_logger.info("Test the process of checking non-existent columns")
    df, schema = DataLoader(f"{DIR_NAME}/unit/dataset/fixtures/data.csv").load_data()
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
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.dropped_columns = set()
    mock_dataset.launch_detection()
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
    "initial_date_format, expected_date_format, upper_case",
    [
        ("%m-%d-%Y", "%m-%d-%Y", False),
        ("%d-%b-%y", "%d-%m-%Y", True),
        ("%d-%m-%y", "%d-%m-%Y", False),
        ("%d-%m-%Y", "%d-%m-%Y", False),
        ("%m/%d/%Y", "%m/%d/%Y", False),
        ("%d/%m/%Y", "%d/%m/%Y", False),
        ("%Y/%m/%d", "%Y/%m/%d", False),
        ("%Y-%m-%d", "%Y-%m-%d", False),
        ("%B %d, %Y", "%B %d, %Y", False),
        ("%b %d, %Y", "%b %d, %Y", False),
        ("%d %B %Y", "%d %B %Y", False),
        ("%b %d %Y", "%b %d %Y", False),
        ("%d.%m.%Y", "%d.%m.%Y", False),
        ("%m-%b-%y", "%d-%m-%Y", False),
    ]
)
@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_define_date_format_with_diff_format(
    mock_fetch_config,
    initial_date_format,
    expected_date_format,
    upper_case,
    rp_logger
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

    if upper_case:
        data = {
            "Date": [(datetime.datetime(2020, 1, 1) + datetime.timedelta(days=x)).
                     strftime(initial_date_format).upper() for x in range(10000)]
        }
    else:
        data = {
            "Date": [(datetime.datetime(2020, 1, 1) + datetime.timedelta(days=x)).
                     strftime(initial_date_format) for x in range(10000)]
        }

    df = pd.DataFrame(data, columns=["Date"])
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.date_mapping == {"Date": expected_date_format}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "initial_date_format, expected_date_format, extreme_values",
    [
        ("%Y-%m-%d", "%Y-%m-%d", ["4723-10-17T07:45:35Z", "9999-12-31T05:22:15Z"]),
        ("%Y/%m/%d", "%Y/%m/%d", ["3/10/17T07:45:35Z", "9/12/31T05:22:15Z"]),
        ("%m/%d/%Y", "%m/%d/%Y", ["11/30/2017T07:45:35Z", "02/27/1999T05:22:15Z"]),
        ("%d-%m-%Y", "%d-%m-%Y", [np.nan, np.nan, np.nan])
    ]
)
@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_define_date_format_with_extreme_values(
    mock_fetch_config,
    initial_date_format,
    expected_date_format,
    extreme_values,
    rp_logger
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
    data.update(
        {"Date": data['Date'] + extreme_values}
    )
    df = pd.DataFrame(data, columns=["Date"])
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.date_mapping == {"Date": expected_date_format}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_is_valid_uuid(rp_logger):
    rp_logger.info(
        "Test the method 'is_valid_uuid' that checks if the given value is a valid UUID",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }
    data = {
        "UUID_1": [str(i) for i in generate_uuids(1, 100)],
        "UUID_2": [str(i) for i in generate_uuids(2, 100)],
        "UUID_3": [str(i) for i in generate_uuids(3, 100)],
        "UUID_4": [str(i) for i in generate_uuids(4, 100)],
        "UUID_5": [str(i) for i in generate_uuids(5, 100)],
        "ULID": [str(i) for i in generate_uuids("ulid", 100)],
        "binary": [random.choice([b'0', b'1']) for _ in range(100)],
        "boolean": [random.choice([True, False]) for _ in range(100)],
        "integer": [random.randint(1, 20) for _ in range(100)],
        "float": [random.uniform(1.0, 20.0) for _ in range(100)],
    }
    df = pd.DataFrame(
        data,
        columns=["UUID_1", "UUID_2", "UUID_3", "UUID_4", "UUID_5"]
    )
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.uuid_columns == {
        "UUID_1", "UUID_2", "UUID_3", "UUID_4", "UUID_5"
    }
    assert mock_dataset.uuid_columns_types == {
        "UUID_1": 1,
        "UUID_2": 2,
        "UUID_3": 3,
        "UUID_4": 4,
        "UUID_5": 5
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_set_email_columns(rp_logger):
    rp_logger.info(
        "Test the method '_set_email_columns' of the class Dataset",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    df, schema = DataLoader(f"{DIR_NAME}/unit/dataset/fixtures/data_with_emails.csv").load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.email_columns == {"ExtractedFrom"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_set_long_text_columns(rp_logger):
    rp_logger.info(
        "Test the method '_set_long_text_columns' of the class Dataset",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    df = pd.DataFrame({
        "column1": range(1, 101),
        "column2": range(101, 201),
        "column3": range(201, 301),
        "column4": range(301, 401),
        "long_text_column":
            ["".join(random.choice(alphabet)
                     for _ in range(250))
             for _ in range(1, 96)] +
            [np.NaN] * 5
    })
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.long_text_columns == {"long_text_column"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_handle_missing_values_in_numeric_columns_in_csv_file(rp_logger):
    rp_logger.info(
        "Test the process of handling missing values "
        "in numeric columns in a '.csv' file",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    data = {
        "column1": range(1, 101),
        "column2": range(101, 201),
        "column3": range(201, 301),
        "column4": range(301, 401),
        "column5": [str(i) for i in range(401, 491)] + ["Not available" for i in range(10)]
    }
    df = pd.DataFrame(data)

    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.int_columns == {"column1", "column2", "column3", "column4", "column5"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_handle_missing_values_in_numeric_columns_in_avro_file(rp_logger):
    rp_logger.info(
        "Test the process of handling missing values "
        "in numeric columns in a '.avro' file",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    data = {
        "column1": range(1, 101),
        "column2": range(101, 201),
        "column3": range(201, 301),
        "column4": range(301, 401),
        "column5": [str(i) for i in range(401, 491)] + ["Not available" for i in range(10)]
    }
    df = pd.DataFrame(data)

    schema = {
        "format": "Avro",
        "fields": {
            "column1": "int",
            "column2": "int",
            "column3": "int",
            "column4": "int",
            "column5": "string"
        }
    }
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.int_columns == {"column1", "column2", "column3", "column4", "column5"}
    assert mock_dataset.nan_labels_dict == {"column5": "Not available"}
    rp_logger.info(SUCCESSFUL_MESSAGE)
