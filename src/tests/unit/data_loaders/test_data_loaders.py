from unittest.mock import patch, mock_open
import pytest
import os

import pandas as pd
from pandas.testing import assert_frame_equal
from cryptography.fernet import Fernet, InvalidToken

from syngen.ml.data_loaders import (
    DataLoader,
    CSVLoader,
    AvroLoader,
    BinaryLoader,
    MetadataLoader,
    YAMLLoader,
    ExcelLoader,
    DataEncryptor
)
from syngen.ml.context import global_context, get_context
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


CSV_SCHEMA = {"fields": {}, "format": "CSV"}


def test_initialize_data_loader_for_local_csv_table_with_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local csv table with existed path"
    )
    path_to_table = f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
    test_data_loader = DataLoader(path_to_table)
    assert test_data_loader.path == path_to_table
    assert test_data_loader.has_existed_path is True
    assert test_data_loader.original_schema is None
    assert isinstance(test_data_loader.file_loader, CSVLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_csv_table_with_not_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local csv table with invalid path"
    )
    test_data_loader = DataLoader("path/to/table.csv")
    assert test_data_loader.path == "path/to/table.csv"
    assert test_data_loader.has_existed_path is False
    assert test_data_loader.original_schema is None
    assert isinstance(test_data_loader.file_loader, CSVLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_avro_table_with_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local avro table with existed path"
    )
    path_to_table = f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/table_with_data.avro"
    test_data_loader = DataLoader(path_to_table)
    assert test_data_loader.path == path_to_table
    assert test_data_loader.has_existed_path is True
    assert test_data_loader.original_schema == {
        "fields": [
            {"name": "gender", "type": ["null", "long"]},
            {"name": "height", "type": ["null", "double"]},
            {"name": "id", "type": ["null", "long"]}
        ],
        "name": "Root",
        "type": "record"
    }
    assert isinstance(test_data_loader.file_loader, AvroLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_avro_table_with_not_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local avro table with invalid path"
    )
    test_data_loader = DataLoader("path/to/table.avro")
    assert test_data_loader.path == "path/to/table.avro"
    assert test_data_loader.has_existed_path is False
    assert isinstance(test_data_loader.file_loader, AvroLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_pickle_table_with_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader "
        "for local pickle table with existed path"
    )
    path_to_table = f"{DIR_NAME}/unit/data_loaders/fixtures/pickle_tables/table_with_data.pkl"
    test_data_loader = DataLoader(path_to_table)
    assert test_data_loader.path == path_to_table
    assert test_data_loader.has_existed_path is True
    assert test_data_loader.original_schema is None
    assert isinstance(test_data_loader.file_loader, BinaryLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_pickle_table_with_not_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader "
        "for local pickle table with invalid path"
    )
    test_data_loader = DataLoader("path/to/table.pkl")
    assert test_data_loader.path == "path/to/table.pkl"
    assert test_data_loader.has_existed_path is False
    assert test_data_loader.original_schema is None
    assert isinstance(test_data_loader.file_loader, BinaryLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_table_in_unsupported_format(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local table in unsupported format"
    )
    with pytest.raises(NotImplementedError) as error:
        DataLoader("path/to/table.test")
        assert str(error.value) == "File format not supported"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_table_with_empty_path(caplog, rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local table with empty path"
    )
    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            DataLoader("")
            assert "It seems that the information of source is absent" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_csv_format(rp_logger):
    rp_logger.info("Loading data from local table in csv format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "csv_tables/table_with_data.csv")
    df, schema = data_loader.load_data()
    assert isinstance(data_loader.file_loader, CSVLoader)
    assert (
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "gender": [0, 1, 0, 1],
                    "height": [
                        157.18518021548246,
                        166.7731072622863,
                        162.91821942384928,
                        173.51448996432848,
                    ],
                    "id": [925, 84, 821, 383],
                }
            ),
        )
        is None
    )

    assert isinstance(df, pd.DataFrame)
    assert schema == CSV_SCHEMA
    assert data_loader.original_schema is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_csv_format_in_not_utf_8(caplog, rp_logger):
    rp_logger.info("Loading data from local table in csv format in not 'utf-8' encoding")
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_in_iso_encoding.csv"
    data_loader = DataLoader(path)
    assert isinstance(data_loader.file_loader, CSVLoader)

    with pytest.raises(UnicodeDecodeError) as error:
        with caplog.at_level("ERROR"):
            data_loader.load_data()
            assert (
                f"It seems that the content of the data in the path - '{path}' "
                f"doesn't have the encoding UTF-8. The details of the error - {error}.\n"
                f"Please, use the data in UTF-8 encoding" in caplog.text
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_csv_format(caplog, rp_logger):
    rp_logger.info("Loading data from local empty table in csv format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "csv_tables/empty_table.csv")
    assert isinstance(data_loader.file_loader, CSVLoader)
    with pytest.raises(pd.errors.EmptyDataError):
        with caplog.at_level("ERROR"):
            data_loader.load_data()
            assert "The empty file was provided. Unable to load data " in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_columns_from_table_in_csv_format(rp_logger):
    rp_logger.info("Get the list of the columns from the table in CSV format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "csv_tables/table_with_data.csv")
    columns = data_loader.get_columns()
    assert isinstance(data_loader.file_loader, CSVLoader)
    assert columns == ["gender", "height", "id"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_columns_from_table_in_csv_format_with_formatting_settings(rp_logger):
    rp_logger.info(
        "Get the list of the columns from the table in CSV format where the separator is '|'"
    )
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "csv_tables/pipe_delimited_text.csv")
    columns = data_loader.get_columns(sep="|")
    assert isinstance(data_loader.file_loader, CSVLoader)
    assert columns == ["id", "first_name", "last_name", "email", "gender", "ip_address"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_columns_from_empty_table_in_csv_format(caplog, rp_logger):
    rp_logger.info("Get the list of the columns from the empty table in csv format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "csv_tables/empty_table.csv")
    assert isinstance(data_loader.file_loader, CSVLoader)
    with pytest.raises(pd.errors.EmptyDataError):
        with caplog.at_level("ERROR"):
            data_loader.get_columns()
            assert "The empty file was provided. Unable to train this table" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_csv_format(test_csv_path, test_df, rp_logger):
    rp_logger.info("Saving data in csv format locally")
    data_loader = DataLoader(test_csv_path)
    data_loader.save_data(test_df)

    assert isinstance(data_loader.file_loader, CSVLoader)
    assert os.path.exists(test_csv_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema == CSV_SCHEMA
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_avro_format(rp_logger):
    rp_logger.info("Loading data from local table in '.avro' format")
    data_loader = DataLoader(
        f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/table_with_data.avro"
    )
    df, schema = data_loader.load_data()

    assert isinstance(data_loader.file_loader, AvroLoader)
    assert (
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "gender": [0, 1, 0, 1],
                    "height": [
                        157.18518021548246,
                        166.7731072622863,
                        162.91821942384928,
                        173.51448996432848,
                    ],
                    "id": [925, 84, 821, 383],
                }
            ),
        )
        is None
    )

    assert isinstance(df, pd.DataFrame)
    assert schema == {
        "fields": {"gender": "int", "height": "float", "id": "int"},
        "format": "Avro",
    }
    assert data_loader.original_schema == {
        "fields": [
            {"name": "gender", "type": ["null", "long"]},
            {"name": "height", "type": ["null", "double"]},
            {"name": "id", "type": ["null", "long"]}
        ],
        "name": "Root",
        "type": "record"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_avro_with_null_column(rp_logger):
    rp_logger.info(
        "Loading data from local table in avro format with a column that has a data type - 'null'"
    )
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "avro_tables/table_with_null_column.avro")
    df, schema = data_loader.load_data()

    assert isinstance(data_loader.file_loader, AvroLoader)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "name", "age", "email"]
    assert df.shape == (10, 4)
    assert schema == {
        "fields": {
            "id": "int",
            "name": "string",
            "age": "null",
            "email": "string"
        },
        "format": "Avro"
    }
    assert data_loader.original_schema == {
        "type": "record",
        "name": "TestRecord",
        "fields": [
            {"name": "id", "type": "int"},
            {"name": "name", "type": "string"},
            {"name": "age", "type": "null"},
            {"name": "email", "type": "string"}
        ]
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_avro_with_nullable_column(rp_logger):
    rp_logger.info(
        "Loading data from local table in avro format with a column that has a data type - "
        "['string', 'null']"
    )
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "avro_tables/table_with_nullable_column.avro")
    df, schema = data_loader.load_data()

    assert isinstance(data_loader.file_loader, AvroLoader)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "first_name", "last_name", "created_date", "Active"]
    assert df.shape == (100, 5)
    assert schema == {
        "fields": {
            "id": "int",
            "first_name": "string",
            "last_name": "string",
            "created_date": "string",
            "Active": "int"
        },
        "format": "Avro"
    }
    assert data_loader.original_schema == {
        "type": "record",
        "name": "avro_format",
        "fields": [
            {"name": "id", "type": "int"},
            {"name": "first_name", "type": "string"},
            {"name": "last_name", "type": "string"},
            {"name": "created_date", "type": ["string", "null"], "default": None},
            {"name": "Active", "type": "boolean"}
        ]
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_avro_format(caplog, rp_logger):
    rp_logger.info("Loading data from local empty table in avro format")
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/empty_table.avro"
    data_loader = DataLoader(path)

    assert isinstance(data_loader.file_loader, AvroLoader)

    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            data_loader.load_data()
            assert "cannot read header - is it an avro file?" in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_original_schema_from_empty_table_in_avro_format(caplog, rp_logger):
    rp_logger.info("Loading the original schema from local empty table in avro format")
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/empty_table.avro"
    data_loader = DataLoader(path)

    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            data_loader.file_loader.load_original_schema()
            assert "Read 0 bytes, expected 4 bytes" in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_columns_from_table_in_avro_format(rp_logger):
    rp_logger.info("Get the list of the columns from the table in avro format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "avro_tables/table_with_data.avro")
    columns = data_loader.get_columns()
    assert isinstance(data_loader.file_loader, AvroLoader)
    assert columns == ["gender", "height", "id"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_columns_from_empty_table_in_avro_format(caplog, rp_logger):
    rp_logger.info("Get the list of the columns from the empty table in avro format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "avro_tables/empty_table.avro")
    assert isinstance(data_loader.file_loader, AvroLoader)
    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            data_loader.get_columns()
            assert "The empty file was provided. Unable to train this table" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_avro_format(test_avro_path, test_df, test_avro_schema, rp_logger):
    rp_logger.info("Saving data in avro format locally")
    data_loader = DataLoader(test_avro_path)
    data_loader.save_data(test_df, schema=test_avro_schema)

    assert isinstance(data_loader.file_loader, AvroLoader)
    assert os.path.exists(test_avro_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema == {
        "fields": {"gender": "int", "height": "float", "id": "int"},
        "format": "Avro",
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_avro_format_without_provided_schema(
    test_avro_path, test_df, rp_logger
):
    rp_logger.info("Saving data in avro format locally without a provided schema")
    data_loader = DataLoader(test_avro_path)
    data_loader.save_data(test_df, schema=None)

    assert isinstance(data_loader.file_loader, AvroLoader)
    assert os.path.exists(test_avro_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema == {
        "fields": {"gender": "int", "height": "float", "id": "int"},
        "format": "Avro",
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_pickle_format(rp_logger):
    rp_logger.info("Loading data from local table in pickle format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "pickle_tables/table_with_data.pkl")
    df, schema = data_loader.load_data()

    assert isinstance(data_loader.file_loader, BinaryLoader)
    assert (
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "gender": [0, 1, 0, 1],
                    "height": [
                        157.18518021548246,
                        166.7731072622863,
                        162.91821942384928,
                        173.51448996432848,
                    ],
                    "id": [925, 84, 821, 383],
                }
            ),
        )
        is None
    )

    assert isinstance(df, pd.DataFrame)
    assert schema is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_pickle_format(caplog, rp_logger):
    rp_logger.info("Loading data from local empty table in pickle format")
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/pickle_tables/empty_table.pkl"
    data_loader = DataLoader(path)

    assert isinstance(data_loader.file_loader, BinaryLoader)

    data, schema = data_loader.load_data()
    assert data == {}
    assert schema is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_pickle_format(test_pickle_path, test_df, rp_logger):
    rp_logger.info("Saving data in pickle format locally")
    data_loader = DataLoader(test_pickle_path)
    data_loader.save_data(test_df)

    assert isinstance(data_loader.file_loader, BinaryLoader)
    assert os.path.exists(test_pickle_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_metadata_loader(rp_logger):
    rp_logger.info("Initializing metadata loader")
    path_to_metadata = f"{DIR_NAME}/unit/data_loaders/fixtures/metadata/metadata.yaml"
    test_metadata_loader = MetadataLoader(path_to_metadata)
    assert test_metadata_loader.path == path_to_metadata
    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_metadata_loader_in_unsupported_format(rp_logger):
    rp_logger.info("Initializing metadata loader in unsupported format")
    with pytest.raises(NotImplementedError) as error:
        MetadataLoader("path/to/table.test")
        assert str(error.value) == "The format of metadata isn't supported"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_in_yaml_format(rp_logger):
    rp_logger.info("Loading metadata in '.yaml' format")
    path_to_metadata = f"{DIR_NAME}/unit/data_loaders/fixtures/metadata/metadata.yaml"
    test_metadata_loader = MetadataLoader(path_to_metadata)

    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    metadata = test_metadata_loader.load_data()
    assert metadata == {
        "global": {},
        "pk_test": {
            "train_settings": {
                "source": "../data/pk_test.csv",
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_in_yml_format(rp_logger):
    rp_logger.info("Loading metadata in yml format")
    path_to_metadata = f"{DIR_NAME}/unit/data_loaders/fixtures/metadata/metadata.yml"
    test_metadata_loader = MetadataLoader(path_to_metadata)

    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    metadata = test_metadata_loader.load_data()
    assert metadata == {
        "global": {},
        "pk_test": {
            "train_settings": {
                "source": "../data/pk_test.csv",
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_by_yaml_loader_in_yaml_format(rp_logger):
    rp_logger.info("Loading metadata by YAMLLoader in '.yaml' format")
    path_to_metadata = f"{DIR_NAME}/unit/data_loaders/fixtures/metadata/metadata.yaml"
    loader = YAMLLoader(path_to_metadata)

    expected_metadata = {
        "pk_test": {
            "train_settings": {
                "source": "../data/pk_test.csv",
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
            "format": {}
        }
    }

    # Mock the open function, yaml.load, and validate_schema
    with patch("builtins.open", mock_open(read_data="data")):
        with patch("yaml.load") as mock_yaml_load:
            mock_yaml_load.return_value = expected_metadata
            metadata = loader.load_data()
            assert metadata == expected_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_by_yaml_loader_in_yml_format_without_validation(rp_logger):
    rp_logger.info("Loading metadata by YAMLLoader in '.yml' format")
    path_to_metadata = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/"
        "metadata/metadata_with_absent_destination.yaml"
    )
    loader = YAMLLoader(path_to_metadata)

    expected_metadata = {
        "pk_test": {
            "train_settings": {
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
            "source": "..\\data\\pk_test.csv",
            "format": {}
        }
    }

    # Mock the open function, yaml.load, and validate_schema
    with patch("builtins.open", mock_open(read_data="data")):
        with patch("yaml.load") as mock_yaml_load:
            mock_yaml_load.return_value = expected_metadata
            metadata = loader.load_data()
            assert metadata == expected_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_metadata_in_yaml_format(test_yaml_path, test_metadata_file, rp_logger):
    rp_logger.info("Saving metadata in yaml format")
    metadata_loader = MetadataLoader(test_yaml_path)
    assert isinstance(metadata_loader.metadata_loader, YAMLLoader)

    metadata_loader.save_data(test_metadata_file)
    assert metadata_loader.load_data() == {
        "global": {},
        "pk_test": {
            "train_settings": {
                "source": "..\\data\\pk_test.csv",
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_metadata_in_yml_format(test_yml_path, test_metadata_file, rp_logger):
    rp_logger.info("Saving metadata in yml format")
    metadata_loader = MetadataLoader(test_yml_path)
    assert isinstance(metadata_loader.metadata_loader, YAMLLoader)

    metadata_loader.save_data(test_metadata_file)
    assert metadata_loader.load_data() == {
        "global": {},
        "pk_test": {
            "train_settings": {
                "source": "..\\data\\pk_test.csv",
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_with_none_params_in_yaml_format(rp_logger):
    rp_logger.info("Loading metadata in yaml format with 'infer_settings', 'keys' defined as None")
    path_to_metadata = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                        "metadata/metadata_with_none_params.yaml")
    test_metadata_loader = MetadataLoader(path_to_metadata)

    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    metadata = test_metadata_loader.load_data()
    assert metadata == {
        "global": {},
        "pk_test": {
            "train_settings": {"source": "../data/pk_test.csv"},
            "infer_settings": {},
            "keys": {},
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_pipe_delimited_csv(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/pipe_delimited_text.csv")
    rp_logger.info("Loading CSV with pipe delimiter")
    global_context({"sep": "|", "quoting": "None"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_pipe_delimited_csv(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/"
                      "fixtures/csv_tables/pipe_delimited_text.csv")
    rp_logger.info("Saving CSV with pipe delimiter")
    format_params = {"sep": "|", "quoting": "None"}
    global_context(format_params)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"sep": "|", "quoting": 3, "skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_semicolon_delimited_csv(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/semicolon_delimited_text.csv")
    rp_logger.info("Loading CSV with semicolon delimiter")
    global_context({"sep": ";", "quoting": "None"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_semicolon_delimited_csv(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/semicolon_delimited_text.csv")
    rp_logger.info("Saving CSV with semicolon delimiter")
    format_settings = {"sep": ";", "quoting": "None"}
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"sep": ";", "quoting": 3, "skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_tab_delimited_csv(rp_logger):
    path_to_source = f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/tab_delimited_text.csv"
    rp_logger.info("Loading CSV with tab delimiter")
    global_context({"sep": "\t", "quoting": "None", "engine": "python"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_tab_delimited_csv(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/"
                      "fixtures/csv_tables/tab_delimited_text.csv")
    rp_logger.info("Saving CSV with tab delimiter")
    format_params = {"sep": "\t", "quoting": "None"}
    global_context(format_params)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"sep": "\t", "quoting": 3, "skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_multiline_bad_line_csv(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multiline_bad_line_text.csv")
    rp_logger.info("Loading CSV with multiline texts")
    global_context(
        {
            "sep": ",",
            "quoting": "all",
            "quotechar": '"',
            "escapechar": "\\",
            "skiprows": 1,
            "encoding": "utf-8",
            "on_bad_lines": "skip",
        }
    )
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (12, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_multiline_bad_line_csv(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multiline_bad_line_text.csv")
    rp_logger.info("Saving CSV with multiline texts")
    format_settings = {
        "sep": ",",
        "quoting": "all",
        "quotechar": '"',
        "escapechar": "\\",
        "skiprows": 1,
        "encoding": "utf-8",
        "on_bad_lines": "skip",
    }
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {
        "sep": ",",
        "quoting": 1,
        "quotechar": '"',
        "escapechar": "\\",
        "skiprows": None,
        "encoding": "utf-8",
        "on_bad_lines": "skip",
    }
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (12, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_double_quoted_csv(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/double_quoted_text.csv")
    rp_logger.info("Loading CSV with double quoted values")
    global_context({"sep": ",", "quotechar": '"'})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_double_quoted_csv(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/double_quoted_text.csv")
    rp_logger.info("Saving CSV with double quoted values")
    format_settings = {"sep": ",", "quotechar": '"'}
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {
        "sep": ",",
        "quotechar": '"',
        "quoting": 0,
        "skiprows": None,
    }
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_escaped_quoted_csv(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/escaped_quoted_text.csv")
    rp_logger.info("Loading CSV with escaped quoted values")
    global_context({"sep": ",", "quotechar": '"'})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_escaped_quoted_csv(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/escaped_quoted_text.csv")
    rp_logger.info("Saving CSV with escaped quoted values")
    format_settings = {"sep": ",", "quotechar": '"'}
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {
        "sep": ",",
        "quotechar": '"',
        "quoting": 0,
        "skiprows": None,
    }
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_without_header(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/text_without_header.csv")
    rp_logger.info("Loading CSV without the header")
    global_context(
        {
            "sep": ",",
            "header": None,
            "encoding": "ascii",
            "quoting": "non-numeric",
            "quotechar": '"',
        }
    )
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (7, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_without_header(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/text_without_header.csv")
    rp_logger.info("Saving CSV without the header")
    format_settings = {
        "sep": ",",
        "header": None,
        "encoding": "ascii",
        "quoting": "non-numeric",
        "quotechar": '"',
    }
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {
        "sep": ",",
        "header": None,
        "encoding": "ascii",
        "quoting": 2,
        "skiprows": None,
        "quotechar": '"',
    }
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (7, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_triple_colons(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multicolon_delimited_text.csv")
    rp_logger.info("Loading CSV contained the fields separated by triple colons")
    global_context({"sep": ":::", "quotechar": "None"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_triple_colons(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multicolon_delimited_text.csv")
    rp_logger.info("Saving CSV contained the fields with triple colons")
    format_settings = {"sep": ":::"}
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"sep": ",", "quoting": 0, "skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    global_context(format_settings)
    data, schema = CSVLoader(test_csv_path, sep=",").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_text_file(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_data.txt")
    rp_logger.info("Loading table with data in '.txt' format")
    global_context({})
    assert get_context().get_config() == {}
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_text_file(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_data.txt")
    rp_logger.info("Saving CSV table in '.txt' format")
    global_context({})
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_pcv_file(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/pipe_delimited_text.psv")
    rp_logger.info("Loading table with data in '.pcv' format")
    global_context({"sep": "|"})
    assert get_context().get_config() == {"sep": "|"}
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_pcv_file(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/pipe_delimited_text.psv")
    rp_logger.info("Saving CSV table in '.pcv' format")
    global_context({"sep": "|"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"sep": "|", "quoting": 0, "skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_tcv_file(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/tab_delimited_text.tsv")
    rp_logger.info("Loading table with data in '.tcv' format")
    global_context({"sep": "\t"})
    assert get_context().get_config() == {"sep": "\t"}
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_tcv_file(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/tab_delimited_text.tsv")
    rp_logger.info("Saving CSV table in '.tcv' format")
    global_context({"sep": "\t"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"sep": "\t", "quoting": 0, "skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_nested_field(rp_logger):
    path_to_source = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/text_contained_nested_field.csv"
    )
    rp_logger.info("Loading CSV with nested field")
    global_context({"quotechar": '"', "quoting": "minimal"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 5)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_nested_field(test_csv_path, rp_logger):
    path_to_source = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/text_contained_nested_field.csv"
    )
    rp_logger.info("Saving CSV with nested field")
    global_context({"quotechar": '"', "quoting": "minimal"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {
        "quotechar": '"',
        "quoting": 0,
        "skiprows": None,
    }
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    data, schema = CSVLoader(test_csv_path).load_data()
    assert data.shape == (15, 5)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_double_pipe_delimited_text(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/double_pipe_delimited_text.csv")
    rp_logger.info("Loading CSV with double pipe delimited text")
    global_context({"sep": r"\|\|"})
    data, schema = CSVLoader(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_double_pipe_delimited_text(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/double_pipe_delimited_text.csv")
    rp_logger.info("Saving CSV contained the fields with double pipe delimited text")
    format_settings = {"sep": r"\|\|"}
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {"sep": ",", "quoting": 0, "skiprows": None}
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    global_context(format_settings)
    data, schema = CSVLoader(test_csv_path, sep=",").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_na_values(rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_na_values.csv")
    rp_logger.info("Loading CSV contained NA values")
    global_context({"na_values": ["-", "Missing"]})
    data, schema = CSVLoader(path_to_source).load_data()

    assert data.shape == (10, 11)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_na_values(test_csv_path, rp_logger):
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_na_values.csv")
    rp_logger.info("Saving CSV contained NA values")
    format_settings = {
        "na_values": ["-", "Missing", "NaN"],
        "quoting": 0,
        "skiprows": None,
    }
    global_context(format_settings)
    data, schema = CSVLoader(path_to_source).load_data()
    assert get_context().get_config() == {
        "na_values": ["-", "Missing", "NaN"],
        "quoting": 0,
        "skiprows": None,
    }
    CSVLoader(test_csv_path).save_data(data, format=get_context().get_config())
    global_context(format_settings)
    data, schema = CSVLoader(test_csv_path, sep=",").load_data()
    assert data.shape == (10, 11)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_excel_table_with_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local Excel table with existed path"
    )
    path_to_table = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                     "excel_tables/table_with_data.xlsx")
    test_data_loader = DataLoader(path_to_table)
    assert test_data_loader.path == path_to_table
    assert test_data_loader.has_existed_path is True
    assert test_data_loader.original_schema is None
    assert isinstance(test_data_loader.file_loader, ExcelLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_excel_table_with_not_existed_path(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataLoader for local Excel table with invalid path"
    )
    test_data_loader = DataLoader("path/to/table.xlsx")
    assert test_data_loader.path == "path/to/table.xlsx"
    assert test_data_loader.has_existed_path is False
    assert test_data_loader.original_schema is None
    assert isinstance(test_data_loader.file_loader, ExcelLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path_to_table",
    [
        f"{DIR_NAME}/unit/data_loaders/fixtures/excel_tables/table_with_data.xls",
        f"{DIR_NAME}/unit/data_loaders/fixtures/excel_tables/table_with_data.xlsx",
    ],
)
def test_load_data_from_table_in_excel_format(path_to_table, rp_logger):
    rp_logger.info("Loading data from local Excel tables in '.xls', '.xlsx' format")
    data_loader = DataLoader(path_to_table)
    df, schema = data_loader.load_data()
    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert (
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "gender": [0, 1, 0, 1],
                    "height": [
                        157.18518021548246,
                        166.7731072622863,
                        162.91821942384928,
                        173.51448996432848,
                    ],
                    "id": [925, 84, 821, 383],
                }
            ),
        )
        is None
    )

    assert isinstance(df, pd.DataFrame)
    assert schema == CSV_SCHEMA
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_excel_format_from_1_sheet_of_2_sheets(rp_logger):
    rp_logger.info("Loading data from local Excel table in '.xls' format from the certain sheet")
    global_context({"sheet_name": "TestName"})
    data_loader = DataLoader(
        f"{DIR_NAME}/unit/data_loaders/fixtures/"
        "excel_tables/table_with_data_and_2_sheets.xls"
    )
    df, schema = data_loader.load_data()
    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert (
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "gender": [0, 1, 0, 1],
                    "height": [157.1852, 166.7731, 162.9182, 173.5145],
                    "id": [925, 84, 821, 383],
                }
            ),
        )
        is None
    )

    assert isinstance(df, pd.DataFrame)
    assert schema == CSV_SCHEMA
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_excel_format_from_2_sheets_of_2_sheets(rp_logger):
    rp_logger.info(
        "Loading data from local Excel table in '.xlsx' format that contains 2 non-empty sheets"
    )
    global_context({"sheet_name": None})
    data_loader = DataLoader(
        f"{DIR_NAME}/unit/data_loaders/fixtures/"
        "excel_tables/table_with_data_and_2_nonempty_sheets.xlsx"
    )
    df, schema = data_loader.load_data()
    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert (
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "gender": [0, 1, 0, 1],
                    "height": [162.9182, 173.5145, 157.1852, 166.7731],
                    "id": [821, 383, 925, 84],
                }
            ),
        )
        is None
    )

    assert isinstance(df, pd.DataFrame)
    assert schema == CSV_SCHEMA
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_excel_table(rp_logger, caplog):
    rp_logger.info("Loading data from local empty Excel table in '.xlsx' format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "excel_tables/empty_table.xlsx")
    assert isinstance(data_loader.file_loader, ExcelLoader)
    data, schema = data_loader.load_data()
    assert data.empty is True
    assert schema == CSV_SCHEMA
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_column_from_table_in_xlsx_format(rp_logger):
    rp_logger.info("Get the list of the columns from the table in '.xlsx' format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "excel_tables/table_with_data.xlsx")
    columns = data_loader.get_columns()
    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert columns == ["gender", "height", "id"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_column_from_table_in_xls_format(rp_logger):
    rp_logger.info("Get the list of the columns from the table in '.xls' format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "excel_tables/table_with_data.xls")
    columns = data_loader.get_columns()
    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert columns == ["gender", "height", "id"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_column_from_table_in_xls_format_with_formatting_settings(rp_logger):
    rp_logger.info(
        "Get the list of the columns from the table in '.xls' format from the certain sheet"
    )
    data_loader = DataLoader(
        f"{DIR_NAME}/unit/data_loaders/fixtures/"
        "excel_tables/table_with_data_and_2_sheets.xls"
    )
    columns = data_loader.get_columns(sheet_name="TestName")
    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert columns == ["gender", "height", "id"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_columns_from_empty_excel_table(caplog, rp_logger):
    rp_logger.info("Get the list of the columns from the empty table in '.xlsx' format")
    data_loader = DataLoader(f"{DIR_NAME}/unit/data_loaders/fixtures/"
                             "excel_tables/empty_table.xlsx")
    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert data_loader.get_columns() == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_excel_table_in_xlx_format(test_xls_path, test_df, rp_logger):
    rp_logger.info("Saving the data to Excel table in '.xls' format")
    data_loader = DataLoader(test_xls_path)
    data_loader.save_data(test_df)

    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert os.path.exists(test_xls_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema == CSV_SCHEMA
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_excel_table_in_xlsx_format(test_xlsx_path, test_df, rp_logger):
    rp_logger.info("Saving the data to Excel table in '.xlsx' format")
    data_loader = DataLoader(test_xlsx_path)
    data_loader.save_data(test_df)

    assert isinstance(data_loader.file_loader, ExcelLoader)
    assert os.path.exists(test_xlsx_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema == CSV_SCHEMA
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_data_encryptor_with_valid_path_and_key(data_encryptor, rp_logger):
    rp_logger.info("Test the initialization of DataEncryptor with valid path and key")
    assert isinstance(data_encryptor, DataEncryptor)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_fernet_key_validation(valid_fernet_key, rp_logger):
    rp_logger.info("Test the validation of the valid Fernet key")
    assert DataEncryptor._validate_fernet_key(valid_fernet_key) is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_round_encrypt_decrypt_data(data_encryptor, valid_simple_dataframe, rp_logger):
    rp_logger.info(
        "Test the round encryption and decryption of the dataframe "
        "with the provided valid Ferney key"
    )
    data_encryptor.save_data(valid_simple_dataframe)
    loaded_df, _ = data_encryptor.load_data()
    pd.testing.assert_frame_equal(loaded_df, valid_simple_dataframe)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_data_encryption(data_encryptor, rp_logger):
    rp_logger.info(
        "Test the method '_check_if_data_encrypted' of the DataEncryptor "
        "with the provided valid path"
    )
    data_encryptor._check_if_data_encrypted()
    assert data_encryptor.path.endswith(".dat")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "invalid_key", [
        "short_key", "a" * 45, "@w3n7X7VO@i0xEHf@fo@rtEa@vgfWW3GZAtmZd@BzlA@"
    ]
)
def test_invalid_fernet_key_validation(invalid_key, rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the Fernet key with the invalid length"
    )

    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            DataEncryptor._validate_fernet_key(invalid_key)
        assert (
                "It seems that the provided Fernet key is invalid. "
                "The Fernet key must be 32 url-safe base64-encoded bytes"
        ) in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_decrypt_data_with_invalid_key(data_encryptor, valid_simple_dataframe, rp_logger, caplog):
    rp_logger.info(
        "Test the decryption of the data with the invalid Fernet key"
    )
    data_encryptor.save_data(valid_simple_dataframe)
    data_encryptor.fernet = Fernet(Fernet.generate_key())  # Change the key
    with pytest.raises(InvalidToken):
        with caplog.at_level("ERROR"):
            data_encryptor.load_data()
        assert (
            "It seems that the decryption process failed due to the following reasons - "
            "the provided Fernet key is invalid or the encrypted data is corrupted"
        ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_data_encryption_with_incorrect_extension(tmp_path, rp_logger, caplog):
    rp_logger.info(
        "Test the method '_check_if_data_encrypted' of the DataEncryptor "
        "with the provided invalid path"
    )
    file_path = tmp_path / "test.txt"
    data_encryptor = DataEncryptor(path=str(file_path))
    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            data_encryptor._check_if_data_encrypted()
        assert (
            "It seems that the decryption process failed "
            "due the data hasn't been encrypted despite the Fernet key presence."
        ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)
