from unittest.mock import patch, mock_open
import pytest
import os

import pandas as pd
from pandas.testing import assert_frame_equal

from syngen.ml.data_loaders import (
    DataLoader,
    CSVLoader,
    AvroLoader,
    BinaryLoader,
    MetadataLoader,
    YAMLLoader
)
from syngen.ml.context import global_context, get_context
from tests.conftest import SUCCESSFUL_MESSAGE


def test_initialize_data_loader_for_local_csv_table_with_existed_path(rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local csv table with existed path")
    test_data_loader = DataLoader("tests/unit/data_loaders/fixtures/csv_tables/table_with_data.csv")
    assert test_data_loader.path == "tests/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
    assert test_data_loader.has_existed_path is True
    assert isinstance(test_data_loader.file_loader, CSVLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_csv_table_with_not_existed_path(rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local csv table with invalid path")
    test_data_loader = DataLoader("path/to/table.csv")
    assert test_data_loader.path == "path/to/table.csv"
    assert test_data_loader.has_existed_path is False
    assert isinstance(test_data_loader.file_loader, CSVLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_avro_table_with_existed_path(rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local avro table with existed path")
    test_data_loader = DataLoader("tests/unit/data_loaders/fixtures/avro_tables/table_with_data.avro")
    assert test_data_loader.path == "tests/unit/data_loaders/fixtures/avro_tables/table_with_data.avro"
    assert test_data_loader.has_existed_path is True
    assert isinstance(test_data_loader.file_loader, AvroLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_avro_table_with_not_existed_path(rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local avro table with invalid path")
    test_data_loader = DataLoader("path/to/table.avro")
    assert test_data_loader.path == "path/to/table.avro"
    assert test_data_loader.has_existed_path is False
    assert isinstance(test_data_loader.file_loader, AvroLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_pickle_table_with_existed_path(rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local pickle table with existed path")
    test_data_loader = DataLoader("tests/unit/data_loaders/fixtures/pickle_tables/table_with_data.pkl")
    assert test_data_loader.path == "tests/unit/data_loaders/fixtures/pickle_tables/table_with_data.pkl"
    assert test_data_loader.has_existed_path is True
    assert isinstance(test_data_loader.file_loader, BinaryLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_pickle_table_with_not_existed_path(rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local pickle table with invalid path")
    test_data_loader = DataLoader("path/to/table.pkl")
    assert test_data_loader.path == "path/to/table.pkl"
    assert test_data_loader.has_existed_path is False
    assert isinstance(test_data_loader.file_loader, BinaryLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_table_in_unsupported_format(rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local table in unsupported format")
    with pytest.raises(NotImplementedError) as error:
        DataLoader("path/to/table.test")
        assert str(error.value) == "File format not supported"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_table_with_empty_path(caplog, rp_logger):
    rp_logger.info("Initializing the instance of the class DataLoader for local table with empty path")
    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            DataLoader("")
        assert "It seems that the information of source is absent" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_csv_format(rp_logger):
    rp_logger.info("Loading data from local table in csv format")
    data_loader = DataLoader("tests/unit/data_loaders/fixtures/csv_tables/table_with_data.csv")
    df, schema = data_loader.load_data()
    assert isinstance(data_loader.file_loader, CSVLoader)
    assert assert_frame_equal(
        df, pd.DataFrame(
            {
                "gender": [0, 1, 0, 1],
                "height": [157.18518021548246, 166.7731072622863, 162.91821942384928, 173.51448996432848],
                "id": [925, 84, 821, 383]
            }
        )
    ) is None

    assert isinstance(df, pd.DataFrame)
    assert schema == {"fields": {}, "format": "CSV"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_csv_format_in_not_utf_8(caplog, rp_logger):
    rp_logger.info("Loading data from local table in csv format in not 'utf-8' encoding")
    path = "tests/unit/data_loaders/fixtures/csv_tables/table_in_iso_encoding.csv"
    data_loader = DataLoader(path)
    assert isinstance(data_loader.file_loader, CSVLoader)

    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            data_loader.load_data()
        assert f"It seems that the content of the data in the path - '{path}' " \
               f"doesn't have the encoding UTF-8. The details of the error - {error}.\n" \
               f"Please, use the data in UTF-8 encoding" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_csv_format(caplog, rp_logger):
    rp_logger.info("Loading data from local empty table in csv format")
    data_loader = DataLoader("tests/unit/data_loaders/fixtures/csv_tables/empty_table.csv")
    assert isinstance(data_loader.file_loader, CSVLoader)
    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            data_loader.load_data()
            assert "The empty file was provided. Unable to load data " in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_csv_format(test_csv_path, test_df, rp_logger):
    rp_logger.info("Saving data in csv format locally")
    data_loader = DataLoader(test_csv_path)
    data_loader.save_data(test_csv_path, test_df)

    assert isinstance(data_loader.file_loader, CSVLoader)
    assert os.path.exists(test_csv_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema == {"fields": {}, "format": "CSV"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_avro_format(rp_logger):
    rp_logger.info("Loading data from local table in avro format")
    data_loader = DataLoader("tests/unit/data_loaders/fixtures/avro_tables/table_with_data.avro")
    df, schema = data_loader.load_data()

    assert isinstance(data_loader.file_loader, AvroLoader)
    assert assert_frame_equal(
        df, pd.DataFrame(
                {
                    "gender": [0, 1, 0, 1],
                    "height": [157.18518021548246, 166.7731072622863, 162.91821942384928, 173.51448996432848],
                    "id": [925, 84, 821, 383]
                }
        )
    ) is None

    assert isinstance(df, pd.DataFrame)
    assert schema == {
        "fields": {
            "gender": "int",
            "height": "float",
            "id": "int"
        },
        "format": "Avro"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_avro_format(caplog, rp_logger):
    rp_logger.info("Loading data from local empty table in avro format")
    path = "tests/unit/data_loaders/fixtures/avro_tables/empty_table.avro"
    data_loader = DataLoader(path)

    assert isinstance(data_loader.file_loader, AvroLoader)

    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            data_loader.load_data()
            assert "The empty file was provided. Unable to load data " in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_avro_format(test_avro_path, test_df, rp_logger):
    rp_logger.info("Saving data in avro format locally")
    data_loader = DataLoader(test_avro_path)
    data_loader.save_data(test_avro_path, test_df)

    assert isinstance(data_loader.file_loader, AvroLoader)
    assert os.path.exists(test_avro_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema == {
        "fields": {
            "gender": "int",
            "height": "float",
            "id": "int"
        },
        "format": "Avro"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_pickle_format(rp_logger):
    rp_logger.info("Loading data from local table in pickle format")
    data_loader = DataLoader("tests/unit/data_loaders/fixtures/pickle_tables/table_with_data.pkl")
    df, schema = data_loader.load_data()

    assert isinstance(data_loader.file_loader, BinaryLoader)
    assert assert_frame_equal(
        df, pd.DataFrame(
            {
                "gender": [0, 1, 0, 1],
                "height": [157.18518021548246, 166.7731072622863, 162.91821942384928, 173.51448996432848],
                "id": [925, 84, 821, 383]
            }
        )
    ) is None

    assert isinstance(df, pd.DataFrame)
    assert schema is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_pickle_format(caplog, rp_logger):
    rp_logger.info("Loading data from local empty table in pickle format")
    path = r"C:\Users\Hanna_Imshenetska\Projects\syngen_open_source\tdm_syngen\src\tests\unit\data_loaders\fixtures\pickle_tables\empty_table.pkl"
    data_loader = DataLoader(path)

    assert isinstance(data_loader.file_loader, BinaryLoader)

    with caplog.at_level("ERROR"):
        data_loader.load_data()
        assert "Loading data from local empty table in pickle format" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_pickle_format(test_pickle_path, test_df, rp_logger):
    rp_logger.info("Saving data in pickle format locally")
    data_loader = DataLoader(test_pickle_path)
    data_loader.save_data(test_pickle_path, test_df)

    assert isinstance(data_loader.file_loader, BinaryLoader)
    assert os.path.exists(test_pickle_path) is True

    loaded_df, schema = data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert schema is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_metadata_loader(rp_logger):
    rp_logger.info("Initializing metadata loader")
    test_metadata_loader = MetadataLoader("tests/unit/data_loaders/fixtures/metadata/metadata.yaml")
    assert test_metadata_loader.metadata_path == "tests/unit/data_loaders/fixtures/metadata/metadata.yaml"
    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_metadata_loader_in_unsupported_format(rp_logger):
    rp_logger.info("Initializing metadata loader in unsupported format")
    with pytest.raises(NotImplementedError) as error:
        MetadataLoader("path/to/table.test")
        assert str(error.value) == "The format of metadata isn't supported"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_in_yaml_format(rp_logger):
    rp_logger.info("Loading metadata in yaml format")
    path_to_metadata = "tests/unit/data_loaders/fixtures/metadata/metadata.yaml"
    test_metadata_loader = MetadataLoader(path_to_metadata)

    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    metadata = test_metadata_loader.load_data()
    assert metadata == {
        "global": {},
        "pk_test": {
            "train_settings": {
                "drop_null": False,
                "epochs": 1,
                "print_report": False,
                "row_limit": 800
            },
            "infer_settings": {
                "print_report": True,
                "random_seed": 1,
                "run_parallel": False,
                "size": 100
            },
            "keys": {
                "pk_id": {
                    "columns": ["Id"],
                    "type": "PK"
                }
            },
            "source": r"..\data\pk_test.csv"
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_in_yml_format(rp_logger):
    rp_logger.info("Loading metadata in yml format")
    path_to_metadata = "tests/unit/data_loaders/fixtures/metadata/metadata.yml"
    test_metadata_loader = MetadataLoader(path_to_metadata)

    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    metadata = test_metadata_loader.load_data()
    assert metadata == {
        "global": {},
        "pk_test": {
            "train_settings": {
                "drop_null": False,
                "epochs": 1,
                "print_report": False,
                "row_limit": 800
            },
            "infer_settings": {
                "print_report": True,
                "random_seed": 1,
                "run_parallel": False,
                "size": 100
            },
            "keys": {
                "pk_id": {
                    "columns": ["Id"],
                    "type": "PK"
                }
            },
            "source": r"..\data\pk_test.csv"
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_by_yaml_loader_in_yaml_format(rp_logger):
    rp_logger.info("Loading metadata by yaml loader in yaml format")
    loader = YAMLLoader()

    expected_metadata = {
        "pk_test": {
            "train_settings": {
                "drop_null": False,
                "epochs": 1,
                "print_report": False,
                "row_limit": 800
            },
            "infer_settings": {
                "print_report": True,
                "random_seed": 1,
                "run_parallel": False,
                "size": 100
            },
            "keys": {
                "pk_id": {
                    "columns": ["Id"],
                    "type": "PK"
                }
            },
            "source": "..\\data\\pk_test.csv"
        }
    }

    # Mock the open function, yaml.load, and validate_schema
    with patch('builtins.open', mock_open(read_data='data')) as m:
        with patch('yaml.load') as mock_yaml_load:
            with patch('syngen.ml.validation_schema.ValidationSchema.validate_schema') as mock_validate_schema:
                mock_yaml_load.return_value = expected_metadata
                metadata = loader.load_data('tests/unit/data_loaders/fixtures/metadata/metadata.yaml')

                # Assert that validate_schema was called with the correct arguments
                # mock_validate_schema.assert_called_once_with(configuration_schema, expected_metadata)
                assert metadata == expected_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_by_yaml_loader_in_yml_format(rp_logger):
    rp_logger.info("Loading metadata by yaml loader in yml format")
    loader = YAMLLoader()

    expected_metadata = {
        "pk_test": {
            "train_settings": {
                "drop_null": False,
                "epochs": 1,
                "print_report": False,
                "row_limit": 800
            },
            "infer_settings": {
                "print_report": True,
                "random_seed": 1,
                "run_parallel": False,
                "size": 100
            },
            "keys": {
                "pk_id": {
                    "columns": ["Id"],
                    "type": "PK"
                }
            },
            "source": "..\\data\\pk_test.csv"
        }
    }

    # Mock the open function, yaml.load, and validate_schema
    with patch('builtins.open', mock_open(read_data='data')) as m:
        with patch('yaml.load') as mock_yaml_load:
            with patch('syngen.ml.validation_schema.ValidationSchema.validate_schema') as mock_validate_schema:
                mock_yaml_load.return_value = expected_metadata
                metadata = loader.load_data('tests/unit/data_loaders/fixtures/metadata/metadata.yml')

                # Assert that validate_schema was called with the correct arguments
                # mock_validate_schema.assert_called_once_with(configuration_schema, expected_metadata)
                assert metadata == expected_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_metadata_in_yaml_format(test_yaml_path, test_df, rp_logger):
    rp_logger.info("Saving metadata in yaml format")
    metadata_loader = MetadataLoader(test_yaml_path)
    assert isinstance(metadata_loader.metadata_loader, YAMLLoader)

    with pytest.raises(NotImplementedError) as error:
        metadata_loader.save_data(test_yaml_path, test_df)
        assert str(error) == "Saving YAML files is not supported"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_metadata_in_yml_format(test_yml_path, test_df, rp_logger):
    rp_logger.info("Saving metadata in yml format")
    metadata_loader = MetadataLoader(test_yml_path)
    assert isinstance(metadata_loader.metadata_loader, YAMLLoader)

    with pytest.raises(NotImplementedError) as error:
        metadata_loader.save_data(test_yml_path, test_df)
        assert str(error) == "Saving YAML files is not supported"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_metadata_with_none_params_in_yaml_format(rp_logger):
    rp_logger.info("Loading metadata in yaml format with 'train_settings', 'infer_settings' defined as None")
    path_to_metadata = "tests/unit/data_loaders/fixtures/metadata/metadata_with_none_params.yaml"
    test_metadata_loader = MetadataLoader(path_to_metadata)

    assert isinstance(test_metadata_loader.metadata_loader, YAMLLoader)
    metadata = test_metadata_loader.load_data()
    assert metadata == {
        "global": {},
        "pk_test": {
            "train_settings": {},
            "infer_settings": {},
            "source": r"..\data\pk_test.csv",
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["Id"]
                }
            }
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_pipe_delimited_csv(rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/pipe_delimited_text.csv"
    rp_logger.info(f"Loading CSV with pipe delimiter")
    global_context({"sep": "|", "quoting": "None"})
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_pipe_delimited_csv(test_csv_path, rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/pipe_delimited_text.csv"
    rp_logger.info(f"Saving CSV with pipe delimiter")
    format_params = {"sep": "|", "quoting": "None"}
    global_context(format_params)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": "|",
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_semicolon_delimited_csv(rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/semicolon_delimited_text.csv"
    rp_logger.info(f"Loading CSV with semicolon delimiter")
    global_context({"sep": ";",  "quoting": "None"})
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_semicolon_delimited_csv(test_csv_path, rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/semicolon_delimited_text.csv"
    rp_logger.info(f"Saving CSV with semicolon delimiter")
    format_settings = {"sep": ";",  "quoting": "None"}
    global_context(format_settings)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": ";",
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_tab_delimited_csv(rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/tab_delimited_text.csv"
    rp_logger.info(f"Loading CSV with tab delimiter")
    global_context({"sep": "\t", "quoting": "None", "engine": "python"})
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_tab_delimited_csv(test_csv_path, rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/tab_delimited_text.csv"
    rp_logger.info(f"Saving CSV with tab delimiter")
    format_params = {"sep": "\t", "quoting": "None"}
    global_context(format_params)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": "\t",
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_multiline_bad_line_csv(rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/multiline_bad_line_text.csv"
    rp_logger.info(f"Loading CSV with multiline texts")
    global_context({
        "sep": ",",
        "quoting": "all",
        "quotechar": '"',
        "escapechar": "\\",
        "skiprows": 1,
        "encoding": "utf-8",
        "on_bad_lines": "skip"
    })
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 12
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_multiline_bad_line_csv(test_csv_path, rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/multiline_bad_line_text.csv"
    rp_logger.info(f"Saving CSV with multiline texts")
    format_settings = {
        "sep": ",",
        "quoting": "all",
        "quotechar": '"',
        "escapechar": "\\",
        "skiprows": 1,
        "encoding": "utf-8",
        "on_bad_lines": "skip"
    }
    global_context(format_settings)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": ",",
        "quoting": 1,
        "quotechar": '"',
        "escapechar": "\\",
        "skiprows": None,
        "encoding": "utf-8",
        "on_bad_lines": "skip"
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 12
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_double_quoted_csv(rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/double_quoted_text.csv"
    rp_logger.info(f"Loading CSV with double quoted values")
    global_context({"sep": ",", "quotechar": '"'})
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_double_quoted_csv(test_csv_path, rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/double_quoted_text.csv"
    rp_logger.info(f"Saving CSV with double quoted values")
    format_settings = {"sep": ",", "quotechar": '"'}
    global_context(format_settings)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": ",",
        "quotechar": '"',
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_escaped_quoted_csv(rp_logger):
    path_source = "tests/unit/data_loaders/fixtures/csv_tables/escaped_quoted_text.csv"
    rp_logger.info(f"Loading CSV with escaped quoted values")
    global_context({"sep": ",", "quotechar": '"'})
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_without_header(rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/text_without_header.csv"
    rp_logger.info(f"Loading CSV without the header")
    global_context({"sep": ",", "header": None, "encoding": "ascii"})
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 7
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_without_header(test_csv_path, rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/text_without_header.csv"
    rp_logger.info(f"Saving CSV without the header")
    format_settings = {
        "sep": ",",
        "header": None,
        "encoding": "ascii"
    }
    global_context(format_settings)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": ",",
        "header": None,
        "encoding": "ascii",
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 7
    rp_logger.info(SUCCESSFUL_MESSAGE)

def test_load_csv_with_json(rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/text_contained_nested_structures.csv"
    rp_logger.info(f"Loading CSV contained the fields with nested structures")
    global_context({
        "sep": ",",
        "quotechar": '"',
        "quoting": "non-numeric",
        "escapechar": "\\",
        "encoding": "utf-8",
        "on_bad_lines": "skip"
    })
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 8
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_nested_structures(test_csv_path, rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/text_contained_nested_structures.csv"
    rp_logger.info(f"Saving CSV contained the fields with nested structures")
    format_settings = {
        "sep": ",",
        "quotechar": '"',
        "quoting": "non-numeric",
        "escapechar": "\\",
        "encoding": "utf-8",
        "on_bad_lines": "skip"
    }
    global_context(format_settings)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": ",",
        "quotechar": '"',
        "quoting": 2,
        "escapechar": "\\",
        "encoding": "utf-8",
        "on_bad_lines": "skip",
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 8
    rp_logger.info(SUCCESSFUL_MESSAGE)



def test_load_csv_with_triple_colons(rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/multicolon_delimited_text.csv"
    rp_logger.info(f"Loading CSV contained the fields separated by triple colons")
    global_context({
        "sep": ":::",
        "quotechar": "None"
    })
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_triple_colons(test_csv_path, rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/multicolon_delimited_text.csv"
    rp_logger.info(f"Saving CSV contained the fields with triple colons")
    format_settings = {
        "sep": ":::",
        "quotechar": "None"
    }
    global_context(format_settings)
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": ",",
        "quotechar": "None",
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    global_context(format_settings)
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_text_file(rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/table_with_data.txt"
    rp_logger.info(f"Loading table with data in '.txt' format")
    global_context({})
    assert get_context().get_config() == {}
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_text_file(test_csv_path, rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/table_with_data.txt"
    rp_logger.info(f"Saving CSV table in '.txt' format")
    global_context({})
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {"skiprows": None}
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_pcv_file(rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/pipe_delimited_text.psv"
    rp_logger.info(f"Loading table with data in '.pcv' format")
    global_context({"sep": "|"})
    assert get_context().get_config() == {"sep": "|"}
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_pcv_file(test_csv_path, rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/pipe_delimited_text.psv"
    rp_logger.info(f"Saving CSV table in '.pcv' format")
    global_context({"sep": "|"})
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": "|",
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_tcv_file(rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/tab_delimited_text.tsv"
    rp_logger.info(f"Loading table with data in '.tcv' format")
    global_context({"sep": "\t"})
    assert get_context().get_config() == {"sep": "\t"}
    data, schema = CSVLoader().load_data(path_source)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_tcv_file(test_csv_path, rp_logger):
    path_source = "./tests/unit/data_loaders/fixtures/csv_tables/tab_delimited_text.tsv"
    rp_logger.info(f"Saving CSV table in '.tcv' format")
    global_context({"sep": "\t"})
    data, schema = CSVLoader().load_data(path_source)
    assert get_context().get_config() == {
        "sep": "\t",
        "quoting": 3,
        "skiprows": None
    }
    CSVLoader().save_data(test_csv_path, data, format=get_context().get_config())
    data, schema = CSVLoader().load_data(test_csv_path)
    assert data.count().max() == 15
    rp_logger.info(SUCCESSFUL_MESSAGE)