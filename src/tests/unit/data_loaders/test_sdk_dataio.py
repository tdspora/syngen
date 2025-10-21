import pytest
import os

import pandas as pd
from pandas.testing import assert_frame_equal
from cryptography.fernet import Fernet

from syngen.ml.data_loaders import (
    CSVLoader,
    AvroLoader,
    BinaryLoader,
    ExcelLoader,
    DataEncryptor
)
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME
from syngen.sdk import DataIO


def test_initialize_dataio_for_local_csv_table(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataIO for the local CSV table"
    )
    path_to_table = f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
    dataio_loader = DataIO(path_to_table)
    assert dataio_loader.path == path_to_table
    assert dataio_loader.fernet_key is None
    assert dataio_loader.metadata == {
        "table": {
            "train_settings": {"source": path_to_table},
            "encryption": {"fernet_key": None},
            "format": {}
        },
    }
    assert isinstance(dataio_loader.data_loader.file_loader, CSVLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_dataio_for_local_avro_table(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataIO for the local AVRO table"
    )
    path_to_table = f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/table_with_data.avro"
    dataio_loader = DataIO(path_to_table)
    assert dataio_loader.path == path_to_table
    assert dataio_loader.fernet_key is None
    assert dataio_loader.metadata == {
        "table": {
            "train_settings": {"source": path_to_table},
            "encryption": {"fernet_key": None},
            "format": {}
        },
    }
    assert isinstance(dataio_loader.data_loader.file_loader, AvroLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_dataio_for_local_pickle_table(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataIO for the local pickle table"
    )
    path_to_table = f"{DIR_NAME}/unit/data_loaders/fixtures/pickle_tables/table_with_data.pkl"
    dataio_loader = DataIO(path_to_table)
    assert dataio_loader.path == path_to_table
    assert dataio_loader.fernet_key is None
    assert dataio_loader.metadata == {
        "table": {
            "train_settings": {"source": path_to_table},
            "encryption": {"fernet_key": None},
            "format": {}
        },
    }
    assert isinstance(dataio_loader.data_loader.file_loader, BinaryLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_dataio_for_local_table_in_unsupported_format(rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataIO "
        "for the local table in the unsupported format"
    )
    with pytest.raises(NotImplementedError) as error:
        DataIO("path/to/table.test")
        assert str(error.value) == "File format not supported for extension '.test'"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_dataio_for_local_table_with_empty_path(caplog, rp_logger):
    rp_logger.info(
        "Initializing the instance of the class DataIO for the local table with the empty path"
    )
    with pytest.raises(NotImplementedError):
        with caplog.at_level("ERROR"):
            DataIO("")
            assert "File format not supported for extension" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_csv_format(rp_logger):
    rp_logger.info("Loading data from the local table in the '.csv' format")
    dataio_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
    )
    df = dataio_loader.load_data()
    assert isinstance(dataio_loader.data_loader.file_loader, CSVLoader)
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_schema_from_table_in_csv_format(rp_logger):
    rp_logger.info("Loading schema from the local table in the '.csv' format")
    dataio_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
    )
    schema = dataio_loader.load_schema()
    assert schema is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_csv_format_in_not_utf_8(caplog, rp_logger):
    rp_logger.info(
        "Loading data from the local table in the '.csv' format in not 'utf-8' encoding"
    )
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_in_iso_encoding.csv"
    dataio_loader = DataIO(path)
    assert isinstance(dataio_loader.data_loader.file_loader, CSVLoader)

    with pytest.raises(UnicodeDecodeError) as error:
        with caplog.at_level("ERROR"):
            dataio_loader.load_data()
            assert (
                f"It seems that the content of the data in the path - '{path}' "
                f"doesn't have the encoding UTF-8. The details of the error - {error}.\n"
                f"Please, use the data in UTF-8 encoding" in caplog.text
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_csv_format(caplog, rp_logger):
    rp_logger.info("Loading data from the local empty table in the '.csv' format")
    dataio_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/empty_table.csv"
    )
    assert isinstance(dataio_loader.data_loader.file_loader, CSVLoader)
    with pytest.raises(pd.errors.EmptyDataError):
        with caplog.at_level("ERROR"):
            dataio_loader.load_data()
            assert "The empty file was provided. Unable to load data" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_csv_format(test_csv_path, test_df, rp_logger):
    rp_logger.info("Saving data in the '.csv' format locally")
    dataio_loader = DataIO(test_csv_path)
    dataio_loader.save_data(test_df)

    assert isinstance(dataio_loader.data_loader.file_loader, CSVLoader)
    assert os.path.exists(test_csv_path) is True

    loaded_df = dataio_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_avro_format(rp_logger):
    rp_logger.info("Loading data from the local table in the '.avro' format")
    dataio_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/table_with_data.avro"
    )
    df = dataio_loader.load_data()

    assert isinstance(dataio_loader.data_loader.file_loader, AvroLoader)
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_schema_from_table_in_avro_format(rp_logger):
    rp_logger.info("Loading the schema from the local table in the '.avro' format")
    dataio_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/table_with_data.avro"
    )

    assert dataio_loader.load_schema() == {
        "fields": [
            {"name": "gender", "type": ["null", "long"]},
            {"name": "height", "type": ["null", "double"]},
            {"name": "id", "type": ["null", "long"]}
        ],
        "name": "Root",
        "type": "record"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_avro_format(caplog, rp_logger):
    rp_logger.info("Loading data from the local empty table in the '.avro' format")
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/empty_table.avro"
    dataio_loader = DataIO(path)

    assert isinstance(dataio_loader.data_loader.file_loader, AvroLoader)

    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            dataio_loader.load_data()
            assert "cannot read header - is it an avro file?" in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_schema_from_empty_table_in_avro_format(caplog, rp_logger):
    rp_logger.info("Loading the schema from the local empty table in the '.avro' format")
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/avro_tables/empty_table.avro"
    dataio_loader = DataIO(path)

    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            dataio_loader.load_schema()
            assert "cannot read header - is it an avro file?" in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_avro_format(test_avro_path, test_df, test_avro_schema, rp_logger):
    rp_logger.info("Saving data in the '.avro' format locally")
    dataio_loader = DataIO(test_avro_path)
    dataio_loader.save_data(test_df, schema=test_avro_schema)

    assert isinstance(dataio_loader.data_loader.file_loader, AvroLoader)
    assert os.path.exists(test_avro_path) is True

    loaded_df = dataio_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    assert dataio_loader.load_schema() == test_avro_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_avro_format_without_provided_schema(test_avro_path, test_df, rp_logger):
    rp_logger.info("Saving data in the '.avro' format locally without a provided schema")
    dataio_loader = DataIO(test_avro_path)
    dataio_loader.save_data(test_df, schema=None)

    assert isinstance(dataio_loader.data_loader.file_loader, AvroLoader)
    assert os.path.exists(test_avro_path) is True

    loaded_df = dataio_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_pickle_format(rp_logger):
    rp_logger.info("Loading data from the local table in the '.pkl' format")
    dataio_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/pickle_tables/table_with_data.pkl"
    )
    df = dataio_loader.load_data()

    assert isinstance(dataio_loader.data_loader.file_loader, BinaryLoader)
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_table_in_pickle_format(caplog, rp_logger):
    rp_logger.info("Loading data from the local empty table in the '.pkl' format")
    path = f"{DIR_NAME}/unit/data_loaders/fixtures/pickle_tables/empty_table.pkl"
    dataio_loader = DataIO(path)

    assert isinstance(dataio_loader.data_loader.file_loader, BinaryLoader)

    data = dataio_loader.load_data()
    assert data == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_data_in_pickle_format(test_pickle_path, test_df, rp_logger):
    rp_logger.info("Saving data in the '.pkl' format locally")
    dataio_loader = DataIO(test_pickle_path)
    dataio_loader.save_data(test_df)

    assert isinstance(dataio_loader.data_loader.file_loader, BinaryLoader)
    assert os.path.exists(test_pickle_path) is True

    loaded_df = dataio_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_pipe_delimited_csv(rp_logger):
    rp_logger.info("Loading data in the '.csv' format with the pipe delimiter")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/pipe_delimited_text.csv")
    dataio_loader = DataIO(path_to_source, sep="|", quoting="none")
    data = dataio_loader.load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_pipe_delimited_csv(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.csv' format with the pipe delimiter")
    path_to_source = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/pipe_delimited_text.csv"
    )
    data = DataIO(path_to_source, sep="|", quoting="none", skiprows=None).load_data()
    DataIO(test_csv_path, sep="|", quoting="none", skiprows=None).save_data(data)
    data = DataIO(test_csv_path, sep="|", quoting="none", skiprows=None).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_semicolon_delimited_csv(rp_logger):
    rp_logger.info("Loading data in the '.csv' format with the semicolon delimiter")
    path_to_source = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/semicolon_delimited_text.csv"
    )
    data = DataIO(path_to_source, sep=";", quoting="none").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_semicolon_delimited_csv(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.csv' format with semicolon delimiter")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/semicolon_delimited_text.csv")
    data = DataIO(path_to_source, sep=";", quoting="none").load_data()
    DataIO(test_csv_path, sep=";", quoting="none").save_data(data)
    data = DataIO(test_csv_path, sep=";", quoting="none").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_tab_delimited_csv(rp_logger):
    rp_logger.info("Loading data in the '.csv' format with tab delimiter")
    path_to_source = f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/tab_delimited_text.csv"
    data = DataIO(path_to_source, sep="\t", quoting="none", engine="python").load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_tab_delimited_csv(test_csv_path, rp_logger):
    rp_logger.info("Saving data in '.csv' format with tab delimiter")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/"
                      "fixtures/csv_tables/tab_delimited_text.csv")
    data = DataIO(path_to_source, sep="\t", quoting="none").load_data()
    DataIO(test_csv_path, sep="\t", quoting="none").save_data(data)
    data = DataIO(test_csv_path, sep="\t", quoting="none").load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_multiline_bad_line_csv(rp_logger):
    rp_logger.info("Loading data in the '.csv' format with multiline texts")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multiline_bad_line_text.csv")
    data = DataIO(
        path_to_source,
        sep=",",
        quoting="all",
        quotechar='"',
        escapechar="\\",
        skiprows=1,
        encoding="utf-8",
        on_bad_lines="skip"
    ).load_data()
    assert data.shape == (12, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_multiline_bad_line_csv(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.csv' format with multiline texts")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multiline_bad_line_text.csv")
    data = DataIO(
        path_to_source,
        sep=",",
        quoting="all",
        quotechar='"',
        escapechar="\\",
        skiprows=1,
        encoding="utf-8",
        on_bad_lines="skip"
    ).load_data()
    DataIO(
        test_csv_path,
        sep=",",
        quoting="all",
        quotechar='"',
        escapechar="\\",
        skiprows=1,
        encoding="utf-8",
        on_bad_lines="skip"
    ).save_data(data)
    data = DataIO(
        test_csv_path,
        sep=",",
        quoting="all",
        quotechar='"',
        escapechar="\\",
        skiprows=None,
        encoding="utf-8",
        on_bad_lines="skip"
    ).load_data()
    assert data.shape == (12, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_double_quoted_csv(rp_logger):
    rp_logger.info("Loading data in the '.csv' format with double quoted values")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/double_quoted_text.csv")
    data = DataIO(
        path_to_source,
        sep=",",
        quotechar='"'
    ).load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_double_quoted_csv(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.csv' format with double quoted values")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/double_quoted_text.csv")
    data = DataIO(path_to_source, sep=",", quotechar='"').load_data()
    DataIO(
        test_csv_path,
        sep=",",
        quotechar='"'
    ).save_data(data)
    data = DataIO(
        test_csv_path,
        sep=",",
        quotechar='"'
    ).load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_escaped_quoted_csv(rp_logger):
    rp_logger.info("Loading data in the '.csv' format with escaped quoted values")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/escaped_quoted_text.csv")
    data = DataIO(path_to_source, sep=",", quotechar='"', escapechar="\\").load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_escaped_quoted_csv(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.csv' format with escaped quoted values")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/escaped_quoted_text.csv")
    data = DataIO(path_to_source, sep=",", quotechar='"').load_data()
    DataIO(
        test_csv_path,
        sep=",",
        quotechar='"',
        escapechar="\\"
    ).save_data(data)
    data = DataIO(
        test_csv_path,
        sep=",",
        quotechar='"',
        escapechar="\\"
    ).load_data()
    assert data.shape == (15, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_without_header(rp_logger):
    rp_logger.info("Loading data in the '.csv' format without the header")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/text_without_header.csv")
    data = DataIO(
        path_to_source,
        sep=",",
        header=None,
        encoding="ascii",
        quoting="non-numeric",
        quotechar='"'
    ).load_data()
    assert data.shape == (7, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_without_header(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.csv' format without the header")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/text_without_header.csv")
    data = DataIO(
        path_to_source,
        sep=",",
        header=None,
        encoding="ascii",
        quoting="non-numeric",
        quotechar='"'
    ).load_data()
    DataIO(
        test_csv_path,
        sep=",",
        header=None,
        encoding="ascii",
        quoting="non-numeric",
        quotechar='"'
    ).save_data(data)
    data = DataIO(
        test_csv_path,
        sep=",",
        header=None,
        encoding="ascii",
        quoting="non-numeric",
        skiprows=None,
        quotechar='"'
    ).load_data()
    assert data.shape == (7, 3)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_triple_colons(rp_logger):
    rp_logger.info(
        "Loading data in the '.csv' format contained the fields separated by triple colons"
    )
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multicolon_delimited_text.csv")
    data = DataIO(path_to_source, sep=":::").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_triple_colons(test_csv_path, rp_logger):
    rp_logger.info(
        "Saving data in the '.csv' format contained the fields with triple colons"
    )
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/multicolon_delimited_text.csv")
    data = DataIO(path_to_source, sep=":::").load_data()
    DataIO(test_csv_path, sep=":::").save_data(data)
    data = DataIO(test_csv_path, sep=",").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_text_file(rp_logger):
    rp_logger.info("Loading data in the '.txt' format")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_data.txt")
    data = DataIO(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_text_file(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.txt' format")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_data.txt")
    data = DataIO(path_to_source).load_data()
    DataIO(test_csv_path).save_data(data)
    data = DataIO(test_csv_path).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_pcv_file(rp_logger):
    rp_logger.info("Loading data in the '.pcv' format")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/pipe_delimited_text.psv")
    data = DataIO(path_to_source).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_pcv_file(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.pcv' format")
    path_to_source = f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/pipe_delimited_text.psv"
    data = DataIO(path_to_source).load_data()
    DataIO(test_csv_path).save_data(data)
    data = DataIO(test_csv_path).load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_tcv_file(rp_logger):
    rp_logger.info("Loading data in the '.tcv' format")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/tab_delimited_text.tsv")
    data = DataIO(path_to_source).load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_tcv_file(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.tcv' format")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/tab_delimited_text.tsv")
    data = DataIO(path_to_source).load_data()
    DataIO(test_csv_path).save_data(data)
    data = DataIO(test_csv_path).load_data()
    assert data.shape == (15, 7)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_nested_field(rp_logger):
    rp_logger.info("Loading data in the '.csv' format with nested field")
    path_to_source = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/text_contained_nested_field.csv"
    )
    data = DataIO(path_to_source, quotechar='"', quoting="minimal").load_data()
    assert data.shape == (15, 5)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_nested_field(test_csv_path, rp_logger):
    path_to_source = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/text_contained_nested_field.csv"
    )
    rp_logger.info("Saving data in the '.csv' format with nested field")
    data = DataIO(path_to_source, quotechar='"', quoting="minimal").load_data()
    DataIO(test_csv_path, quotechar='"', quoting="minimal").save_data(data)
    data = DataIO(
        test_csv_path,
        quotechar='"',
        quoting="minimal"
    ).load_data()
    assert data.shape == (15, 5)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_double_pipe_delimited_text(rp_logger):
    rp_logger.info("Loading CSV with the double pipe delimited text")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/double_pipe_delimited_text.csv")
    data = DataIO(path_to_source, sep=r"\|\|").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_double_pipe_delimited_text(test_csv_path, rp_logger):
    rp_logger.info(
        "Saving data in the '.csv' format contained the fields with double pipe delimited text"
    )
    path_to_source = (
        f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/double_pipe_delimited_text.csv"
    )
    data = DataIO(path_to_source, sep=r"\|\|").load_data()
    DataIO(test_csv_path, sep=r"\|\|").save_data(data)
    data = DataIO(test_csv_path, sep=",").load_data()
    assert data.shape == (15, 6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_csv_with_na_values(rp_logger):
    rp_logger.info("Loading data in the '.csv' format contained NA values")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_na_values.csv")
    data = DataIO(path_to_source, na_values=["-", "Missing"]).load_data()
    assert data.shape == (10, 11)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_csv_with_na_values(test_csv_path, rp_logger):
    rp_logger.info("Saving data in the '.csv' format contained NA values")
    path_to_source = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                      "csv_tables/table_with_na_values.csv")
    data = DataIO(path_to_source, na_values=["-", "Missing", "NaN"]).load_data()
    DataIO(test_csv_path, na_values=["-", "Missing", "NaN"]).save_data(data)
    data = DataIO(test_csv_path, na_values=["-", "Missing", "NaN"]).load_data()
    assert data.shape == (10, 11)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialize_data_loader_for_local_excel_table(rp_logger):
    rp_logger.info("Initializing the instance of the class DataIO for the local Excel table")
    path_to_table = (f"{DIR_NAME}/unit/data_loaders/fixtures/"
                     "excel_tables/table_with_data.xlsx")
    test_data_loader = DataIO(path_to_table)
    assert test_data_loader.path == path_to_table
    assert test_data_loader.fernet_key is None
    assert test_data_loader.metadata == {
        "table": {
            "train_settings": {"source": path_to_table},
            "encryption": {"fernet_key": None},
            "format": {}
        },
    }
    assert isinstance(test_data_loader.data_loader.file_loader, ExcelLoader)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path_to_table",
    [
        f"{DIR_NAME}/unit/data_loaders/fixtures/excel_tables/table_with_data.xls",
        f"{DIR_NAME}/unit/data_loaders/fixtures/excel_tables/table_with_data.xlsx",
    ],
)
def test_load_data_from_table_in_excel_format(path_to_table, rp_logger):
    rp_logger.info("Loading data from the local Excel tables in '.xls', '.xlsx' format")
    test_data_loader = DataIO(path_to_table)
    df = test_data_loader.load_data()
    assert isinstance(test_data_loader.data_loader.file_loader, ExcelLoader)
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_excel_format_from_1_sheet_of_2_sheets(rp_logger):
    rp_logger.info(
        "Loading data from the local Excel table in the '.xls' format from the certain sheet"
    )
    test_data_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/"
        "excel_tables/table_with_data_and_2_nonempty_sheets.xlsx",
        sheet_name="TestName"
    )
    df = test_data_loader.load_data()
    assert isinstance(test_data_loader.data_loader.file_loader, ExcelLoader)
    assert (
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "gender": [0, 1],
                    "height": [162.9182, 173.5145],
                    "id": [821, 383],
                }
            ),
        )
        is None
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_table_in_excel_format_from_2_sheets_of_2_sheets(rp_logger):
    rp_logger.info(
        "Loading data from the local Excel table in '.xlsx' format "
        "that contains 2 non-empty sheets"
    )
    test_data_loader = DataIO(
        path=f"{DIR_NAME}/unit/data_loaders/fixtures/"
        "excel_tables/table_with_data_and_2_nonempty_sheets.xlsx",
        sheet_name=None
    )
    df = test_data_loader.load_data()
    assert isinstance(test_data_loader.data_loader.file_loader, ExcelLoader)
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

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_data_from_empty_excel_table(rp_logger, caplog):
    rp_logger.info("Loading data from the local empty Excel table in the '.xlsx' format")
    test_data_loader = DataIO(
        f"{DIR_NAME}/unit/data_loaders/fixtures/excel_tables/empty_table.xlsx"
    )
    assert isinstance(test_data_loader.data_loader.file_loader, ExcelLoader)
    data = test_data_loader.load_data()
    assert data.empty is True
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_excel_table_in_xls_format(test_xls_path, test_df, rp_logger):
    rp_logger.info("Saving the data to the Excel table in the '.xls' format")
    dataio_data_loader = DataIO(test_xls_path)
    dataio_data_loader.save_data(test_df)

    assert isinstance(dataio_data_loader.data_loader.file_loader, ExcelLoader)
    assert os.path.exists(test_xls_path) is True

    loaded_df = dataio_data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_excel_table_in_xlsx_format(test_xlsx_path, test_df, rp_logger):
    rp_logger.info("Saving data to the Excel table in the '.xlsx' format")
    dataio_data_loader = DataIO(test_xlsx_path)
    dataio_data_loader.save_data(test_df)

    assert isinstance(dataio_data_loader.data_loader.file_loader, ExcelLoader)
    assert os.path.exists(test_xlsx_path) is True

    loaded_df = dataio_data_loader.load_data()
    pd.testing.assert_frame_equal(loaded_df, test_df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_data_encryptor_with_dat_path_and_fernet_key(monkeypatch, rp_logger):
    rp_logger.info(
        "Test the initialization of DataIO with path ended with '.dat' "
        "and provided Fernet key"
    )
    monkeypatch.setenv("FERNET_KEY", Fernet.generate_key().decode())
    path = "path/to/data.dat"
    dataio_data_loader = DataIO(
        path=path,
        fernet_key="FERNET_KEY"
    )
    assert isinstance(dataio_data_loader.data_loader.file_loader, DataEncryptor)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_round_encrypt_decrypt_data(dataio_data_encryptor, valid_simple_dataframe, rp_logger):
    rp_logger.info(
        "Test the round encryption and decryption of the dataframe "
        "with the provided valid Fernet key"
    )
    dataio_data_encryptor.save_data(valid_simple_dataframe)
    loaded_df = dataio_data_encryptor.load_data()
    pd.testing.assert_frame_equal(loaded_df, valid_simple_dataframe)
    rp_logger.info(SUCCESSFUL_MESSAGE)
