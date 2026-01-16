from unittest.mock import patch

import pytest
import os

from marshmallow import ValidationError
from cryptography.fernet import Fernet

from syngen.sdk import Syngen
from syngen.ml.worker import Worker
from syngen.ml.validation_schema import ReportTypes
from syngen.ml.reporters import Report
from syngen.ml.data_loaders import DataEncryptor
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME, get_dataframe

TABLE_NAME = "test_table"
PATH_TO_TABLE = "path/to/source.csv"
PATH_TO_METADATA = f"{DIR_NAME}/unit/sdk/fixtures/metadata.yaml"
TRAIN_REPORT_TYPES = ReportTypes().train_report_types
INFER_REPORT_TYPES = ReportTypes().infer_report_types
LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_initialization_with_source_and_table_name(rp_logger):
    rp_logger.info(
        "Initialization of the instance 'Syngen' by providing with 'source' and 'table_name'"
    )
    instance = Syngen(source=PATH_TO_TABLE, table_name=TABLE_NAME)
    assert instance.table_name == TABLE_NAME
    assert instance.source == PATH_TO_TABLE
    assert instance.metadata_path is None
    assert instance.loader is None
    assert instance.list_of_tables == ["test_table"]
    assert instance.execution_artifacts == dict()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_loader_and_table_name(rp_logger):
    rp_logger.info(
        "Initialization of the instance 'Syngen' by providing with 'loader' and 'table_name'"
    )
    instance = Syngen(loader=get_dataframe, table_name=TABLE_NAME)
    assert instance.table_name == TABLE_NAME
    assert instance.source is None
    assert callable(instance.loader) is True
    assert instance.metadata_path is None
    assert instance.list_of_tables == ["test_table"]
    assert instance.execution_artifacts == dict()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path(rp_logger):
    rp_logger.info("Initialization of the instance 'Syngen' by providing with 'metadata_path'")
    instance = Syngen(metadata_path=PATH_TO_METADATA)
    assert instance.table_name is None
    assert instance.source is None
    assert instance.loader is None
    assert instance.metadata_path == PATH_TO_METADATA
    assert instance.list_of_tables == ["test_table"]
    assert instance.execution_artifacts == dict()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path_and_source(rp_logger, caplog):
    rp_logger.info(
        "Initialization of the instance 'Syngen' by providing with 'metadata_path' and 'source'"
    )
    with caplog.at_level("WARNING"):
        instance = Syngen(metadata_path=PATH_TO_METADATA, source=PATH_TO_TABLE)
        assert instance.table_name is None
        assert instance.source == PATH_TO_TABLE
        assert instance.loader is None
        assert instance.metadata_path == PATH_TO_METADATA
        assert instance.list_of_tables == ["test_table"]
        assert instance.execution_artifacts == dict()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'source' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path_and_table_name(rp_logger, caplog):
    rp_logger.info(
        "Initialization of the instance 'Syngen' "
        "by providing with 'metadata_path' and 'table_name'"
    )
    with caplog.at_level("WARNING"):
        instance = Syngen(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME)
        assert instance.table_name == TABLE_NAME
        assert instance.source is None
        assert instance.loader is None
        assert instance.metadata_path == PATH_TO_METADATA
        assert instance.list_of_tables == ["test_table"]
        assert instance.execution_artifacts == dict()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path_and_loader(rp_logger):
    rp_logger.info(
        "Initialization of the instance 'Syngen' by providing 'metadata_path' and 'loader'"
    )
    instance = Syngen(metadata_path=PATH_TO_METADATA, loader=get_dataframe)
    assert instance.table_name is None
    assert instance.source is None
    assert callable(instance.loader) is True
    assert instance.metadata_path == PATH_TO_METADATA
    assert instance.list_of_tables == ["test_table"]
    assert instance.execution_artifacts == dict()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path_and_table_name_and_source(rp_logger, caplog):
    rp_logger.info(
        "Initialization of the instance 'Syngen' "
        "by providing 'metadata_path', 'table_name' and 'source'"
    )
    with caplog.at_level("WARNING"):
        instance = Syngen(
            metadata_path=PATH_TO_METADATA,
            table_name=TABLE_NAME,
            source=PATH_TO_TABLE
        )
        assert instance.table_name == TABLE_NAME
        assert instance.source == PATH_TO_TABLE
        assert instance.loader is None
        assert instance.metadata_path == PATH_TO_METADATA
        assert instance.list_of_tables == ["test_table"]
        assert instance.execution_artifacts == dict()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'source' and 'table_name' will be ignored."
            in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path_and_table_name_and_loader(rp_logger, caplog):
    rp_logger.info(
        "Initialization of the instance 'Syngen' "
        "by providing 'metadata_path', 'table_name' and 'loader'"
    )
    with caplog.at_level("WARNING"):
        instance = Syngen(
            metadata_path=PATH_TO_METADATA,
            table_name=TABLE_NAME,
            loader=get_dataframe
        )
        assert instance.table_name == TABLE_NAME
        assert instance.source is None
        assert callable(instance.loader) is True
        assert instance.metadata_path == PATH_TO_METADATA
        assert instance.list_of_tables == ["test_table"]
        assert instance.execution_artifacts == dict()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'loader' and 'table_name' will be ignored."
            in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path_and_table_name_and_loader_and_source(rp_logger, caplog):
    rp_logger.info(
        "Initialization of the instance 'Syngen' "
        "by providing 'metadata_path', 'table_name', 'loader' and 'source'"
    )
    with caplog.at_level("WARNING"):
        instance = Syngen(
            metadata_path=PATH_TO_METADATA,
            table_name=TABLE_NAME,
            loader=get_dataframe,
            source="path/to/source.csv"
        )
        assert instance.table_name == TABLE_NAME
        assert instance.source == "path/to/source.csv"
        assert callable(instance.loader) is True
        assert instance.metadata_path == PATH_TO_METADATA
        assert instance.list_of_tables == ["test_table"]
        assert instance.execution_artifacts == dict()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' and 'source' "
            "and 'loader' will be ignored."
        ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_only_with_table_name(rp_logger, caplog):
    rp_logger.info(
        "Initialization of the instance 'Syngen' by providing only 'table_name'"
    )
    with pytest.raises(AttributeError) as error:
        Syngen(table_name=TABLE_NAME)
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'source' (or 'loader') "
            "is absent. Please provide either the information about 'metadata_path' or "
            "the information of 'source' (or 'loader') and 'table_name'."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_only_with_source(rp_logger):
    rp_logger.info("Initialization of the instance 'Syngen' by providing only 'source'")
    with pytest.raises(AttributeError) as error:
        Syngen(source=PATH_TO_TABLE)
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information about 'metadata_path' or "
            "the information about 'source' (or 'loader') and 'table_name'."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_only_with_loader(rp_logger):
    rp_logger.info("Initialization of the instance 'Syngen' by providing only 'loader'")
    with pytest.raises(AttributeError) as error:
        Syngen(loader=get_dataframe)
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information about 'metadata_path' or "
            "the information about 'source' (or 'loader') and 'table_name'."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_source_and_loader(rp_logger):
    rp_logger.info(
        "Initialization of the instance 'Syngen' by providing 'source' and 'loader'"
    )
    with pytest.raises(AttributeError) as error:
        Syngen(source=PATH_TO_TABLE, loader=get_dataframe)
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information about 'metadata_path' or "
            "the information about 'source' (or 'loader') and 'table_name'."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_source_and_loader_and_table_name(rp_logger):
    rp_logger.info(
        "Initialization of the instance 'Syngen' "
        "by providing 'source', 'loader', and 'table_name'"
    )
    with pytest.raises(AttributeError) as error:
        Syngen(source=PATH_TO_TABLE, loader=get_dataframe, table_name=TABLE_NAME)
        assert str(error.value) == (
            "The information of both 'source' and 'loader' was provided. "
            "Please provide only one of them along with 'table_name'."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initialization_with_metadata_path_and_source_and_loader(caplog, rp_logger):
    rp_logger.info(
        "Initialization of the instance 'Syngen' by providing "
        "with 'metadata_path', 'loader', and 'source'"
    )
    with caplog.at_level("WARNING"):
        Syngen(source=PATH_TO_TABLE, loader=get_dataframe, metadata_path=PATH_TO_METADATA)
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'source' and 'loader' will be ignored."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)



def test_initialization_without_parameters(rp_logger):
    rp_logger.info("Initialization of the instance 'Syngen' without providing any attributes")
    with pytest.raises(AttributeError) as error:
        Syngen()
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'table_name' "
            "and 'source' (or 'loader') is absent. Please provide either the information about "
            "'metadata_path' or the information about ('source' or 'loader') and 'table_name'."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_epochs(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info("Launch the training process with the valid 'epochs' parameter")
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(epochs=20)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_epochs(rp_logger, caplog):
    rp_logger.info("Launch the training process with the invalid 'epochs' parameter")
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(epochs=0)
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "epochs": [\n'
                '            "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "epochs": [\n'
                '            "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", [True, False])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_drop_null(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process with the valid 'drop_null' parameter "
        f"equals to '{valid_value}'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(drop_null=valid_value)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_drop_null(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process with the invalid 'drop_null' parameter equals 'test'"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(drop_null="test")
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "drop_null": [\n'
                '             "Not a valid boolean."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "drop_null": [\n'
                '             "Not a valid boolean."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_row_limit(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info(
        "Launch the training process with the valid 'row_limit' parameter equals 100"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(row_limit=100)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_row_limit(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process with the invalid 'row_limit' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(row_limit=0)
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "row_limit": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "row_limit": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", TRAIN_REPORT_TYPES + ["none", "all"])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process "
        f"with the valid 'reports' the parameter equals '{valid_value}'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(reports=valid_value)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [
        [pv, i] for pv in TRAIN_REPORT_TYPES
        for i in TRAIN_REPORT_TYPES
    ]
)
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_several_valid_values_in_reports(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, value, rp_logger
):
    rp_logger.info(
        "Launch the training process with several valid values "
        f"in the 'reports' parameter equals '{value}'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(reports=value)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "invalid_value", ["test", ("none", "all"), ("none", "test"), ("all", "test")]
)
def test_train_table_with_invalid_parameter_reports(invalid_value, rp_logger):
    rp_logger.info(
        "Launch the training process "
        f"with the invalid 'reports' parameter equals '{invalid_value}'"
    )
    with pytest.raises(ValueError):
        Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(reports=invalid_value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [[pv, i] for pv in ["all", "none"] for i in TRAIN_REPORT_TYPES]
)
def test_train_table_with_redundant_parameter_reports(value, rp_logger):
    rp_logger.info(
        f"Launch the training process with the redundant 'reports' parameter: '{value}'"
    )
    with pytest.raises(ValueError) as error:
        Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(reports=value)
        assert str(error.value) == (
            "Invalid input: When 'reports' parameter is set to 'none' or 'all', "
            "no other values should be provided."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_batch_size(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info(
        "Launch the training process "
        "with the valid 'batch_size' parameter equals 100"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(batch_size=100)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_batch_size(rp_logger, caplog):
    rp_logger.info("Launch the training process with the invalid 'batch_size' parameter equals 0")
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(batch_size=0)
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "batch_size": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "batch_size": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_existing_fernet_key(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info(
        "Launch the training process with the 'fernet_key' parameter "
        "equals to the value of the existing environment variable 'FERNET_KEY'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(fernet_key="FERNET_KEY")
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_nonexistent_fernet_key(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process with the 'fernet_key' parameter "
        "equals to the non-existent environment variable name"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            Syngen(
                table_name=TABLE_NAME,
                source=PATH_TO_TABLE
            ).train(fernet_key="FERNET_KEY_NONEXISTENT")
            assert str(error.value) == (
                "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
                "Please, check whether it is set correctly."
            )
            assert caplog.text == (
                "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
                "Please, check whether it is set correctly."
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", LOG_LEVELS)
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_log_level(
    mock_post_init, mock_launch_train, mock_set_execution_artifacts, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch the training process with the valid 'log_level' parameter equals to {valid_value}"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(log_level=valid_value)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="train")
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_log_level(rp_logger):
    rp_logger.info(
        "Launch the training process with the invalid 'log_level' parameter equals 'test'"
    )
    with pytest.raises(ValueError) as error:
        Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).train(log_level="test")
        assert str(error.value) == "ValueError: Level 'test' does not exist"

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_size(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info(
        "Launch the inference process with the valid 'size' parameter equals 10"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(size=10)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_size(rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process with the invalid 'size' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(size=0)
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "size": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "size": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", [True, False])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_run_parallel(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process with the valid 'run_parallel' parameter "
        f"equals to '{valid_value}"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(run_parallel=True)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_run_parallel(rp_logger, caplog):
    rp_logger.info(
        "Launch infer process with the invalid 'run_parallel' parameter equals 'test'"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(run_parallel="test")
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "run_parallel": [\n'
                '             "Not a valid boolean."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "run_parallel": [\n'
                '             "Not a valid boolean."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_batch_size(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info("Launch infer process with the valid 'batch_size' parameter equals 100")
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(batch_size=100)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_batch_size(rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process with the invalid 'batch_size' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR") as caplog:
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(batch_size=0)
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "batch_size": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "batch_size": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_random_seed(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info("Launch the inference process with the valid 'random_seed' parameter equals 1")
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(random_seed=1)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_random_seed(rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the invalid 'random_seed' parameter equals -1"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR") as caplog:
            Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(random_seed=-1)
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "random_seed": [\n'
                '             "Must be greater than or equal to 0."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "random_seed": [\n'
                '             "Must be greater than or equal to 0."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", INFER_REPORT_TYPES + ["none", "all"])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch the inference process with the valid 'reports' parameter equals '{valid_value}'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(reports=valid_value)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [
        [pv, i] for pv in INFER_REPORT_TYPES
        for i in INFER_REPORT_TYPES
    ]
)
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_several_valid_parameter_reports(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, value, rp_logger
):
    rp_logger.info(
        f"Launch the inference process with several values "
        f"in the 'reports' parameter equals '{value}'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(reports=value)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("invalid_value", [
    "sample", "test", ("none", "all"), ("none", "test"), ("all", "test")
])
def test_infer_table_with_invalid_parameter_reports(invalid_value, rp_logger):
    rp_logger.info(
        "Launch the inference process "
        f"with the invalid 'reports' parameter equals '{invalid_value}'"
    )
    with pytest.raises(ValueError) as error:
        Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(reports=invalid_value)
        assert str(error.value) == (
            "Invalid input: Acceptable values for the parameter 'reports' "
            "are none, all, accuracy, metrics_only."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [[pv, i] for pv in ["all", "none"] for i in INFER_REPORT_TYPES]
)
def test_infer_table_with_redundant_parameter_reports(value, rp_logger):
    rp_logger.info(f"Launch the inference process with redundant 'reports' parameter: '{value}'")
    with pytest.raises(ValueError) as error:
        Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(reports=value)
        assert str(error.value) == (
            "Invalid input: When 'reports' option is set to 'none' or 'all', "
            "no other values should be provided."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_existing_fernet_key(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, rp_logger
):
    rp_logger.info(
        "Launch the inference process with the 'fernet_key' parameter "
        "equals to the value of the existing environment variable 'FERNET_KEY'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(fernet_key="FERNET_KEY")
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_non_existent_fernet_key(rp_logger):
    rp_logger.info(
        "Launch the inference process with the 'fernet_key' parameter "
        "equals to the non-existent environment variable name"
    )
    with pytest.raises(ValueError) as error:
        Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(
            fernet_key="FERNET_KEY_NONEXISTENT"
        )
        assert str(error.value) == (
            "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
            "Please, check whether it is set correctly."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", LOG_LEVELS)
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_log_level(
    mock_post_init, mock_launch_infer, mock_set_execution_artifacts, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process "
        f"with the valid 'log_level' parameter equals to '{valid_value}'"
    )
    Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(log_level=valid_value)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(type_of_process="infer")
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_log_level(rp_logger):
    rp_logger.info(
        "Launch the inference process with the invalid 'log_level' parameter equals 'test'"
    )
    with pytest.raises(ValueError) as error:
        Syngen(table_name=TABLE_NAME, source=PATH_TO_TABLE).infer(log_level="test")
        assert str(error.value) == "ValueError: Level 'test' does not exist"
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", [["sample"], "sample"])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Report, "clear_report")
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_Syngen__get_sample_reporter")
@patch.object(Syngen, "_Syngen__get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
@patch("syngen.sdk.setup_log_process")
def test_generate_sample_report(
    mock_setup_log_process,
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    mock_clear_report,
    mock_set_execution_artifacts,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the 'sample' report")
    Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
        table_name="test_table",
        reports=report,
        log_level="DEBUG"
    )
    mock_validate_artifacts.assert_called_once_with(
        table_name="test_table", completed_processes={"train"}
    )
    mock_get_sample_reporter.assert_called_once_with("test_table", None)
    mock_get_accuracy_reporter.assert_not_called()
    mock_register_reporter.assert_called_once()
    mock_generate_report.assert_called_once()
    mock_clear_report.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(
        type_of_process="report", table_name="test_table", reports=["sample"]
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", [["accuracy"], "accuracy"])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Report, "clear_report")
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_Syngen__get_sample_reporter")
@patch.object(Syngen, "_Syngen__get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
@patch("syngen.sdk.setup_log_process")
def test_generate_accuracy_report(
    mock_setup_log_process,
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    mock_clear_report,
    mock_set_execution_artifacts,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the 'accuracy' report")
    Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
        table_name="test_table",
        reports=report,
        log_level="DEBUG"
    )
    mock_validate_artifacts.assert_called_once_with(
        table_name="test_table", completed_processes={"infer"}
    )
    mock_get_accuracy_reporter.assert_called_once_with("test_table", "accuracy", None)
    mock_get_sample_reporter.assert_not_called()
    mock_register_reporter.assert_called_once()
    mock_generate_report.assert_called_once()
    mock_clear_report.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(
        type_of_process="report", table_name="test_table", reports=["accuracy"]
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", [["metrics_only"], "metrics_only"])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Report, "clear_report")
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_Syngen__get_sample_reporter")
@patch.object(Syngen, "_Syngen__get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
@patch("syngen.sdk.setup_log_process")
def test_generate_metrics_only_report(
    mock_setup_log_process,
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    mock_clear_report,
    mock_set_execution_artifacts,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the 'metrics_only' report")
    Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
        table_name="test_table",
        reports=report,
        log_level="DEBUG"
    )
    mock_validate_artifacts.assert_called_once_with(
        table_name="test_table", completed_processes={"infer"}
    )
    mock_get_accuracy_reporter.assert_called_once_with("test_table", "metrics_only", None)
    mock_get_sample_reporter.assert_not_called()
    mock_register_reporter.assert_called_once()
    mock_generate_report.assert_called_once()
    mock_clear_report.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(
        type_of_process="report", table_name="test_table", reports=["metrics_only"]
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", ["all", ["sample", "accuracy"]])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Report, "clear_report")
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_Syngen__get_sample_reporter")
@patch.object(Syngen, "_Syngen__get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
@patch("syngen.sdk.setup_log_process")
def test_generate_all_reports(
    mock_setup_log_process,
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    mock_clear_report,
    mock_set_execution_artifacts,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the 'all' report")
    Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
        table_name="test_table",
        reports=report,
        log_level="DEBUG"
    )
    mock_validate_artifacts.assert_called_once_with(
        table_name="test_table", completed_processes={"train", "infer"}
    )
    mock_get_accuracy_reporter.assert_called_once()
    mock_get_sample_reporter.assert_called_once()
    assert mock_register_reporter.call_count == 2
    mock_generate_report.assert_called_once()
    mock_clear_report.assert_called_once()
    mock_set_execution_artifacts.assert_called_once()

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", ["none", []])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Report, "clear_report")
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_Syngen__get_sample_reporter")
@patch.object(Syngen, "_Syngen__get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
@patch("syngen.sdk.setup_log_process")
def test_generate_none_reports(
    mock_setup_log_process,
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    mock_clear_report,
    mock_set_execution_artifacts,
    report,
    rp_logger,
    caplog
):
    rp_logger.info("Launch the generation of the 'none' report")
    with caplog.at_level("WARNING"):
        Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
            table_name="test_table",
            reports=report,
            log_level="DEBUG"
        )
        mock_validate_artifacts.assert_not_called()
        mock_get_accuracy_reporter.assert_not_called()
        mock_get_sample_reporter.assert_not_called()
        mock_register_reporter.assert_not_called()
        mock_generate_report.assert_not_called()
        mock_clear_report.assert_not_called()
        mock_set_execution_artifacts.assert_called_once_with(
            type_of_process="report", table_name="test_table", reports=[]
        )
        assert (
            "No reports to generate. Please specify the report type "
            "from 'accuracy', 'metrics_only', 'sample', 'none', 'all'."
        ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Report, "clear_report")
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_Syngen__get_sample_reporter")
@patch.object(Syngen, "_Syngen__get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
@patch("syngen.sdk.setup_log_process")
def test_generate_full_set_of_reports(
    mock_setup_log_process,
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    mock_clear_report,
    mock_set_execution_artifacts,
    rp_logger
):
    rp_logger.info(
        "Launch the generation of all types of reports - "
        "'accuracy', 'metrics_only', 'sample' reports"
    )
    Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
        table_name="test_table",
        reports=["accuracy", "metrics_only", "sample"],
        log_level="DEBUG"
    )
    mock_validate_artifacts.assert_called_once_with(
        table_name="test_table", completed_processes={"train", "infer"}
    )
    assert mock_get_accuracy_reporter.call_count == 2
    mock_get_sample_reporter.assert_called_once()
    assert mock_register_reporter.call_count == 3
    mock_generate_report.assert_called_once()
    mock_clear_report.assert_called_once()
    mock_set_execution_artifacts.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.sdk.setup_log_process")
def test_generate_report_with_wrong_report_type(mock_setup_log_process, rp_logger, caplog):
    rp_logger.info("Launch the generation of the report with the wrong report type")
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
                table_name="test_table", reports="test"
            )
            assert str(error.value) == (
                "Invalid input: Acceptable values for the parameter 'reports' "
                "are 'none', 'all', 'accuracy', 'metrics_only', 'sample'"
            )
            assert caplog.text == (
                "Invalid input: Acceptable values for the parameter 'reports' "
                "are 'none', 'all', 'accuracy', 'metrics_only', 'sample'"
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", [["sample"], "sample"])
@patch.object(Syngen, "_set_execution_artifacts")
@patch.object(Report, "clear_report")
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_Syngen__get_sample_reporter")
@patch.object(Syngen, "_Syngen__get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
@patch.object(DataEncryptor, "validate_fernet_key")
@patch("syngen.sdk.setup_log_process")
def test_generate_report_for_encrypted_data(
    mock_setup_log_process,
    mock_validate_fernet_key,
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    mock_clear_report,
    mock_set_execution_artifacts,
    report,
    monkeypatch,
    rp_logger
):
    rp_logger.info("Launch the generation of the 'sample' report for the encrypted data")
    fernet_key = Fernet.generate_key().decode()
    monkeypatch.setenv("FERNET_KEY", fernet_key)
    Syngen(metadata_path=PATH_TO_METADATA).generate_quality_reports(
        table_name="test_table",
        reports=report,
        fernet_key="FERNET_KEY",
        log_level="DEBUG"
    )
    mock_validate_fernet_key.assert_called_once_with(fernet_key)
    mock_validate_artifacts.assert_called_once_with(
        table_name="test_table", completed_processes={"train"}
    )
    mock_get_sample_reporter.assert_called_once_with("test_table", os.getenv("FERNET_KEY"))
    mock_get_accuracy_reporter.assert_not_called()
    mock_register_reporter.assert_called_once()
    mock_generate_report.assert_called_once()
    mock_clear_report.assert_called_once()
    mock_set_execution_artifacts.assert_called_once_with(
        type_of_process="report", table_name="test_table", reports=["sample"]
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)
