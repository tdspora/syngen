from unittest.mock import patch
import pytest

from marshmallow import ValidationError

from syngen.sdk import Syngen, DataIO
from syngen.ml.worker import Worker
from syngen.ml.validation_schema import ReportTypes
from syngen.ml.reporters import Report
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME

TABLE_NAME = "test_table"
PATH_TO_TABLE = f"{DIR_NAME}/unit/launchers/fixtures/table_with_data.csv"
PATH_TO_METADATA = f"{DIR_NAME}/unit/launchers/fixtures/metadata.yaml"
TRAIN_REPORT_TYPES = ReportTypes().train_report_types
INFER_REPORT_TYPES = ReportTypes().infer_report_types
LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_source_and_table_name(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info("Launch the training process with parameters 'source' and 'table_name'")
    Syngen().train(source=PATH_TO_TABLE, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_metadata_path(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info("Launch the training process with the parameter 'metadata_path'")
    Syngen().train(metadata_path=PATH_TO_METADATA)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_launch_table_with_metadata_path_and_source(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Launch the training process with parameters 'metadata_path' and 'source'"
    )
    with caplog.at_level("WARNING"):
        Syngen().train(metadata_path=PATH_TO_METADATA, source=PATH_TO_TABLE)
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'source' will be ignored" in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_train_table_with_metadata_path_and_table_name(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info("Launch the training process with parameters 'metadata_path' and 'table_name'")
    with caplog.at_level("WARNING"):
        Syngen().train(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME)
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' will be ignored" in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_train_table_with_metadata_path_and_table_name_and_source(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Launch the training process with parameters 'metadata_path', 'table_name' and 'source'"
    )
    with caplog.at_level("WARNING"):
        Syngen().train(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME, source=PATH_TO_TABLE)
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' and 'source' will be ignored"
            in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_table_name_and_without_source(rp_logger, caplog):
    rp_logger.info("Launch the training process only with the parameter 'table_name'")
    with pytest.raises(AttributeError) as error:
        Syngen().train(table_name=TABLE_NAME)
        assert str(error.value) == (
            "It seems that the information of 'metadata_path' or 'source' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'source' and 'table_name'",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_source_and_without_table_name(rp_logger):
    rp_logger.info("Launch the training process only with the parameter 'source'")
    with pytest.raises(AttributeError) as error:
        Syngen().train(source=PATH_TO_TABLE)
        assert str(error.value) == (
            "It seems that the information of 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'source' and 'table_name'",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_without_parameters(rp_logger):
    rp_logger.info("Launch the training process without parameters")
    with pytest.raises(AttributeError) as error:
        Syngen().train()
        assert str(error.value) == (
            "It seems that the information of 'metadata_path' or 'table_name' "
            "and 'source' is absent. Please provide either the information of "
            "'metadata_path' or the information of 'source' and 'table_name'",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_epochs(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info("Launch the training process with the valid 'epochs' parameter")
    Syngen().train(epochs=20, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_epochs(rp_logger, caplog):
    rp_logger.info("Launch the training process with the invalid 'epochs' parameter")
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen().train(epochs=0, table_name=TABLE_NAME, source=PATH_TO_TABLE)
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


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_drop_null(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch the training process with the valid 'drop_null' parameter equals 'True'"
    )
    Syngen().train(drop_null=True, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_drop_null(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process with the invalid 'drop_null' parameter equals 'test'"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen().train(drop_null="test", table_name=TABLE_NAME, source=PATH_TO_TABLE)
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


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_row_limit(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process with the valid 'row_limit' parameter equals 100"
    )
    Syngen().train(row_limit=100, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_row_limit(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process with the invalid 'row_limit' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen().train(row_limit=0, table_name=TABLE_NAME, source=PATH_TO_TABLE)
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
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch the training process "
        f"with the valid 'reports' the parameter equals '{valid_value}'"
    )
    Syngen().train(reports=valid_value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [
        [pv, i] for pv in TRAIN_REPORT_TYPES
        for i in TRAIN_REPORT_TYPES
    ]
)
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_several_valid_values_in_reports(
    mock_post_init, mock_launch_train, value, rp_logger
):
    rp_logger.info(
        "Launch the training process with several valid values "
        f"in the 'reports' parameter equals '{value}'"
    )
    Syngen().train(reports=value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
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
        Syngen().train(reports=invalid_value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
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
        Syngen().train(reports=value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
        assert str(error.value) == (
            "Invalid input: When 'reports' parameter is set to 'none' or 'all', "
            "no other values should be provided."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_batch_size(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process "
        "with the valid 'batch_size' parameter equals 100"
    )
    Syngen().train(batch_size=100, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_invalid_batch_size(rp_logger, caplog):
    rp_logger.info("Launch the training process with the invalid 'batch_size' parameter equals 0")
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen().train(batch_size=0, table_name=TABLE_NAME, source=PATH_TO_TABLE)
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


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_fernet_key(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process with the valid 'fernet_key' parameter "
        "equals to the value of the environment variable 'FERNET_KEY'"
    )
    Syngen().train(fernet_key="FERNET_KEY", table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_train_table_with_nonexistent_fernet_key(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process with the invalid 'fernet_key' parameter "
        "equals to non-existent environment variable name"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            Syngen().train(
                fernet_key="FERNET_KEY_NONEXISTENT",
                table_name=TABLE_NAME,
                source=PATH_TO_TABLE
            )
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
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_log_level(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch the training process with the valid 'log_level' parameter equals {valid_value}"
    )
    Syngen().train(log_level=valid_value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_log_level(rp_logger):
    rp_logger.info(
        "Launch the training process with the invalid 'log_level' parameter equals 'test'"
    )
    with pytest.raises(ValueError) as error:
        Syngen().train(log_level="test", table_name=TABLE_NAME)
        assert str(error.value) == "ValueError: Level 'test' does not exist"

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_table_name(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info("Launch the inference process with the parameter 'table_name'")
    Syngen().infer(table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_metadata_path(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info("Launch the inference process with the parameter 'metadata_path'")
    Syngen().infer(metadata_path=PATH_TO_METADATA)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_infer_table_with_metadata_path_and_table_name(
    mock_logger, mock_post_init, mock_launch_infer, rp_logger, caplog
):
    rp_logger.info(
        "Launch the inference process with parameters 'metadata_path' and 'table_name'"
    )
    with caplog.at_level("WARNING"):
        Syngen().infer(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME)
        mock_post_init.assert_called_once()
        mock_launch_infer.assert_called_once()
        assert (
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' will be ignored" in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_without_parameters(rp_logger):
    rp_logger.info("Launch the inference process without parameters")
    with pytest.raises(AttributeError) as error:
        Syngen().infer()
        assert str(error.value) == (
            "It seems that the information of 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'table_name'",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_size(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch the inference process with the valid 'size' parameter equals 10"
    )
    Syngen().infer(size=10, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_size(rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process with the invalid 'size' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen().infer(size=0, table_name=TABLE_NAME)
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


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_run_parallel(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch the inference process with the valid 'run_parallel' parameter equals True"
    )
    Syngen().infer(run_parallel=True, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_run_parallel(rp_logger, caplog):
    rp_logger.info(
        "Launch infer process with the invalid 'run_parallel' parameter equals 'test'"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            Syngen().infer(run_parallel="test", table_name=TABLE_NAME)
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


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_batch_size(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info("Launch infer process with the valid 'batch_size' parameter equals 100")
    Syngen().infer(batch_size=100, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_batch_size(rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process with the invalid 'batch_size' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR") as caplog:
            Syngen().infer(batch_size=0, table_name=TABLE_NAME)
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


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_random_seed(
    mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info("Launch the inference process with the valid 'random_seed' parameter equals 1")
    Syngen().infer(random_seed=1, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_infer_table_with_invalid_random_seed(rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the invalid 'random_seed' parameter equals -1"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR") as caplog:
            Syngen().infer(random_seed=-1, table_name=TABLE_NAME)
            assert str(error.value) == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "random_seed": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert caplog.text == (
                'The error(s) found in - "test_table": {\n'
                '    "infer_settings": {\n'
                '        "random_seed": [\n'
                '             "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", INFER_REPORT_TYPES + ["none", "all"])
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_infer, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch the inference process with the valid 'reports' parameter equals '{valid_value}'"
    )
    Syngen().infer(reports=valid_value, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [
        [pv, i] for pv in INFER_REPORT_TYPES
        for i in INFER_REPORT_TYPES
    ]
)
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_several_valid_parameter_reports(
    mock_post_init, mock_launch_infer, value, rp_logger
):
    rp_logger.info(
        f"Launch the inference process with several values "
        f"in the 'reports' parameter equals '{value}'"
    )
    Syngen().infer(reports=value, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("invalid_value", [
    "sample", "test", ("none", "all"), ("none", "test"), ("all", "test")
])
def test_infer_table_with_invalid_parameter_reports(invalid_value, rp_logger):
    rp_logger.info(
        "Launch the inference process "
        "with the invalid 'reports' parameter equals '{invalid_value}'"
    )
    with pytest.raises(ValueError) as error:
        Syngen().infer(reports=invalid_value, table_name=TABLE_NAME)
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
        Syngen().infer(reports=value, table_name=TABLE_NAME)
        assert str(error.value) == (
            "Invalid input: When 'reports' option is set to 'none' or 'all', "
            "no other values should be provided."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_fernet_key(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch the inference process with the valid 'fernet_key' parameter "
        "equals to the value of the environment variable 'FERNET_KEY'"
    )
    Syngen().infer(fernet_key="FERNET_KEY", table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_non_existent_fernet_key(rp_logger):
    rp_logger.info(
        "Launch the inference process with the invalid 'fernet_key' parameter "
        "equals to non-existent environment variable name"
    )
    with pytest.raises(ValueError) as error:
        Syngen().infer(fernet_key="FERNET_KEY_NONEXISTENT", table_name=TABLE_NAME)
        assert str(error.value) == (
            "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
            "Please, check whether it is set correctly."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", LOG_LEVELS)
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_log_level(
    mock_post_init, mock_launch_infer, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process "
        f"with the valid 'log_level' parameter equals - '{valid_value}'"
    )
    Syngen().infer(log_level=valid_value, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_log_level(rp_logger):
    rp_logger.info(
        "Launch the inference process with the invalid 'log_level' parameter equals 'test'"
    )
    with pytest.raises(ValueError) as error:
        Syngen().infer(log_level="test", table_name=TABLE_NAME)
        assert str(error.value) == "ValueError: Level 'test' does not exist"

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", [["sample"], "sample"])
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_get_sample_reporter")
@patch.object(Syngen, "_get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
def test_generate_sample_report(
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the sample report")
    Syngen().generate_reports(table_name="test_table", reports=report)
    mock_validate_artifacts.assert_called_once_with("test_table", None, ["sample"])
    mock_get_sample_reporter.assert_called_once_with("test_table")
    mock_get_accuracy_reporter.assert_not_called()
    mock_register_reporter.assert_called_once()
    mock_generate_report.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", [["accuracy"], "accuracy"])
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_get_sample_reporter")
@patch.object(Syngen, "_get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
def test_generate_accuracy_report(
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the accuracy report")
    Syngen().generate_reports(table_name="test_table", reports=report)
    mock_validate_artifacts.assert_called_once_with("test_table", None, ["accuracy"])
    mock_get_accuracy_reporter.assert_called_once_with("test_table", "accuracy")
    mock_get_sample_reporter.assert_not_called()
    mock_register_reporter.assert_called_once()
    mock_generate_report.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", [["metrics_only"], "metrics_only"])
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_get_sample_reporter")
@patch.object(Syngen, "_get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
def test_generate_metrics_only_report(
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the 'metrics_only' report")
    Syngen().generate_reports(table_name="test_table", reports=report)
    mock_validate_artifacts.assert_called_once_with("test_table", None, ["metrics_only"])
    mock_get_accuracy_reporter.assert_called_once_with("test_table", "metrics_only")
    mock_get_sample_reporter.assert_not_called()
    mock_register_reporter.assert_called_once()
    mock_generate_report.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", ["all", ["accuracy", "sample"]])
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_get_sample_reporter")
@patch.object(Syngen, "_get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
def test_generate_all_reports(
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    report,
    rp_logger
):
    rp_logger.info("Launch the generation of the 'metrics_only' report")
    Syngen().generate_reports(table_name="test_table", reports=report)
    mock_validate_artifacts.assert_called_once()
    assert mock_get_accuracy_reporter.call_count == 1
    mock_get_sample_reporter.assert_called_once()
    assert mock_register_reporter.call_count == 2
    mock_generate_report.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("report", ["none", []])
@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_get_sample_reporter")
@patch.object(Syngen, "_get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
def test_generate_none_reports(
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    report,
    rp_logger,
    caplog
):
    rp_logger.info("Launch the generation of the 'none' report")
    Syngen().generate_reports(table_name="test_table", reports=report)
    with caplog.at_level("WARNING"):
        mock_validate_artifacts.assert_not_called()
        mock_get_accuracy_reporter.assert_not_called()
        mock_get_sample_reporter.assert_not_called()
        mock_register_reporter.assert_not_called()
        mock_generate_report.assert_not_called()
        assert (
            "No reports to generate. Please specify the report type "
            "from 'accuracy', 'metrics_only' or 'sample'."
        ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Report, "generate_report")
@patch.object(Report, "register_reporter")
@patch.object(Syngen, "_get_sample_reporter")
@patch.object(Syngen, "_get_accuracy_reporter")
@patch.object(Syngen, "_validate_artifacts")
def test_generate_full_set_of_reports(
    mock_validate_artifacts,
    mock_get_accuracy_reporter,
    mock_get_sample_reporter,
    mock_register_reporter,
    mock_generate_report,
    rp_logger
):
    rp_logger.info(
        "Launch the generation of all types of reports - "
        "'accuracy', 'metrics_only', 'sample' reports"
    )
    Syngen().generate_reports(
        table_name="test_table",
        reports=["accuracy", "metrics_only", "sample"]
    )
    mock_validate_artifacts.assert_called_once()
    assert mock_get_accuracy_reporter.call_count == 2
    mock_get_sample_reporter.assert_called_once()
    assert mock_register_reporter.call_count == 3
    mock_generate_report.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_report_with_wrong_report_type(rp_logger, caplog):
    rp_logger.info("Launch the generation of the report with the wrong report type")
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            Syngen().generate_reports(table_name="test_table", reports="test")
            assert str(error.value) == (
                f"Invalid report type - 'test'. "
                "Use 'accuracy', 'metrics_only' or 'sample'."
            )
            assert caplog.text == (
                f"Invalid report type - 'test'. "
                "Use 'accuracy', 'metrics_only' or 'sample'."
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)
