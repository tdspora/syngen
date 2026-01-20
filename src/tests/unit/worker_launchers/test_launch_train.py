from unittest.mock import patch

import pandas as pd
import pytest

from click.testing import CliRunner
from marshmallow import ValidationError

from syngen.train import launch_train, cli_launch_train, validate_required_parameters
from syngen.ml.worker import Worker
from syngen.ml.validation_schema import ReportTypes
from syngen.ml.utils import ValidationError as UtilsValidationError
from tests.conftest import SUCCESSFUL_MESSAGE, get_dataframe


TABLE_NAME = "test_table"
PATH_TO_TABLE = "path/to/test_table.csv"
PATH_TO_METADATA = "path/to/metadata.yaml"
TRAIN_REPORT_TYPES = ReportTypes().train_report_types
LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_cli_launch_train_table_with_source_and_table_name(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI with parameters '--source' and '--table_name'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--source", PATH_TO_TABLE, "--table_name", TABLE_NAME]
    )
    assert result.exit_code == 0
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_source_and_table_name(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with parameters 'source' and 'table_name'"
    )
    launch_train(source=PATH_TO_TABLE, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_cli_launch_train_table_with_metadata_path(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI only "
        "with the parameter '--metadata_path'"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_train, ["--metadata_path", PATH_TO_METADATA])
    assert result.exit_code == 0
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_metadata_path(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the parameter 'metadata_path'"
    )
    launch_train(metadata_path=PATH_TO_METADATA)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_cli_launch_train_table_with_metadata_path_and_source(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Launch the training process through CLI with parameters '--metadata_path' and '--source'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            cli_launch_train,
            ["--metadata_path", PATH_TO_METADATA, "--source", PATH_TO_TABLE],
        )
        assert result.exit_code == 0
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'source' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_validate_parameters_with_metadata_path_and_source(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Validate required parameters before launching of the training process "
        "with parameters 'metadata_path' and 'source'"
    )
    with caplog.at_level("WARNING"):
        validate_required_parameters(metadata_path=PATH_TO_METADATA, source=PATH_TO_TABLE)
        launch_train(metadata_path=PATH_TO_METADATA, source=PATH_TO_TABLE)
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'source' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_cli_launch_train_table_with_metadata_path_and_table_name(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Launch the training process through CLI "
        "with parameters '--metadata_path' and '--table_name'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            cli_launch_train,
            ["--metadata_path", PATH_TO_METADATA, "--table_name", TABLE_NAME],
        )
        assert result.exit_code == 0
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_validate_parameters_with_metadata_path_and_table_name(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Validate required parameters before launching of the training process "
        "with parameters 'metadata_path' and 'table_name'"
    )
    with caplog.at_level("WARNING"):
        validate_required_parameters(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME)
        launch_train(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME)
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_cli_launch_train_table_with_metadata_path_and_table_name_and_source(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Launch the training process through CLI with parameters "
        "'--metadata_path', '--table_name' and '--source'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            cli_launch_train,
            [
                "--metadata_path",
                PATH_TO_METADATA,
                "--table_name",
                TABLE_NAME,
                "--source",
                PATH_TO_TABLE,
            ],
        )
        assert result.exit_code == 0
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' and 'source' will be ignored."
            in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_validate_parameters_with_metadata_path_and_table_name_and_source(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Validate required parameters before the launching training process with parameters "
        "'metadata_path', 'table_name' and 'source'"
    )
    with caplog.at_level("WARNING"):
        validate_required_parameters(
            metadata_path=PATH_TO_METADATA,
            table_name=TABLE_NAME,
            source=PATH_TO_TABLE
        )
        launch_train(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME, source=PATH_TO_TABLE)
        mock_post_init.assert_called_once()
        mock_launch_train.assert_called_once()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' and 'source' will be ignored."
            in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_table_name_and_without_source(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI only with the parameter '--table_name'"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_train, ["--table_name", TABLE_NAME])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information about 'metadata_path' or 'source' is absent. "
        "Please provide either the information about 'metadata_path' or "
        "the information about 'source' and 'table_name'.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_parameters_with_table_name_and_without_source(rp_logger, caplog):
    rp_logger.info(
        "Validate required parameters before launching the training process "
        "only with the parameter 'table_name'"
    )
    with pytest.raises(AttributeError) as error:
        validate_required_parameters(table_name=TABLE_NAME)
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'source' is absent. "
            "Please provide either the information about 'metadata_path' or "
            "the information about 'source' and 'table_name'.",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_source_and_without_table_name(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI only with the parameter '--source'"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_train, ["--source", PATH_TO_TABLE])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information about 'metadata_path' or 'table_name' is absent. "
        "Please provide either the information about 'metadata_path' or "
        "the information about 'source' and 'table_name'.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_parameters_with_source_and_without_table_name(rp_logger):
    rp_logger.info(
        "Validate required parameters before launching the training process "
        "only with the parameter 'source'"
    )
    with pytest.raises(AttributeError) as error:
        validate_required_parameters(source=PATH_TO_TABLE)
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information about 'metadata_path' or "
            "the information about 'source' and 'table_name'.",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_without_parameters(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI without parameters"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_train, [])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information about 'metadata_path' or "
        "'table_name' and 'source' is absent. "
        "Please provide either the information about 'metadata_path' or "
        "the information about 'source' and 'table_name'.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_parameters_without_parameters(rp_logger):
    rp_logger.info(
        "Validate required parameters before launching of training process without parameters"
    )
    with pytest.raises(AttributeError) as error:
        validate_required_parameters()
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'table_name' "
            "and 'source' is absent. Please provide either the information about "
            "'metadata_path' or the information about 'source' and 'table_name'.",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_cli_launch_train_table_with_valid_epochs(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process through CLI with the valid '--epochs' parameter"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--epochs", 20, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_valid_epochs(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the valid 'epochs' parameter"
    )
    launch_train(epochs=20, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_invalid_epochs(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI with the invalid '--epochs' parameter"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--epochs", 0, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_launch_train_table_with_invalid_epochs(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the invalid 'epochs' parameter"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_train(epochs=0, table_name=TABLE_NAME, source=PATH_TO_TABLE)
            message = (
                'The error(s) found in - "test_table": {\n'
                '    "train_settings": {\n'
                '        "epochs": [\n'
                '            "Must be greater than or equal to 1."\n'
                '        ]\n'
                '    }\n'
                '}'
            )
            assert message in str(error.value)
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", [True, False])
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_cli_launch_train_table_with_valid_drop_null(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI "
        f"with the valid '--drop_null' parameter equals '{valid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--drop_null", valid_value, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", [True, False])
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_valid_drop_null(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        f"with the valid 'drop_null' parameter equals '{valid_value}'"
    )
    launch_train(drop_null=valid_value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_invalid_drop_null(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI "
        "with the invalid '--drop_null' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--drop_null", "test", "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_launch_train_table_with_invalid_drop_null(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the invalid 'drop_null' parameter equals 'test'"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_train(drop_null="test", table_name=TABLE_NAME, source=PATH_TO_TABLE)
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
def test_cli_launch_train_table_with_valid_row_limit(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process through CLI with the valid '--row_limit' parameter equals 100"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--row_limit", 100, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_valid_row_limit(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the valid 'row_limit' parameter equals 100"
    )
    launch_train(row_limit=100, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_invalid_row_limit(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI with the invalid '--row_limit' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--row_limit", 0, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_launch_train_table_with_invalid_row_limit(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the invalid 'row_limit' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_train(row_limit=0, table_name=TABLE_NAME, source=PATH_TO_TABLE)
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
def test_cli_launch_train_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI "
        f"with the valid '--reports' the parameter equals '{valid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--reports", valid_value, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", TRAIN_REPORT_TYPES + ["none", "all"])
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        f"with the valid 'reports' the parameter equals '{valid_value}'"
    )
    launch_train(reports=valid_value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "first_value, second_value",
    [
        (pv, i) for pv in TRAIN_REPORT_TYPES
        for i in TRAIN_REPORT_TYPES
    ]
)
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_cli_launch_train_table_with_several_valid_parameter_reports(
    mock_post_init, mock_launch_train, first_value, second_value, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI "
        f"with several valid '--reports' parameters equals '{first_value}' and '{second_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        [
            "--reports",
            first_value,
            "--reports",
            second_value,
            "--table_name", TABLE_NAME,
            "--source", PATH_TO_TABLE
        ],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
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
def test_launch_train_table_with_several_valid_values_in_reports(
    mock_post_init, mock_launch_train, value, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        f"with several valid values in the 'reports' parameter equals '{value}'"
    )
    launch_train(reports=value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "invalid_value", ["test", ("none", "all"), ("none", "test"), ("all", "test")]
)
def test_cli_launch_train_table_with_invalid_parameter_reports(invalid_value, rp_logger):
    rp_logger.info(
        "Launch the training process through CLI "
        f"with the invalid '--reports' parameter equals '{invalid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        [
            "--reports",
            invalid_value,
            "--table_name",
            TABLE_NAME,
            "--source",
            PATH_TO_TABLE,
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "invalid_value", ["test", ("none", "all"), ("none", "test"), ("all", "test")]
)
def test_launch_train_table_with_invalid_parameter_reports(invalid_value, rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        f"with the invalid 'reports' parameter equals '{invalid_value}'"
    )
    with pytest.raises(ValueError):
        launch_train(reports=invalid_value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "prior_value, value",
    [(pv, i) for pv in ["all", "none"] for i in TRAIN_REPORT_TYPES]
)
def test_cli_launch_train_table_with_redundant_parameter_reports(prior_value, value, rp_logger):
    rp_logger.info(
        f"Launch the training process through CLI "
        f"with the redundant '--reports' parameter: '{value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train, [
            "--reports",
            prior_value,
            "--reports",
            value,
            "--table_name",
            TABLE_NAME,
            "--source",
            PATH_TO_TABLE
        ]
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == (
        "Invalid input: When 'reports' option is set to 'none' or 'all', "
        "no other values should be provided.",)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [[pv, i] for pv in ["all", "none"] for i in TRAIN_REPORT_TYPES]
)
def test_launch_train_table_with_redundant_parameter_reports(value, rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        f"with the redundant 'reports' parameter: '{value}'"
    )
    with pytest.raises(ValueError) as error:
        launch_train(reports=value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
        assert str(error.value) == (
            "Invalid input: When 'reports' parameter is set to 'none' or 'all', "
            "no other values should be provided."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_cli_launch_train_table_with_valid_batch_size(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI "
        "with the valid '--batch_size' parameter equals 100"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--batch_size", 100, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_valid_batch_size(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the valid 'batch_size' parameter equals 100"
    )
    launch_train(batch_size=100, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_invalid_batch_size(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI "
        "with the invalid '--batch_size' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--batch_size", 0, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_launch_train_table_with_invalid_batch_size(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the invalid 'batch_size' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_train(batch_size=0, table_name=TABLE_NAME, source=PATH_TO_TABLE)
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
def test_cli_launch_train_table_with_existed_fernet_key(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI with the '--fernet_key' parameter "
        "equals to the value of the existed environment variable 'FERNET_KEY'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--fernet_key", "FERNET_KEY", "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_existed_fernet_key(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the 'fernet_key' parameter "
        "equals to the value of the existed environment variable 'FERNET_KEY'"
    )
    launch_train(fernet_key="FERNET_KEY", table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_nonexistent_fernet_key(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI with the '--fernet_key' parameter "
        "equals to non-existent environment variable name"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        [
            "--fernet_key", "FERNET_KEY_NONEXISTENT",
            "--table_name", TABLE_NAME,
            "--source", PATH_TO_TABLE
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
        "Please, check whether it is set correctly.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.setup_log_process")
def test_launch_train_table_with_nonexistent_fernet_key(rp_logger, caplog):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the 'fernet_key' parameter equals to non-existent environment variable name"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            launch_train(
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
def test_cli_launch_train_table_with_valid_log_level(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process through CLI "
        f"with the valid '--log_level' parameter equals '{valid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--log_level", valid_value, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", LOG_LEVELS)
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_valid_log_level(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        f"with the valid 'log_level' parameter equals - '{valid_value}'"
    )
    launch_train(log_level=valid_value, table_name=TABLE_NAME, source=PATH_TO_TABLE)
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_train_table_with_invalid_log_level(rp_logger):
    rp_logger.info(
        "Launch the training process through CLI "
        "with the invalid '--log_level' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_train,
        ["--log_level", "test", "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_launch_train_table_with_invalid_log_level(rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the invalid 'log_level' parameter equals 'test'"
    )
    with pytest.raises(ValueError) as error:
        launch_train(log_level="test", table_name=TABLE_NAME, source=PATH_TO_TABLE)
        assert str(error.value) == "ValueError: Level 'test' does not exist"

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_train_table_with_loader(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the provided valid callback function to the 'loader' parameter"
    )
    launch_train(loader=get_dataframe, table_name="table")
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch("syngen.train.setup_log_process")
def test_launch_train_table_with_not_callable_loader(
    mock_logger, mock_launch_train, caplog, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the provided 'loader' parameter that is not callable"
    )
    error_message = (
        "The provided loader for the table - 'table' isn't callable. "
        "Please, provide a valid callback function."
    )
    with pytest.raises(UtilsValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_train(loader="not_callable", table_name="table")
            assert error_message in str(error.value)
            assert error_message in caplog.text

        mock_launch_train.assert_not_called()

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch("syngen.train.setup_log_process")
def test_launch_train_table_with_loader_with_wrong_signature(
    mock_logger, mock_launch_train, caplog, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the provided 'loader' parameter containing a function with wrong signature"
    )
    error_message = (
        "The provided loader for the table - 'table_name' doesn't accept "
        "'table_name' as an argument or requires additional arguments "
        "besides 'table_name' signature 'loader(table_name)'."
    )
    with pytest.raises(UtilsValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_train(loader=lambda x: pd.DataFrame(), table_name="table")
        assert error_message in str(error.value)
        assert error_message in caplog.text

        mock_launch_train.assert_not_called()

    rp_logger.info(SUCCESSFUL_MESSAGE)
