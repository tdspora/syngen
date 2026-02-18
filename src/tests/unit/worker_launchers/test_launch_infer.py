from unittest.mock import patch, Mock
import pytest
import shutil

from click.testing import CliRunner
from marshmallow import ValidationError
import pandas as pd

from syngen.infer import launch_infer, cli_launch_infer, validate_required_parameters
from syngen.ml.worker import Worker
from syngen.ml.validation_schema import ReportTypes
from syngen.ml.utils import ValidationError as UtilsValidationError
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME, get_dataframe


TABLE_NAME = "test_table"
PATH_TO_METADATA = f"{DIR_NAME}/unit/launchers/fixtures/metadata.yaml"
INFER_REPORT_TYPES = ReportTypes().infer_report_types
LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_table_name(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info("Launch the inference process through CLI with the parameter '--table_name'")
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--table_name", TABLE_NAME])
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_table_name(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the parameter 'table_name'"
    )
    launch_infer(table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_metadata_path(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info("Launch the inference process through CLI with the parameter '--metadata_path'")
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--metadata_path", PATH_TO_METADATA])
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_metadata_path(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the parameter 'metadata_path'"
    )
    launch_infer(metadata_path=PATH_TO_METADATA)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_metadata_path_and_table_name(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger, caplog
):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with parameters '--metadata_path' and '--table_name'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            cli_launch_infer,
            ["--metadata_path", PATH_TO_METADATA, "--table_name", TABLE_NAME],
        )
        mock_setup_log.assert_called_once()
        mock_post_init.assert_called_once()
        mock_launch_infer.assert_called_once()
        assert result.exit_code == 0
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_validate_parameters_with_metadata_path_and_table_name(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger, caplog
):
    rp_logger.info(
        "Validate required parameters before launching the inference process "
        "with parameters 'metadata_path' and 'table_name'"
    )
    with caplog.at_level("WARNING"):
        validate_required_parameters(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME)
        launch_infer(metadata_path=PATH_TO_METADATA, table_name=TABLE_NAME)
        mock_setup_log.assert_called_once()
        mock_post_init.assert_called_once()
        mock_launch_infer.assert_called_once()
        assert (
            "The information about 'metadata_path' was provided. "
            "In this case the information about 'table_name' will be ignored." in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_infer_table_without_parameters(rp_logger):
    rp_logger.info("Launch the inference process through CLI without parameters")
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, [])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information about 'metadata_path' or 'table_name' is absent. "
        "Please provide either the information about 'metadata_path' or "
        "the information about 'table_name'.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_parameters_without_parameters(rp_logger):
    rp_logger.info(
        "Validate required parameters before launching the inference process without parameters"
    )
    with pytest.raises(AttributeError) as error:
        validate_required_parameters()
        assert str(error.value) == (
            "It seems that the information about 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information about 'metadata_path' or "
            "the information about 'table_name'.",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_valid_size(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI with the valid '--size' parameter equals 10"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--size", 10, "--table_name", TABLE_NAME])
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_valid_size(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the valid 'size' parameter equals 10"
    )
    launch_infer(size=10, table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_infer_table_with_invalid_size(rp_logger):
    rp_logger.info(
        "Launch the inference process through CLI with the invalid '--size' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--size", 0, "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_invalid_size(mock_setup_log, rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the invalid 'size' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_infer(size=0, table_name=TABLE_NAME)
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
            mock_setup_log.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_valid_run_parallel(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with the valid '--run_parallel' parameter equals True"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer, ["--run_parallel", True, "--table_name", TABLE_NAME]
    )
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_valid_run_parallel(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the valid 'run_parallel' parameter equals True"
    )
    launch_infer(run_parallel=True, table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_infer_table_with_invalid_run_parallel(rp_logger):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with the invalid '--run_parallel' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer, ["--run_parallel", "test", "--table_name", TABLE_NAME]
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_invalid_run_parallel(mock_setup_log, rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the invalid 'run_parallel' parameter equals 'test'"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_infer(run_parallel="test", table_name=TABLE_NAME)
            mock_setup_log.assert_called_once()
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
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_valid_batch_size(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with the valid '--batch_size' parameter equals 100"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--batch_size", 100, "--table_name", TABLE_NAME])
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_valid_batch_size(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function "
        "'launch_infer' with the valid 'batch_size' parameter equals 100"
    )
    launch_infer(batch_size=100, table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_infer_table_with_invalid_batch_size(rp_logger):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with the invalid '--batch_size' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--batch_size", 0, "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_invalid_batch_size(mock_setup_log, rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the invalid 'batch_size' parameter equals 0"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR") as caplog:
            launch_infer(batch_size=0, table_name=TABLE_NAME)
            mock_setup_log.assert_called_once()
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
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_valid_random_seed(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with the valid '--random_seed' parameter equals 1"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--random_seed", 1, "--table_name", TABLE_NAME])
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_valid_random_seed(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the valid 'random_seed' parameter equals 1"
    )
    launch_infer(random_seed=1, table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_infer_table_with_invalid_random_seed(rp_logger):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with the invalid '--random_seed' parameter equals -1"
    )
    runner = CliRunner()
    result = runner.invoke(cli_launch_infer, ["--random_seed", -1, "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_invalid_random_seed(mock_setup_log, rp_logger, caplog):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the invalid 'random_seed' parameter equals -1"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR") as caplog:
            launch_infer(random_seed=-1, table_name=TABLE_NAME)
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
            mock_setup_log.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", INFER_REPORT_TYPES + ["none", "all"])
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_valid_parameter_reports(
    mock_setup_log, mock_post_init, mock_launch_infer, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch the inference process through CLI "
        f"with the valid '--reports' parameter equals '{valid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer, ["--reports", valid_value, "--table_name", TABLE_NAME]
    )
    assert result.exit_code == 0
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", INFER_REPORT_TYPES + ["none", "all"])
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_valid_parameter_reports(
    mock_setup_log, mock_post_init, mock_launch_infer, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        f"with the valid 'reports' parameter equals '{valid_value}'"
    )
    launch_infer(reports=valid_value, table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "first_value, second_value",
    [
        (pv, i) for pv in INFER_REPORT_TYPES
        for i in INFER_REPORT_TYPES
    ]
)
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_several_valid_parameter_reports(
    mock_setup_log, mock_post_init, mock_launch_infer, first_value, second_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI "
        f"with several valid '--reports' parameters equals '{first_value}' and '{second_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer,
        [
            "--reports",
            first_value,
            "--reports",
            second_value,
            "--table_name",
            TABLE_NAME,
        ],
    )
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
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
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_several_valid_parameter_reports(
    mock_setup_log, mock_post_init, mock_launch_infer, value, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        f"with several values in the 'reports' parameter equals '{value}'"
    )
    launch_infer(reports=value, table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("invalid_value", [
    "sample", "test", ("none", "all"), ("none", "test"), ("all", "test")
])
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_invalid_parameter_reports(
    mock_setup_log, invalid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI "
        f"with the invalid '--reports' parameter equals '{invalid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer, ["--reports", invalid_value, "--table_name", TABLE_NAME]
    )
    mock_setup_log.assert_called_once()
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == (
        "Invalid input: Acceptable values for the parameter 'reports' "
        "are 'none', 'all', 'accuracy', 'metrics_only'.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("invalid_value", [
    "sample", "test", ("none", "all"), ("none", "test"), ("all", "test")
])
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_invalid_parameter_reports(
    mock_setup_log, invalid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        f"with the invalid 'reports' parameter equals '{invalid_value}'"
    )
    with pytest.raises(ValueError) as error:
        launch_infer(reports=invalid_value, table_name=TABLE_NAME)
        mock_setup_log.assert_called_once()
        assert str(error.value) == (
            "Invalid input: Acceptable values for the parameter 'reports' "
            "are 'none', 'all', 'accuracy', 'metrics_only'."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "prior_value, value",
    [(pv, i) for pv in ["all", "none"] for i in INFER_REPORT_TYPES]
)
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_redundant_parameter_reports(
    mock_setup_log, prior_value, value, rp_logger
):
    rp_logger.info(
        f"Launch the inference process through CLI with redundant '--reports' parameter: '{value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer, [
            "--reports",
            prior_value,
            "--reports",
            value,
            "--table_name",
            TABLE_NAME
        ]
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == (
        "Invalid input: When 'reports' option is set to 'none' or 'all', "
        "no other values should be provided.",)
    mock_setup_log.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "value",
    [[pv, i] for pv in ["all", "none"] for i in INFER_REPORT_TYPES]
)
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_redundant_parameter_reports(mock_setup_log, value, rp_logger):
    rp_logger.info(
        f"Launch the inference process by using the function 'launch_infer' "
        f"with redundant 'reports' parameter: '{value}'"
    )
    with pytest.raises(ValueError) as error:
        launch_infer(reports=value, table_name=TABLE_NAME)
        mock_setup_log.assert_called_once()
        assert str(error.value) == (
            "Invalid input: When 'reports' option is set to 'none' or 'all', "
            "no other values should be provided."
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_existed_fernet_key(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI with the valid '--fernet_key' parameter "
        "equals to the value of the environment variable 'FERNET_KEY'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer,
        ["--fernet_key", "FERNET_KEY", "--table_name", TABLE_NAME],
    )
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_existed_fernet_key(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the valid 'fernet_key' parameter "
        "equals to the value of the environment variable 'FERNET_KEY'"
    )
    launch_infer(fernet_key="FERNET_KEY", table_name=TABLE_NAME)
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_non_existent_fernet_key(mock_setup_log, rp_logger):
    rp_logger.info(
        "Launch the inference process through CLI with the '--fernet_key' parameter "
        "equals to the non-existent environment variable name"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer,
        [
            "--fernet_key", "FERNET_KEY_NONEXISTENT",
            "--table_name", TABLE_NAME
        ],
    )
    mock_setup_log.assert_called_once()
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
        "Please, check whether it is set correctly.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_non_existent_fernet_key(mock_setup_log, rp_logger):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the 'fernet_key' parameter "
        "equals to the non-existent environment variable name"
    )
    with pytest.raises(ValueError) as error:
        launch_infer(fernet_key="FERNET_KEY_NONEXISTENT", table_name=TABLE_NAME)
        assert str(error.value) == (
            "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
            "Please, check whether it is set correctly."
        )
        mock_setup_log.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", LOG_LEVELS)
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_cli_launch_infer_table_with_valid_log_level(
    mock_setup_log, mock_post_init, mock_launch_infer, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process through CLI "
        f"with the valid '--log_level' parameter equals '{valid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer,
        ["--log_level", valid_value, "--table_name", TABLE_NAME],
    )
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", LOG_LEVELS)
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_launch_infer_table_with_valid_log_level(
    mock_post_init, mock_launch_infer, valid_value, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        f"with the valid 'log_level' parameter equals '{valid_value}'"
    )
    launch_infer(log_level=valid_value, table_name=TABLE_NAME)
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    shutil.rmtree("model_artifacts/")
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cli_launch_infer_table_with_invalid_log_level(rp_logger):
    rp_logger.info(
        "Launch the inference process through CLI "
        "with the invalid '--log_level' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_launch_infer,
        ["--log_level", "test", "--table_name", TABLE_NAME],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


# def test_launch_infer_table_with_invalid_log_level(rp_logger):
#     rp_logger.info(
#         "Launch the inference process by using the function 'launch_infer' "
#         "with the invalid 'log_level' parameter equals 'test'"
#     )
#     with pytest.raises(ValueError) as error:
#         launch_infer(log_level="test", table_name=TABLE_NAME)
#         assert str(error.value) == "ValueError: Level 'test' does not exist"
#
#     rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_loader(
    mock_setup_log, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the provided valid callback function to the 'loader' parameter"
    )
    launch_infer(loader=get_dataframe, table_name="table")
    mock_setup_log.assert_called_once()
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_not_callable_loader(
    mock_setup_log, mock_launch_infer, caplog, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the provided 'loader' parameter that is not callable"
    )
    error_message = (
        "The provided loader for the table - 'table' isn't callable. "
        "Please, provide a valid callback function."
    )
    with pytest.raises(UtilsValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_infer(loader="not_callable", table_name="table")
            assert error_message in str(error.value)
            assert error_message in caplog.text

        mock_setup_log.assert_called_once()
        mock_launch_infer.assert_not_called()

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("loader, error_message", [
    (
        lambda x: pd.DataFrame(),
        "The provided loader should accept `table_name` as an argument. "
        "Please, adjust the signature of the loader."
    ),
    (
        lambda table_name, y: pd.DataFrame(),
        "The provided loader should accept only the argument - `table_name`. "
        "Please, adjust the signature of the loader."
    )
])
@patch.object(Worker, "launch_infer")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_loader_with_wrong_signature(
    mock_setup_log, mock_launch_infer, loader, error_message, caplog, rp_logger
):
    rp_logger.info(
        "Launch the inference process by using the function 'launch_infer' "
        "with the provided 'loader' parameter containing a function with wrong signature"
    )
    with pytest.raises(UtilsValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_infer(loader=loader, table_name="table")
            assert error_message in str(error.value)
            assert error_message in caplog.text

        mock_setup_log.assert_called_once()
        mock_launch_infer.assert_not_called()

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch("syngen.infer.setup_log_process")
def test_launch_infer_table_with_loader_with_wrong_return_value(
    mock_setup_log, mock_launch_infer, caplog, rp_logger
):
    rp_logger.info(
        "Launch the training process by using the function 'launch_train' "
        "with the provided 'loader' parameter that returns a non-DataFrame object"
    )
    error_message = "The provided loader raised an exception when called"
    with pytest.raises(UtilsValidationError) as error:
        with caplog.at_level("ERROR"):
            launch_infer(loader=lambda table_name: Mock(), table_name="table")
            assert error_message in str(error.value)
            assert error_message in caplog.text

        mock_launch_infer.assert_not_called()
        mock_setup_log.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)
