from unittest.mock import patch
import pytest

from click.testing import CliRunner

from syngen.infer import launch_infer
from syngen.ml.worker import Worker
from syngen.ml.validation_schema import ReportTypes
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


TABLE_NAME = "test_table"
PATH_TO_METADATA = f"{DIR_NAME}/unit/launchers/fixtures/metadata.yaml"
INFER_REPORT_TYPES = ReportTypes().infer_report_types


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_table_name(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info("Launch infer process through CLI with parameter '--table_name'")
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--table_name", TABLE_NAME])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_metadata_path(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info("Launch infer process through CLI with parameter '--metadata_path'")
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--metadata_path", PATH_TO_METADATA])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.setup_log_process")
def test_infer_table_with_metadata_path_and_table_name(
    mock_logger, mock_post_init, mock_launch_infer, rp_logger, caplog
):
    rp_logger.info(
        "Launch infer process through CLI with parameters '--metadata_path' and '--table_name'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            launch_infer,
            ["--metadata_path", PATH_TO_METADATA, "--table_name", TABLE_NAME],
        )
        mock_post_init.assert_called_once()
        mock_launch_infer.assert_called_once()
        assert result.exit_code == 0
        assert (
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' will be ignored" in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_without_parameters(rp_logger):
    rp_logger.info("Launch infer process through CLI without parameters")
    runner = CliRunner()
    result = runner.invoke(launch_infer, [])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information of 'metadata_path' or 'table_name' is absent. "
        "Please provide either the information of 'metadata_path' or "
        "the information of 'table_name'",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_size(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with valid 'size' parameter equals 10"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--size", 10, "--table_name", TABLE_NAME])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_size(rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with invalid 'size' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--size", 0, "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_run_parallel(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with valid 'run_parallel' parameter equals True"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--run_parallel", True, "--table_name", TABLE_NAME])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_run_parallel(rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with invalid 'run_parallel' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_infer, ["--run_parallel", "test", "--table_name", TABLE_NAME]
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_batch_size(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with valid 'batch_size' parameter equals 100"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--batch_size", 100, "--table_name", TABLE_NAME])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_batch_size(rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with invalid 'batch_size' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--batch_size", 0, "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_random_seed(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with valid 'random_seed' parameter equals 1"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--random_seed", 1, "--table_name", TABLE_NAME])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_random_seed(rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with invalid 'random_seed' parameter equals -1"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--random_seed", -1, "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", INFER_REPORT_TYPES + ["none", "all"])
@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_infer, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch infer process through CLI with valid 'reports' parameter equals '{valid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_infer, ["--reports", valid_value, "--table_name", TABLE_NAME]
    )
    assert result.exit_code == 0
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
def test_infer_table_with_several_valid_parameter_reports(
    mock_post_init, mock_launch_infer, first_value, second_value, rp_logger
):
    rp_logger.info(
        f"Launch infer process through CLI "
        f"with several valid 'reports' parameters equals '{first_value}' and '{second_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_infer,
        [
            "--reports",
            first_value,
            "--reports",
            second_value,
            "--table_name",
            TABLE_NAME,
        ],
    )
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("invalid_value", [
    "sample", "test", ("none", "all"), ("none", "test"), ("all", "test")
])
def test_infer_table_with_invalid_parameter_reports(invalid_value, rp_logger):
    rp_logger.info(
        f"Launch infer process through CLI "
        f"with invalid 'reports' parameter equals '{invalid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_infer, ["--reports", invalid_value, "--table_name", TABLE_NAME]
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == (
        "Invalid input: Acceptable values for the parameter '--reports' "
        "are none, all, accuracy, metrics_only.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "prior_value, value",
    [(pv, i) for pv in ["all", "none"] for i in INFER_REPORT_TYPES]
)
def test_infer_table_with_redundant_parameter_reports(prior_value, value, rp_logger):
    rp_logger.info(
        f"Launch infer process through CLI with redundant 'reports' parameter: '{value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_infer, [
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
        "Invalid input: When '--reports' option is set to 'none' or 'all', "
        "no other values should be provided.",)
    rp_logger.info(SUCCESSFUL_MESSAGE)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
def test_infer_table_with_valid_fernet_key(mock_post_init, mock_launch_infer, rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with valid 'fernet_key' parameter "
        "equals to the value of the environment variable 'FERNET_KEY'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_infer,
        ["--fernet_key", "FERNET_KEY", "--table_name", TABLE_NAME],
    )
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_fernet_key(rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with invalid 'fernet_key' parameter "
        "equals to non-existent environment variable name"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_infer,
        [
            "--fernet_key", "FERNET_KEY_NONEXISTENT",
            "--table_name", TABLE_NAME
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
        "Please, check whether it is set correctly.",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)
