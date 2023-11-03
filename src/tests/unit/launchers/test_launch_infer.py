from unittest.mock import patch
from click.testing import CliRunner

from syngen.infer import launch_infer
from syngen.ml.worker import Worker
from tests.conftest import SUCCESSFUL_MESSAGE


TABLE_NAME = "test_table"
PATH_TO_METADATA = "./tests/unit/launchers/fixtures/metadata.yaml"


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.set_mlflow")
def test_infer_table_with_table_name(
        mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info("Launch infer process through CLI with parameter '--table_name'")
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--table_name", TABLE_NAME])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.set_mlflow")
def test_infer_table_with_metadata_path(
        mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info("Launch infer process through CLI with parameter '--metadata_path'")
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--metadata_path", PATH_TO_METADATA])
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.set_mlflow")
@patch("syngen.infer.setup_logger")
def test_infer_table_with_metadata_path_and_table_name(
    mock_logger, mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger, caplog
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


@patch("syngen.infer.set_mlflow")
def test_infer_table_without_parameters(mock_set_mlflow, rp_logger):
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
@patch("syngen.infer.set_mlflow")
def test_infer_table_with_valid_size(mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger):
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
@patch("syngen.infer.set_mlflow")
def test_infer_table_with_valid_run_parallel(
    mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger
):
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
    result = runner.invoke(launch_infer, ["--run_parallel", "test", "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.set_mlflow")
def test_infer_table_with_valid_batch_size(
        mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger
):
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
@patch("syngen.infer.set_mlflow")
def test_infer_table_with_valid_random_seed(
        mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger
):
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


@patch.object(Worker, "launch_infer")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.infer.set_mlflow")
def test_infer_table_with_valid_print_report(
        mock_set_mlflow, mock_post_init, mock_launch_infer, rp_logger
):
    rp_logger.info(
        "Launch infer process through CLI with valid 'print_report' parameter equals True"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--print_report", True, "--table_name", TABLE_NAME])
    assert result.exit_code == 0
    mock_post_init.assert_called_once()
    mock_launch_infer.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_print_report(rp_logger):
    rp_logger.info(
        "Launch infer process through CLI with invalid 'print_report' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(launch_infer, ["--print_report", "test", "--table_name", TABLE_NAME])
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)
