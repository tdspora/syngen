from unittest.mock import patch
from click.testing import CliRunner

from syngen.infer import launch_infer
from syngen.ml.worker import Worker
from tests.conftest import SUCCESSFUL_MESSAGE


TABLE_NAME = "test_table"
PATH_TO_METADATA = "./tests/unit/entrypoints/fixtures/metadata.yaml"


def test_infer_table_with_table_name(rp_logger):
    rp_logger.info("Launch infer process through CLI with parameter '--table_name'")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_metadata_path(rp_logger):
    rp_logger.info("Launch infer process through CLI with parameter '--metadata_path'")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--metadata_path", PATH_TO_METADATA
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_metadata_path_and_table_name(rp_logger, caplog):
    rp_logger.info("Launch infer process through CLI with parameters '--metadata_path' and '--table_name'")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        with caplog.at_level("WARNING"):
            runner = CliRunner()
            result = runner.invoke(
                launch_infer, [
                    "--metadata_path", PATH_TO_METADATA,
                    "--table_name", TABLE_NAME
                ]
            )
            assert result.exit_code == 0
            mock_method.assert_called_once()
            assert "The information of 'metadata_path' was provided. " \
                   "In this case the information of 'table_name' will be ignored" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_without_parameters(rp_logger):
    rp_logger.info("Launch infer process through CLI without parameters")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, []
        )
        assert result.exit_code == 1
        assert type(result.exception) == AttributeError
        assert result.exception.args == (
            "It seems that the information of 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'table_name'",)
        mock_method.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_valid_size(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with valid 'size' parameter equals 10")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--size", 10,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_size(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with invalid 'size' parameter equals 0")
    with patch.object(Worker, "launch_infer", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--size", 0,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_valid_run_parallel(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with valid 'run_parallel' parameter equals True")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--run_parallel", True,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_run_parallel(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with invalid 'run_parallel' parameter equals 'test'")
    with patch.object(Worker, "launch_infer", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--run_parallel", "test",
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_valid_batch_size(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with valid 'batch_size' parameter equals 100")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--batch_size", 100,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_batch_size(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with invalid 'batch_size' parameter equals 0")
    with patch.object(Worker, "launch_infer", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--batch_size", 0,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)

def test_infer_table_with_valid_random_seed(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with valid 'random_seed' parameter equals 1")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--random_seed", 1,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_random_seed(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with invalid 'random_seed' parameter equals -1")
    with patch.object(Worker, "launch_infer", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--random_seed", -1,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_valid_print_report(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with valid 'print_report' parameter equals True")
    with patch.object(Worker, "launch_infer", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--print_report", True,
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_infer_table_with_invalid_print_report(rp_logger):
    rp_logger.info(f"Launch infer process through CLI with invalid 'print_report' parameter equals 'test'")
    with patch.object(Worker, "launch_infer", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_infer, [
                "--print_report", "test",
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)
