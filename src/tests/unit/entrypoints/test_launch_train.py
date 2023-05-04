from unittest.mock import patch
from click.testing import CliRunner

from syngen.train import launch_train
from syngen.ml.worker import Worker
from tests.conftest import SUCCESSFUL_MESSAGE

TABLE_NAME = "test_table"
PATH_TO_TABLE = "./tests/unit/entrypoints/fixtures/table_with_data.csv"
PATH_TO_METADATA = "./tests/unit/entrypoints/fixtures/metadata.yaml"


def test_train_table_with_source_and_table_name(rp_logger):
    rp_logger.info("Launch train process through CLI with parameters '--source' and '--table_name'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                '--source', PATH_TO_TABLE,
                '--table_name', TABLE_NAME
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_metadata_path(rp_logger):
    rp_logger.info("Launch train process through CLI with parameters '--metadata_path'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                '--metadata_path', PATH_TO_METADATA
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_metadata_path_and_source(rp_logger, caplog):
    rp_logger.info("Launch train process through CLI with parameters '--metadata_path' and '--source'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        with caplog.at_level("WARNING"):
            runner = CliRunner()
            result = runner.invoke(
                    launch_train, [
                        "--metadata_path", PATH_TO_METADATA,
                        "--source", PATH_TO_TABLE
                    ]
                )
            assert result.exit_code == 0
            mock_method.assert_called_once()
            assert "The information of 'metadata_path' was provided. " \
                   "In this case the information of 'source' will be ignored" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_metadata_path_and_table_name(rp_logger, caplog):
    rp_logger.info("Launch train process through CLI with parameters '--metadata_path' and '--table_name'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        with caplog.at_level("WARNING"):
            runner = CliRunner()
            result = runner.invoke(
                launch_train, [
                    "--metadata_path", PATH_TO_METADATA,
                    "--table_name", TABLE_NAME
                ]
            )
            assert result.exit_code == 0
            mock_method.assert_called_once()
            assert "The information of 'metadata_path' was provided. " \
                   "In this case the information of 'table_name' will be ignored" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_metadata_path_and_table_name_and_source(rp_logger, caplog):
    rp_logger.info("Launch train process through CLI with parameters '--metadata_path', '--table_name' and '--source'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        with caplog.at_level("WARNING"):
            runner = CliRunner()
            result = runner.invoke(
                launch_train, [
                    "--metadata_path", PATH_TO_METADATA,
                    "--table_name", TABLE_NAME,
                    "--source", PATH_TO_TABLE
                ]
            )
            assert result.exit_code == 0
            mock_method.assert_called_once()
            assert "The information of 'metadata_path' was provided. " \
                   "In this case the information of 'table_name' and 'source' will be ignored" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_table_name_and_without_source(rp_logger):
    rp_logger.info("Launch train process through CLI only with parameter '--table_name'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--table_name", TABLE_NAME
            ]
        )
        assert result.exit_code == 1
        assert type(result.exception) == AttributeError
        assert result.exception.args == (
            "It seems that the information of 'metadata_path' or 'source' is absent. "
            "Please provide either the information of 'metadata_path' or the information of 'source' and 'table_name'",)
        mock_method.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_source_and_without_table_name(rp_logger):
    rp_logger.info("Launch train process through CLI only with parameter '--source'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 1
        assert type(result.exception) == AttributeError
        assert result.exception.args == (
            "It seems that the information of 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'source' and 'table_name'",)
        mock_method.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_without_parameters(rp_logger):
    rp_logger.info("Launch train process through CLI without parameters")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, []
        )
        assert result.exit_code == 1
        assert type(result.exception) == AttributeError
        assert result.exception.args == (
            "It seems that the information of 'metadata_path' or 'table_name' and 'source' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'source' and 'table_name'",)
        mock_method.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_valid_epochs(monkeypatch, rp_logger):
    rp_logger.info("Launch train process through CLI with valid 'epochs' parameter")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--epochs", 20,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_epochs(monkeypatch, rp_logger):
    rp_logger.info("Launch train process through CLI with invalid 'epochs' parameter")
    with patch.object(Worker, "launch_train", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--epochs", 0,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_valid_drop_null(rp_logger):
    rp_logger.info(f"Launch train process through CLI with valid 'drop_null' parameter equals 'True'")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--drop_null", True,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_drop_null(rp_logger):
    rp_logger.info(f"Launch train process through CLI with valid 'drop_null' parameter equals 'test'")
    with patch.object(Worker, "launch_train", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--drop_null", "test",
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_valid_row_limit(rp_logger):
    rp_logger.info(f"Launch train process through CLI with valid 'row_limit' parameter equals 100")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--row_limit", 100,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_row_limit(rp_logger):
    rp_logger.info(f"Launch train process through CLI with invalid 'drop_null' parameter equals 0")
    with patch.object(Worker, "launch_train", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--row_limit", 0,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_valid_print_report(rp_logger):
    rp_logger.info(f"Launch train process through CLI with valid 'print_report' parameter equals True")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--print_report", True,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_print_report(rp_logger):
    rp_logger.info(f"Launch train process through CLI with invalid 'print_report' parameter equals 'test'")
    with patch.object(Worker, "launch_train", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--print_report", "test",
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_valid_batch_size(rp_logger):
    rp_logger.info(f"Launch train process through CLI with valid 'batch_size' parameter equals 100")
    with patch.object(Worker, "launch_train", return_value=None) as mock_method:
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--batch_size", 100,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 0
        mock_method.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_batch_size(rp_logger):
    rp_logger.info(f"Launch train process through CLI with invalid 'batch_size' parameter equals 0")
    with patch.object(Worker, "launch_train", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            launch_train, [
                "--batch_size", 0,
                "--table_name", TABLE_NAME,
                "--source", PATH_TO_TABLE
            ]
        )
        assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)
