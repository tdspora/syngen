from unittest.mock import patch
import pytest

from click.testing import CliRunner

from syngen.train import launch_train
from syngen.ml.worker import Worker
from syngen.ml.validation_schema import ReportTypes
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME

TABLE_NAME = "test_table"
PATH_TO_TABLE = f"{DIR_NAME}/unit/launchers/fixtures/table_with_data.csv"
PATH_TO_METADATA = f"{DIR_NAME}/unit/launchers/fixtures/metadata.yaml"
TRAIN_REPORT_TYPES = ReportTypes().train_report_types


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_source_and_table_name(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch train process through CLI with parameters '--source' and '--table_name'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--source", PATH_TO_TABLE, "--table_name", TABLE_NAME]
    )
    assert result.exit_code == 0
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_metadata_path(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info("Launch train process through CLI with parameters '--metadata_path'")
    runner = CliRunner()
    result = runner.invoke(launch_train, ["--metadata_path", PATH_TO_METADATA])
    assert result.exit_code == 0
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_log_process")
def test_train_table_with_metadata_path_and_source(
    mock_logger, mock_post_init, mock_launch_train, rp_logger, caplog
):
    rp_logger.info(
        "Launch train process through CLI with parameters '--metadata_path' and '--source'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            launch_train,
            ["--metadata_path", PATH_TO_METADATA, "--source", PATH_TO_TABLE],
        )
        assert result.exit_code == 0
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
    rp_logger.info(
        "Launch train process through CLI with parameters '--metadata_path' and '--table_name'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            launch_train,
            ["--metadata_path", PATH_TO_METADATA, "--table_name", TABLE_NAME],
        )
        assert result.exit_code == 0
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
        "Launch train process through CLI with parameters "
        "'--metadata_path', '--table_name' and '--source'"
    )
    with caplog.at_level("WARNING"):
        runner = CliRunner()
        result = runner.invoke(
            launch_train,
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
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' and 'source' will be ignored"
            in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_table_name_and_without_source(rp_logger):
    rp_logger.info("Launch train process through CLI only with parameter '--table_name'")
    runner = CliRunner()
    result = runner.invoke(launch_train, ["--table_name", TABLE_NAME])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information of 'metadata_path' or 'source' is absent. "
        "Please provide either the information of 'metadata_path' or "
        "the information of 'source' and 'table_name'",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_source_and_without_table_name(rp_logger):
    rp_logger.info("Launch train process through CLI only with parameter '--source'")
    runner = CliRunner()
    result = runner.invoke(launch_train, ["--source", PATH_TO_TABLE])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information of 'metadata_path' or 'table_name' is absent. "
        "Please provide either the information of 'metadata_path' or "
        "the information of 'source' and 'table_name'",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_without_parameters(rp_logger):
    rp_logger.info("Launch train process through CLI without parameters")
    runner = CliRunner()
    result = runner.invoke(launch_train, [])
    assert result.exit_code == 1
    assert isinstance(result.exception, AttributeError)
    assert result.exception.args == (
        "It seems that the information of 'metadata_path' or 'table_name' and 'source' is absent. "
        "Please provide either the information of 'metadata_path' or "
        "the information of 'source' and 'table_name'",
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_epochs(
    mock_post_init, mock_launch_train, monkeypatch, rp_logger
):
    rp_logger.info(
        "Launch train process through CLI with valid 'epochs' parameter"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--epochs", 20, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_epochs(rp_logger):
    rp_logger.info(
        "Launch train process through CLI with invalid 'epochs' parameter"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--epochs", 0, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_drop_null(
    mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch train process through CLI with valid 'drop_null' parameter equals 'True'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--drop_null", True, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_drop_null(rp_logger):
    rp_logger.info(
        "Launch train process through CLI with valid 'drop_null' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--drop_null", "test", "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_row_limit(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch train process through CLI with valid 'row_limit' parameter equals 100"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--row_limit", 100, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_row_limit(rp_logger):
    rp_logger.info(
        "Launch train process through CLI with invalid 'drop_null' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--row_limit", 0, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("valid_value", TRAIN_REPORT_TYPES + ["none", "all"])
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_parameter_reports(
    mock_post_init, mock_launch_train, valid_value, rp_logger
):
    rp_logger.info(
        f"Launch train process through CLI with valid 'reports' parameter equals '{valid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--reports", valid_value, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
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
def test_train_table_with_several_valid_parameter_reports(
    mock_post_init, mock_launch_train, first_value, second_value, rp_logger
):
    rp_logger.info(
        f"Launch train process through CLI "
        f"with several valid 'reports' parameters equals '{first_value}' and '{second_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
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
    "invalid_value", ["test", ("none", "all"), ("none", "test"), ("all", "test")]
)
def test_train_table_with_invalid_parameter_reports(invalid_value, rp_logger):
    rp_logger.info(
        f"Launch train process through CLI "
        f"with invalid 'reports' parameter equals '{invalid_value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
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
    "prior_value, value",
    [(pv, i) for pv in ["all", "none"] for i in TRAIN_REPORT_TYPES]
)
def test_train_table_with_redundant_parameter_reports(prior_value, value, rp_logger):
    rp_logger.info(
        f"Launch train process through CLI with redundant 'reports' parameter: '{value}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train, [
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


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_batch_size(mock_post_init, mock_launch_train, rp_logger):
    rp_logger.info(
        "Launch train process through CLI with valid 'batch_size' parameter equals 100"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--batch_size", 100, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_batch_size(rp_logger):
    rp_logger.info(
        "Launch train process through CLI with invalid 'batch_size' parameter equals 0"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--batch_size", 0, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)
