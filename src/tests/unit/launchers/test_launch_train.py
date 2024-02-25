from unittest.mock import patch
from click.testing import CliRunner

from syngen.train import (
    launch_train,
    get_json_columns,
    get_flattened_df
)
from syngen.ml.worker import Worker
from syngen.ml.data_loaders import DataLoader
from tests.conftest import SUCCESSFUL_MESSAGE

TABLE_NAME = "test_table"
PATH_TO_TABLE = "./tests/unit/launchers/fixtures/table_with_data.csv"
PATH_TO_METADATA = "./tests/unit/launchers/fixtures/metadata.yaml"


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_source_and_table_name(
        mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch train process through CLI with parameters '--source' and '--table_name'"
    )
    runner = CliRunner()
    result = runner.invoke(launch_train, ["--source", PATH_TO_TABLE, "--table_name", TABLE_NAME])
    assert result.exit_code == 0
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.preprocess_data")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_metadata_path(
        mock_post_init, mock_launch_train, mock_preprocess_data, rp_logger
):
    rp_logger.info("Launch train process through CLI with parameters '--metadata_path'")
    runner = CliRunner()
    result = runner.invoke(launch_train, ["--metadata_path", PATH_TO_METADATA])
    assert result.exit_code == 0
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    mock_post_init.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.preprocess_data")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_logger")
def test_train_table_with_metadata_path_and_source(
    mock_logger, mock_post_init, mock_launch_train, mock_preprocess_data, rp_logger, caplog
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
        mock_preprocess_data.assert_called_once()
        assert (
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'source' will be ignored" in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.preprocess_data")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_logger")
def test_train_table_with_metadata_path_and_table_name(
    mock_logger, mock_post_init, mock_launch_train, mock_preprocess_data, rp_logger, caplog
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
        mock_preprocess_data.assert_called_once()
        assert (
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' will be ignored" in caplog.text
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.train.preprocess_data")
@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
@patch("syngen.train.setup_logger")
def test_train_table_with_metadata_path_and_table_name_and_source(
    mock_logger, mock_post_init, mock_launch_train, mock_preprocess_data, rp_logger, caplog
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
        mock_preprocess_data.assert_called_once()
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
def test_train_table_with_valid_row_limit(
        mock_post_init, mock_launch_train, rp_logger
):
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


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_print_report(
        mock_post_init, mock_launch_train, rp_logger
):
    rp_logger.info(
        "Launch train process through CLI with valid 'print_report' parameter equals True"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        ["--print_report", True, "--table_name", TABLE_NAME, "--source", PATH_TO_TABLE],
    )
    mock_post_init.assert_called_once()
    mock_launch_train.assert_called_once()
    assert result.exit_code == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_table_with_invalid_print_report(rp_logger):
    rp_logger.info(
        "Launch train process through CLI with invalid 'print_report' parameter equals 'test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        launch_train,
        [
            "--print_report",
            "test",
            "--table_name",
            TABLE_NAME,
            "--source",
            PATH_TO_TABLE,
        ],
    )
    assert result.exit_code == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "launch_train")
@patch.object(Worker, "__attrs_post_init__")
def test_train_table_with_valid_batch_size(
        mock_post_init, mock_launch_train, rp_logger
):
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


def test_get_json_columns_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_json_columns' method "
        "for the dataframe contained one json column"
    )
    path_to_data = "tests/unit/launchers/fixtures/data_with_one_json_column.csv"
    data, schema = DataLoader(path_to_data).load_data()
    assert get_json_columns(data) == ["_details"]


def test_get_flattened_df_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_flattened_df' method "
        "for the dataframe contained one json column"
    )
    path_to_data = "tests/unit/launchers/fixtures/data_with_one_json_column.csv"
    data, schema = DataLoader(path_to_data).load_data()
    json_columns = ["_details"]
    flattened_data, flattening_mapping = get_flattened_df(data, json_columns)
    assert flattened_data.columns.to_list() == [
        "id",
        "created_at",
        "updated_at",
        "name",
        "description",
        "owner_id",
        "is_default",
        "is_encrypted",
        "status",
        "cluster_type",
        "master",
        "ssh_config",
        "username",
        "password",
        "authentication_type",
        "log_level",
        "env_variables",
        "key_passphrase",
        "private_key"
    ]
    assert flattening_mapping == {
        "_details": [
            "cluster_type",
            "master",
            "ssh_config",
            "username",
            "password",
            "authentication_type",
            "log_level",
            "env_variables",
            "key_passphrase",
            "private_key"
        ]
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_json_columns' method "
        "for the dataframe contained more than one json column"
    )
    path_to_data = "tests/unit/launchers/fixtures/data_with_two_json_columns.csv"
    data, schema = DataLoader(path_to_data).load_data()
    assert get_json_columns(data) == ["progress", "details"]


def test_get_flattened_df_with_df_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_flattened_df' method "
        "for the dataframe contained more than one json column"
    )
    path_to_data = "tests/unit/launchers/fixtures/data_with_two_json_columns.csv"
    data, schema = DataLoader(path_to_data).load_data()
    json_columns = ["progress", "details"]
    flattened_data, flattening_mapping = get_flattened_df(data, json_columns)
    assert flattened_data.columns.to_list() == [
        "id",
        "created_at",
        "updated_at",
        "start_time",
        "finish_time",
        "status",
        "pipeline_id",
        "user_id",
        "info.finished",
        "info.total",
        "step",
        "description",
        "source.id",
        "source.name",
        "source.connection_string",
        "target.id",
        "target.name",
        "target.connection_string",
        "cluster.id",
        "cluster.name",
        "cluster.master_webui",
        "cluster.history_server",
        "integrity_type",
        "total_tables",
        "included_tables",
        "base_table"
    ]
    assert flattening_mapping == {
        "progress": [
            "info.finished",
            "info.total",
            "step"
        ],
        "details": [
            "description",
            "source.id",
            "source.name",
            "source.connection_string",
            "target.id",
            "target.name",
            "target.connection_string",
            "cluster.id",
            "cluster.name",
            "cluster.master_webui",
            "cluster.history_server",
            "integrity_type",
            "total_tables",
            "included_tables",
            "base_table"
        ]
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)
