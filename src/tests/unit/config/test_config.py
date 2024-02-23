from unittest.mock import patch

from syngen.ml.config import TrainConfig

from tests.conftest import SUCCESSFUL_MESSAGE


@patch.object(TrainConfig, "__post_init__", lambda x: None)
def test_get_flattened_df_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_flattened_df' method returns a flattened dataframe "
        "for the dataframe contained one json column"
    )
    config = TrainConfig(
        source="tests/unit/config/fixtures/data_with_one_json_column.csv",
        epochs=10,
        drop_null=False,
        row_limit=1000,
        table_name="data_with_one_json_column",
        metadata_path="not/existed/path/to/metadata.yaml",
        print_report=False,
        batch_size=32
    )
    config.paths = {}
    data, schema = config._extract_data()
    config._get_json_columns(data)
    assert config.json_columns == ["_details"]
    assert config._get_flattened_df(data).columns.to_list() == [
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
    assert config.flattening_mapping == {
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


@patch.object(TrainConfig, "__post_init__", lambda x: None)
def test_get_flattened_df_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_flattened_df' method returns a flattened dataframe "
        "for the dataframe contained more than one json column"
    )
    config = TrainConfig(
        source="tests/unit/config/fixtures/data_with_two_json_columns.csv",
        epochs=10,
        drop_null=False,
        row_limit=1000,
        table_name="data_with_two_json_columns",
        metadata_path="not/existed/path/to/metadata.yaml",
        print_report=False,
        batch_size=32
    )
    config.paths = {}
    data, schema = config._extract_data()
    config._get_json_columns(data)
    assert config.json_columns == ["progress", "details"]
    assert config._get_flattened_df(data).columns.to_list() == [
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
    assert config.flattening_mapping == {
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
