from syngen.ml.preprocess import PreprocessHandler
from syngen.ml.data_loaders import DataLoader
from tests.conftest import SUCCESSFUL_MESSAGE


def test_get_json_columns_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_json_columns' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained one json column"
    )
    path_to_data = "tests/unit/preprocess/fixtures/data_with_one_json_column.csv"
    path_to_metadata = ("tests/unit/preprocess/fixtures/"
                        "metadata_for_table_with_one_json_column.yaml")
    handler = PreprocessHandler(path_to_metadata)
    data, schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]


def test_get_flattened_df_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_flattened_df' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained one json column"
    )
    path_to_data = "tests/unit/preprocess/fixtures/data_with_one_json_column.csv"
    path_to_metadata = ("tests/unit/preprocess/fixtures/"
                        "metadata_for_table_with_one_json_column.yaml")
    handler = PreprocessHandler(path_to_metadata)
    data, schema = DataLoader(path_to_data).load_data()
    json_columns = ["_details"]
    flattened_data, flattening_mapping = handler._get_flattened_df(data, json_columns)
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
        "Test that the method 'get_json_columns' method of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column"
    )
    path_to_data = "tests/unit/preprocess/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = ("tests/unit/preprocess/fixtures/"
                        "metadata_for_table_with_two_json_columns.yaml")
    handler = PreprocessHandler(path_to_metadata)
    data, schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["progress", "details"]


def test_get_flattened_df_with_df_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method 'get_flattened_df' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column"
    )
    path_to_data = "tests/unit/preprocess/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = ("tests/unit/preprocess/fixtures/"
                        "metadata_for_table_with_two_json_columns.yaml")
    data, schema = DataLoader(path_to_data).load_data()
    handler = PreprocessHandler(path_to_metadata)
    json_columns = ["progress", "details"]
    flattened_data, flattening_mapping = handler._get_flattened_df(data, json_columns)
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
