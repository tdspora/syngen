from syngen.ml.processors import PreprocessHandler, PostprocessHandler
from syngen.ml.data_loaders import DataLoader
from tests.conftest import SUCCESSFUL_MESSAGE


def test_get_json_columns_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained one json column"
    )
    path_to_data = "tests/unit/processors/fixtures/data_with_one_json_column.csv"
    path_to_metadata = ("tests/unit/processors/fixtures/"
                        "metadata_for_table_with_one_json_column.yaml")
    handler = PreprocessHandler(path_to_metadata, None, {})
    data, schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]


def test_get_flattened_df_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_flattened_df' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained one json column"
    )
    path_to_data = "tests/unit/processors/fixtures/data_with_one_json_column.csv"
    path_to_metadata = ("tests/unit/processors/fixtures/"
                        "metadata_for_table_with_one_json_column.yaml")
    handler = PreprocessHandler(path_to_metadata, None, {})
    data, schema = DataLoader(path_to_data).load_data()
    json_columns = ["_details"]
    (flattened_data,
     flattening_mapping,
     duplicated_columns) = handler._get_flattened_df(data, json_columns)
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
    assert duplicated_columns == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' method of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column"
    )
    path_to_data = "tests/unit/processors/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = ("tests/unit/processors/fixtures/"
                        "metadata_for_table_with_two_json_columns.yaml")
    handler = PreprocessHandler(path_to_metadata, None, {})
    data, schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["progress", "details"]


def test_get_flattened_df_with_df_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_flattened_df' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column"
    )
    path_to_data = "tests/unit/processors/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = ("tests/unit/processors/fixtures/"
                        "metadata_for_table_with_two_json_columns.yaml")
    data, schema = DataLoader(path_to_data).load_data()
    handler = PreprocessHandler(path_to_metadata, None, {})
    json_columns = ["progress", "details"]
    (flattened_data,
     flattening_mapping,
     duplicated_columns) = handler._get_flattened_df(data, json_columns)
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
    assert duplicated_columns == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_post_process_generated_data_with_one_json_column(rp_logger):
    """
    Test the postprocessing of the flattened generated data contained one json column
    """
    rp_logger.info(
        "Test the method '_post_process_generated_data' fo the class PostprocessHandler "
        "for the flattened generated data with one json column"
    )
    path_to_metadata = ("tests/unit/processors/fixtures/"
                        "metadata_for_table_with_one_json_column.yaml")
    path_to_flattened_data = ("tests/unit/processors/fixtures/"
                              "flattened_data_with_one_json_column.csv")
    handler = PostprocessHandler(path_to_metadata, None, {})
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._post_process_generated_data(
        data=data,
        flattening_mapping={
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
        },
        duplicated_columns=[]
    )
    assert un_flattened_data.columns.to_list() == [
            "id",
            "created_at",
            "updated_at",
            "name",
            "description",
            "owner_id",
            "is_default",
            "is_encrypted",
            "status",
            "_details"
        ]


def test_post_process_generated_data_with_two_json_columns(rp_logger):
    """
    Test the postprocessing of the flattened generated data contained two json columns
    """
    rp_logger.info(
        "Test the method '_post_process_generated_data' fo the class PostproccesorHandler "
        "for the flattened generated data with one json column"
    )
    path_to_metadata = ("tests/unit/processors/fixtures/"
                        "metadata_for_table_with_two_json_columns.yaml")
    path_to_flattened_data = ("tests/unit/processors/fixtures/"
                              "flattened_data_with_two_json_columns.csv")
    handler = PostprocessHandler(path_to_metadata, None, {})
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._post_process_generated_data(
        data=data,
        flattening_mapping={
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
        },
        duplicated_columns=[]
    )
    assert un_flattened_data.columns.to_list() == [
        "id",
        "created_at",
        "updated_at",
        "start_time",
        "finish_time",
        "status",
        "pipeline_id",
        "user_id",
        "progress",
        "details"
    ]