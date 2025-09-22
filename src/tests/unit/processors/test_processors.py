from unittest.mock import patch
import pytest

import pandas as pd
from pandas.testing import assert_series_equal
from syngen.ml.processors import PreprocessHandler, PostprocessHandler
from syngen.ml.data_loaders import DataLoader, MetadataLoader
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_get_json_columns_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "for the dataframe contained one json column"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_one_json_column.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]


def test_get_artifacts_contained_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained one json column"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_one_json_column.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, schema = DataLoader(path_to_data).load_data()
    json_columns = handler._get_json_columns(data)
    assert json_columns == ["_details"]

    (flattened_data,
     flattening_mapping,
     duplicated_columns) = handler._get_artifacts(data, json_columns)
    assert flattened_data.columns.to_list() == [
        "id", "created_at", "updated_at", "name",
        "description", "owner_id", "is_default",
        "is_encrypted", "status", "cluster_type",
        "master", "log_level", "env_variables",
        "ssh_config", "authentication_type",
        "username", "key_passphrase", "private_key",
        "_details_"
    ]
    assert_series_equal(
        flattened_data["_details_"],
        pd.Series([None, None, None]),
        check_names=False
    )
    assert flattening_mapping == {
        "_details": [
            "cluster_type", "master", "log_level",
            "env_variables", "ssh_config", "authentication_type",
            "username", "key_passphrase", "private_key"
        ]
    }
    assert duplicated_columns == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' method of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_two_json_columns.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["progress", "details"]


def test_get_artifacts_with_df_contained_more_than_one_json_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_two_json_columns.yaml"
    )
    data, schema = DataLoader(path_to_data).load_data()
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    json_columns = ["progress", "details"]

    (flattened_data,
     flattening_mapping,
     duplicated_columns) = handler._get_artifacts(data, json_columns)
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
        "progress_",
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
        "base_table",
        "details_"
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
    assert_series_equal(
        flattened_data["progress_"],
        pd.Series([None, None, None]),
        check_names=False
    )
    assert_series_equal(
        flattened_data["details_"],
        pd.Series([None, None, None]),
        check_names=False
    )
    assert duplicated_columns == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_column_contained_mixed_data(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "for the dataframe contained the column with mixed data types"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_column_contained_mixed_data.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_column_contained_mixed_data.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]


def test_get_artifacts_contained_column_with_mixed_data(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method of the class 'PreprocessHandler' "
        "for the dataframe contained the column with mixed data"
    )
    path_to_data = (
        f"{DIR_NAME}/unit/processors/fixtures/data_with_column_contained_mixed_data.csv"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_column_contained_mixed_data.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, schema = DataLoader(path_to_data).load_data()
    json_columns = handler._get_json_columns(data)
    assert json_columns == ["_details"]

    (flattened_data,
     flattening_mapping,
     duplicated_columns) = handler._get_artifacts(data, json_columns)
    assert flattened_data.columns.to_list() == [
        "id", "created_at", "updated_at", "name",
        "description", "owner_id", "is_default",
        "is_encrypted", "status", "cluster_type",
        "master", "log_level", "env_variables",
        "ssh_config", "authentication_type",
        "username", "key_passphrase", "private_key",
        "_details_"
    ]
    values_list = flattened_data["_details_"].fillna("?").tolist()
    for value in ["some_string", "1", "1234"]:
        assert value in values_list
    assert flattening_mapping == {
        "_details": [
            "cluster_type", "master", "log_level",
            "env_variables", "ssh_config", "authentication_type",
            "username", "key_passphrase", "private_key"
        ]
    }
    assert duplicated_columns == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_postprocess_generated_data_with_one_json_column(rp_logger):
    """
    Test the postprocessing of the flattened generated data contained one json column
    """
    rp_logger.info(
        "Test the method '_postprocess_generated_data' of the class PostprocessHandler "
        "for the flattened generated data with one json column"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column.yaml"
    )
    path_to_flattened_data = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "flattened_data_with_one_json_column.csv"
    )

    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PostprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name=None
    )
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._postprocess_generated_data(
        data=data,
        flattening_mapping={
            "_details": [
                "cluster_type",
                "master",
                "ssh_config",
                "username",
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


def test_postprocess_generated_data_with_two_json_columns(rp_logger):
    """
    Test the postprocessing of the flattened generated data contained two json columns
    """
    rp_logger.info(
        "Test the method '_postprocess_generated_data' of the class PostprocessHandler "
        "for the flattened generated data with two json columns"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        f"metadata_for_table_with_two_json_columns.yaml"
    )
    path_to_flattened_data = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "flattened_data_with_two_json_columns.csv"
    )

    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PostprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name=None
    )
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._postprocess_generated_data(
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


def test_postprocess_generated_data_with_mixed_data(rp_logger):
    """
    Test the postprocessing of the flattened generated data with the column
    contained mixed data
    """
    rp_logger.info(
        "Test the method '_postprocess_generated_data' of the class PostprocessHandler "
        "for the flattened generated data with the column contained mixed data"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        f"metadata_for_table_with_column_contained_mixed_data.yaml"
    )
    path_to_flattened_data = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "flattened_data_with_column_contained_mixed_data.csv"
    )

    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PostprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name=None
    )
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._postprocess_generated_data(
        data=data,
        flattening_mapping={
            "_details": [
                "cluster_type",
                "master",
                "ssh_config",
                "username",
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


@pytest.mark.parametrize("drop_null, row_limit, expected_result", [
    (False, None, 10),
    (True, None, 5),
    (False, 5, 5),
    (True, 6, 5),
    (True, 2, 2)
])
@patch.object(PreprocessHandler, "_save_original_schema")
def test_run(
    mock_save_original_schema,
    drop_null,
    row_limit,
    expected_result,
    rp_logger
):
    rp_logger.info(
        "Test the method 'run' of the class 'PreprocessHandler'"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_na_values.csv"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_data,
                "drop_null": drop_null,
                "row_limit": row_limit,
                "reports": []
            }
        }
    }
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=None,
        table_name="test_table"
    )
    data, schema = handler.run()
    assert schema == {
        "fields": {"registration": "removed"},
        "format": "CSV"
    }
    assert handler.row_subset == expected_result
    assert handler.initial_data_shape == (10, 12)
    assert data.shape == (expected_result, 11)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_original_schema")
def test_run_with_avro_file(
    mock_save_original_schema,
    rp_logger
):
    rp_logger.info(
        "Test the method 'run' of the class 'PreprocessHandler' "
        "by providing the file in the '.avro' format"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_na_values.avro"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_data,
                "drop_null": False,
                "row_limit": None,
                "reports": []
            }
        }
    }
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=None,
        table_name="test_table"
    )
    data, schema = handler.run()
    assert schema == {
        "fields": {
            "id_number": "int",
            "first_name": "string",
            "last_name": "string",
            "created_date": "string",
            "active": "int",
            "id": "string",
            "region": "string",
            "email": "string",
            "ratio": "float",
            "address": "string",
            "alternative_address": "string",
            "registration": "removed",
            "description": "string"
        },
        "format": "Avro"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("drop_null, row_limit, row_subset", [
    (False, None, 3),
    (True, None, 1),
    (False, 2, 2),
    (True, 2, 1),
])
@patch.object(PreprocessHandler, "_save_flatten_metadata")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_run_with_table_containing_json_columns(
    mock_save_original_schema,
    mock_save_flatten_metadata,
    drop_null,
    row_limit,
    row_subset,
    rp_logger
):
    rp_logger.info(
        "Test the method 'run' of the class 'PreprocessHandler' "
        "for the table containing one JSON column"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_one_json_column.csv"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_data,
                "drop_null": drop_null,
                "row_limit": row_limit,
                "reports": []
            }
        }
    }
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=None,
        table_name="test_table"
    )
    data, _ = handler.run()
    assert handler.row_subset == row_subset
    assert handler.initial_data_shape == (3, 10)
    assert data.shape == (row_subset, 19)
    assert data.columns.to_list() == [
        "id", "created_at", "updated_at",
        "name", "description", "owner_id",
        "is_default", "is_encrypted", "status",
        "cluster_type", "master", "log_level",
        "env_variables", "ssh_config", "authentication_type",
        "username", "key_passphrase", "private_key", "_details_"
    ]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_if_data_is_empty(rp_logger):
    rp_logger.info(
        "Test the method '_check_if_data_is_empty' of the class 'PreprocessHandler'"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/empty_table.csv"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_data,
                "drop_null": False,
                "row_limit": None,
                "reports": []
            }
        }
    }

    with pytest.raises(ValueError) as error:
        handler = PreprocessHandler(
            metadata=metadata,
            metadata_path=None,
            table_name="test_table"
        )
        handler.prepare_data()
    assert (
        str(error.value)
        == "The empty table was provided. Unable to train the table - 'test_table'"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)
