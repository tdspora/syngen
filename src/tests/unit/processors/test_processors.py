from unittest.mock import patch
import pytest

import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from types import SimpleNamespace
from syngen.ml.processors import PreprocessHandler, PostprocessHandler
from syngen.ml.data_loaders import DataLoader, MetadataLoader
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_get_json_columns_contained_one_json_column_in_csv(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "for the dataframe contained one json column. The data is provided in the '.csv' format"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_one_json_column.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column_in_csv.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_contained_one_json_column_in_avro(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "for the dataframe contained one json column. The data is provided in the '.avro' format"
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_one_json_column.avro"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column_in_avro.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_artifacts_contained_one_json_column_in_csv(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained one json column. "
        "The data is provided in the '.csv' format."
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_one_json_column.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column_in_csv.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
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
    assert handler.schema == {"fields": {"_details": "removed"}, "format": "CSV"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_artifacts_contained_one_json_columns_in_avro(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained one json column. "
        "The data is provided in the '.avro' format."
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_one_json_column.avro"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column_in_avro.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
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
    assert handler.schema == {
        "fields": {
            "id": "string",
            "created_at": "string",
            "updated_at": "string",
            "name": "string",
            "description": "string",
            "_details": "removed",
            "_details_": "string",
            "authentication_type": "string",
            "cluster_type": "string",
            "env_variables": "string",
            "key_passphrase": "string",
            "log_level": "string",
            "master": "string",
            "username": "string",
            "private_key": "string",
            "ssh_config": "string",
            "username": "string",
            "owner_id": "string",
            "is_default": "boolean",
            "is_encrypted": "boolean",
            "status": "string"
        },
        "format": "Avro",
        "date_types_to_restore": {}
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_contained_more_than_one_json_column_in_csv(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' method of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column. "
        "The data is provided in the '.csv' format."
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_two_json_columns_in_csv.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["progress", "details"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_contained_more_than_one_json_column_in_avro(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' method of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column. "
        "The data is provided in the '.avro' format."
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_two_json_columns.avro"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_two_json_columns_in_avro.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["progress", "details"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_artifacts_with_df_contained_more_than_one_json_column_in_csv(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column. "
        "The data is provided in the '.csv' format."
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_two_json_columns.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_two_json_columns_in_csv.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
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
        flattened_data["details_"],
        pd.Series([None, None, None]),
        check_names=False
    )
    assert_series_equal(
        flattened_data["progress_"],
        pd.Series([None, None, None]),
        check_names=False
    )
    assert duplicated_columns == []
    assert handler.schema == {
        "fields": {"progress": "removed", "details": "removed"},
        "format": "CSV"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_artifacts_with_df_contained_more_than_one_json_column_in_avro(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method "
        "of the class 'PreprocessHandler' "
        "for the dataframe contained more than one json column. "
        "The data is provided in the '.avro' format."
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_two_json_columns.avro"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_two_json_columns_in_avro.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
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
        flattened_data["details_"],
        pd.Series([None, None, None]),
        check_names=False
    )
    assert_series_equal(
        flattened_data["progress_"],
        pd.Series([None, None, None]),
        check_names=False
    )
    assert duplicated_columns == []
    assert handler.schema == {
        "fields": {
            "id": "string",
            "created_at": "string",
            "updated_at": "string",
            "start_time": "string",
            "finish_time": "string",
            "status": "string",
            "pipeline_id": "string",
            "user_id": "string",
            "progress": "removed",
            "details": "removed",
            "info.finished": "string",
            "info.total": "string",
            "step": "string",
            "progress_": "string",
            "description": "string",
            "source.id": "string",
            "source.name": "string",
            "source.connection_string": "string",
            "target.id": "string",
            "target.name": "string",
            "target.connection_string": "string",
            "cluster.id": "string",
            "cluster.name": "string",
            "cluster.master_webui": "string",
            "cluster.history_server": "string",
            "integrity_type": "string",
            "total_tables": "string",
            "included_tables": "string",
            "base_table": "string",
            "details_": "string"
        },
        "format": "Avro",
        "date_types_to_restore": {}
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_column_contained_mixed_data_in_csv(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "for the dataframe contained the column with mixed data types. "
        "The data is provided in the '.csv' format."
    )
    path_to_data = f"{DIR_NAME}/unit/processors/fixtures/data_with_column_contained_mixed_data.csv"
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_column_contained_mixed_data_in_csv.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_column_contained_mixed_data_in_avro(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "for the dataframe contained the column with mixed data types. "
        "The data is provided in the '.avro' format."
    )
    path_to_data = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "data_with_column_contained_mixed_data.avro"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_column_contained_mixed_data_in_avro.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
    assert handler._get_json_columns(data) == ["_details"]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_artifacts_contained_column_with_mixed_data_in_csv(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method of the class 'PreprocessHandler' "
        "for the dataframe contained the column with mixed data. "
        "The data is provided in the '.csv' format."
    )
    path_to_data = (
        f"{DIR_NAME}/unit/processors/fixtures/data_with_column_contained_mixed_data.csv"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_column_contained_mixed_data_in_csv.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
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
    assert handler.schema == {"fields": {"_details": "removed"}, "format": "CSV"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_artifacts_contained_column_with_mixed_data_in_avro(rp_logger):
    rp_logger.info(
        "Test that the method '_get_artifacts' method of the class 'PreprocessHandler' "
        "for the dataframe contained the column with mixed data. "
        "The data is provided in the '.avro' format."
    )
    path_to_data = (
        f"{DIR_NAME}/unit/processors/fixtures/data_with_column_contained_mixed_data.avro"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_column_contained_mixed_data_in_avro.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name="test_table"
    )
    data, handler.schema = DataLoader(path_to_data).load_data()
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
    assert handler.schema == {
        "fields": {
            "_details": "removed",
            "_details_": "string",
            "authentication_type": "string",
            "cluster_type": "string",
            "created_at": "string",
            "description": "string",
            "env_variables": "string",
            "id": "string",
            "is_default": "boolean",
            "is_encrypted": "boolean",
            "key_passphrase": "string",
            "log_level": "string",
            "master": "string",
            "name": "string",
            "owner_id": "string",
            "private_key": "string",
            "ssh_config": "string",
            "status": "string",
            "updated_at": "string",
            "username": "string"
        },
        "format": "Avro",
        "date_types_to_restore": {}
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_no_json(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "returns an empty list when no JSON columns are present"
    )
    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": ["hello", "world", "foo"]
    })
    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
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
    handler.schema = {"fields": {}, "format": "CSV"}
    assert handler._get_json_columns(df) == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_all_null_column(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "returns an empty list when a column contains all null values"
    )
    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": [None, None, None]
    })
    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
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
    handler.schema = {"fields": {}, "format": "CSV"}
    assert handler._get_json_columns(df) == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_json_columns_with_json_list_not_dict(rp_logger):
    rp_logger.info(
        "Test that the method '_get_json_columns' of the class 'PreprocessHandler' "
        "returns an empty list for a column containing JSON arrays (not dicts)"
    )
    df = pd.DataFrame({
        "col_a": ['[1, 2, 3]', '[4, 5, 6]', '[7, 8, 9]']
    })
    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
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
    handler.schema = {"fields": {}, "format": "CSV"}
    assert handler._get_json_columns(df) == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_unflatten_generated_data_with_one_json_column(rp_logger):
    """
    Test the unflattening of the flattened generated data contained one json column
    """
    rp_logger.info(
        "Test the method '_unflatten_generated_data' of the class PostprocessHandler "
        "for the flattened generated data with one json column"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "metadata_for_table_with_one_json_column_in_csv.yaml"
    )
    path_to_flattened_data = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "flattened_data_with_one_json_column.csv"
    )

    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PostprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name=None,
        type_of_process="infer"
    )
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._unflatten_generated_data(
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_unflatten_generated_data_with_two_json_columns(rp_logger):
    """
    Test the unflattening of the flattened generated data contained two json columns
    """
    rp_logger.info(
        "Test the method '_unflatten_generated_data' of the class PostprocessHandler "
        "for the flattened generated data with two json columns"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        f"metadata_for_table_with_two_json_columns_in_csv.yaml"
    )
    path_to_flattened_data = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "flattened_data_with_two_json_columns.csv"
    )

    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PostprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name=None,
        type_of_process="infer"
    )
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._unflatten_generated_data(
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_unflatten_generated_data_with_mixed_data(rp_logger):
    """
    Test the unflattening of the flattened generated data with the column
    contained mixed data
    """
    rp_logger.info(
        "Test the method '_unflatten_generated_data' of the class PostprocessHandler "
        "for the flattened generated data with the column contained mixed data"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        f"metadata_for_table_with_column_contained_mixed_data_in_csv.yaml"
    )
    path_to_flattened_data = (
        f"{DIR_NAME}/unit/processors/fixtures/"
        "flattened_data_with_column_contained_mixed_data.csv"
    )

    metadata = MetadataLoader(path_to_metadata).load_data()

    handler = PostprocessHandler(
        metadata=metadata,
        metadata_path=path_to_metadata,
        table_name=None,
        type_of_process="infer"
    )
    data = handler._load_generated_data(path_to_flattened_data)
    un_flattened_data = handler._unflatten_generated_data(
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("drop_null, row_limit, expected_result", [
    (False, None, 10),
    (True, None, 5),
    (False, 5, 5),
    (True, 6, 5),
    (True, 2, 2)
])
@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_run(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
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


@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_run_with_avro_file(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
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
    _, schema = handler.run()
    assert schema == {
        "fields": {
            "id_number": "int",
            "first_name": "string",
            "last_name": "string",
            "created_date": "string",
            "active": "boolean",
            "id": "string",
            "region": "string",
            "email": "string",
            "ratio": "float",
            "address": "string",
            "alternative_address": "string",
            "registration": "removed",
            "description": "string"
        },
        "format": "Avro",
        "date_types_to_restore": {}
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("drop_null, row_limit, row_subset", [
    (False, None, 3),
    (True, None, 1),
    (False, 2, 2),
    (True, 2, 1),
])
@patch.object(PreprocessHandler, "_save_flatten_metadata")
@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_run_with_table_containing_json_columns(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
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


@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_remove_empty_columns(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_remove_empty_columns' of the class 'PreprocessHandler' "
        "removes completely empty columns and updates the schema"
    )
    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": [None, None, None],
        "col_c": ["x", "y", "z"],
        "col_d": [np.nan, np.nan, np.nan]
    })

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
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
    handler.original_df = df
    handler.initial_order_of_columns = df.columns.to_list()
    handler.schema = {"fields": {}, "format": "CSV"}

    result = handler._remove_empty_columns()
    assert "col_b" not in result.columns
    assert "col_d" not in result.columns
    assert "col_a" in result.columns
    assert "col_c" in result.columns
    assert handler.schema["fields"]["col_b"] == "removed"
    assert handler.schema["fields"]["col_d"] == "removed"
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_remove_empty_columns_no_empty(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_remove_empty_columns' does not remove "
        "columns aren't completely empty"
    )
    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": [4, None, 6],
        "col_c": ["x", "y", "z"]
    })

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
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
    handler.original_df = df
    handler.initial_order_of_columns = df.columns.to_list()
    handler.schema = {"fields": {}, "format": "CSV"}

    result = handler._remove_empty_columns()
    assert result.columns.to_list() == ["col_a", "col_b", "col_c"]
    assert handler.schema["fields"] == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_flatten_metadata")
@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_handle_json_columns_no_json(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    mock_save_flatten_metadata,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_handle_json_columns' of the class 'PreprocessHandler' "
        "returns data unchanged when no JSON columns are present"
    )
    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": ["hello", "world", "foo"]
    })
    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
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
    handler.schema = {"fields": {}, "format": "CSV"}
    result = handler._handle_json_columns(df)
    assert result.equals(df)
    mock_save_flatten_metadata.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_flatten_metadata")
@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_handle_json_columns_with_json(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    mock_save_flatten_metadata,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_handle_json_columns' of the class 'PreprocessHandler' "
        "flattens JSON columns and saves flatten metadata"
    )
    df = pd.DataFrame({
        "id": [1, 2],
        "data": ['{"key1": "val1", "key2": "val2"}', '{"key1": "val3", "key2": "val4"}']
    })
    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
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
    handler.schema = {"fields": {}, "format": "CSV"}
    result = handler._handle_json_columns(df)
    assert result.columns.to_list() == ["id", "key1", "key2", "data_"]
    assert handler.schema["fields"]["data"] == "removed"
    mock_save_flatten_metadata.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_load_source_with_custom_loader(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_load_source' of the class 'PreprocessHandler' "
        "loads data using a custom loader callback"
    )
    expected_df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": ["x", "y", "z"]
    })

    def custom_loader(table_name):
        return expected_df

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "unused_path.csv",
                "drop_null": False,
                "row_limit": None,
                "reports": []
            }
        }
    }
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=None,
        table_name="test_table",
        loader=custom_loader
    )
    handler._load_source()
    assert handler.original_df.equals(expected_df)
    assert handler.schema == {"fields": {}, "format": "CSV"}
    assert handler.initial_data_shape == (3, 2)
    assert handler.initial_order_of_columns == ["col_a", "col_b"]
    assert handler.row_subset == 3
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_preprocess_data_with_loader_ignores_drop_null_and_row_limit(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_preprocess_data' of the class 'PreprocessHandler' "
        "ignores 'drop_null' and 'row_limit' when a custom loader is used"
    )
    df = pd.DataFrame({
        "col_a": [1, None, 3, 4, 5],
        "col_b": ["x", "y", "z", "w", "v"]
    })

    def custom_loader(table_name):
        return df

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "unused_path.csv",
                "drop_null": True,
                "row_limit": 2,
                "reports": []
            }
        }
    }
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=None,
        table_name="test_table",
        loader=custom_loader
    )
    handler._load_source()
    handler.schema = {"fields": {}, "format": "CSV"}
    result = handler._preprocess_data()
    # With loader, drop_null and row_limit are ignored,
    # so the full dataframe is returned
    assert len(result) == 5
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_preprocess_data_drop_null_results_in_empty_df(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_preprocess_data' of the class 'PreprocessHandler' "
        "ignores 'drop_null' when it results in an empty dataframe"
    )
    # All rows have at least one null value
    df = pd.DataFrame({
        "col_a": [1, None, 3],
        "col_b": [None, "y", None]
    })

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
                "drop_null": True,
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
    handler.original_df = df
    handler.initial_data_shape = df.shape
    handler.initial_order_of_columns = df.columns.to_list()
    handler.row_subset = len(df)
    handler.schema = {"fields": {}, "format": "CSV"}

    result = handler._preprocess_data()
    # drop_null is ignored because it results in empty df
    assert len(result) == 3
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("os.system")
@patch("os.path.exists")
def test_run_script_when_script_exists(
    mock_exists,
    mock_system,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_run_script' of the class 'PreprocessHandler' "
        "runs the script when it exists"
    )
    mock_exists.return_value = True
    PreprocessHandler._run_script()
    mock_system.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("os.system")
@patch("os.path.exists")
def test_run_script_when_script_does_not_exist(
    mock_exists,
    mock_system,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_run_script' of the class 'PreprocessHandler' "
        "does not run the script when it does not exist"
    )
    mock_exists.return_value = False
    PreprocessHandler._run_script()
    mock_system.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_flatten_metadata")
@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_run_with_custom_loader(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    mock_save_flatten_metadata,
    rp_logger
):
    rp_logger.info(
        "Test the method 'run' of the class 'PreprocessHandler' "
        "with a custom loader callback"
    )
    df = pd.DataFrame({
        "col_a": list(range(100)),
        "col_b": [f"val_{i}" for i in range(100)]
    })

    def custom_loader(table_name):
        return df

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "unused_path.csv",
                "drop_null": True,
                "row_limit": 50,
                "reports": []
            }
        }
    }
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=None,
        table_name="test_table",
        loader=custom_loader
    )
    data, schema = handler.run()
    # With loader, drop_null and row_limit are ignored
    assert len(data) == 100
    assert schema == {"fields": {}, "format": "CSV"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(PreprocessHandler, "_save_initial_order_of_columns")
@patch.object(PreprocessHandler, "_save_original_schema")
def test_preprocess_data_row_limit_larger_than_data(
    mock_save_original_schema,
    mock_save_initial_order_of_columns,
    rp_logger
):
    rp_logger.info(
        "Test that the method '_preprocess_data' of the class 'PreprocessHandler' "
        "handles row_limit larger than the data size"
    )
    df = pd.DataFrame({
        "col_a": list(range(10)),
        "col_b": list(range(10))
    })

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "dummy_path.csv",
                "drop_null": False,
                "row_limit": 100,
                "reports": []
            }
        }
    }
    handler = PreprocessHandler(
        metadata=metadata,
        metadata_path=None,
        table_name="test_table"
    )
    handler.original_df = df
    handler.initial_data_shape = df.shape
    handler.initial_order_of_columns = df.columns.to_list()
    handler.row_subset = len(df)
    handler.schema = {"fields": {}, "format": "CSV"}

    result = handler._preprocess_data()
    assert len(result) == 10
    assert handler.row_subset == 10
    rp_logger.info(SUCCESSFUL_MESSAGE)


# tests for PostprocessHandler.restore_int_dtypes
def _make_dataset_config(
        int_columns,
        nan_labels_dict=None,
        table_name="test_table"
):
    return SimpleNamespace(
        int_columns=set(int_columns),
        nan_labels_dict=nan_labels_dict or {},
        table_name=table_name,
    )


def test_restore_int_dtypes_plain_int_column_cast_to_int64(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' casts a plain integer column "
        "(stored as float, no NaN, no nan-label) to 'int64'."
    )
    data = pd.DataFrame({"age": [1.0, 2.0, 3.0]})
    config = _make_dataset_config(["age"])
    result = PostprocessHandler.restore_int_dtypes(data, config)
    assert result["age"].dtype == np.dtype("int64")
    assert list(result["age"]) == [1, 2, 3]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_restore_int_dtypes_int_column_with_nan_cast_to_Int64(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' casts an integer column "
        "with NaN values to nullable 'Int64'."
    )
    data = pd.DataFrame({"age": [1.0, np.nan, 3.0]})
    config = _make_dataset_config(["age"])
    result = PostprocessHandler.restore_int_dtypes(data, config)
    assert result["age"].dtype == pd.Int64Dtype()
    assert result["age"][0] == 1
    assert pd.isna(result["age"][1])
    assert result["age"][2] == 3
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_restore_int_dtypes_nan_label_sentinel_preserved(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' preserves the nan-label sentinel string "
        "while converting numeric values to int in an object column."
    )
    data = pd.DataFrame({"score": [1.0, "missing", 3.0, "missing"]})
    config = _make_dataset_config(["score"], nan_labels_dict={"score": "missing"})
    result = PostprocessHandler.restore_int_dtypes(data, config)
    assert result["score"].dtype == object
    assert result["score"][0] == 1
    assert result["score"][1] == "missing"
    assert result["score"][2] == 3
    assert result["score"][3] == "missing"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_restore_int_dtypes_nan_label_and_actual_nan_both_preserved(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' preserves both the nan-label sentinel "
        "and actual NaN values alongside integer values."
    )
    data = pd.DataFrame({"score": [1.0, "missing", np.nan, 4.0]})
    config = _make_dataset_config(["score"], nan_labels_dict={"score": "missing"})
    result = PostprocessHandler.restore_int_dtypes(data, config)
    assert result["score"].dtype == object
    assert result["score"][0] == 1
    assert result["score"][1] == "missing"
    assert pd.isna(result["score"][2])
    assert result["score"][3] == 4
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_restore_int_dtypes_column_not_in_data_is_skipped(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' silently skips columns listed in "
        "'int_columns' that are not present in the DataFrame."
    )
    data = pd.DataFrame({"age": [1.0, 2.0]})
    config = _make_dataset_config(["age", "nonexistent"])
    result = PostprocessHandler.restore_int_dtypes(data, config)
    assert "nonexistent" not in result.columns
    assert result["age"].dtype == np.dtype("int64")
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_restore_int_dtypes_multiple_columns_mixed(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' handles multiple columns: "
        "one plain int, one with NaN, one with a nan-label sentinel."
    )
    data = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [1.0, np.nan, 3.0],
        "c": [10.0, "n/a", 30.0],
    })
    config = _make_dataset_config(["a", "b", "c"], nan_labels_dict={"c": "n/a"})
    result = PostprocessHandler.restore_int_dtypes(data, config)
    assert result["a"].dtype == np.dtype("int64")
    assert result["b"].dtype == pd.Int64Dtype()
    assert result["c"].dtype == object
    assert result["c"][0] == 10
    assert result["c"][1] == "n/a"
    assert result["c"][2] == 30
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_restore_int_dtypes_no_int_columns_data_unchanged(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' leaves the DataFrame unchanged "
        "when 'int_columns' is empty."
    )
    data = pd.DataFrame({"name": ["Alice", "Bob"], "score": [1.5, 2.5]})
    config = _make_dataset_config([])
    result = PostprocessHandler.restore_int_dtypes(data, config)
    assert_series_equal(result["score"], data["score"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_restore_int_dtypes_debug_log_not_emitted_when_no_columns_restored(rp_logger):
    rp_logger.info(
        "Test that 'restore_int_dtypes' does not emit a debug log "
        "when 'int_columns' is empty (nothing to restore)."
    )
    data = pd.DataFrame({"score": [1.5, 2.5]})
    config = _make_dataset_config([])
    with patch("syngen.ml.processors.processors.logger") as mock_logger:
        PostprocessHandler.restore_int_dtypes(data, config)
    mock_logger.debug.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)
