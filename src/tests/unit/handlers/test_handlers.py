from unittest.mock import patch
import pytest

import pandas as pd

from syngen.ml.handlers import VaeInferHandler
from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.utils import fetch_training_config
from tests.conftest import SUCCESSFUL_MESSAGE


@patch("os.path.exists", return_value=True)
@pytest.mark.parametrize(
    "path_to_metadata, expected_path, type_of_process",
    [
        (
            "tests/unit/handlers/fixtures/metadata.yaml",
            "path/to/merged_infer_parent-table.csv",
            "train",
        ),
        (
            "tests/unit/handlers/fixtures/metadata_with_absent_destination.yaml",
            "path/to/merged_infer_parent-table.csv",
            "train",
        ),
        (
            "tests/unit/handlers/fixtures/metadata.yaml",
            "../data/parent_table_generated.csv",
            "infer",
        ),
        (
            "tests/unit/handlers/fixtures/metadata_with_absent_destination.yaml",
            "model_artifacts/tmp_store/parent-table/merged_infer_parent-table.csv",
            "infer",
        ),
    ],
)
def test_get_pk_path(
        mock_os_path_exists, path_to_metadata, expected_path, type_of_process, rp_logger
):
    """
    Test the method '_get_pk_path' of the VaeInferHandler
    """
    rp_logger.info("Test the method '_get_pk_path' of the VaeInferHandler")
    with patch.object(VaeInferHandler, "__post_init__", lambda x: None):
        metadata = MetadataLoader(path_to_metadata).load_data()
        handler = VaeInferHandler(
            metadata=metadata,
            table_name="parent_table",
            paths={"path_to_merged_infer": "path/to/merged_infer_parent-table.csv"},
            metadata_path=path_to_metadata,
            random_seed=0,
            size=100,
            batch_size=100,
            run_parallel=False,
            print_report=False,
            get_infer_metrics=False,
            wrapper_name="MMDVAEWrapper",
            log_level="INFO",
            type_of_process=type_of_process,
        )
        assert handler._get_pk_path("parent_table", "child_table") == expected_path
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path_to_generated_data, "
    "path_to_train_config_pickle, "
    "expected_result, "
    "expected_json_columns, "
    "expected_flattening_mapping",
    [
        (
            "tests/unit/handlers/fixtures/flattened_data_with_one_json_column.csv",
            "tests/unit/handlers/fixtures/flattened_table_with_one_json_column_train_config.pkl",
            [
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
            ],
            ["_details"],
            {
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
        ),
        (
            "tests/unit/handlers/fixtures/flattened_data_with_two_json_columns.csv",
            "tests/unit/handlers/fixtures/flattened_table_with_two_json_columns_train_config.pkl",
            [
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
            ],
            ["progress", "details"],
            {
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
        )
    ],
)
def test_post_process_generated_data(
        path_to_generated_data,
        path_to_train_config_pickle,
        expected_result,
        expected_json_columns,
        expected_flattening_mapping,
        rp_logger
):
    """
    Test the postprocessing of the flattened generated data
    """
    with patch.object(VaeInferHandler, "__post_init__", lambda x: None):
        handler = VaeInferHandler(
            metadata={},
            table_name="flattened_table",
            paths={
                "train_config_pickle_path":
                    path_to_train_config_pickle
            },
            metadata_path="",
            random_seed=0,
            size=100,
            batch_size=100,
            run_parallel=False,
            print_report=False,
            get_infer_metrics=False,
            wrapper_name="MMDVAEWrapper",
            log_level="INFO",
            type_of_process="infer",
        )
        data = pd.read_csv(path_to_generated_data)
        config = fetch_training_config(path_to_train_config_pickle)
        assert config.json_columns == expected_json_columns
        assert config.flattening_mapping == expected_flattening_mapping
        un_flattened_data = handler._post_process_generated_data(data)
        assert un_flattened_data.columns.to_list() == expected_result
