from unittest.mock import patch
import pytest
import math

from syngen.ml.handlers import VaeInferHandler
from syngen.ml.data_loaders import MetadataLoader
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


@patch("os.path.exists", return_value=True)
@pytest.mark.parametrize(
    "path_to_metadata, expected_path, type_of_process",
    [
        (
            f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml",
            "path/to/merged_infer_parent-table.csv",
            "train",
        ),
        (
            f"{DIR_NAME}/unit/handlers/fixtures/metadata_with_absent_destination.yaml",
            "path/to/merged_infer_parent-table.csv",
            "train",
        ),
        (
            f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml",
            "../data/parent_table_generated.csv",
            "infer",
        ),
        (
            f"{DIR_NAME}/unit/handlers/fixtures/metadata_with_absent_destination.yaml",
            "model_artifacts/tmp_store/parent-table/merged_infer_parent-table.csv",
            "infer",
        ),
    ],
)
def test_get_pk_path(
        mock_os_path_exists, path_to_metadata, expected_path, type_of_process, rp_logger
):
    """
    Test the method '_get_pk_path' of the class VaeInferHandler
    """
    rp_logger.info("Test the method '_get_pk_path' of the class VaeInferHandler")
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
    "size, batch_size, nodes, expected_result",
    [
        (100, 30, 4, [30, 30, 30, 10]),
        (200, 40, 5, [40, 40, 40, 40, 40]),
        (100, 100, 1, [100]),
        (100, 25, 4, [25, 25, 25, 25]),
        (100, 20, 5, [20, 20, 20, 20, 20]),
        (50, 20, 3, [20, 20, 10]),
    ],
)
@patch("os.path.exists", return_value=True)
@patch.object(VaeInferHandler, "__post_init__")
def test_split_by_batches(
        mock_handler_post_init,
        mock_os_path_exists,
        size,
        batch_size,
        nodes,
        expected_result,
        rp_logger
):
    """
    Test the method 'split_by_batches' of the class VaeInferHandler
    """
    rp_logger.info("Test the method 'split_by_batches' of the class VaeInferHandler")
    path_to_metadata = f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml"
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = VaeInferHandler(
        metadata=metadata,
        table_name="parent_table",
        paths={"path_to_merged_infer": "path/to/merged_infer_parent-table.csv"},
        metadata_path=path_to_metadata,
        random_seed=0,
        size=size,
        batch_size=batch_size,
        run_parallel=False,
        print_report=False,
        get_infer_metrics=False,
        wrapper_name="MMDVAEWrapper",
        log_level="INFO",
        type_of_process="infer",
        )
    handler.batch_num = math.ceil(handler.size / handler.batch_size)
    assert handler.split_by_batches() == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)
