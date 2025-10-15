from unittest.mock import patch, MagicMock
import pytest
import math
import functools

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
    with patch.object(VaeInferHandler, "__attrs_post_init__", lambda x: None):
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
            reports=[],
            wrapper_name="MMDVAEWrapper",
            log_level="INFO",
            type_of_process=type_of_process,
            loader=None
        )
        assert handler._get_pk_path("parent_table", "child_table") == expected_path
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "size, batch_size, nodes, run_parallel, expected_result",
    [
        # non-parallel cases
        (200, 40, 5, False, [40, 40, 40, 40, 40]),
        (100, 30, 4, False, [30, 30, 30, 10]),
        (50, 20, 3, False, [20, 20, 10]),
        (50, 20, 4, False, [20, 20, 10]),
        (50, 20, 5, False, [20, 20, 10]),
        # if batch_size is not provided one batch is created
        (100, 100, 6, False, [100]),
        (10, 10, 8, False, [10]),
        (100, 100, 16, False, [100]),

        # parallel cases
        # if batch_size is provided split exactly by batch_size
        (200, 40, 5, True, [40, 40, 40, 40, 40]),
        (100, 30, 4, True, [30, 30, 30, 10]),
        (50, 20, 3, True, [20, 20, 10]),
        (50, 20, 4, True, [20, 20, 10]),
        (50, 20, 5, True, [20, 20, 10]),
        # if batch_size is not provided split by the number of nodes
        (100, 100, 6, True, [17, 17, 17, 17, 17, 15]),
        (1000, 1000, 16, True, [63] * 15 + [55]),
        # # if not enough points for the last batch, decrease the batch_size
        (10, 10, 8, True, [1, 1, 1, 1, 1, 1, 1, 3]),
        (100, 100, 16, True, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 10])
    ],
)
@patch("os.path.exists", return_value=True)
@patch.object(VaeInferHandler, "__attrs_post_init__")
@patch("multiprocessing.cpu_count")
def test_split_by_batches(
        mock_cpu_count,
        mock_handler_post_init,
        mock_os_path_exists,
        size,
        batch_size,
        nodes,
        run_parallel,
        expected_result,
        rp_logger
):
    """
    Test the method 'split_by_batches' of the class VaeInferHandler
    """
    rp_logger.info(
        "Test the method 'split_by_batches' of the class VaeInferHandler"
    )

    mock_cpu_count.return_value = nodes
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
        run_parallel=run_parallel,
        reports=[],
        wrapper_name="MMDVAEWrapper",
        log_level="INFO",
        type_of_process="infer",
        loader=None
        )

    # attributes from the __attrs_post_init__ method
    handler.batch_num = math.ceil(handler.size / handler.batch_size)
    handler._pool = None
    handler.dataset = MagicMock()

    if run_parallel:
        with patch('multiprocessing.Pool') as mock_pool, \
             patch.object(VaeInferHandler, '_initialize_worker_vae_model'):

            mock_pool_instance = MagicMock()
            mock_pool.return_value = mock_pool_instance

            handler._setup_parallel_processing()

    result = handler.split_by_batches()

    assert result == expected_result, (
        f"split_by_batches() failed for case: "
        f"size={size}, batch_size={batch_size}, nodes={nodes}, parallel={run_parallel}\n"
        f"Expected: {expected_result}\n"
        f"Got: {result}\n"
        f"Final state: batch_size={handler.batch_size}, batch_num={handler.batch_num}"
    )

    # ensure total equals original size
    assert sum(result) == size, (
        f"Total of batches ({sum(result)}) doesn't equal original size ({size})"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "size, batch_size, cpu_count, "
    "expected_batch_num, expected_batch_size, expected_n_jobs",
    [
        (100, 20, 8, 5, 20, 5),   # batch_num > 1, batch_num < cpu_count
        (100, 24, 4, 5, 24, 4),  # batch_num > 1, batch_num > cpu_count
        (16, 4, 32, 4, 4, 4),  # batch_num > 1, size < cpu_count, even division
        (16, 5, 32, 4, 5, 4),  # batch_num > 1, size < cpu_count, uneven division
        (100, 100, 8, 8, 13, 8),  # batch_num == 1 (size == batch_size),
        (10, 10, 8, 8, 1, 8),  # batch_num == 1, case when batch_size is decreased because no points are left for the last batch
        (1, 1, 8, 1, 1, 1),  # size == 1
    ],
)
@patch('multiprocessing.cpu_count')
@patch('multiprocessing.Pool')
@patch("os.path.exists", return_value=True)
@patch.object(VaeInferHandler, "__attrs_post_init__")
def test_setup_parallel_processing(
    mock_post_init,
    mock_os_exists,
    mock_pool,
    mock_cpu_count,
    size,
    batch_size,
    cpu_count,
    expected_batch_num,
    expected_batch_size,
    expected_n_jobs,
    rp_logger
):
    """
    Test the method '_setup_parallel_processing' of the VaeInferHandler class
    """
    rp_logger.info("Test the method '_setup_parallel_processing' "
                   "of the VaeInferHandler class")

    mock_cpu_count.return_value = cpu_count

    mock_pool_instance = MagicMock()
    mock_pool.return_value = mock_pool_instance

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
        run_parallel=True,
        reports=[],
        wrapper_name="MMDVAEWrapper",
        log_level="INFO",
        type_of_process="infer",
        loader=None
    )

    # attributes from the __attrs_post_init__ method
    handler.batch_num = math.ceil(size / batch_size)
    handler._pool = None
    handler.dataset = MagicMock()

    # Patch the _initialize_worker_vae_model method
    with patch.object(VaeInferHandler, '_initialize_worker_vae_model'):
        handler._setup_parallel_processing()

        assert handler.batch_num == expected_batch_num, (
            f"Expected batch_num to be {expected_batch_num}, "
            f"got {handler.batch_num}"
        )

        assert handler.batch_size == expected_batch_size, (
            f"Expected batch_size to be {expected_batch_size}, "
            f"got {handler.batch_size}"
        )

        # Check that Pool was initialized with the correct number of processes
        mock_pool.assert_called_once()
        call_args = mock_pool.call_args
        assert call_args[1]['processes'] == expected_n_jobs
        assert call_args[1]['initializer'] == VaeInferHandler.worker_init

        # verify that partial function is callable
        assert callable(call_args[1]['initargs'][0])

    assert handler._pool is not None, "Expected _pool to be initialized"

    rp_logger.info(SUCCESSFUL_MESSAGE)
