from unittest.mock import patch, MagicMock
import pytest
import math
import functools

import numpy as np
import pandas as pd
import dill
from scipy.stats import gaussian_kde

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
        # if batch_size is not provided split by (nodes - 1)
        (100, 100, 6, True, [20, 20, 20, 20, 20]),
        (1000, 1000, 16, True, [67] * 14 + [62]),
        (10, 10, 8, True, [1, 1, 1, 1, 1, 1, 4]),
        (100, 100, 16, True, [7] * 14 + [2]),
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
        (100, 20, 8, 5, 20, 5),      # batch_num > 1, batch_num < cpu_count
        (100, 24, 4, 5, 24, 3),      # batch_num > 1, batch_num > cpu_count
        (16, 4, 32, 4, 4, 4),        # batch_num > 1, size < cpu_count, even division
        (16, 5, 32, 4, 5, 4),        # batch_num > 1, size < cpu_count, uneven division
        (100, 100, 8, 7, 15, 7),     # batch_num == 1 (size == batch_size),
        (10, 10, 8, 7, 1, 7),        # batch_num == 1, case when batch_size is decreased because no points are left for the last batch
        (1, 1, 8, 1, 1, 1),          # size == 1
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

# ---------------------------------------------------------------------------
# kde_gen: slugified filename lookup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fk_label, expected_slug",
    [
        # underscore → hyphen after slugify
        ("brand_id", "brand-id"),
        # no special chars → unchanged
        ("brandid", "brandid"),
        # mixed case + underscore
        ("Brand_ID", "brand-id"),
    ],
)
@patch("os.path.exists", return_value=True)
@patch.object(VaeInferHandler, "__attrs_post_init__", lambda x: None)
def test_kde_gen_loads_kde_with_slugified_filename(
    mock_os_path_exists, fk_label, expected_slug, tmp_path, rp_logger
):
    """
    Test that kde_gen opens the KDE pickle using the slugified fk_label,
    ensuring consistency with _save_kde_artifacts (decorated with @slugify_parameters).
    """
    rp_logger.info(
        "Test that kde_gen loads KDE file with slugified fk_label"
    )
    # create a KDE from simple numeric data and save it with the slugified name
    kde = gaussian_kde([0, 1, 2])
    kde_file = tmp_path / f"{expected_slug}.pkl"
    with open(kde_file, "wb") as f:
        dill.dump(kde, f)

    fk_kde_path = str(tmp_path) + "/"
    pk_table = pd.DataFrame({"pk_col": [10, 20, 30]})

    path_to_metadata = f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml"
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = VaeInferHandler(
        metadata=metadata,
        table_name="parent_table",
        paths={
            "path_to_merged_infer": "path/to/merged_infer_parent-table.csv",
            "fk_kde_path": fk_kde_path,
        },
        metadata_path=path_to_metadata,
        random_seed=0,
        size=100,
        batch_size=100,
        run_parallel=False,
        reports=[],
        wrapper_name="MMDVAEWrapper",
        log_level="INFO",
        type_of_process="infer",
        loader=None,
    )

    result = handler.kde_gen(pk_table, "pk_col", size=5, fk_label=fk_label)
    assert len(result) == 5
    assert set(result[fk_label].unique()).issubset({10, 20, 30})
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# kde_gen: is_string_dtype check for text PK columns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pk_values, dtype_desc",
    [
        # object dtype – classic pandas string storage
        (pd.Series(["a", "b", "c"], dtype="object"), "object"),
        # nullable StringDtype
        (pd.Series(["a", "b", "c"], dtype=pd.StringDtype()), "StringDtype"),
        # string[python] alias
        (pd.Series(["a", "b", "c"], dtype="string[python]"), "string_python"),
    ],
)
@patch("os.path.exists", return_value=True)
@patch.object(VaeInferHandler, "__attrs_post_init__", lambda x: None)
def test_kde_gen_handles_string_pk_dtypes(
    mock_os_path_exists, pk_values, dtype_desc, tmp_path, rp_logger
):
    """
    Test that kde_gen correctly detects string PK columns for all string-like
    dtypes (object, StringDtype, string[python]) and falls back to positional
    indices when evaluating the KDE.
    """
    rp_logger.info(
        f"Test that kde_gen handles string PK dtype: {dtype_desc}"
    )
    kde = gaussian_kde([0, 1, 2])
    kde_file = tmp_path / "fk-col.pkl"
    with open(kde_file, "wb") as f:
        dill.dump(kde, f)

    fk_kde_path = str(tmp_path) + "/"
    pk_table = pd.DataFrame({"pk_col": pk_values})

    path_to_metadata = f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml"
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = VaeInferHandler(
        metadata=metadata,
        table_name="parent_table",
        paths={
            "path_to_merged_infer": "path/to/merged_infer_parent-table.csv",
            "fk_kde_path": fk_kde_path,
        },
        metadata_path=path_to_metadata,
        random_seed=0,
        size=100,
        batch_size=100,
        run_parallel=False,
        reports=[],
        wrapper_name="MMDVAEWrapper",
        log_level="INFO",
        type_of_process="infer",
        loader=None,
    )

    result = handler.kde_gen(pk_table, "pk_col", size=5, fk_label="fk_col")
    assert len(result) == 5
    # all generated FK values should come from the PK table
    assert set(result["fk_col"].unique()).issubset(set(pk_values))
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "pk_values, dtype_desc",
    [
        # int PK – should be passed directly to KDE (no index substitution)
        (pd.Series([10, 20, 30]), "int"),
        # float PK
        (pd.Series([1.1, 2.2, 3.3]), "float"),
    ],
)
@patch("os.path.exists", return_value=True)
@patch.object(VaeInferHandler, "__attrs_post_init__", lambda x: None)
def test_kde_gen_handles_numeric_pk_dtypes(
    mock_os_path_exists, pk_values, dtype_desc, tmp_path, rp_logger
):
    """
    Test that kde_gen passes numeric PK values directly to the KDE evaluator
    without index substitution.
    """
    rp_logger.info(
        f"Test that kde_gen handles numeric PK dtype: {dtype_desc}"
    )
    kde = gaussian_kde(pk_values.astype(float).values)
    kde_file = tmp_path / "fk-col.pkl"
    with open(kde_file, "wb") as f:
        dill.dump(kde, f)

    fk_kde_path = str(tmp_path) + "/"
    pk_table = pd.DataFrame({"pk_col": pk_values})

    path_to_metadata = f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml"
    metadata = MetadataLoader(path_to_metadata).load_data()
    handler = VaeInferHandler(
        metadata=metadata,
        table_name="parent_table",
        paths={
            "path_to_merged_infer": "path/to/merged_infer_parent-table.csv",
            "fk_kde_path": fk_kde_path,
        },
        metadata_path=path_to_metadata,
        random_seed=0,
        size=100,
        batch_size=100,
        run_parallel=False,
        reports=[],
        wrapper_name="MMDVAEWrapper",
        log_level="INFO",
        type_of_process="infer",
        loader=None,
    )

    result = handler.kde_gen(pk_table, "pk_col", size=5, fk_label="fk_col")
    assert len(result) == 5
    assert set(result["fk_col"].unique()).issubset(set(pk_values))
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_kde_gen_fallback_on_missing_kde(tmp_path, rp_logger):
    """
    Test that kde_gen falls back to simple sampling when the KDE .pkl file
    does not exist, without raising an exception.
    """
    rp_logger.info(
        "Test that kde_gen falls back to simple sampling on missing KDE file"
    )
    fk_kde_path = str(tmp_path) + "/"
    pk_table = pd.DataFrame({"pk_col": [10, 20, 30]})

    path_to_metadata = f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml"
    metadata = MetadataLoader(path_to_metadata).load_data()
    with patch.object(VaeInferHandler, "__attrs_post_init__", lambda x: None):
        handler = VaeInferHandler(
            metadata=metadata,
            table_name="parent_table",
            paths={
                "path_to_merged_infer": "path/to/merged_infer_parent-table.csv",
                "fk_kde_path": fk_kde_path,
            },
            metadata_path=path_to_metadata,
            random_seed=0,
            size=100,
            batch_size=100,
            run_parallel=False,
            reports=[],
            wrapper_name="MMDVAEWrapper",
            log_level="INFO",
            type_of_process="infer",
            loader=None,
        )

    result = handler.kde_gen(pk_table, "pk_col", size=5, fk_label="nonexistent_fk")
    assert len(result) == 5
    assert set(result.values.flatten()).issubset({10, 20, 30})
    rp_logger.info(SUCCESSFUL_MESSAGE)