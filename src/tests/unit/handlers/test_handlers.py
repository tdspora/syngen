from unittest.mock import patch, MagicMock
import multiprocessing as mp
from collections import OrderedDict
import pytest
import math
import warnings

import numpy as np
import pandas as pd
import dill
import torch
from scipy.stats import gaussian_kde

from syngen.ml.handlers import VaeInferHandler
from syngen.ml.handlers.handlers import _select_mp_start_method, LongTextsHandler
from syngen.ml.data_loaders import MetadataLoader
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def _get_worker_ordinal(_):
    """Module-level (not nested) so it stays picklable under `spawn`."""
    return mp.current_process()._identity[0]


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


@patch("os.path.exists", return_value=True)
def test_get_wrapper_forwards_device_without_nameerror(mock_os_path_exists, rp_logger):
    """
    Regression test for a 'device' NameError in VaeInferHandler._get_wrapper:
    the non-parallel infer path (__attrs_post_init__'s
    `self._get_wrapper(dataset_to_preload=self.dataset)` call, made with no
    'device' kwarg) crashed because `_get_wrapper` built its wrapper_kwargs
    dict from a bare `device` name that was never declared as a parameter -
    a NameError on every non-parallel infer run, regardless of device.
    """
    rp_logger.info(
        "Test that VaeInferHandler._get_wrapper accepts and forwards 'device'"
    )
    with patch.object(VaeInferHandler, "__attrs_post_init__", lambda x: None):
        path_to_metadata = f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml"
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
            type_of_process="infer",
            loader=None
        )
        with patch.object(VaeInferHandler, "create_wrapper") as mock_create_wrapper:
            handler._get_wrapper(dataset_to_preload="fake_dataset")
            assert mock_create_wrapper.call_args.kwargs["device"] is None

            explicit_device = torch.device("cpu")
            handler._get_wrapper(dataset_to_preload="fake_dataset", device=explicit_device)
            assert mock_create_wrapper.call_args.kwargs["device"] == explicit_device
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
@patch("syngen.ml.handlers.handlers.get_thread_parallelism_budget")
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
    "expected_batch_num, expected_batch_size, expected_n_jobs, "
    "expected_threads_per_worker",
    [
        # batch_num > 1, batch_num < cpu_count
        (100, 20, 8, 5, 20, 5, 1),
        # batch_num > 1, batch_num > cpu_count
        (100, 24, 4, 5, 24, 3, 1),
        # batch_num > 1, size < cpu_count, even division
        (16, 4, 32, 4, 4, 4, 7),
        # batch_num > 1, size < cpu_count, uneven division
        (16, 5, 32, 4, 5, 4, 7),
        # batch_num == 1 (size == batch_size)
        (100, 100, 8, 7, 15, 7, 1),
        # batch_num == 1, case when batch_size is decreased
        # because no points are left for the last batch
        (10, 10, 8, 7, 1, 7, 1),
        # size == 1
        (1, 1, 8, 1, 1, 1, 7),
    ],
)
@patch('syngen.ml.handlers.handlers.get_thread_parallelism_budget')
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
    expected_threads_per_worker,
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
        assert call_args[1]['initargs'][1] == expected_threads_per_worker

        # verify that partial function is callable
        assert callable(call_args[1]['initargs'][0])

    assert handler._pool is not None, "Expected _pool to be initialized"

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.handlers.handlers.torch.set_num_threads")
def test_worker_init_limits_native_threads(mock_set_num_threads, rp_logger):
    mock_wrapper = MagicMock()

    VaeInferHandler.worker_init(mock_wrapper, threads_per_worker=2)

    mock_set_num_threads.assert_called_once_with(2)
    mock_wrapper.assert_called_once_with()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.handlers.handlers.torch.set_num_threads")
def test_worker_init_keeps_legacy_single_argument_usage(mock_set_num_threads, rp_logger):
    mock_wrapper = MagicMock()

    VaeInferHandler.worker_init(mock_wrapper)

    mock_set_num_threads.assert_not_called()
    mock_wrapper.assert_called_once_with()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "platform, gpu_count, expected",
    [
        ("linux", 0, "fork"),
        ("linux", 1, "spawn"),
        ("linux", 4, "spawn"),
        ("win32", 0, "spawn"),
        ("win32", 4, "spawn"),
    ],
)
def test_select_mp_start_method(platform, gpu_count, expected, rp_logger):
    """`spawn` is required whenever any GPU is visible (or on Windows) -
    forking after the parent has touched CUDA is unsafe regardless of how
    many GPUs this specific pool will use. On CPU-only POSIX machines this
    must stay byte-identical to the pre-existing `fork` behavior."""
    rp_logger.info(
        "Test '_select_mp_start_method' picks spawn whenever a GPU is visible or on Windows"
    )
    with (
        patch("syngen.ml.handlers.handlers.sys.platform", platform),
        patch("syngen.ml.handlers.handlers.cuda_device_count", return_value=gpu_count),
    ):
        assert _select_mp_start_method() == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.handlers.handlers.torch.set_num_threads")
def test_worker_init_pins_gpu_by_pool_worker_ordinal(mock_set_num_threads, rp_logger):
    """Unlike training, the VAE model is built once per worker and reused for
    every task that worker later handles, so GPU assignment must happen once,
    at init time, pinned to mp.current_process()._identity - the stable,
    1-based pool-worker ordinal CPython's multiprocessing.Pool assigns (no
    built-in per-worker ordinal exists otherwise)."""
    rp_logger.info("Test 'worker_init' pins each worker to a GPU index via its pool ordinal")
    get_wrapper_func_from_main = MagicMock()
    gpu_indices = [0, 1, 2]

    with patch("syngen.ml.handlers.handlers.mp.current_process") as mock_current_process:
        mock_current_process.return_value._identity = (2,)  # 1-based -> ordinal 1
        VaeInferHandler.worker_init(
            get_wrapper_func_from_main, threads_per_worker=1, gpu_indices=gpu_indices
        )

    get_wrapper_func_from_main.assert_called_once_with(gpu_index=1)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.handlers.handlers.torch.set_num_threads")
def test_worker_init_skips_gpu_pinning_when_no_gpu_indices(mock_set_num_threads, rp_logger):
    """On a CPU-only machine gpu_indices is falsy (empty list), so worker_init
    must call the wrapper factory unmodified - byte-identical to today."""
    rp_logger.info("Test 'worker_init' skips GPU pinning when gpu_indices is empty")
    get_wrapper_func_from_main = MagicMock()

    VaeInferHandler.worker_init(
        get_wrapper_func_from_main, threads_per_worker=1, gpu_indices=[]
    )

    get_wrapper_func_from_main.assert_called_once_with()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_pool_worker_ordinal_is_stable_1_based():
    """De-risks worker_init's reliance on mp.current_process()._identity: a
    real (non-CUDA) spawn Pool must actually yield stable 1..N ordinals
    across its workers, without needing GPU hardware to exercise it."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=3) as pool:
        ordinals = sorted(pool.map(_get_worker_ordinal, range(3)))
    assert ordinals == [1, 2, 3]


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


# --------------------------------------------------------------------------- #
# EPMCTDM-7631 - long text generation                                          #
# --------------------------------------------------------------------------- #

ALPHABET = OrderedDict((char, index) for index, char in enumerate("abcdefgh", start=1))
CHAR_COUNTS = OrderedDict((char, 10) for char in ALPHABET)


def _build_long_text_features(char_lengths, word_counts, min_word_count, null_share=0.0):
    """
    Build the per-column dict that LongTextsHandler pickles, from an explicit
    (char_len, word_count) sample, mirroring the anti-singularity noise and the
    size-derived bandwidth used at fit time.
    """
    text_structure = np.array([char_lengths, word_counts], dtype="float64")
    noise = np.random.uniform(low=-1e-4, high=1e-4, size=text_structure.shape)
    bw_width = text_structure.shape[1] / text_structure.shape[1] ** 1.3
    return {
        "counts": CHAR_COUNTS,
        "indexes": ALPHABET,
        "kde": gaussian_kde(text_structure + noise, bw_method=bw_width),
        "min_word_count": min_word_count,
        "null_share": null_share,
    }


def _make_infer_handler(features, tmp_path, size):
    """Build a VaeInferHandler wired to a pickled long-text features dict."""
    path_to_no_ml = str(tmp_path / "kde_params.pkl")
    with open(path_to_no_ml, "wb") as file:
        dill.dump(features, file)

    path_to_metadata = f"{DIR_NAME}/unit/handlers/fixtures/metadata.yaml"
    metadata = MetadataLoader(path_to_metadata).load_data()
    with patch.object(VaeInferHandler, "__attrs_post_init__", lambda x: None):
        return VaeInferHandler(
            metadata=metadata,
            table_name="parent_table",
            paths={"path_to_no_ml": path_to_no_ml},
            metadata_path=path_to_metadata,
            random_seed=0,
            size=size,
            batch_size=size,
            run_parallel=False,
            reports=[],
            wrapper_name="MMDVAEWrapper",
            log_level="INFO",
            type_of_process="infer",
            loader=None,
        )


def test_generate_long_texts_no_empty_values_for_degenerate_word_count(
    tmp_path, rp_logger
):
    """
    EPMCTDM-7631 guardrail: a column whose every source text is a single word
    (e.g. a JWT, which contains no whitespace) must not generate empty values.
    Its word count carries no variance beyond the anti-singularity noise, so
    truncating the resampled value collapsed roughly half the rows to zero words.
    """
    rp_logger.info(
        "Test that a source column with a constant word count of 1 "
        "generates no empty values and no NULLs"
    )
    size = 500
    features = {
        "auth_jwt": _build_long_text_features(
            char_lengths=np.random.randint(240, 260, size),
            word_counts=np.ones(size),
            min_word_count=1,
        )
    }
    handler = _make_infer_handler(features, tmp_path, size)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = handler.generate_long_texts(size, pd.DataFrame())

    generated = result["auth_jwt"]
    assert len(generated) == size
    assert generated.isna().sum() == 0
    assert (generated == "").sum() == 0
    assert not [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_long_texts_no_empty_values_for_wide_word_count_tail(
    tmp_path, rp_logger
):
    """
    EPMCTDM-7631 guardrail: with a widely spread word count the resampled value
    can land well below 1, so rounding alone is not enough - the floor is what
    keeps the generated text non-empty.
    """
    rp_logger.info(
        "Test that a source column with a bimodal word count "
        "generates no empty values"
    )
    size = 500
    word_counts = np.concatenate([np.ones(size // 2), np.random.randint(50, 200, size // 2)])
    features = {
        "mixed_text": _build_long_text_features(
            char_lengths=np.random.randint(100, 900, size),
            word_counts=word_counts,
            min_word_count=1,
        )
    }
    handler = _make_infer_handler(features, tmp_path, size)

    result = handler.generate_long_texts(size, pd.DataFrame())

    generated = result["mixed_text"]
    assert (generated == "").sum() == 0
    assert generated.isna().sum() == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_long_texts_reproduces_source_null_share(tmp_path, rp_logger):
    """
    EPMCTDM-7631: NULLs are reproduced from the share measured at fit time and
    are emitted as real NaN, not as an empty string.
    """
    rp_logger.info(
        "Test that the generated long text column reproduces the source NULL share"
    )
    size = 2000
    features = {
        "comment": _build_long_text_features(
            char_lengths=np.random.randint(200, 300, size),
            word_counts=np.random.randint(25, 36, size),
            min_word_count=25,
            null_share=0.2,
        )
    }
    handler = _make_infer_handler(features, tmp_path, size)

    result = handler.generate_long_texts(size, pd.DataFrame())

    generated = result["comment"]
    assert generated.isna().mean() == pytest.approx(0.2, abs=0.05)
    # every value that is not NULL is a real text, never an empty string
    assert (generated.dropna() == "").sum() == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_long_texts_keeps_genuine_empty_values(tmp_path, rp_logger):
    """
    EPMCTDM-7631: a source that genuinely contains empty texts has
    'min_word_count' of 0, so the floor must not suppress empty generated values.
    """
    rp_logger.info(
        "Test that empty values are still generated when the source contains them"
    )
    size = 1000
    empty_rows = size // 4
    word_counts = np.concatenate(
        [np.zeros(empty_rows), np.random.randint(20, 40, size - empty_rows)]
    )
    char_lengths = np.concatenate(
        [np.zeros(empty_rows), np.random.randint(150, 250, size - empty_rows)]
    )
    features = {
        "optional_text": _build_long_text_features(
            char_lengths=char_lengths,
            word_counts=word_counts,
            min_word_count=0,
        )
    }
    handler = _make_infer_handler(features, tmp_path, size)

    result = handler.generate_long_texts(size, pd.DataFrame())

    assert (result["optional_text"] == "").sum() > 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_long_texts_does_not_censor_lower_tail(tmp_path, rp_logger):
    """
    EPMCTDM-7631: the floor is the structural 'at least one word', not the
    minimum word count observed in the source. Flooring a prose column at its
    own minimum would censor the lower tail the KDE is meant to model.
    """
    rp_logger.info(
        "Test that a high minimum word count does not floor the generated word count"
    )
    size = 1000
    features = {
        "review": _build_long_text_features(
            char_lengths=np.random.randint(200, 300, size),
            word_counts=np.random.randint(29, 60, size),
            min_word_count=29,
        )
    }
    handler = _make_infer_handler(features, tmp_path, size)

    result = handler.generate_long_texts(size, pd.DataFrame())

    word_counts = result["review"].apply(lambda text: len(text.split()))
    assert word_counts.min() < 29
    assert (result["review"] == "").sum() == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("word_count", [0, 1, 3])
def test_synth_text_word_count(word_count, tmp_path, rp_logger):
    """
    EPMCTDM-7631: '_synth_text' builds exactly 'word_count' words, and a word
    count of 0 yields an empty text without a divide-by-zero warning.
    """
    rp_logger.info(
        f"Test that '_synth_text' builds {word_count} word(s) without warnings"
    )
    handler = _make_infer_handler({}, tmp_path, size=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = handler._synth_text(60, word_count, ALPHABET, CHAR_COUNTS)

    assert len(text.split()) == word_count
    assert (text == "") is (word_count == 0)
    assert not [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.handlers.handlers.fetch_config")
def test_long_texts_handler_fits_kde_on_non_null_values(
    mock_fetch_config, tmp_path, rp_logger
):
    """
    EPMCTDM-7631: the KDE must be fitted on real texts only. Filling NULLs with
    an empty string skews the modelled text length towards zero and turns NULL
    reproduction into an accident of where the KDE tail lands.
    """
    rp_logger.info(
        "Test that LongTextsHandler fits the KDE on non-null values "
        "and stores the NULL share and the minimum word count"
    )
    dataset = MagicMock()
    dataset.long_text_columns = {"comment"}
    mock_fetch_config.return_value = dataset

    texts = [" ".join(["word"] * 40) for _ in range(80)] + [None] * 20
    data = pd.DataFrame({"comment": texts})

    handler = LongTextsHandler(
        metadata={},
        table_name="parent_table",
        paths={
            "no_ml_state_path": f"{tmp_path}/",
            "dataset_pickle_path": "dataset.pkl",
        },
        schema=None,
    )
    handler.handle(data)

    with open(f"{tmp_path}/kde_params.pkl", "rb") as file:
        features = dill.load(file)

    assert features["comment"]["null_share"] == pytest.approx(0.2)
    assert features["comment"]["min_word_count"] == 40
    # the NULL rows are excluded from the fitted sample
    assert features["comment"]["kde"].dataset.shape[1] == 80
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.handlers.handlers.fetch_config")
def test_long_texts_handler_handles_almost_fully_null_column(
    mock_fetch_config, tmp_path, rp_logger
):
    """
    EPMCTDM-7631: 'gaussian_kde' needs at least two points. A column left with
    fewer than two non-null texts must fall back instead of failing training.
    """
    rp_logger.info(
        "Test that a column with fewer than two non-null values does not raise"
    )
    dataset = MagicMock()
    dataset.long_text_columns = {"comment"}
    mock_fetch_config.return_value = dataset

    data = pd.DataFrame({"comment": ["a text value"] + [None] * 9})

    handler = LongTextsHandler(
        metadata={},
        table_name="parent_table",
        paths={
            "no_ml_state_path": f"{tmp_path}/",
            "dataset_pickle_path": "dataset.pkl",
        },
        schema=None,
    )
    handler.handle(data)

    with open(f"{tmp_path}/kde_params.pkl", "rb") as file:
        features = dill.load(file)

    assert features["comment"]["null_share"] == 0.0
    rp_logger.info(SUCCESSFUL_MESSAGE)
