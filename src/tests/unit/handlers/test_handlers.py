from unittest.mock import patch
import pytest
import math

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
@patch.object(VaeInferHandler, "__attrs_post_init__")
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
        reports=[],
        wrapper_name="MMDVAEWrapper",
        log_level="INFO",
        type_of_process="infer",
        loader=None
        )
    handler.batch_num = math.ceil(handler.size / handler.batch_size)
    assert handler.split_by_batches() == expected_result
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
