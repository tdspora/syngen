import os
import random
import subprocess
import sys
import textwrap

import pandas as pd
import numpy as np
import pytest

from unittest.mock import patch

from syngen.ml.metrics.metrics_classes.metrics import (
    BivariateMetric,
    Clustering,
    Utility,
    UnivariateMetric,
    encode_categories,
    MAX_ROWS_FOR_CLUSTER_SEARCH,
    MIN_NUMBER_OF_ROWS_FOR_CLUSTERING,
)

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_clustering_calculate_all(rp_logger):
    """
    Testing the clustering metric score
    """
    rp_logger.info(
        "Testing the clustering metric score"
    )

    path_to_original = f"{DIR_NAME}/unit/metrics/fixtures/clusters-2.csv"
    path_to_synthetic = f"{DIR_NAME}/unit/metrics/fixtures/synthetic_clusters-2_10k.csv"

    threshold = 0.95

    original = pd.read_csv(path_to_original)
    synthetic = pd.read_csv(path_to_synthetic)

    clustering = Clustering(original, synthetic, plot=False, reports_path="")

    categ_columns = []
    cont_columns = original.columns.to_list()

    mean_score = clustering.calculate_all(categ_columns, cont_columns)

    assert mean_score >= threshold, f"Mean score shouldn't be less than {threshold}"

    rp_logger.info(SUCCESSFUL_MESSAGE)


def _make_clustered_frame(n, seed, n_groups=3):
    """
    A frame with `n_groups` well-separated blobs, so clustering has genuine
    structure to find and a "perfect generator" (a resample of this frame)
    is expected to score close to 1.
    """
    rng = np.random.RandomState(seed)
    group = rng.randint(0, n_groups, size=n)
    centers = np.arange(n_groups) * 10
    x = centers[group] + rng.normal(scale=0.5, size=n)
    y = centers[group] + rng.normal(scale=0.5, size=n)
    cat = pd.Series(group).map({i: f"region_{i}" for i in range(n_groups)})
    return pd.DataFrame({"x": x, "y": y, "cat": cat})


def test_encode_categories_is_order_independent(rp_logger):
    """
    'encode_categories' must not depend on the iteration order of its inputs:
    this is the fix for EPMCTDM-7127, where 'enumerate(set(...))' produced a
    different mapping (and therefore a different clustering score) on every
    process run because Python randomizes string hashing per process.
    """
    rp_logger.info("Testing 'encode_categories' is deterministic and order independent")

    original = pd.Series(["b", "a", "c", "a"])
    synthetic = pd.Series(["c", "d"])

    forward = encode_categories(original, synthetic)
    backward = encode_categories(original.iloc[::-1], synthetic.iloc[::-1])

    assert forward == backward, "mapping must not depend on row order"
    assert set(forward.keys()) == {"a", "b", "c", "d"}, \
        "the union must cover categories present in only one of the two frames"

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_encode_categories_handles_mixed_types_and_nan(rp_logger):
    """
    Columns coming through the reporter can mix strings, ints and NaN
    (e.g. after upstream preprocessing); a bare 'sorted(set(...))' would
    raise on that mix, so 'encode_categories' sorts by the string form.
    """
    rp_logger.info("Testing 'encode_categories' tolerates mixed types and NaN")

    original = pd.Series(["a", 1, np.nan])
    synthetic = pd.Series(["b", 2])

    mapping = encode_categories(original, synthetic)

    assert len(mapping) == 5
    assert set(mapping.values()) == {1, 2, 3, 4, 5}

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_unequal_row_counts_does_not_collapse_score(rp_logger):
    """
    Regression test for the 'level_1' leak (EPMCTDM-7127): the merged frame's
    reset_index() used to leave the row index in as a numeric feature, and
    because the smaller of the two frames is sampled in full while the larger
    contributes a random subset of its index range, that axis silently leaked
    which dataset a row came from. A perfect generator with a different row
    count than the original used to score ~0.4 instead of ~1.0.
    """
    rp_logger.info(
        "Testing the clustering score stays high when original and synthetic "
        "have different row counts"
    )

    original = _make_clustered_frame(1000, seed=1)
    synthetic = _make_clustered_frame(3000, seed=2)

    clustering = Clustering(
        original.copy(), synthetic.copy(), plot=False, reports_path=""
    )
    mean_score = clustering.calculate_all(["cat"], ["x", "y"])

    assert mean_score >= 0.9, (
        f"Mean score shouldn't collapse when row counts differ, got {mean_score}"
    )
    assert "level_0" not in clustering.feature_columns
    assert "level_1" not in clustering.feature_columns

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_uses_one_scaler_for_both_spaces(rp_logger):
    """
    The number of clusters must be chosen in the same geometry it is applied
    to: a single 'MinMaxScaler', fit once on the merged data, has to serve
    both the k search (on the original data) and the measurement (on the
    merged data).
    """
    rp_logger.info("Testing clustering selects k using the scaler fit on merged data")

    original = _make_clustered_frame(500, seed=3)
    synthetic = _make_clustered_frame(500, seed=4)

    clustering = Clustering(
        original.copy(), synthetic.copy(), plot=False, reports_path=""
    )
    clustering.calculate_all(["cat"], ["x", "y"])

    assert clustering.scaler.n_features_in_ == len(clustering.feature_columns)
    # the same scaler instance must be able to transform the original data
    # (already categorical-encoded in place by calculate_all) without
    # refitting - this is what keeps k and the measurement consistent
    reconstructed = clustering.scaler.transform(
        clustering.original[clustering.feature_columns].dropna()
    )
    assert reconstructed.shape[1] == len(clustering.feature_columns)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_k_search_is_capped(rp_logger):
    """
    The k search must not run on the full original dataset above
    'MAX_ROWS_FOR_CLUSTER_SEARCH' rows; the measurement itself is not capped.
    """
    rp_logger.info("Testing the k-search row cap is applied")

    n_original = MAX_ROWS_FOR_CLUSTER_SEARCH + 500
    original = _make_clustered_frame(n_original, seed=5)
    # kept small so the final KMeans fit (on the merged data) stays fast;
    # only the k-search cap on the *original* data is under test here
    synthetic = _make_clustered_frame(60, seed=6)

    clustering = Clustering(
        original.copy(), synthetic.copy(), plot=False, reports_path=""
    )

    seen_sizes = []
    original_method = Clustering._Clustering__get_optimal_number_of_clusters

    def spy(self, dataset):
        seen_sizes.append(len(dataset))
        return original_method(self, dataset)

    with patch.object(
        Clustering, "_Clustering__get_optimal_number_of_clusters", spy
    ):
        clustering.calculate_all(["cat"], ["x", "y"])

    assert seen_sizes == [MAX_ROWS_FOR_CLUSTER_SEARCH], (
        "the k search must see at most MAX_ROWS_FOR_CLUSTER_SEARCH rows, "
        f"got {seen_sizes}"
    )

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_edge_cases_do_not_raise(rp_logger):
    """
    None of these inputs are expected to produce a clustering score, but none
    of them should raise either.
    """
    rp_logger.info("Testing clustering edge cases return None without raising")

    # exactly MIN_NUMBER_OF_ROWS_FOR_CLUSTERING rows on each side must still work
    minimal = _make_clustered_frame(MIN_NUMBER_OF_ROWS_FOR_CLUSTERING, seed=7)
    clustering = Clustering(minimal.copy(), minimal.copy(), plot=False, reports_path="")
    clustering.calculate_all(["cat"], ["x", "y"])  # must not raise

    # a column declared as continuous that cannot be coerced to numeric
    # must be silently dropped from the feature set, not raise
    original = _make_clustered_frame(50, seed=8)
    synthetic = _make_clustered_frame(50, seed=9)
    original["not_numeric"] = "text"
    synthetic["not_numeric"] = "text"
    clustering = Clustering(original.copy(), synthetic.copy(), plot=False, reports_path="")
    clustering.calculate_all(["cat"], ["x", "y", "not_numeric"])
    assert "not_numeric" not in clustering.feature_columns

    # all-NaN continuous column must return None with a warning, not raise
    original = _make_clustered_frame(50, seed=10)
    synthetic = _make_clustered_frame(50, seed=11)
    original["all_nan"] = np.nan
    synthetic["all_nan"] = np.nan
    clustering = Clustering(original.copy(), synthetic.copy(), plot=False, reports_path="")
    with patch(
        "syngen.ml.metrics.metrics_classes.metrics.logger"
    ) as mock_logger:
        result = clustering.calculate_all(["cat"], ["x", "y", "all_nan"])
    assert result is None
    mock_logger.warning.assert_called_once()

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metrics_are_deterministic_across_hash_seeds(rp_logger):
    """
    EPMCTDM-7127: every categorical->integer map in this file used to be built
    as 'enumerate(set(...))' over strings, and Python randomizes string
    hashing per process, so the codes - and therefore the clustering score -
    changed on every run. This is the test that would have caught it: run the
    clustering metric in fresh subprocesses under different PYTHONHASHSEED
    values and assert identical results.
    """
    rp_logger.info(
        "Testing the clustering score is stable across PYTHONHASHSEED values"
    )

    script = textwrap.dedent(
        """
        import numpy as np
        import pandas as pd

        from syngen.ml.metrics.metrics_classes.metrics import Clustering

        rng = np.random.RandomState(0)
        n = 300
        group = rng.randint(0, 3, size=n)
        centers = np.arange(3) * 10
        x = centers[group] + rng.normal(scale=0.5, size=n)
        y = centers[group] + rng.normal(scale=0.5, size=n)
        cat = pd.Series(group).map(
            {0: "region_a", 1: "region_b", 2: "region_c"}
        )
        original = pd.DataFrame({"x": x, "y": y, "cat": cat})
        synthetic = original.sample(frac=1, random_state=1).reset_index(drop=True)

        clustering = Clustering(
            original.copy(), synthetic.copy(), plot=False, reports_path=""
        )
        score = clustering.calculate_all(["cat"], ["x", "y"])
        print(score)
        """
    )

    scores = []
    for seed in range(5):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        scores.append(float(completed.stdout.strip().splitlines()[-1]))

    assert len(set(scores)) == 1, (
        f"clustering score must not depend on PYTHONHASHSEED, got {scores}"
    )

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "col, model_y, task_type, expected_result, expected_log_message",
    [
        ("column_1", np.array([1, 1, 1, 1]), "binary classification", False, True),
        ("column_2", np.array([0, 0, 0, 1]), "binary classification", False, True),
        ("column_3", np.array([0, 1, 0, 1]), "binary classification", True, False),
        ("column_4", np.array([0, 0, 1, 1, 2, 2]), "multiclass classification", True, False),
        ("column_5", np.array([0, 0, 0, 1, 2, 2]), "multiclass classification", False, True),
        ("column_6", np.array([1, 1, 1]), "multiclass classification", False, True),
        ("column_7", np.array([1, 1, 1, 1]), "regression", False, True),
        ("column_8", np.array([1, 2, 3, 4, 4]), "regression", True, False),
    ]
)
def test_utility_valid_target(
        rp_logger, col, model_y, task_type,
        expected_result, expected_log_message):
    """
    Testing the _valid_target function in the Utility class
    """
    rp_logger.info(
        "Testing the _valid_target function in the Utility class"
    )

    with patch(
        'syngen.ml.metrics.metrics_classes.metrics.logger'
    ) as mock_logger:
        result = Utility._valid_target(col, model_y, task_type)

    assert result == expected_result, \
        f"Expected result is {expected_result}, got {result}"

    if expected_log_message:
        mock_logger.info.assert_called_once()
    else:
        mock_logger.info.assert_not_called()

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "original_val, synthetic_val, expected_result",
    [
        (0, 0, 1.0),         # Both zero -> perfect match
        (0, 5, 5e10),        # Original zero -> synthetic/epsilon
        (5, 0, 0),          # Synthetic zero -> 0
        (1e-11, 1e-11, 1.0),  # Both near zero -> perfect match
        (10, 20, 2.0),      # Regular case
    ]
)
def test_calculate_ratio(
    rp_logger, original_val, synthetic_val, expected_result
):
    """Test _calculate_ratio method of UnivariateMetric class"""
    rp_logger.info(
        f"Testing _calculate_ratio with original_val={original_val}, "
        f"synthetic_val={synthetic_val}, expected_result={expected_result}"
    )
    epsilon = 1e-10
    metric = UnivariateMetric(
        pd.DataFrame(), pd.DataFrame(), plot=False, reports_path=""
    )

    ratio = metric._calculate_ratio(original_val, synthetic_val, epsilon)

    assert ratio == expected_result, f"Expected {expected_result}, got {ratio}"

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_categorical_tick_selection_is_deterministic(rp_logger):
    """EPMCTDM-7630: which categorical ticks the bivariate heatmap labels must not
    depend on the global RNG.

    `_get_categorical_ticks` keeps the first and last ticks and samples the middle.
    Drawing that middle slice from the module-level `random` meant the same data
    produced differently-labelled axes on every run.

    Driven through a stub instance because `BivariateMetric.__init__` requires a
    pickled dataset config, which this method does not touch.
    """
    rp_logger.info("Test 'BivariateMetric._get_categorical_ticks' is deterministic")
    values = [f"cat_{i:03d}" for i in range(80)]
    frame = pd.DataFrame({"c": values})

    stub = object.__new__(BivariateMetric)
    stub.original = frame
    stub.synthetic = frame

    selections = []
    for seed in (1, 7919, 104729):
        random.seed(seed)          # perturb the global RNG between calls
        selections.append(tuple(stub._get_categorical_ticks("c")))

    assert len(set(selections)) == 1, (
        "tick selection must be stable regardless of global RNG state"
    )
    # seeding must not change the shape of the selection
    ticks = selections[0]
    assert values[0] in ticks and values[-1] in ticks, "extremes must be kept"
    assert len(ticks) == 50, "default ticks_count is unchanged"
    rp_logger.info(SUCCESSFUL_MESSAGE)
