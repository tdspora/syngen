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
    # the merged frame carries the clustering features and nothing else - the
    # dataset label lives in 'origin', so no index column can become a feature
    assert clustering.feature_columns == list(clustering.merged.columns)
    assert set(clustering.feature_columns) == {"x", "y", "cat"}

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_uses_one_scaler_fit_on_the_original_data(rp_logger):
    """
    The number of clusters must be chosen in the same geometry it is applied
    to, so one 'MinMaxScaler' has to serve both the k search (on the original
    data) and the measurement (on the merged data). That scaler is fit on the
    original data alone, so the range of the synthetic data cannot move the
    space in which k is chosen.
    """
    rp_logger.info("Testing clustering selects k using the scaler fit on original data")

    original = _make_clustered_frame(500, seed=3)
    synthetic = _make_clustered_frame(500, seed=4)

    clustering = Clustering(
        original.copy(), synthetic.copy(), plot=False, reports_path=""
    )
    clustering.calculate_all(["cat"], ["x", "y"])

    assert clustering.scaler.n_features_in_ == len(clustering.feature_columns)
    # the scaler's bounds must come from the original data only - 'calculate_all'
    # encodes the categoricals in place, so 'clustering.original' is the encoded
    # frame the scaler was fit on
    expected = clustering.original[clustering.feature_columns].dropna()
    assert clustering.scaler.data_min_ == pytest.approx(expected.min().values)
    assert clustering.scaler.data_max_ == pytest.approx(expected.max().values)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_scaler_is_not_moved_by_synthetic_outliers(rp_logger):
    """
    A 'MinMaxScaler' is defined by a single min and max per column, so fitting
    it on the merged data would let one extreme synthetic row rescale every
    other row and shift the chosen number of clusters. Fitting on the original
    data alone keeps both the scaler and k independent of the synthetic data.
    """
    rp_logger.info("Testing synthetic outliers do not move the clustering scaler")

    original = _make_clustered_frame(500, seed=12)
    synthetic = _make_clustered_frame(500, seed=13)
    outlying = synthetic.copy()
    outlying.loc[0, "x"] = 1e6

    clean_run = Clustering(original.copy(), synthetic.copy(), plot=False, reports_path="")
    clean_run.calculate_all(["cat"], ["x", "y"])

    outlier_run = Clustering(original.copy(), outlying.copy(), plot=False, reports_path="")
    outlier_run.calculate_all(["cat"], ["x", "y"])

    assert outlier_run.scaler.data_max_ == pytest.approx(clean_run.scaler.data_max_), (
        "the scaler's bounds must come from the original data, "
        "so a synthetic outlier cannot move them"
    )
    # the outlier still lands where it belongs: outside the original [0, 1] range
    assert outlier_run.merged_transformed.max() > 1

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_warns_when_a_column_is_numeric_in_only_one_dataset(rp_logger):
    """
    A column that coerces to numeric in one dataset but not the other cannot be
    clustered on, so it is excluded from the feature set - but that is a data
    quality signal rather than a routine event, so it must not be dropped
    silently.
    """
    rp_logger.info(
        "Testing clustering warns about the columns dropped from the feature set"
    )

    original = _make_clustered_frame(50, seed=18)
    synthetic = _make_clustered_frame(50, seed=19)
    original["half_numeric"] = 1.0
    synthetic["half_numeric"] = "text"

    clustering = Clustering(
        original.copy(), synthetic.copy(), plot=False, reports_path=""
    )
    with patch(
        "syngen.ml.metrics.metrics_classes.metrics.logger"
    ) as mock_logger:
        clustering.calculate_all(["cat"], ["x", "y", "half_numeric"])

    assert "half_numeric" not in clustering.feature_columns
    messages = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert any("half_numeric" in message for message in messages), (
        f"the excluded column must be named in a warning, got {messages}"
    )

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_unparseable_value_costs_a_row_not_a_column(rp_logger):
    """
    EPMCTDM-7127: a value that cannot be parsed as a number must cost the single
    row holding it, not the whole column. Dropping the column would remove a
    dimension from the joint feature space - and with it any discrepancy living
    along that axis, which biases the score upwards, the wrong direction for a
    quality metric.
    """
    rp_logger.info(
        "Testing an unparseable value drops its row rather than its column"
    )

    original = _make_clustered_frame(500, seed=20)
    # a generator that reproduces 'y' and 'cat' but gets 'x' badly wrong
    synthetic = _make_clustered_frame(500, seed=21)
    synthetic["x"] = synthetic["x"] + 200

    poisoned = synthetic.copy()
    poisoned["x"] = poisoned["x"].astype(object)
    poisoned.loc[poisoned.index[0], "x"] = "corrupted"

    clean_run = Clustering(original.copy(), synthetic.copy(), plot=False, reports_path="")
    clean_score = clean_run.calculate_all(["cat"], ["x", "y"])

    poisoned_run = Clustering(original.copy(), poisoned.copy(), plot=False, reports_path="")
    poisoned_score = poisoned_run.calculate_all(["cat"], ["x", "y"])

    assert "x" in poisoned_run.feature_columns, \
        "the column must survive a single unparseable value"
    # the offending row is gone, and the merged frame stays balanced - losing one
    # synthetic row costs the original one too, because 'row_limit' re-balances
    counts = poisoned_run.merged.index.get_level_values("origin").value_counts()
    assert counts["original"] == counts["synthetic"]
    assert len(poisoned_run.merged) == len(clean_run.merged) - 2
    # and the defect in 'x' is still detected, instead of being hidden by the
    # column's removal
    assert clean_score < 0.1, f"the shifted 'x' must be detected, got {clean_score}"
    assert poisoned_score == pytest.approx(clean_score, abs=0.05), (
        "one unparseable value must not change the score materially; "
        f"clean={clean_score} poisoned={poisoned_score}"
    )

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_merged_frame_stays_balanced(rp_logger):
    """
    EPMCTDM-7127: 'calculate_diversity' compares raw counts, so if one frame
    contributes fewer rows to the merged frame than the other, the score drops
    on that alone. Nulls must therefore be dropped from each frame BEFORE
    'row_limit' samples them - dropping after the concat used to score a perfect
    generator at ~0.50 when half its rows carried an unparseable value.
    """
    rp_logger.info("Testing the merged frame stays balanced when one side loses rows")

    original = _make_clustered_frame(2000, seed=22)
    synthetic = _make_clustered_frame(2000, seed=23)

    # a perfect generator whose first half carries an unparseable 'x'
    poisoned = synthetic.copy()
    poisoned["x"] = poisoned["x"].astype(object)
    poisoned.iloc[:1000, poisoned.columns.get_loc("x")] = "junk"

    # matched control: the very same surviving rows, without the junk ones present.
    # Comparing against the full 2000-row frame instead would confound the balance
    # effect with the sampling noise floor, which grows as the row count falls.
    control = synthetic.iloc[1000:]

    poisoned_run = Clustering(original.copy(), poisoned.copy(), plot=False, reports_path="")
    poisoned_score = poisoned_run.calculate_all(["cat"], ["x", "y"])

    control_run = Clustering(original.copy(), control.copy(), plot=False, reports_path="")
    control_score = control_run.calculate_all(["cat"], ["x", "y"])

    counts = poisoned_run.merged.index.get_level_values("origin").value_counts()
    assert counts["original"] == counts["synthetic"] == 1000, (
        f"both frames must contribute equally, got {dict(counts)}"
    )
    assert poisoned_score > 0.9, (
        "a perfect generator must not be penalised for losing rows on one side "
        f"(this scored ~0.50 before the fix), got {poisoned_score}"
    )
    assert poisoned_score == pytest.approx(control_score, abs=0.02), (
        "dropping the junk rows must match feeding the surviving rows directly; "
        f"poisoned={poisoned_score} control={control_score}"
    )

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_clustering_requires_enough_rows_after_dropping_nulls(rp_logger):
    """
    The row guard counts usable rows, not raw ones. A frame can be large and still
    have almost nothing to cluster on - it used to pass on its raw length and then
    contribute a handful of rows against the other frame's thousands.
    """
    rp_logger.info("Testing the row guard counts rows left after dropping nulls")

    original = _make_clustered_frame(2000, seed=24)
    synthetic = _make_clustered_frame(2000, seed=25)
    synthetic.loc[synthetic.index[:1998], "x"] = np.nan   # 2 usable rows remain

    clustering = Clustering(
        original.copy(), synthetic.copy(), plot=False, reports_path=""
    )
    with patch(
        "syngen.ml.metrics.metrics_classes.metrics.logger"
    ) as mock_logger:
        result = clustering.calculate_all(["cat"], ["x", "y"])

    assert result is None
    message = mock_logger.warning.call_args_list[-1].args[0]
    assert "2 rows in the synthetic dataset after dropping null values" in message, (
        f"the warning must report the usable row count, got: {message}"
    )

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

    # no usable feature at all must return None with a warning, not raise
    # inside the scaler
    original = _make_clustered_frame(50, seed=16)
    synthetic = _make_clustered_frame(50, seed=17)
    clustering = Clustering(original.copy(), synthetic.copy(), plot=False, reports_path="")
    with patch(
        "syngen.ml.metrics.metrics_classes.metrics.logger"
    ) as mock_logger:
        result = clustering.calculate_all([], [])
    assert result is None
    mock_logger.warning.assert_called_once()

    # an all-NaN continuous column holds no parseable value, so it is excluded and
    # the metric is still computed on the remaining columns - rather than its NaNs
    # wiping out every row
    original = _make_clustered_frame(50, seed=10)
    synthetic = _make_clustered_frame(50, seed=11)
    original["all_nan"] = np.nan
    synthetic["all_nan"] = np.nan
    clustering = Clustering(original.copy(), synthetic.copy(), plot=False, reports_path="")
    with patch(
        "syngen.ml.metrics.metrics_classes.metrics.logger"
    ) as mock_logger:
        result = clustering.calculate_all(["cat"], ["x", "y", "all_nan"])
    assert result is not None
    assert "all_nan" not in clustering.feature_columns
    mock_logger.warning.assert_called_once()

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metrics_are_deterministic_across_hash_seeds(rp_logger):
    """
    EPMCTDM-7127: every categorical->integer map used to be built as
    'enumerate(set(...))' over strings, and Python randomizes string hashing per
    process, so the codes - and therefore the clustering score - changed on every
    run. 'PYTHONHASHSEED' is fixed at interpreter start-up, so no in-process test
    can see this; only fresh subprocesses can.

    The fixture has to be chosen with care. An earlier version used three
    categories and a synthetic frame that was a shuffle of the original - it
    scored 1.0 under every seed even with the bug present, because both frames
    share one mapping, so any numbering leaves them identical and perfectly mixed.
    This fixture instead uses many categories and breaks the category-to-cluster
    association in the synthetic frame, which makes the score depend on the codes:
    measured against the pre-fix encoder it swings 0.62 - 0.86 across seeds.
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
        n = 400
        n_cat = 12
        group = rng.randint(0, 3, size=n)
        centers = np.arange(3) * 10

        # the category carries the cluster in the ORIGINAL...
        original = pd.DataFrame({
            "x": centers[group] + rng.normal(scale=0.5, size=n),
            "y": centers[group] + rng.normal(scale=0.5, size=n),
            "cat": [f"cat_{(g * 4 + rng.randint(0, 4)):02d}" for g in group],
        })
        # ...but the synthetic breaks that association, so the ordinal codes
        # actually move rows relative to each other
        synth_group = rng.randint(0, 3, size=n)
        synthetic = pd.DataFrame({
            "x": centers[synth_group] + rng.normal(scale=0.5, size=n),
            "y": centers[synth_group] + rng.normal(scale=0.5, size=n),
            "cat": [f"cat_{rng.randint(0, n_cat):02d}" for _ in range(n)],
        })

        clustering = Clustering(
            original.copy(), synthetic.copy(), plot=False, reports_path=""
        )
        print(clustering.calculate_all(["cat"], ["x", "y"]))
        """
    )

    scores = []
    for seed in range(3):
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
