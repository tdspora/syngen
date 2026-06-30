import pandas as pd
import numpy as np
import pytest
from sdmetrics.single_column import RangeCoverage

from unittest.mock import patch

from syngen.ml.metrics.metrics_classes.metrics import (
    Clustering,
    Utility,
    UnivariateMetric,
    CoverageMetric,
    DetectionMetric,
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


# ---------------------------------------------------------------------------
# CoverageMetric tests
# ---------------------------------------------------------------------------

@pytest.fixture
def coverage_dataframes():
    """Shared original / synthetic DataFrames for CoverageMetric tests."""
    original = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0],
            "salary": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "gender": ["male", "female", "male", "female", "male"],
            "category": ["A", "B", "C", "A", "B"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "age": [25.0, 35.0, 45.0, 55.0],
            "salary": [1500.0, 2500.0, 3500.0, 4500.0],
            "gender": ["male", "female", "male", "male"],
            "category": ["A", "B", "A", "B"],
        }
    )
    return original, synthetic


def test_coverage_metric_numerical_columns(rp_logger, coverage_dataframes):
    """
    CoverageMetric should compute RangeCoverage for numerical columns.
    Synthetic data within [1500, 4500] covers part of [1000, 5000], so score < 1.0.
    """
    rp_logger.info("Testing CoverageMetric with numerical columns only")
    original, synthetic = coverage_dataframes

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(
        cont_columns=["age", "salary"],
        categorical_columns=[],
    )

    assert set(scores.keys()) == {"age", "salary"}
    for col, score in scores.items():
        assert 0.0 <= score <= 1.0, (
            f"RangeCoverage for '{col}' should be in [0, 1], got {score}"
        )
    # age: synthetic range [25, 55] vs original [20, 60]  →  coverage = 1 - 0.125 - 0.125 = 0.75
    assert scores["age"] == pytest.approx(0.75, abs=1e-6)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_symmetrical_score_matches_directional_minimum(rp_logger):
    """
    CoverageMetric score should match min(original->synthetic, synthetic->original).
    """
    rp_logger.info("Testing directional minimum behavior with real RangeCoverage")
    original = pd.DataFrame({"value": [0.0, 10.0, 20.0, np.nan]})
    synthetic = pd.DataFrame({"value": [5.0, 15.0, np.nan]})

    forward = RangeCoverage.compute(original["value"].dropna(), synthetic["value"].dropna())
    reverse = RangeCoverage.compute(synthetic["value"].dropna(), original["value"].dropna())

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(cont_columns=["value"], categorical_columns=[])

    assert forward != reverse
    assert scores["value"] == pytest.approx(min(forward, reverse), abs=1e-9)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_synthetic_range_wider_than_original(rp_logger):
    """
    When synthetic range is wider than original, symmetric score should
    use min(original->synthetic, synthetic->original) and be < 1.
    """
    rp_logger.info("Testing CoverageMetric when synthetic range is wider than original")
    original = pd.DataFrame({"value": [10.0, 15.0, 20.0]})
    synthetic = pd.DataFrame({"value": [0.0, 10.0, 20.0, 30.0]})

    forward = RangeCoverage.compute(original["value"], synthetic["value"])
    reverse = RangeCoverage.compute(synthetic["value"], original["value"])

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(cont_columns=["value"], categorical_columns=[])

    assert forward == pytest.approx(1.0, abs=1e-9)
    assert reverse < 1.0
    assert scores["value"] == pytest.approx(min(forward, reverse), abs=1e-9)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_categorical_columns(rp_logger, coverage_dataframes):
    """
    CoverageMetric should compute CategoryCoverage for categorical columns.
    Synthetic covers only 2 out of 3 categories in 'category', so score = 2/3.
    """
    rp_logger.info("Testing CoverageMetric with categorical columns only")
    original, synthetic = coverage_dataframes

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(
        cont_columns=[],
        categorical_columns=["gender", "category"],
    )

    assert set(scores.keys()) == {"gender", "category"}
    # gender: synthetic contains both 'male' and 'female' → score = 1.0
    assert scores["gender"] == pytest.approx(1.0, abs=1e-6)
    # category: synthetic contains 'A' and 'B' but not 'C' → score = 2/3
    assert scores["category"] == pytest.approx(2 / 3, abs=1e-6)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_mixed_columns(rp_logger, coverage_dataframes):
    """
    CoverageMetric applies RangeCoverage to continuous and CategoryCoverage
    to categorical columns simultaneously.
    """
    rp_logger.info("Testing CoverageMetric with mixed column types")
    original, synthetic = coverage_dataframes

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(
        cont_columns=["age", "salary"],
        categorical_columns=["gender", "category"],
    )

    assert set(scores.keys()) == {"age", "salary", "gender", "category"}
    for col, score in scores.items():
        assert 0.0 <= score <= 1.0, (
            f"Coverage score for '{col}' should be in [0, 1], got {score}"
        )

    mean_value = metric.get_value()
    expected_mean = np.mean([scores["age"], scores["salary"],
                             scores["gender"], scores["category"]])
    assert mean_value == pytest.approx(expected_mean, abs=1e-6)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_get_value_stored(rp_logger, coverage_dataframes):
    """
    After calculate_all(), get_value() must return the mean of all valid scores.
    """
    rp_logger.info("Testing that CoverageMetric.get_value() returns mean coverage")
    original, synthetic = coverage_dataframes

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(
        cont_columns=["salary"],
        categorical_columns=["category"],
    )

    expected_mean = np.mean(list(scores.values()))
    assert metric.get_value() == pytest.approx(expected_mean, abs=1e-6)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_missing_column_logs_warning(rp_logger, coverage_dataframes):
    """
    A column present in the list but absent from the DataFrame should be skipped
    and a warning should be logged.
    """
    rp_logger.info("Testing CoverageMetric warning for missing columns")
    original, synthetic = coverage_dataframes

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    with patch("syngen.ml.metrics.metrics_classes.metrics.logger") as mock_logger:
        scores = metric.calculate_all(
            cont_columns=["nonexistent_num"],
            categorical_columns=["nonexistent_cat"],
        )

    assert scores == {}, "Missing columns should produce an empty scores dict"
    assert mock_logger.warning.call_count == 2

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_zero_variance_column(rp_logger):
    """
    RangeCoverage returns NaN when the original column has zero variance.
    The metric should handle it gracefully and exclude it from the mean.
    """
    rp_logger.info("Testing CoverageMetric with a zero-variance numerical column")
    original = pd.DataFrame({"constant": [5.0, 5.0, 5.0], "value": [1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({"constant": [5.0, 5.0], "value": [1.5, 2.5]})

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(
        cont_columns=["constant", "value"],
        categorical_columns=[],
    )

    assert np.isnan(scores["constant"]), (
        "RangeCoverage for a constant column should be NaN"
    )
    assert not np.isnan(scores["value"]), (
        "RangeCoverage for 'value' column should not be NaN"
    )
    # Mean should be computed only from valid (non-NaN) scores
    assert metric.get_value() == pytest.approx(scores["value"], abs=1e-6)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_coverage_metric_empty_columns(rp_logger, coverage_dataframes):
    """
    Calling calculate_all() with empty column lists should return an empty dict
    and set self.value to NaN.
    """
    rp_logger.info("Testing CoverageMetric with empty column lists")
    original, synthetic = coverage_dataframes

    metric = CoverageMetric(original, synthetic, plot=False, reports_path="")
    scores = metric.calculate_all(cont_columns=[], categorical_columns=[])

    assert scores == {}
    assert np.isnan(metric.get_value())

    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# DetectionMetric tests
# ---------------------------------------------------------------------------

@pytest.fixture
def detection_dataframes():
    """Shared DataFrames for DetectionMetric tests (large enough for 3-fold CV)."""
    rng = np.random.default_rng(42)
    size = 60
    original = pd.DataFrame(
        {
            "num": rng.normal(0, 1, size),
            "cat": rng.choice(["a", "b", "c"], size),
        }
    )
    # Synthetic data drawn from the same distribution → classifier should struggle
    synthetic = pd.DataFrame(
        {
            "num": rng.normal(0, 1, size),
            "cat": rng.choice(["a", "b", "c"], size),
        }
    )
    return original, synthetic


@pytest.mark.parametrize("method", ["logistic", "svc"])
def test_detection_metric_score_range(rp_logger, detection_dataframes, method):
    """
    DetectionMetric score must be in [0.0, 1.0] for both 'logistic' and 'svc'.
    """
    rp_logger.info(f"Testing DetectionMetric score range with method='{method}'")
    original, synthetic = detection_dataframes

    metric = DetectionMetric(original, synthetic, plot=False, reports_path="", method=method)
    score = metric.calculate_all()

    assert 0.0 <= score <= 1.0, (
        f"DetectionMetric ({method}) score should be in [0, 1], got {score}"
    )
    assert metric.get_value() == pytest.approx(score, abs=1e-9)

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_detection_metric_default_method_is_logistic(rp_logger, detection_dataframes):
    """
    When no method is specified, DetectionMetric should default to 'logistic'.
    """
    rp_logger.info("Testing that DetectionMetric defaults to 'logistic'")
    original, synthetic = detection_dataframes

    metric = DetectionMetric(original, synthetic, plot=False, reports_path="")
    assert metric.method == "logistic"

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_detection_metric_invalid_method_raises(rp_logger, detection_dataframes):
    """
    Passing an unknown method name should raise ValueError immediately at construction.
    """
    rp_logger.info("Testing that DetectionMetric raises ValueError for invalid method")
    original, synthetic = detection_dataframes

    with pytest.raises(ValueError, match="Unknown detection method"):
        DetectionMetric(original, synthetic, plot=False, reports_path="", method="unknown")

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_detection_metric_get_value_before_calculate(rp_logger, detection_dataframes):
    """
    Before calculate_all() is called, get_value() should return None
    (inherited BaseMetric default).
    """
    rp_logger.info("Testing DetectionMetric.get_value() before calculate_all()")
    original, synthetic = detection_dataframes

    metric = DetectionMetric(original, synthetic, plot=False, reports_path="")
    assert metric.get_value() is None

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_detection_metric_indistinguishable_data(rp_logger):
    """
    When synthetic data is identical to real data the classifier cannot distinguish them,
    so the score should be close to 1.0.
    """
    rp_logger.info("Testing DetectionMetric with identical real and synthetic data")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(5, 2, 100)})

    metric = DetectionMetric(df, df.copy(), plot=False, reports_path="", method="logistic")
    score = metric.calculate_all()

    assert score >= 0.5, (
        f"Score for identical datasets should be >= 0.5, got {score}"
    )

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_detection_metric_clearly_different_data(rp_logger):
    """
    When synthetic data comes from a very different distribution the classifier
    easily separates them → score should be close to 0.0.
    """
    rp_logger.info("Testing DetectionMetric with clearly different distributions")
    rng = np.random.default_rng(1)
    original = pd.DataFrame({"x": rng.normal(0, 0.1, 100)})
    synthetic = pd.DataFrame({"x": rng.normal(1000, 0.1, 100)})

    metric = DetectionMetric(original, synthetic, plot=False, reports_path="", method="logistic")
    score = metric.calculate_all()

    assert score < 0.5, (
        f"Score for clearly different datasets should be < 0.5, got {score}"
    )

    rp_logger.info(SUCCESSFUL_MESSAGE)

