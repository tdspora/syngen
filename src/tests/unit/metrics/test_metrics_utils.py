import random

import pandas as pd
import numpy as np
import pytest

from syngen.ml.metrics.utils import (
    encode_categories,
    encode_categorical_features,
    get_outlier_ratio_iqr,
    sanitize_labels,
    get_ratio_counts,
    plot_dist
)
from tests.conftest import SUCCESSFUL_MESSAGE


@pytest.fixture
def normal_distribution_data():
    """Fixture providing normal distribution data"""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 1, 1000))


@pytest.fixture
def data_with_outliers():
    """Fixture providing data with clear outliers"""
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200])


@pytest.fixture
def uniform_data():
    """Fixture providing uniform data"""
    return pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])


@pytest.fixture
def mixed_data_with_negatives():
    """Fixture providing data with negative and positive values"""
    return pd.Series([-100, -10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 100])


class TestGetOutlierRatioIqr:
    """Test suite for 'get_outlier_ratio_iqr' function"""

    def test_normal_distribution_no_outliers(self, normal_distribution_data):
        """Test with normal distribution data that has few outliers"""
        result = get_outlier_ratio_iqr(normal_distribution_data)
        assert 0.0 <= result <= 0.05

    def test_data_with_clear_outliers(self, data_with_outliers):
        """Test with data containing clear outliers"""
        result = get_outlier_ratio_iqr(data_with_outliers)
        expected_ratio = 2 / len(data_with_outliers)
        assert abs(result - expected_ratio) < 1e-10

    def test_uniform_data_no_outliers(self, uniform_data):
        """Test with uniform data that has no outliers"""
        result = get_outlier_ratio_iqr(uniform_data)

        # Uniform data should have no outliers
        assert result == 0.0

    def test_single_value_no_outliers(self):
        """Test with series containing only one value"""
        data = pd.Series([42])
        result = get_outlier_ratio_iqr(data)

        # Single value cannot be an outlier
        assert result == 0.0

    def test_empty_series_returns_nan(self):
        """Test with empty series"""
        data = pd.Series([], dtype=float)
        result = get_outlier_ratio_iqr(data)

        # Empty series should return NaN
        assert pd.isna(result)

    @pytest.mark.parametrize("factor, expected_result", [
        (1.5, 2 / 12),
        (3.0, 2 / 12),
        (5.0, 2 / 12),
        (10.0, 2 / 12),
        (20.0, 1 / 12),
    ])
    def test_different_factors(self, data_with_outliers, factor, expected_result):
        """Test with different factor values"""
        result = get_outlier_ratio_iqr(data_with_outliers, factor=factor)

        assert abs(result - expected_result) < 1e-10

    def test_factor_sensitivity(self):
        """Test that smaller factors detect more outliers"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])

        result_small_factor = get_outlier_ratio_iqr(data, factor=0.5)
        result_large_factor = get_outlier_ratio_iqr(data, factor=3.0)

        # Smaller factor should detect more or equal outliers
        assert result_small_factor >= result_large_factor

    def test_negative_values(self, mixed_data_with_negatives):
        """Test with negative values"""

        result = get_outlier_ratio_iqr(mixed_data_with_negatives)
        expected_ratio = 2 / len(mixed_data_with_negatives)

        assert abs(result - expected_ratio) < 1e-10

    def test_large_dataset(self):
        """Test with large dataset"""
        np.random.seed(42)
        # Create large dataset with some outliers
        normal_data = np.random.normal(50, 10, 10000)
        outliers = [0, 0, 200, 200, 300]  # Add some clear outliers
        data = pd.Series(list(normal_data) + outliers)

        result = get_outlier_ratio_iqr(data)

        # Should detect the added outliers but ratio should be small
        assert 0.0 < result < 0.05

    def test_zero_iqr_edge_case(self):
        """Test edge case where IQR is zero (Q1 = Q3)"""
        # Create data where Q1 = Q3 (most values are the same)
        data = pd.Series([5, 5, 5, 5, 5, 5, 5, 10])  # 7 fives, 1 ten

        result = get_outlier_ratio_iqr(data)

        # When IQR is 0, any different value should be considered an outlier
        assert result == 1 / len(data)

    def test_symmetric_outliers(self):
        """Test with symmetric outliers on both ends"""
        data = pd.Series([-100, 1, 2, 3, 4, 5, 6, 7, 8, 100])
        result = get_outlier_ratio_iqr(data)
        assert result == 2 / len(data)


@pytest.mark.parametrize("label, expected_result", [
    ("$100", "100"),
    ("Price: $50", "Price: 50"),
    ("$", ""),
    ("x^2", "x2"),
    ("y = a^b + c", "y = ab + c"),
    ("^", ""),
    ("$100^2", "1002"),
    ("Cost: $50^discount", "Cost: 50discount"),
    ("$^$^", ""),
    ("$$100$$", "100"),
    ("x^^2^^", "x2"),
    ("$100$ + $200$", "100 + 200"),
    ("a^b^c^d", "abcd"),
    ("normal_label", "normal_label"),
    ("123", "123"),
    ("test label with spaces", "test label with spaces"),
    ("special@#%&*()", "special@#%&*()"),
    ("", "")
])
def test_sanitize_labels(label, expected_result):
    """Test sanitize_labels function"""
    assert sanitize_labels(label) == expected_result


class TestGetRatioCounts:
    """Test suite for 'get_ratio_counts' function"""

    def test_basic_functionality(self):
        """Test basic functionality with default count"""
        # Create test data with known frequencies
        data = ["a"] * 10 + ["b"] * 8 + ["c"] * 6 + ["d"] * 4 + ["e"] * 2
        result = get_ratio_counts(data)

        # Should return a dictionary
        assert isinstance(result, dict)
        # Should contain the most and least common items
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" in result
        assert "e" in result
        assert result == {"a": 10, "b": 8, "c": 6, "d": 4, "e": 2}

    def test_small_dataset(self):
        """Test with dataset smaller than requested count"""
        data = ["a", "b", "c", "d", "e"]
        result = get_ratio_counts(data)
        assert len(result) == 5
        assert set(result.keys()) == {"a", "b", "c", "d", "e"}

    def test_empty_data(self):
        """Test with empty data"""
        data = []
        result = get_ratio_counts(data)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_single_item_repeated(self):
        """Test with single item repeated multiple times"""
        data = ["a"] * 100
        result = get_ratio_counts(data)

        assert len(result) == 1
        assert "a" in result
        assert result["a"] == 100


class TestPlotDist:
    """Test suite for 'plot_dist' function"""

    def test_basic_functionality_without_full_set(self):
        """Test basic functionality without full_set parameter"""
        data = ["a", "a", "b", "b", "b", "c"]  # a:2, b:3, c:1
        result = plot_dist(data, sort=False, full_set=None)

        assert result is None

    def test_basic_functionality_with_full_set_no_sort(self):
        """Test basic functionality with full_set parameter, no sorting"""
        data = ["a", "a", "b", "b", "b", "c"]  # a:2, b:3, c:1, total=6
        full_set = {"a", "b", "c", "d"}  # d is missing from data
        result = plot_dist(data, sort=False, full_set=full_set)

        # Should return dictionary with percentages
        assert isinstance(result, dict)
        assert abs(result["a"] - (2/6 * 100)) < 1e-10  # 33.33%
        assert abs(result["b"] - (3/6 * 100)) < 1e-10  # 50%
        assert abs(result["c"] - (1/6 * 100)) < 1e-10  # 16.67%
        assert result["d"] == 0.0  # Missing key gets 0

    def test_with_full_set_and_sorting(self):
        """Test with full_set parameter and sorting enabled"""
        data = ["a", "a", "b", "b", "b", "c"]  # a:2, b:3, c:1
        full_set = {"a", "b", "c", "d"}
        result = plot_dist(data, sort=True, full_set=full_set)

        # Should return dictionary with percentages, sorted by count
        assert isinstance(result, dict)

        # Should be sorted by count (ascending): d(0), c(1), a(2), b(3)
        expected_order = ["d", "c", "a", "b"]
        assert list(result.keys()) == expected_order

        # Check percentages
        assert result["d"] == 0.0
        assert abs(result["c"] - (1/6 * 100)) < 1e-10
        assert abs(result["a"] - (2/6 * 100)) < 1e-10
        assert abs(result["b"] - (3/6 * 100)) < 1e-10

    def test_single_value_repeated(self):
        """Test with single value repeated multiple times"""
        data = ["x"] * 5
        full_set = {"x", "y", "z"}
        result = plot_dist(data, sort=True, full_set=full_set)

        assert isinstance(result, dict)
        assert result["x"] == 100.0  # 5/5 * 100 = 100%
        assert result["y"] == 0.0
        assert result["z"] == 0.0

    def test_no_missing_keys_in_full_set(self):
        """Test when full_set contains no missing keys"""
        data = ["a", "b", "c", "a", "b"]
        full_set = {"a", "b", "c"}  # All keys present in data
        result = plot_dist(data, sort=False, full_set=full_set)

        assert isinstance(result, dict)
        assert abs(result["a"] - (2/5 * 100)) < 1e-10  # 40%
        assert abs(result["b"] - (2/5 * 100)) < 1e-10  # 40%
        assert abs(result["c"] - (1/5 * 100)) < 1e-10  # 20%


def test_get_ratio_counts_selection_is_deterministic(rp_logger):
    """EPMCTDM-7630: which categorical bars the univariate plot shows must not depend
    on the global RNG.

    `get_ratio_counts` keeps the most and least common items plus a random selection
    from the middle. Drawing that middle slice from the module-level `random` meant the
    same data produced a different accuracy report on every run.
    """
    rp_logger.info("Test 'get_ratio_counts' selects the same items across runs")
    ratio_counts = {f"v{i}": 100 - i for i in range(60)}

    selections = []
    for seed in (1, 7919, 104729):
        random.seed(seed)          # perturb the global RNG between calls
        selections.append(tuple(get_ratio_counts(dict(ratio_counts)).keys()))

    assert len(set(selections)) == 1, (
        "selection must be stable regardless of global RNG state"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_ratio_counts_keeps_extremes_and_size(rp_logger):
    """Seeding must not change *what kind* of selection is made: the most and least
    common items are still always present and the count is unchanged, so metric
    magnitudes stay comparable with previously published reports."""
    rp_logger.info("Test 'get_ratio_counts' still keeps extremes and count")
    ratio_counts = {f"v{i}": 100 - i for i in range(60)}

    result = get_ratio_counts(dict(ratio_counts), count=30)

    assert "v0" in result, "most common item must be kept"
    assert "v59" in result, "least common item must be kept"
    assert len(result) == 30
    rp_logger.info(SUCCESSFUL_MESSAGE)


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


def test_encode_categories_is_exported_from_the_metrics_package(rp_logger):
    """
    'tdm_syngen' reaches this helper through 'syngen.ml.metrics', the way it already
    imports 'encode_categorical_features'. Keep the public name importable.
    """
    rp_logger.info("Test 'encode_categories' is importable from 'syngen.ml.metrics'")
    from syngen.ml.metrics import encode_categories as exported

    assert exported is encode_categories
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_encode_categorical_features_returns_shared_codes_as_arrays(rp_logger):
    """
    The return type is a list of bare float arrays, not DataFrames - the column names
    are lost, which is why 'encode_categories' exists for callers that need them. The
    encoder is fitted on the concatenation, so codes mean the same thing in both.
    """
    rp_logger.info("Test 'encode_categorical_features' shares codes across frames")
    original = pd.DataFrame({"c": ["x", "y", None]})
    synthetic = pd.DataFrame({"c": ["y", "z", "x"]})

    encoded = encode_categorical_features([original, synthetic])

    assert [type(frame) for frame in encoded] == [np.ndarray, np.ndarray]
    # 'x' is the second category of ('?', 'x', 'y', 'z') in both frames
    assert encoded[0][0, 0] == encoded[1][2, 0]
    # NaN becomes its own '?' category rather than being dropped
    assert encoded[0][2, 0] == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)
