import random

import pandas as pd
import numpy as np
import pytest

from unittest.mock import patch

from syngen.ml.metrics.metrics_classes.metrics import (
    BivariateMetric,
    Clustering,
    Utility,
    UnivariateMetric
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
