import pandas as pd
import pytest

from benchmarks.text_sequence_training import (
    _generate_frame,
    _legacy_rnn_units,
    run_scenario,
)
from tests.conftest import SUCCESSFUL_MESSAGE


@pytest.mark.parametrize("architecture", ["legacy", "optimized"])
def test_text_sequence_benchmark_returns_finite_result(architecture, rp_logger):
    result = run_scenario(
        architecture=architecture,
        text_max_len=6,
        columns=1,
        rows=4,
        batch_size=2,
        warmup_steps=1,
        measured_steps=1,
    )

    assert set(result) == {
        "architecture",
        "batch_size",
        "columns",
        "compiled_train_step",
        "finite_loss",
        "length",
        "loss",
        "median_train_step_ms",
        "parameters",
        "rows",
        "transformed_input_bytes",
    }
    assert result["architecture"] == architecture
    assert result["compiled_train_step"] is (architecture == "optimized")
    assert result["finite_loss"] is True
    assert result["median_train_step_ms"] > 0
    assert result["parameters"] > 0
    assert result["transformed_input_bytes"] > 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "text_max_len, expected_units",
    [
        (0, 16),
        (1, 32),
        (6, 32),
        (7, 128),
        (12, 128),
        (13, 256),
        (16, 256),
        (17, 512),
        (200, 512),
    ],
)
def test_legacy_rnn_units_match_historical_boundaries(
    text_max_len,
    expected_units,
    rp_logger,
):
    assert _legacy_rnn_units(text_max_len) == expected_units
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_frame_is_seeded_and_exercises_padding(rp_logger):
    first = _generate_frame(rows=20, text_max_len=7, columns=2, seed=1729)
    second = _generate_frame(rows=20, text_max_len=7, columns=2, seed=1729)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["text_0", "text_1"]
    for column in first:
        lengths = first[column].str.len()
        assert lengths.max() == 7
        assert lengths.min() < 7
    rp_logger.info(SUCCESSFUL_MESSAGE)
