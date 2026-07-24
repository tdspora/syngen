import numpy as np
import pandas as pd
import pytest

from benchmarks.text_sequence_training import (
    _build_model,
    _generate_frame,
    _generate_structured_frame,
    _legacy_rnn_units,
    _true_token_ids,
    run_quality_scenario,
    run_scenario,
    structured_prefix_length,
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
        "rnn_units",
        "rows",
        "transformed_input_bytes",
    }
    assert result["architecture"] == architecture
    assert result["compiled_train_step"] is (architecture == "optimized")
    assert result["finite_loss"] is True
    assert result["median_train_step_ms"] > 0
    assert result["parameters"] > 0
    assert result["rnn_units"] > 0
    assert result["transformed_input_bytes"] > 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_text_sequence_benchmark_supports_multiple_columns(rp_logger):
    result = run_scenario(
        architecture="optimized",
        text_max_len=6,
        columns=2,
        rows=4,
        batch_size=2,
        warmup_steps=1,
        measured_steps=1,
    )

    assert result["columns"] == 2
    assert result["finite_loss"] is True
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_units_override_replaces_optimized_rnn_units_and_restores_default(rp_logger):
    overridden = run_scenario(
        architecture="optimized",
        text_max_len=6,
        columns=1,
        rows=4,
        batch_size=2,
        warmup_steps=1,
        measured_steps=1,
        units_override=64,
    )
    restored = run_scenario(
        architecture="optimized",
        text_max_len=6,
        columns=1,
        rows=4,
        batch_size=2,
        warmup_steps=1,
        measured_steps=1,
    )
    legacy_unaffected = run_scenario(
        architecture="legacy",
        text_max_len=6,
        columns=1,
        rows=4,
        batch_size=2,
        warmup_steps=1,
        measured_steps=1,
        units_override=64,
    )

    assert overridden["rnn_units"] == 64
    assert overridden["parameters"] > restored["parameters"]
    # A later call without an override must not leak the previous
    # units_override into the module-level TEXT_RNN_UNITS constant.
    assert restored["rnn_units"] == 32
    # units_override only applies to the optimized architecture.
    assert legacy_unaffected["rnn_units"] == _legacy_rnn_units(6)
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


@pytest.mark.parametrize("architecture_version", [1, 2])
def test_true_token_ids_are_identical_across_architectures(architecture_version, rp_logger):
    """The fair-accuracy ground truth must not depend on architecture_version.

    Only vocab_size (and therefore the one-hot/int32 branch inside
    ``transform``) differs between legacy and optimized; the underlying
    tokenizer id sequence used as the accuracy label must stay identical so
    the metric is comparable across architectures.
    """
    frame = _generate_frame(rows=10, text_max_len=6, columns=1, seed=1729)
    _, _, features, _units = _build_model(
        frame, text_max_len=6, architecture_version=architecture_version
    )

    ids = _true_token_ids(features[0], frame["text_0"])

    assert ids.dtype == np.int32
    assert ids.shape == (10, 6)
    # Every non-padding id must be a valid tokenizer-assigned id, and no id
    # can exceed len(word_index), regardless of architecture_version.
    assert ids.max() <= len(features[0].tokenizer.word_index)
    assert ids.min() >= 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("architecture", ["legacy", "optimized"])
def test_quality_scenario_returns_finite_fair_metrics(architecture, rp_logger):
    result = run_quality_scenario(
        architecture=architecture,
        text_max_len=6,
        columns=1,
        rows=8,
        batch_size=4,
        epochs=2,
        report_every=1,
    )

    assert set(result) == {
        "architecture",
        "pattern",
        "length",
        "columns",
        "rows",
        "batch_size",
        "epochs",
        "rnn_units",
        "initial_loss",
        "final_loss",
        "initial_character_accuracy",
        "final_character_accuracy",
        "loss_trace",
        "character_accuracy_trace",
    }
    assert result["architecture"] == architecture
    assert result["pattern"] == "random"
    assert np.isfinite(result["initial_loss"])
    assert np.isfinite(result["final_loss"])
    assert 0.0 <= result["initial_character_accuracy"] <= 1.0
    assert 0.0 <= result["final_character_accuracy"] <= 1.0
    assert len(result["loss_trace"]) == 2
    assert len(result["character_accuracy_trace"]) == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_quality_scenario_rejects_unknown_pattern(rp_logger):
    with pytest.raises(ValueError):
        run_quality_scenario(
            architecture="optimized",
            text_max_len=6,
            columns=1,
            rows=8,
            batch_size=4,
            epochs=1,
            report_every=1,
            pattern="not-a-real-pattern",
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_quality_scenario_structured_pattern_reports_prefix_fidelity(rp_logger):
    result = run_quality_scenario(
        architecture="optimized",
        text_max_len=10,
        columns=1,
        rows=8,
        batch_size=4,
        epochs=2,
        report_every=1,
        pattern="structured",
        prefix_ratio=0.4,
    )

    assert result["pattern"] == "structured"
    assert result["prefix_length"] == structured_prefix_length(10, 0.4)
    assert 0.0 <= result["final_prefix_accuracy"] <= 1.0
    assert 0.0 <= result["final_prefix_confidence"] <= 1.0
    assert 0.0 <= result["final_prefix_generation_match_rate"] <= 1.0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_quality_scenario_structured_pattern_recovers_generation_fidelity(rp_logger):
    """Regression test for the long-structured-column fix.

    Reproduces the reported bug (URLs like 'https://www...' rendering as
    'httpps//ww..x' under generation) at a smaller, CI-friendly scale: a
    templated column with a 10-character deterministic prefix. Under the
    current production default records positions whose tokens are invariant
    across every training row and restores those exact tokens after the real
    stochastic top_p sampling path, so fidelity does not depend on model
    confidence or training duration -- see
    tmp/22-07-26/syngen-text-lstm-perf-findings.md for the full derivation.
    """
    result = run_quality_scenario(
        architecture="optimized",
        text_max_len=20,
        columns=1,
        rows=8,
        batch_size=4,
        epochs=2,
        report_every=1,
        pattern="structured",
        prefix_ratio=0.5,
    )

    assert result["rnn_units"] == 32
    assert result["final_prefix_generation_match_rate"] == 1.0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_quality_scenario_units_override_increases_optimized_capacity(rp_logger):
    small = run_quality_scenario(
        architecture="optimized",
        text_max_len=10,
        columns=1,
        rows=8,
        batch_size=4,
        epochs=1,
        report_every=1,
        units_override=8,
    )
    large = run_quality_scenario(
        architecture="optimized",
        text_max_len=10,
        columns=1,
        rows=8,
        batch_size=4,
        epochs=1,
        report_every=1,
        units_override=64,
    )

    assert small["rnn_units"] == 8
    assert large["rnn_units"] == 64
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_quality_scenario_trace_sampling_always_includes_last_epoch(rp_logger):
    result = run_quality_scenario(
        architecture="optimized",
        text_max_len=6,
        columns=1,
        rows=8,
        batch_size=4,
        epochs=7,
        report_every=5,
    )

    assert result["loss_trace"][-1] == result["final_loss"]
    assert result["character_accuracy_trace"][-1] == result["final_character_accuracy"]
    # Sampled at stride 5 over 7 epochs: indices 0, 5, plus the forced last (6).
    assert len(result["loss_trace"]) == 3
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_quality_scenario_rejects_non_positive_report_every(rp_logger):
    with pytest.raises(ValueError):
        run_quality_scenario(
            architecture="optimized",
            text_max_len=6,
            columns=1,
            rows=8,
            batch_size=4,
            epochs=2,
            report_every=0,
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_structured_frame_shares_a_deterministic_prefix(rp_logger):
    first = _generate_structured_frame(rows=20, text_max_len=10, columns=2, seed=1729, prefix_ratio=0.4)
    second = _generate_structured_frame(rows=20, text_max_len=10, columns=2, seed=1729, prefix_ratio=0.4)

    pd.testing.assert_frame_equal(first, second)
    prefix_len = structured_prefix_length(10, 0.4)
    assert prefix_len == 4
    for column in first:
        lengths = first[column].str.len()
        assert (lengths == 10).all()
        prefixes = first[column].str.slice(0, prefix_len)
        assert prefixes.nunique() == 1
        suffixes = first[column].str.slice(prefix_len)
        assert suffixes.nunique() > 1
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_generate_structured_frame_rejects_oversized_prefix(rp_logger):
    with pytest.raises(ValueError):
        _generate_structured_frame(rows=4, text_max_len=1000, columns=1, seed=1729, prefix_ratio=0.9)
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
