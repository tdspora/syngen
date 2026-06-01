"""Diversity / mode coverage (Part-A 14.2 entropy, 14.3 coverage, 15.3 imbalance,
plus a bimodal mode-coverage check standing in for 16.1 additivity, and a 13.3/14.1
mode-collapse guard)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import mr_asserts as mr        # noqa: E402

EPOCHS = 8


@pytest.mark.metamorphic
@pytest.mark.slow
def test_categorical_entropy_preserved(mrun):
    """14.2: categorical diversity (entropy) doesn't shrink much."""
    res = mrun("categorical_explicit", "csv", epochs=EPOCHS)
    assert not mr.entropy_preserved(res, "categorical_explicit")


@pytest.mark.metamorphic
@pytest.mark.slow
def test_numeric_space_coverage(mrun):
    """14.3: synthetic numeric data covers most of the real value space."""
    res = mrun("numeric_explicit", "csv", epochs=EPOCHS)
    assert not mr.coverage_numeric(res, "numeric_explicit")


@pytest.mark.metamorphic
@pytest.mark.slow
def test_rare_category_present(mrun):
    """15.3: the rare imbalanced category (BR ~5%) is not lost."""
    res = mrun("categorical_explicit", "csv", epochs=EPOCHS)
    gen = set(res.generated["categorical_explicit"]["country"].dropna().astype(str))
    assert "BR" in gen, f"rare category 'BR' dropped; got {sorted(gen)}"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_bimodal_both_modes_present(mrun):
    """16.1 / 14.x: both modes of the bimodal column survive (no sub-population dropped)."""
    res = mrun("numeric_explicit", "csv", epochs=EPOCHS)
    b = pd.to_numeric(res.generated["numeric_explicit"]["bimodal"], errors="coerce")
    assert (b < 45).any() and (b >= 45).any(), "a bimodal mode was dropped (mode collapse)"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_no_mode_collapse(mrun):
    """13.3 / 14.1 proxy: no numeric column collapses to a constant, no categorical to one value.

    Run at EPOCHS (not the 2-epoch default): on the wide "kitchen-sink" table a binary/
    imbalanced categorical legitimately underfits to its majority value after only 2
    epochs, which is low-epoch underfitting, not the catastrophic collapse this guards."""
    res = mrun("mixed_wide", "csv", epochs=EPOCHS)
    assert not mr.generation_diverse(res, "mixed_wide")
