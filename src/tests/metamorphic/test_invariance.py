"""Invariance to representation changes (Part-A 1.2 column order, 1.3 category
renaming, 17.1 category encoding).

The generator should learn from the *data*, not from incidental representation:
re-ordering columns, renaming category labels, or encoding categories as integers
must not change the structure of the output.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import mr_asserts as mr        # noqa: E402


def _freq_shape_close(a, b, tol=0.15):
    """Compare two label-agnostic frequency *shapes* (sorted descending)."""
    fa = a.dropna().astype(str).value_counts(normalize=True).sort_values(ascending=False).to_numpy()
    fb = b.dropna().astype(str).value_counts(normalize=True).sort_values(ascending=False).to_numpy()
    n = max(len(fa), len(fb))
    fa = np.pad(fa, (0, n - len(fa)))
    fb = np.pad(fb, (0, n - len(fb)))
    return float(np.abs(fa - fb).max()) <= tol


@pytest.mark.metamorphic
@pytest.mark.slow
def test_column_order_invariance(mrun):
    """1.2: shuffling the column order of the source must not collapse the output."""
    base = mrun("numeric_explicit", "csv")
    reordered = base.originals["numeric_explicit"].sample(frac=1.0, axis=1, random_state=3)
    res = mrun("numeric_explicit", "csv", tables_override={"numeric_explicit": reordered})
    # column order legitimately differs (we reordered the input); 1.2 is about the
    # per-column *distributions* being invariant, so ignore order/row-count diffs.
    problems = [d for d in mr.distribution_equivalent(base, res, tol=mr.preservation_tol())
                if "column order" not in d and "row_count" not in d]
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_category_rename_invariance(mrun):
    """1.3: renaming category labels must not change the frequency structure."""
    base = mrun("categorical_explicit", "csv")
    df = base.originals["categorical_explicit"].copy()
    mapping = {"US": "United_States", "UK": "Britain", "DE": "Germany",
               "FR": "France", "IN": "India", "BR": "Brazil"}
    df["country"] = df["country"].map(mapping)
    res = mrun("categorical_explicit", "csv", tables_override={"categorical_explicit": df})
    assert _freq_shape_close(base.generated["categorical_explicit"]["country"],
                             res.generated["categorical_explicit"]["country"]), \
        "country frequency shape changed after renaming labels"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_category_encoding_invariance(mrun):
    """17.1: encoding a category as integer codes must not change its structure."""
    base = mrun("categorical_explicit", "csv")
    df = base.originals["categorical_explicit"].copy()
    codes = {c: i for i, c in enumerate(sorted(df["country"].dropna().unique()))}
    df["country"] = df["country"].map(codes)
    res = mrun("categorical_explicit", "csv", tables_override={"categorical_explicit": df})
    assert _freq_shape_close(base.generated["categorical_explicit"]["country"],
                             res.generated["categorical_explicit"]["country"]), \
        "country frequency shape changed after integer-encoding"
