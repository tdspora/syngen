"""MR-T — transformation relations: properties that must hold for any correct backend.

These transform the *input* and assert a predictable relation on the *output*,
so they need no golden oracle and validate the backend's invariances directly.
All run on ``numeric_explicit`` (single table, fast) with a fixed seed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import specs, mr_asserts as mr        # noqa: E402

DS = "numeric_explicit"
TBL = "numeric_explicit"


def _ignore_rowcount(discrepancies):
    return [d for d in discrepancies if "row_count" not in d]


# Permutation / duplication change the TRAINING data, so the model (and its
# stochastic output) legitimately differs run-to-run. The metamorphic guarantee is
# therefore "no distribution COLLAPSE" (anti-collapse tolerance), not tight equality
# — a backend that pathologically depended on row order would still collapse and fail.
@pytest.mark.metamorphic
@pytest.mark.slow
def test_permutation_invariance(mrun):
    """MR-T1: shuffling the training rows must not collapse the output distribution."""
    base = mrun(DS, "csv", seed=10)
    shuffled = base.originals[TBL].sample(frac=1.0, random_state=1).reset_index(drop=True)
    res = mrun(DS, "csv", seed=10, tables_override={TBL: shuffled})
    problems = mr.distribution_equivalent(base, res, tol=mr.preservation_tol())
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_row_duplication_preserves_distribution(mrun):
    """MR-T2: duplicating every row (2x) must not collapse the marginal distributions."""
    import pandas as pd
    base = mrun(DS, "csv", seed=10)
    orig = base.originals[TBL]
    doubled = pd.concat([orig, orig], ignore_index=True)
    doubled["id"] = range(1, len(doubled) + 1)          # keep PK unique in the source
    res = mrun(DS, "csv", seed=10, tables_override={TBL: doubled})
    problems = _ignore_rowcount(mr.distribution_equivalent(base, res, tol=mr.preservation_tol()))
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_affine_shift_propagates(mrun):
    """MR-T3: shifting a numeric column by +C shifts the generated column's mean by ~C
    (tests scaler fit + inverse_transform end to end)."""
    shift = 1000.0
    base = mrun(DS, "csv", seed=10)
    shifted = base.originals[TBL].copy()
    shifted["score"] = shifted["score"] + shift
    res = mrun(DS, "csv", seed=10, tables_override={TBL: shifted})
    delta = res.generated[TBL]["score"].mean() - base.generated[TBL]["score"].mean()
    assert abs(delta - shift) < 0.20 * shift, \
        f"generated score mean shifted by {delta:.1f}, expected ~{shift}"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_size_scaling(mrun):
    """MR-T4: requesting N vs 2N rows changes only the count, not the marginals."""
    base = mrun(DS, "csv", seed=10)                      # size = spec default (5000)
    half = mrun(DS, "csv", seed=10, size_override=2000)
    assert len(half.generated[TBL]) == 2000
    leftover = _ignore_rowcount(mr.distribution_equivalent(base, half))
    assert not leftover, "; ".join(leftover)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_determinism(mrun):
    """MR-T5: two independent runs with the same seed produce (near) identical output."""
    base = mrun(DS, "csv", seed=7)
    twin = mrun(DS, "csv", seed=7, tables_override={TBL: base.originals[TBL]})
    assert not mr.determinism(base, twin), "; ".join(mr.determinism(base, twin))
