"""Robustness & seed behaviour (Part-A 8.1 noise, 8.3 outliers, 6.2 seed distinguishability)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import mr_asserts as mr        # noqa: E402

DS = "numeric_explicit"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_small_noise_robustness(mrun):
    """8.1: a tiny perturbation of the inputs causes only a small change in the output."""
    base = mrun(DS, "csv")
    df = base.originals[DS].copy()
    rng = np.random.default_rng(0)
    for col in ("income", "score", "bimodal", "exponential"):
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s + rng.normal(0, 0.001 * (s.std() or 1.0), size=len(s))
    res = mrun(DS, "csv", tables_override={DS: df})
    problems = mr.distribution_equivalent(base, res, tol=mr.preservation_tol())
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_outlier_robustness(mrun):
    """8.3: a handful of extreme outliers must not blow up the generated median."""
    base = mrun(DS, "csv")
    df = base.originals[DS].copy()
    df.loc[df.index[:25], "score"] = 1e4                       # 0.5% extreme outliers
    res = mrun(DS, "csv", tables_override={DS: df})
    base_med = pd.to_numeric(base.generated[DS]["score"], errors="coerce").median()
    out_med = pd.to_numeric(res.generated[DS]["score"], errors="coerce").median()
    spread = pd.to_numeric(base.originals[DS]["score"], errors="coerce").std() or 1.0
    assert abs(out_med - base_med) <= 3 * spread, \
        f"median moved {out_med - base_med:.2f} (> 3*std) after adding outliers"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_seed_distinguishability(mrun):
    """6.2: different seeds -> different rows (the model samples), but close distributions."""
    a = mrun(DS, "csv", seed=10)
    b = mrun(DS, "csv", seed=20)
    ga, gb = a.generated[DS], b.generated[DS]
    cols = [c for c in ga.columns if c != "id"]
    identical = (ga[cols].reset_index(drop=True)
                 .astype(str).eq(gb[cols].reset_index(drop=True).astype(str)).all(axis=1).mean())
    assert identical < 0.99, "different seeds produced (near-)identical rows"
    assert not mr.cross_format(a, b), "different seeds drifted out of distribution"
