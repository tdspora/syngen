"""Equivariance to a date shift (Part-A 2.2).

Shifting every date in the input by +1 year should shift the generated dates by
about +1 year (the numeric-affine case 2.1 is covered in test_transformations).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DS = "datetime_patterns"
COL = "signup_date"          # plain %Y-%m-%d column (unambiguous to parse)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_date_shift_equivariance(mrun):
    base = mrun(DS, "csv")
    df = base.originals[DS].copy()
    shifted = (pd.to_datetime(df[COL]) + pd.Timedelta(days=365)).dt.strftime("%Y-%m-%d")
    df[COL] = shifted
    res = mrun(DS, "csv", tables_override={DS: df})

    base_mean = pd.to_datetime(base.generated[DS][COL], errors="coerce").dropna().map(pd.Timestamp.toordinal).mean()
    shft_mean = pd.to_datetime(res.generated[DS][COL], errors="coerce").dropna().map(pd.Timestamp.toordinal).mean()
    delta_days = shft_mean - base_mean
    # expect ~ +365 days; allow generous slack for stochastic generation
    assert 365 - 180 <= delta_days <= 365 + 180, \
        f"generated {COL} shifted by {delta_days:.0f} days, expected ~365"
