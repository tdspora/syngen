"""MR-D — distribution preservation vs the original (anti-collapse) + null ratios.

A stochastic VAE at low epochs cannot reproduce a distribution tightly, so these
fail only on *catastrophic* range/category collapse (the 18-90 -> 18-40 failure)
plus an explicit per-column null-ratio drift check.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import specs, mr_asserts as mr        # noqa: E402

# Distribution preservation is asserted only for datasets designed to test
# distributions. Key-focused datasets (keys_relational_*, relations_chain) exist to
# test PK/UQ/FK on small dimension tables (50-800 rows) that a stochastic VAE cannot
# reproduce per-column from so few rows — they are covered by MR-K instead.
DIST_DATASETS = [n for n, s in specs.all_datasets().items() if not s.key_focused]


@pytest.mark.metamorphic
@pytest.mark.slow
@pytest.mark.parametrize("dataset", DIST_DATASETS)
def test_no_distribution_collapse(dataset, mrun):
    res = mrun(dataset, "csv")
    problems = []
    for table in res.spec.tables:
        problems += mr.preservation(res, table)
        problems += mr.null_ratio_preserved(res, table)
        problems += mr.datetime_parse_preserved(res, table)
    assert not problems, "; ".join(problems)
