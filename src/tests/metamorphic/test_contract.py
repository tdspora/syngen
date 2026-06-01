"""MR-C — output contract: row count == requested size; columns & order preserved."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import specs, mr_asserts as mr        # noqa: E402

DATASETS = list(specs.all_datasets())


@pytest.mark.metamorphic
@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_contract(dataset, mrun):
    res = mrun(dataset, "csv")
    problems = []
    for table in res.spec.tables:
        problems += mr.contract(res, table)
    assert not problems, "; ".join(problems)
