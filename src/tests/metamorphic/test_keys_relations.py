"""MR-K — keys & relations: PK/UQ uniqueness, regex conformance, FK integrity."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import specs, mr_asserts as mr        # noqa: E402

DATASETS = list(specs.all_datasets())
FK_DATASETS = [n for n, s in specs.all_datasets().items() if s.fks]


@pytest.mark.metamorphic
@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_pk_uq_and_regex(dataset, mrun):
    """MR-K1/K2/K5: PK/UQ uniqueness >= 0.999 and regex-key conformance."""
    res = mrun(dataset, "csv")
    assert not mr.keys(res), "; ".join(mr.keys(res))


@pytest.mark.metamorphic
@pytest.mark.slow
@pytest.mark.parametrize("dataset", FK_DATASETS)
def test_fk_integrity(dataset, mrun):
    """MR-K3/K4: every generated FK value exists in the generated parent PK set."""
    res = mrun(dataset, "csv")
    assert not mr.fk_integrity(res), "; ".join(mr.fk_integrity(res))
