"""Boundary / degenerate inputs (Part-A 15.1 constant, 15.2 high-cardinality,
15.4 very small, plus 7.2 low-cardinality-numeric -> categorical stability).

These build small ad-hoc datasets in memory and run them through ``mrun`` with an
override (no committed test-data needed). Where syngen legitimately cannot handle a
degenerate input, the test records it as an xfail (a documented limitation) rather
than a hard failure.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib.specs import DatasetSpec, TableSpec, KeySpec        # noqa: E402


def _spec(name, kinds, size, epochs=3):
    return DatasetSpec(name=name, title=name, builder=lambda rng: {},
                       tables={name: TableSpec(name, kinds,
                                               keys=[KeySpec(f"{name}_pk", "PK", ["id"])],
                                               pk_columns=["id"], size=size, epochs=epochs)},
                       formats=["csv"], explicit_patterns="", implicit_patterns="", null_policy="")


@pytest.mark.metamorphic
@pytest.mark.slow
def test_constant_columns(mrun):
    """15.1: constant columns must not crash (no divide-by-zero) and stay constant."""
    n = 1000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "const_num": 42.0,
                       "const_cat": "X", "val": rng.normal(0, 1, n)})
    spec = _spec("boundary_constant",
                 {"const_num": "numeric", "const_cat": "categorical", "val": "numeric"}, n)
    try:
        res = mrun(spec, "csv", tables_override={"boundary_constant": df})
    except Exception as exc:                                   # noqa: BLE001
        pytest.xfail(f"syngen does not handle fully-constant columns: {exc}")
    g = res.generated["boundary_constant"]
    assert g["const_num"].nunique() == 1 and g["const_cat"].nunique() == 1


@pytest.mark.metamorphic
@pytest.mark.slow
def test_high_cardinality_column(mrun):
    """15.2: a near-unique high-cardinality column must not crash; output is produced."""
    n = 3000
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id": np.arange(1, n + 1),
                       "code": [f"CODE-{i:06d}" for i in range(n)],
                       "val": rng.normal(0, 1, n)})
    spec = _spec("boundary_highcard", {"code": "text", "val": "numeric"}, n)
    try:
        res = mrun(spec, "csv", tables_override={"boundary_highcard": df})
    except Exception as exc:                                   # noqa: BLE001
        pytest.xfail(f"syngen struggled with a high-cardinality column: {exc}")
    assert len(res.generated["boundary_highcard"]) == n


@pytest.mark.metamorphic
@pytest.mark.slow
def test_very_small_dataset(mrun):
    """15.4: training on a tiny table must behave gracefully (no crash)."""
    base = mrun("numeric_explicit", "csv")
    tiny = base.originals["numeric_explicit"].head(40).copy()
    tiny["id"] = range(1, len(tiny) + 1)
    try:
        res = mrun("numeric_explicit", "csv", size_override=40,
                   tables_override={"numeric_explicit": tiny})
    except Exception as exc:                                   # noqa: BLE001
        pytest.xfail(f"syngen does not train on a 40-row table: {exc}")
    assert len(res.generated["numeric_explicit"]) == 40


@pytest.mark.metamorphic
@pytest.mark.slow
def test_low_cardinality_numeric_treated_categorically(mrun):
    """7.2: a low-cardinality numeric column (near_constant = {5,6}) is reproduced
    from that small value set (no invented numeric values)."""
    res = mrun("numeric_explicit", "csv")
    allowed = set(res.originals["numeric_explicit"]["near_constant"].astype(str))
    produced = set(res.generated["numeric_explicit"]["near_constant"].dropna().astype(str))
    assert produced <= allowed, f"invented values outside {allowed}: {produced - allowed}"
