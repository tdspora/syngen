"""MR-IO1 — loader round-trip fidelity (no model, fast).

A DataFrame written to each supported file format and read back must equal the
original (dtype-aware). Typed formats (avro/parquet/pkl) must preserve NaN vs
empty-string exactly; CSV/Excel are allowed the documented empty<->null ambiguity.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import specs, mr_asserts as mr               # noqa: E402
from _lib.runner import read_table, source_path        # noqa: E402
from _lib.generate_datasets import SEED                 # noqa: E402

TYPED_FORMATS = {"avro", "parquet", "pkl"}              # preserve "" vs NaN exactly

# (dataset, table, fmt) for every materialised file
_CASES = []
for _name, _spec in specs.all_datasets().items():
    _frames = _spec.builder(np.random.default_rng(SEED))
    for _table in _spec.tables:
        for _fmt in _spec.formats:
            _CASES.append((_name, _table, _fmt))


@pytest.mark.metamorphic
@pytest.mark.parametrize("dataset,table,fmt", _CASES,
                         ids=[f"{d}-{t}-{f}" for d, t, f in _CASES])
def test_loader_roundtrip(dataset, table, fmt):
    spec = specs.all_datasets()[dataset]
    builder_df = spec.builder(np.random.default_rng(SEED))[table]
    path = source_path(dataset, table, fmt)
    assert os.path.exists(path), f"missing generated file {path}"
    loaded = read_table(path, fmt)
    discrepancies = mr.frame_equivalent(
        builder_df, loaded, text_null_equiv=(fmt not in TYPED_FORMATS))
    assert not discrepancies, f"[{dataset}/{table}/{fmt}] " + "; ".join(discrepancies)
