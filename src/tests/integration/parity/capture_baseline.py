"""Capture golden statistical baselines from the current backend.

Run this on the TensorFlow code (before/at the start of the migration) to freeze
the behavior the PyTorch path will be measured against:

    python src/tests/integration/parity/capture_baseline.py            # all fixtures
    python src/tests/integration/parity/capture_baseline.py numeric_wide categorical

Each baseline JSON stores the profile of the original data, the profile of the
generated data, key uniqueness, FK validity, and the tolerances in force. The
committed ``baselines/*.json`` are the acceptance reference for ``test_parity.py``.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import List

from stats import Tolerances, profile_table, key_profile, fk_validity
from runner import FIXTURES, BASELINES_DIR, run_fixture, FixtureSpec, RunResult


def build_baseline(spec: FixtureSpec, result: RunResult, tol: Tolerances) -> dict:
    baseline = {"fixture": spec.name, "tolerances": asdict(tol), "tables": {}}
    for table, kinds in spec.column_kinds.items():
        domains = spec.email_domains if table in spec.column_kinds else {}
        entry = {
            "original": profile_table(result.originals[table], kinds, domains),
            "generated": profile_table(result.generated[table], kinds, domains),
            "keys": {},
        }
        for key_col in spec.pk_columns.get(table, []):
            if key_col in result.generated[table].columns:
                entry["keys"][key_col] = key_profile(result.generated[table], key_col)
        for uuid_col in spec.uuid_columns.get(table, []):
            if uuid_col in result.generated[table].columns:
                entry["keys"][uuid_col] = key_profile(result.generated[table], uuid_col)
        baseline["tables"][table] = entry

    baseline["fk_validity"] = {}
    for fk in spec.fks:
        child = result.generated[fk["child"]]
        parent = result.generated[fk["parent"]]
        label = f"{fk['child']}.{fk['fk_col']}->{fk['parent']}.{fk['parent_pk']}"
        baseline["fk_validity"][label] = fk_validity(
            child, fk["fk_col"], parent[fk["parent_pk"]]
        )
    return baseline


def capture(names: List[str]) -> None:
    os.makedirs(BASELINES_DIR, exist_ok=True)
    for name in names:
        spec = FIXTURES[name]
        # Key/FK-focused fixtures record catastrophic-collapse-only tolerances;
        # distribution-focused fixtures record the strict defaults.
        tol = (Tolerances.catastrophic_collapse_only()
               if spec.key_focused else Tolerances())
        print(f"[capture] running fixture {name!r} through TF train -> infer ...")
        result = run_fixture(spec)
        baseline = build_baseline(spec, result, tol)
        out_path = os.path.join(BASELINES_DIR, f"{name}.json")
        with open(out_path, "w") as fp:
            json.dump(baseline, fp, indent=2)
        print(f"[capture] wrote {out_path}")


if __name__ == "__main__":
    requested = sys.argv[1:] or list(FIXTURES)
    capture(requested)
