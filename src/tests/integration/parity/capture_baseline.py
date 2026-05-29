"""Capture golden statistical baselines from the current backend.

Run this on the TensorFlow code (before/at the start of the migration) to freeze
the behavior the PyTorch path will be measured against:

    python src/tests/integration/parity/capture_baseline.py            # all fixtures
    python src/tests/integration/parity/capture_baseline.py numeric_wide categorical
    N_RUNS=5 python src/tests/integration/parity/capture_baseline.py    # ensemble size

This captures an **ensemble**: each fixture is run ``N_RUNS`` times (the model is
stochastic, so a single run cannot reproduce itself within tight tolerances), and
per-column statistical bands (mean/std/min/max) are stored. ``test_parity.py``
then accepts a PyTorch run that falls within TF's own observed band and rejects a
genuine collapse. The committed ``baselines/*.json`` are the acceptance reference.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import List

import numpy as np

from stats import (
    EnsembleTolerances, profile_table, key_profile, fk_validity, aggregate_ensemble,
)
from runner import FIXTURES, BASELINES_DIR, run_fixture, FixtureSpec, RunResult

DEFAULT_N_RUNS = int(os.environ.get("N_RUNS", "5"))


def build_baseline(spec: FixtureSpec, results: List[RunResult], tol: EnsembleTolerances) -> dict:
    baseline = {
        "fixture": spec.name,
        "method": "ensemble",
        "n_runs": len(results),
        "tolerances": asdict(tol),
        "tables": {},
    }
    for table, kinds in spec.column_kinds.items():
        domains = spec.email_domains
        profiles = [profile_table(r.generated[table], kinds, domains) for r in results]
        entry = {
            "original": profile_table(results[0].originals[table], kinds, domains),
            "ensemble": aggregate_ensemble(profiles),
            "keys": {},
        }
        key_cols = spec.pk_columns.get(table, []) + spec.uuid_columns.get(table, [])
        for key_col in key_cols:
            uniqs = [key_profile(r.generated[table], key_col)["uniqueness"]
                     for r in results if key_col in r.generated[table].columns]
            if uniqs:
                entry["keys"][key_col] = {
                    "uniqueness_min": float(min(uniqs)),
                    "uniqueness_mean": float(np.mean(uniqs)),
                }
        baseline["tables"][table] = entry

    baseline["fk_validity"] = {}
    for fk in spec.fks:
        vals = [fk_validity(r.generated[fk["child"]], fk["fk_col"],
                            r.generated[fk["parent"]][fk["parent_pk"]]) for r in results]
        label = f"{fk['child']}.{fk['fk_col']}->{fk['parent']}.{fk['parent_pk']}"
        baseline["fk_validity"][label] = {"min": float(min(vals)), "mean": float(np.mean(vals))}
    return baseline


def capture(names: List[str], n_runs: int = DEFAULT_N_RUNS) -> None:
    os.makedirs(BASELINES_DIR, exist_ok=True)
    tol = EnsembleTolerances()
    for name in names:
        spec = FIXTURES[name]
        print(f"[capture] {name!r}: running TF train -> infer {n_runs}x ...", flush=True)
        results = [run_fixture(spec, random_seed=10 + i) for i in range(n_runs)]
        baseline = build_baseline(spec, results, tol)
        out_path = os.path.join(BASELINES_DIR, f"{name}.json")
        with open(out_path, "w") as fp:
            json.dump(baseline, fp, indent=2)
        print(f"[capture] wrote {out_path} (n_runs={n_runs})", flush=True)


if __name__ == "__main__":
    requested = sys.argv[1:] or list(FIXTURES)
    capture(requested)
