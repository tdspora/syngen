"""Parity tests: PyTorch (or any candidate backend) vs the frozen TF baseline.

These exercise the real public API (``launch_train`` -> ``launch_infer``) per
fixture, profile the generated output, and assert it has not drifted from the
committed TF baseline beyond tolerance. The decisive assertion is anti-collapse:
numeric range coverage and categorical category coverage.

Heavy and backend-dependent, so opt-in:

    pytest src/tests/integration/parity -m parity

The ``test_collapse_is_detected`` self-test needs no backend and proves the guard
catches the exact regression we are defending against.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from stats import Tolerances, profile_table, key_profile, fk_validity, compare_profiles
from runner import FIXTURES, BASELINES_DIR, run_fixture

pytestmark = [pytest.mark.parity, pytest.mark.slow]


def _require_backend():
    """Skip backend-dependent tests when syngen / its model backend is absent."""
    try:
        import syngen.train  # noqa: F401
        import syngen.infer  # noqa: F401
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"syngen backend not importable: {error}")


def _load_baseline(name: str) -> dict:
    path = os.path.join(BASELINES_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"No committed baseline for {name!r}; run capture_baseline.py on TF first")
    with open(path) as fp:
        return json.load(fp)


@pytest.mark.parametrize("fixture_name", list(FIXTURES))
def test_parity_against_baseline(fixture_name):
    spec = FIXTURES[fixture_name]
    baseline = _load_baseline(fixture_name)
    _require_backend()
    tol = Tolerances(**{k: v for k, v in baseline["tolerances"].items()})

    result = run_fixture(spec)

    all_discrepancies = []
    for table, kinds in spec.column_kinds.items():
        domains = spec.email_domains
        gen_profile = profile_table(result.generated[table], kinds, domains)
        base_table = baseline["tables"][table]

        # Contract: row count matches request, column order/dtypes preserved.
        assert gen_profile["row_count"] == base_table["generated"]["row_count"], (
            f"{table}: row count {gen_profile['row_count']} != "
            f"{base_table['generated']['row_count']}"
        )

        # Anti-collapse / statistical parity vs the baseline's *generated* profile.
        discrepancies = compare_profiles(base_table["generated"], gen_profile, tol)
        all_discrepancies += [f"{table}: {d}" for d in discrepancies]

        # Key and UUID uniqueness must hold.
        unique_cols = (spec.pk_columns.get(table, [])
                       + spec.uuid_columns.get(table, []))
        for key_col in unique_cols:
            if key_col in result.generated[table].columns:
                prof = key_profile(result.generated[table], key_col)
                assert prof["uniqueness"] >= tol.uniqueness_min, (
                    f"{table}.{key_col} uniqueness {prof['uniqueness']:.4f} "
                    f"< {tol.uniqueness_min}"
                )

    for fk in spec.fks:
        child = result.generated[fk["child"]]
        parent = result.generated[fk["parent"]]
        valid = fk_validity(child, fk["fk_col"], parent[fk["parent_pk"]])
        assert valid >= tol.fk_valid_min, (
            f"FK {fk['child']}.{fk['fk_col']} -> {fk['parent']}.{fk['parent_pk']} "
            f"validity {valid:.4f} < {tol.fk_valid_min}"
        )

    assert not all_discrepancies, (
        "Distribution parity failures vs TF baseline:\n  - "
        + "\n  - ".join(all_discrepancies)
    )


@pytest.mark.parametrize("fixture_name", ["numeric_wide", "categorical", "mixed_complex"])
def test_determinism(fixture_name):
    """Two seeded runs must profile-equal within tolerance."""
    _require_backend()
    spec = FIXTURES[fixture_name]
    tol = Tolerances()
    r1 = run_fixture(spec, random_seed=123)
    r2 = run_fixture(spec, random_seed=123)
    for table, kinds in spec.column_kinds.items():
        p1 = profile_table(r1.generated[table], kinds, spec.email_domains)
        p2 = profile_table(r2.generated[table], kinds, spec.email_domains)
        det_tol = Tolerances(range_min=0.0, num_rel=tol.det, cat_coverage=0.0,
                             cat_js=tol.det, ratio_abs=tol.det)
        discrepancies = compare_profiles(p1, p2, det_tol)
        assert not discrepancies, (
            f"{table}: non-deterministic under fixed seed:\n  - "
            + "\n  - ".join(discrepancies)
        )


def test_collapse_is_detected():
    """Self-test (no backend): clamping a wide-range column must trip the guard.

    Proves the harness catches the 18-90 -> 18-40 failure before PyTorch exists.
    """
    kinds = {"age": "numeric"}
    original = pd.DataFrame({"age": list(range(18, 91)) * 10})
    healthy = profile_table(original, kinds)

    collapsed_df = pd.DataFrame({"age": [min(a, 40) for a in original["age"]]})
    collapsed = profile_table(collapsed_df, kinds)

    tol = Tolerances()
    assert compare_profiles(healthy, healthy, tol) == [], "baseline vs itself must pass"
    discrepancies = compare_profiles(healthy, collapsed, tol)
    assert any("RANGE COLLAPSE" in d for d in discrepancies), (
        f"range collapse not detected; got {discrepancies}"
    )


def test_category_collapse_is_detected():
    """Self-test: dropping most categories must trip the category-coverage guard."""
    kinds = {"city": "categorical"}
    original = pd.DataFrame({"city": [f"city_{i:02d}" for i in range(40)] * 10})
    healthy = profile_table(original, kinds)
    collapsed = profile_table(pd.DataFrame({"city": ["city_00", "city_01"] * 200}), kinds)

    discrepancies = compare_profiles(healthy, collapsed, Tolerances())
    assert any("CATEGORY COLLAPSE" in d for d in discrepancies), (
        f"category collapse not detected; got {discrepancies}"
    )
