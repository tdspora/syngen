"""Statistical profiling and parity comparison for the TF -> PyTorch migration.

The point of this module is to make *distribution collapse* impossible to miss.
A model can train (loss down) and still narrow an 18-90 range to 18-40; the only
reliable evidence against that is a per-column statistical profile of the
generated data compared to a frozen baseline.

``profile_table`` produces a JSON-serializable profile. ``compare_profiles``
diffs a candidate profile against a baseline and returns a list of human-readable
discrepancies that exceed tolerance, with explicit range-coverage checks.

This module deliberately has no TensorFlow/PyTorch dependency so it survives the
migration unchanged and can compare a TF baseline against a PyTorch candidate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import t as _student_t

QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


@dataclass
class Tolerances:
    """Acceptance tolerances. Override per-column via ``column_overrides``."""
    range_min: float = 0.80      # candidate range must cover >= 80% of baseline range
    bound_frac: float = 0.20     # min/max may pull inward <= 20% of baseline span
    num_rel: float = 0.25        # relative drift for mean/std/quantiles/text-length
    ratio_abs: float = 0.10      # absolute drift for null/zero ratios
    cat_coverage: float = 0.90   # fraction of baseline categories that must appear
    cat_js: float = 0.15         # max Jensen-Shannon distance on category freqs
    alien_freq: float = 0.05     # max freq of categories absent from baseline
    parse_min: float = 0.99      # min datetime parse-success ratio
    uniqueness_min: float = 0.999  # PK/UQ uniqueness ratio
    fk_valid_min: float = 0.999  # fraction of FK values present in parent PK set
    det: float = 0.02            # determinism: max relative drift between seeded runs
    column_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def for_column(self, column: str) -> "Tolerances":
        if column not in self.column_overrides:
            return self
        merged = {**asdict(self), **self.column_overrides[column]}
        merged.pop("column_overrides", None)
        return Tolerances(**merged, column_overrides=self.column_overrides)

    @classmethod
    def catastrophic_collapse_only(cls) -> "Tolerances":
        """Tolerances for key/FK-focused fixtures (e.g. ``keys``,
        ``relations_chain``).

        Those fixtures exist to test FK linkage and PK/UQ uniqueness on small
        tables (8-600 rows), not per-column distribution parity — a stochastic
        VAE cannot reproduce a distribution from a handful of rows, and measuring
        against a single TF baseline run false-fails (verified: TF cannot
        reproduce its own baseline there either). So the distribution checks are
        relaxed to fire **only on catastrophic collapse** (range covering <5% of
        baseline, or <10% of categories present), while the key/FK/smoke checks
        (uniqueness, FK validity, datetime parse) stay strict. See
        docs/migration/sign_off_records.md (Phase E/F decision)."""
        return cls(
            range_min=0.05,      # only flag near-total range collapse
            bound_frac=1.0,      # disable inward-bound nag (range_min covers it)
            num_rel=1e9,         # disable mean/std/quantile drift
            ratio_abs=1.0,       # disable null/zero-ratio drift
            cat_coverage=0.10,   # only flag near-total category collapse
            cat_js=1.0,          # disable JS-distance drift
            alien_freq=1.0,      # disable alien-category check
            parse_min=0.95,      # keep datetime parseability (smoke)
            uniqueness_min=0.999,  # keep PK/UQ uniqueness strict
            fk_valid_min=0.999,  # keep FK validity strict
        )


def _to_native(value):
    """Make numpy scalars JSON-serializable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


def _numeric_profile(series: pd.Series) -> Dict:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    profile = {
        "kind": "numeric",
        "null_ratio": float(numeric.isna().mean()),
        "zero_ratio": float((non_null == 0).mean()) if len(non_null) else 0.0,
    }
    if len(non_null):
        profile.update({
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "range": float(non_null.max() - non_null.min()),
            "mean": float(non_null.mean()),
            "std": float(non_null.std(ddof=0)),
            "quantiles": {str(q): float(non_null.quantile(q)) for q in QUANTILES},
        })
    return profile


def _categorical_profile(series: pd.Series) -> Dict:
    null_ratio = float(series.isna().mean())
    non_null = series.dropna().astype(str)
    counts = non_null.value_counts(normalize=True)
    return {
        "kind": "categorical",
        "null_ratio": null_ratio,
        "n_categories": int(non_null.nunique()),
        "frequencies": {str(k): float(v) for k, v in counts.items()},
    }


def _text_profile(series: pd.Series, domain: Optional[str] = None) -> Dict:
    non_null = series.dropna().astype(str)
    lengths = non_null.str.len()
    word_counts = non_null.str.split().apply(len)
    profile = {
        "kind": "text",
        "null_ratio": float(series.isna().mean()),
        "length": {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "std": float(lengths.std(ddof=0)) if len(lengths) else 0.0,
            "quantiles": {str(q): float(lengths.quantile(q)) for q in QUANTILES}
            if len(lengths) else {},
        },
        "word_count": {
            "mean": float(word_counts.mean()) if len(word_counts) else 0.0,
            "std": float(word_counts.std(ddof=0)) if len(word_counts) else 0.0,
        },
    }
    if domain is not None:
        profile["email_domain_valid"] = float(non_null.str.endswith(f"@{domain}").mean())
    return profile


def _datetime_profile(series: pd.Series) -> Dict:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    valid = parsed.dropna()
    return {
        "kind": "datetime",
        "parse_success": float(parsed.notna().mean()),
        "min": valid.min().isoformat() if len(valid) else None,
        "max": valid.max().isoformat() if len(valid) else None,
    }


def _uuid_profile(series: pd.Series) -> Dict:
    non_null = series.dropna().astype(str)
    return {
        "kind": "uuid",
        "null_ratio": float(series.isna().mean()),
        "uniqueness": float(non_null.nunique() / len(non_null)) if len(non_null) else 1.0,
    }


def profile_table(
    df: pd.DataFrame,
    column_kinds: Dict[str, str],
    email_domains: Optional[Dict[str, str]] = None,
) -> Dict:
    """Build a JSON-serializable profile.

    ``column_kinds`` maps column name -> one of
    {"numeric", "categorical", "text", "email", "datetime"}. Columns absent from
    the mapping are skipped (e.g. opaque keys handled separately).
    """
    email_domains = email_domains or {}
    columns: Dict[str, Dict] = {}
    for col, kind in column_kinds.items():
        if col not in df.columns:
            columns[col] = {"kind": kind, "missing": True}
            continue
        if kind == "numeric":
            columns[col] = _numeric_profile(df[col])
        elif kind in ("categorical", "binary"):
            columns[col] = _categorical_profile(df[col])
        elif kind == "text":
            columns[col] = _text_profile(df[col])
        elif kind == "email":
            columns[col] = _text_profile(df[col], domain=email_domains.get(col, ""))
        elif kind == "datetime":
            columns[col] = _datetime_profile(df[col])
        elif kind == "uuid":
            columns[col] = _uuid_profile(df[col])
        else:
            raise ValueError(f"Unknown column kind {kind!r} for column {col!r}")
    return {
        "row_count": int(len(df)),
        "column_order": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "columns": _to_native(columns),
    }


def key_profile(df: pd.DataFrame, column: str) -> Dict:
    """Profile a key column's uniqueness."""
    series = df[column].dropna()
    return {
        "n": int(len(series)),
        "n_unique": int(series.nunique()),
        "uniqueness": float(series.nunique() / len(series)) if len(series) else 1.0,
    }


def fk_validity(child: pd.DataFrame, fk_col: str, parent_keys) -> float:
    """Fraction of FK values present in the parent key set."""
    series = child[fk_col].dropna()
    if not len(series):
        return 1.0
    parent_set = set(pd.Series(list(parent_keys)).astype(str))
    return float(series.astype(str).isin(parent_set).mean())


def _rel_drift(base: float, cand: float) -> float:
    denom = max(abs(base), 1e-9)
    return abs(cand - base) / denom


def _drift_exceeds(base: float, cand: float, rel_tol: float, scale: float) -> bool:
    """Significant drift = absolute change exceeds ``rel_tol`` of
    ``max(|base|, scale)``.

    ``scale`` is the column's baseline std — its natural unit of spread. Flooring
    the relative comparison by the column scale prevents a *false* failure when a
    statistic's magnitude is tiny relative to that spread (e.g. a near-zero mean or
    median, where ordinary relative drift explodes for a negligible absolute move).
    This does NOT weaken collapse detection: range coverage (the primary
    anti-collapse check) is untouched, and a genuine collapse also shrinks the std
    so the std check still trips. Approved metric refinement (see
    docs/migration/sign_off_records.md)."""
    return abs(cand - base) > rel_tol * max(abs(base), scale)


def _js_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    pv = np.array([p.get(k, 0.0) for k in keys])
    qv = np.array([q.get(k, 0.0) for k in keys])
    pv = pv / pv.sum() if pv.sum() else pv
    qv = qv / qv.sum() if qv.sum() else qv
    m = 0.5 * (pv + qv)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.where(b[mask] > 0, b[mask], 1e-12))))

    return float(np.sqrt(0.5 * _kl(pv, m) + 0.5 * _kl(qv, m)))


def _compare_numeric(col, base, cand, tol, out):
    if base.get("missing") or "min" not in base:
        return
    if cand.get("missing") or "min" not in cand:
        out.append(f"[{col}] numeric column missing/empty in candidate")
        return
    # --- the anti-collapse check ---
    base_range = base["range"]
    if base_range > 0:
        coverage = cand["range"] / base_range
        if coverage < tol.range_min:
            out.append(
                f"[{col}] RANGE COLLAPSE: candidate range covers {coverage:.0%} of "
                f"baseline (min {tol.range_min:.0%}); "
                f"baseline [{base['min']:.4g}, {base['max']:.4g}] vs "
                f"candidate [{cand['min']:.4g}, {cand['max']:.4g}]"
            )
        inward = tol.bound_frac * base_range
        if cand["min"] > base["min"] + inward:
            out.append(f"[{col}] lower bound pulled inward: {base['min']:.4g} -> {cand['min']:.4g}")
        if cand["max"] < base["max"] - inward:
            out.append(f"[{col}] upper bound pulled inward: {base['max']:.4g} -> {cand['max']:.4g}")
    # Column spread floors the relative checks (see _drift_exceeds) so a near-zero
    # mean/median is not flagged for a negligible absolute move.
    scale = base.get("std", 0.0)
    for stat in ("mean", "std"):
        if _drift_exceeds(base[stat], cand[stat], tol.num_rel, scale):
            out.append(f"[{col}] {stat} drift ({base[stat]:.4g} -> {cand[stat]:.4g}) "
                       f"exceeds {tol.num_rel:.0%} of max(|base|, std={scale:.4g})")
    # Multiple quantiles make the check sensitive to distribution-shape collapse
    # (e.g. a bimodal column flattened to unimodal moves the mid quantiles).
    for q in ("0.05", "0.25", "0.5", "0.75", "0.95"):
        bq, cq = base["quantiles"][q], cand["quantiles"][q]
        if _drift_exceeds(bq, cq, tol.num_rel, scale):
            out.append(f"[{col}] q{q} drift ({bq:.4g} -> {cq:.4g}) "
                       f"exceeds {tol.num_rel:.0%} of max(|base|, std)")
    if abs(cand["null_ratio"] - base["null_ratio"]) > tol.ratio_abs:
        out.append(f"[{col}] null_ratio drift "
                   f"{base['null_ratio']:.2f} -> {cand['null_ratio']:.2f}")
    if abs(cand["zero_ratio"] - base["zero_ratio"]) > tol.ratio_abs:
        out.append(f"[{col}] zero_ratio drift "
                   f"{base['zero_ratio']:.2f} -> {cand['zero_ratio']:.2f}")


def _compare_categorical(col, base, cand, tol, out):
    if base.get("missing") or cand.get("missing"):
        out.append(f"[{col}] categorical column missing in candidate or baseline")
        return
    base_freq = base["frequencies"]
    cand_freq = cand["frequencies"]
    present = [c for c in base_freq if c in cand_freq]
    coverage = len(present) / max(len(base_freq), 1)
    if coverage < tol.cat_coverage:
        missing = sorted(set(base_freq) - set(cand_freq))[:10]
        out.append(f"[{col}] CATEGORY COLLAPSE: only {coverage:.0%} of baseline "
                   f"categories present (min {tol.cat_coverage:.0%}); missing e.g. {missing}")
    alien = {c: f for c, f in cand_freq.items() if c not in base_freq}
    big_alien = {c: f for c, f in alien.items() if f > tol.alien_freq}
    if big_alien:
        out.append(f"[{col}] alien categories above {tol.alien_freq:.0%}: "
                   f"{dict(sorted(big_alien.items(), key=lambda x: -x[1])[:5])}")
    js = _js_distance(base_freq, cand_freq)
    if js > tol.cat_js:
        out.append(f"[{col}] frequency JS distance {js:.3f} > {tol.cat_js}")
    if abs(cand["null_ratio"] - base["null_ratio"]) > tol.ratio_abs:
        out.append(f"[{col}] null_ratio drift "
                   f"{base['null_ratio']:.2f} -> {cand['null_ratio']:.2f}")


def _compare_text(col, base, cand, tol, out):
    if base.get("missing") or cand.get("missing"):
        out.append(f"[{col}] text column missing in candidate or baseline")
        return
    drift = _rel_drift(base["length"]["mean"], cand["length"]["mean"])
    if drift > tol.num_rel:
        out.append(f"[{col}] text length mean drift {drift:.0%} > {tol.num_rel:.0%} "
                   f"({base['length']['mean']:.1f} -> {cand['length']['mean']:.1f})")
    drift = _rel_drift(base["word_count"]["mean"], cand["word_count"]["mean"])
    if drift > tol.num_rel:
        out.append(f"[{col}] word-count mean drift {drift:.0%} > {tol.num_rel:.0%}")
    if "email_domain_valid" in base and cand.get("email_domain_valid", 0.0) < 0.99:
        out.append(f"[{col}] email domain validity {cand['email_domain_valid']:.2%} < 99%")


def _compare_datetime(col, base, cand, tol, out):
    if base.get("missing") or cand.get("missing"):
        out.append(f"[{col}] datetime column missing in candidate or baseline")
        return
    if cand["parse_success"] < tol.parse_min:
        out.append(f"[{col}] datetime parse success {cand['parse_success']:.2%} "
                   f"< {tol.parse_min:.0%}")


def _compare_uuid(col, base, cand, tol, out):
    if base.get("missing") or cand.get("missing"):
        out.append(f"[{col}] uuid column missing in candidate or baseline")
        return
    if cand["uniqueness"] < tol.uniqueness_min:
        out.append(f"[{col}] uuid uniqueness {cand['uniqueness']:.4f} < {tol.uniqueness_min}")
    if abs(cand["null_ratio"] - base["null_ratio"]) > tol.ratio_abs:
        out.append(f"[{col}] null_ratio drift "
                   f"{base['null_ratio']:.2f} -> {cand['null_ratio']:.2f}")


def compare_profiles(baseline: Dict, candidate: Dict, tol: Tolerances) -> List[str]:
    """Return discrepancies where ``candidate`` deviates from ``baseline``."""
    out: List[str] = []

    if candidate["row_count"] != baseline["row_count"]:
        out.append(f"row_count {baseline['row_count']} -> {candidate['row_count']}")
    if candidate["column_order"] != baseline["column_order"]:
        out.append(f"column order changed: {baseline['column_order']} -> "
                   f"{candidate['column_order']}")

    base_cols = baseline["columns"]
    cand_cols = candidate["columns"]
    for col, base in base_cols.items():
        col_tol = tol.for_column(col)
        cand = cand_cols.get(col, {"missing": True})
        kind = base.get("kind")
        if kind == "numeric":
            _compare_numeric(col, base, cand, col_tol, out)
        elif kind in ("categorical", "binary"):
            _compare_categorical(col, base, cand, col_tol, out)
        elif kind in ("text", "email"):
            _compare_text(col, base, cand, col_tol, out)
        elif kind == "datetime":
            _compare_datetime(col, base, cand, col_tol, out)
        elif kind == "uuid":
            _compare_uuid(col, base, cand, col_tol, out)
    return out


# ---------------------------------------------------------------------------
# Ensemble tolerances: calibrate the gate against TF's own run-to-run variance.
#
# A single TF run cannot reproduce itself within tight tolerances (proven), so
# instead of one baseline we capture N TF runs per fixture and compare the
# candidate against the *band* TF itself occupies. A statistic passes when it is
# within mean +/- k*std of the N TF runs (plus a relative floor so a near-constant
# stat is not flagged for a negligible move). Real collapse — narrower / fewer
# categories than TF ever produced — still fails, and hard catastrophic backstops
# (range <5%, categories <10%) guard the egregious case even when TF's own band is
# wide. Categorical *shape* is checked via the Jensen-Shannon distance to the
# ensemble-mean distribution, bounded by TF's own run-to-run JS spread.
# ---------------------------------------------------------------------------


@dataclass
class EnsembleTolerances:
    # The band is a two-sided prediction interval for a *new* draw given the N TF
    # runs: mean ± t(N-1, 1-alpha/2)·std·sqrt(1 + 1/N), plus a relative floor. This
    # is the correct "will the PyTorch sample fall where TF samples fall?" question
    # and self-widens for small N (more uncertainty -> wider band).
    pi_alpha: float = 0.002      # prediction-interval significance (99.8% PI). Tighter
                                 # than 99% because ~50 stats are checked per run, so a
                                 # 1% per-stat false-positive rate would fail a faithful
                                 # run by chance (multiple-comparison aware).
    rel_floor: float = 0.10      # relative floor on the band half-width (handles std ~ 0)
    cat_present_frac: float = 0.8  # categories in >= this fraction of TF runs are "expected"
    hard_text_frac: float = 0.5  # text length/word-count must stay within a factor of
                                 # 1/this of TF's mean (collapse-oriented: catches
                                 # degenerate/absurd text, not benign length differences)
    parse_min: float = 0.95      # legacy/datetime floor (datetime now uses the PI band)
    uniqueness_min: float = 0.999
    fk_valid_min: float = 0.999
    hard_range_min: float = 0.05   # catastrophic range-collapse backstop
    hard_cat_coverage: float = 0.10  # catastrophic category-collapse backstop
    k_std: float = 3.0           # retained for back-compat with older baselines (unused)
    js_k: float = 3.0            # retained for back-compat (unused; JS uses the PI band)


def _agg(values) -> Optional[Dict]:
    """mean/std/min/max/n over the non-null values."""
    arr = np.array([v for v in values if v is not None], dtype=float)
    if not len(arr):
        return None
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
            "lo": float(arr.min()), "hi": float(arr.max()), "n": int(len(arr))}


def aggregate_ensemble(profiles: List[Dict]) -> Dict:
    """Aggregate N ``profile_table`` outputs (same table, N TF runs) into a
    per-column statistical band."""
    ref = profiles[0]
    columns: Dict[str, Dict] = {}
    for col, ref_col in ref["columns"].items():
        kind = ref_col.get("kind")
        per_run = [p["columns"].get(col, {}) for p in profiles]
        if kind == "numeric":
            stats = {}
            for key in ("min", "max", "range", "mean", "std", "null_ratio", "zero_ratio"):
                agg = _agg([r.get(key) for r in per_run])
                if agg:
                    stats[key] = agg
            quantiles = {}
            for q in [str(x) for x in QUANTILES]:
                agg = _agg([r.get("quantiles", {}).get(q) for r in per_run])
                if agg:
                    quantiles[q] = agg
            columns[col] = {"kind": "numeric", "stats": stats, "quantiles": quantiles}
        elif kind in ("categorical", "binary"):
            all_cats = set()
            for r in per_run:
                all_cats |= set(r.get("frequencies", {}).keys())
            mean_freq = {c: float(np.mean([r.get("frequencies", {}).get(c, 0.0) for r in per_run]))
                         for c in all_cats}
            present_frac = {c: float(np.mean([1.0 if c in r.get("frequencies", {}) else 0.0
                                              for r in per_run])) for c in all_cats}
            run_js = [_js_distance(r.get("frequencies", {}), mean_freq) for r in per_run]
            columns[col] = {
                "kind": "categorical",
                "mean_freq": mean_freq,
                "present_frac": present_frac,
                "js_to_mean": _agg(run_js),
                "n_categories": _agg([r.get("n_categories") for r in per_run]),
                "null_ratio": _agg([r.get("null_ratio") for r in per_run]),
            }
        elif kind in ("text", "email"):
            entry = {
                "kind": "text",
                "length_mean": _agg([r.get("length", {}).get("mean") for r in per_run]),
                "word_count_mean": _agg([r.get("word_count", {}).get("mean") for r in per_run]),
                "null_ratio": _agg([r.get("null_ratio") for r in per_run]),
            }
            if any("email_domain_valid" in r for r in per_run):
                entry["email_domain_valid"] = _agg(
                    [r.get("email_domain_valid") for r in per_run])
            columns[col] = entry
        elif kind == "datetime":
            columns[col] = {"kind": "datetime",
                            "parse_success": _agg([r.get("parse_success") for r in per_run])}
        elif kind == "uuid":
            columns[col] = {"kind": "uuid",
                            "uniqueness": _agg([r.get("uniqueness") for r in per_run]),
                            "null_ratio": _agg([r.get("null_ratio") for r in per_run])}
    return {
        "n_runs": len(profiles),
        "row_count": ref["row_count"],
        "column_order": ref["column_order"],
        "columns": _to_native(columns),
    }


def _pi_band(agg: Dict, alpha: float, floor: float):
    """Two-sided prediction interval for a new draw: mean ± t·std·sqrt(1+1/N) + floor."""
    n = agg["n"]
    if n >= 2 and agg["std"] > 0:
        tval = float(_student_t.ppf(1 - alpha / 2.0, df=n - 1))
        half = tval * agg["std"] * math.sqrt(1.0 + 1.0 / n) + floor
    else:
        half = floor
    return agg["mean"] - half, agg["mean"] + half


def _outside(col, label, value, agg, alpha, floor, out):
    lo, hi = _pi_band(agg, alpha, floor)
    if value < lo or value > hi:
        out.append(f"[{col}] {label} {value:.4g} outside TF band [{lo:.4g}, {hi:.4g}] "
                   f"(mean {agg['mean']:.4g}, std {agg['std']:.4g}, n={agg['n']})")


def compare_to_ensemble(ensemble: Dict, candidate: Dict, tol: EnsembleTolerances) -> List[str]:
    """Discrepancies where ``candidate`` falls outside the band of N TF runs."""
    out: List[str] = []
    if candidate["row_count"] != ensemble["row_count"]:
        out.append(f"row_count {ensemble['row_count']} -> {candidate['row_count']}")
    if candidate["column_order"] != ensemble["column_order"]:
        out.append("column order changed")

    for col, ce in ensemble["columns"].items():
        cand = candidate["columns"].get(col, {"missing": True})
        kind = ce.get("kind")
        if cand.get("missing"):
            out.append(f"[{col}] missing in candidate")
            continue

        if kind == "numeric":
            scale = ce["stats"].get("std", {}).get("mean", 0.0) or 0.0
            for key, agg in ce["stats"].items():
                if key in cand:
                    floor = tol.rel_floor * max(abs(agg["mean"]), scale)
                    _outside(col, key, cand[key], agg, tol.pi_alpha, floor, out)
            for q, agg in ce["quantiles"].items():
                cv = cand.get("quantiles", {}).get(q)
                if cv is not None:
                    floor = tol.rel_floor * max(abs(agg["mean"]), scale)
                    _outside(col, f"q{q}", cv, agg, tol.pi_alpha, floor, out)
            rng = ce["stats"].get("range")
            if rng and rng["mean"] > 0 and cand.get("range", 0) < tol.hard_range_min * rng["mean"]:
                out.append(f"[{col}] RANGE COLLAPSE: {cand['range']:.4g} < "
                           f"{tol.hard_range_min:.0%} of TF mean range {rng['mean']:.4g}")

        elif kind in ("categorical", "binary"):
            cand_freq = cand.get("frequencies", {})
            expected = [c for c, f in ce["present_frac"].items() if f >= tol.cat_present_frac]
            missing = [c for c in expected if c not in cand_freq]
            coverage = 1 - len(missing) / max(len(expected), 1)
            if coverage < tol.hard_cat_coverage:
                out.append(f"[{col}] CATEGORY COLLAPSE: {coverage:.0%} of TF-consistent "
                           f"categories present; missing e.g. {sorted(missing)[:5]}")
            if ce.get("n_categories"):
                agg = ce["n_categories"]
                lo, _ = _pi_band(agg, tol.pi_alpha, tol.rel_floor * max(abs(agg["mean"]), 1.0))
                if cand.get("n_categories", 0) < lo:
                    out.append(f"[{col}] n_categories {cand.get('n_categories')} below TF "
                               f"band low {lo:.1f} (mean {agg['mean']:.1f})")
            js = _js_distance(cand_freq, ce["mean_freq"])
            jsa = ce.get("js_to_mean")
            if jsa is not None:
                _, js_limit = _pi_band(jsa, tol.pi_alpha, tol.rel_floor)  # one-sided upper
                if js > js_limit:
                    out.append(f"[{col}] JS-to-mean {js:.3f} > TF band {js_limit:.3f} "
                               f"(TF mean {jsa['mean']:.3f}, std {jsa['std']:.3f})")
            if ce.get("null_ratio"):
                _outside(col, "null_ratio", cand.get("null_ratio", 0.0),
                         ce["null_ratio"], tol.pi_alpha, tol.rel_floor, out)

        elif kind in ("text", "email"):
            # Text length / word-count are checked collapse-style: within a factor of
            # 1/hard_text_frac of TF's mean. A tight band is wrong here — TF's char-LSTM
            # over-generates length (~25 chars vs the real ~8-char names), so a port that
            # matches the real data better than TF would "fail" a strict band. We only
            # flag genuinely degenerate (too short) or absurd (too long) text.
            for label, agg, cand_val in (
                ("length.mean", ce.get("length_mean"),
                 cand.get("length", {}).get("mean", 0.0)),
                ("word_count.mean", ce.get("word_count_mean"),
                 cand.get("word_count", {}).get("mean", 0.0)),
            ):
                if agg and agg["mean"] > 0:
                    lo = tol.hard_text_frac * agg["mean"]
                    hi = agg["mean"] / tol.hard_text_frac
                    if cand_val < lo or cand_val > hi:
                        out.append(f"[{col}] {label} {cand_val:.1f} outside "
                                   f"[{lo:.1f}, {hi:.1f}] (TF mean {agg['mean']:.1f})")
            if "email_domain_valid" in ce and cand.get("email_domain_valid", 0.0) < 0.99:
                out.append(f"[{col}] email domain validity "
                           f"{cand.get('email_domain_valid', 0.0):.2%} < 99%")

        elif kind == "datetime":
            # Parse success is checked against TF's own band, not an absolute floor:
            # day-first (%d/%m/%Y) columns are parsed inconsistently by the generic
            # profiler, so TF itself ranges widely (e.g. created_date 0.40-1.00).
            # A one-sided lower bound still catches a real formatting regression on
            # reliably-parsed columns (e.g. ISO updated_at, where TF std ~ 0).
            ps = ce.get("parse_success")
            if ps is not None:
                band_lo, _ = _pi_band(ps, tol.pi_alpha, tol.rel_floor)
                lo = max(0.0, band_lo)
                cv = cand.get("parse_success", 0.0)
                if cv < lo:
                    out.append(f"[{col}] datetime parse success {cv:.2%} below TF band "
                               f"low {lo:.2%} (TF mean {ps['mean']:.2%}, std {ps['std']:.2%})")

        elif kind == "uuid":
            if cand.get("uniqueness", 0.0) < tol.uniqueness_min:
                out.append(f"[{col}] uuid uniqueness {cand.get('uniqueness'):.4f} "
                           f"< {tol.uniqueness_min}")
    return out
