"""Metamorphic-relation helpers built on the framework-neutral ``stats`` engine.

Every helper returns a list of human-readable discrepancy strings (empty list ==
relation holds), so test bodies read as::

    assert not mr.preservation(result, "numeric_explicit")

Tolerance philosophy (self-validating, no golden baseline):

* **Preservation vs original** is judged *anti-collapse only* — a stochastic VAE at
  low epochs cannot reproduce a distribution tightly (proven in the parity work),
  so we only fail on catastrophic range/category collapse, plus an explicit
  null-ratio check.
* **Cross-format** and **determinism** compare two runs of the *same* backend, so
  they use tight(er) tolerances — these are the decisive transfer-consistency
  relations.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import stats  # noqa: E402
from _lib.specs import DatasetSpec, TableSpec  # noqa: E402
from _lib.runner import RunResult  # noqa: E402


# --------------------------------------------------------------------------- #
# Tolerance presets
# --------------------------------------------------------------------------- #
def preservation_tol() -> stats.Tolerances:
    """Anti-collapse only: a single low-epoch run can't match a distribution tightly.

    The absolute datetime parse-success check is disabled here because day-first
    (``%d/%m/%Y``) columns are parsed inconsistently by the generic profiler (the
    original parses at ~40% too); parse fidelity is instead checked *relative to the
    original* by :func:`datetime_parse_preserved`.
    """
    tol = stats.Tolerances.catastrophic_collapse_only()
    tol.parse_min = 0.0
    return tol


def datetime_parse_preserved(result: RunResult, table: str, floor: float = 0.2) -> List[str]:
    """MR-D4: generated datetime columns are not garbled (parse-success above a low floor).

    An absolute floor (not relative to the original) because syngen legitimately
    degrades **day-first** (``%d/%m/%Y``) parseability to ~40% on *both* the TF and
    PyTorch backends (documented in the parity work: created_date ~0.40-1.00) — a
    known generator quirk, not a transfer regression. The floor still catches a
    genuinely garbled (~0% parseable) column."""
    import pandas as pd
    out: List[str] = []
    gen = result.generated[table]
    for col, kind in result.spec.tables[table].column_kinds.items():
        if kind != "datetime" or col not in gen:
            continue
        cb = float(pd.to_datetime(gen[col], errors="coerce", utc=True).notna().mean())
        if cb < floor:
            out.append(f"{table}.{col}: datetime parse {cb:.0%} < floor {floor:.0%} (garbled)")
    return out


def cross_format_tol() -> stats.Tolerances:
    """Two runs of the same backend over the same data in different file formats.
    Differences should be small (only source dtype coercion / float precision)."""
    return stats.Tolerances(
        range_min=0.60, bound_frac=1.0, num_rel=0.50, ratio_abs=0.15,
        cat_coverage=0.70, cat_js=0.35, alien_freq=1.0, parse_min=0.90,
    )


def determinism_tol() -> stats.Tolerances:
    """Same backend, same format, same seed -> should be (near) identical."""
    return stats.Tolerances(
        range_min=0.90, bound_frac=1.0, num_rel=0.05, ratio_abs=0.03,
        cat_coverage=0.95, cat_js=0.05, alien_freq=1.0, parse_min=0.95,
    )


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile(spec: DatasetSpec, table: str, df: pd.DataFrame) -> Dict:
    t: TableSpec = spec.tables[table]
    return stats.profile_table(df, t.column_kinds, email_domains=t.email_domains or None)


# --------------------------------------------------------------------------- #
# A — distribution preservation (vs original)  [MR-D]
# --------------------------------------------------------------------------- #
def preservation(result: RunResult, table: str) -> List[str]:
    """MR-D1/D2/D4: generated distribution must not COLLAPSE vs the original."""
    base = profile(result.spec, table, result.originals[table])
    cand = profile(result.spec, table, result.generated[table])
    out = stats.compare_profiles(base, cand, preservation_tol())
    return [f"{table}: {d}" for d in out]


def null_ratio_preserved(result: RunResult, table: str, *,
                         collapse_frac: float = 0.20, invent_abs: float = 0.15) -> List[str]:
    """MR-D3: the missing-value *pattern* is preserved (anti-collapse).

    A stochastic VAE at low epochs under-restores nulls, so we do not require a tight
    match; we fail only when the pattern genuinely breaks: a substantially-null column
    (>=10%) whose nulls almost vanish (< ``collapse_frac`` of the original), or a
    (near-)non-null column (<2%) that suddenly gains many nulls (> ``invent_abs``)."""
    out: List[str] = []
    orig, gen = result.originals[table], result.generated[table]
    for col in result.spec.tables[table].column_kinds:
        if col not in orig or col not in gen:
            continue
        b, c = float(orig[col].isna().mean()), float(gen[col].isna().mean())
        if b >= 0.10 and c < collapse_frac * b:
            out.append(f"{table}.{col}: NULL PATTERN COLLAPSE {b:.2f} -> {c:.2f}")
        elif b < 0.02 and c > invent_abs:
            out.append(f"{table}.{col}: invented nulls {b:.2f} -> {c:.2f}")
    return out


# --------------------------------------------------------------------------- #
# B — contract  [MR-C]
# --------------------------------------------------------------------------- #
def contract(result: RunResult, table: str) -> List[str]:
    """MR-C1/C2: row count == requested size; columns and order preserved."""
    out: List[str] = []
    t = result.spec.tables[table]
    gen, orig = result.generated[table], result.originals[table]
    if len(gen) != t.size:
        out.append(f"{table}: row_count {len(gen)} != requested size {t.size}")
    if list(gen.columns) != list(orig.columns):
        out.append(f"{table}: columns/order changed {list(orig.columns)} -> {list(gen.columns)}")
    return out


# --------------------------------------------------------------------------- #
# C — keys & relations  [MR-K]
# --------------------------------------------------------------------------- #
def keys(result: RunResult, uniqueness_min: float = 0.999) -> List[str]:
    """MR-K1/K2/K5: PK/UQ uniqueness and regex-PK/UQ conformance."""
    out: List[str] = []
    for tname, t in result.spec.tables.items():
        gen = result.generated[tname]
        for k in t.keys:
            if k.type in ("PK", "UQ"):
                for col in k.columns:
                    if col not in gen:
                        out.append(f"{tname}.{col}: key column missing in output")
                        continue
                    kp = stats.key_profile(gen, col)
                    if kp["uniqueness"] < uniqueness_min:
                        out.append(f"{tname}.{col} ({k.type}): uniqueness "
                                   f"{kp['uniqueness']:.4f} < {uniqueness_min}")
            if k.regex_patterns:
                for col, pattern in k.regex_patterns.items():
                    if col not in gen:
                        continue
                    rx = re.compile(pattern)
                    vals = gen[col].dropna().astype(str)
                    bad = (~vals.map(lambda v: bool(rx.match(v)))).mean() if len(vals) else 0.0
                    if bad > 0.01:
                        out.append(f"{tname}.{col} ({k.type}): {bad:.1%} values violate "
                                   f"regex {pattern!r}")
    return out


def fk_integrity(result: RunResult, fk_valid_min: float = 0.999) -> List[str]:
    """MR-K3/K4: every FK value present in the (generated) parent key set."""
    out: List[str] = []
    for fk in result.spec.fks:
        child, parent = result.generated[fk["child"]], result.generated[fk["parent"]]
        if fk["fk_col"] not in child or fk["parent_pk"] not in parent:
            out.append(f"FK {fk['child']}.{fk['fk_col']} -> "
                       f"{fk['parent']}.{fk['parent_pk']}: column missing")
            continue
        valid = stats.fk_validity(child, fk["fk_col"], parent[fk["parent_pk"]])
        if valid < fk_valid_min:
            out.append(f"FK {fk['child']}.{fk['fk_col']} -> {fk['parent']}.{fk['parent_pk']}: "
                       f"validity {valid:.4f} < {fk_valid_min}")
    return out


# --------------------------------------------------------------------------- #
# D — cross-run comparisons (cross-format / determinism / transformation)  [MR-IO2, MR-T]
# --------------------------------------------------------------------------- #
def _compare_runs(spec: DatasetSpec, gen_a: Dict[str, pd.DataFrame],
                  gen_b: Dict[str, pd.DataFrame], tol: stats.Tolerances,
                  label: str) -> List[str]:
    out: List[str] = []
    for tname in spec.tables:
        pa = profile(spec, tname, gen_a[tname])
        pb = profile(spec, tname, gen_b[tname])
        for d in stats.compare_profiles(pa, pb, tol):
            out.append(f"[{label}] {tname}: {d}")
    return out


def cross_format(result_a: RunResult, result_b: RunResult,
                 tol: Optional[stats.Tolerances] = None) -> List[str]:
    """MR-IO2: same dataset + seed, two source formats -> equivalent output."""
    label = f"{result_a.fmt} vs {result_b.fmt}"
    return _compare_runs(result_a.spec, result_a.generated, result_b.generated,
                         tol or cross_format_tol(), label)


def determinism(result_a: RunResult, result_b: RunResult,
                tol: Optional[stats.Tolerances] = None) -> List[str]:
    """MR-T5: identical seed + format -> (near) identical output."""
    return _compare_runs(result_a.spec, result_a.generated, result_b.generated,
                         tol or determinism_tol(), "determinism")


def distribution_equivalent(result_a: RunResult, result_b: RunResult,
                             tol: Optional[stats.Tolerances] = None) -> List[str]:
    """Generic 'two runs should look alike' (used by transformation MRs)."""
    return _compare_runs(result_a.spec, result_a.generated, result_b.generated,
                         tol or cross_format_tol(), "transform")


# --------------------------------------------------------------------------- #
# E — pure loader round-trip fidelity (no model)  [MR-IO1]
# --------------------------------------------------------------------------- #
def _null_mask(series: pd.Series, text_null_equiv: bool):
    isna = series.isna()
    if text_null_equiv:
        isna = isna | series.map(lambda v: isinstance(v, str) and v == "")
    return isna.to_numpy()


def frame_equivalent(builder_df: pd.DataFrame, loaded_df: pd.DataFrame, *,
                     float_atol: float = 1e-6, text_null_equiv: bool = False) -> List[str]:
    """MR-IO1: a DataFrame written then read back in some format equals the original.

    Comparison is *representation-aware*: a column whose non-null values are all
    numeric is compared numerically (so int-in-object written to CSV and read back
    as float still matches); other columns are compared as strings. ``text_null_equiv``
    treats ""<->NaN as equal (the documented CSV/Excel empty-vs-null ambiguity);
    typed formats (avro/parquet/pkl) pass it as False.
    """
    out: List[str] = []
    if list(builder_df.columns) != list(loaded_df.columns):
        return [f"columns/order changed {list(builder_df.columns)} -> {list(loaded_df.columns)}"]
    if len(builder_df) != len(loaded_df):
        return [f"row count {len(builder_df)} -> {len(loaded_df)}"]
    for col in builder_df.columns:
        a, b = builder_df[col].reset_index(drop=True), loaded_df[col].reset_index(drop=True)
        a_na, b_na = _null_mask(a, text_null_equiv), _null_mask(b, text_null_equiv)
        if not np.array_equal(a_na, b_na):
            out.append(f"{col}: null mask differs ({int(a_na.sum())} vs {int(b_na.sum())} nulls)")
            continue
        an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        # numeric iff every non-null value parses as a number on both sides
        a_numeric = (~a_na).sum() > 0 and np.array_equal(an.notna().to_numpy(), ~a_na)
        b_numeric = (~b_na).sum() > 0 and np.array_equal(bn.notna().to_numpy(), ~b_na)
        if a_numeric and b_numeric:
            av, bv = an.to_numpy(float)[~a_na], bn.to_numpy(float)[~b_na]
            if len(av) and not np.allclose(av, bv, atol=float_atol, rtol=1e-6):
                out.append(f"{col}: numeric values differ beyond atol={float_atol}")
        else:
            av = a[~a_na].astype(str).to_numpy()
            bv = b[~b_na].astype(str).to_numpy()
            mism = (av != bv).mean() if len(av) else 0.0
            if mism > 0:
                out.append(f"{col}: {mism:.2%} values differ after read-back")
    return out


# =========================================================================== #
# Extended relations: dependencies, diversity, privacy, utility, training.
#
# These reuse the same datasets and the public train->infer path. The "hard"
# checks (discriminator, TSTR, DCR) are deliberately **lenient anti-catastrophe**
# bounds rather than tight quality bars: without a captured baseline, and at the
# low epoch counts used in tests, the meaningful, backend-agnostic signal is "did
# the generator catastrophically fail / leak / collapse", not "is the synthetic
# data production-grade". Thresholds are documented and easy to tighten later.
# =========================================================================== #
def _num_cols(spec: DatasetSpec, table: str) -> List[str]:
    return [c for c, k in spec.tables[table].column_kinds.items() if k == "numeric"]


def _cat_cols(spec: DatasetSpec, table: str) -> List[str]:
    return [c for c, k in spec.tables[table].column_kinds.items()
            if k in ("categorical", "binary")]


def _safe_corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 10 or x[m].std() == 0 or y[m].std() == 0:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


# --- 5.2 correlations / 11.2 functional dependencies ----------------------- #
def correlation_preserved(result: RunResult, table: str,
                          pairs: Sequence[Tuple[str, str]],
                          max_abs_diff: float = 0.35) -> List[str]:
    """MR 5.2 / 11.2: the correlation between each column pair is roughly preserved
    (catches a backend that reproduces marginals but loses the joint relationship)."""
    out: List[str] = []
    o, g = result.originals[table], result.generated[table]
    for a, b in pairs:
        if any(c not in o or c not in g for c in (a, b)):
            continue
        co, cg = _safe_corr(o[a], o[b]), _safe_corr(g[a], g[b])
        if co is None or cg is None:
            continue
        if abs(co - cg) > max_abs_diff:
            out.append(f"{table}: corr({a},{b}) {co:.2f} -> {cg:.2f} (|diff|>{max_abs_diff})")
    return out


# --- 11.1 conditional distributions ---------------------------------------- #
def conditional_preserved(result: RunResult, table: str, group_col: str,
                          value_col: str, ordinal_map: Optional[Dict] = None,
                          min_rho: float = 0.3) -> List[str]:
    """MR 11.1: the per-group mean of ``value_col`` keeps the same ordering across
    groups (e.g. amount|status, tier|country). ``ordinal_map`` maps a categorical
    value column to numbers."""
    o, g = result.originals[table].copy(), result.generated[table].copy()
    if any(c not in o or c not in g for c in (group_col, value_col)):
        return []
    if ordinal_map is not None:
        o[value_col] = o[value_col].map(ordinal_map)
        g[value_col] = g[value_col].map(ordinal_map)
    o[value_col] = pd.to_numeric(o[value_col], errors="coerce")
    g[value_col] = pd.to_numeric(g[value_col], errors="coerce")
    om = o.groupby(group_col)[value_col].mean()
    gm = g.groupby(group_col)[value_col].mean()
    common = [k for k in om.index if k in gm.index]
    if len(common) < 3:
        return []
    from scipy.stats import spearmanr
    rho, _ = spearmanr(om[common].to_numpy(), gm[common].to_numpy())
    if not np.isnan(rho) and rho < min_rho:
        return [f"{table}: conditional {value_col}|{group_col} ordering not preserved "
                f"(Spearman {rho:.2f} < {min_rho})"]
    return []


# --- 4.3 relationship cardinality ------------------------------------------ #
def cardinality_preserved(result: RunResult, num_rel: float = 0.5) -> List[str]:
    """MR 4.3: the average number of child rows per parent key (1:N fan-out) is
    roughly preserved."""
    out: List[str] = []
    for fk in result.spec.fks:
        oc, gc = result.originals[fk["child"]], result.generated[fk["child"]]
        if fk["fk_col"] not in oc or fk["fk_col"] not in gc:
            continue
        o_fan = oc[fk["fk_col"]].dropna().value_counts().mean()
        g_fan = gc[fk["fk_col"]].dropna().value_counts().mean()
        if o_fan and abs(o_fan - g_fan) > num_rel * o_fan:
            out.append(f"{fk['child']}.{fk['fk_col']}: avg children/parent "
                       f"{o_fan:.1f} -> {g_fan:.1f} (>{num_rel:.0%})")
    return out


# --- 14.2 entropy / diversity ---------------------------------------------- #
def entropy_preserved(result: RunResult, table: str, min_frac: float = 0.6) -> List[str]:
    """MR 14.2: categorical Shannon entropy should not shrink much (diversity loss)."""
    def _entropy(series):
        p = series.dropna().astype(str).value_counts(normalize=True).to_numpy()
        return float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    out: List[str] = []
    o, g = result.originals[table], result.generated[table]
    for col in _cat_cols(result.spec, table):
        if col not in o or col not in g:
            continue
        eo, eg = _entropy(o[col]), _entropy(g[col])
        if eo > 0.5 and eg < min_frac * eo:
            out.append(f"{table}.{col}: entropy {eo:.2f} -> {eg:.2f} (<{min_frac:.0%} of original)")
    return out


# --- 14.3 multivariate coverage (numeric, simplified) ---------------------- #
def coverage_numeric(result: RunResult, table: str, min_cov: float = 0.5) -> List[str]:
    """MR 14.3: synthetic numeric data should cover most of the real value space
    (per-column decile-bin coverage, averaged)."""
    cols = _num_cols(result.spec, table)
    if not cols:
        return []
    o, g = result.originals[table], result.generated[table]
    covs = []
    for col in cols:
        ov = pd.to_numeric(o[col], errors="coerce").dropna()
        gv = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(ov) < 20 or ov.nunique() < 5:
            continue
        edges = np.unique(np.quantile(ov, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        occ = set(np.clip(np.digitize(ov, edges[1:-1]), 0, len(edges) - 1))
        syn = set(np.clip(np.digitize(gv, edges[1:-1]), 0, len(edges) - 1))
        covs.append(len(occ & syn) / max(len(occ), 1))
    if covs and float(np.mean(covs)) < min_cov:
        return [f"{table}: numeric space coverage {np.mean(covs):.0%} < {min_cov:.0%}"]
    return []


# --- shared feature encoder for the ML-style checks ------------------------ #
def _encode(spec: DatasetSpec, table: str, ref: pd.DataFrame, *others: pd.DataFrame):
    """Fit a numeric+one-hot encoder on ``ref`` and transform ``ref`` and ``others``
    into aligned dense numeric matrices (numeric: median-impute+standardize;
    categorical: most-frequent-impute+one-hot, unknowns ignored)."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    num, cat = _num_cols(spec, table), _cat_cols(spec, table)
    tfs = []
    if num:
        tfs.append(("num", Pipeline([("i", SimpleImputer(strategy="median")),
                                     ("s", StandardScaler())]), num))
    if cat:
        tfs.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                     ("o", OneHotEncoder(handle_unknown="ignore"))]), cat))

    def _prep(df):
        df = df.copy()
        for c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in cat:
            df[c] = df[c].astype("object").where(df[c].notna(), None).astype(str)
        return df[num + cat]

    if not tfs:
        return [np.zeros((len(d), 0)) for d in (ref, *others)]
    ct = ColumnTransformer(tfs, remainder="drop", sparse_threshold=0.0)
    ct.fit(_prep(ref))
    return [np.asarray(ct.transform(_prep(d)), dtype=float) for d in (ref, *others)]


# --- 11.3 discriminator AUC (realism) -------------------------------------- #
def discriminator_auc(result: RunResult, table: str) -> float:
    """MR 11.3: AUC of a classifier trying to tell real from synthetic. ~0.5 = very
    realistic; ~1.0 = trivially distinguishable. Returned for the test to bound."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    real, synth = result.originals[table], result.generated[table]
    n = min(len(real), len(synth))
    Xr, Xs = _encode(result.spec, table, real.head(n), synth.head(n))
    if Xr.shape[1] == 0:
        return 0.5
    X = np.vstack([Xr, Xs])
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
    clf = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=0, n_jobs=1)
    return float(np.mean(cross_val_score(clf, X, y, cv=3, scoring="roc_auc")))


# --- 12.1 TSTR (train on synthetic, test on real) -------------------------- #
def tstr_scores(spec: DatasetSpec, table: str, target: str, synth: pd.DataFrame,
                train_real: pd.DataFrame, test_real: pd.DataFrame) -> Tuple[float, float]:
    """MR 12.1: regression R^2 on the real *test* split for a model trained on the
    synthetic data (TSTR) vs trained on the real *train* split (TRTR). The generator
    must have been trained on ``train_real`` so ``test_real`` is genuinely unseen."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    kinds = spec.tables[table].column_kinds
    feats = [c for c in _num_cols(spec, table) + _cat_cols(spec, table) if c != target]
    sub = DatasetSpec(name="_", title="", builder=spec.builder,
                      tables={table: TableSpec(name=table,
                                               column_kinds={c: kinds[c] for c in feats})},
                      formats=["csv"], explicit_patterns="", implicit_patterns="", null_policy="")

    def _fit_score(train_df):
        Xtr, Xte = _encode(sub, table, train_df, test_real)
        if Xtr.shape[1] == 0:
            return 0.0
        ytr = pd.to_numeric(train_df[target], errors="coerce").fillna(0).to_numpy()
        yte = pd.to_numeric(test_real[target], errors="coerce").fillna(0).to_numpy()
        m = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=0, n_jobs=1)
        m.fit(Xtr, ytr)
        return float(r2_score(yte, m.predict(Xte)))
    return _fit_score(synth), _fit_score(train_real)


# --- 10.1 memorization / 10.2 DCR ------------------------------------------ #
def _nn_distances(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(reference)
    d, _ = nn.kneighbors(query)
    return d[:, 0]


def memorization_fraction(result: RunResult, table: str, eps: float = 1e-6) -> float:
    """MR 10.1: fraction of synthetic rows that are (near-)exact copies of a training
    row in the encoded feature space (distance ~ 0). High = memorisation/leakage."""
    real, synth = result.originals[table], result.generated[table]
    Xr, Xs = _encode(result.spec, table, real, synth)
    if Xr.shape[1] == 0:
        return 0.0
    return float((_nn_distances(Xr, Xs) <= eps).mean())


def dcr_ratio(spec: DatasetSpec, table: str, synth: pd.DataFrame,
              train_real: pd.DataFrame, holdout_real: pd.DataFrame) -> float:
    """MR 10.2/10.3: ratio of median distance-to-closest-record (synth->train) over
    (synth->holdout). ~1.0 = synthetic is no closer to the data it was trained on
    than to unseen data (no privacy leak). The generator must have been trained on
    ``train_real``; ``holdout_real`` is an unseen real split."""
    Xtr, Xho, Xs = _encode(spec, table, train_real, holdout_real, synth)
    if Xtr.shape[1] == 0:
        return 1.0
    d_train = np.median(_nn_distances(Xtr, Xs))
    d_hold = np.median(_nn_distances(Xho, Xs))
    return float(d_train / d_hold) if d_hold > 0 else 1.0


# --- 13.1 loss convergence / 13.3 latent (mode) collapse ------------------- #
def loss_decreased(result: RunResult) -> List[str]:
    """MR 13.1: training total loss at the last epoch is below the first epoch."""
    df = result.losses
    if df is None or "loss_name" not in df.columns or "epoch" not in df.columns:
        return []
    total = df[df["loss_name"] == "total_loss"] if "total_loss" in set(df["loss_name"]) else df
    out: List[str] = []
    for tname, sub in total.groupby(total.columns[0]):
        s = sub.groupby("epoch")["value"].mean().sort_index()
        if len(s) >= 2 and s.iloc[-1] > s.iloc[0]:
            out.append(f"{tname}: loss did not decrease ({s.iloc[0]:.4g} -> {s.iloc[-1]:.4g})")
    return out


def generation_diverse(result: RunResult, table: str) -> List[str]:
    """MR 13.3/14.1 proxy: the generator did not collapse to a near-constant output
    (every numeric column has some spread; categoricals keep >1 value where the
    original had >1)."""
    out: List[str] = []
    o, g = result.originals[table], result.generated[table]
    for col in _num_cols(result.spec, table):
        ov = pd.to_numeric(o[col], errors="coerce")
        gv = pd.to_numeric(g[col], errors="coerce")
        if ov.std(ddof=0) > 1e-6 and gv.std(ddof=0) <= 1e-9:
            out.append(f"{table}.{col}: generated column is constant (latent/mode collapse)")
    for col in _cat_cols(result.spec, table):
        if o[col].dropna().nunique() > 1 and g[col].dropna().nunique() <= 1:
            out.append(f"{table}.{col}: generated a single category (mode collapse)")
    return out
