import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis

ORIG = "examples/example-data/housing.csv"
TF = "model_artifacts/tmp_store/housing-tf/merged_infer_housing-tf.csv"
PT = "model_artifacts/tmp_store/housing-pt/merged_infer_housing-pt.csv"

orig = pd.read_csv(ORIG)
tf = pd.read_csv(TF)
pt = pd.read_csv(PT)

COLS = ["latitude", "longitude"]


def jsd(a, b, bins=100):
    """Jensen-Shannon distance over a shared histogram grid."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = pa / pa.sum() if pa.sum() else pa
    pb = pb / pb.sum() if pb.sum() else pb
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = (p > 0) & (q > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    return np.sqrt(0.5 * kl(pa, m) + 0.5 * kl(pb, m))


def desc(s):
    s = s.dropna().astype(float)
    return dict(
        n=len(s), mean=s.mean(), std=s.std(), min=s.min(),
        p25=s.quantile(.25), median=s.median(), p75=s.quantile(.75),
        max=s.max(), skew=skew(s), kurt=kurtosis(s),
    )


for col in COLS:
    o = orig[col].dropna().astype(float)
    t = tf[col].dropna().astype(float)
    p = pt[col].dropna().astype(float)
    print("=" * 78)
    print(f"COLUMN: {col}")
    print("=" * 78)
    do, dt, dp = desc(o), desc(t), desc(p)
    rows = ["n", "mean", "std", "min", "p25", "median", "p75", "max", "skew", "kurt"]
    print(f"{'stat':<8}{'ORIGINAL':>14}{'TF':>14}{'PyTorch':>14}"
          f"{'TF err%':>10}{'PT err%':>10}")
    for r in rows:
        ov, tv, pv = do[r], dt[r], dp[r]
        if r == "n":
            print(f"{r:<8}{ov:>14.0f}{tv:>14.0f}{pv:>14.0f}{'':>10}{'':>10}")
            continue
        # relative error vs original (guard tiny denominators)
        denom = abs(ov) if abs(ov) > 1e-9 else 1.0
        te = 100 * (tv - ov) / denom
        pe = 100 * (pv - ov) / denom
        print(f"{r:<8}{ov:>14.4f}{tv:>14.4f}{pv:>14.4f}{te:>10.2f}{pe:>10.2f}")

    print("\n-- distribution distance vs ORIGINAL (lower = closer) --")
    ks_t = ks_2samp(o, t).statistic
    ks_p = ks_2samp(o, p).statistic
    w_t = wasserstein_distance(o, t)
    w_p = wasserstein_distance(o, p)
    j_t = jsd(o.values, t.values)
    j_p = jsd(o.values, p.values)
    print(f"{'metric':<26}{'TF':>14}{'PyTorch':>14}{'winner':>10}")
    print(f"{'KS statistic':<26}{ks_t:>14.4f}{ks_p:>14.4f}"
          f"{('TF' if ks_t < ks_p else 'PT'):>10}")
    print(f"{'Wasserstein (earth-mover)':<26}{w_t:>14.4f}{w_p:>14.4f}"
          f"{('TF' if w_t < w_p else 'PT'):>10}")
    print(f"{'Jensen-Shannon distance':<26}{j_t:>14.4f}{j_p:>14.4f}"
          f"{('TF' if j_t < j_p else 'PT'):>10}")
    print()
