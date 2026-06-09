import numpy as np
import pandas as pd
import json

ORIG = "examples/example-data/housing.csv"
SEEDS = {
    1: "/tmp/latent_analysis/pt_seed1.csv",
    2: "/tmp/latent_analysis/pt_seed2.csv",
    3: "/tmp/latent_analysis/pt_seed3.csv",
}
# the original report run output too (random/no fixed seed)
SEEDS["report"] = "model_artifacts/tmp_store/housing-pt/merged_infer_housing-pt.csv"

orig = pd.read_csv(ORIG)
COLS = ["latitude", "longitude"]

out = {"original": {}, "pytorch_runs": {}}
for col in COLS:
    o = orig[col].dropna().astype(float)
    out["original"][col] = {"mean": round(o.mean(), 4), "std": round(o.std(), 4)}

print(f"{'col':<11}{'run':<10}{'mean':>12}{'std':>12}")
print("-" * 45)
for col in COLS:
    o = orig[col].dropna().astype(float)
    print(f"{col:<11}{'ORIGINAL':<10}{o.mean():>12.4f}{o.std():>12.4f}")
    stds = []
    for tag, path in SEEDS.items():
        s = pd.read_csv(path)[col].dropna().astype(float)
        print(f"{col:<11}{('seed='+str(tag)):<10}{s.mean():>12.4f}{s.std():>12.4f}")
        out["pytorch_runs"].setdefault(col, {})[str(tag)] = {
            "mean": round(s.mean(), 4), "std": round(s.std(), 4)
        }
        stds.append(s.std())
    stds = np.array(stds)
    print(f"{col:<11}{'--> std across runs':<10}  mean={stds.mean():.4f}  "
          f"sd={stds.std():.4f}  (original std={o.std():.4f}, "
          f"retention={100*stds.mean()/o.std():.1f}%)")
    out["pytorch_runs"][col]["_summary"] = {
        "std_mean_across_runs": round(float(stds.mean()), 4),
        "std_sd_across_runs": round(float(stds.std()), 4),
        "original_std": round(float(o.std()), 4),
        "std_retention_pct": round(float(100 * stds.mean() / o.std()), 1),
    }
    print()

with open("/tmp/latent_analysis/step1_seed_variance.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved -> /tmp/latent_analysis/step1_seed_variance.json")
