# PyTorch Migration — Findings Checkpoint

**Date:** 2026-06-09
**Branches:** TF baseline = `main` · PyTorch = `tf-to-pytorch-migration`
**Dataset under test:** `examples/example-data/housing.csv` (20,640 rows, 10 columns: 8 numeric, 1 categorical `ocean_proximity`, `total_bedrooms` ~1% NaN)
**Default config:** epochs=10, batch_size=32, latent_dim=10, intermediate_dim=128, BGM `latent_components`=min(30, latent_dim·2), lr = `1e-4·√(batch/32)` = **1e-4**, Adam.
**Determinism:** training is hard-seeded (`_TRAIN_SEED=42`, `wrappers.py`), so weights + BGM fit are deterministic; only generation (`infer --random_seed`) is random.

**Environments used**
| Purpose | venv | key dep |
|---|---|---|
| TF baseline (`main`) | `venv-tf-main` | tensorflow 2.15.1 |
| PyTorch (`tf-to-pytorch-migration`) | `venv-pt` | torch 2.12.0 |

> Scripts, JSON results and logs for every number below are committed under
> `pytorch_migration/analysis/` (see **Reproduction** at the end). The large seeded-generation
> CSVs are regenerable and were intentionally left out of the repo.

---

## 0. Executive summary

1. **KL loss:** computed and logged in both backends but added to the optimized loss with **weight 0**. The non-zero `kl_loss` in the logs (e.g. ~45.6) is **report-only**; it does not affect training. This is intended and matches TF. **KL is not involved in any issue below.**
2. **Overall accuracy parity (housing, 10 epochs):** PyTorch matches TF on most metrics; slightly better univariate similarity, slightly worse on joint-structure metrics (clustering, multiclass utility).
3. **Concrete defect found:** PyTorch synthetic **latitude/longitude distributions are far too narrow** (std retention ~40–50% vs TF's ~87–91%). Visible on the report as a generated curve that doesn't follow the real bimodal shape.
4. **Root cause (localized by experiment):** **decoder under-fitting of hard continuous features at 10 epochs** — NOT the BGM, NOT KL, NOT generation sampling. Continuous features use a point-estimate `Linear→ReLU→Linear(1)` head + **MSE** on MinMax-scaled targets, which mean-regresses on weakly-predictable / multimodal columns.
5. **Validated fix:** **raise lr 1e-4 → ~5e-4**. Recovers 92–99% spread at the same 10 epochs and same wall-time, with lower final loss. `lr×10` overshoots; switching BatchNorm to `train()` mode is worse.
6. **Deeper/structural fixes proposed (not yet built):** distributional/**quantile output head** for continuous features, **multi-task loss balancing** across feature types, and an **lr schedule** for long runs.

---

## 1. KL loss question (answered)

- `kl_divergence(mu, log_sigma)` is computed every step and logged/reported (`kl_loss` metric, losses CSV, MLflow).
- In the optimized objective it is multiplied by **0**:
  `loss = recon_total + 0.0 * kl_loss` (`wrappers.py` train step).
- Evidence from a real run (PyTorch, epoch 9): `kl_loss = 45.64` while `total_loss = 0.0634` (= reconstruction only).
- **Both TF and PyTorch use KL weight 0.** So re-enabling KL is a separate design decision; it is *not* the cause of the distribution issues documented here.

---

## 2. TF vs PyTorch — accuracy parity (housing, 10 epochs, `--reports accuracy`)

| Metric | TF (main) | PyTorch | Δ | Better |
|---|---:|---:|---:|:--:|
| Final total loss (epoch 10) | 0.0402 | 0.0634 | +0.023 | TF |
| Median accuracy (univariate similarity) | 0.9281 | 0.9390 | +0.011 | PT |
| Mean correlation difference | 0.0031 | 0.0095 | +0.0064 | TF |
| Clusters homogeneity | 0.8239 | 0.7377 | −0.086 | TF |
| Utility — multiclass (`ocean_proximity`) | 0.5796 | 0.4779 | −0.102 | TF |
| Utility — regression (`households`) | 0.0 | 0.0 | 0 | tie |

**Read:** same regime, no collapse/divergence. PyTorch slightly better on marginals, slightly worse on joint structure (clustering / multiclass utility) — consistent with the lat/lon spread defect in §3. Regression utility is 0 in **both** → that weakness is config/dataset-related (10 epochs), not a migration regression.

> Caveat: single stochastic run each; different RNG backends. ±0.05–0.10 on clustering/utility is plausibly within run-to-run noise. For decision-grade numbers, average 3–5 seeded runs.

---

## 3. Latitude / longitude distribution collapse — investigation

### 3.1 Observation (confirmed numerically)

| Column | Original std | TF std (gen) | PyTorch std (gen) | TF retention | PT retention |
|---|---:|---:|---:|---:|---:|
| latitude | 2.136 | 1.857 | 1.081 | 87% | **51%** |
| longitude | 2.004 | 1.814 | 0.801 | 91% | **40%** |

Full descriptive stats:

| stat | ORIG lat | TF lat | PT lat | ORIG lon | TF lon | PT lon |
|---|---:|---:|---:|---:|---:|---:|
| mean | 35.632 | 35.057 | 35.398 | −119.570 | −118.760 | −119.913 |
| std | 2.136 | 1.857 | 1.105 | 2.004 | 1.814 | 0.761 |
| skew | 0.466 | 0.863 | 0.388 | −0.298 | 0.134 | −0.224 |
| kurtosis | −1.118 | 1.273 | 0.234 | −1.330 | −0.089 | 0.816 |

Distribution distance vs original (lower = closer):

| metric | TF lat | PT lat | TF lon | PT lon |
|---|---:|---:|---:|---:|
| KS statistic | 0.250 | 0.417 | 0.211 | 0.471 |
| Wasserstein | 0.669 | 1.062 | 0.857 | 1.345 |
| Jensen-Shannon | 0.496 | 0.691 | 0.457 | 0.735 |

**Interpretation:** means/medians match → **not** a location bias; it is a **variance/spread collapse**. Real lat/lon are bimodal (negative kurtosis ≈ −1.1…−1.3 = Northern vs Southern California). PyTorch flattens the two modes into a single narrow central peak (positive kurtosis). TF preserves most of the spread.

### 3.2 Step 1 — Is it systematic? **YES**

Ran `infer` 3× with seeds 1/2/3 on the fixed trained model:

| Column | PT std mean (3 seeds) | run-to-run sd | retention |
|---|---:|---:|---:|
| latitude | 1.081 | 0.0022 | 50.6% |
| longitude | 0.801 | 0.0065 | 40.0% |

Generation seed has ~zero effect → the collapse is a deterministic property of the trained model, not sampling noise.

### 3.3 Step 2 — Is the BGM (latent sampler) collapsed? **NO**

| | PyTorch | TensorFlow |
|---|---:|---:|
| effective components (weight > 0.01) | 10 / 10 | 10 / 10 |
| weight entropy ratio (1.0 = uniform) | 0.957 | 0.926 |
| mean sampled latent std (all dims) | 1.496 | 1.396 |

The BGM is healthy in both; PyTorch's latent space is **wider** than TF's. So the problem is **downstream of the latent space**.

### 3.4 Step 3 — Where is the spread lost? **IN THE DECODER**

Encoded the real data and measured lat/lon std at each stage (PyTorch):

| Stage | latitude std | longitude std |
|---|---:|---:|
| ORIGINAL | 2.136 | 2.004 |
| (A) decode(mu) — reconstruct real latents | 1.078 (50%) | 0.785 (39%) |
| (B) decode(mu+σ·ε) — reparam reconstruction | 1.051 (49%) | 0.772 (39%) |
| (C) BGM-sample → decode — generation | 1.071 (50%) | 0.791 (39%) |

- encoded mu std/dim mean = 1.497; BGM sampled std = 1.495 → latent spread fine.
- posterior σ ≈ 0.043 → near-deterministic encoder (expected with KL off).
- Even feeding the decoder the **real data's own latents** (A) recovers only ~50/39%.
  ⇒ pure **decoder mean-regression**, independent of BGM/KL/sampling.

### 3.5 Root cause

Continuous features decode via a point-estimate head `Linear→ReLU→Linear(1)` (no final activation) trained with **MSE** on **MinMax-scaled** targets (`features.py`). The MSE-optimal output is the *conditional mean*; for targets that are weakly predictable from the other columns and **multimodal** (lat/lon), that mean sits between the modes and shrinks variance.

Confirmed by per-feature loss at the final epoch (10 ep) — lat/lon are the **two worst-fit features**:

```
latitude_loss_num  = 0.0143   ← worst
longitude_loss_num = 0.0119   ← 2nd worst
(other numeric features:  0.0009 .. 0.0099)
numeric_loss = 0.0597 | categorical_loss = 0.0038 | text_loss = 0.0 | kl_loss = 45.64 (×0) | total = 0.0634
```

Scaler chosen for lat/lon = **MinMaxScaler** (bimodal, non-normal, kurtosis below the QuantileTransformer threshold of 50).

### 3.6 Step 3b — TF reconstruction diagnostic (the comparison)

Same encode→decode test on the TF model:

| Stage | TF latitude | TF longitude | PT latitude | PT longitude |
|---|---:|---:|---:|---:|
| decode(mu) reconstruction | **86%** | **91%** | 50% | 39% |
| generation (BGM→decode) | 85% | 89% | 50% | 39% |

Same architecture, BatchNorm, **eval-mode training** (verified: TF's loop calls `self.model(batch)` *without* `training=True` → Keras inference mode; PyTorch replicates with `eval()`), same MSE, same lr formula, same seed. **Yet the TF decoder fits lat/lon and the PyTorch decoder under-fits them in 10 epochs.** ⇒ a per-epoch **convergence-speed** gap, not architecture.

---

## 4. Solution experiments (housing, lat/lon std retention)

| Config | epochs | lr | lat recon | lon recon | lat gen | lon gen | final loss | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline PT | 10 | 1e-4 | 50% | 39% | 50% | 39% | 0.063 | collapsed |
| **TF reference** | 10 | 1e-4 | 86% | 91% | 85% | 89% | — | good |
| PT 50 epochs | 50 | 1e-4 | 85% | 82% | 83% | 78% | — | works (5× time) |
| PT `train()` BN mode | 10 | 1e-4 | 32% | 39% | 34% | 39% | 1.62 | **worse** |
| **PT lr×5 (5e-4)** | 10 | 5e-4 | 99% | 93% | 95% | 92% | 0.029 | **best** |
| PT lr×10 (1e-3) | 10 | 1e-3 | 72% | 62% | 72% | 61% | 0.079 | overshoots |
| PT lr×5 (5e-4) | 50 | 5e-4 | 97% | 90% | 92% | 87% | 0.0079 | stable, no extra spread |

### 4.1 Recommended quick fix — APPLIED

Raise the PyTorch learning rate ~5× — one line, `wrappers.py` (`__create_optimizer`).
**This change has been committed on the `tf-to-pytorch-migration` branch:**

```python
# was: learning_rate = 1e-04 * np.sqrt(self.batch_size / BATCH_SIZE_DEFAULT)
learning_rate = 5e-04 * np.sqrt(self.batch_size / BATCH_SIZE_DEFAULT)
```

Recovers 92–99% of the lat/lon spread at the same 10 epochs / same wall-time, with **lower** final loss than baseline.

**Caveats before changing the default:**
- It is a **global** change (every table). Only validated on `housing` so far — must be checked on categorical-heavy, date, PK/FK, and larger datasets for stability (NaN / overshoot). `lr×10` already overshoots, so margin matters.
- The *reason* the identical lr=1e-4 converges so differently between Keras-Adam and Torch-Adam is still a hypothesis (optimizer-default / gradient-scale difference; see Open Questions).

### 4.2 Do we need to lower lr as epochs increase?

Tested lr×5 at 10 vs 50 epochs:
- **No divergence/blow-up** at 50 epochs (stayed 87–97%, never overshot >100%); bounded by Adam + early stopping.
- A constant high lr over many epochs **oscillates** around the minimum (loss bumped ep14 0.022 → ep19 0.054 → recovered) and gave **no extra benefit** for lat/lon (slightly worse than 10-epoch).
- There is **no lr scheduler**; early stopping (patience=10, min_delta=0.005, restores best-loss weights) is the only guard. **Note:** early stopping watches *total loss*, not lat/lon spread — best-loss weights ≠ best-spread weights.
- **Conclusion:** not strictly required for safety; but if training long, pair the higher lr with a **decay schedule** rather than a constant high lr. With the 10-epoch default, lr×5 alone suffices.

---

## 5. Deeper / structural solutions (proposed, not yet built)

### 5.1 Distributional / quantile output head for continuous features

**Problem it solves:** the MSE point head can only emit one value (the conditional mean) and therefore *cannot represent* "this column could be here OR there." On multimodal/uncertain columns (lat/lon, multi-cluster prices, bimodal ages) it averages the modes → variance collapse. This is the structural cause behind §3.5; lr×5 only mitigates it by training the means to spread more, and it can re-tighten on long runs.

**What it is:**
- *Quantile regression head:* output K quantiles (e.g. 10th…90th), train with **pinball/quantile loss**; at generation sample a random quantile level. Recovers full spread **and** shape.
- *Mixture Density Network:* output a small Gaussian mixture (means/vars/weights), train by likelihood, sample at generation. Represents multimodality directly (one component per hump).

**Trade-offs:** touches head + loss + generation/inverse-transform paths; changes saved-model format; needs tuning (#quantiles or #components; monotonicity constraint to avoid quantile crossing). Win concentrated on hard multimodal columns; sharply-determined columns are already fine with MSE.

### 5.2 Multi-task loss balancing across feature types

**Problem it solves:** the objective is a **sum of per-feature losses on incomparable scales** — numeric MSE on [0,1] is tiny (~0.001–0.06), categorical cross-entropy is medium (~0.5–3, grows with #classes), text/sequence CE is large. Whichever term is largest dominates the gradient; the quiet features (numeric) get starved of gradient and **under-train** — the same mechanism that produced the lat/lon collapse, and the reason "numeric loss small while categorical/text loss big" happens.

**Options (ranked):**
1. **Per-feature loss weights** (static). Infra already exists (`loss_weight` / `weight_randomizer`, default 1) — up-weight numeric. Cheap; dataset-dependent tuning.
2. **Loss normalization** — divide each feature loss by a running estimate of its scale (or by output dims / #classes) so all contribute equally regardless of raw scale. Robust, dataset-agnostic. *Best effort/payoff.*
3. **Uncertainty weighting (Kendall & Gal)** — learnable per-task log-variance `s_i`: `Σ exp(−s_i)·L_i + s_i`. Auto-balances; no manual weights. Few extra params.
4. **GradNorm / PCGrad** — balance by gradient norms / project conflicting gradients. Most powerful, most complex.

**Caveat:** balance is about **gradient share**, not printed loss magnitudes — measure per-task gradient contributions before tuning. Prefer equal-contribution methods (2/3) over aggressive manual weights so categorical/text don't degrade.

### 5.3 Learning-rate schedule

Add warmup + decay (cosine) or `ReduceLROnPlateau` so high lr early (fast convergence) decays later (stable settling), decoupling lr from the epoch budget. Pairs with §4 to make long runs safe.

---

## 6. Open questions

1. **Why does identical lr=1e-4 converge ~5× slower in Torch-Adam than Keras-Adam?** Hypotheses: Adam epsilon default (Keras 1e-7 vs Torch 1e-8), bias-correction differences, or per-feature loss **reduction** (sum vs mean) changing effective gradient scale. → confirm and pick a *principled* lr, not an empirical ×5.
2. **Cross-dataset validation of lr×5:** does it stay stable on categorical-heavy, date, PK/FK, large (>100k row) tables? Any NaNs/overshoot?
3. **Joint-structure metrics:** is the clustering/multiclass-utility gap (§2) real or run-to-run noise? → 3–5 seeded runs per backend, compare means.
4. **Early stopping criterion:** should it track a distribution-fidelity metric (or per-feature spread) rather than total loss, given best-loss ≠ best-spread?
5. **Quantile vs MDN head:** which gives better fidelity/stability on multimodal numeric columns at the default budget, and what is the model-format/back-compat cost?
6. **Loss balancing method choice:** static weights vs normalization vs uncertainty weighting — which best fixes "numeric under-trained" without hurting categorical/text? Measure gradient contributions first.
7. **Regression utility = 0 in both backends** — is this expected at 10 epochs, or a deeper representational limit for continuous targets?

---

## 7. Reproduction

All commands run from repo root. Switch branch + matching venv before each backend.

```bash
# --- TF baseline ---
git checkout main
python3 -m venv venv-tf-main && venv-tf-main/bin/pip install -e .   # tensorflow 2.15
venv-tf-main/bin/train --source ./examples/example-data/housing.csv \
    --table_name housing_tf --epochs 10 --reports accuracy

# --- PyTorch ---
git checkout tf-to-pytorch-migration
python3 -m venv venv-pt && venv-pt/bin/pip install -e .             # torch 2.x
venv-pt/bin/train --source ./examples/example-data/housing.csv \
    --table_name housing_pt --epochs 10 --reports accuracy

# seeded generation variance (Step 1)
venv-pt/bin/infer --table_name housing_pt --size 20640 --random_seed 1 --reports none
```

**Committed analysis artifacts (`pytorch_migration/analysis/`):**

| file | purpose |
| --- | --- |
| `scripts/dist_compare.py` | descriptive stats + KS/Wasserstein/JSD for lat/lon (orig vs TF vs PT) |
| `scripts/step1_seed_variance.py` + `results/step1_seed_variance.json` | 3-seed generation variance |
| `scripts/step2_bgm_inspect.py` + `results/step2_bgm_inspect.json` | BGM component/weight/latent-std comparison |
| `scripts/step3_encode_decode.py` + `results/step3_encode_decode.json` | PyTorch encode→decode reconstruction vs generation |
| `scripts/step3b_tf_reconstruction.py` + `results/step3b_tf_reconstruction.json` | same diagnostic on the TF model |
| `results/step4_solutions.json` | summary of all fix experiments |
| `scripts/measure.py` | reusable: `python measure.py <checkpoints_dir>` → lat/lon recon+gen std |
| `logs/train_e50.log`, `train_tm.log`, `train_lr{5,10}.log`, `train_lr5_e50.log` | experiment logs |

To reproduce the lr / BatchNorm-mode experiments in §4, the runs used temporary env-gated
toggles (`SYNGEN_LR_MULT`, `SYNGEN_TRAIN_MODE`) in `__create_optimizer` / `_train`; these were
scratch and are not part of the committed code. The seeded generations (`pt_seed{1,2,3}.csv`)
are regenerable via the `infer --random_seed` command above.

---

## 8. Key code references (PyTorch branch)

| What | Location |
|---|---|
| KL computed, weight 0 | `src/syngen/ml/vae/wrappers/wrappers.py` train step (`loss = recon_total + 0.0*kl_loss`) |
| `kl_divergence` | `src/syngen/ml/vae/models/model.py` |
| Training in `eval()` mode | `src/syngen/ml/vae/wrappers/wrappers.py` `_train` |
| Learning rate | `wrappers.py` `__create_optimizer` (`1e-4·√(batch/32)`) |
| Early stopping | `wrappers.py` `_train` (patience=10, min_delta=0.005) |
| Continuous head + MSE | `src/syngen/ml/vae/models/features.py` `ContinuousFeature` / `_numeric_decoder_head` |
| Scaler selection | `features.py` `_select_scaler` (KURTOSIS_THRESHOLD=50) |
| BGM fit on `mu` only | `model.py` `fit_sampler` |
| Decoder / heads | `model.py` `CVAEModule` |
