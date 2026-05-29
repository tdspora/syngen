# TF → PyTorch parity harness

A runnable safety net for the TensorFlow → PyTorch migration. It captures the
**current TF generation behavior** as golden per-column statistics, then lets the
PyTorch branch be diffed against it so any **distribution collapse** (e.g. an age
column 18–90 coming out 18–40) fails loudly instead of slipping through because
"the loss went down."

See `docs/migration/tf_to_pytorch_migration_plan.md` for the full acceptance
checklist and the six code-grounded collapse hypotheses this guards against.

## Layout

| File | Purpose |
| --- | --- |
| `make_fixtures.py` | Deterministically (re)generate the fixture CSVs + metadata YAMLs. |
| `fixtures/` | Committed datasets covering numeric / categorical / text / email / date / UUID / PK / UQ / FK, plus a wide mixed-type table (`mixed_complex`, incl. a bimodal column) and a four-table relational chain with multiple FKs (`relations_chain`). |
| `stats.py` | `profile_table()` + `compare_profiles()` — no TF/PyTorch dependency. |
| `runner.py` | Runs the real `launch_train` → `launch_infer` per fixture in an isolated dir. |
| `capture_baseline.py` | Runs the current backend and writes `baselines/<fixture>.json`. |
| `baselines/` | Committed golden statistics (capture these **on TF**). |
| `test_parity.py` | Re-runs, profiles, and asserts no drift vs the baseline. |

## Workflow

### 1. (Re)generate fixtures — only if the matrix changes
```bash
python src/tests/integration/parity/make_fixtures.py
```

### 2. Capture the TF golden baseline (do this on the TF code, then commit)
```bash
cd src/tests/integration/parity
python capture_baseline.py            # all fixtures
python capture_baseline.py numeric_wide categorical   # subset
git add baselines/*.json
```

### 3. Run parity on the PyTorch branch
```bash
pytest src/tests/integration/parity -m parity
```
`test_parity_against_baseline` and `test_determinism` **skip** until a baseline is
committed and a backend is installed, so they never break default CI. The two
self-tests (`test_collapse_is_detected`, `test_category_collapse_is_detected`)
need no backend and always run — they prove the guard itself works.

## Interpreting a failure → which collapse hypothesis

| Failure message | Likely root cause (see migration doc) |
| --- | --- |
| `RANGE COLLAPSE` on a numeric column | KL re-enabled (#1), latent under-dispersion, or missing `eval()` for BatchNorm/Dropout (#3); scaler not applied (#5); wrong eps/sampling (#6). |
| `q0.25/q0.5/q0.75 drift` on `bimodal` (range still OK) | distribution flattened to unimodal — latent posterior collapse / KL pressure (#1); mode-dropping decoder. |
| `uuid uniqueness < …` on `external_uuid` | UUID regeneration path (`generate_uuid`) regressed. |
| `CATEGORY COLLAPSE` / `alien categories` | Feature-order mismatch between transform / decoder / inverse_transform (#4); softmax head wrong size. |
| generator produces garbage everywhere | Decoder weights not shared with the generator, or `vae_generator` state loaded into the wrong module (#2). |
| `non-deterministic under fixed seed` | Unseeded torch/numpy RNG, or train-mode dropout/BN active during generation (#3). |
| `datetime parse success < …` | Date scaler / `inverse_transform` / format restoration regressed. |
| FK validity / uniqueness assertion | PK/UQ restoration (`__make_pk_uq_unique`) or FK KDE path regressed. |

## Tolerances

Defaults live in `stats.Tolerances` (e.g. numeric range coverage ≥ 80 %,
category coverage ≥ 90 %, mean/std/quantile drift ≤ 25 %). Per-column overrides
go in `Tolerances.column_overrides` and are persisted into each baseline JSON so
acceptance thresholds are explicit and reviewable.
