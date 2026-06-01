# Metamorphic test suite (open-source `syngen`)

Self-validating metamorphic relations (MRs) over real `launch_train` → `launch_infer`
round-trips, across every supported file format. No golden baseline is captured — each
test asserts an input→output relation true for *any* correct backend, so the suite
proves **transfer consistency** for the TF→PyTorch migration. Full design &
category↔MR mapping: `tmp/metamorphic tests/plan-metamorph-test.md`.

## Run

```bash
# 1. Generate the committed datasets + cards (deterministic; once / on change):
PYTHONPATH=src python -m tests.metamorphic._lib.generate_datasets

# 2a. Fast (loader fidelity only, no model):
pytest src/tests/metamorphic -m "metamorphic and not slow"

# 2b. Full suite (real train->infer; opt-in, slow):
pytest src/tests/metamorphic -m metamorphic
```

## Layout

| Path | What |
| --- | --- |
| `_lib/specs.py` | dataset registry (schema, kinds, keys, fks, formats, pattern docs) |
| `_lib/generate_datasets.py` | deterministic generator → `test-data/` + `dataset-cards/` + `manifest.json` |
| `_lib/runner.py` | format-parameterized train→infer harness (public API) |
| `_lib/stats.py` | **copy** of `src/tests/integration/parity/stats.py` (canonical source — keep in sync) |
| `_lib/mr_asserts.py` | MR helpers on top of `stats.py` |
| `test_io_roundtrip.py` | MR-IO1 loader fidelity (fast, no model) |
| `test_cross_format.py` | MR-IO2 cross-format equivalence, MR-IO3 encryption round-trip |
| `test_contract.py` | MR-C row count / columns / order |
| `test_keys_relations.py` | MR-K PK/UQ/FK / regex |
| `test_distribution.py` | MR-D anti-collapse + null ratio |
| `test_transformations.py` | MR-T permutation / duplication / affine / size / determinism |
| `test-data/`, `dataset-cards/` | generated data + one card per dataset |

## Notes
- Tests are opt-in via the `metamorphic` marker; train→infer tests are also `slow`.
- `pkl` and `.dat` are loader-only (not trainable sources in syngen); see the plan doc.
- The enterprise mirror lives at `tdm_syngen/src/tests/metamorphic/` (adds Parquet/Delta
  and the EE-only datasets/relations).
