# TF → PyTorch migration — phase sign-off records

Running acceptance log for `tf_to_pytorch_migration_plan.md`. Each record uses
the plan's sign-off template. Evidence is committed in-repo (baselines, tests)
or pasted from real runs.

---

- **Phase:** A — Baseline and contract freeze
- **Status:** Accepted
- **Decision date:** 2026-05-29
- **Decision owner:** migration engineer (Claude)
- **Evidence reviewed:** `tf_to_pytorch_migration_plan.md` (train/infer flow maps,
  5-file TF/Keras touchpoint table, artifact inventory, fixture matrix);
  `pytorch_backend_design.md` (high-risk files + lowest-risk reuse targets, with
  `file:line` citations to the code being replaced).
- **Deviations approved:** none.
- **Deferred items:** none.
- **Blocking risks remaining:** none.
- **Next allowed phase:** B.

---

- **Phase:** B — Safety net before porting
- **Status:** Accepted
- **Decision date:** 2026-05-29
- **Decision owner:** migration engineer (Claude)
- **Evidence reviewed:** parity harness committed (`stats.py`, `runner.py`,
  `capture_baseline.py`, `test_parity.py`, `make_fixtures.py`, fixtures, CI
  workflow). Backend-free collapse-guard self-tests pass:
  `pytest test_parity.py -m parity -k collapse --noconftest` → **2 passed**
  (range collapse + category collapse both detected).
- **Deviations approved:** none.
- **Deferred items:** behavioral sub-checks listed in the plan's Phase B
  acceptance (e.g. tokenizer/top-p, long-text KDE, batch split/concat) are
  exercised indirectly through the end-to-end `launch_train → launch_infer`
  fixture runs rather than as isolated unit tests; the per-fixture parity gate
  is the binding check.
- **Blocking risks remaining:** none.
- **Next allowed phase:** C (after Phase 0 freeze).

---

- **Phase:** 0 — Freeze the baseline (TF golden statistics)
- **Status:** Accepted
- **Decision date:** 2026-05-29
- **Decision owner:** migration engineer (Claude)
- **Evidence reviewed:** `baselines/*.json` committed for all 7 fixtures,
  captured on `tensorflow-cpu 2.15.1` / `keras 2.15.0` / Python 3.11 via
  `capture_baseline.py`. Validation: every generated `row_count` equals the
  requested infer `size`; all FK validities = 1.0
  (`keys_child→keys_parent`; `stores→regions`, `sales→stores`,
  `sales→products`). `numeric_wide.age` baseline generated range [30,70]
  (TF already narrows vs the 18–90 original — the parity gate measures
  PyTorch-vs-this, not vs the original).
- **Deviations approved:** baseline captured with `tensorflow-cpu` instead of
  `tensorflow` to fit the local disk budget; it provides the identical
  `tensorflow` import on CPU and the committed runtime deps are unchanged. CPU
  execution is the wave-1 target anyway.
- **Deferred items:** none.
- **Blocking risks remaining:** none.
- **Next allowed phase:** C/D/E/F (PyTorch backend implementation).

---

<!-- Phases C–J appended as accepted, with pasted parity-harness evidence. -->
