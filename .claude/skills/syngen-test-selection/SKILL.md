---
name: syngen-test-selection
description: Use for selecting the correct test, lint, CI, or packaging command for a given Syngen change.
---

# Syngen Test Selection

## Pre-flight

If you have not run `syngen-bootstrap` yet in this session, do so first. The structural pytest-collection break that historically affected this repo is fixed (2026-05-27) but runtime imports (`exrex`, `reportportal_client`, etc.) may still be missing locally and would show up as conftest-level collection failures.

## Fast path decision tree

All commands assume you are at the repo root (`cd "$(git rev-parse --show-toplevel)"`) and use `${PY}` resolved as in `syngen-bootstrap` (project venv if present, else system Python).

| Change type | Command |
|---|---|
| Documentation only | No runtime tests required |
| [[validation schema]] only | `${PY} -m pytest src/tests/unit/validation_schema/ -q` |
| Python utility change | `${PY} -m pytest src/tests/unit/<path> -q` + `${PY} -m flake8 src/syngen/<path>` |
| [[CLI entry points]] / [[SDK]] change | Targeted unit tests + **downstream impact check on [[tdm_syngen]]** before merge |
| File I/O change | Fixture-based unit tests; integration tests if connector behavior changes |
| Packaging change | `${PY} -m build --no-isolation --wheel .` + `${PY} -m twine check dist/*.whl` (both gated by `syngen-bootstrap` confirming `build` and `twine` are installed) |
| Runtime generation change | Targeted unit tests first; [[smoke run]] only when justified |
| `src/syngen/VERSION` bump | Confirm semver matches commit type per `syngen-implement-feature` step 7 |

## Lint commands

```bash
# Fatal errors only (fast)
${PY} -m flake8 src/syngen --count --select=E9,F63,F7,F82 --show-source --statistics

# Project-config style check (uses .flake8 in repo root)
${PY} -m flake8 src/syngen --count --statistics
```

Notes:
- `--max-line-length` and `--max-complexity` are intentionally NOT passed on the command line. The repo's `.flake8` file (and `setup.cfg`) is the source of truth for those values; passing flags would diverge from CI.
- Scope lint to changed paths (`src/syngen/<subpath>`), not the whole repo, to keep feedback fast.

## Escalation

Run the full unit-test suite (`${PY} -m pytest src/tests/unit/ -q`) before final handoff for broad or cross-module changes. Flag any [[public API]] / [[CLI entry points]] / [[validation schema]] change for downstream [[tdm_syngen]] review per CLAUDE.md "Approval contract".
