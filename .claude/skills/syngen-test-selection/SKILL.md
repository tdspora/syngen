---
name: syngen-test-selection
description: Use for selecting the correct test, lint, CI, or packaging command for a given Syngen change.
---

# Syngen Test Selection

## Fast path decision tree

| Change type | Command |
|---|---|
| Documentation only | No runtime tests required |
| Schema / validation only | `pytest src/tests/unit/<validation_schema_path>` |
| Python utility change | `pytest src/tests/unit/<path>` + `flake8 src/syngen/<path>` |
| CLI / SDK change | Targeted CLI tests + downstream impact check (tdm_syngen) |
| File I/O change | Fixture-based unit tests |
| Packaging change | `python -m build .` + `python -m twine check dist/*.whl` |
| Runtime generation change | Targeted unit tests first; smoke test only when justified |

## Lint commands

```bash
# Fatal errors only (fast)
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Broader style check
python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

## Escalation

Run full unit test suite (`pytest src/tests/unit/`) before final handoff for broad or cross-module changes. Flag any public API change for downstream `tdm_syngen` review.
