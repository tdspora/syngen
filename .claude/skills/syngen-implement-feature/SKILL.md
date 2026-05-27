---
name: syngen-implement-feature
description: Use for implementing production-grade Syngen features or non-trivial code changes.
---

# Syngen Feature Implementation

## Vocabulary

Use the names defined in `docs/agent-harness/glossary.md` (business / product, plus the metadata-YAML keys). If a draft uses an undefined term, add it to the glossary in the same change rather than inventing a synonym.

## Workflow

0. **Bootstrap check.** If pytest fails to collect or any `python -m …` command reports `ModuleNotFoundError`, invoke the `syngen-bootstrap` skill before continuing. The harness fixed a structural collection break on 2026-05-27 by adding `src/tests/__init__.py` and a `[tool.pytest.ini_options]` block in `pyproject.toml`; runtime deps from `requirements.txt` may still be missing locally.
1. **Frame the change.** Restate the requested change and identify affected workflows: [[training run]], [[inference run]], [[metadata YAML]], [[reports]].
2. **Research.** Inspect relevant source under `src/syngen/`, tests under `src/tests/unit/`, examples in `examples/`, demo notebooks, CI configuration.
3. **Identify public-surface impact.** [[CLI entry points]] (`train`, `infer`, `syngen` — note these are the binary names; `launch_train` / `launch_infer` are the function names), [[SDK]] (`Syngen` class in `src/syngen/sdk.py`), [[metadata YAML]] schema, or serialization format. Any break is an [[escalation trigger]].
4. **Downstream impact.** Every public-surface change propagates to **[[tdm_syngen]]** (enterprise edition) and external PyPI users. Flag and escalate per CLAUDE.md "Approval contract" before implementing.
5. **Plan.** Draft a minimal implementation plan listing files to touch, schemas/models to add or update, tests to add (`src/tests/unit/`, plus integration tests if CLI / file I/O / metadata execution is affected).
6. **Implement.** Smallest viable change aligned with existing patterns in `syngen.ml`.
7. **Version bump.** Update `src/syngen/VERSION` — semver patch by default; minor when public API or CLI changes; major requires `APPROVAL REQUIRED:` and explicit `approved` per the contract.
8. **Test.** Per `testing-contract.md`:
   - Unit test for every new branch / validator rule.
   - Positive and negative [[validation schema]] tests for every marshmallow schema change.
   - [[Regression test]] for every bug fix.
9. **Validate locally**, all from the repo root (`cd "$(git rev-parse --show-toplevel)"`):
   ```bash
   PY="${REPO_ROOT}/.venv/bin/python"; test -x "${PY}" || PY="$(command -v python)"
   "${PY}" -m pytest src/tests/unit/<touched_subpath> -q
   "${PY}" -m flake8 src/syngen/<touched_subpath>
   "${PY}" -m pytest src/tests/unit/ -q          # full unit suite before handoff
   ```
   If any reports `ModuleNotFoundError`, run `syngen-bootstrap` once and retry.
10. **Subagents.** Invoke `syngen-code-reviewer` for non-trivial code changes; `syngen-test-engineer` if test coverage needs review; `syngen-security-reviewer` for any change touching encryption, credentials, data flow, or logging.
11. **Handoff.** State: changed files, tests run, lint results, downstream-impact assessment, documentation updated (glossary, README, examples).
