---
name: syngen-implement-feature
description: Use for implementing production-grade Syngen features or non-trivial code changes.
---

# Syngen Feature Implementation

## Workflow

1. Restate the requested change and identify affected workflows (train, infer, metadata, reports).
2. Inspect relevant source files under `src/syngen/`, tests under `src/tests/unit/`, examples, and CI configuration.
3. Identify public API, CLI, SDK, metadata schema, or serialization impact.
4. **Check downstream impact**: any public interface change propagates to `tdm_syngen` — flag and escalate if needed.
5. Draft a minimal implementation plan aligned with existing patterns in `syngen.ml`.
6. Implement in small, reviewable changes.
7. Update the version in `src/syngen/VERSION` (semver patch bump by default; minor bump when public API or CLI interface changes; ask the user if unsure).
8. Add or update unit tests in `src/tests/unit/`.
9. Add integration tests when CLI, file I/O, or metadata execution behavior changes.
10. Run targeted validation: `python -m pytest src/tests/unit/<path>` and `python -m flake8 src/syngen/<path>`.
11. Invoke `syngen-code-reviewer` and `syngen-test-engineer` for non-trivial changes.
12. Invoke `syngen-security-reviewer` when encryption, credentials, or data flow changes.
13. Provide final handoff: changed files, tests run, results, skipped checks with reason, residual risks.
