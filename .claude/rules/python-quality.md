---
paths:
  - "src/**/*.py"
  - "src/tests/**/*.py"
---

# Python Quality Rules

- Follow existing style and module boundaries in `syngen.ml`.
- Prefer explicit validation and clear error messages aligned with existing marshmallow patterns.
- Keep functions focused and testable.
- Add type hints when they improve clarity without disrupting existing style.
- Do not introduce hidden global state.
- Do not swallow exceptions without preserving diagnostic context.
- Add unit tests for new branches, edge cases, and failure modes.
- Run targeted pytest before broader validation.
- Do not introduce new dependencies without explicit approval — changes affect downstream `tdm_syngen` packaging.
