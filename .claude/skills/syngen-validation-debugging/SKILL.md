---
name: syngen-validation-debugging
description: Use for marshmallow validation errors, metadata schema failures, and CLI argument validation issues in Syngen.
---

# Syngen Validation Debugging

## Workflow

1. Capture the exact failing metadata YAML or CLI input and the full error message.
2. Locate the responsible marshmallow schema in `syngen/ml/validation_schema/`.
3. Locate existing tests for the same validation area in `src/tests/unit/`.
4. Reproduce the failure with the smallest possible fixture.
5. Fix validation logic without weakening unrelated constraints.
6. Improve error messages when they are unclear or missing context.
7. Add a regression test for the failing path.
8. Run targeted schema tests: `python -m pytest src/tests/unit/<validation_path>`.
9. Confirm positive cases (valid input still passes) and negative cases (invalid input still fails).
