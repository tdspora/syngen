# Project Architecture Rules

- Package code lives under `src/syngen`.
- Tests live under `src/tests/` (unit under `src/tests/unit/`).
- CLI entry points: `launch_train`, `launch_infer`, `Syngen` SDK (`src/syngen/sdk.py`).
- ML components live in `src/syngen/ml/`: VAE models, strategies, workers, processors, data_loaders, reporters, metrics, validation_schema.
- Metadata behavior must be verified against `examples/`, README guidance, and validation schemas in `syngen/ml/validation_schema/`.
- Generated outputs (`model_artifacts/`, `tmp_store/`, reports) must not be edited as source files.
- Avoid broad refactoring unless required by the task.
- Before changing public behavior, inspect README, CLI definitions, tests, and validation schemas.
- For any runtime generation change, consider both single-table and metadata-driven (multi-table) flows.
- Any public API, CLI flag, or SDK method change propagates to `tdm_syngen` — treat as high-risk.
