---
paths:
  - "**/*.yaml"
  - "**/*.yml"
---

# Metadata YAML Rules

- Use existing examples in `examples/` and marshmallow schemas in `syngen/ml/validation_schema/` as the source of truth.
- Define `global` defaults only when multiple tables share settings.
- Keep table sections explicit and readable.
- Configure `train_settings` and `infer_settings` separately.
- Validate `keys` definitions carefully:
  - PK and UQ keys do not require references.
  - FK keys require references.
  - Composite keys must preserve column ordering.
- Use `regex_patterns` only for PK/UQ key generation unless the schema explicitly supports broader use.
- Treat `encryption.fernet_key` as an environment variable name — never the key value itself.
- Do not place real credentials, keys, or sensitive source paths in committed metadata.
