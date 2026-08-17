# Security and Secrets Rules

- Never commit secrets or local credentials.
- Never read `.env`, `.env.*`, local MCP config, cloud credential files, or private token files unless explicitly authorized.
- Never inline Fernet keys, AWS keys, Azure keys, PyPI tokens, GitHub tokens, or other credentials.
- Treat all credential configuration values as environment variable names.
- Keep generated synthetic data, reports, trained model artifacts, and temporary stores out of source control unless they are small, non-sensitive test fixtures.
- Logs must be data-free at all levels; validate this claim when logging code changes.
- If a change touches encryption, credential handling, data loading, or generated artifacts, invoke the `syngen-security-reviewer` subagent.
