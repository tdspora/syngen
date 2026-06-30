# Syngen Claude Code Operating Context

## Project facts

- `syngen` is the open-source base library for synthetic tabular data generation.
- Package root: `src/syngen`.
- Supported Python versions: 3.10 and 3.11
- Primary workflows: `train`, `infer`, and YAML metadata-driven execution.
- Console-script entry points (per `setup.cfg [options.entry_points]`): `train` (→ `syngen.train:launch_train`), `infer` (→ `syngen.infer:launch_infer`), `syngen` (→ `syngen:main`). The `launch_train` / `launch_infer` names refer to the *function* objects, not the CLI binaries.
- Python SDK: `Syngen` class in `src/syngen/sdk.py`.
- Metadata validation is schema-driven (marshmallow, `syngen/ml/validation_schema/`); must remain backward compatible.
- **Downstream dependency**: `tdm_syngen` (enterprise edition) depends on this library as a versioned pip package. Any breaking change in public API, CLI flags, SDK methods, or metadata schema propagates to `tdm_syngen`.
- Runtime outputs (`model_artifacts/`, `tmp_store/`, reports, generated inference files) are not source assets.

## Engineering standards

- Implement production code only with corresponding tests.
- Prefer minimal, isolated changes over broad rewrites.
- Preserve public CLI/API/SDK backward compatibility — breakage propagates to `tdm_syngen` and external users.
- Use existing architecture, naming, validation style, logging conventions, and test patterns.
- Add regression tests for every bug fix.
- Add positive and negative tests for every validation-schema change.
- Avoid full TensorFlow-heavy train/infer runs unless runtime generation behavior is affected.

## Security standards

- Never commit secrets, raw customer data, local credentials, generated datasets, or model artifacts.
- Treat `fernet_key` and cloud credential fields as environment variable names — never literal values.
- Do not inspect `.env`, `.env.*`, local secret files, or developer-specific settings unless explicitly authorized.

## Default implementation workflow

1. Inspect relevant source code, tests, examples, and CI definitions.
2. Produce a concise implementation plan.
3. Implement the smallest viable change.
4. Add or update unit tests (`src/tests/unit/`).
5. Add integration tests when CLI, file I/O, metadata execution, or packaging behavior changes.
6. Run the smallest relevant validation command first.
7. Escalate to broader CI-style checks before final handoff.
8. Report changed files, tests run, results, and residual risks.

## Canonical vocabulary

- Business / product terms (for PRDs, user stories, acceptance criteria): `docs/agent-harness/glossary.md`.
- Technical metadata-YAML terms are described in code under `src/syngen/ml/validation_schema/` and in `examples/`. There is no separate technical dictionary file in this repo.
- When a new term appears anywhere in this repository, add it to `docs/agent-harness/glossary.md` in the same change. Cross-link with `[[term]]`; do not re-define.

## Approval contract

Anything an agent is told to "request human approval for" must follow this contract:

1. The agent prints a single line: `APPROVAL REQUIRED: <one-sentence summary of the action>`.
2. The agent stops and yields the turn to the human.
3. The human authorizes by typing exactly one of:
   - `approved` — proceed exactly as proposed.
   - `approved: <free-text constraint>` — proceed, but apply the constraint.
   - `deny` — abort and explain the reason in a brief follow-up.
4. Any other response (silence, `ok`, thumbs-up, emoji, paraphrase) is **not** approval. The agent must re-ask.

This contract applies to every escalation trigger below, every `ask` permission in `.claude/settings.json`, and every harness-update action.

## Escalation triggers (require human approval)

- Any change to public API signatures or CLI flags.
- Dependency version changes.
- Release or publish workflow changes.
- Credential or encryption behavior changes.
- Full train/infer smoke runs on real data.
