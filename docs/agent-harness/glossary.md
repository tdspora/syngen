# Syngen Glossary (Product Owner & Agent Harness)

Single source of truth for business-facing terms a Product Owner or Business Analyst can use in user stories, acceptance criteria, and conversations with delivery teams. `syngen` is the open-source base library; the enterprise-only and rules-generation vocabulary lives in the downstream repositories (`tdm_syngen`, `tdm_rulesgen`).

When a term appears in `[[brackets]]` it is defined elsewhere in this document. Cross-project terms are flagged with their repository.

---

## 1. Product surface

| Term | Definition you can use in requirements |
|---|---|
| **Syngen (base)** | The open-source Python library for synthetic tabular data generation. Distributed as the `syngen` wheel on PyPI; consumed by **[[tdm_syngen]]** as a pinned dependency. |
| **CLI entry points** | Three console scripts shipped by the wheel (per `setup.cfg`): `train`, `infer`, and `syngen`. The "launch" names (`launch_train`, `launch_infer`) refer to the Python functions inside the package, not the CLI binaries. |
| **SDK** | `Syngen` class in `src/syngen/sdk.py`. Programmatic Python interface for embedding the library. |
| **Public API** | The set of names this library guarantees to keep stable for downstream consumers: the three [[CLI entry points]], the [[SDK]] class and its public methods, the [[validation schema]] surface, and the [[metadata YAML]] vocabulary. Any change here is a [[backward-compatibility]] event. |

## 2. Core workflows (the verbs in user stories)

| Term | Definition |
|---|---|
| **[[training run]]** | Execution of the `train` workflow. Inputs: a [[source dataset]] and a [[metadata YAML]]. Output: [[model artifacts]]. Triggered by `train --metadata_path=…`. |
| **[[inference run]]** | Execution of the `infer` workflow. Inputs: [[model artifacts]] + [[metadata YAML]]. Output: a [[synthetic dataset]]. Triggered by `infer --metadata_path=…`. |
| **[[metadata-driven execution]]** | Any workflow driven by a [[metadata YAML]] that describes one or more tables. The non-metadata-driven alternative is direct [[SDK]] invocation with a single-table configuration. |
| **[[smoke run]]** | A short end-to-end execution of `train` + `infer` on a small dataset, used to validate that nothing in the runtime pipeline is broken. Heavy; only run when runtime generation behavior is plausibly affected. |

## 3. Data objects

| Term | Definition |
|---|---|
| **[[source dataset]]** | The original tabular data supplied to a [[training run]]. Never committed; supplied via `train_settings.source`. |
| **[[synthetic dataset]]** | The generated tabular data emitted by an [[inference run]]. Statistically faithful to the [[source dataset]] but contains no real values (modulo [[passthrough columns]]). |
| **[[metadata YAML]]** | The configuration file driving every workflow. Top-level keys: `global` (defaults), one or more table sections, each with `train_settings`, `infer_settings`, `format`, `keys`, `column_types`, `encryption`. The marshmallow [[validation schema]] is the source of truth. |
| **[[model artifacts]]** | Files emitted by a [[training run]] (generator weights, vocabularies, fitted preprocessors). Stored under `model_artifacts/`; **never committed**. |
| **[[generator model]]** | The model architecture used by the library — **VAE** (Variational Autoencoder) is the default. |
| **[[VAE]]** | Variational Autoencoder. The default generator architecture. |
| **[[passthrough columns]]** | Columns copied verbatim from source to synthetic. Treated as sensitive in stories and security review because the original values survive. |
| **[[tmp_store]]** | Per-run scratch space for intermediate files. |
| **[[reports]]** | Output controlled by the `reports` metadata key. Allowed values: `none`, `accuracy`, `sample`, `metrics_only`, `all`. |

## 4. Configuration vocabulary (metadata YAML)

| Term | Definition |
|---|---|
| **`global` section** | Defaults applied to every table section when not overridden. |
| **table section** | A non-`global` top-level YAML key naming an internal table configuration. |
| **`train_settings`** | Source data and model training configuration (path, row limit, drop-null behavior, epochs). |
| **`infer_settings`** | Synthetic data generation configuration (destination, size, random seed). |
| **`format`** | File parser settings: separator, quote character, encoding, header presence, skip rows, Excel sheet handling. |
| **`keys`** | Primary (`PK`), unique (`UQ`), and foreign (`FK`) key declarations. |
| **`regex_patterns`** | Value-generation patterns scoped to [[keys]] (PK/UQ). |
| **`column_types.categorical`** | Force-cast columns to categorical even when the heuristic would not. |
| **`column_types.passthrough`** | Mark columns as [[passthrough columns]]. |
| **[[Fernet key]]** | Symmetric AES-128-CBC + HMAC key. The `encryption.fernet_key` field carries the **env-var name** holding the key — never the value. |

## 5. Validation & quality terms

| Term | Definition |
|---|---|
| **[[validation schema]]** | A marshmallow schema in `src/syngen/ml/validation_schema/` that gates [[metadata YAML]] acceptance. Source of truth for what a metadata YAML may contain. |
| **[[positive test]]** / **[[negative test]]** | For any [[validation schema]] change we require both: a positive test asserting an accepted YAML loads successfully, and a negative test asserting a rejected YAML raises a `ValidationError` with the expected field path. |
| **[[regression test]]** | A test that reproduces a previously reported bug before the fix and passes after. Required for every bug fix. |
| **[[backward compatibility]]** | A change is backward-compatible when: (a) existing [[CLI entry points]] accept the same flags, (b) existing [[SDK]] callers compile and run, (c) existing [[metadata YAML]] files validate against the [[validation schema]] without modification, (d) [[tdm_syngen]] consumer code continues to pass its own tests. A break propagates to [[tdm_syngen]] and external users; treat as an [[escalation trigger]]. |
| **[[acceptance criteria]] format** | Gherkin (`GIVEN…WHEN…THEN…`) for user-facing flows; tabular contracts for [[metadata YAML]] shape changes. |

## 6. Security & privacy terms

| Term | Definition |
|---|---|
| **[[secret hygiene]]** | Three rules: (a) no value in source / commit / log, (b) all credential fields are env-var **names**, (c) `.env*`, `secrets/`, `~/.ssh/**`, `~/.aws/**`, `~/.azure/**`, `~/.pypirc`, `./.git/config` are denied for read by the harness. |
| **[[data-free logging]]** | Logs MUST NOT contain raw dataset values, generated samples, or LLM payloads at any level — including `DEBUG` and `TRACE`. Verified on every logging-touching change. |
| **[[customer-controlled storage]]** | `model_artifacts/`, `tmp_store/`, generated [[reports]]. Never committed; retention defined by the customer. |
| **Test fixture credentials** | Note: `src/tests/conftest.py` historically contained literal Fernet keys for fast test bootstrap. PO awareness: these are short-lived test-only keys, but they are technically a [[secret hygiene]] violation because they are *literal values* in committed code. Move to env-var lookups before next public release. |

## 7. Agent harness terms

| Term | Definition |
|---|---|
| **[[harness]]** | The collection of files under `.claude/` plus `CLAUDE.md` that conditions Claude Code's behavior. |
| **[[CLAUDE.md]]** | Always-loaded project context. Authoritative for project facts and escalation triggers. |
| **[[CLAUDE.local.md]]** | Gitignored developer-local overrides. |
| **[[agent (subagent)]]** | A focused role under `.claude/agents/` with its own model, tools, and prompt. Today: `syngen-code-reviewer`, `syngen-security-reviewer`, `syngen-codebase-researcher`, `syngen-implementation-engineer`, `syngen-test-engineer`. |
| **[[skill]]** | A trigger-based prompt fragment under `.claude/skills/<name>/SKILL.md`. Loaded only when its trigger matches. |
| **[[rule]]** | A persistent constraint under `.claude/rules/` always loaded as context. |
| **[[settings.json]]** | The `allow` / `ask` / `deny` permission matrix for tool calls. |
| **[[escalation trigger]]** | A change category that requires explicit human approval before the agent may proceed. |
| **[[approval text]]** | Literal strings a human types to authorize a paused action: `approved`, `approved: <constraint>`, or `deny`. Anything else is not approval. See CLAUDE.md "Approval contract". |
| **[[harness change]]** | An edit to files under `.claude/` or `docs/agent-harness/`. Requires the `syngen-harness-update` skill + explicit human [[approval text]]. |

## 8. Cross-project family terms

| Term | Definition |
|---|---|
| **[[syngen]]** | This repository. The open-source base library. |
| **[[tdm_syngen]]** | Enterprise edition built on top of `syngen`. Consumes this library as a pinned pip dependency; any [[public API]] break here propagates there. |
| **[[tdm_rulesgen]]** | The Rulesgen FastAPI service. Not used by `syngen`; used by `tdm_syngen` via `column_rules` metadata. |

---

## 9. How to use this glossary in requirements

Recommended PRD/User-story template:

> **As a** *<role>*, **I want** *<feature>* **so that** *<value>*.
>
> **Acceptance criteria** (Gherkin):
> - **GIVEN** a *[[metadata YAML]]* with `train_settings.source=<path>` and `keys.PK=<col>` …
> - **WHEN** I run *`train --metadata_path=<yaml>`* against the *[[source dataset]]* …
> - **THEN** *[[model artifacts]]* are written under `model_artifacts/<table>/` …
> - **AND** the [[validation schema]] accepts the YAML without raising a `ValidationError`.
>
> **Definition of Done**: implementation + tests merged behind a passing CI run; any change to [[public API]] is reflected in the release notes; downstream [[tdm_syngen]] test impact assessed.

Always link out via `[[bracket]]` syntax so reviewers can click through to definitions. If you find a term in a draft that is **not** in this glossary, add it here first or rewrite the draft using an existing term — undefined vocabulary is a leading cause of acceptance-test churn.

---

## 10. Maintenance

- When a new term enters PRDs, JIRA, or design docs, add it here in the same PR.
- Cross-link via `[[term]]` to other sections; do not re-define.
- Any change to this file is a [[harness change]] and follows the `syngen-harness-update` skill flow.
- The downstream glossaries in `tdm_syngen/docs/agent-harness/glossary.md` and `tdm_rulesgen/docs/agent-harness/glossary.md` mirror this structure — keep cross-project terms in sync.
