---
name: syngen-harness-update
description: Use when a coding or testing session reveals that a harness file (rule, skill, agent, or doc) is inaccurate, incomplete, or blocking correct work. Requires human approval before any harness file is changed.
---

# Syngen Harness Update Flow

Harness files (`.claude/rules/`, `.claude/skills/`, `.claude/agents/`) are the operating instructions for all agents in this project. They must not be changed autonomously — every update requires explicit human approval.

## Trigger conditions

Invoke this skill when, during a coding or testing session, you encounter any of the following:

- A rule blocks a valid, required action (e.g., a permission is too narrow for the task).
- A skill step is wrong, missing, or leads to a test failure.
- An agent's tool list, constraint, or responsibility description is inaccurate.
- A new recurring pattern emerges that should be encoded as a rule or skill step.

## Workflow

### Step 1 — Identify the issue

State precisely:

- **File**: which harness file is affected (relative path from repo root).
- **Current content**: quote the exact line(s) that are wrong or missing.
- **Issue found during**: which task or test surfaced this (brief description).
- **Impact**: how the current content caused incorrect behavior or blocked progress.

### Step 2 — Propose the change

Provide a concrete diff or replacement text:

```
File: .claude/rules/testing-contract.md

CURRENT:
  - Python utility change: ...

PROPOSED:
  - Python utility change: ... (add specific clarification)

Reason: <why the current text caused a problem>
```

### Step 3 — Stop and present to human

**Do not apply the change yet.** Output the proposal in full and wait for explicit human approval:

```
HARNESS UPDATE PROPOSAL
-----------------------
File: <path>
Issue: <one sentence>
Proposed change: <diff or replacement>

Please review and reply "approved" to apply, or provide corrections.
```

### Step 4 — Apply only after explicit approval

Apply the change **only** after the human replies with clear approval.

Apply the minimal targeted change — do not refactor surrounding content.

### Step 5 — Document the update

After applying, append a one-line entry to `docs/agent-harness/harness-changelog.md` (create if missing):

```
YYYY-MM-DD | <file changed> | <one-line reason> | approved by human
```

### Constraints

- Never apply a harness file change without explicit human approval in this session.
- Never batch multiple harness changes into one approval request — propose each separately.
- Never change `settings.json` autonomously.
- Never remove a rule or constraint without documenting why.
