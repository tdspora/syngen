---
name: syngen-bootstrap
description: Use to verify or set up the local Syngen development environment — creates a `.venv` if missing, installs `requirements.txt`, probes runtime imports, and confirms pytest collection works. Invoke this on a fresh checkout, when pytest reports `ModuleNotFoundError`, or when `python -m build / twine / mypy` says the tool is not installed.
---

# Syngen Bootstrap

## Purpose

Guarantee that the local Python environment has every dependency the test suite and quality gates need, and that pytest can collect tests without the structural break this repo had before 2026-05-27.

## Historical context (load-bearing)

- **2026-05-27 — pytest collection break fixed.** `src/tests/__init__.py` was added so `tests/` is importable as a package; `pyproject.toml` now sets `[tool.pytest.ini_options] pythonpath = ["src"]` and `testpaths = ["src/tests"]`. Before that fix, `python -m pytest src/tests/unit` produced 20 `ModuleNotFoundError: No module named 'tests'` errors.
- **2026-05-27 — literal Fernet keys removed from `src/tests/conftest.py`.** `FERNET_KEY` and `FERNET_KEY_2` are now generated per pytest session via `Fernet.generate_key().decode()` and assigned via `os.environ.setdefault(...)`. CI can still inject specific values by setting the env vars before invoking pytest.

If either regression returns, the test suite will silently degrade — restore the fixes before further work.

## Workflow

### Step 0 — repo root

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
```

### Step 1 — locate or create the project venv

The harness prefers a `.venv` at the repo root for parity with the other `tdm-*` repositories.

```bash
VENV_DIR="${REPO_ROOT}/.venv"
PY="${VENV_DIR}/bin/python"
```

**Branch A — venv exists.** Verify the interpreter still works:

```bash
test -x "${PY}" && "${PY}" --version
```

If the version is not 3.10.x or 3.11.x (the supported range), the venv is stale — escalate per the Approval contract before recreating.

**Branch B — venv missing.** Creating a venv is a workspace-state change. Emit:

```
APPROVAL REQUIRED: create .venv at ${REPO_ROOT}/.venv via
    python3.11 -m venv "${REPO_ROOT}/.venv"
    "${REPO_ROOT}/.venv/bin/python" -m pip install --upgrade pip
    "${REPO_ROOT}/.venv/bin/python" -m pip install -r "${REPO_ROOT}/requirements.txt"
    "${REPO_ROOT}/.venv/bin/python" -m pip install -e "${REPO_ROOT}"
    "${REPO_ROOT}/.venv/bin/python" -m pip install build twine
```

Wait for `approved` per CLAUDE.md "Approval contract". After approval, run each command in order and report the result of each. Notes:

- Use `python3.11` (Python 3.10 also works; the project supports both).
- The `pip install --upgrade pip` step is a benign upgrade of pip itself inside the new venv — this is the only `--upgrade` use the harness permits, and it is bounded to the freshly created venv. All other `pip install --upgrade` invocations remain denied by `.claude/settings.json`.
- The `-e .` step installs the package in editable mode so the `train`, `infer`, and `syngen` console scripts are created on the venv's PATH.
- `build` and `twine` are dev-only release tools; install them with the venv since the project does not list them in `requirements.txt`.

**Branch C — user denies venv creation.** Fall back to the system Python:

```bash
PY="$(command -v python)"
"${PY}" --version
```

Continue with steps 2-5; quality gates that require missing dev tools (`mypy`, `build`, `twine`) will be skipped with an explicit note in the handoff.

### Step 2 — probe runtime imports

```bash
"${PY}" - <<'PY'
groups = [
    ("runtime",  ["syngen", "marshmallow", "pandas", "numpy", "pyarrow", "loguru", "yaml"]),
    ("ml",       ["tensorflow"]),
    ("regex",    ["exrex"]),
    ("fernet",   ["cryptography.fernet"]),
    ("logging",  ["reportportal_client"]),
    ("dev",      ["pytest", "flake8", "build", "twine"]),
]
missing = []
for label, mods in groups:
    for m in mods:
        try:
            __import__(m)
        except Exception as e:
            missing.append((label, m, f"{type(e).__name__}: {e}"))
            print(f"  FAIL {label:8s} {m:30s} {type(e).__name__}: {e}")
        else:
            print(f"  OK   {label:8s} {m}")
print()
print("missing:" if missing else "all imports OK")
for label, m, e in missing:
    print(f"  - [{label}] {m}: {e}")
PY
```

Each missing module that you did *not* just resolve via the venv creation in step 1 is an escalation. Emit:

```
APPROVAL REQUIRED: install missing module(s) <list> via
    "${PY}" -m pip install <list>
```

### Step 2b — installed-vs-source version parity

This library is consumed downstream by `tdm_syngen`, whose tests frequently run against this source tree via `PYTHONPATH=<syngen>/src` rather than an install. An **installed `syngen` wheel of a different version** then silently shadows or mismatches the source you are editing, producing failures (or false passes) that have nothing to do with your change. Before trusting any test result, compare the installed version to this source tree's `src/syngen/VERSION`:

```bash
"${PY}" - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
src = Path("src/syngen/VERSION").read_text().strip()
try:
    inst = version("syngen")
except PackageNotFoundError:
    inst = "(not installed)"
print(f"source={src}  installed={inst}")
print("MATCH" if inst == src else "DRIFT — prefer PYTHONPATH=<syngen>/src, or reinstall the matching wheel (escalation)")
PY
```

On `DRIFT`, do **not** `pip install --upgrade` (denied by `.claude/settings.json` and a dependency change requiring escalation). Either run against the source tree with `PYTHONPATH` set, or escalate to reinstall the exact matching wheel. Report the drift in your handoff.

### Step 3 — verify pytest collection

```bash
"${PY}" -m pytest --collect-only -q src/tests/unit 2>&1 | tail -5
```

Expected: a `<N> tests collected` line with no collection errors.

- If you see `ModuleNotFoundError: No module named 'tests'`, the structural fix has been reverted. Restore `src/tests/__init__.py` (empty file) and the `[tool.pytest.ini_options]` block in `pyproject.toml`. This is a [[harness change]] — coordinate via `syngen-harness-update`.
- If you see `ModuleNotFoundError: No module named 'exrex'` (or any other runtime module), go back to Step 2.

### Step 4 — verify entry-point binaries and imports

```bash
ls -la "${VENV_DIR}/bin/train" "${VENV_DIR}/bin/infer" "${VENV_DIR}/bin/syngen" 2>/dev/null
"${PY}" -c "import syngen; print(syngen.__file__)"
```

If the console scripts are missing, the editable install in step 1 was skipped or failed — re-run `"${PY}" -m pip install -e .`.

### Step 5 — confirm runtime cache directories are absent or empty

```bash
ls -d "${REPO_ROOT}/model_artifacts" "${REPO_ROOT}/tmp_store" 2>/dev/null | head
```

These should not exist on a fresh checkout. If they do, they are leftover from a prior run and should be cleaned up before committing. They are denied for read by the harness (`./model_artifacts/**`, `./tmp_store/**` in `settings.json`).

## When to invoke

- Fresh `git clone`.
- Any `python -m pytest …` reports `ModuleNotFoundError`.
- Before running a release wheel build, `mypy`, or `pip-audit`.
- After a change to `requirements.txt`, `setup.cfg`, `pyproject.toml`, or `MANIFEST.in`.
- After a Python interpreter upgrade.

## Do not

- Do not `pip install --upgrade <package>` for any package other than `pip` itself inside a freshly created venv. All other `--upgrade` / `--force-reinstall` / `-U` invocations are denied by `.claude/settings.json` because they constitute dependency changes that must go through `syngen-release-engineer`.
- Do not edit `requirements.txt` to satisfy a missing import — escalate via the Approval contract.
- Do not commit the `.venv` directory.
- Do not restore the literal Fernet keys that used to live in `src/tests/conftest.py`. The current `os.environ.setdefault(..., Fernet.generate_key().decode())` pattern is the correct one. If a test fails because it expects a specific encrypted payload, the test is wrong — fix the test, not the fixture.

## Handoff

State:

- Which Python interpreter was used (venv path or system).
- Whether the venv was created or already existed.
- Which (if any) modules were installed and the exact commands run.
- Whether `pytest --collect-only` reported zero collection errors and the number of tests collected.
- Whether `train` / `infer` / `syngen` console scripts are on the venv PATH.
