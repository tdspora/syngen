# Agent Harness Changelog

| Date | File | Reason | Approved by |
|------|------|--------|-------------|
| 2026-05-27 | .claude/skills/syngen-implement-feature/SKILL.md | Add step 7: bump version in src/syngen/VERSION after each feature | human |
| 2026-07-02 | .claude/skills/syngen-bootstrap/SKILL.md | Session-reflection P2 — add Step 2b installed-vs-source version parity check (`importlib.metadata.version("syngen")` vs `src/syngen/VERSION`); on DRIFT prefer PYTHONPATH=<syngen>/src or escalate a matching reinstall. Motivated by a session where a stale installed `syngen 0.10.45` shadowed a 0.12.x source tree during downstream tdm_syngen testing | human (`appchanges approved`) |
