# Phase 0 Implementation Plan — Project bootstrap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (or just execute inline — this phase is small).

**Goal:** Bring the PQuantLib skeleton from PyCharm scaffold to green `uv sync` + `uv run pytest` + `uv run pyright` + `uv run ruff check`, then tag `pquantlib-phase0-bootstrap`.

**Architecture:** uv workspace, 4 packages (pquantlib + pquantlib-{contrib,helpers,samples}), Python 3.14, pyright strict, ruff lint+format, pytest with 5 tier markers.

**Tech Stack:** Python 3.14, uv, hatchling, pytest, pyright, ruff, numpy 2.1+, mpmath 1.3+.

---

## Tasks

### Task 1: Install Python 3.14 if not already present

```bash
# Check current Python
python3 --version

# If not 3.14, install via uv:
uv python install 3.14

# Pin the workspace to 3.14
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
echo "3.14" > .python-version  # already created by bootstrap

# Verify
uv python pin 3.14
```

### Task 2: Initial sync + verify smoke test

```bash
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
uv sync
uv run pytest -v
```

Expected: 2 tests pass (`test_version_is_present`, `test_import_does_not_raise`).

### Task 3: Type-check

```bash
uv run pyright
```

Expected: 0 errors.

### Task 4: Lint

```bash
uv run ruff check
uv run ruff format --check
```

Expected: 0 errors.

### Task 5: One-time GitHub setup (if not done)

```bash
# If multiple gh accounts, switch to jlmoya
gh auth switch -u jlmoya

# Verify
gh auth status
```

### Task 6: Commit + push the bootstrap

```bash
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
git add -A
git status

# Verify what's being committed (no .venv, no .DS_Store)
git diff --cached --stat | head -30

# Commit
git commit -s -m "infra(bootstrap): PQuantLib project skeleton (Phase 0)

uv workspace with 4 packages (pquantlib + pquantlib-{contrib,helpers,samples})
on Python 3.14. pyright strict + ruff lint+format + pytest with 5 tolerance-
tier markers. migration-harness/ scaffolded for C++ probe infra (submodule
deferred to Phase 1). CLAUDE.md, README.md, LICENSE, docs/migration/ docs.

Sister-project anchor: jquantlib-final.
C++ ground truth: QuantLib v1.42.1 @ 099987f0.

Verified:
- uv sync: ok
- uv run pytest: 2/0/0
- uv run pyright: 0 errors
- uv run ruff check: clean

Next: Phase 1 — L1 math primitives."

git push origin main
```

### Task 7: Tag Phase 0 closure

```bash
git tag -a pquantlib-phase0-bootstrap -m "PQuantLib Phase 0 — project bootstrap

Skeleton complete:
- uv workspace, 4 packages on Python 3.14
- pyright strict + ruff + pytest configured
- migration-harness/ scaffolded
- CLAUDE.md + README.md + docs/migration/ docs in place
- Smoke test green (2/0/0)

Sister project: jquantlib-final
C++ ground truth: QuantLib v1.42.1 @ 099987f0

Phase 1 (L1 math primitives) next."

git push origin pquantlib-phase0-bootstrap
```

### Task 8: Write Phase 1 design doc

Open `docs/migration/phase1-design.md` and draft it. Mirror jquantlib's
`docs/migration/phase2-l1-plan.md` shape, adapted for Python:

- Scope: 5 thematic clusters (A=time+simple-math pilot, B=copulas, C=optimization+RNG, D=distributions+integrals, E=interpolations)
- Use jquantlib's cluster decompositions as starting reference
- Python-specific notes: dataclasses where Java used records, `match-case` where Java used sealed types, numpy where Java used Array/Matrix

This task hands off to a fresh Claude session in this repo.

---

## Definition of done

- [ ] Tasks 1-7 complete; tag pushed
- [ ] Task 8 produces a complete Phase 1 design doc

## Carve-outs

None for Phase 0.
