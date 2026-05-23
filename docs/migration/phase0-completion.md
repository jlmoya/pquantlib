# Phase 0 — Project bootstrap (completion)

**Closed:** 2026-05-23
**Tag:** `pquantlib-phase0-bootstrap` @ `85018e5`
**Commits:**
- `b9569d2 infra(bootstrap): PQuantLib project skeleton (Phase 0)`
- `85018e5 infra(bootstrap): fix uv sync to install workspace members + pyright import resolution`

## What landed

A green PQuantLib skeleton on `main`:

- uv workspace with 4 members (`pquantlib`, `pquantlib-contrib`, `pquantlib-helpers`, `pquantlib-samples`) on Python 3.14.
- Root `pyproject.toml` declares the four members as `dependencies` so plain `uv sync` installs them editable in one shot.
- pyright strict (`typeCheckingMode = "strict"`, `pythonVersion = "3.14"`), with `extraPaths` + `venvPath`/`venv` so `import pquantlib` resolves to `pquantlib/src/pquantlib/__init__.py` (not the bare workspace-member directory namespace).
- ruff (`E,F,W,I,N,UP,B,C4,SIM,RUF,PL,PT`) + ruff-format, line length 110, target `py314`.
- pytest 8+ with 5 tolerance-tier markers (`slow`, `integration`, `exact`, `tight`, `loose`) and `filterwarnings = error`.
- `migration-harness/` scaffolded (README, `build-cpp.sh`, `generate-references.sh`, `cpp/probes/CMakeLists.txt` placeholder, `references/` empty dir). C++ submodule clone deferred to Phase 1 (one-time `git submodule add`, documented in the harness README).
- Smoke test (`pquantlib/tests/test_smoke.py`) asserting `__version__ == "1.0.0"` and `import pquantlib` doesn't raise.
- `CLAUDE.md` with Python-specific Java→Python cheatsheet, ground-truth principle, operational rules, pause triggers.
- `README.md` with "What is PQuantLib?" + sister-project link to `jquantlib-final`.
- BSD `LICENSE`.
- `docs/migration/README.md` + `phase0-design.md` + `phase0-plan.md` + this completion doc.
- `uv.lock` committed for reproducible builds.

## Verification (final)

| Check | Result |
|---|---|
| `uv sync` | Installs 17 packages incl. 4 editable workspace members |
| `uv run pytest` | 2 passed, 0 failed, 0 skipped (0.02s) |
| `uv run pyright` | 0 errors, 0 warnings, 0 informations |
| `uv run ruff check` | All checks passed |
| `uv run ruff format --check` | 9 files already formatted |

## Surprises / lessons learned

- **Workspace member install is not automatic.** A uv `[tool.uv.workspace]` block alone does not install the members into the venv; the root project must list them under `[project] dependencies = [...]` (or every command must be `uv sync --all-packages`). The plain `uv sync` UX the design doc and `CLAUDE.md` prescribe needs the dependency declaration. Fixed in `85018e5`.
- **Pyright + src layout + workspace member named identically to the package.** Because the workspace member directory is `pquantlib/` and the actual package is at `pquantlib/src/pquantlib/`, pyright (and CPython) will try to interpret the outer directory as a PEP 420 namespace package and miss the real one. `extraPaths` pointing pyright at each `<member>/src` fixed it for pyright; `uv sync` installing the wheels into `.venv` fixed it for runtime. Lesson: in a workspace where members and packages share names, both pyright and pytest need explicit hints.
- **uv.lock should be committed.** Standard practice; the bootstrap missed it because the initial sync produced an empty lock that was never staged. Committed in `85018e5`.

## Carry-overs into Phase 1

None blocking — these are deferred-by-design from the Phase 0 scope:

- `pquantlib.testing.tolerance` helper module (with `exact`, `tight`, `loose`, `custom`) — write alongside the first cross-validated test.
- `pquantlib.exceptions.LibraryException` + `pquantlib.QL.require` — write alongside the first ported class that needs it.
- C++ submodule clone (`migration-harness/cpp/quantlib/`) — `git submodule add` when the first probe is written.
- First C++ probe + reference-value JSON — see Phase 1 design for the chosen pilot class.

## Phase 1 entry conditions

All met:

- [x] `main` is green (pytest + pyright + ruff)
- [x] Tag `pquantlib-phase0-bootstrap` exists on remote
- [x] CLAUDE.md describes the operational rules an implementer needs
- [x] `migration-harness/` skeleton exists so the first probe can be dropped in

Proceed to draft `docs/migration/phase1-design.md`.
