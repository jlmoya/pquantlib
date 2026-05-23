# Phase 0 — Project bootstrap (design)

**Date:** 2026-05-23
**Status:** in progress (about to be closed)
**Predecessor:** none (greenfield)
**Sister-project anchor:** [jquantlib-final](https://github.com/jlmoya/jquantlib/releases/tag/jquantlib-final)

## Goal

Create a working PQuantLib skeleton that:

1. Builds via `uv sync` on Python 3.14
2. Passes `uv run pytest` (smoke test only)
3. Passes `uv run pyright` strict mode
4. Passes `uv run ruff check`
5. Has the `migration-harness/` scaffolding ready for the first C++ probe
6. Has CLAUDE.md, README.md, LICENSE in place
7. Is committed + pushed to `main` at tag `pquantlib-phase0-bootstrap`

## Scope (in)

- Workspace structure (4 packages: pquantlib, pquantlib-contrib, pquantlib-helpers, pquantlib-samples)
- Tooling config (pyright strict, ruff lint+format, pytest with 5 markers: slow/integration/exact/tight/loose)
- migration-harness scaffolding (README, build-cpp.sh, generate-references.sh, probes/CMakeLists.txt with placeholder)
- Smoke test that imports `pquantlib` and asserts `__version__`
- BSD LICENSE (matches JQuantLib / QuantLib)
- README.md with "What is PQuantLib?" header + sister-project link
- CLAUDE.md with Python-specific translation cheatsheet (Java → Python)

## Scope (out — comes in Phase 1+)

- Any actual ported QuantLib class (math, time, instruments, etc.)
- The C++ submodule clone (deferred until Phase 1 needs the first probe; documented in migration-harness/README.md so the new Claude knows the one-time setup command)
- The first C++ probe (deferred to Phase 1 with concrete need)
- A `tolerance.py` helper module (deferred to Phase 1 with the first cross-validated test that needs it)
- `LibraryException` + `QL.require` helpers (deferred to Phase 1 with concrete use)

## Approach

Mirror JQuantLib's repo shape:

| JQuantLib (Java/Maven) | PQuantLib (Python/uv) |
|---|---|
| `jquantlib/` (core module) | `pquantlib/` (core package) |
| `jquantlib-contrib/` | `pquantlib-contrib/` |
| `jquantlib-helpers/` | `pquantlib-helpers/` |
| `jquantlib-samples/` | `pquantlib-samples/` |
| Root `pom.xml` (aggregator + parent) | Root `pyproject.toml` (uv workspace + tooling config) |
| `migration-harness/` (C++ submodule + probes) | `migration-harness/` (same C++ submodule + same probe pattern) |
| `docs/migration/` (per-phase docs) | `docs/migration/` (per-phase docs) |
| `CLAUDE.md` | `CLAUDE.md` (Python-flavored) |
| `README.md` (Java audience) | `README.md` ("What is PQuantLib?" Python audience) |

Why this shape:

- Python doesn't need 4 separate distributable packages for a small lib, BUT mirroring JQuantLib's structure makes cross-port reasoning easier (a class at `org.jquantlib.math.copulas.ClaytonCopula` maps obviously to `pquantlib.math.copulas.clayton_copula.ClaytonCopula`).
- uv workspaces give us the multi-package Maven-multi-module feel while still being trivial to install with one `uv sync`.

## Tolerance discipline (binding from Phase 1 onward)

| Tier | Predicate | Use case |
|---|---|---|
| **EXACT** | `struct.pack('!d', actual) == struct.pack('!d', expected)` | Bit-identical (deterministic algorithms with no FP non-associativity) |
| **TIGHT** | `math.isclose(actual, expected, abs_tol=1e-14, rel_tol=1e-12)` | Default for closed-form math after light arithmetic |
| **LOOSE** | `math.isclose(actual, expected, abs_tol=1e-8, rel_tol=1e-8)` | Iterative methods, FDM, MC; document inline why TIGHT isn't achievable |

Helper module to be added in Phase 1: `pquantlib.testing.tolerance` with `exact()`, `tight()`, `loose()` predicates + a `custom(actual, expected, abs_tol, rel_tol, reason)` for documented per-test exceptions.

## Worktree topology (binding from Phase 1 onward)

Per JQuantLib's proven pattern:
- Up to 5 worktrees in parallel: `pquantlib-<phase>-{A,B,C,D,E}`
- Each cluster owns a thematic subset of the phase's scope
- Each cluster ends in FF-merge to `main`
- Worktrees + their branches are cleaned up local+remote after merge

## Pause triggers (A1–A8 — binding)

Carried over from JQuantLib design §7.3:

| ID | Condition | Action |
|---|---|---|
| A1 | Phase scope > 1000 classes after audit | Pause, re-scope |
| A2 | Tolerance looser than 1e-8 needed | Pause, document rationale |
| A3 | Cross-validation suggests v1.42.1 is wrong | Pause, log decision |
| A4 | Stub needs Python dep not in workspace | Pause, ask for add-dep approval |
| A5 | Architectural divergence from Java that affects multiple ported classes | Pause, design adapter |
| A6 | End of every phase | Report summary, wait for ack |
| A7 | Test failure that can't be explained by porting choice | Pause, escalate |
| A8 | Credential / repo access needed (e.g., `gh` token issue) | Pause, ask |

## Decision log (Phase 0)

| # | Decision | Why |
|---|---|---|
| 1 | Python 3.14 (not 3.12 LTS or 3.13) | Latest stable — analogue to JQuantLib's "JDK 25 LTS / modernized" choice. PEP 750 t-strings, PEP 765/779. |
| 2 | uv (not Poetry, not pip) | Fastest dep resolver, native workspace support, lockfile by default. Best modern-Maven analogue. |
| 3 | hatchling backend (not setuptools, not flit) | PEP-517 compliant, minimal, well-supported by uv/Hatch family. |
| 4 | 4-package workspace (not single src-layout) | Faithful mirror of JQuantLib's 5-module reactor (with `jquantlib-parent` collapsed into root). |
| 5 | pyright strict (not mypy strict) | Faster than mypy, used by Pylance/PyCharm Pro/VS Code natively. Best javac-equivalent rigor. |
| 6 | ruff for lint+format (not flake8+black) | Single tool, Rust-fast, well-maintained. |
| 7 | pytest 8+ (not unittest) | Universal Python standard. JUnit 4 analogue. |
| 8 | Numpy `ndarray[float64]` for `Array` analogue (not custom class) | Java's `Cells.$` raw-array deprecation pain → solved by using numpy from day one. |
| 9 | `mpmath` for arbitrary-precision reference values (not CORE-MATH C++ port) | mpmath already has 100+ digits at hand; no need to port JQuantMath's correctly-rounded transcendentals because mpmath suffices for cross-validation. |
| 10 | Defer C++ submodule clone to Phase 1 | Phase 0 has no probe-validated test; the one-time `git submodule add` is documented in migration-harness/README.md and the new Claude can run it when Phase 1 needs the first probe. |
| 11 | `pquantlib.exceptions.LibraryException` + `pquantlib.QL.require` to mirror jquantlib semantics | Allows direct translation of C++ `QL_REQUIRE` / `QL_FAIL` idioms without inventing new exception hierarchies. **DO NOT** add side-effect logging in the ctor (the JQuantLib `LibraryException` originally did, then had to be fixed in commit `de95bb17`). |
| 12 | Direct-to-main per cluster (no PRs) | Solo single-owner repo, same as JQuantLib. |

## Definition of done (Phase 0)

- [x] Repo structure created (4 packages + migration-harness/ + docs/migration/)
- [x] Root pyproject.toml with uv workspace + pyright strict + ruff + pytest config
- [x] Each member package has pyproject.toml + src/<pkg>/__init__.py + py.typed + tests/__init__.py
- [x] CLAUDE.md written (full Python-specific cheatsheet)
- [x] README.md written ("What is PQuantLib?" + sister-project link)
- [x] LICENSE in place (BSD)
- [x] migration-harness/ scaffolded (README, build script, probe CMakeLists with placeholder, references/ empty dir)
- [x] Smoke test that asserts `pquantlib.__version__`
- [x] `.python-version` pinning 3.14
- [ ] `uv sync` succeeds (run after committing — needs Python 3.14 on the machine)
- [ ] `uv run pytest` passes the smoke test
- [ ] `uv run pyright` clean
- [ ] `uv run ruff check` clean
- [ ] Tag `pquantlib-phase0-bootstrap` pushed
- [ ] Initial commit + push

## Next phase preview

**Phase 1 — math primitives (L1).** Mirrors JQuantLib Phase 2 L1 (5 clusters: A=time+simple-math pilot, B=copulas, C=optimization+RNG, D=distributions+integrals, E=interpolations). Scope: ~80 classes. Estimated 5-10 sessions if dispatched at JQuantLib's pace.

Phase 1 design doc to be drafted by the new Claude as the first task after Phase 0 closes.
