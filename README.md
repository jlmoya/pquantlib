# PQuantLib

> A 100%-Python port of [QuantLib](https://www.quantlib.org/) — the de-facto open-source library for quantitative finance — being systematically rebuilt from C++ v1.42.1 with bit-exact precision guarantees.

[![Tag](https://img.shields.io/badge/tag-pquantlib--phase0--bootstrap-yellow)](#migration-status)
[![Branch](https://img.shields.io/badge/branch-phase1--A%20(Stages%200--3%20closed)-orange)](#migration-status)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](#migration-status)
[![Build](https://img.shields.io/badge/build-uv%20workspace-success)](#repo-layout)
[![C%2B%2B%20pin](https://img.shields.io/badge/C%2B%2B%20pin-v1.42.1-informational)](#ground-truth)
[![Sister%20project](https://img.shields.io/badge/sister-jquantlib-blueviolet)](#sister-project)
[![License](https://img.shields.io/badge/license-BSD-green)](#license)

---

## What is PQuantLib?

PQuantLib provides Python developers and quants with the mathematical, statistical, and modelling toolset needed to value equities, options, futures, swaps, fixed-income instruments, and a wide range of derivatives. It mirrors QuantLib's C++ API as faithfully as Python idioms allow, offering a precise, type-checked Python alternative to:

- The official [QuantLib-Python](https://pypi.org/project/QuantLib/) SWIG bindings (which wrap the C++ library) — PQuantLib is a **pure-Python reimplementation**, not a binding.
- Building your own QuantLib bridge over JPype/Py4J to JQuantLib (slower, JVM-coupled).
- Re-implementing risk math piecewise on top of NumPy/SciPy (no QuantLib parity).

PQuantLib is being built as a **systematic, full-fidelity port from C++ QuantLib v1.42.1**, using the same migration discipline that delivered the sister project [JQuantLib](https://github.com/jlmoya/jquantlib) (Java port). Every functional change is cross-validated against C++ reference values via probe programs that link against the pinned QuantLib commit.

## Sister project

This port is the **Python sibling of JQuantLib** (tag `jquantlib-final`). Both projects:

- Pin the same C++ ground truth (`v1.42.1` @ `099987f0`)
- Use the same migration patterns (subagent-driven, worktree-parallel, tier-stratified tolerances, probe-based cross-validation, direct-to-main per cluster)
- Share the same `migration-harness/` design (C++ submodule + probe directory + JSON reference values)
- Pass the same test-suite scope (faithful ports of C++ `test-suite/*.cpp` files)

The two projects are independent but borrow heavily from each other's plans. Bugs surfaced in one port are checked in the other.

## Project posture (2026-05 → present)

| Principle | What it means in practice |
|-----------|---------------------------|
| **C++ is source of truth** | Where Python diverges from QuantLib v1.42.1 — signatures, implementations, constants, behavior — the Python code is adapted to match C++ behavior. Python idioms (dataclasses, type hints, `match`/`case`, async) are used where natural. |
| **Cross-validation before commit** | Every functional change is backed by a C++ "probe" (a small program linked against the pinned QuantLib submodule) that emits reference values to JSON. Python tests load those JSONs and assert against them via tolerance helpers. No expected value is invented inline. |
| **Tier-stratified tolerances** | Comparisons land in one of three tiers — **EXACT** (bit-identical via `struct.pack('!d', x)`), **TIGHT** (`math.isclose(abs_tol=1e-14, rel_tol=1e-12)`), **LOOSE** (`math.isclose(abs_tol=1e-8, rel_tol=1e-8)`). Per-test exceptions require an inline written justification. |
| **Bulletproof, not fast** | Every commit passes `uv run pytest` + `uv run pyright` + `uv run ruff check`. One stub fix = one commit. No `--no-verify`, no skipped hooks. Mid-port architectural divergence becomes a separate `align(...)` commit, never bundled. |
| **Direct-to-main** | Solo single-owner repo; no PR overhead. Each phase ends with a signed git tag (`pquantlib-phase<N>-complete`) and a completion document under `docs/migration/`. |

## Migration status

| Phase | Tag | What landed | Tests | Date |
|-------|-----|-------------|-------|------|
| 0 | `pquantlib-phase0-bootstrap` | Project skeleton (uv workspace, 4 packages, pyright strict, ruff lint+format, pytest), CLAUDE.md, migration-harness/ scaffold, BSD LICENSE | 2/0/0 (smoke) | 2026-05-23 |
| 1 L1-A (in progress) | _(branch `phase1-A`)_ | Stages 0–3 closed: harness bootstrap (QuantLib submodule + sentinel probe) → foundations (`exceptions`, `qassert`, `testing.tolerance`, `testing.reference_reader`, all 5 pattern modules) → **time core** (6 IntEnums, `Period`, `Date`, parsers, `Calendar` abstract + Null/WeekendsOnly/Joint/Bespoke, `Schedule` + `MakeSchedule`, `IMM`, `ASX`, `ECB`, `TimeGrid`, `TimeSeries`) → **day counters** (DayCounter abstract + 11 concretes covering 24+ convention aliases). Stages 4–5 (sovereign/exchange calendars, first math batch) pending. | 311/0/0 | 2026-05-23..2026-05-24 |

Per-phase scoping mirrors JQuantLib's layer sequencing:
- **Phase 1:** L1 — math primitives (`Array` via numpy, `Date`, `Calendar`, `DayCounter`, distributions, integrals, interpolations, RNGs)
- **Phase 2:** L2 — termstructures + indexes
- **Phase 3:** L3 — instruments + pricingengines
- **Phase 4:** L4 — models
- **Phase 5:** L5 — experimental + L6 test-suite parity
- **Phase 6:** Python 3.14 modernization sweep
- **Phase 7:** Final closure + carve-out documentation + tag `pquantlib-final`

**Current tip on `main`:** `ec4fed0 docs(migration): draft L1-A design + plan` (Phase 0 closed via `pquantlib-phase0-bootstrap` tag).
**Current tip on `phase1-A`:** `601aa1e feat(daycounters): port ActualActual` (Stages 0–3 of L1-A landed, 311/0/0 green). See [`docs/migration/phase1-l1-A-design.md`](docs/migration/phase1-l1-A-design.md) + [`phase1-l1-A-plan.md`](docs/migration/phase1-l1-A-plan.md) + [`phase1-l1-A-progress.md`](docs/migration/phase1-l1-A-progress.md).

## Repo layout

```
pquantlib/                    # repo root
├── pquantlib/                # core package
│   ├── pyproject.toml
│   ├── src/pquantlib/        # the actual sources
│   │   ├── __init__.py
│   │   ├── py.typed
│   │   └── ...               # ported modules land here
│   └── tests/                # pytest tests
├── pquantlib-contrib/        # community contributions / extras
│   └── (same shape)
├── pquantlib-helpers/        # convenience builders + utilities
│   └── (same shape)
├── pquantlib-samples/        # example scripts (not packaged for distribution)
│   └── (same shape)
├── migration-harness/        # C++ ground-truth infrastructure
│   ├── cpp/quantlib/         # git submodule → QuantLib v1.42.1 @ 099987f0
│   ├── cpp/probes/           # one-off C++ probes emitting reference JSONs
│   ├── references/           # JSON reference value files consumed by pytest
│   ├── build-cpp.sh          # builds QuantLib + all probes
│   └── generate-references.sh # runs all probes, emits JSONs
├── docs/migration/           # per-phase design / plan / progress / completion docs
├── pyproject.toml            # workspace root (members = the 4 packages above)
├── .python-version           # 3.14
├── CLAUDE.md                 # binding instructions for Claude Code sessions
├── README.md                 # this file
└── LICENSE                   # BSD
```

## Architecture of a phase

Every phase follows a uniform shape (refined from JQuantLib's discipline):

```
brainstorm → design → plan → execute (subagent-driven) → review → tag → memory
```

### 1. Brainstorm & design
A binding spec (`docs/migration/phase<N>-design.md`) is approved before any code is written. Sections include scope (in/out), approach comparison, worktree topology, tolerance & probe discipline, pause triggers (A1–A8), decision log.

### 2. Plan
A bite-sized, checkbox-tracked task list (`docs/migration/phase<N>-plan.md`) with exact file paths, code snippets, and expected test-count deltas per task. No "TODO" or "TBD" — every step is concrete.

### 3. Execute
Each phase runs across **2–5 git worktrees** (named `pquantlib-<phase>-A`, `-B`, `-C`, ...). A controller dispatches one fresh subagent per cluster with two-stage review:

1. **Spec compliance review** — does the code match the spec exactly?
2. **Code quality review** — pyright strict, ruff clean, idiomatic Python, no dead code, no half-finished implementations.

After both reviews pass, the cluster fast-forwards to `main`.

### 4. Review & tag
Final code-quality reviewer for the entire phase. Signed annotated tag with a comprehensive commit message. Completion doc written.

## Getting started

```bash
git clone git@github.com:jlmoya/pquantlib.git
cd pquantlib

# Install dependencies into the workspace venv
uv sync

# Run the full suite
uv run pytest

# Type-check
uv run pyright

# Lint
uv run ruff check

# Build the C++ harness (one-time setup before probe-validated tests)
./migration-harness/build-cpp.sh
```

## Ground truth

C++ QuantLib is pinned to `v1.42.1` @ commit [`099987f0ca2c11c505dc4348cdb9ce01a598e1e5`](https://github.com/lballabio/QuantLib/commit/099987f0ca2c11c505dc4348cdb9ce01a598e1e5) (2026-04-16 — same pin as the sister project JQuantLib). The submodule lives at `migration-harness/cpp/quantlib/`.

When v1.42.1 has a documented bug (caught via cross-validation), PQuantLib mirrors the buggy behavior in production code and documents the bug inline. Fixing the bug is a separate decision logged in the phase-design's decision log.

## License

BSD (same as QuantLib).

## Acknowledgements

- The [QuantLib](https://www.quantlib.org/) team — foundational C++ codebase and ongoing reference.
- The [JQuantLib](https://github.com/jlmoya/jquantlib) sister project — proved the migration discipline that PQuantLib reuses.
- The [CORE-MATH](https://core-math.gitlabpages.inria.fr/) project (Sibidanov et al., Inria) — provably correctly-rounded transcendental algorithms (used by JQuantLib's JQuantMath; PQuantLib leverages `mpmath` for the equivalent precision guarantees).
