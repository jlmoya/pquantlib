# Phase 2 L2-A pilot — completion

**Date closed:** 2026-05-26
**Tag:** `pquantlib-phase2-l2-A-complete` @ `4ace1f0`
**Predecessor:** `pquantlib-phase1-complete` @ `edcadbc`
**Test delta:** 581 → 649 (+68). Triad clean (pytest + pyright + ruff).
**Successor:** merged into `pquantlib-phase2-complete` @ `b5d2519` via 4 parallel L2-B/C/D/E clusters.

## Stages closed

| Stage | Topic | Classes | Tests | Approach |
|---|---|---|---|---|
| 0 | foundations mega-probe | — | — | C++ probe → `quotes/foundations.json` |
| 1 | Quote / Simple / Derived / Composite | 4 | +19 | EXACT-tier vs probe |
| 2 | TermStructure + Extrapolator | 2 | +15 | behavioral via `_StubTS` |
| 3 | BootstrapHelper + PillarChoice | 2 | +9 | behavioral via `_StubHelper`; PEP 695 generic |
| 4 | Index + IndexManager singleton | 2 | +20 | case-insensitive name; behavioral via `_StubIndex` |
| 5 | cross-cluster Protocols | 4 | +5 | `@runtime_checkable` structural typing for L2-B/C/D/E |

## Notable decisions

- **Protocol-based cross-cluster glue** (Stage 5) — `YieldTermStructureProtocol`, `IborIndexProtocol`, `OvernightIndexProtocol`, `SwapIndexProtocol`. Let 4 parallel clusters reference each other's concretes structurally at merge time. Validated end-to-end at Phase 2 close.
- **`BootstrapError` class skipped** — deprecated in C++ v1.40; use `Callable[[float], float]` instead. Documented inline in `bootstrap_helper.py`.
- **`IndexManager` per-index notifier subsystem skipped** — deprecated in C++ v1.42.1. `Index.update()` is the canonical observer hook.
- **`TermStructure` moving-reference-date mode deferred** — needs `Settings.evaluation_date` observable wiring (still pending across PQuantLib). Subclasses currently use modes 1 (delegated) + 2 (fixed reference date).
- **Compile-time technique for probes**: one-off `clang++` linking against main worktree's already-built `libQuantLib.dylib`, sidestepping the 5–10 min per-worktree CMake rebuild. Reused by all 4 L2-B/C/D/E subagents.

## Files of note (relative to repo root)

- `pquantlib/src/pquantlib/quotes/` — Quote hierarchy.
- `pquantlib/src/pquantlib/termstructures/{term_structure,extrapolator,bootstrap_helper,protocols}.py` — termstructure scaffolding.
- `pquantlib/src/pquantlib/indexes/{index,index_manager}.py` — index core.
- `migration-harness/cpp/probes/quotes/foundations_probe.cpp` + `migration-harness/references/quotes/foundations.json`.
