# Phase 3 L3-A pilot — completion

**Date closed:** 2026-05-27
**Tag:** `pquantlib-phase3-l3-A-complete` @ `e72bcdf`
**Predecessor:** `pquantlib-phase2-complete` @ `b5d2519`
**Test delta:** 922 → 1037 (+115; exceeded +76 target by 51%). Triad clean.
**Successor:** merged into `pquantlib-phase3-complete` @ `aacc2c2` via 4 parallel L3-B/C/D/E clusters.

## Stages closed

| Stage | Topic | Tests | Approach |
|---|---|---|---|
| 0 | foundations mega-probe | — | C++ probe → `l3a/foundations.json` |
| 1 | `Settings.evaluation_date` observable + 4 retroactive cleanups | +12-15 | ObservableSettings multi-inherits Singleton + Observable; Schedule null-effective-date fallback, TermStructure moving-mode, RelativeDateBootstrapHelper, SmileSection floating-mode all closed |
| 2 | Payoff hierarchy | +18 | PlainVanilla + Cash/Asset OrNothing + Gap + SuperFund + SuperShare |
| 3 | Exercise hierarchy | +12 | European / American / Bermudan |
| 4 | Instrument + PricingEngine + GenericEngine | +12 | `GenericEngine[ArgsT, ResultsT]` PEP 695 generic with typed bounds |
| 5 | BlackFormula family | +14 | lognormal + bachelier + implied-vol solvers (Halley + Jaeckel 2017) + Black-vega derivatives |
| 6 | Option + OneAssetOption + cross-cluster Protocols | +8 | `@runtime_checkable` Instrument / PricingEngine / StochasticProcess Protocols |

## Notable decisions

- **`Settings.evaluation_date` wired in this pilot** rather than deferring further. Unblocked 4 L1/L2 deferrals at once (Schedule, TermStructure-moving, RelativeDateBootstrapHelper, SmileSection-floating). ObservableSettings multi-inheritance pattern documented inline.
- **`ObservableValue<Date>` push semantics → Python property setter** that calls `notify_observers()` only on actual value change.
- **`GenericEngine[ArgsT, ResultsT]`** PEP 695 generic with typed bounds (`PricingEngineArguments`, `PricingEngineResults`).
- **`LazyObject` metaclass**: C++ uses plain `type` with `@abstractmethod` informational. PQuantLib's `Instrument` multi-inherits `ABC` to bring in `ABCMeta` so `is_expired` is enforced abstract.
- **`Exercise` / `EarlyExercise` abstract-by-convention** (no `@abstractmethod`) — matches C++ public constructors + empty default `dates_`.
- **`bachelierBlackFormulaImpliedVolChoi` (approximation) carved out** — only the exact Jaeckel 2017 variant ported.
- **C++ `shared_ptr<PlainVanillaPayoff>` overloads of `blackFormula*` not ported** — Python callers can unwrap themselves.

## Files of note (relative to repo root)

- `pquantlib/src/pquantlib/payoffs.py` — Payoff hierarchy (single module, mirroring C++ `ql/payoffs.hpp`).
- `pquantlib/src/pquantlib/exercise.py` — Exercise hierarchy.
- `pquantlib/src/pquantlib/option.py` — Option abstract.
- `pquantlib/src/pquantlib/instruments/instrument.py` — Instrument abstract.
- `pquantlib/src/pquantlib/instruments/one_asset_option.py` — OneAssetOption abstract.
- `pquantlib/src/pquantlib/instruments/protocols.py` — cross-cluster Protocols.
- `pquantlib/src/pquantlib/pricingengines/{pricing_engine,generic_engine,black_formula}.py`.
- `pquantlib/src/pquantlib/patterns/observable_settings.py` — extended with `evaluation_date` + observer plumbing.
- `migration-harness/cpp/probes/l3a/foundations_probe.cpp` + `migration-harness/references/l3a/foundations.json`.
