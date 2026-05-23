# Phase 1 L1-A — Running progress log

Branch: `phase1-A` (worktree `../pquantlib-phase1-A`)
Pinned base: `main` @ `ec4fed0`
QuantLib C++ ground truth: v1.42.1 @ `099987f0`

## Stage 0 — Harness bootstrap (closed)

| Commit | Subject |
|---|---|
| `557b7bf` | `fix(harness): add Boost include path to probes CMakeLists` |
| `5a82562` | `infra(harness): bootstrap C++ submodule + sentinel probe` |

Outcome: QuantLib submodule cloned, built (`libQuantLib.dylib` linked clean),
sentinel probe emits `references/harness/sentinel.json` with v1.42.1 + sqrt(2)
+ pi at full precision.

Surprises:
- Phase 0 probes CMakeLists missed `find_package(Boost)` — QL's `qldefines.hpp`
  transitively `#include`s `<boost/config.hpp>`, surfaces only when a real probe
  exists. Fixed before sentinel commit.

## Stage 1 — Foundations (closed)

| Commit | Subject | Tests added |
|---|---|---|
| `b592349` | `fix(infra): exclude QuantLib submodule from ruff format` | 0 |
| `4d2162a` | `feat(exceptions): port LibraryException + qassert helpers` | 13 |
| `d1c1e7e` | `feat(testing): add tolerance-tier assertion helpers` | 11 |
| `2d4464c` | `feat(testing): add reference_reader for C++ probe JSON outputs` | 7 |
| `ff45d94` | `feat(patterns): port Observer, Singleton, LazyObject, ObservableSettings, Visitor` | 18 |

Outcome: 51 tests pass, pyright strict clean, ruff lint+format clean. The full
foundation surface (exceptions, qassert, tolerance, reference_reader, all 5
pattern modules) is now usable by Stages 2–5.

Surprises:
- Phase 0 ruff config excluded `migration-harness/**` from lint via
  `per-file-ignores` but did NOT exclude from format. Fixed with `extend-exclude`
  under `[tool.ruff]`.
- pyright pyright-strict caught `target: T` in generic `Visitor[T]` Protocol as
  partially-unknown when subclassed with a narrower `target: float`. Collapsed
  Visitor to non-generic Protocol — decision recorded in L1-A design log #10,
  refined here: the `T` parameter is dropped in Python because structural typing
  + Protocol checks fire on method-name matching; specific target-type narrowing
  happens at the implementing class's own visit signature.
- pyright also flagged `dict[type, Any]` keyed by `cls` inside the Singleton
  metaclass `__call__` (pyright sees `cls` as `Self@_SingletonMeta`, not `type`).
  Relaxed to `dict[Any, Any]` with an inline rationale comment.

## Stage 2 — Time core (pending)

Up next: time enums → Period → Date → Date/PeriodParser → Calendar abstract +
Null/WeekendsOnly/Joint/Bespoke → Schedule + MakeSchedule → IMM/ASX/ECB →
TimeGrid/TimeSeries/Series.

## Stage 3 — Day counters (pending)

## Stage 4 — Calendars (pending)

## Stage 5 — First math batch (pending)
