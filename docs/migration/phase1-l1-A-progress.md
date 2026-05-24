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

## Stage 2 — Time core (closed)

| Commit | Subject | Tests added |
|---|---|---|
| `fa8b7b6` | `feat(time): port enums (Weekday, Month, TimeUnit, Frequency, BusinessDayConvention, DateGeneration)` | 76 |
| `26b1749` | `fix(infra): allow PLR0911 (too-many-returns) for switch-on-enum ports` | 0 |
| `373f67a` | `feat(time): port Period` | 22 |
| `67e055d` | `feat(time): port Date` | 23 |
| `2b41823` | `feat(time): port DateParser + PeriodParser` | 14 |
| `87553ea` | `fix(infra): allow PLR0912 (too-many-branches) for switch-on-enum ports` | 0 |
| `4349930` | `align(time): add @overload signatures to Date.__add__ and __sub__` | 0 |
| `3047166` | `feat(time): port Calendar + NullCalendar + WeekendsOnly + JointCalendar + BespokeCalendar` | 17 |
| `e7ad2e6` | `feat(time): port IMM helpers` (reordered before Schedule — see below) | 9 |
| `11f00f9` | `feat(time): port Schedule + MakeSchedule` | 18 |
| `e55f4f1` | `feat(time): port ASX + ECB date helpers` | 15 |
| `100b608` | `feat(time): port TimeGrid + TimeSeries` | 13 |

Outcome: **258/0/0 tests** (added 207 over Stage 2), pyright strict clean, ruff
lint+format clean. The full time-layer surface is usable — enums, Period, Date,
parsers, Calendar abstract + 4 trivial calendars, Schedule + MakeSchedule, IMM,
ASX, ECB, TimeGrid, TimeSeries.

Surprises / decisions:
- **`IMM` ported before `Schedule`** (Task 2.7a inserted between 2.5 and 2.6).
  `Schedule.from_rule` validates `first_date` / `next_to_last_date` against
  `is_imm_date` for the ThirdWednesday rule. Reordering was lighter than
  stubbing IMM inside Schedule.
- **Date.month() bug** surfaced during Date cross-validation: C++ `month()`
  probes `monthOffset(m+1)` as the year-end bracket; for `m=12` that's `Month(13)`
  which C++ allows via unchecked enum cast but Python `IntEnum` rejects. Fix:
  widened `_month_offset` signature to accept `Month | int`.
- **Date.__add__ / __sub__ overloads** added (`align(time): ...` commit).
  Pyright cannot narrow the union `Date | int` return type from a runtime-
  dispatched body alone — downstream callers like `Calendar.adjust` need
  `Date - 1: Date`. `@overload` signatures provide the narrowing; no runtime
  change.
- **`PLR0911` + `PLR0912` ignored globally** — C++→Python switch-on-enum ports
  hit these limits unavoidably. Each case in the C++ switch becomes an early
  return / branch in Python. Per-method `# noqa` would be noise; global ignore
  is the right precedent.
- **`Schedule.from_rule` divergence**: C++ `Settings::evaluationDate()` fallback
  for null `effective_date` in Backward rule is NOT ported. Python callers must
  pass an explicit effective date; the divergence is captured in the module
  docstring and the error message.
- **`DateParser.parse_formatted` divergence**: C++ uses boost::date_time facets
  with boost-style format strings; the Python port uses `datetime.strptime`
  format codes instead. Common codes (`%Y`, `%m`, `%d`) work identically.
- **`Visitor` collapsed to non-generic Protocol** earlier in Stage 1 is the
  precedent for Python-vs-C++ template simplification.
- **`name()` strings preserved verbatim**: `WeekendsOnly` returns the lowercase
  `"weekends only"` (with space); `JointCalendar.name()` formats as
  `"JoinHolidays(A, B, ...)"`. Required for the C++ `operator==` (name-based
  equality).
- **`Calendar` is `abc.ABC` direct inheritance** (no pImpl Bridge). Per-instance
  `added_holidays` / `removed_holidays` rather than the C++ shared-via-Impl
  pattern.
- **`ECB.knownDates` is a hardcoded 200-entry serial-number table** (2005–2024)
  embedded verbatim from the C++ source. The C++ source uses a `std::set<Date>`;
  Python uses a mutable `set[Date]` extracted from a constant tuple.
- **`Series` (mentioned in the L1-A plan) does NOT exist in C++ v1.42.1.** It
  was a jquantlib-port-specific class. No Series module is created.

## Stage 3 — Day counters (pending)

## Stage 4 — Calendars (pending)

## Stage 5 — First math batch (pending)
