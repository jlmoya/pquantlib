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

## Stage 3 — Day counters (closed)

| Commit | Subject | Tests added |
|---|---|---|
| `79cca9b` | `feat(daycounters): port DayCounter abstract + OneDayCounter` | 7 |
| `3e05ee6` | `feat(daycounters): port Actual360/364/36525/365Fixed/366 family` | 13 |
| `7f7d87e` | `feat(daycounters): port Thirty360 (9 conventions) + Thirty365` | 14 |
| `7c10922` | `feat(daycounters): port SimpleDayCounter + Business252` | 6 |
| `601aa1e` | `feat(daycounters): port ActualActual (7 conventions, 4 underlying impls)` | 13 |

Outcome: **311/0/0 tests** (added 53 over Stage 3), pyright strict clean, ruff
lint+format clean. The full daycounters layer is usable: abstract DayCounter
+ 11 concretes covering every day-count convention C++ QuantLib v1.42.1 ships.

Surprises / decisions:
- **Probe-emit path naming**: `generate-references.sh` translates underscores
  to slashes in the executable name to derive the JSON output path. So
  `daycounters_one_day_counter_probe` writes to `references/daycounters/one/day/counter.json`
  (not `daycounters/one_day_counter.json`). Tests load with the slash-separated key.
  Pre-existing quirk from the harness scaffolding; harmless once you know it.
- **C++ pImpl + 6 Impl classes for Thirty360 → single Python class** dispatching
  on a `Convention` IntEnum. Same pattern for `Actual365Fixed` (3 impls) and
  `ActualActual` (4 impls). Aliases (e.g. `ISMA ≡ BondBasis` in Thirty360,
  `ISDA ≡ Historical ≡ Actual365` in ActualActual) become shared `name()` output
  and shared dispatch — keeps C++-equality semantics intact.
- **Business252 documented divergences (2 of them):**
  - The C++ default `Business252()` constructor uses `Brazil()` calendar.
    Brazil is not yet ported (Stage 4); the Python port requires an explicit
    calendar argument. Probe uses `WeekendsOnly`.
  - The C++ impl maintains module-level monthly + yearly business-day caches
    keyed by calendar name to amortize cost over multi-year ranges. The Python
    port skips the cache and delegates straight to
    `Calendar.business_days_between` — algorithmically equivalent, only matters
    for performance on very long spans. Cache can be added if profiling
    identifies it as a hotspot.
- **ActualActual ISMA-with-schedule reconstruction**: the test rebuilds the
  same Schedule the probe used (NullCalendar / Unadjusted / Forward / 6-month
  tenor over 2024-01-01..2025-01-01) and asserts the serial-number list
  matches the probe's `schedule_dates_serials` field before exercising
  year-fraction. This pins the probe to the Python schedule-generator's
  behaviour, not just to the year-fraction output.
- **C++ `Old_ISMA_Impl::yearFraction` is recursive** — the Python port mirrors
  the recursion exactly (the only nontrivial recursion in the daycounters
  layer). Defensive: in the split-into-whole-periods branch the while-loop
  uses `d2 >= new_ref_end` instead of C++'s `d2 < new_ref_end break`
  inversion, which would otherwise double-count the trailing period.
- **`OneDayCounter.day_count` returns +1/-1 sign-only**, not `d2 - d1`. The
  base `DayCounter.day_count` defaults to `d2 - d1`; concretes override it
  freely (ActualActual `Old_ISMA` actually uses the default day_count even
  though year_fraction is bespoke; the C++ inherits the default the same way).

## Stage 4 — Calendars (closed)

| Commit | Subject | Tests added |
|---|---|---|
| `ca11f74` | `infra(harness): emit reference holiday-sets for all 41 sovereign/exchange calendars` (single mega-probe → 4506 holiday dates in one JSON) | 0 |
| `5a0b1a9` | `merge: batch A (10 European Western + Orthodox Ukraine)` — `worktree-agent-a90c246249f3d88bb` | 20 |
| `94ea196` | `merge: batch B (Western Europe + Mexico + Orthodox Romania)` — `worktree-agent-a84664e0f52e86d08` | 16 |
| `400540a` | `merge: batch C (Americas + Oceania + Africa)` — `worktree-agent-a5824a55f2c9b4fdc` | 16 |
| `8d779b4` | `merge: batch D (Middle East + Russia/Orthodox + East Asia)` — `worktree-agent-a9bd4436d3446f5b0` | 14 |
| `239079b` | `merge: batch E (China + HongKong + India + Indonesia + SouthKorea + Thailand + UK + US)` — `worktree-agent-af9e9e0b6bdca7b41` | 16 |

Outcome: **41 sovereign/exchange calendars** ported, **82 cross-validation tests** added, all green. Total Stage 4 contribution: 5 subagent branches × 13 underlying feat commits + 5 merge commits + 1 probe-bootstrap commit = 19 commits.

**Parallelization strategy**:
- Single mega-probe at `time/calendars/all_probe.cpp` emits 11-year (2020-2030) non-weekend holiday sets for all 41 default-constructed calendars into one 4506-entry JSON.
- Probe + JSON committed first (`ca11f74`) so all subagents could pull from a shared reference.
- 5 isolated-worktree subagents dispatched in parallel — each handles ~8 calendars, writes Python modules + tests against the committed JSON, verifies pytest+pyright+ruff, commits and pushes their own branch.
- Main session merges each branch back with `--no-ff` after all 5 return. Disjoint file sets per batch meant zero conflicts.
- Total wall-clock: ~25 minutes for 41 calendars (vs ~6+ hours sequential).

Surprises / decisions:
- **Subagents caught two Western/Orthodox prompt errors**: Ukraine (batch A) and Romania (batch B) inherit from `Calendar::OrthodoxImpl` in C++, not WesternImpl. Agents verified by checking Orthodox-Easter dates in the reference JSON and corrected the inheritance. Documented inline in each module.
- **Saudi Arabia + Israel TelAvivImpl** subclass `Calendar` directly (not `WesternCalendar`) because their weekend is Fri+Sat, not Sat+Sun. Batch D agent handled this correctly with a `_is_weekend` override.
- **UnitedStates** has no default constructor in C++ (`explicit UnitedStates(Market)`). The Python port adds `Market.Settlement` as the default to match the probe's reference and typical usage.
- **Multi-market dispatch pattern reused**: every multi-market calendar (China, US, UK, Brazil, Canada, Germany, Russia, Israel, etc.) dispatches on `self._market` inside `_is_business_day` and `name()` — mirrors the pImpl pattern we established for Thirty360, Actual365Fixed, ActualActual.
- **Probe-emit-path quirk**: the `time_calendars_all_probe` executable name translates to JSON path `references/time/calendars/all.json` (single underscore split → single slash). Tests load via key `"time/calendars/all"`.
- **One worktree-isolation gotcha**: 2 of the 5 subagent worktrees spawned at `main` rather than `phase1-A`. Both agents detected the mismatch (their worktree was missing recent commits) and self-corrected via `git reset --hard phase1-A` + `git submodule update`. No data loss, just a minor extra step.

## Stage 5 — First math batch (closed)

| Commit | Subject | Tests added |
|---|---|---|
| `384fa8d` | `feat(math): port Stage 5 first batch — constants, closeness, rounding, factorial, error_function, beta, bernstein, pascal` | 18 |

Outcome: 8 first-math-batch modules under `pquantlib.math` — `constants`, `closeness`, `rounding`, `factorial`, `error_function`, `beta`, `bernstein_polynomial`, `pascal_triangle`. Single commit (small batch, tightly coupled). **329/0/0 → 411/0/0** combined with Stage 4.

Surprises / decisions:
- **Math constants pulled from Python `math` + `sys.float_info`** rather than re-declared as float literals. `M_PI = math.pi`, `QL_EPSILON = sys.float_info.epsilon`. Documented divergence — C++ has `M_PI` as a CPP macro from `<cmath>`; Python's `math` module imports the same C99 constant.
- **Factorial + Beta fallbacks use `math.lgamma`** instead of porting `GammaFunction.logValue` (lives in distributions/, deferred). For tabulated factorials (n ≤ 27) the values are EXACT; for n > 27 the C++ vs Python GammaFunction-divergence is ~1e-9 relative, so those test cases use LOOSE tier with inline rationale.
- **ErrorFunction delegates to `math.erf`** (stdlib C99) rather than reproducing the Sun-Microsystems polynomial fit used by C++. Both agree to ~1e-14 typically; LOOSE tier in tests for safety at extreme |x|.
- **Rounding `Type.None_`** with trailing underscore to avoid the Python `None` keyword collision (RUF100 `noqa: PIE796` originally added but removed when ruff didn't flag it; the suffix stays as the convention).
- **PascalTriangle uses `ClassVar`** for the cache (RUF012). Bootstrap rows 0..3 copied verbatim from C++; iterative `_next_order` extends as needed.
- **Single math probe** rather than per-class probes — 8 modules in one probe file (`math/first_batch_probe.cpp`) emits closeness × 8 cases, rounding × 17 (every Type × digit × precision), factorial × 11, error_function × 15, beta × 14, bernstein × 11, pascal × 9 orders.
