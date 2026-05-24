# Phase 1 L1-A — Pilot cluster (design)

**Date:** 2026-05-23
**Status:** drafted, awaiting ack to start
**Phase:** 1 (L1 math primitives + time + foundations)
**Cluster:** A (pilot — runs alone before B/C/D/E dispatch)
**Predecessor:** `pquantlib-phase0-bootstrap` @ `7e2c6ec`
**Sister-project anchor:** jquantlib `phase2-l1-plan.md` (jquantlib had no L1-A plan doc — A was the implicit pilot subset; pquantlib makes it explicit because the foundation surface is larger here)
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Goal

Establish all the infrastructure that L1-B/C/D/E will depend on, plus a sufficiently broad pilot scope to prove every part of the workflow (probe → reference JSON → tolerance helpers → cross-validated test → port) before parallel clusters dispatch.

L1-A closes when:

1. C++ submodule cloned at `migration-harness/cpp/quantlib/` pinned to `099987f0` and built successfully via `./migration-harness/build-cpp.sh`.
2. Foundation modules landed: `exceptions`, `testing.tolerance`, `testing.reference_reader`, `patterns` (Observer/Observable/LazyObject/Singleton/ObservableSettings/Visitor), `util` (Pair, Std).
3. `time` core landed: enums (Weekday, Month, TimeUnit, Frequency, BusinessDayConvention, DateGeneration), Period, Date, DateParser, PeriodParser, Calendar abstract + impls (NullCalendar, WeekendsOnly, JointCalendar, BespokeCalendar), Schedule, MakeSchedule, IMM, ASX, ECB, TimeGrid, TimeSeries, Series.
4. All 12 day counters landed: DayCounter abstract + Actual360, Actual364, Actual36525, Actual365Fixed, Actual366, ActualActual (all conventions), Business252, OneDayCounter, SimpleDayCounter, Thirty360, Thirty365.
5. All ~45 calendars landed: per-sovereign + per-exchange.
6. First math batch landed: Constants, Closeness, Rounding (Closest/Up/Down/Floor/Ceiling/None), Factorial, BernsteinPolynomial, PascalTriangle, ErrorFunction, Beta.
7. Every test cross-validated against a C++ probe (or, where C++ has no probeable surface, mpmath with inline rationale).
8. `uv run pytest`, `uv run pyright`, `uv run ruff check`, `uv run ruff format --check` all clean.
9. FF-merged to `main`; tag `pquantlib-phase1-l1-A-complete` pushed.

## Scope estimate

| Sub-layer | Count | Notes |
|---|---|---|
| `exceptions` | 1 | LibraryException + `pquantlib.qassert` module (require/fail helpers) |
| `testing` | 2 | tolerance.py + reference_reader.py |
| `patterns` | ~7 | Observer, Observable, LazyObject, Singleton, ObservableSettings, Visitor, Visitable (Python-flavored — see decision log) |
| `util` | ~3 | Pair (or just stop using; see decision log), Std, ComparablePair |
| `time` enums | 6 | Weekday, Month, TimeUnit, Frequency, BusinessDayConvention, DateGeneration |
| `time` core | ~13 | Period, Date, DateParser, PeriodParser, Calendar (abstract), NullCalendar, WeekendsOnly, JointCalendar, BespokeCalendar, Schedule, MakeSchedule, IMM, ASX, ECB, TimeGrid, TimeSeries, Series |
| `daycounters` | 12 | base + 11 concretes |
| `time.calendars` | ~45 | Western + Eastern variants of Calendar; ~43 sovereign/exchange impls |
| `math` first batch | ~10 | Constants, Closeness, Rounding (~5 variants), Factorial, BernsteinPolynomial, PascalTriangle, ErrorFunction, Beta |

**Total: ~99 classes / modules.** A bit lower than the ~120 the Phase 1 design estimated, because the Python translation collapses Java's Visitor/Visitable/PolymorphicVisitor/PolymorphicVisitable family and several inner classes into single modules using `Protocol` / `typing.runtime_checkable`.

## Approach

### Stage decomposition (binding, sequential within L1-A)

L1-A does NOT fan out across sub-worktrees. It's one worktree (`../pquantlib-phase1-A`) executing 5 stages serially:

| Stage | Scope | Why this order |
|---|---|---|
| **0. Harness bootstrap** | Submodule clone + first C++ build + harness verification | Everything that follows depends on the harness working. |
| **1. Foundations** | `exceptions`, `testing` (tolerance + reference_reader), `patterns`, `util` | Tests in subsequent stages need these to exist. Order within stage: exceptions → tolerance → reference_reader → patterns → util. |
| **2. Time core** | Enums → Period → Date → DateParser/PeriodParser → Calendar abstract + Null/Weekends/Joint/Bespoke → **IMM** (reordered before Schedule — Schedule.from_rule validates against `is_imm_date` for ThirdWednesday rule) → Schedule + MakeSchedule → ASX/ECB → TimeGrid/TimeSeries (no Series — C++ v1.42.1 has no such class) | Date is the foundation; every calendar and day counter consumes it. |
| **3. Day counters** | DayCounter abstract → 11 concretes | Independent of calendars, depends only on Date + enums. Land in parallel within stage. |
| **4. Calendars** | Western/Eastern base → ~43 concretes alphabetically | Depend on time core; can be batched by region. |
| **5. First math batch** | Constants → Closeness → Rounding family → Factorial → ErrorFunction → Beta → BernsteinPolynomial → PascalTriangle | Independent of time; proves the harness on numerical (non-date) primitives. |

Each stage closes with a **stage-checkpoint commit** that has the form:

```
feat(<stage>): close L1-A stage <N> — <stage-title>

<N-line summary of what landed>

Verified:
- uv run pytest: <count>/0/0
- uv run pyright: 0 errors
- uv run ruff check: clean
```

Stages may contain multiple intermediate commits (per-class or per-batch). The stage checkpoint is the natural FF-merge unit if mid-stage rollback is ever needed.

### Per-class TDD loop (repeated for every functional change)

1. **Probe.** Write `migration-harness/cpp/probes/<topic>/<class_snake>_probe.cpp` calling the C++ API and printing key outputs at `setprecision(17)`. Add it to `migration-harness/cpp/probes/CMakeLists.txt`.
2. **Reference.** Run `./migration-harness/generate-references.sh <topic>/<class_snake>_probe`. Produces `migration-harness/references/<topic>/<class_snake>.json`. Commit the JSON.
3. **Failing test.** Write the pytest test that loads the JSON and asserts (initially failing). Confirm `uv run pytest -k <test>` is red.
4. **Port.** Implement the Python class with a `# C++ parity:` line citing v1.42.1 source.
5. **Green.** `uv run pytest -k <test>` passes.
6. **Lint+types.** `uv run pyright`, `uv run ruff check`, `uv run ruff format`. Fix.
7. **Commit.** `feat(<topic>): port <ClassName>` (or batch-equivalent for tiny sibling classes that share a probe).

### Exceptions to the probe rule

The following classes get a documented inline exemption from the C++-probe requirement:

- **Pure interface / abstract base classes** with no executable code: Calendar (abstract part), DayCounter (abstract part), Visitor/Visitable Protocols. Tested via subclass behavior.
- **Pure enum classes** with values trivially mirroring C++: Weekday, Month, TimeUnit, Frequency, BusinessDayConvention, DateGeneration. Test fixtures assert each enum value's integer matches C++ via one probe per enum.
- **The harness/testing modules themselves** (tolerance, reference_reader): tested with synthetic fixtures, not C++ probes.
- **The foundation pattern modules** (Observer, Observable, LazyObject, Singleton, ObservableSettings, Visitor): behavioral tests; the C++ implementations are template-heavy and don't expose stable probe surfaces.
- **Calendar impls**: tested via probes that enumerate holidays for a known year range (≥10 representative years per calendar), not per-method.

All other classes (Date, Period, Schedule, DayCounter concretes, math primitives) require C++ probes.

## Decision log (L1-A)

| # | Decision | Why |
|---|---|---|
| 1 | `LibraryException` does NOT print on construction | jquantlib commit `de95bb17` fixed this exact bug in the Java port (the original 2007 code called `QL.error(this)` in ctor, leaking caught-for-control-flow exceptions to stderr). PQuantLib starts clean: `LibraryException(RuntimeError)` with no side effects. CLAUDE.md decision #11 reinforces this. |
| 2 | `pquantlib.qassert` is a module of free functions, not a class | `QL.require(cond, msg)` and `QL.fail(msg)` are the only two helpers; a module is the simplest Python idiom and avoids stateful coupling. Import as `from pquantlib import qassert; qassert.require(...)` or `from pquantlib.qassert import require`. |
| 3 | `Date` is a `@dataclass(frozen=True, slots=True)` over an integer serial day | C++ uses `Date::serialNumber()` integer arithmetic with epoch `1899-12-30` (matching MS Excel) for compatibility. Frozen+slots gives free `__hash__`, `__eq__`, `__lt__`, low memory. Mutation operations (`++` in C++) return new instances. |
| 4 | `Date` epoch: serial 1 = 1901-01-01 (matching `ql/time/date.hpp`) | The Excel-style 1899-12-30 alignment is implementation detail; cross-validation pins both sides to the same convention. |
| 5 | Enums use `enum.IntEnum`, not `enum.Enum` | Boundary-value tests against C++ probes need to compare integer values. C++ enums are integral; Python `IntEnum` matches that semantics. Trade-off: lose strict typing of enum-only ops, but pyright still catches mismatches via `IntEnum` subclass identity. |
| 6 | `Calendar` is an abstract base class (`abc.ABC`) with two protected helpers: `_is_weekend`, `_is_business_day` | Python doesn't have C++'s `Calendar::Impl` PIMPL pattern. Direct inheritance is fine. Western vs Eastern weekend convention captured by overriding `_is_weekend`. |
| 7 | `JointCalendar` uses `enum.IntFlag` for join rules (Joining/JoinHolidays/JoinBusinessDays) | Mirrors C++. |
| 8 | `Observer`/`Observable` use `weakref.WeakSet` for observers | Mirrors C++ boost::signals2 weak-binding to prevent cycles. The original jquantlib `DefaultObservable`/`WeakReferenceObservable` split is collapsed to one `Observable` class. |
| 9 | `Pair` / `ComparablePair` are NOT ported as classes; the C++ uses get `tuple[A, B]` | Python tuples already support comparison, hashing, unpacking. Only if a specific call site needs name fields does it get a `@dataclass(frozen=True, slots=True)`. Documented inline at every C++ `Pair<A,B>` translation site. |
| 10 | `Visitor`/`Visitable` are `Protocol`s, not abstract classes | Visitor pattern in Python is naturally structural. `runtime_checkable` Protocol gives both static (pyright) and runtime checks. The `PolymorphicVisitor`/`PolymorphicVisitable` Java split disappears (Python's duck typing makes it unnecessary). |
| 11 | `LazyObject` is a mixin class with `_calculated: bool` + `_calculate(self) -> None` template method | C++ uses CRTP; Java uses abstract method. Python uses an explicit boolean cache + `calculate()` method that calls `_calculate()` if `not self._calculated`. |
| 12 | `Singleton` uses metaclass `_SingletonMeta` (not `__init_subclass__`) | Metaclass gives the cleanest "single instance per class" semantics; `_instances: dict[type, Any]` keyed by subclass. Test: subclass `S` has `S() is S()` true. |
| 13 | `ObservableSettings` is a `Singleton` holding global mutation flags (e.g., `enforces_business_day_convention`) | Mirrors C++ `Settings::instance()`. |
| 14 | `TimeSeries` is a wrapper over `dict[Date, T]` with `__getitem__` interpolation hooks | C++ uses `std::map<Date, T>`; Python `dict` (since 3.7) preserves insertion order. Wrapper layer adds bisect-based lookup for "value as of date" queries. |
| 15 | Calendar holiday tests use a "year-range enumeration" probe (10+ years per calendar) | Probing every individual method (`isBusinessDay(date)`) per calendar is wasteful. One probe per calendar emits a JSON of `{year: [list of holiday dates]}` for 2000–2030; the test asserts the same set. This bulk-validation gives stronger coverage per LOC. |
| 16 | `DayCounter.year_fraction` is the only required method on the abstract class | C++'s `dayCount` is derived from year_fraction × constants; expose both as methods with default impls where possible. |
| 17 | `Schedule` is a `@dataclass(frozen=True, slots=True)` carrying `tuple[Date, ...]` + the original generation params | C++ stores both the date list and the rule; preserving the rule in the dataclass means `Schedule.previous_date(d)` / `next_date(d)` can re-derive without external state. |
| 18 | First math batch uses `numpy` for vectorized ops where natural (`Factorial.cached(n)`, `BernsteinPolynomial.eval_array(...)`) | Idiomatic Python; performance free. Document divergence where the C++ uses scalar loops. |
| 19 | `mpmath` is the ground truth for `ErrorFunction` and `Beta` when C++ ground truth disagrees with mpmath at TIGHT tolerance | Per Phase 1 pause trigger A3' — if C++ and mpmath disagree, log it and pin the test to mpmath (likely C++ approximation rounding). |
| 20 | Calendar implementations use **module-level holiday lists** that the abstract method consults | Per-instance state is wasteful (calendars are stateless). Class attribute or module-level frozen-set. Saves memory if many `UnitedStates()` instances exist. |

## Tolerance assignments (L1-A defaults)

| Test category | Tier | Justification |
|---|---|---|
| Date arithmetic (`d + 7 == d2`, `d.serial_number()`, `d.weekday()`) | EXACT | All integer-valued under the hood. |
| Period parsing/composition (`Period(7, Days).years() == 7/365.25` etc.) | TIGHT | Float division but determinate. |
| Calendar holiday enumeration | EXACT (set equality) | Holiday dates are integer-valued. |
| DayCounter year_fraction | TIGHT | Float division; conventions encoded by C++ exactly. |
| ActualActual.ISMA corner cases | LOOSE (per-test justification inline) | Known floating-point sensitivities in C++ around month-end boundaries; mpmath confirms LOOSE is the true achievable precision. |
| Rounding | EXACT | Bit-deterministic. |
| Factorial.tabulated(n) for n ≤ 20 | EXACT | Integer values up to 20!. |
| Factorial.ln_factorial(n) | TIGHT | log(n!) via Stirling for n > 170. |
| ErrorFunction, Beta | TIGHT (with mpmath cross-check) | Closed-form transcendental. |
| BernsteinPolynomial, PascalTriangle | EXACT for integer binomials, TIGHT for evaluation at non-integer x | |

## Pause triggers (in addition to Phase 1's A1'–A6)

| ID | Condition | Action |
|---|---|---|
| A_LA_1 | C++ submodule clone fails (network, gh auth, submodule lock) | Pause, ask user for credentials check |
| A_LA_2 | First C++ build (`build-cpp.sh`) fails | Pause — likely environment (missing boost, cmake version, compiler). Don't proceed; environment-fix is a user concern. |
| A_LA_3 | A foundation module's API shape needs to differ from C++ in a way that ripples (e.g., Observer signature changes) | Pause, document divergence design, get ack before propagating |
| A_LA_4 | Calendar holiday probe disagrees with a public-record holiday list (e.g., wrong Easter date) | Pause — this is potentially an A3' (v1.42.1 bug). Verify against an external authoritative source before deciding. |
| A_LA_5 | Audit during stage 1–2 reveals scope materially differs from estimate (±25%) | Pause, re-allocate B–E plan estimates |

## Definition of done (L1-A)

- [ ] Stage 0: C++ submodule cloned, built, first probe runs and emits a parseable JSON
- [ ] Stage 1: foundations land — `exceptions`, `testing.tolerance`, `testing.reference_reader`, `patterns.*`, `util.*`
- [ ] Stage 2: time core lands — enums + Period + Date + Date/PeriodParser + Calendar abstract + 4 trivial calendars + Schedule + MakeSchedule + IMM/ASX/ECB + TimeGrid/TimeSeries/Series
- [ ] Stage 3: 12 day counters land
- [ ] Stage 4: ~43 sovereign/exchange calendars land
- [ ] Stage 5: 10 first-math-batch classes land
- [ ] `uv run pytest`: green across all tiers, count ≥ 300 (rough estimate: ~3 tests per class × 100 classes)
- [ ] `uv run pyright`: 0 errors
- [ ] `uv run ruff check` + `uv run ruff format --check`: clean
- [ ] Every ported class has `# C++ parity:` line citing v1.42.1 source
- [ ] FF-merged to `main`
- [ ] Tag `pquantlib-phase1-l1-A-complete` pushed

## Carry-outs (deferred from L1-A to L1-B–E or later phases)

Document explicitly if any of these slip:

- Specific calendar with unusual holiday rules (e.g., `SaudiArabia` Hijri calendar) — may defer to a follow-up cluster with rationale.
- `MultidimIntegral` or other math primitives that turn out to only be used by L2+. Defer if appropriate.

## Next-step preview

Once L1-A is acked:

1. Spawn worktree at `../pquantlib-phase1-A` off current `main`.
2. Execute `phase1-l1-A-plan.md` task-by-task in that worktree.
3. Review (spec-compliance + code-quality) + fix-up loop.
4. FF-merge.
5. Draft B/C/D/E design+plan docs using L1-A's audit refinements.
