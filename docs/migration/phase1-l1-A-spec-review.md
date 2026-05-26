# Phase 1 L1-A — Spec-compliance review

**Date**: 2026-05-26
**Reviewer**: spec-compliance subagent
**Range**: `main..phase1-A` (47 commits)
**Worktree**: `/Users/josemoya/Projects/PycharmProjects/pquantlib-phase1-A`
**Final test count**: 411/0/0 (re-verified live)
**Final pyright/ruff**: clean (0 errors / 0 lint / 177 files formatted)

## TL;DR

PASS WITH FIXUPS. The L1-A cluster meets every binding correctness rule from `CLAUDE.md` / `phase1-design.md` / `phase1-l1-A-design.md`: zero `Co-authored-by: Claude` trailers, every ported class carries a `# C++ parity:` reference, every functional test loads from `migration-harness/references/...`, every `tolerance.loose(...)` use has inline justification, all the Western/Orthodox calendar inheritance flagged in the progress log is correct, and `uv run pytest / pyright / ruff check / ruff format --check` all run green on the tip commit. Two process-discipline gaps would be NITs to fix before the next cluster but do not block FF-merge: the five `--no-ff` calendar batch merge commits are unsigned (no `Signed-off-by:` trailer carried through `git merge --no-ff`), and the 13 individual subagent calendar commits inside those merges omit the `uv run pytest/pyright/ruff` verification fence (Stage 4 progress log states the subagents verified locally before pushing; the fence was simply not transcribed). One stale-doc NIT: `business_252.py` still says "Brazil is not yet ported" although `brazil.py` landed in Stage 4. One design-vs-implementation discrepancy: `Schedule` ships as a regular class with `__init__`, not a `@dataclass` as design Decision #17 specified — but the divergence is documented inline.

## Findings

### 1. Five merge commits are unsigned [NIT]

The five `--no-ff` merge commits that integrated the Stage 4 subagent calendar batches into `phase1-A` carry no `Signed-off-by:` trailer. The underlying per-calendar commits inside each merge ARE signed (the subagents signed each `feat(calendars): port <X>` commit individually before pushing their isolated branches). `git merge --no-ff` by default does not propagate a `-s` to the merge commit even when both branch heads are signed, so the merge itself has to be re-signed explicitly.

**Evidence**:
- `5a0b1a9` `merge: batch A (10 European Western calendars + Orthodox Ukraine)`
- `94ea196` `merge: batch B (Western Europe + Mexico + Orthodox Romania)`
- `400540a` `merge: batch C (Americas + Oceania + Africa)`
- `8d779b4` `merge: batch D (Middle East + Russia/Orthodox + East Asia)`
- `239079b` `merge: batch E (China + HongKong + India + Indonesia + SouthKorea + Thailand + UK + US)`

42 of 47 commits ARE signed. The closing commit `2ffeab43` is also signed.

**Recommendation**: For future clusters, pass `-s` to `git merge --no-ff` (or set `commit.gpgsign` analogue for `merge.signoff`). For L1-A, an option is to soft-reset the five merge tips and re-create with `-s`, but that rewrites the history that subagents already pushed. Pragmatic: accept the five unsigned merges as an FF-merge precondition, add a paragraph to `phase1-l1-A-completion.md` acknowledging the gap, and put a check in the dispatch protocol for B-E.

### 2. Stage 4 subagent commits omit the verification fence [NIT]

The 13 per-calendar feat commits emitted by the Stage 4 subagent worktrees (`d5823f10` through `03330bf5`) do not include the `Verified: uv run pytest / pyright / ruff check / ruff format --check` fence inside the commit body. The progress log (`phase1-l1-A-progress.md` Stage 4 section, line 180) explicitly states each subagent ran pytest+pyright+ruff in their isolated worktree before pushing, but the fence text was not transcribed into the commit messages.

**Evidence**: every commit between `c9d93f27` (batch A subagent's first feat commit) and `d5823f10` (batch E US commit) — see the `--format='%B'` dump for any of those hashes.

The five integration commits authored on `phase1-A` directly (`c9d93f27`, `8789e62d`, `4f56e76c`, `68b7fe0c`) DO carry the fence — those were the maintainer's commits on the main worktree's HEAD that integrated each batch's audit + fixups.

**Recommendation**: For future parallel-cluster dispatch, prompt the subagent template to append the fence to every commit body. Non-blocking for L1-A — the tip-of-`phase1-A` is verifiably green at 411/0/0.

### 3. Two infra commits have only partial fences [NIT]

- `557b7bf` `fix(harness): add Boost include path to probes CMakeLists` — relies on the next commit (`5a82562` sentinel probe) to demonstrate it works. Body says "Verified by Stage 0.3 sentinel probe (next commit)" but does not list pytest/pyright/ruff (none ran because there was no code change to test).
- `b592349` `fix(infra): exclude QuantLib submodule from ruff format` — has only `uv run ruff format --check: 13 files already formatted` (no pytest/pyright lines because there was no code change). This is the right minimum but doesn't match the canonical 4-line fence.

**Evidence**: see commit bodies via `git show <hash>`.

**Recommendation**: None — these are minor infra commits with the right level of verification for what they changed. Document the precedent in `phase1-l1-A-completion.md` that pure-infra commits may include only the relevant fence line.

### 4. `Business252` divergence note is stale [NIT]

`pquantlib/src/pquantlib/daycounters/business_252.py` lines 8-11 says "The C++ default constructor uses `Brazil()` calendar. Brazil is not yet ported (Stage 4); the Python port requires an explicit calendar." Brazil DID land in Stage 4 (`pquantlib/src/pquantlib/time/calendars/brazil.py`). The decision to keep `Business252(calendar)` as a required-arg constructor is defensible (explicit > implicit-via-default-Brazil) but the rationale should be re-cast as a deliberate Python idiom choice, not a "not yet ported" temporary.

**Evidence**: `pquantlib/src/pquantlib/daycounters/business_252.py:8-11` vs `pquantlib/src/pquantlib/time/calendars/brazil.py` (present).

**Recommendation**: Re-word the divergence block in `business_252.py` to make explicit that requiring `calendar` is the intentional Python idiom (no hidden-default-Brazil surprise) — Brazil is now importable for anyone who wants `Business252(Brazil())`.

### 5. `Schedule` deviates from design Decision #17 [OK — documented]

L1-A design Decision #17 specified `Schedule` as `@dataclass(frozen=True, slots=True)`. The implementation ships as a regular class with an `__init__` method that mutates state during rule-based generation. The rationale is documented inline (`pquantlib/src/pquantlib/time/schedule.py:13-18`): "Schedule is a regular class with `__init__` (not a `@dataclass`) because the rule-based constructor mutates internal state during generation. Once constructed it is immutable from the caller's point of view (`dates` is exposed as a `tuple`...)."

**Evidence**: `pquantlib/src/pquantlib/time/schedule.py:13-18` (docstring) vs `phase1-l1-A-design.md` Decision #17.

**Recommendation**: None — documented divergence. Update the L1-A design log retrospectively (or in `phase1-l1-A-completion.md`) so the next reader knows the as-built shape.

### 6. `Date` design log says "serial 1 = 1901-01-01" but C++ has serial 367 = 1901-01-01 [OK — typo in design]

L1-A design Decision #4 says "Date epoch: serial 1 = 1901-01-01 (matching ql/time/date.hpp)". The C++ source (`migration-harness/cpp/quantlib/ql/time/date.cpp:749-755`) says `minimumSerialNumber() == 367` = Jan 1, 1901. The Python port correctly uses serial 367 for 1901-01-01 (`pquantlib/src/pquantlib/time/date.py:38`). Decision #3 in the same log states the Excel-style 1899-12-30 epoch correctly. Decision #4's wording is a typo, not a code bug.

**Evidence**: `phase1-l1-A-design.md` Decision #4 vs `pquantlib/src/pquantlib/time/date.py:5-8` + C++ source.

**Recommendation**: Fix the design doc wording — easy and clarifies for future readers. Code is correct.

### 7. Western/Orthodox calendar inheritance is correct [OK]

Reviewer-flagged concerns are all clean:

- `Ukraine` (`pquantlib/src/pquantlib/time/calendars/ukraine.py:38`) subclasses `OrthodoxCalendar`.
- `Romania` (`pquantlib/src/pquantlib/time/calendars/romania.py:31`) subclasses `OrthodoxCalendar`.
- `Russia` (`pquantlib/src/pquantlib/time/calendars/russia.py:158, 197`) subclasses `OrthodoxCalendar` for both Settlement + Exchange impls.
- `SaudiArabia` (`pquantlib/src/pquantlib/time/calendars/saudi_arabia.py:112`) subclasses `Calendar` directly and provides `_is_weekend` returning `Friday or Saturday`.
- `Israel.TelAviv` (`pquantlib/src/pquantlib/time/calendars/israel.py:411,417-418`) subclasses `Calendar` with `_is_weekend` returning `Friday or Saturday`; `Israel.Shir` subclasses `WesternCalendar` (Sat+Sun) per the C++ source.

**Recommendation**: None — meets spec.

### 8. Every ported source file has `# C++ parity:` [OK]

`grep -L '# C++ parity:' pquantlib/src/pquantlib/**/*.py` excluding `__init__.py` returns empty. The harness modules (`testing/tolerance.py`, `testing/reference_reader.py`) correctly mark themselves `# C++ parity: none — this is harness, not a port.`. The `__init__.py` files are short re-export modules with `# C++ parity:` references at the subpackage level.

**Recommendation**: None — meets spec.

### 9. Every functional test loads from `references/...` [OK]

All 15 functional test files (`test_date.py`, `test_period.py`, `test_schedule.py`, `test_enums.py`, `test_imm.py`, `test_asx_ecb.py`, `test_parsers.py`, `test_calendar.py`, `test_time_grid_and_series.py`, `test_first_batch.py`, `test_actual_family.py`, `test_thirty_family.py`, `test_simple_business.py`, `test_actualactual.py`, `test_one_day_counter.py`) call `reference_reader.load(...)`. All 41 calendar test files load the shared `time/calendars/all` reference. Behavioral tests (`test_observer.py`, `test_singleton.py`, etc.) correctly omit the reference loader because they exercise Python-side semantics with no C++ probeable surface — this matches the L1-A "Exceptions to the probe rule" design clause.

**Recommendation**: None — meets spec.

### 10. Every `tolerance.loose(...)` use has inline justification [OK]

Only `pquantlib/tests/math/test_first_batch.py` uses `loose()` in functional tests; each call is preceded by a 2-6 line comment block citing the specific `lgamma`-vs-`GammaFunction.logValue` ULP divergence, the Sun-Microsystems polynomial fit divergence, or the Lentz-continued-fraction precision floor. `tolerance.custom(...)` is only used in `test_tolerance.py` (self-test), with `reason=` keyword-only argument required.

**Recommendation**: None — meets spec.

### 11. No `Co-authored-by: Claude` trailers [OK]

`git log main..HEAD --format='%B' | grep -ic 'co-authored'` returns `0`.

**Recommendation**: None.

### 12. Probe-validated tests exist for every functional class [OK]

Cross-check of `pquantlib/src/pquantlib/` (non-test) against `pquantlib/tests/` shows every ported module is covered either by a same-name test or by an aggregated test (`test_first_batch.py` covers `constants/closeness/rounding/factorial/error_function/beta/bernstein_polynomial/pascal_triangle`; `test_actual_family.py` covers `actual_360/364/365_25/365_fixed/366`; `test_thirty_family.py` covers `thirty_360/thirty_365`; `test_simple_business.py` covers `simple_day_counter` + `business_252`; `test_calendar.py` covers `null_calendar/weekends_only/joint_calendar/bespoke_calendar`; `test_enums.py` covers all six time enums; `test_parsers.py` covers both `date_parser` + `period_parser`; `test_asx_ecb.py` covers both; `test_time_grid_and_series.py` covers both). Reference JSONs match (`references/time/calendars/all.json` for calendars; per-family probes for day counters; `references/math/first/batch.json` for the math batch).

**Recommendation**: None — meets spec.

## Checklist results

- [x] # C++ parity: lines present on all ported classes (including subpackage `__init__.py` aggregate refs and explicit "none — harness" markers)
- [x] Tests load from `references/...` (15 functional test files, all calendars via shared `time/calendars/all`)
- [x] Tolerance tiers justified (every `tolerance.loose` use in `test_first_batch.py` has 2-6 lines of inline rationale)
- [PARTIAL] All commits have verification fence (5 `--no-ff` merge bodies + 13 subagent feat commits lack the canonical fence; subagents verified in their worktrees per progress log but didn't transcribe the fence)
- [x] No `Co-authored-by:` trailers (grep returns 0)
- [PARTIAL] All commits signed (42/47 signed; the 5 unsigned ones are `--no-ff` calendar batch merges)
- [x] Divergences documented inline (lgamma vs GammaFunction, erf vs Sun-Microsystems polynomial, DateParser.parse_formatted, Schedule.from_rule eval-date fallback, Business252 default-Brazil, TimeSeries no-Series, Schedule non-dataclass, Date Month(13) widening, Date __add__/__sub__ overloads, ECB 200-entry table, Western/Orthodox 299-entry tables, Singleton metaclass dict[Any, Any], Visitor non-generic Protocol, UnitedStates default Settlement market, SaudiArabia + Israel-TelAviv Fri+Sat weekend, 30/360 USA Feb-end check ordering)
- [x] Design decisions implemented as written (only Schedule deviates from #17 with inline doc; all other decisions including #4 Date custom class, #5 IntEnum, #6 abc.ABC Calendar, #8 weakref.WeakSet Observable, #11 LazyObject mixin, #12 Singleton metaclass, #14 TimeSeries dict[Date, T]+PEP-695 generics, #17 Schedule [documented deviation])
- [x] Calendar Western/Orthodox inheritance correct (Ukraine + Romania + Russia × 2 → OrthodoxCalendar; Saudi Arabia + Israel-TelAviv → Calendar with Fri+Sat override; Israel-Shir → WesternCalendar per C++)
- [x] Probe-validated tests for every functional class (15 functional test files; behavioral patterns/tolerance/reference_reader correctly exempt per design)

## Divergence catalogue

Complete list of C++ → Python divergences documented inline across L1-A:

| # | Location | Divergence | Rationale |
|---|---|---|---|
| 1 | `math/factorial.py:7-10` | `math.lgamma` for n > 27 instead of `GammaFunction.logValue` | GammaFunction is in `ql/math/distributions/`, deferred to a later cluster; stdlib `math.lgamma` is C99-equivalent. LOOSE tier in tests. |
| 2 | `math/beta.py:6-8` | `math.lgamma` again | Same as #1. |
| 3 | `math/error_function.py:10-12` | `math.erf` (C99) instead of Sun-Microsystems polynomial fit | Both agree to ~1e-14 typically; LOOSE tier for safety at extreme \|x\|. |
| 4 | `math/constants.py` | `M_PI = math.pi`, `QL_EPSILON = sys.float_info.epsilon`, etc. | Pulled from stdlib rather than re-declared as float literals. Same C99 source. |
| 5 | `math/rounding.py:43` | `Type.None_` with trailing underscore | Avoids Python `None` keyword collision. |
| 6 | `math/pascal_triangle.py` | `ClassVar` cache (RUF012-clean) | Bootstrap rows 0..3 verbatim from C++. |
| 7 | `time/date.py:144-153` | `_month_offset(m: Month | int, leap: bool)` accepts `int` for `Month(13)` widening | C++ allows unchecked enum cast to `Month(13)` as the year-end bracket; Python `IntEnum` rejects, so the helper accepts `int`. |
| 8 | `time/date.py:257-300` | `@overload` signatures on `Date.__add__` and `__sub__` | Pyright cannot narrow `Date | int` return from runtime dispatch alone; @overload pure type-narrow, no runtime change. (`align(time): ...` commit.) |
| 9 | `time/date_parser.py:9-15` | `datetime.strptime` format codes instead of boost::date_time facet format strings | Common codes `%Y %m %d` behave identically; boost-specific codes not needed for the probe round-trip. |
| 10 | `time/schedule.py:13-18` | Regular class with `__init__`, not `@dataclass(frozen=True, slots=True)` | Rule-based constructor mutates state during generation; immutable from caller's perspective via `tuple` exposure. Deviates from L1-A design Decision #17 with inline rationale. |
| 11 | `time/schedule.py:21-23` | `Settings::evaluationDate()` fallback for null `effective_date` in Backward rule is NOT ported | Python callers must pass an explicit effective_date. |
| 12 | `time/time_series.py:11-13` | No `Series` module | `Series` is a jquantlib-port artifact; C++ v1.42.1 has only `TimeSeries`. |
| 13 | `time/time_series.py:25` | `TimeSeries[T]` PEP 695 generic over `dict[Date, T]` | Python 3.14 native generic syntax replaces C++ template + `std::map` container parameter. |
| 14 | `time/ecb.py:43-244` | 200-entry hardcoded `_KNOWN_DATE_SERIALS` table | Verbatim port of C++ `ecbKnownDateSet`. |
| 15 | `time/calendar.py:305+` and `_WESTERN_EASTER_MONDAY` table | 299-entry hardcoded Western Easter Monday table | Verbatim port. |
| 16 | `time/calendar.py:307+` (Orthodox) | 299-entry hardcoded Orthodox Easter Monday table | Verbatim port. |
| 17 | `time/calendar.py` `Calendar(ABC)` direct inheritance | No PIMPL Bridge pattern | Python doesn't need the C++ `Calendar::Impl` pImpl trick. Per-instance `added_holidays` / `removed_holidays`. |
| 18 | `daycounters/business_252.py:8-11` | Required `calendar` constructor argument (no default) | C++ defaults to `Brazil()`. Python port keeps the requirement explicit. **NOTE: doc says "Brazil not yet ported" but Brazil landed in Stage 4 — see finding #4.** |
| 19 | `daycounters/business_252.py:12-17` | No module-level monthly/yearly business-day cache | Delegate directly to `Calendar.business_days_between`. Cache addable if profiling identifies as a hotspot. |
| 20 | `daycounters/thirty_360.py:35` | Single class dispatching on `Convention` IntEnum, not 6 Impl pImpl classes | C++ pImpl pattern collapsed; aliases (ISMA ≡ BondBasis etc.) become shared dispatch + name() output. |
| 21 | `daycounters/actual_365_fixed.py:47` | Single class with `Convention` IntEnum dispatch | Same pattern as Thirty360 (3 impls collapsed). |
| 22 | `daycounters/actual_actual.py:36` | Single class with `Convention` IntEnum dispatch | Same pattern (4 impls collapsed, 7 conventions: ISMA, Bond, ISDA, Historical, Actual365, AFB, Euro). |
| 23 | `time/calendars/united_states.py:14` | `Market.Settlement` default for `UnitedStates()` | C++ has no default constructor (`explicit UnitedStates(Market)`). Python adds Settlement as default to match the probe. |
| 24 | `time/calendars/saudi_arabia.py:5-11, 124-126` | Subclasses `Calendar` directly with `_is_weekend(Friday or Saturday)` | Friday+Saturday weekend. Plus historical: weekend was Thursday+Friday before 29-June-2013 — `_is_true_weekend` consults date. |
| 25 | `time/calendars/israel.py:5-8, 417-418` | TelAviv + Settlement markets subclass `Calendar` with `_is_weekend(Friday or Saturday)`; Shir subclasses `WesternCalendar` | Mirrors C++ — Tel Aviv works on Sunday, Shir follows ECB. |
| 26 | `time/calendars/ukraine.py:5-7, 38` | Subclasses `OrthodoxCalendar` | C++ derives from `Calendar::OrthodoxImpl` (not WesternImpl). Stage 4 subagent caught this. |
| 27 | `time/calendars/romania.py:5-8, 31` | Subclasses `OrthodoxCalendar` | Same as Ukraine — C++ Public impl derives from OrthodoxImpl. |
| 28 | `time/calendars/russia.py:5-7, 158, 197` | Both Settlement + Exchange impls subclass `OrthodoxCalendar` | Russian Orthodox Easter table. |
| 29 | `daycounters/thirty_360.py:103-140 `_dc_us`` | "Order of checks is important" comment | USA Feb-end ordering rule preserved verbatim from C++ source. |
| 30 | `patterns/singleton.py:20-33` | `_SingletonMeta` metaclass with `_instances: dict[Any, Any]` (not `dict[type, Any]`) | Pyright sees `cls` inside `__call__` as `Self@_SingletonMeta`, not `type` — so the dict key annotation has to widen to `Any`. |
| 31 | `patterns/visitor.py:6-11` | Single non-generic `Visitor` Protocol + `Visitable` Protocol | C++/Java four-class Visitor/Visitable/PolymorphicVisitor/PolymorphicVisitable hierarchy collapses to two Protocols. The generic `T` parameter is dropped because Python structural typing fires on method-name matching. |
| 32 | `patterns/observer.py` | `weakref.WeakSet` of observers (not boost::signals2) | Mirrors C++ semantic; cycle-free observer registration. |
| 33 | `exceptions.py:9-22` | `LibraryException(RuntimeError)` with NO side effects in `__init__` | Explicitly avoids the jquantlib pre-de95bb17 bug where `QL.error(this)` in ctor leaked to stderr. |
| 34 | `qassert.py` | Module of free functions (`require`, `fail`) rather than a class with static methods | Cleanest Python idiom; no stateful coupling. |
| 35 | All time/calendar/daycounter enums | `IntEnum` (not `Enum`) | Boundary tests need integer comparison with C++ probes. 30+ IntEnum classes in the cluster. |
| 36 | Joint calendars | `JointCalendarRule = IntFlag` | Mirrors C++ bitfield-style flags (Joining/JoinHolidays/JoinBusinessDays). |

## Process gaps to address before L1-B/C/D/E dispatch

1. Subagent commit template must include the full 4-line verification fence (`uv run pytest: N/0/0` / `uv run pyright: 0 errors` / `uv run ruff check: clean` / `uv run ruff format --check: ...`).
2. `git merge --no-ff` of subagent branches should pass `-s` for the Signed-off-by trailer on the merge commit itself (the underlying commits being signed is not enough).
3. Optional: have the dispatching session run `phase1-l1-A-spec-review` (this doc) as a per-cluster gate before tagging `pquantlib-phase1-l1-A-complete`.
