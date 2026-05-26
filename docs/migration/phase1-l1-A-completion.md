# Phase 1 L1-A — Completion summary

**Closed:** 2026-05-24
**Tag:** [`pquantlib-phase1-l1-A-complete`](../../../../releases/tag/pquantlib-phase1-l1-A-complete) @ `03d0ce8`
**Predecessor tag:** `pquantlib-phase0-bootstrap` @ `85018e5`
**Branch:** `phase1-A` (FF-merged to `main`, branch deleted local + remote)
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Sister project parity:** `jquantlib-final`

## Final state

- **51 commits** on `phase1-A`, all signed (`-s`), zero `Co-authored-by: Claude` trailers.
- **415 / 0 / 0** pytest, **pyright strict** clean, **ruff lint + format** clean.
- **~5000 reference values** committed under `migration-harness/references/`.
- **11 C++ probe binaries** in `migration-harness/cpp/probes/`.
- **177 Python files** under `pquantlib/`, all formatted.

## What landed (Stages 0–5)

### Stage 0 — Harness bootstrap

- `migration-harness/cpp/quantlib/` cloned as a git submodule pinned at `099987f0`.
- `migration-harness/cpp/probes/CMakeLists.txt` Boost include path fix (caught only when the first real probe linked).
- Sentinel probe `harness_sentinel_probe` emits `references/harness/sentinel.json` with `QL_VERSION`, `sqrt(2)`, `pi` at full precision.

### Stage 1 — Foundations

- `pquantlib.exceptions.LibraryException` — side-effect-free constructor (deliberate, per Phase 0 decision #11; jquantlib pre-`de95bb17` had a stderr leak).
- `pquantlib.qassert.require` / `.fail` — module of free functions, no `QL` namespace class.
- `pquantlib.testing.tolerance.{exact, tight, loose, custom}` — assertion helpers, mandatory `reason` for `custom`.
- `pquantlib.testing.reference_reader.load("<topic>/<class>")` — upward-walk JSON loader resilient to worktree layout.
- `pquantlib.patterns.{observer, singleton, lazy_object, observable_settings, visitor}` — 5 modules.

### Stage 2 — Time core

- **6 IntEnums**: `Weekday`, `Month`, `TimeUnit`, `Frequency`, `BusinessDayConvention`, `DateGeneration`.
- **`Period`** — `@dataclass(frozen=True, slots=True)` over `(length, units)`; `from_frequency` alt-constructor; full arithmetic; `normalized()`; `years/months/weeks/days` free functions.
- **`Date`** — `@dataclass(frozen=True, slots=True)` over `serial: int`; Excel-compatible epoch (1899-12-30) so the C++ `yearOffset` table aligns; valid range `[367, 109574]` = 1901-01-01..2199-12-31; `from_ymd` classmethod; full inspector / arithmetic / `nth_weekday` / `end_of_month` API; `__add__`/`__sub__` `@overload`-annotated so `Date - 1` narrows to `Date` (not `Date | int`).
- **`DateParser` + `PeriodParser`** — module-of-free-functions form. `parse_formatted` uses Python `strptime` codes (documented divergence from boost::date_time).
- **`Calendar`** abstract (`abc.ABC`) + `WesternCalendar` / `OrthodoxCalendar` bases — 299-entry Easter Monday tables (1901–2199) embedded verbatim from C++.
- **4 trivial calendar concretes**: `NullCalendar`, `WeekendsOnly` (preserving the lowercase "weekends only" name from C++), `JointCalendar` with `JointCalendarRule` IntEnum, `BespokeCalendar` with bitfield weekend mask.
- **`Schedule`** + **`MakeSchedule`** — generates date lists for all 10 `DateGeneration` rules. Documented divergence: `Settings.evaluation_date` fallback not ported.
- **`IMM`, `ASX`, `ECB`** — date helpers as module-of-free-functions. ECB embeds the 200-entry known-dates table (2005–2024) verbatim from C++.
- **`TimeGrid`** — 3 construction modes (regular / mandatory-only / mandatory-with-steps); **`TimeSeries[T]`** PEP-695 generic over `dict[Date, T]`.

### Stage 3 — Day counters (12 modules, 24+ convention aliases)

- `DayCounter` abstract base with name-based equality.
- `OneDayCounter` (1/1, sign-only day count).
- `Actual360` / `Actual364` / `Actual36525` / `Actual366` with optional `include_last_day`.
- `Actual365Fixed` with `Convention` IntEnum (Standard / Canadian / NoLeap). Canadian Bond requires explicit `ref_period_{start,end}`.
- `Thirty360` with `Convention` IntEnum (USA / BondBasis / European / EurobondBasis / Italian / German / ISMA / ISDA / NASD) — 9 conventions collapsing to 6 distinct algorithms via alias.
- `Thirty365` (ISO 20022 30/365).
- `SimpleDayCounter` (whole-month clean fractions + Thirty360 BondBasis fallback).
- `Business252` (calendar-dependent; documented divergence: drops the C++ `Brazil()` default and the in-memory month/year caches).
- `ActualActual` with `Convention` IntEnum (ISMA / Bond / ISDA / Historical / Actual365 / AFB / Euro) — 7 conventions collapsing to 4 underlying algorithms: ISDA family (leap-year fraction), AFB family (1-year-walk with Feb-28 leap bump), `Old_ISMA` (recursive, no schedule), schedule-based ISMA (quasi-coupon dates + per-period overlap-day fractions).

### Stage 4 — 41 sovereign / exchange calendars (parallelized)

Default-market for each of: Argentina, Australia, Austria, Botswana, Brazil, Canada, Chile, China, CzechRepublic, Denmark, Finland, France, Germany, HongKong, Hungary, Iceland, India, Indonesia, Israel, Italy, Japan, Mexico, NewZealand, Norway, Poland, Romania, Russia, SaudiArabia, Singapore, Slovakia, SouthAfrica, SouthKorea, Sweden, Switzerland, Taiwan, TARGET, Thailand, Turkey, Ukraine, UnitedKingdom, UnitedStates.

**Stage 4 was parallelized via 5 isolated-worktree subagents.** See "Lessons learned" below.

### Stage 5 — First math batch (8 modules)

- `pquantlib.math.constants` — `M_PI / M_E / M_SQRT2 / ...` from `math`; `QL_EPSILON / QL_MAX_REAL / ...` from `sys.float_info`.
- `pquantlib.math.closeness` — `close(x, y, n=42)` (AND) + `close_enough(x, y, n=42)` (OR) Knuth-style predicates with one-side-zero + equal-input + infinity short-circuits.
- `pquantlib.math.rounding` — `Rounding(precision, Type, digit=5)` + `Type` IntEnum (None_ / Up / Down / Closest / Floor / Ceiling) + 5 convenience subclasses. fast_pow10 LUT-backed.
- `pquantlib.math.factorial` — tabulated 0..27 verbatim from C++; fallback uses `math.lgamma`. LOOSE-tier divergence ~1e-9 for n > 27.
- `pquantlib.math.error_function` — `ErrorFunction()(x)` delegates to `math.erf`. LOOSE-tier for safety.
- `pquantlib.math.beta` — `beta_function(z, w)` via `lgamma`; `beta_continued_fraction(a, b, x, ...)` Lentz iteration; `incomplete_beta_function(a, b, x, ...)`.
- `pquantlib.math.bernstein_polynomial` — `B_i^n(x) = C(n,i) x^i (1-x)^(n-i)` via Factorial.
- `pquantlib.math.pascal_triangle` — ClassVar-cached lazy row generator.

## Lessons learned

### 1. Subagent fan-out for parallelizable stages — pattern works

**Stage 4** (41 calendars) ran across **5 isolated-worktree subagents in parallel** for ~25 min wall-clock. Estimated 6+ hours sequential.

The pattern that worked:
1. Emit a single mega-probe + JSON containing reference values for *all* units of work at once.
2. Commit + push that probe + JSON before fanning out, so subagents see it as ground truth.
3. Dispatch N agents with `isolation: "worktree"`. Each gets a fresh worktree off `phase1-A`, ~8 calendars assigned, writes Python + tests + verifies pytest/pyright/ruff + commits + pushes its own branch.
4. Main session merges each branch with `--no-ff` after all return. Zero conflicts because file sets are disjoint per batch.

**Caveats discovered:**
- 2 of 5 subagent worktrees provisioned at `main` (`ec4fed0`) instead of `phase1-A`. Both agents detected the missing-commits state and self-corrected via `git reset --hard phase1-A` + `git submodule update`. **Next time:** explicitly pass the desired base in the dispatch prompt.
- Subagents independently caught **two prompt errors**: Ukraine and Romania inherit from `Calendar::OrthodoxImpl` in v1.42.1 (not Western). My prompts said "all Western" for those batches; the agents verified against the reference JSON Easter dates and corrected the inheritance. **Lesson:** subagents add useful adversarial review even on rote tasks.

### 2. Two-stage review subagent pass surfaces real bugs

After all 6 stages landed (411/0/0), two review subagents ran in parallel:
- **Spec-compliance** reviewer: PASS WITH FIXUPS (0 BLOCKER, 4 NIT). Catalogued 36 documented divergences from C++ for future reference.
- **Code-quality** reviewer: 0 BLOCKER, 3 MAJOR, 6 MINOR, 4 NIT.

The 3 MAJOR findings were all real correctness bugs that would have shipped:
- `time_grid.py` close_enough tolerance was 4500× looser than C++ (`42 * 1e-12` vs intended `42 * QL_EPSILON`). Caused silent point merging in `_dedupe_close` / `TimeGrid.index`.
- `Factorial.get(-1)` / `PascalTriangle.get(-1)` / `BernsteinPolynomial.get(i > n, ...)` silently returned wrong values via Python negative-array indexing (C++ uses `Natural` (unsigned)).
- `_coupons_per_year` would divide by zero if called outside the 15-day-floor guard.

All 3 MAJOR + 3 selected MINOR findings fixed in `03d0ce8 fix(l1-A): apply L1-A code-review fixes`. **Lesson:** always run both review passes before the cluster tag, even when the implementation pytest suite is green.

### 3. C++ pImpl + N-Impl-per-Convention → single Python class dispatching on IntEnum

The pattern emerged in Stage 3 (`Actual365Fixed`, `Thirty360`, `ActualActual`) and got reused everywhere in Stage 4 (every multi-market calendar). Shared `name()` output and shared dispatch preserve C++ `operator==` (name-based) semantics across convention aliases.

This is now a project precedent — captured in `/Users/josemoya/.claude/projects/.../memory/project_python_translation_choices.md`.

### 4. Documented divergences (full catalogue in spec review doc)

The non-bug divergences from C++ that we deliberately accept:

- **`math.lgamma` instead of `GammaFunction.logValue`** (Factorial + Beta + IncompleteBeta) — LOOSE tier divergence ~1e-9 to ~1e-12 relative.
- **`math.erf` instead of Sun-Microsystems polynomial fit** (ErrorFunction) — LOOSE tier.
- **`datetime.strptime` instead of boost::date_time facets** (`DateParser.parse_formatted`).
- **No `Settings.evaluation_date` fallback in `Schedule.from_rule`** — explicit `effective_date` required.
- **`Business252()` default `Brazil()` dropped** — explicit calendar required as an API-clarity choice.
- **`Business252` in-memory month/year caches dropped** — algorithmically equivalent to direct `Calendar.business_days_between` delegation, just slower for very long ranges.
- **`Date.month()` `Month(13)` index sentinel** — C++ unchecked enum cast; Python widened `_month_offset` signature to accept `Month | int`.
- **`Visitor` non-generic Protocol** — C++ `template<class T> class Visitor` collapses (structural matching at `.visit` call site does the type narrowing).
- **`Series` class deliberately not created** — does not exist in v1.42.1, was a jquantlib artifact.
- **ECB known-dates as Python tuple** — 200 entries embedded verbatim; module-level `set[Date]` for mutable add_date / remove_date.

### 5. Stage 4 was bigger than the design estimated

The Phase 1 design estimated ~45 calendars; v1.42.1 has 41 sovereign/exchange + 4 we already had (Null/Weekends/Joint/Bespoke) = 45 total. The Phase 1 L1-A design estimated ~99 modules; we ended at ~115 modules + 41 calendar files. Scope held — no A1 (>500-class) trigger fired.

### 6. Per-class commits with verification fence are worth the cost

Every per-class commit includes:
```
Verified:
- uv run pytest: <count>/0/0
- uv run pyright: 0 errors
- uv run ruff check: clean
- uv run ruff format --check: <N> files already formatted
```

This let the spec-compliance reviewer mechanically check the discipline rather than re-running tests. **NIT-2 from the spec review** noted that batch-E's 13 per-calendar feat commits skipped the fence (verified by the subagent in its isolated worktree). Future fan-out dispatches should require the agent to transcribe the fence.

## Carry-overs into L1-B / C / D / E

These are deferred-by-design, not gaps:

- **GammaFunction port** (`ql/math/distributions/gammadistribution.hpp`) — needed by Factorial + Beta exact-precision paths. Currently `math.lgamma`. Lands in L1-D or L1-E.
- **Currency / Money classes** (`ql/currencies/`) — Phase 1 design placed these in L1-B alongside copulas. ~9 classes.
- **Random number generators** (`ql/math/randomnumbers/`) — L1-D, ~30 classes.
- **Distributions** (`ql/math/distributions/`) — L1-B, ~30 classes.
- **Optimization** (`ql/math/optimization/`) — L1-D, ~30 classes.
- **Integrals** (`ql/math/integrals/`) — L1-C, ~45 classes.
- **Solvers1D** (`ql/math/solvers1D/`) — L1-C, 9 classes.
- **Interpolations** (`ql/math/interpolations/`) — L1-E, ~50 classes.
- **Matrix utilities** (`ql/math/matrixutilities/`) — L1-E, ~25 classes.
- **Multi-market support for already-ported calendars** — only default market currently. ~30 calendars have secondary markets (UnitedStates has 7, China has 6, etc.).

The L1-{B,C,D,E} design+plan docs will be drafted next session.

## What's pending

1. ~~FF-merge `phase1-A` → `main`~~ ✅ (commit `03d0ce8`)
2. ~~Tag `pquantlib-phase1-l1-A-complete`~~ ✅ (annotated tag pushed)
3. ~~Spec-compliance + code-quality review subagent passes + fixup commit~~ ✅
4. ~~Write `docs/migration/phase1-l1-A-completion.md`~~ ✅ (this document)
5. **Draft `docs/migration/phase1-l1-B-design.md` + `…-plan.md`** — math.copulas + distributions + statistics + currencies cluster.
6. **Draft `docs/migration/phase1-l1-C-design.md` + `…-plan.md`** — math.integrals + solvers1D + ode.
7. **Draft `docs/migration/phase1-l1-D-design.md` + `…-plan.md`** — math.randomnumbers + optimization.
8. **Draft `docs/migration/phase1-l1-E-design.md` + `…-plan.md`** — math.matrixutilities + interpolations + remaining math root.
9. Dispatch L1-B/C/D/E clusters in parallel (4 worktrees, mirroring the Stage 4 fan-out pattern).
10. After all four merge: tag `pquantlib-phase1-complete`.

## Links

- [Phase 1 design](phase1-design.md) — binding spec for the whole L1 layer
- [L1-A design](phase1-l1-A-design.md) — binding spec for this cluster, with 20-row decision log + pause triggers
- [L1-A plan](phase1-l1-A-plan.md) — executable task list (now historical)
- [L1-A progress](phase1-l1-A-progress.md) — running stage-by-stage log with per-stage surprises
- [L1-A spec review](phase1-l1-A-spec-review.md) — spec-compliance reviewer findings + 36-entry divergence catalogue
- [L1-A code review](phase1-l1-A-code-review.md) — code-quality reviewer findings (3 MAJOR + 6 MINOR + 4 NIT)
- [Phase 0 design](phase0-design.md), [Phase 0 plan](phase0-plan.md), [Phase 0 completion](phase0-completion.md) — predecessor cluster
