# PQuantLib

> A 100%-Python port of [QuantLib](https://www.quantlib.org/) — the de-facto open-source library for quantitative finance — being systematically rebuilt from C++ v1.42.1 with bit-exact precision guarantees.

[![Tag](https://img.shields.io/badge/tag-pquantlib--phase2--complete-green)](#migration-status)
[![Tests](https://img.shields.io/badge/tests-922%2F0%2F0-brightgreen)](#migration-status)
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
| 1 L1-A (pilot) | `pquantlib-phase1-l1-A-complete` | Foundations + time core (41 sovereign/exchange calendars via 5-agent fan-out) + 11 day counters + 8 first-math modules | 415/0/0 | 2026-05-23..2026-05-24 |
| 1 L1-B/C/D/E (parallel) | _(merged into Phase 1 tag)_ | L1-B (12 copulas + 3 normal distributions + 2 statistics + 5 currencies), L1-C (9 Solver1D + 5 simple integrals), L1-D (5 RNGs all EXACT-tier + BoxMuller + 7 optimization scaffolding), L1-E (4 interpolations + bilinear + scipy-backed Cholesky). 4 isolated-worktree subagents in parallel, ~25 min wall-clock. | +166 → 581/0/0 | 2026-05-24 |
| **1 complete** | **`pquantlib-phase1-complete`** | **Full L1 layer** — math primitives, time, foundations. 581 tests across 16 C++ probes / ~10,000 reference values. | **581/0/0** | **2026-05-24** |
| 2 L2-A (pilot) | `pquantlib-phase2-l2-A-complete` | Foundations: quotes + TermStructure + Extrapolator + BootstrapHelper + Index/IndexManager + 4 cross-cluster Protocols (Yield/Ibor/Overnight/Swap) | 649/0/0 | 2026-05-26 |
| 2 L2-B/C/D/E (parallel) | _(merged into Phase 2 tag)_ | L2-B (yield curves + Compounding + InterestRate), L2-C (8 ibors + 2 swap + 7 rate helpers), L2-D (cashflows + Coupon hierarchy + leg generators + CashFlows aggregator + Duration), L2-E (vol termstructures: SmileSection + BlackVol/LocalVol family). 4 isolated-worktree subagents in parallel, ~35 min wall-clock. | +273 → 922/0/0 | 2026-05-26 |
| **2 complete** | **`pquantlib-phase2-complete`** | **Full L2 layer** — termstructures + indexes + cashflows + quotes. 922 tests; cross-cluster Protocols proven as fan-out glue. | **922/0/0** | **2026-05-26** |

Per-phase scoping mirrors JQuantLib's layer sequencing:
- **Phase 1:** L1 — math primitives (`Array` via numpy, `Date`, `Calendar`, `DayCounter`, distributions, integrals, interpolations, RNGs)
- **Phase 2:** L2 — termstructures + indexes
- **Phase 3:** L3 — instruments + pricingengines
- **Phase 4:** L4 — models
- **Phase 5:** L5 — experimental + L6 test-suite parity
- **Phase 6:** Python 3.14 modernization sweep
- **Phase 7:** Final closure + carve-out documentation + tag `pquantlib-final`

**Current tip on `main`:** `b5d2519 merge: L2-E` (Phase 2 closed via `pquantlib-phase2-complete` tag). See [`docs/migration/phase2-completion.md`](docs/migration/phase2-completion.md) for the closure summary.

## What's available today (Phase 1 L1 + Phase 2 L2)

Phase 1 ships the foundation: math primitives, time machinery, day counters, currencies, distributions, random number generators, simple optimization scaffolding, and a starter set of 1-D/2-D interpolations. Importable as `pquantlib.<module>`.

### Foundations

- **`pquantlib.exceptions`** — `LibraryException` (C++ `QL::Error` analogue), `RootNotBracketed`, etc.
- **`pquantlib.qassert`** — `require(cond, msg)` / `ensure(...)` (C++ `QL_REQUIRE` / `QL_ENSURE`).
- **`pquantlib.testing.tolerance`** — `exact(a, e)` / `tight(a, e)` / `loose(a, e)` / `custom(a, e, abs_tol, rel_tol)`.
- **`pquantlib.testing.reference_reader`** — `load("<topic>/<class>")` returns probe-emitted JSON.
- **`pquantlib.patterns`** — `Observer` / `Observable` (weakref-backed), `Singleton` (metaclass), `Visitor` / `Visitable` (Protocols), `LazyObject`, `CuriouslyRecurringTemplate`.

### Time core

- **`pquantlib.time.Date`** — frozen `@dataclass`, Excel-1900 epoch, `+`/`-`/`<`/etc. with `@overload`-narrowed return types.
- **`pquantlib.time.Period`** — `frozen+slots`, normalized arithmetic.
- **`pquantlib.time.DateParser` / `PeriodParser`** — string ↔ value-type.
- **`pquantlib.time.Calendar`** — abstract base + 4 trivial concretes (`NullCalendar`, `WeekendsOnly`, `JointCalendar`, `BespokeCalendar`) + **41 sovereign/exchange calendars** (US/UK/Germany/France/Italy/Japan/Switzerland/Sweden/Brazil/China/India/Canada/Australia/Russia/Israel/etc., each with default-market `Convention` enum).
- **`pquantlib.time.Schedule`** + **`MakeSchedule`** — builder pattern for date schedules.
- **`pquantlib.time.IMM` / `ASX` / `ECB`** — module-of-free-functions exporting `next_date`, `is_imm_date`, `known_date`, etc.
- **`pquantlib.time.TimeGrid`** — mandatory + close-enough times bundle.
- **`pquantlib.time.TimeSeries[T]`** — PEP 695 generic; sorted-by-date container.

### Day counters

`pquantlib.daycounters` — `DayCounter` abstract + 11 concretes (`Actual360`, `Actual365Fixed` w/ Standard/NoLeap/Actual365 convention dispatch, `ActualActual` w/ ISMA/Bond/ISDA/Historical/Actual365/AFB/Euro variants, `Thirty360` w/ USA/BondBasis/European/Italian/German/ISMA/ISDA/NASD/EurobondBasis variants, `Business252`, `One`, `Simple`, `Thirty365`).

### Math primitives

- **`pquantlib.math.constants`** — `M_PI`, `QL_EPSILON`, `QL_MIN_REAL`, `QL_MAX_REAL` (from `math` + `sys.float_info`).
- **`pquantlib.math.closeness.close_enough`** — relative-tolerance comparison.
- **`pquantlib.math.rounding`** — 6 `Type` enums (Up/Down/Closest/Floor/Ceiling/None).
- **`pquantlib.math.factorial`** — table to n=27 + `math.lgamma` fallback (LOOSE-tier).
- **`pquantlib.math.error_function`** — `math.erf` delegate (LOOSE-tier).
- **`pquantlib.math.beta`** + `incomplete_beta` + `beta_continued_fraction` — closed-form via `math.lgamma`.
- **`pquantlib.math.bernstein_polynomial`** + **`pascal_triangle`** — combinatorial helpers with `qassert.require` guards.

### Distributions, copulas, statistics

- **`pquantlib.math.distributions`** — `NormalDistribution`, `CumulativeNormalDistribution` (via `math.erf`), `InverseCumulativeNormal` (Acklam algorithm), `MoroInverseCumulativeNormal`.
- **`pquantlib.math.copulas`** — 12 closed-form 2-D copulas: AliMikhailHaq, Clayton, FarlieGumbelMorgenstern, Frank, Galambos, Gaussian, Gumbel, HuslerReiss, Independent, MaxCopula, MinCopula, Plackett. (MarshallOlkin deferred.) All `@dataclass(frozen=True, slots=True)` with `__call__`.
- **`pquantlib.math.statistics`** — `GeneralStatistics` (running mean/variance/kurtosis/skew), `IncrementalStatistics` (Welford-style aggregator).

### Random number generators (all EXACT-tier bit-exact vs C++)

`pquantlib.math.randomnumbers` — `MersenneTwisterUniformRng` (MT19937), `KnuthUniformRng`, `LecuyerUniformRng`, `Ranlux3UniformRng`, `Xoshiro256StarStarUniformRng`, `BoxMullerGaussianRng`. Every PRNG reproduces C++ outputs bit-for-bit via `struct.pack('!d', x)` comparison.

### Solvers and integrals

- **`pquantlib.math.solvers1d`** — `Solver1D` abstract + 9 concretes (`Bisection`, `Brent`, `FalsePosition`, `FiniteDifferenceNewtonSafe`, `Halley`, `Newton`, `NewtonSafe`, `Ridder`, `Secant`).
- **`pquantlib.math.integrals`** — `Integrator` abstract + 5 simple concretes (`SimpsonIntegral`, `TrapezoidIntegral`, `SegmentIntegral`, `GaussKronrodAdaptive`, `GaussLobattoIntegral`).

### Optimization scaffolding

`pquantlib.math.optimization` — `Constraint` family (`NoConstraint`, `PositiveConstraint`, `BoundaryConstraint`), `CostFunction`, `EndCriteria`, `OptimizationMethod` abstract, `Problem`. (Concrete LM/BFGS/Simplex/CG/SA optimizers deferred to a follow-up cluster.)

### Interpolations + matrix utilities

- **`pquantlib.math.interpolations`** — `Interpolation` abstract + 4 1-D concretes (`LinearInterpolation`, `LogLinearInterpolation`, `BackwardFlatInterpolation`, `ForwardFlatInterpolation`) + `BilinearInterpolation` 2-D.
- **`pquantlib.math.matrixutilities`** — `Array` / `Matrix` typing aliases over `npt.NDArray[np.float64]`, plus `CholeskyDecomposition` (scipy delegate).

### Currencies

`pquantlib.currencies` — 5 ISO descriptors (USD, EUR, GBP, JPY, CHF) as `@dataclass(frozen=True, slots=True)`. (More currencies follow in L2 termstructures work.)

### Carve-outs (deferred from Phase 1)

Full `GaussianOrthogonalPolynomial` hierarchy (12+ subclass tree), `SobolRsg` + `Burley2020SobolRsg` low-discrepancy, `LevenbergMarquardt`/`Bfgs`/`Simplex`/`ConjugateGradient`/`SimulatedAnnealing` optimizers, 8+ cubic-spline variants (`AkimaCubicInterpolation`, `KrugerCubic`, `FritschButland`, etc.), `QRDecomposition`/`EigenvalueDecomposition`/`SVD`/`SparseMatrix` utilities, full `GammaFunction` (currently delegated to `math.lgamma`). See [`docs/migration/phase1-completion.md`](docs/migration/phase1-completion.md).

### Phase 2 L2 modules

#### Quotes

- **`pquantlib.quotes`** — `Quote` abstract + `SimpleQuote` (mutable, observer-aware) + `DerivedQuote(quote, f)` + `CompositeQuote(quote1, quote2, f)` — market-observable handles used by every termstructure and rate-helper input.

#### Term structure scaffolding

- **`pquantlib.termstructures.term_structure.TermStructure`** — abstract base; reference_date / max_date / day_counter / calendar / extrapolation support.
- **`pquantlib.termstructures.extrapolator.Extrapolator`** — enable/disable extrapolation flag mixin.
- **`pquantlib.termstructures.bootstrap_helper.BootstrapHelper[TS]`** — abstract; PEP 695 generic; `PillarChoice` enum.
- **`pquantlib.termstructures.protocols`** — `YieldTermStructureProtocol`, `IborIndexProtocol`, `OvernightIndexProtocol`, `SwapIndexProtocol` — `@runtime_checkable` cross-cluster glue.

#### Yield termstructures

`pquantlib.termstructures.yield_term_structure.YieldTermStructure` (abstract); concretes under `pquantlib.termstructures.yield_.*`:

- **`FlatForward`** (+ `FlatForward.from_rate(...)` classmethod) — constant-rate curve.
- **`InterpolatedZeroCurve`** / **`InterpolatedForwardCurve`** / **`InterpolatedDiscountCurve`** — parameterized by an `InterpolationFactory` (PEP 695 generics; default Linear).
- **`ZeroCurve`** / **`ForwardCurve`** / **`DiscountCurve`** — PEP 695 `type` aliases over the linear-interp variants.
- **`ForwardSpreadedTermStructure`** / **`ZeroSpreadedTermStructure`** / **`DiscountSpreadedTermStructure`** — Quote-driven spread overlays.
- **`ImpliedTermStructure`** — forward-shifted view of an existing curve.

#### Rate helpers (under `pquantlib.termstructures.yield_`)

- **`DepositRateHelper`** / **`FraRateHelper`** / **`FuturesRateHelper`** / **`SwapRateHelper`** / **`OISRateHelper`** / **`BondHelper`** / **`FxSwapRateHelper`** — all subclass `BootstrapHelper`. SwapRateHelper / OISRateHelper / BondHelper `implied_quote()` deferred to L3 (need pricing engines).

#### Compounding + InterestRate

- **`pquantlib.time.compounding.Compounding`** — IntEnum (Simple / Compounded / Continuous / SimpleThenCompounded / CompoundedThenSimple).
- **`pquantlib.interest_rate.InterestRate`** — rate + day-counter + compounding + frequency; `compound_factor(t)` / `discount_factor(t)` / `equivalent_rate(...)` / `implied_rate(...)` factory; null sentinel via `InterestRate.null()` + `is_null()`.

#### Index hierarchy

`pquantlib.indexes.index.Index` (abstract base, observable) + `IndexManager` singleton (fixings repo, case-insensitive); under `pquantlib.indexes.*`:

- **`InterestRateIndex`** / **`IborIndex`** / **`OvernightIndex`** / **`SwapIndex`** — abstract bases.
- **8 ibor concretes**: `Euribor`, `USDLibor`, `GBPLibor`, `Eonia`, `Sofr`, `Sonia`, `FedFunds`, `Estr`. Multi-tenor via `Period` arg + classmethod shortcuts (e.g. `Euribor.three_months()`).
- **2 swap indexes**: `EuriborSwapIsdaFixA`, `UsdLiborSwapIsdaFixAm`.

#### Cashflows

`pquantlib.cashflows.*`:

- **`CashFlow`** (abstract) + **`SimpleCashFlow`** (+ `Redemption` + `AmortizingPayment`).
- **`Coupon`** (abstract) + **`FixedRateCoupon`** + **`FloatingRateCoupon`** + **`IborCoupon`** + **`OvernightIndexedCoupon`**.
- **`fixed_rate_leg(...)`** / **`ibor_leg(...)`** / **`overnight_leg(...)`** — free-function leg generators (Pythonic replacement for C++ Builder pattern).
- **`CouponPricer`** (abstract) + **`IborCouponPricer`** + **`BlackIborCouponPricer`** + **`CompoundingOvernightIndexedCouponPricer`**.
- **`CashFlows`** — static methods aggregator: `npv` / `bps` / `irr` / `simple_duration` / `macaulay_duration` / `modified_duration` / `convexity`.
- **`Duration`** — IntEnum (Simple / Macaulay / Modified).

#### Volatility termstructures (equity / FX minimum)

`pquantlib.termstructures.volatility.*`:

- **`VolatilityTermStructure`** abstract + **`SmileSection`** abstract + **`FlatSmileSection`** concrete.
- **`BlackVolTermStructure`** (+ `BlackVolatilityTermStructure` + `BlackVarianceTermStructure` adapters) abstracts + **`BlackConstantVol`** + **`BlackVarianceCurve`** + **`BlackVarianceSurface`** (bilinear) concretes.
- **`LocalVolTermStructure`** abstract + **`LocalConstantVol`** + **`LocalVolCurve`** + **`LocalVolSurface`** (Dupire, flat-curve simplification) concretes.

### Carve-outs (deferred from Phase 2)

All inflation termstructures/indexes/cashflows; all credit termstructures; ZABR/SABR/XABR volatility models; capfloor/optionlet/swaption volatility; 35 specialty ibors beyond the 8 must-port; specialized cashflows (Digital / Cms / CapFloored / AverageBmaCoupon); advanced curve construction (`FittedBondDiscountCurve` / `MultiCurve` / `GlobalBootstrap` / spline-fitting variants); `PiecewiseYieldCurve` full bootstrap; SwapIndex.forecast_fixing (needs L3 VanillaSwap); `Settings.evaluation_date` observable wiring (used in TermStructure moving mode + SmileSection floating mode + RelativeDateBootstrapHelper). Full list in [`docs/migration/phase2-completion.md`](docs/migration/phase2-completion.md).

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
