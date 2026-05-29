# PQuantLib

> A 100%-Python port of [QuantLib](https://www.quantlib.org/) — the de-facto open-source library for quantitative finance — being systematically rebuilt from C++ v1.42.1 with bit-exact precision guarantees.

[![Tag](https://img.shields.io/badge/tag-pquantlib--phase9--complete-green)](#migration-status)
[![Tests](https://img.shields.io/badge/tests-2464%2F0%2F0-brightgreen)](#migration-status)
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
| 3 L3-A (pilot) | `pquantlib-phase3-l3-A-complete` | Foundations: Settings.evaluation_date observable + 4 retroactive L1/L2 cleanups; Payoff + Exercise + Instrument + PricingEngine + GenericEngine + BlackFormula + Option + 3 cross-cluster Protocols (Instrument / PricingEngine / StochasticProcess) | 1037/0/0 | 2026-05-27 |
| 3 L3-B/C/D/E (parallel) | _(merged into Phase 3 tag)_ | L3-B (bonds: Bond + 4 concretes + Callability + DiscountingBondEngine + BondForward), L3-C (swaps: Swap + VanillaSwap + OIS + ZeroCoupon + Make-factories + DiscountingSwapEngine + 3 L2-C carry-over closures), L3-D (equity options + processes: StochasticProcess hierarchy + GBSM family + VanillaOption + EuropeanOption + AnalyticEuropeanEngine + BinomialVanillaEngine), L3-E (forwards: Forward + Position + FxForward + FRA + DiscountingFwdEngine + L2-C FraRateHelper carry-over). 4 isolated-worktree subagents in parallel, ~50 min wall-clock. | +247 → 1284/0/0 | 2026-05-27 |
| **3 complete** | **`pquantlib-phase3-complete`** | **Full L3 layer** — instruments + pricingengines. 1284 tests across ~50 classes; vanilla pricing path (bonds + swaps + European options + FX forwards) end-to-end. | **1284/0/0** | **2026-05-27** |
| 4 L4-A (pilot) | `pquantlib-phase4-l4-A-complete` | Foundations: Phase 1 LM + Simplex carry-overs closed (scipy-backed); Parameter hierarchy + Model + CalibratedModel + CalibrationHelper bases + 3 cross-cluster Protocols (Model / CalibrationHelper / ShortRateModel) | 1351/0/0 | 2026-05-27 |
| 4 L4-B/C/D/E (parallel) | _(merged into Phase 4 tag)_ | L4-B (short-rate: Vasicek + HW + CIR + ExtendedCIR + OU/CIR processes), L4-C (Heston + Bates + AnalyticHestonEngine), L4-D (G2++ + TwoFactorModel + multi-process suite), L4-E (Swaption + CapFloor instruments closing Phase 3 carve-out + Black/Bachelier/Jamshidian/G2 swaption engines + Black/Bachelier/AnalyticCapFloor engines + SwaptionHelper + CapHelper). 4 parallel cluster subagents, ~50 min wall-clock. | +193 → 1544/0/0 | 2026-05-27 |
| **4 complete** | **`pquantlib-phase4-complete`** | **Full L4 layer** — models (short-rate + equity stochastic-vol + calibration helpers + analytic swaption/capfloor engines). 1544 tests across ~40 classes. Phase 1 optimizer carry-overs + Phase 3 Swaption/CapFloor carve-outs closed. | **1544/0/0** | **2026-05-27** |
| 5 L5-A (pilot) | `pquantlib-phase5-l5-A-complete` | Foundations: Phase 1 Sobol + Burley2020 + GammaFunction Lanczos + AkimaCubic carry-overs closed; Tree[T] + Lattice + DiscretizedAsset hierarchy + 3 cross-cluster Protocols | 1614/0/0 | 2026-05-28 |
| 5 L5-B/C/D/E (parallel) | _(merged into Phase 5 tag)_ | L5-B (TrinomialTree + BlackScholesLattice + TreeSwaption/CapFloorEngine + **BlackKarasinski** — closes Phase 4 carve-outs), L5-C (MC framework + MCEuropeanEngine + MCAsianEngine), L5-D (FD framework + **FdBlackScholesVanillaEngine + VanillaOption.implied_volatility** — closes Phase 3 carve-out), L5-E (6 exotic instrument families + 6 analytic engines + **BivariateCumulativeNormalDistribution** — closes Phase 1 carve-out). 4 parallel cluster subagents, ~50 min wall-clock. | +269 → 1883/0/0 | 2026-05-28 |
| **5 complete** | **`pquantlib-phase5-complete`** | **Full L5 layer** — tree/lattice + MC + FD + exotic instruments. 1883 tests across ~50 classes. Closed 5 distinct phases of carry-overs (Phase 1 Sobol/GammaFunction/Akima/Bivariate; Phase 3 VanillaOption.implied_volatility; Phase 4 BlackKarasinski/TreeSwaption/TreeCapFloor). | **1883/0/0** | **2026-05-28** |
| 6 L6-A/B/C (parallel) | _(merged into Phase 6 + final tags)_ | L6-A (LongstaffSchwartz American MC — closes Phase 5 carve-out), L6-B (BatesEngine — closes Phase 4 carve-out via add_on_term hook), L6-C (DoubleBarrierOption + AnalyticDoubleBarrierEngine Ikeda-Kunitomo 1992 — closes Phase 5 carve-out). Modernization sweep deleted after audit (codebase already modern from day 1). 3 parallel cluster subagents, ~30 min wall-clock. | +75 → 1958/0/0 | 2026-05-28 |
| **6 complete** | **`pquantlib-phase6-complete`** | **High-impact Phase 4+5 carve-outs closed** + final closure tooling (`docs/carve-outs.md` + 4 sample programs). 1958 tests; closes Phase 4 BatesEngine + Phase 5 American MC + DoubleBarrier. | **1958/0/0** | **2026-05-28** |
| **Project complete** | **`pquantlib-final`** | **End of planned migration.** ~340 classes ported across ~1958 tests (54.2% of jquantlib-final). Vanilla pricing + calibration end-to-end + American MC + analytic exotics covered. See `docs/carve-outs.md` for comprehensive carve-out documentation. | **1958/0/0** | **2026-05-28** |
| 7 inflation (opt-in extension) | `pquantlib-phase7-complete` | Inflation Tier-1 carve-out closed: 5 region indexes + termstructure abstracts + Seasonality + Interpolated curves + inflation cashflows + pricers + ZeroCouponInflationSwap + YearOnYearInflationSwap + CPISwap + YoYInflationCapFloor + vol surfaces + 3 YoY analytic engines (Bachelier/Black/UnitDisplaced). L7-B Piecewise + helpers deferred to L7-Bb follow-up. 4 clusters; ~32 classes; +151 tests. | **2109/0/0** | **2026-05-28** |
| 8 piecewise inflation + credit + capfloor-vol (opt-in extension) | `pquantlib-phase8-complete` | **L8-A** Piecewise{Zero,YoY}InflationCurve + IterativeBootstrap (closes L7-Bb + L2-B); **L8-B** Tier-1 credit (DefaultProbabilityTermStructure family + FlatHazardRate + 3 interpolated curves + probability traits + PiecewiseDefaultCurve scaffold + Spread/UpfrontCdsHelper + CreditDefaultSwap + Claim + MidPoint/Integral CDS engines); **L8-C** capfloor/optionlet/swaption vol surfaces (CapFloorTermVolatilityStructure family + OptionletVolatilityStructure family + OptionletStripper1 + SwaptionVolatilityStructure family + SwaptionVolatilityMatrix; closes Phase 2 capfloor-vol). 3 parallel-no-pilot clusters, ~60 min wall-clock. ~40 classes; +194 tests. | **2303/0/0** | **2026-05-28** |
| 9 cubic/bicubic + post-L8 ergonomics + SABR cube (opt-in extension) | `pquantlib-phase9-complete` | **L9-A pilot** Cubic + Bicubic spline interpolators (CubicNaturalSpline + MonotonicCubicNaturalSpline + BicubicSpline via scipy delegation; opt-in `interpolator=` kwarg on L8-C surfaces; closes L1-E cubic-family); **L9-B** post-L8 ergonomics (PiecewiseYieldCurve + Discount/ZeroYield/ForwardRate traits + PiecewiseDefaultCurve bootstrap wiring + IsdaCdsEngine + implied_hazard_rate + conventional_spread + MakeCDS); **L9-C** SABR swaption smile cube (sabr_volatility + sabr_normal_volatility (Hagan 2002) + SabrInterpolation + SmileSection abstract + Flat/Interpolated/Sabr/Spreaded SmileSection + SwaptionVolatilityCube + SabrSwaptionVolatilityCube + InterpolatedSwaptionVolatilityCube; closes Phase-8 SABR). Pilot + 2-parallel, ~90 min wall-clock. ~22 classes; +161 tests. | **2464/0/0** | **2026-05-28** |

Per-phase scoping mirrors JQuantLib's layer sequencing:
- **Phase 1:** L1 — math primitives (`Array` via numpy, `Date`, `Calendar`, `DayCounter`, distributions, integrals, interpolations, RNGs)
- **Phase 2:** L2 — termstructures + indexes
- **Phase 3:** L3 — instruments + pricingengines
- **Phase 4:** L4 — models
- **Phase 5:** L5 — experimental + L6 test-suite parity
- **Phase 6:** Python 3.14 modernization sweep
- **Phase 7:** Final closure + carve-out documentation + tag `pquantlib-final`

**Current tip on `main`:** Phase 9 closed via `pquantlib-phase9-complete` (opt-in extension beyond `pquantlib-final` — cubic/bicubic interpolators + post-L8 ergonomics + SABR swaption smile cube). See [`docs/migration/phase9-completion.md`](docs/migration/phase9-completion.md) for the closure summary + [`docs/carve-outs.md`](docs/carve-outs.md) for the comprehensive carve-out catalog.

**Sample programs**: Run `uv run python -m pquantlib_samples.{vanilla_swap_pricing,heston_calibration,american_option_mc,double_barrier_analytic}` for end-to-end demos.

## What's available today (Phase 1 L1 + Phase 2 L2 + Phase 3 L3 + Phase 4 L4 + Phase 5 L5)

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

All inflation termstructures/indexes/cashflows; all credit termstructures; ZABR/SABR/XABR volatility models; capfloor/optionlet/swaption volatility; 35 specialty ibors beyond the 8 must-port; specialized cashflows (Digital / Cms / CapFloored / AverageBmaCoupon); advanced curve construction (`FittedBondDiscountCurve` / `MultiCurve` / `GlobalBootstrap` / spline-fitting variants); `PiecewiseYieldCurve` full bootstrap. See [`docs/migration/phase2-completion.md`](docs/migration/phase2-completion.md).

(Phase 3 closed the previously-deferred `Settings.evaluation_date` wiring, `SwapRateHelper.implied_quote`, `OISRateHelper.implied_quote`, `SwapIndex.forecast_fixing`, and `FraRateHelper(useIndexedCoupon=True)` carry-overs.)

### Phase 3 L3 modules

#### Foundations (`pquantlib.*`)

- **`pquantlib.payoffs`** — `Payoff` abstract + `PlainVanillaPayoff(option_type, strike)` + `CashOrNothingPayoff` + `AssetOrNothingPayoff` + `GapPayoff` + `SuperFundPayoff` + `SuperSharePayoff`. `OptionType` enum (Call=1, Put=−1).
- **`pquantlib.exercise`** — `Exercise` abstract + `EuropeanExercise(date)` + `AmericanExercise(earliest, latest=None, payoff_at_expiry=False)` + `BermudanExercise(dates, ...)`.
- **`pquantlib.option.Option`** — abstract base for option instruments.
- **`pquantlib.patterns.observable_settings.ObservableSettings`** — now exposes `evaluation_date` field with observer notification.

#### Instruments (`pquantlib.instruments.*`)

- **`Instrument`** abstract + **`OneAssetOption`** abstract.
- **`pquantlib.instruments.protocols`** — `InstrumentProtocol`, `PricingEngineProtocol`, `StochasticProcessProtocol`.
- **Bonds** (`pquantlib.instruments.bonds.*`): `Bond` + `FixedRateBond` + `ZeroCouponBond` + `FloatingRateBond` + `AmortizingFixedRateBond` + `Callability` + `CallabilitySchedule` + `BondForward`.
- **Swaps**: `Swap` + `FixedVsFloatingSwap` + `VanillaSwap` + `OvernightIndexedSwap` + `ZeroCouponSwap` + `make_vanilla_swap(...)` + `make_ois(...)` free-function factories.
- **Options**: `VanillaOption` + `EuropeanOption`.
- **Forwards**: `Forward` abstract + `ForwardTypePayoff` + `Position` enum + `FxForward` + `ForwardRateAgreement`.

#### Pricing engines (`pquantlib.pricingengines.*`)

- **`PricingEngine`** abstract + **`GenericEngine[ArgsT, ResultsT]`** PEP 695 generic.
- **`pquantlib.pricingengines.black_formula`** — `black_formula(...)` (lognormal) / `bachelier_black_formula(...)` / `black_formula_implied_std_dev(...)` / `bachelier_black_formula_implied_vol(...)` + Black-vega derivatives.
- **`pquantlib.pricingengines.bond.discounting_bond_engine.DiscountingBondEngine`**.
- **`pquantlib.pricingengines.swap.discounting_swap_engine.DiscountingSwapEngine`**.
- **`pquantlib.pricingengines.forward.discounting_fwd_engine.DiscountingFwdEngine`**.
- **`pquantlib.pricingengines.vanilla.analytic_european_engine.AnalyticEuropeanEngine`** — closed-form BSM + Greeks (delta/gamma/vega/theta/rho).
- **`pquantlib.pricingengines.vanilla.binomial_engine.BinomialVanillaEngine`** — parameterized by `TreeBuilder` enum (CRR / JarrowRudd / Tian / LeisenReimer); supports American/Bermudan via numpy backward induction.
- **`pquantlib.pricingengines.vanilla.black_calculator.BlackCalculator`** — Visitor-replacement helper.

#### Stochastic processes (`pquantlib.processes.*`)

- **`StochasticProcess`** + **`StochasticProcess1D`** abstracts.
- **`EulerDiscretization`** — drift/diffusion stepper.
- **`GeneralizedBlackScholesProcess`** (risk-free + dividend + Black-vol curves), **`BlackScholesProcess`** (no dividends), **`BlackProcess`** (no rates — Black 76), **`BlackScholesMertonProcess`** (full BSM).

### Carve-outs (deferred from Phase 3)

All exotic instruments (Asian/Barrier/Basket/Cliquet/Lookback/Quanto/etc.); specialty bonds; specialty swaps; all MC + FD engines; lattice/tree hierarchy; `VanillaOption.implied_volatility` (needs FD engine). Full list in [`docs/migration/phase3-completion.md`](docs/migration/phase3-completion.md).

(Phase 4 closed the previously-deferred Swaption + CapFloor instruments, plus the Heston / Hull-White / G2 / Vasicek / CIR analytic engines.)

### Phase 4 L4 modules

#### Optimizers (Phase 1 carry-overs closed)

- **`pquantlib.math.optimization.levenberg_marquardt.LevenbergMarquardt`** — `scipy.optimize.least_squares(method='lm')` wrapper.
- **`pquantlib.math.optimization.simplex.Simplex`** — `scipy.optimize.minimize(method='Nelder-Mead')` wrapper.

#### Model foundations (`pquantlib.models.*`)

- **`Model`** abstract + **`TermStructureConsistentModel`** + **`CalibratedModel`** (with `calibrate(instruments, method, end_criteria, ...)` orchestration).
- **`Parameter`** hierarchy: `NullParameter` / `ConstantParameter` / `PiecewiseConstantParameter` / `TermStructureFittedParameter`.
- **`CalibrationHelper`** + **`BlackCalibrationHelper`** abstract bases.
- **`pquantlib.models.protocols`** — `ModelProtocol`, `CalibrationHelperProtocol`, `ShortRateModelProtocol`.

#### Short-rate models (`pquantlib.models.shortrate.*`)

- **`ShortRateModel`** + **`OneFactorModel`** + **`OneFactorAffineModel`** abstracts (closed-form `A(t,T)`, `B(t,T)`, `discount_bond_option` via Jamshidian).
- **`Vasicek(r0, a, b, sigma, lambda_=0.0)`**.
- **`HullWhite(termStructure, a=0.1, sigma=0.01)`** — extended-Vasicek with `phi(t)` closed-form + `convexity_bias`.
- **`CoxIngersollRoss(r0, theta, k, sigma)`**.
- **`ExtendedCoxIngersollRoss(termStructure, theta, k, sigma, r0)`**.
- **`TwoFactorModel`** abstract.
- **`G2(termStructure, a, sigma, b, eta, rho)`** — Brigo-Mercurio G2++ with closed-form `discount_bond` + `discount_bond_option` + `swaption`.

#### Equity stochastic-vol (`pquantlib.models.equity.*`)

- **`HestonModel(process)`** + **`HestonModelHelper`** (calibration helper).
- **`BatesModel(process)`** — HestonModel + Merton jumps.

#### Stochastic processes (added in Phase 4)

`pquantlib.processes.*`:
- **`OrnsteinUhlenbeckProcess`**, **`CoxIngersollRossProcess`**, **`HestonProcess`** (2-D: S, V), **`BatesProcess`**, **`G2Process`**, **`G2ForwardProcess`**, **`HullWhiteForwardProcess`**, **`ForwardMeasureProcess`** (1-D + multi-D).

#### Instruments (Phase 3 carve-outs closed)

- **`pquantlib.instruments.swaption.Swaption`** + **`pquantlib.instruments.cap_floor.{CapFloor, Cap, Floor}`**.

#### Calibration helpers (`pquantlib.models.*`)

- **`SwaptionHelper`** + **`CapHelper`** (concrete `BlackCalibrationHelper`s).

#### Analytic engines

`pquantlib.pricingengines.*`:
- **`vanilla.analytic_heston_engine.AnalyticHestonEngine(model, integration_order=144)`** — Gatheral characteristic function + `scipy.integrate.quad`.
- **`swaption.black_swaption_engine.BlackSwaptionEngine`** (model-free Black).
- **`swaption.bachelier_swaption_engine.BachelierSwaptionEngine`** (model-free normal).
- **`swaption.jamshidian_swaption_engine.JamshidianSwaptionEngine(model)`** — analytic under any `OneFactorAffineModel` via Jamshidian decomposition.
- **`swaption.g2_swaption_engine.G2SwaptionEngine(model, range_, intervals)`** — analytic G2 via 1-D SegmentIntegral + nested Brent solve.
- **`capfloor.black_capfloor_engine.BlackCapFloorEngine`**, **`bachelier_capfloor_engine.BachelierCapFloorEngine`**.
- **`capfloor.analytic_capfloor_engine.AnalyticCapFloorEngine(model)`** — uses `model.discount_bond_option` for each caplet/floorlet (HW/Vasicek/CIR/ExtendedCIR).

### Carve-outs (deferred from Phase 4)

MarketModels (125 files of LMM machinery); specialty short-rate (Gaussian1d / GSR / MarkovFunctional / HestonSLV / GJR-GARCH / BatesDoubleExp); volatility models (GARCH / GarmanKlass / etc.). See [`docs/migration/phase4-completion.md`](docs/migration/phase4-completion.md).

(Phase 5 closed: BlackKarasinski + TreeSwaptionEngine + TreeCapFloorEngine + MC framework + FD framework + exotic instruments.)

### Phase 5 L5 modules

#### Phase 1 carry-overs closed
- **`pquantlib.math.randomnumbers.sobol_rsg.SobolRsg`** + **`Burley2020SobolRsg`** — low-discrepancy via scipy.stats.qmc.Sobol.
- **`pquantlib.math.distributions.gamma_function.GammaFunction`** — Lanczos approximation; replaces math.lgamma in Factorial.
- **`pquantlib.math.distributions.bivariate_cumulative_normal.BivariateCumulativeNormalDistribution`** + `Dr78` alias — Genz-Bretz via scipy.stats.multivariate_normal.
- **`pquantlib.math.interpolations.akima_cubic_interpolation.AkimaCubicInterpolation`** — scipy.interpolate.Akima1DInterpolator.

#### Tree / lattice (`pquantlib.methods.lattices.*`)
- **`Tree[T]`** + **`Lattice`** abstracts; **`BinomialTree`** concretes (CRR / JarrowRudd / Tian / LeisenReimer); **`TrinomialTree`**; **`TreeLattice1D`**; **`BlackScholesLattice`**.
- **`DiscretizedAsset`** + **`DiscretizedOption`** + **`DiscretizedDiscountBond`** + **`DiscretizedSwap`** + **`DiscretizedSwaption`** + **`DiscretizedCapFloor`**.

#### Tree-based engines + BlackKarasinski (Phase 4 carve-outs closed)
- **`pquantlib.pricingengines.swaption.tree_swaption_engine.TreeSwaptionEngine(model, time_steps)`**.
- **`pquantlib.pricingengines.capfloor.tree_capfloor_engine.TreeCapFloorEngine(model, time_steps)`**.
- **`pquantlib.models.shortrate.onefactor.black_karasinski.BlackKarasinski(termStructure, a=0.1, sigma=0.1)`**.
- `ShortRateModel.tree(grid)` across Vasicek / HW / CIR / ExtendedCIR / BlackKarasinski.

#### Monte Carlo framework (`pquantlib.methods.montecarlo.*`)
- **`Path`** + **`MultiPath`** + **`BrownianBridge`** + **`PathGenerator`** + **`MultiPathGenerator`** + **`PathPricer`** + **`McSimulation`** + **`MonteCarloModel`**.
- **`MCVanillaEngine`** abstract + **`MCEuropeanEngine`** + **`MCDiscreteArithmeticAveragePriceEngine`**.
- **`DiscreteAveragingAsianOption`** + **`AnalyticDiscreteGeometricAveragePriceAsianEngine`** (Levy 1997).

#### Finite-difference framework (`pquantlib.methods.finitedifferences.*`)
- 18 modules: layout + meshers (FdmBlackScholesMesher) + operators (TripleBandLinearOp + FirstDeriv/SecondDeriv + FdmBlackScholesOp) + schemes (ExplicitEuler/ImplicitEuler/CrankNicolson) + step conditions (FdmAmericanStepCondition + Composite) + solver (FdmBackwardSolver) + config DTOs.
- **`pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine.FdBlackScholesVanillaEngine(process, t_grid=100, x_grid=100, scheme=CrankNicolson)`** — supports European + American.
- **`pquantlib.instruments.vanilla_option.VanillaOption.implied_volatility(...)`** — Brent solver dispatching to AnalyticEuropean or FD as appropriate. **Closes Phase 3 carve-out.**

#### Exotic instruments + analytic engines (`pquantlib.instruments.*` + `pquantlib.pricingengines.*`)
- **`AsianOption`** family (`ContinuousAveragingAsianOption` + `DiscreteAveragingAsianOption`).
- **`BarrierOption`** + `BarrierType` IntEnum (DownIn / UpIn / DownOut / UpOut).
- **`BasketOption`** + `BasketPayoff` hierarchy (Min / Max / Average / Spread).
- **`ContinuousFloatingLookbackOption`** + **`ContinuousFixedLookbackOption`**.
- **`CliquetOption`**, **`DigitalOption`**.
- **Analytic engines**: `AnalyticContinuousGeometricAveragePriceAsianEngine` (Kemna-Vorst), `AnalyticBarrierEngine` (Reiner-Rubinstein), `AnalyticBinaryBarrierEngine`, `StulzEngine` (2-asset basket), `AnalyticContinuousFloatingLookbackEngine` (Conze-Viswanathan).

#### Additional payoffs (`pquantlib.payoffs`)
- **`FloatingTypePayoff`**, **`PercentageStrikePayoff`**.

### Carve-outs (deferred from Phase 5)

Tree/lattice: TreeLattice2D + G2.tree(), Joshi4/AdditiveEQP/Trigeorgis builders. MC: LongstaffSchwartz American MC, Heston/G2/Bates/HW MC engines, low-discrepancy MC, all exotic MC engines. FD: multi-asset FD (Heston/G2/Bates), time-dependent operators, operator-splitting (HV/MS/CS), BoundaryCondition framework, Concentrating1dMesher. Exotic: DoubleBarrier, PartialTimeBarrier, SoftBarrier, HolderExtensible, ComplexChooser, Compound, 3+ asset baskets. See [`docs/migration/phase5-completion.md`](docs/migration/phase5-completion.md). Phase 6 addresses high-impact remainder + modernization.

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
