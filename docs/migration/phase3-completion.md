# Phase 3 (L3 instruments + pricingengines) — completion

**Date closed:** 2026-05-27
**Tag:** [`pquantlib-phase3-complete`](../../README.md#migration-status) @ `aacc2c2`
**Predecessor:** `pquantlib-phase2-complete` @ `b5d2519`
**Test count:** 922 → **1284/0/0** (+362). pyright + ruff clean.
**Design spec:** [`phase3-design.md`](phase3-design.md). **Plan:** [`phase3-plan.md`](phase3-plan.md).

## Cluster contribution table

| Cluster | Mode | Commits | Tests added | Coverage |
|---|---|---|---|---|
| **L3-A pilot** | sequential, 6 stages | 7 | +115 | Settings.evaluation_date observable wired; 4 retroactive L1/L2 cleanups (Schedule null-effective-date fallback, TermStructure moving-mode, RelativeDateBootstrapHelper, SmileSection floating-mode); Payoff hierarchy (PlainVanilla + Cash/AssetOrNothing + Gap + SuperFund + SuperShare); Exercise hierarchy (European/American/Bermudan); Instrument + PricingEngine + GenericEngine[ArgsT, ResultsT]; BlackFormula family (lognormal + bachelier + implied-vol + Black-vega); Option + OneAssetOption + cross-cluster Protocols (Instrument / PricingEngine / StochasticProcess) |
| **L3-B** bonds | parallel | 6 | +81 | Bond abstract + FixedRateBond + ZeroCouponBond + FloatingRateBond + AmortizingFixedRateBond + Callability + CallabilitySchedule + DiscountingBondEngine + BondForward; extended CashFlows with leg-walking helpers |
| **L3-C** swaps + carry-over closures | parallel | 4 | +41 | Swap abstract + FixedVsFloatingSwap + VanillaSwap + OvernightIndexedSwap + ZeroCouponSwap + make_vanilla_swap + make_ois + DiscountingSwapEngine. Closed L2-C carry-overs: SwapRateHelper.implied_quote, OISRateHelper.implied_quote, SwapIndex.forecast_fixing/underlying_swap. Added par-coupons forecast to IborCouponPricer |
| **L3-D** options + processes | parallel | 4 | +97 | StochasticProcess + StochasticProcess1D + EulerDiscretization + GeneralizedBlackScholesProcess + BlackScholesProcess + BlackProcess + BlackScholesMertonProcess; VanillaOption + EuropeanOption; BlackCalculator helper; AnalyticEuropeanEngine + BinomialVanillaEngine (CRR/JarrowRudd/Tian/LeisenReimer via direct numpy backward induction) |
| **L3-E** forwards + FRAs | parallel | 4 | +28 | Forward abstract + ForwardTypePayoff + Position enum + FxForward + DiscountingFwdEngine + ForwardRateAgreement. Closed L2-C FraRateHelper(useIndexedCoupon=True). Modified LazyObject.calculate for bootstrap recursion |
| **Post-merge alignment** | sequential | 1 | 0 | YieldTermStructureProtocol.discount param name unified to `t` across Protocol + 3 test-mock classes |
| **Total** | | **~26** | **+362** | **~50 classes** |

L3-A pilot was tagged separately as `pquantlib-phase3-l3-A-complete` @ `e72bcdf` mid-phase; that tag is preserved as the predecessor for parallel cluster dispatch.

All 4 parallel clusters significantly exceeded their test targets — L3-B +81 vs +30 target, L3-D +97 vs +43 target, L3-C +41 vs +35 target, L3-E +28 vs +20 target. Subagents added richer behavioral coverage than the floor estimates.

## Parallelization wins

Same fan-out pattern as Phases 1 + 2, now proven across three phases:

- **Wall-clock**: L3-A pilot ~50 min (largest pilot to date — Settings observable wiring + 14 classes across 6 stages). L3-B/C/D/E in parallel ~50 min wall-clock for ~36 classes across 4 disjoint topic areas (bonds / swaps / equity options + processes / forwards).
- **Three Protocols** (Instrument / PricingEngine / StochasticProcess) defined in L3-A glued the 4 parallel clusters at merge time. The pattern continues to scale.
- **No probe rebuild per worktree**: subagents continued using the main worktree's already-built libQuantLib via one-off `clang++` linking.
- **One real alignment surprise**: parameter-name mismatch between L3-C's concrete YieldTermStructure (`t`) and L3-E's slimmed Protocol comment (which still said `arg`). Caught by pyright in 8 test-mock locations; resolved in a single `align(...)` commit.

## Merge reconciliation

Three issues caught + resolved at merge:

1. **`YieldTermStructureProtocol.discount` parameter name** drifted between subagents (L3-C renamed concrete `arg→t`; L3-E slimmed the Protocol but its docstring + 3 test-mock classes still used `arg`). Resolved by unifying to `t` everywhere; structural typing requires positional-or-keyword arg names to match.
2. **L3-B's `BondForwardPosition` placeholder** vs **L3-E's `Position` enum** — no actual conflict at merge. L3-B inlined position values directly into `BondForward` rather than importing from a not-yet-existent module; L3-E ported the canonical `Position` enum. Future BondForward refactor can adopt L3-E's enum without breaking changes.
3. **`CMakeLists.txt`** — 4 parallel `add_executable` additions for cluster probes, stacked at merge.

## Cross-cluster Protocol design (validated again)

The Protocol-glue pattern continues to deliver:

- L3-B's `DiscountingBondEngine` takes a concrete `YieldTermStructure` (because it needs `zero_rate()` / `forward_rate()` which the slim Protocol can't offer — documented).
- L3-C's `DiscountingSwapEngine` takes `YieldTermStructure` directly (same reason).
- L3-D's `AnalyticEuropeanEngine` takes a `GeneralizedBlackScholesProcess` (concrete) rather than `StochasticProcessProtocol` (Protocol can't structurally narrow to multi-D `evolve` signature).
- L3-E's `DiscountingFwdEngine` takes `YieldTermStructureProtocol` (only needs `discount()`).

Lesson: Protocols work for **narrow** consumers; concrete bases work for **rich** consumers. Both patterns coexist cleanly.

## Cumulative documented divergences (L1+L2+L3)

In addition to Phase 1+2's:

### Foundations
- **`ObservableSettings`** multi-inherits `Singleton + Observable` with an `_observable_settings_initialized` guard to prevent `Observable.__init__` running twice (Singleton metaclass caches the instance).
- **`Settings.evaluationDate()` global** replaced by `ObservableSettings().evaluation_date_or_today()` — explicit None means "today".

### Type-level
- **`GenericEngine[ArgsT, ResultsT]`** PEP 695 generic with typed bounds (`PricingEngineArguments`, `PricingEngineResults`).
- **`BinomialVanillaEngine`** collapses C++ `template<class T>` tree hierarchy into a single class parameterized by `TreeBuilder` enum (CRR / JarrowRudd / Tian / LeisenReimer) + direct numpy backward induction. Bypasses the entire `Tree` / `Lattice` / `DiscretizedAsset` / `DiscretizedVanillaOption` class hierarchy.
- **`BlackCalculator`** replaces the C++ Visitor pattern (`BlackCalculator::Calculator`) with `isinstance` dispatch.
- **1-D scalar overloads** on StochasticProcess renamed `drift_1d` / `diffusion_1d` / `expectation_1d` / `variance_1d` / `std_deviation_1d` / `evolve_1d` to coexist with multi-D `drift(t, NDArray) -> NDArray` signatures.
- **EulerDiscretization** uses `typing.overload` to satisfy both multi-D and 1-D abstract discretization bases with one body (C++ achieves via multiple-inheritance + name-overloading).
- **`YieldTermStructureProtocol` is slim** — only `discount()`; concrete `YieldTermStructure` extends with `zero_rate()` / `forward_rate()` (richer signatures that can't structurally narrow).

### Idiomatic Python
- **`Forward` → `ForwardTypePayoff`** (replaces C++ `Position::Type` casting).
- **`Position` IntEnum** for Long/Short forward position.
- **`Bond.yield_rate`** (Python `yield` is a reserved keyword — same trailing-underscore pattern as `yield_` subpackage from Phase 2).
- **`make_vanilla_swap` / `make_ois`** free functions with kwargs (Pythonic replacement for C++ `MakeVanillaSwap` / `MakeOIS` Builders).
- **`Exercise` / `EarlyExercise` abstract-by-convention** (no `@abstractmethod`) — matches C++ public constructors with empty default `dates_`.

### Algorithmic
- **`IborCouponPricer`** defaults to par-coupons forecast (C++ `Settings::usingAtParCoupons=true` analog). Fixed a 1.8e-5 rel drift in 5y swap NPV caught by L3-C testing.
- **`LazyObject.calculate`** sets `_calculated=True` BEFORE invoking `_perform_calculations` (C++ parity) with rollback on exception — supports bootstrap recursion `Forward.forward_value → calculate → perform_calculations → forward_value`.
- **`GeneralizedBlackScholesProcess.evolve_1d`** bypasses the StochasticProcess1D.evolve_1d default in the Euler-fallback branch (which routes through `expectation_1d` → "not implemented"); calls discretization drift directly to match C++ semantics.
- **`Bond.yield_rate` Brent solver** at LOOSE tier (1e-8) rather than NewtonSafe with derivative — easier in Python without templated function-with-derivative.

### Deferred deprecated paths (continuation)
- C++ `shared_ptr<PlainVanillaPayoff>` overloads of `blackFormula*` not ported — Python callers unwrap themselves.
- C++ `bachelierBlackFormulaImpliedVolChoi` (approximation) carved out; only the exact Jaeckel 2017 variant ported.

## Carve-outs (deferred from Phase 3)

All intentionally not ported. Each lands either as Phase 4/5 work or a dedicated follow-up.

### Exotic instruments (Phase 5)
Asian / Barrier / Basket / Cliquet / Lookback / Quanto / DoubleBarrier / ComplexChooser / Compound / HolderExtensible / Variance options.

### Swaption + capfloor (Phase 4 — model-coupled)
Swaption, NonStandardSwaption, FloatFloatSwaption, MakeSwaption, CapFloor, MakeCapFloor.

### Credit (Phase 4 or 5)
CDS / CreditDefaultSwap / MakeCDS / ConvertibleBond.

### Specialty swaps
BMA swap, Float/Float swap, NonStandardSwap, MultipleResetsSwap, EquityTotalReturnSwap. RateAveraging.Simple for OIS.

### Model-coupled engines (Phase 4)
- All Heston / Bates / GJR-GARCH / Hull-White / G2 / CEV / SABR engines under `pricingengines/vanilla/`.
- All MC engines (Monte Carlo).
- All FD (finite-difference) engines.
- `VanillaOption.implied_volatility` (depends on FdBlackScholesVanillaEngine).

### Specialty processes (Phase 4)
HestonProcess, BatesProcess, GJRGARCHProcess, G2Process, Hull-White, CIR, Vasicek.

### Specialty bonds (on demand)
BTP, CmsRateBond, CpiBond, AmortizingCmsRateBond, AmortizingFloatingRateBond.

### Lattice / tree hierarchy (Phase 4)
Tree, BlackScholesLattice, DiscretizedAsset, DiscretizedVanillaOption, TimeGrid plumbing; Joshi4 / AdditiveEQP / Trigeorgis trees. (`BinomialVanillaEngine` ported via direct numpy backward induction, bypassing the hierarchy.)

### Phase-1+2 carry-overs still open
- Full `GaussianOrthogonalPolynomial` hierarchy.
- SobolRsg / Burley2020SobolRsg low-discrepancy.
- LM / BFGS / Simplex / ConjugateGradient / SimulatedAnnealing optimizers.
- 8+ cubic-spline variants.
- QR / Eigenvalue / SVD / SparseMatrix utilities.
- Full GammaFunction.
- All inflation termstructures + indexes + cashflows.
- All credit termstructures.
- ZABR / SABR / XABR vol; capfloor / optionlet / swaption vol.
- 35 specialty ibors beyond the 8 must-port.
- Advanced curve construction (FittedBondDiscountCurve / MultiCurve / GlobalBootstrap / spline-fitting variants); PiecewiseYieldCurve full bootstrap.

## Lessons learned

- **Settings.evaluation_date wiring in L3-A unblocked everything as planned.** 4 retroactive L1/L2 cleanups landed in the same commit. The "defer the observable, plan to wire in the next phase that needs it" strategy works well.
- **Test-floor estimates underestimate by 2× for instrument + engine ports.** Phase 3 targeted +204 tests; landed +362 (+77% over). Instruments + engines naturally compose into more behavioral assertions per class than termstructure/index ports.
- **Protocol parameter names matter for structural typing.** The post-merge `arg→t` alignment cost ~10 minutes and demonstrated that Protocols are positional-or-keyword-strict in pyright. For Phase 4: lock the Protocol parameter-name convention in the pilot and tell subagents not to rename.
- **Subagents make conservative scope choices that pay off.** L3-D's BinomialVanillaEngine bypassed the entire Tree/Lattice/DiscretizedAsset class hierarchy with direct numpy backward induction — cleaner than porting the C++ template machinery, still bit-exact convergence to the analytical solution.
- **L2 carry-overs all closed cleanly.** L3-C completed SwapRateHelper / OISRateHelper / SwapIndex; L3-E completed FraRateHelper(useIndexedCoupon=True). The deferred-with-clear-unblocker pattern works.
- **Cross-cluster type duplication still happens.** L3-B and L3-E independently invented `Position`-like enums (BondForwardPosition vs Position). L3-B's was inlined so no merge conflict, but the pre-pilot canonical-type discipline from Phase 2 should be extended to Phase 4 — port commonly-needed enums up front.

## Next: Phase 4 (L4 models)

Sister-project anchor: `jquantlib/docs/migration/phase2-L4-models-plan.md`. Expected scope: short-rate models + Heston + Hull-White + G2 + their calibration engines + the full optimizer suite (LM / BFGS / Simplex from Phase 1 carve-outs).

Phase 4 will unblock:
- Swaption pricing engines + CapFloor pricing engines (instruments already ported in carve-out; engines need short-rate models).
- BlackIborCouponPricer cap/floor branches (need OptionletVolatilityStructure).
- VanillaOption.implied_volatility (needs FdBlackScholesVanillaEngine).
- All Heston / Bates / GJR-GARCH equity option engines.

Phase 4 should also retroactively close Phase 1's optimizer carve-outs (LM / BFGS / Simplex / ConjugateGradient) — they're needed for model calibration.
