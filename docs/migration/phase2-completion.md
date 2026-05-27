# Phase 2 (L2 termstructures + indexes + cashflows) — completion

**Date closed:** 2026-05-26
**Tag:** [`pquantlib-phase2-complete`](../../README.md#migration-status) @ `b5d2519`
**Predecessor:** `pquantlib-phase1-complete` @ `edcadbc`
**Test count:** 581 → **922/0/0** (+341). pyright + ruff clean.
**Design spec:** [`phase2-design.md`](phase2-design.md). **Plan:** [`phase2-plan.md`](phase2-plan.md).

## Cluster contribution table

| Cluster | Mode | Commits | Tests added | Coverage |
|---|---|---|---|---|
| **L2-A pilot** | sequential, 6 stages | 6 | +68 | quotes (Quote/SimpleQuote/Derived/Composite) + TermStructure abstract + Extrapolator + BootstrapHelper + PillarChoice + Index/IndexManager + 4 cross-cluster Protocols (Yield/Ibor/Overnight/Swap) |
| **L2-B** yield curves | parallel | 7 | +50 | YieldTermStructure + ZeroYieldStructure + FlatForward + Interpolated{Zero,Forward,Discount}Curve + 3 spreaded variants + ImpliedTermStructure + Compounding + InterestRate |
| **L2-C** indexes + rate helpers | parallel | 4 | +77 | InterestRateIndex/IborIndex/OvernightIndex/SwapIndex abstracts + 8 ibor concretes (Euribor/USDLibor/GBPLibor/Eonia/Sofr/Sonia/FedFunds/Estr) + 2 swap indexes (EuriborSwapIsdaFixA/UsdLiborSwapIsdaFixAm) + 7 rate helpers (Deposit/FRA/Futures/Swap/OIS/Bond/FxSwap) |
| **L2-D** cashflows | parallel | 6 | +50 (post-dedup) | CashFlow + SimpleCashFlow + Coupon + FixedRateCoupon + FloatingRateCoupon + IborCoupon + OvernightIndexedCoupon + 3 leg generators + 3 coupon pricers + CashFlows aggregator + Duration enum |
| **L2-E** vol termstructures | parallel | 5 | +96 | VolatilityTermStructure + SmileSection + FlatSmileSection + BlackVol/LocalVol family (TermStructure abstract + Constant + Curve + Surface variants) |
| **Total** | | **~28** | **+341** | **~70 classes** |

L2-A pilot was tagged separately as `pquantlib-phase2-l2-A-complete` @ `4ace1f0` mid-phase; that tag is preserved as the predecessor for the parallel cluster dispatch.

## Parallelization wins

Same fan-out pattern as Phase 1 L1-B/C/D/E, now proven across two phases:

- **Wall-clock**: 4 cluster subagents (B/C/D/E) all returned within ~35 min wall-clock. Total +273 tests across 4 disjoint topic areas (yield curves / indexes / cashflows / vol surfaces).
- **Zero deadlock**: cross-cluster Protocols (`YieldTermStructureProtocol`, `IborIndexProtocol`, `OvernightIndexProtocol`, `SwapIndexProtocol` — defined in L2-A) let agents reference each other's concretes structurally without import cycles. At merge time, structural typing matched up automatically.
- **No probe rebuild per worktree**: subagents used the main worktree's already-built libQuantLib via one-off `clang++` linking with absolute paths (saved 4×5–10 min QL rebuilds).

## Merge reconciliation

Four issues caught + resolved at merge:

1. **`Compounding` enum + `InterestRate` class duplicated** across L2-B and L2-D. Both clusters needed the type; both ported it independently to different locations (L2-B → `pquantlib.time.compounding` + `pquantlib.interest_rate`; L2-D → `pquantlib.cashflows.{compounding,interest_rate}`). Resolved by keeping L2-B's locations (which match C++ `ql/compounding.hpp` + `ql/interestrate.hpp` placement), removing L2-D's duplicate files, rewriting L2-D's imports + attribute accesses to method calls (L2-B uses traditional class-with-methods; L2-D had used `@dataclass(frozen=True, slots=True)` with public attrs).
2. **L2-D's duplicate `test_interest_rate.py`** removed (L2-B's covers the same surface; 16 vs 12 tests, ~70% overlap).
3. **`pquantlib.termstructures.yield_` subpackage `__init__.py`** docstring conflict (L2-B and L2-C both created it). Resolved by merging the two docstrings.
4. **`CMakeLists.txt`** had four parallel `add_executable` additions for cluster probes. Resolved by stacking all four entries.

## Cross-cluster Protocol design (validated)

L2-A's `@runtime_checkable` Protocols proved out as expected:

- L2-C's `Euribor` (concrete `IborIndex`) → satisfies `IborIndexProtocol`.
- L2-C's `Eonia`, `Sofr`, etc. → satisfy `OvernightIndexProtocol`.
- L2-B's `FlatForward` → satisfies `YieldTermStructureProtocol`.
- L2-D's `IborCoupon(index: IborIndexProtocol, ...)` → at merge time, L2-C's `Euribor` plugs in via structural matching, no glue code.
- L2-D's `CashFlows.npv(legs, discount_curve: YieldTermStructureProtocol, ...)` → L2-B's `FlatForward` plugs in.

The cost of defining the Protocols upfront (5 tests, ~100 LOC in L2-A) paid for itself many times over by eliminating cluster sequencing.

## Cumulative documented divergences from C++ v1.42.1

In addition to Phase 1's divergences:

### Type-level
- **`Compounding` + `InterestRate` placements**: C++ has both at namespace root (`ql/compounding.hpp`, `ql/interestrate.hpp`). PQuantLib places `Compounding` under `pquantlib.time` (next to `Frequency`) and `InterestRate` at `pquantlib.interest_rate` root.
- **`PiecewiseYieldCurve<Traits, Interpolator>` C++ template** → single Python class parameterized by an `Interpolation` instance + bootstrap traits enum.
- **L2-B's `InterpolatedZeroCurve` / `InterpolatedForwardCurve` / `InterpolatedDiscountCurve`** use a single Python class parameterized by an `InterpolationFactory = Callable[[Array, Array], Interpolation]`. C++'s template-on-Interpolator is the same idea.
- **PEP 695 type aliases** (`type ZeroCurve = InterpolatedZeroCurve`) for the C++ default-interpolator typedefs.
- **`yield_` subpackage** — `yield` is a Python keyword; trailing underscore mirrors a common idiom. C++ `ql/termstructures/yield/` — same content.

### Construction modes
- **TermStructure / VolatilityTermStructure moving-reference-date mode** (mode 3 — settlement_days + `Settings.evaluation_date` observer wiring) NOT ported. Subclasses using moving mode would currently need to override `reference_date()` themselves.
- **`Settings.evaluation_date` observable** NOT yet wired in PQuantLib's `ObservableSettings`. Schedule, TermStructure-moving-mode, RelativeDateBootstrapHelper, and SmileSection-floating-mode all defer to it.
- **L2-D's `CashFlows` aggregator**: methods take an explicit `settlement_date` arg (no default to `Settings.evaluation_date`).
- **Free-function leg generators** (`fixed_rate_leg`, `ibor_leg`, `overnight_leg`) replace C++'s Builder pattern (`FixedRateLeg(...).withNotionals(...).withSchedule(...)`).
- **`InterestRate` null sentinel** uses NaN + `is_null()` (L2-B); not the C++ `Null<Real>()` poison-double.

### Deprecated C++ paths not ported
- **`BootstrapError` template class** (deprecated in C++ v1.40 — "Use a lambda instead"). Use `Callable[[float], float]` directly.
- **`IndexManager` per-index notifier subsystem** (deprecated in C++ v1.42.1). Use `Index.update()` directly.
- **`RelativeDateBootstrapHelper`** (depends on Settings.evaluation_date observer).
- **`Visitor` accept() at cashflow level** (cap on cashflows.bps() reimplemented without visitor dispatch).
- **`LazyObject` / `performCalculations` deferred evaluation** — Python's coupons just compute eagerly each call.

### Algorithmic simplifications
- **`OvernightIndexedCoupon`**: defaults to `RateAveraging.Compound` with no lookback/lockout/observation-shift/compound-spread-daily/telescopic-value-dates. The 6 modes deferred to a follow-up.
- **`BlackIborCouponPricer`** cap/floor methods raise `LibraryException` — they need `OptionletVolatilityStructure` (deferred). Plain swaplet behaves identically to `IborCouponPricer`.
- **`LocalVolSurface`** flat-curve simplification (zero risk-free, zero dividend, forward = spot). Full version with yield curves deferred.
- **`BlackVarianceCurve` BlackVolTimeExtrapolation** strategies (FlatVolatility/UseInterpolator/LinearVariance) collapsed to default FlatVolatility.

## Carve-outs (deferred from Phase 2)

These were intentionally not ported. Each gets either a dedicated follow-up cluster or lands as part of the layer that consumes it.

### Whole subsurfaces
- **Inflation** (everything inflation-related): termstructures/inflation, indexes/inflation, volatility/inflation, cashflows/cpi_coupon. ~25+ classes. Dedicated Phase 5 (or pre-Phase-3 inflation cluster).
- **Credit** (DefaultProbabilityTermStructure, PiecewiseDefaultCurve, hazard-rate / survival-probability traits, CdsHelper). 11 headers. Phase 4 or Phase 5.
- **ZABR / SABR / XABR volatility models** (depend on optimization concretes deferred from Phase 1).
- **Capfloor / Optionlet / Swaption volatility surfaces** (29 headers; specialized interest-rate volatility — Phase 4 alongside the relevant models).

### Specialty / advanced
- **35 of 43 region-specialty ibor concretes**: AUDLibor, BBSW, Bibor, BKBM, CADLibor, CDI, Cdor, CHFLibor, Corra, Custom, Destr, DKKLibor, EURLibor, Jibar, JPYLibor, Kofr, Mosprime, NZDLibor, NZOCR, Pribor, Robor, Saron, SEKLibor, Shibor, Swestr, Thbfix, Tibor, Tona, Tonar, TRLibor, Wibor, Zibor, BMA, Equity, Aonia. On-demand.
- **Specialty cashflows**: DigitalCoupon, DigitalIborCoupon, DigitalCmsCoupon, CmsCoupon, CmsSpreadCoupon, AverageBmaCoupon, CapFlooredCoupon, CapFlooredInflationCoupon, EquityCashflow, Dividend, ConundrumPricer, LinearTsrPricer.
- **Yield-curve specialty**: FittedBondDiscountCurve, CompositeZeroYieldStructure, MultiCurve, GlobalBootstrap, LocalBootstrap, MultipleResetsSwapHelper, OvernightIndexFutureRateHelper, QuantoTermStructure, NonlinearFittingMethods, CubicBSplinesFitting, ExponentialSplinesFitting, NaturalCubicFitting, SpreadFittingMethod, SpreadTraits, CPIBondHelper, InterpolatedSimpleZeroCurve, PiecewiseSpreadYieldCurve, PiecewiseForwardSpreadedTermStructure, PiecewiseZeroSpreadedTermStructure, ForwardStructure.
- **PiecewiseYieldCurve full bootstrap** (the scaffolding from L2-B exists but the per-trait bootstrap was deferred — needs L2-C rate helpers wired in).
- **SwapIndex.forecast_fixing() + underlying_swap()** (need VanillaSwap from L3).
- **SwapRateHelper.implied_quote() / OISRateHelper.implied_quote() / BondHelper.implied_quote()** (need L3 pricing engines).
- **FraRateHelper(useIndexedCoupon=True)** (needs L2-D IborCoupon — wire-up pending).

### Phase-1 carry-overs still open
- Full `GaussianOrthogonalPolynomial` hierarchy.
- SobolRsg / Burley2020SobolRsg low-discrepancy.
- LM / BFGS / Simplex / ConjugateGradient / SimulatedAnnealing optimizers.
- 8+ cubic-spline variants (AkimaCubic, KrugerCubic, FritschButland, etc.).
- QR / Eigenvalue / SVD / SparseMatrix utilities.
- Full GammaFunction (currently delegated to `math.lgamma`).

## Lessons learned

- **Pre-defining cross-cluster Protocols saved significant integration time.** L2-A spent ~30 min on 4 Protocols + 5 tests. That up-front cost eliminated cluster sequencing entirely — all 4 of B/C/D/E ran truly parallel.
- **Subagents duplicating peripheral types is inevitable when their work overlaps.** L2-B and L2-D both needed `Compounding` + `InterestRate`. The fix at merge time (rename imports + attribute→method) was mechanical but real (15 min of cleanup). For Phase 3+, consider pre-porting any cross-cluster types in the pilot before dispatching.
- **One-off `clang++` compile against main worktree's library** saved 4×5–10 min QL rebuilds. Worth documenting in the standard subagent prompt template.
- **Subagent file leakage**: two subagents wrote probe files to the main worktree path via shared absolute paths. Caught at merge time; cleanup was easy because the files were also committed in the subagents' own branches. For Phase 3, instruct subagents to never write to main worktree paths even when using main's library/headers for compile.
- **The `_between` vs `_dates` naming clash** between L2-B's and L2-D's InterestRate cost ~10 min of rewriting at merge. Both ports independently chose different names for the date-based overloads. Pre-naming convention in the pilot would have prevented this.
- **Test coverage exploded faster than class count.** Phase 2 ported ~70 classes but added 341 tests (5×). This is healthy — every class has 4–6 behavioral + numerical-parity tests. Test count is now ~922 and growing at the ratio needed to lock in correctness as L3 instruments start composing L2 primitives.

## Next: Phase 3 (L3 instruments + pricingengines)

Sister-project anchor: `jquantlib/docs/migration/phase2-L3-instruments-pricingengines-plan.md`. Expected scope: ~80–120 classes (bonds, vanilla swaps, European options, pricing engines, payoffs). Largest L-layer.

Phase 1 carry-overs that may need to land mid-Phase-3:
- `Settings.evaluation_date` observable wiring (unblocks Schedule fallback + TermStructure moving mode + RelativeDateBootstrapHelper + several pricing engines that rely on the global eval date).
- `OptionletVolatilityStructure` (unblocks BlackIborCouponPricer cap/floor branches).
- Concrete optimizer (LM or BFGS) for any calibration engine that needs one.
