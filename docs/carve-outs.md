# PQuantLib carve-outs

Comprehensive list of C++ QuantLib v1.42.1 surface area NOT ported to PQuantLib through `pquantlib-final`. Each item includes the C++ source location, the rationale for deferral, and (where applicable) the access pattern for users who need the functionality.

Carve-out categories:

1. **Deferred-deliberately**: out of scope for the vanilla pricing + calibration mission; not blocked by any technical issue.
2. **Specialty domain**: would benefit from a dedicated follow-up phase.
3. **Engine-pair carry-over**: instrument ported but its engine is deferred (or vice-versa).
4. **Tooling boundary**: replaced by Python ecosystem tooling.

## Category 1 — Specialty domains (Phase 7+ candidates)

### Inflation

C++: `ql/instruments/{cpicapfloor,cpiswap,cpibond}.hpp`, `ql/cashflows/{cpicoupon,cpicouponpricer,capflooredinflationcoupon,yearonyearinflationcoupon}.hpp`, `ql/indexes/inflation/*` (8 files), `ql/termstructures/inflation/*` (8 files), `ql/termstructures/volatility/inflation/*` (4 files), `ql/pricingengines/inflation/*` (~15 files), `ql/experimental/inflation/*`.

**Why deferred:** Specialized domain; not on the vanilla pricing critical path. Users requiring CPI/YoY pricing should pin a target market (UK, US, EUR) and port the index families + CPISwap + YoYInflationSwap + ZeroCouponInflationSwap + their helpers individually. The inflation index machinery is mostly self-contained — Phase 5's `InflationIndex` Protocol would be a good starting point.

**Access**: none — direct port required.

### Credit — **CLOSED** by Phase 8 L8-B (Tier-1 CDS pricing)

C++: `ql/instruments/creditdefaultswap.hpp`, `ql/termstructures/credit/*` (11 files), `ql/pricingengines/credit/*`, `ql/experimental/credit/*` (~30 files), `ql/instruments/makecds.hpp`.

**Status:** Tier-1 vanilla CDS pricing is now ported. `pquantlib-phase8-complete` ships DefaultProbabilityTermStructure + 3 intermediates + FlatHazardRate + 3 interpolated curves + probability traits + PiecewiseDefaultCurve scaffold + Spread/UpfrontCdsHelper + CreditDefaultSwap + Claim + MidPoint/Integral CDS engines.

**Remaining (post-phase-8 follow-ups):** `IsdaCdsEngine` (ISDA-standard convention), `MakeCDS` factory, `implied_hazard_rate` + `conventional_spread` Brent wrappers, `FaceValueAccrualClaim` (accrual-rebate convention), Quanto CDS, all of `ql/experimental/credit/*` (CDO, basket CDS, CDS-on-CDS). `PiecewiseDefaultCurve` iterative bootstrap is a ~1-hour wiring follow-up against the L8-A `IterativeBootstrap[TS, Traits]` generic.

### Capfloor / optionlet / swaption volatility surfaces — **CLOSED** by Phase 8 L8-C + Phase 9 L9-C

C++: `ql/termstructures/volatility/capfloor/*` (5 files), `ql/termstructures/volatility/optionlet/*` (11 files), `ql/termstructures/volatility/swaption/*` (13 files), `ql/termstructures/volatility/{smilesection,*smilesection}.hpp`, `ql/termstructures/volatility/sabr.hpp`, `ql/math/interpolations/sabrinterpolation.hpp`.

**Status (Phase 8):** Term-vol + flat-vol + 1-D + 2-D bilinear surfaces. `pquantlib-phase8-complete` ships CapFloorTermVolatilityStructure family (4 classes) + OptionletVolatilityStructure family (7 classes incl. OptionletStripper1) + SwaptionVolatilityStructure family (5 classes incl. SwaptionVolatilityMatrix).

**Status (Phase 9):** Full SABR swaption smile cube + smile-section family. `pquantlib-phase9-complete` ships `sabr_volatility` + `sabr_normal_volatility` (Hagan 2002) + `SabrInterpolation` (scipy.optimize.least_squares) + `SmileSection` abstract + `FlatSmileSection` + `InterpolatedSmileSection` (uses L9-A `CubicNaturalSpline`) + `SabrSmileSection` + `SpreadedSmileSection` + `SwaptionVolatilityCube` abstract + `SabrSwaptionVolatilityCube` + `InterpolatedSwaptionVolatilityCube`. `SmileSection` base extended with `option_price` / `digital_option_price` / `density` helpers.

**Remaining (future specialty-cluster candidates):** `KahaleSmileSection` (no-arbitrage smile reformulation), `ZabrSmileSection` + `ZabrInterpolatedSmileSection` (ZABR specialty), `AtmAdjustedSmileSection` + `AtmSmileSection` (ATM-targeted adapters), `SabrInterpolatedSmileSection` (composition wrapper — ~1 hr plumbing), `XabrSwaptionVolatilityCube` template generalisation, `Gaussian1dSwaptionVolatility`, `CmsMarket`, `CmsMarketCalibration`, `OptionletStripper2`.

### MarketModels (LIBOR Market Model)

C++: `ql/models/marketmodels/*` (~125 files).

**Why deferred:** Large self-contained domain (forward LIBOR dynamics + LMM calibration + drift approximations + BGM). Not on the vanilla critical path. Users requiring LMM should port the `marketmodels/` tree as a dedicated cluster.

**Access:** none — direct port required. Or use QuantLib via the SWIG `QuantLib-Python` binding for LMM-only.

### Specialty volatility models

C++: `ql/models/volatility/{constantestimator,garch,garmanklass,parkinson,simplelocalestimator,yangzhang}.hpp`.

**Why deferred:** Historical-volatility estimators; replaced in modern practice by direct numpy/pandas implementations.

**Access:** Use `numpy` directly: `np.diff(np.log(prices)).std() * np.sqrt(252)` for annualized Parkinson-style estimators; GARCH via `arch` package.

### Specialty short-rate models — **partially CLOSED** by Phase 10 L10-B

C++: `ql/models/shortrate/onefactormodels/{gaussian1dmodel,gsr,markovfunctional}.hpp`.

**Status:** `Gaussian1dModel` abstract + `Gsr` concrete + `GsrProcess` + `Gaussian1dSwaptionVolatility` all ported in `pquantlib-phase10-complete`. Gaussian1d + GSR cover the Tier-1 specialty-short-rate use cases.

**Remaining (future-cluster candidates):** `MarkovFunctional` (542 LOC; second Gaussian1dModel concrete with bootstrap-vs-swaption-strip calibration — its own dedicated cluster). Plus `Gaussian1dGsrProcess` companion + `Gaussian1dCapFloorEngine` + `Gaussian1dFloatFloatSwaptionEngine` + `Gaussian1dNonStandardSwaptionEngine` (Gaussian1d-driven engines on top of the model). `Gaussian1dSmileSection` IborIndex ctor (depends on Gaussian1dCapFloorEngine).

### Specialty Heston variants — **partially CLOSED** by Phase 11 W1-D

Phase 11 W1-D landed:
- `GjrGarchModel` + `AnalyticGjrGarchEngine` (Duan et al. 2006 Edgeworth expansion).
- `PiecewiseTimeDependentHestonModel` + `AnalyticPiecewiseTimeDependentHestonEngine` (Gatheral form, time-segmented).
- `HestonSlvMcModel` (MC-bucketing leverage-function calibration).
- `HestonSlvFdmModel` (**scaffold only** — public API + unit-leverage fallback; the Fokker-Planck FDM solver depends on 2-D Heston FD operators scheduled for Phase 11 W5-C).

Still deferred (covered elsewhere in Phase 11 plan):
- `BatesDetJumpModel` + `BatesDoubleExpModel` + `BatesDoubleExpDetJumpModel` and their analytic engines → Phase 11 W1-C.
- `HestonSlvFdmModel` full implementation → Phase 11 W5-C (depends on `Fdm2dHestonOp` / `FdmHestonFwdOp` / `Concentrating1dMesher`).

**Access:** `from pquantlib.models.equity.gjr_garch_model import GjrGarchModel`, `from pquantlib.models.equity.piecewise_time_dependent_heston_model import PiecewiseTimeDependentHestonModel`, `from pquantlib.models.equity.heston_slv_mc_model import HestonSlvMcModel`, `from pquantlib.models.equity.heston_slv_fdm_model import HestonSlvFdmModel` (returns unit leverage).

## Category 2 — Engine-pair carry-overs

### Tree/lattice multi-asset

- **TreeLattice2D + G2.tree()**: 2-D lattice for G2 tree-based engines. G2 swaption is supported via `G2SwaptionEngine` (1-D analytic integration); the tree variant defers.
- **TFLattice variants** (TF-lattice scheme): one-off specialty.
- **Joshi4 / AdditiveEQP / Trigeorgis tree builders**: equity tree alternatives to CRR / JarrowRudd / Tian / LeisenReimer.

**Why deferred:** TreeLattice2D needs care + 2-D backward induction; the analytic G2SwaptionEngine covers the common case.

### Multi-asset finite-difference

- **`ql/methods/finitedifferences/*`** beyond the 1-D Black-Scholes subset ported in L5-D (~110 of 120 files).
  - 2-D Heston FD (`FdmHestonOp`, `FdmHestonVarianceMesher`, etc.).
  - 2-D G2 FD (`FdG2Op`).
  - 2-D Bates FD.
  - 2-D CIR / SABR FD.
  - Time-dependent operators (`FdmTimeDepBlackScholesOp`, `FdmAffineModelTermStructureFwdOp`).
  - Multi-D schemes (Craig-Sneyd, Hundsdorfer, MethodOfLines, TR-BDF2).
  - BoundaryCondition framework (`FdmDirichletBoundary`, `FdmNeumannBoundary`).
  - `Concentrating1dMesher` (mesh concentration around features).
  - Iterative solvers (BiCGstab, GMRES) for sparse non-tridiagonal systems.

**Why deferred:** 1-D BSM FD (L5-D) covers the most-cited vanilla American-option use case. Multi-asset FD is a research-grade undertaking that would benefit from its own dedicated phase.

**Access:** For Heston/Bates analytic pricing, `AnalyticHestonEngine` / `BatesEngine` from L4-C / L6-B cover characteristic-function-based pricing.

### Monte Carlo engine carry-overs

- **MC for Heston / G2 / Bates / HW**: process-specific MC engines.
- **Exotic MC**: `MCBarrierEngine`, `MCBasketEngine`, `MCLookbackEngine`, `MCCliquetEngine`, `MCAmericanBasketEngine`, `MCEuropeanBasketEngine`.
- **Sobol-based MC**: low-discrepancy MC path generators.

**Why deferred:** The MC framework (L5-C) + Longstaff-Schwartz American MC (L6-A) cover the highest-demand cases. Process- and instrument-specific MC engines are mostly mechanical applications of the framework with `PathPricer` specializations.

**Access:** Subclass `MCVanillaEngine` (from L5-C) with the appropriate `PathPricer` and the desired process — the framework supports this directly.

### Exotic option carry-overs

- **DoubleBarrier**: instrument + analytic engine ported (L6-C); **AnalyticDoubleBarrierBinaryEngine carved out** (C.H.Hui 307-line series).
- **PartialTimeBarrier**: barrier active only over a sub-period — Heynen-Kat formulas.
- **SoftBarrier**: Hart-Ross / Carr formulas.
- **HolderExtensibleOption**: holder can extend expiry by paying premium.
- **ComplexChooserOption**: option to choose option type at an intermediate date.
- **CompoundOption** (Geske 1979).
- **3+ asset baskets**: Stulz (L5-E) covers 2-asset; 3+ asset basket engines require multi-D bivariate normal.
- **`DoubleBarrierOption.implied_volatility`**: same pattern as Phase 3 `VanillaOption.implied_volatility` — needs the FD engine for double-barrier (deferred multi-asset FD).
- **AnalyticContinuousFixedLookbackEngine**: floating-strike lookback. (Floating-rate lookback is supported via L5-E.)

**Why deferred:** Each is closed-form but uses bespoke bivariate / trivariate normal CDF combinations. The Reiner-Rubinstein barrier framework (L5-E) covers the standard single-barrier cases.

**Access:** Use the appropriate analytic engine for the standard variants; deferred engines require porting the specific closed-form.

### Cashflow carry-overs

- **DigitalCoupon / DigitalIborCoupon / DigitalCmsCoupon**: coupon types with embedded digital options.
- **CmsCoupon / CmsSpreadCoupon**: CMS-rate-indexed coupons (need swap rate forecasting + convexity adjustment).
- **AverageBmaCoupon**: BMA-index-averaged coupon.
- **CapFlooredCoupon / CapFlooredOvernightIndexedCoupon**: coupons with cap/floor embedded.

**Why deferred:** Specialized; each requires a corresponding pricer that involves convexity adjustments or option-replication. The base Coupon + Pricer infrastructure (L2-D) supports adding these as needed.

### Bond carry-overs

- **BTP** (Italian government bond): specific BTP coupon conventions.
- **CmsRateBond**: bond paying CMS coupons.
- **ConvertibleBond**: bond convertible into equity (needs embedded option modeling).
- **CpiBond**: inflation-linked bond (defers with all inflation).
- **AmortizingCmsRateBond / AmortizingFloatingRateBond**: amortizing variants.

**Why deferred:** The Bond + FixedRateBond + FloatingRateBond + AmortizingFixedRateBond + ZeroCouponBond (L3-B) cover the vast majority of bonds. Specialty bonds defer per cashflow specialization.

### Ibor carry-overs

- 35 region-specialty ibors beyond the 8 ported (Euribor, USDLibor, GBPLibor, Eonia, Sofr, Sonia, FedFunds, Estr): AUDLibor, BBSW, Bibor, BKBM, CADLibor, CDI, Cdor, CHFLibor, Corra, Custom, Destr, DKKLibor, EURLibor, Jibar, JPYLibor, Kofr, Mosprime, NZDLibor, NZOCR, Pribor, Robor, Saron, SEKLibor, Shibor, Swestr, Thbfix, Tibor, Tona, Tonar, TRLibor, Wibor, Zibor, BMA, Equity, Aonia.

**Why deferred:** On-demand per market.

**Access:** subclass `IborIndex` (L2-C) with the appropriate (calendar, day-counter, fixing-days, currency) tuple.

## Category 3 — Tooling-boundary carry-outs

### Replaced by scipy / numpy / mpmath

- **`InverseCumulativeNormal` Acklam variant**: scipy.special.ndtri.
- **GammaFunction `Sankaran` approximation**: scipy.stats.ncx2 (Phase 5 L5-B).
- **Custom Gauss-Laguerre quadrature**: scipy.integrate.quad (Phase 4 L4-C).
- **Custom Gauss-Hermite quadrature**: scipy.integrate.quad / scipy.special.roots_hermite.
- **Custom matrix decompositions**: scipy.linalg.{cholesky, lu, svd, qr, eigh}.
- **Custom sparse iterative solvers**: scipy.sparse.linalg.{bicgstab, gmres, spsolve}.
- **Custom random number generation**: numpy.random + scipy.stats (Sobol/Halton via scipy.stats.qmc).
- **Direction integer tables for Sobol** (Jaeckel default / Unit / SobolLevitan / Kuo / etc.): scipy.stats.qmc uses Joe-Kuo by default.

### Replaced by Python language features

- **C++ `Visitor` pattern**: replaced by `isinstance` dispatch.
- **C++ `LazyObject` deferred-calc pattern**: replaced by direct method calls (Python is already late-binding).
- **C++ `template<class T>` strategy classes (e.g. `BinomialVanillaEngine<T>`)**: replaced by IntEnum + match-case.
- **C++ `Handle<X>`**: replaced by direct object references (Python is already reference-semantic).
- **C++ `Settings::instance().evaluationDate()`**: replaced by `ObservableSettings().evaluation_date_or_today()`.

## Category 4 — Deliberately deferred Phase 5 + 6 follow-ups

These were deferred during specific clusters with clear access patterns:

- **L4-B `tree(grid)` carve-out** for `OneFactorModel` subclasses: closed in L5-B.
- **L4-C `BatesEngine`**: closed in L6-B.
- **L4-E carry-overs**: SwapRateHelper / OISRateHelper / SwapIndex / FraRateHelper(useIndexedCoupon=True): closed in L3-C / L3-E.
- **L5-C American MC**: closed in L6-A.
- **L5-D `VanillaOption.implied_volatility`**: closed in L5-D.
- **L5-E `BivariateCumulativeNormalDistribution`**: closed in L5-E.

## Category 5 — Items not in C++ v1.42.1

Some PQuantLib-only additions don't have C++ analogues:

- Cross-cluster `@runtime_checkable` Protocols (`YieldTermStructureProtocol`, `IborIndexProtocol`, `OvernightIndexProtocol`, `SwapIndexProtocol`, `InstrumentProtocol`, `PricingEngineProtocol`, `StochasticProcessProtocol`, `ModelProtocol`, `CalibrationHelperProtocol`, `ShortRateModelProtocol`, `DiscretizedAssetProtocol`, `LatticeProtocol`, `PathGeneratorProtocol`).
- `pquantlib.testing.tolerance` + `pquantlib.testing.reference_reader` — testing harness.
- The `migration-harness/` C++ probe + JSON reference system.

## How to access carved-out functionality

If you need a carved-out feature:

1. **Check the C++ source** at `migration-harness/cpp/quantlib/ql/...` — the QuantLib v1.42.1 submodule is pinned in the harness.
2. **Check QuantLib-Python** (the SWIG binding): `pip install QuantLib` gives you the entire QuantLib via Python. Useful for one-off pricing of carved-out instruments.
3. **Use scipy / numpy / mpmath directly** for the numerical-tooling category — see "Category 3" above.
4. **Port the specific class** following the PQuantLib pattern: write a C++ probe, generate JSON reference values, write a failing Python test, port, verify. The full pattern is documented in `docs/migration/phase1-l1-A-design.md`.

## Statistics

- **C++ v1.42.1 surface**: ~2300 .hpp files.
- **PQuantLib ported**: ~454 classes across ~2652 tests (after Phase 7 inflation + Phase 8 piecewise/credit/capfloor-vol + Phase 9 cubic/post-L8-ergonomics/SABR-cube + Phase 10 vol-tail/Gaussian1d/interpolator-tail/ZABR opt-in extensions).
- **Test parity with `jquantlib-final`**: 2652 / 3610 = **73.5%**.

PQuantLib covers the **vanilla + American + analytic-exotic pricing + calibration** surface end-to-end, with substantial coverage of specialty short-rate models (Vasicek, HullWhite, CIR, ExtendedCIR, G2++, BlackKarasinski, **Gsr**), equity stochastic-vol (Heston, Bates), and the full Monte Carlo / Finite-Difference / Tree pricing infrastructure. Phase 7 added the full inflation cluster (Tier-1 indexes/curves/cashflows/instruments/vol). Phase 8 added piecewise inflation + IterativeBootstrap + Tier-1 credit (CDS pricing + hazard-rate term structures + MidPoint/Integral engines) + capfloor/optionlet/swaption vol surfaces (incl. OptionletStripper1 + SwaptionVolatilityMatrix). Phase 9 added cubic/bicubic spline interpolators + post-L8 ergonomics (PiecewiseYieldCurve + 3 yield traits + IsdaCdsEngine + MakeCDS + implied_hazard_rate + conventional_spread + PiecewiseDefaultCurve bootstrap wiring) + the full SABR swaption smile cube (5-class SmileSection family + Hagan 2002 SABR closed-form + SabrInterpolation scipy least-squares fitter + Sabr/Interpolated SwaptionVolatilityCube). Phase 10 closed the vol surface tail (KahaleSmileSection + AtmSmileSection + AtmAdjustedSmileSection + SabrInterpolatedSmileSection + OptionletStripper2 + SabrInterpolation Halton multi-start + HaltonRsg) + the Gaussian1d short-rate cluster (Gaussian1dModel + Gsr + GsrProcess + Gaussian1dSwaptionVolatility) + the interpolator tail (HymanFilteredCubic + ChebyshevInterpolation + MultiCubicSpline + AbcdInterpolation) + the ZABR family closed-form (zabr_volatility + ZabrSmileSection).

Remaining specialty domains (MarketModels, MarkovFunctional, experimental credit basket/CDO, CmsMarket/CmsMarketCalibration, ZABR FD modes, Gaussian1d-driven engines) defer to dedicated future work or QuantLib-Python wrapping. Post-Phase-10 backlog (none blocking): ZabrInterpolation 5-param fitter + ZabrInterpolatedSmileSection (~3 hr), XabrSwaptionVolatilityCube template generalisation (~3 hr), KahaleSmileSection.core_smile deep-iteration testing (~2 hr), Gaussian1d engines (~6 hr each), Hyman-1983 alt monotonic cubic semantics, ConvexMonotoneInterpolation (Hagan-West 2009), BootstrapError / LocalBootstrap alt bootstrap algorithms.
