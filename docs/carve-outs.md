# PQuantLib carve-outs

Comprehensive list of C++ QuantLib v1.42.1 surface area NOT ported to PQuantLib as of the terminal release (`pquantlib-final` / `pquantlib-100-complete` @ `1fdb1db`). Each item includes the C++ source location, the rationale for deferral, and (where applicable) the access pattern for users who need the functionality.

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

### MarketModels (LIBOR Market Model) — **CLOSED** by Phase 11 W9 + W10 + W11

C++: `ql/models/marketmodels/*` (~111 class-bearing headers).

**Status:** the entire BGM/LMM domain is ported across three Phase-11 waves:
- **W9 (core):** EvolutionDescription + CurveState (+ LMM/CMSwap/CoterminalSwap) + MarketModel/MarketModelEvolver/MarketModelMultiProduct/PathwiseMultiProduct/BrownianGenerator abstracts + discounters + swap-forward mappings + PiecewiseConstantCorrelation + exponential/time-homogeneous/cot-swap correlations + LMM/LMMNormal/SMM/CMSMM drift calculators + MT/Sobol Brownian generators + AccountingEngine + PathwiseAccountingEngine + ProxyGreekEngine + ConstrainedEvolver + SequenceStatistics.
- **W10 (models + evolvers):** FlatVol + AbcdVol + PiecewiseConstant{,Abcd}Variance + PseudoRootFacade + Fwd↔CotSwap/FwdPeriod adapters + VolatilityInterpolationSpecifier + 12 evolvers (LogNormalFwdRate Pc/Euler/Ipc/Balland/IBalland + Normal/CotSwap/CmSwap/SVD + SquareRootAndersen) + AlphaForm/AlphaFinder + CTSMM caplet-coterminal calibration (Original/MaxHomogeneity/AlphaForm/periodic).
- **W11 (products + callability + pathwise greeks):** the full MultiProduct framework + ~25 concrete products (MultiStep/OneStep + pathwise) + callability (exercise values/strategies + LS basis systems + LongstaffSchwartzExerciseStrategy + Andersen-Broadie UpperBoundEngine) + pathwise-greeks Jacobians (RatePseudoRoot + Swaption/Cap + VegaBumpCluster + VolatilityBumpInstrumentJacobian).

**Two canonical end-to-end tests pass** — `AccountingEngine(LogNormalFwdRatePc(FlatVol), MultiStepOptionlets)` → Black caplet prices, and the callable-swap Longstaff-Schwartz Bermudan test — proving the full stack works together.

**Remaining (single documented follow-up):** `PathwiseVegasAccountingEngine` (the ~1200-line Giles-Glasserman smoking-adjoints sweep) is unblocked (both `browniansThisStep` and RatePseudoRootJacobian exist) but left as a follow-up beyond the W11-D scope.

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

### Multi-asset finite-difference — **partially CLOSED** by Phase 11 W5-C

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

**Status (Phase 11 W5-C):** **Partially closed.** Now ported:
  - `NinePointLinearOp` + `SecondOrderMixedDerivativeOp` (2-D 9-stencil operator base + central mixed-derivative).
  - `FdmZabrOp` (2-D ZABR pricing PDE — `0.5 V^2 F^{2β} D_FF + 0.5 ν^2 V^{2γ} D_VV + νρ|V|^{γ+1} F^β D_FV`).
  - `FdmDupire1dOp` (1-D Dupire local-volatility op — `0.5 σ(S)^2 D_SS`).
  - `FdmOrnsteinUhlenbeckOp` (1-D OU PDE — `a(b-x) D_x + 0.5 σ^2 D_xx - rI`).
  - `FdmSimpleProcess1dMesher` (process-driven 1-D mesh anchored at x0).
  - `FdOrnsteinUhlenbeckVanillaEngine` (end-to-end FD engine for vanillas under arithmetic OU dynamics).
  - `FdmExtOUJumpModelInnerValue` (inner-value calculator for Kluge-style payoffs).
  - `FdmLinearOpComposite` Protocol (the schemes + backward solver now accept any conforming op, not just `FdmBlackScholesOp`).

**Remaining (W5-C deferred):**
  - `FdmExtOUJumpOp` + `ExtOUWithJumpsProcess` + `ExponentialJump1dMesher`: 2-D op for OU+jumps. Needs the Kluge process family and the exponential-jump 1-D mesher.
  - `FdmKlugeExtOUOp` + `KlugeExtOUProcess`: 3-D op composition for power+gas spread under Kluge+ExtOU.
  - `FdExtOUJumpVanillaEngine` + `FdKlugeExtOUSpreadEngine`: process-specific FD vanilla / spread engines. Each requires its op + a 2-D / 3-D solver decomposition (Hundsdorfer / Craig-Sneyd / TR-BDF2 — none of which is ported yet).
  - `FdmHestonFwdOp` + `FdmSquareRootFwdOp`: forward (Fokker-Planck) operators for the Heston SLV calibration.
  - `Concentrating1dMesher`: mesh concentration around strike/spot — required by both the BSM mesher's `c_point` parameter and the SLV calibration.
  - `LocalVolRNDCalculator` + `FdmHestonGreensFct`: Green's-function machinery underpinning the SLV's local-vol density target.
  - **Heston SLV Fokker-Planck FDM calibration**: the `HestonSlvFdmModel.leverage_function()` real implementation. The W1-D scaffold returns unit leverage (model degenerates to pure Heston). Real calibration requires all four bullets above plus 6 different time-stepping schemes with Rannacher smoothing.

**Why deferred:** The W5-C scope (per the cluster brief) was sized for 7 classes of straightforward 1-D / 2-D FD operators + 1 vanilla engine + 1 SLV calibration. The first 7 landed; the SLV FDM piece alone is 537 LOC of C++ depending on ~5 cross-cluster operators + meshers (~1500 LOC total). Lands as a dedicated Phase-12 cluster (or as an opt-in extension if a user surfaces the requirement).

**Access:**
  - For Heston/Bates *analytic* pricing, `AnalyticHestonEngine` / `BatesEngine` from L4-C / L6-B cover characteristic-function-based pricing.
  - For Heston SLV, the `HestonSlvFdmModel` scaffold accepts the public API but `leverage_function()` returns the unit-leverage degenerate; round-trip tests vs pure `HestonModel` pricing pass.
  - For ZABR via FD, `FdmZabrOp` provides the spatial operator. A `ZabrModel.fdmPrice()` wrapper still needs a 2-D solver loop (Craig-Sneyd-style) — separate follow-up.
  - For energy/power-gas, port the missing process + mesher + op trio against the existing `FdmExtOUJumpModelInnerValue` Python inner-value calculator. The W5-C ports give you the inner-value piece for free.

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
- **`moorePenroseInverse` custom SVD pseudo-inverse** (Phase 11 W6-C): `numpy.linalg.pinv`. The C++ hand-rolls `V·diag(1/sigma)·Uᵀ` from QuantLib's own `SVD` with cut-off `max(m,n)·eps·sigma_max`; numpy.linalg.pinv applies the identical relative cut-off (`rcond·sigma_max`) over LAPACK's divide-and-conquer SVD. The W6-C port reproduces the C++ default cut-off via a relative `rcond = max(m,n)·eps`.
- **Custom sparse iterative solvers**: scipy.sparse.linalg.{bicgstab, gmres, spsolve}.
- **`LaplaceInterpolation` FD-operator assembly** (Phase 11 W6-C): the C++ threads `Predefined1dMesher` + `FdmMesherComposite` + `SecondDerivativeOp.toMatrix()` + `BiCGstab`. The W6-C port inlines the identical second-derivative stencil (`2/zeta` weights) + Numerical-Recipes 3.8.6 corner weighting, assembles the system with `scipy.sparse.csr_matrix`, and solves it with `scipy.sparse.linalg.bicgstab` (the same Bi-CG-Stab algorithm). Inner/boundary/corner in-fills match the C++ reference grids exactly.
- **`GaussNonCentralChiSquaredPolynomial` weight** (Phase 11 W6-C): `scipy.stats.ncx2.pdf` for `w(x)`; the orthogonal-polynomial recurrence is ported directly (the C++ `MomentBasedGaussianPolynomial<mp_real>` arbitrary-precision moment accumulation is run in double precision only — matches the C++ `Real` specialization at the 1e-5 quadrature test tolerance; mpmath multiprecision is a deferred carve-out).
- **`MultidimGaussianQuadrature` node/weight solve** (Phase 11 W6-C): Golub-Welsch via `numpy.linalg.eigh` of the Hermite Jacobi matrix (C++ uses `TqrEigenDecomposition`); QuantLib's weight convention (`w_i = mu_0·ev[0,i]²/w(x_i)`) is reproduced so results match.
- **`GaussianQuadrature` 1-D node solve** (Phase 11 W8-D): standalone `pquantlib.math.integrals.GaussianQuadrature` drives Golub-Welsch via `scipy.linalg.eigh_tridiagonal` of the symmetric tridiagonal Jacobi matrix (diag `alpha(i)`, off-diag `sqrt(beta(i))`) — the C++ `GaussianQuadrature` uses its own `TqrEigenDecomposition`. The weight formula (`w_i = mu_0·ev[0,i]²/w(x_i)`, the Lebesgue convention) is ported line-for-line; nodes match the C++ TQR to ~1e-11. Used by `SquareRootCLVModel`'s collocation.
- **Gauss-Hermite collocation abscissae** (Phase 11 W8-D): `NormalCLVModel` uses `sqrt(2)·scipy.special.roots_hermite(n)` for the OU collocation nodes (the C++ `M_SQRT2·GaussHermiteIntegration(n).x()`); scipy returns ascending physicists' Hermite roots, reversed to match the C++ descending node order.
- **Non-central chi-squared quantile** (Phase 11 W8-D): `SquareRootCLVModel`'s `pMin`/`pMax` collocation clamps use `scipy.stats.ncx2.ppf` (the C++ used `boost::math::quantile`); the CDF uses the ported `NonCentralCumulativeChiSquareDistribution` (itself `scipy.stats.ncx2.cdf`).
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
- **W6-C `LatentModel<Impl>::FactorSampler` specializations**: the general `experimental.math.LatentModel` ports the integration/inspector surface (factor loadings, per-variable correlation, copula cumulative/inverse passthroughs, `latent_var_value`, `integrated_expected_value`); the C++ `Impl`-driven random-sample machinery (the `FactorSampler` partial specializations for Gaussian/T copulas) is deferred — the W6-C copula RNGs (`ClaytonCopulaRng` / `FrankCopulaRng` / `FarlieGumbelMorgensternCopulaRng` / `PolarStudentTRng`) and `GaussianCopulaPolicy.all_factor_cumul_inverter` provide direct factor-sample generation when a simulation is needed.
- **W8-C `experimental.mcbasket` Longstaff-Schwartz multipath engines**: the W8-C cluster ports the deterministic mcbasket scaffolding — `PathPayoff` (abstract) + `AdaptedPathPayoff` (the `ValuationData` adapted-accessor, enforcing non-anticipating payoffs) + `PathMultiAssetOption` (instrument) + `MCPathBasketEngine` (European path-dependent basket MC, with `EuropeanPathMultiPathPricer`). The **American multi-asset** variant — `LongstaffSchwartzMultiPathPricer` (309-line LS regression over multi-asset paths) + `MCLongstaffSchwartzPathEngine` + `MCAmericanPathEngine` — is deferred: a ~3-hour follow-up that extends the L6-A `LongstaffSchwartzPathPricer` / `LsmBasisSystem` to multi-asset states using the W8-C `PathPayoff.basis_system_dimension()` + the `states`/`exercises` outputs already plumbed through `AdaptedPathPayoff.set_exercise_data`. The European engine + the deterministic scaffolding cover the non-exercise path; for American multi-asset basket pricing today, the L6-A single-asset `MCAmericanEngine` (regression LS) is the closest in-tree option.

### Arithmetic-average OIS — **removed upstream (deprecated v1.41)**

The W8-D plan called for `ArithmeticAverageOIS` + `ArithmeticOISRateHelper` +
`MakeArithmeticAverageOIS` (C++ `ql/experimental/averageois/`). On inspection
of v1.42.1 (`099987f0`), **all three headers are empty deprecation stubs**:

```cpp
// ql/experimental/averageois/arithmeticaverageois.hpp (v1.42.1)
// Deprecated in version 1.41
#pragma message("Warning: this file is empty and will disappear in a future release; do not include it.")
```

The class definitions were deleted upstream — there is no
`class ArithmeticAverageOIS` anywhere in the v1.42.1 tree (verified by
`grep -rl "class ArithmeticAverageOIS"`). Since **C++ v1.42.1 is the source
of truth** and the source of truth contains no implementation, these are
carved out: there is nothing to port. The arithmetic-averaging convexity
adjustment they once provided over compounded OIS is, in modern QuantLib,
subsumed by the standard `OvernightIndexedSwap` (L3-C, already ported) plus
the ibor-ibor / overnight-ibor basis-swap rate helpers added in W8-D
(`IborIborBasisSwapRateHelper`, `OvernightIborBasisSwapRateHelper`). If the
historical arithmetic-average behaviour is ever needed, recover it from a
pre-1.41 QuantLib tag or from QuantLib-Python's matching version.

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

## Statistics (final — after Phase 11 full-closure)

- **C++ v1.42.1 surface**: 1320 class-bearing `.hpp` files (excl. `all.hpp`) / 2024 distinct class+struct names.
- **PQuantLib ported**: ~895 Python modules across **4048 tests** / 499 test files.
- **Test parity with `jquantlib-final`**: 4048 / 3610 = **112%** — PQuantLib now exceeds JQuantLib's final test count (JQuantLib never reached the `experimental/*` + full `marketmodels/*` surface).
- **Raw name-match coverage** (`migration-harness/check_coverage.py`): 1024 / 2024 = 50.6%. **This is a heavily-distorted floor, not the functional measure** (see the audit-triage note below).

### The 50.6% raw figure is a floor — the functional surface is essentially complete

The audit script matches by `class Name`. The ~1000 raw "missing" names split into four buckets, only the first of which is a real gap (and Phase 11 W12 closed the bulk of it):

1. **Real gaps — now CLOSED by W12** (core `cashflows`): CmsCoupon + CmsCouponPricer + CmsLeg + GFunction family + HaganPricer/AnalyticHaganPricer/NumericHaganPricer + ConundrumIntegrand + CappedFlooredCoupon/Ibor/Cms/Overnight + DigitalCoupon/Ibor/Cms + DigitalReplication + StrippedCappedFlooredCoupon + Dividend/FixedDividend/FractionalDividend + AverageBMACoupon/Leg + overnight-indexed coupon pricers + LognormalCmsSpreadPricer (also closed the W8-A CMS-spread deferral) + upgraded the L2-D BlackIborCouponPricer stub to a real optionlet pricer. **Remaining real gaps:** EquityCashFlow + EquityIndex + QuantoTermStructure (a small self-contained equity-return-cashflow batch deferred during W12 — see Category 2); CMS/overnight cap-floor *pricing* (needs a Hagan-replication / Black-overnight caplet pricer).

2. **Representation mismatches — NOT gaps** (the bulk of the raw "missing"). The name-match audit cannot see these:
   - `currencies` (5/111): every per-currency class (`USDCurrency`, `EURCurrency`, …, 100+ of them) is ported as a `Currency` *instance / registry entry*, not a `class X` subclass. Functionally complete.
   - Leg builders (`IborLeg`, `FixedRateLeg`, `CmsLeg`, …) are ported as `ibor_leg(...)` / `fixed_rate_leg(...)` *functions*, not classes.
   - Engine `Arguments` / `Results` nested structs are folded into the Python engine (no separate class).
   - `pricingengines` (62/221 raw) and `math` (89/286 raw) are dominated by these nested-helper / template-tag / `detail::` names.

3. **Superseded / not-in-scope** (formal carve-out): `ql/legacy/*` = the LIBOR Forward Model (`LiborForwardModel` + `Lm{Correlation,Volatility}Model` family + `LfmSwaptionEngine`) — **superseded** by the `marketmodels/*` (BGM) domain that Phase 11 W9–W11 fully ported; `ql/utilities/*` C++ idioms (`Clone`, `Null`, `Tracing`, `DateParser`, `PeriodParser`, `ObservableValue`) that Python handles natively (smart-pointer clone → `.clone()`, sentinel → `None`, RAII tracing → logging).

4. **Permanently-delegated** (Category 3): historical-vol estimators (Parkinson, GarmanKlass, YangZhang, …) + GARCH → numpy / scipy / arch. Not ported by design.

### What PQuantLib covers (functional summary)

The **complete** vanilla + American + analytic-exotic + calibration surface, plus: all short-rate models (Vasicek/HullWhite/CIR/ExtendedCIR/G2++/BlackKarasinski/Gsr/**MarkovFunctional**) + equity stochastic-vol (Heston/Bates + **all Bates variants** + **GjrGarch** + **PiecewiseTimeDependentHeston** + **HestonSLV** MC) + Gaussian1d (model + engines + swaption vol) + the full SABR/ZABR/SVI/NoArbSABR smile-cube family + full inflation (Tier-1 + piecewise + experimental YoY-optionlet-stripping) + full credit (Tier-1 CDS + experimental CDO/basket/NthToDefault) + the complete commodity/energy stack + variance-gamma + FFT engines + CLV models + **the entire MarketModels/BGM/LMM domain** (W9–W11, with two passing canonical end-to-end validations) + the experimental finite-difference / exotic-option / heuristic-optimizer trees + the core cashflows CMS/CappedFloored/Digital coupon families (W12).

**Remaining backlog (none blocking; all documented):** EquityCashFlow batch; CMS/overnight cap-floor pricing; PathwiseVegasAccountingEngine (Giles-Glasserman); MarkovFunctional caplet-calibration + Gaussian1d-engine edge cases; the W8-B `cluster_w8b/probe.cpp` reconstruction (reference JSON committed, tests pass). These are small follow-ups against a functionally-complete library.
