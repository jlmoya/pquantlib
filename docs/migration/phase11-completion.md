# Phase 11 — completion (in progress)

**Started:** 2026-05-29
**Predecessor:** `pquantlib-phase10-complete` @ `d3746e4` — 2652/0/0
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

Living document — updated incrementally as each wave closes.

## Wave summary

| Wave | Tag | Tests delta | Cumulative | Date |
|---|---|---|---|---|
| W1 | `pquantlib-phase11-w1-complete` @ `07712fa` | +123 | 2775 | 2026-05-29 |
| W2 | `pquantlib-phase11-w2-complete` @ `d345bde` | +100 | 2875 | 2026-05-29 |
| W3 | `pquantlib-phase11-w3-complete` @ `ea41029` | +229 | 3104 | 2026-05-29 |
| W4 | `pquantlib-phase11-w4-complete` @ `eb901ae` | +68 | 3172 | 2026-05-29 |
| W5 | `pquantlib-phase11-w5-complete` @ `6e0f8d2` | +127 | 3299 | 2026-05-29 |
| W6 | `pquantlib-phase11-w6-complete` @ `ac69e2f` | +169 | 3468 | 2026-05-31 |
| W7 | `pquantlib-phase11-w7-complete` @ `6ae1cfd` | +152 | 3620 | 2026-05-31 |
| W8 | `pquantlib-phase11-w8-complete` @ `8573dd9` | +143 | 3763 | 2026-05-31 |
| W9 | `pquantlib-phase11-w9-complete` @ `4d09234` | +82 | 3845 | 2026-05-31 |
| W10 | `pquantlib-phase11-w10-complete` @ `1d559de` | +64 | **3909** | 2026-05-31 |

## W1 — Specialty model completion

4 parallel clusters off `pquantlib-phase10-complete`.

| Cluster | Branch tip | Classes | Tests | Carve-out closed / deferred |
|---|---|---|---|---|
| W1-A MarkovFunctional | `1f0aa63` | 3 (MarkovFunctional + MarkovFunctionalSwaptionEngine + Gaussian1dGsrProcess) | +24 | closes MarkovFunctional carve-out (caplet calibration / Kahale-SABR pretreatment / Bermudan exercise / AdjustYts / AdjustDigitals deferred) |
| W1-B Gaussian1d engines | `b66589b` | 3 engines + 4 instruments (FloatFloatSwap/Swaption + NonstandardSwap/Swaption) | +8 | closes L10-B Gaussian1d engines carve-out (BasketGeneratingEngine.calibrationBasket / Probabilities enum / OAS / RebatedExercise / CMS dispatch deferred) |
| W1-C Bates variants | `47cb885` | 3 models + 2 engines | +43 | closes Bates variant carve-out (AnalyticBatesDoubleExpDetJumpEngine still deferred — model landed, engine pending) |
| W1-D Heston SLV + GjrGarch + time-dep | `f272df5` | 4 models + 2 engines + FixedLocalVolSurface | +48 | closes specialty-Heston carve-out (HestonSlvFdmModel scaffold-only — full FDM scheduled for W5-C; PTD engine AndersenPiterbarg branch deferred) |

**Total W1:** 14 classes / +123 tests / 2652 → 2775.

## W2 — ZABR + smile + bootstrap follow-ups

2 parallel clusters off `pquantlib-phase11-w1-complete`. Originally planned as pilot + 2-parallel; ZABR FD evaluation modes (W2-A's planned scope) punted to W5 (multi-asset FDM cluster), simplifying to 2-parallel-no-pilot.

| Cluster | Branch tip | Classes | Tests | Carve-out closed / deferred |
|---|---|---|---|---|
| W2-A ZABR fitter + smile + Xabr cube | `3a3473d` | 4 (ZabrInterpolation + ZabrInterpolatedSmileSection + XabrSwaptionVolatilityCube + ZabrSwaptionVolatilityCube). SabrSwaptionVolatilityCube refactored to subclass Xabr (back-compat). | +39 | closes L10-C ZabrInterpolatedSmileSection + L9-C/L10-C XabrSwaptionVolatilityCube generalization carve-outs (ZABR FD modes still deferred to W5) |
| W2-B Bootstrap + smile + CMS | `1084461` | 7 (ConvexMonotoneInterpolation Hagan-West 2009 + BootstrapError + LocalBootstrap + CmsMarket + CmsMarketCalibration + AbcdCalibration + KahaleSmileSection.core_smile deep) | +61 | closes L10-C ConvexMonotone + L9-B BootstrapError/LocalBootstrap + L9-C CmsMarket/CmsMarketCalibration + L10-C AbcdCalibration + L10-A KahaleSmileSection.core_smile carve-outs |

**Total W2:** 11 classes / +100 tests / 2775 → 2875.

## W3 — experimental/credit/*

Pilot (W3-A) + 3-parallel (W3-B/C/D). +229 tests / 2875 → 3104. Closed the full vanilla credit-experimental surface: foundation types (DefaultType/Event/Key + Issuer/Pool + RecoveryRate + Loss/Distribution/LossDistribution), copulas + correlation + 5 default-loss models + 4 latent models, hazard-rate specialty curves + Basket + SyntheticCDO + Integral/MidPoint CDO engines, NthToDefault + CDSOption + BlackCDSOptionEngine + RandomDefaultModel + RiskyAssetSwap(Option). Bonus: closed L1 BinomialDistribution carry-over. Mid-merge align: split `DefaultLossModel` into a Protocol (W3-C/engine interface) + `DefaultLossModelBase` ABC (W3-B concrete interface). Deferred: InterpolatedAffineHazardRateCurve + OneFactorAffineSurvival (affine-model glue), CDO generic-copula variant, heterogeneous-basket NTD.

## W4 — experimental/exoticoptions + barrieroption + varianceoption

3-parallel. +68 tests / 3104 → 3172. W4-A multi-asset exotics (Himalaya/Everest/Pagoda + MC engines + TwoAssetBarrier/Correlation + StochasticProcessArray + MultiAssetOption); W4-B compound/chooser/Kirk-spread/holder-writer-extensible + AnalyticPdfHeston + continuous-Asian Levy/Vecer; W4-C PartialTimeBarrier + BinomialDoubleBarrier + SoftBarrier + VarianceOption + IntegralHestonVarianceOptionEngine. Deferred: VannaVolga barrier engines (FX DeltaVolQuote/BlackDeltaCalculator chain → W8), AnalyticBatesDoubleExpDetJumpEngine (W1-C carry-over).

## W5 — experimental/finitedifferences/*

3-parallel-no-pilot (each cluster off the W4 base). +127 tests / 3172 → 3299. Tag `pquantlib-phase11-w5-complete` @ `6e0f8d2`. **Note:** this wave was paused mid-flight for a computer restart (see [`phase11-w5-resume-checkpoint.md`](phase11-w5-resume-checkpoint.md) — now resolved) and the three branches were merged on resume.

| Cluster | Tests | Scope |
|---|---|---|
| W5-A ExtOU/Kluge FD infra | +49 | Glued1dMesher + FdmExtendedOrnsteinUhlenbeckOp + FdmExtOUJumpOp + FdmKlugeExtOUOp + NinePointLinearOp + SecondOrderMixedDerivativeOp + 3 processes + 2 inner-value calcs + 4 Fdm*Solver scaffolds |
| W5-B VPP/swing/storage | +53 | SwingExercise + VanillaVPPOption + VanillaStorage/SwingOption + FdmVPPStepCondition family + DynProgVPPIntrinsicValueEngine + 3 FdSimple* scaffolds + process Protocols |
| W5-C ZABR/Dupire/OU FD | +25 | FdmZabrOp (closes W2-A ZABR-FD carve-out at op level) + FdmDupire1dOp + FdmOrnsteinUhlenbeckOp + FdmSimpleProcess1dMesher + FdmExtOUJumpModelInnerValue + FdOrnsteinUhlenbeckVanillaEngine + FdmLinearOpComposite Protocol refactor |

**Merge reconciliations:** add/add docstring-union on `experimental/finitedifferences/__init__.py` (all 3 branches created it docstring-only); CMakeLists stacking ×3. W5-C's `FdmLinearOpComposite` Protocol refactor of the scheme classes (ExplicitEuler/ImplicitEuler/CrankNicolson/FdmBackwardSolver + FdmBlackScholesOp) preserved all existing L5-D FD-engine behavior (full pytest green post-merge).

**W5 follow-ups (carve-outs):** multi-D backward FDM framework (Fdm2DimSolver/Fdm3DimSolver/Hundsdorfer scheme) blocks full pricing of the 7 FdSimple*/FdExtOU* engine scaffolds; HestonSlvFdmModel Fokker-Planck wiring still scaffold-only; `NinePointLinearOp` + `SecondOrderMixedDerivativeOp` duplicated across `experimental/finitedifferences/` (W5-A) and `methods/finitedifferences/operators/` (W5-C) — dedup pending.

## W6 — experimental/volatility + experimental/math

4-parallel-no-pilot. +169 tests / 3299 → 3468. Tag `pquantlib-phase11-w6-complete` @ `ac69e2f`. The 4 `zabr*` files in `experimental/volatility` were correctly skipped (already on main from L10-C + W2-A).

| Cluster | Tests | Scope |
|---|---|---|
| W6-A NoArbSABR + SVI | +30 | NoArbSabrModel (Doust 2012, byte-exact 1.2M-entry absorption-prob table as a 3.6 MB gz numpy asset) + D0Interpolator + NoArbSabr{Interpolation,SmileSection,InterpolatedSmileSection,SwaptionVolatilityCube} + SVI (svi_volatility/total_variance + SviInterpolation + Svi{,Interpolated}SmileSection). Extends W2-A `XabrModelKind` with `NOARB_SABR`. |
| W6-B experimental vol surfaces | +28 | BlackVolSurface/BlackAtmVolCurve/EquityFXVolSurface/InterestRateVolSurface/VolatilityCube abstracts + ExtendedBlackVariance{Curve,Surface} + AbcdAtmVolCurve + SabrVolSurface + SABRVolTermStructure |
| W6-C experimental math foundations | +61 | Gaussian/T copula policies + 4 copula RNGs (Clayton/Frank/FGM/PolarStudentT) + ConvolvedStudentT + GaussianNonCentralChiSquaredPolynomial + moore_penrose_inverse + LaplaceInterpolation + Multidim{Integral,GaussianQuadrature} + Piecewise{Function,Integral} + generic LatentModel |
| W6-D heuristic optimizers | +50 | ParticleSwarmOptimization + FireflyAlgorithm + HybridSimulatedAnnealing(+functors) + ZigguratRng + LevyFlightDistribution + IsotropicRandomWalk |

**Notable divergences:**
- **C++ `ExtendedBlackVarianceSurface` is broken in v1.42.1** — out-of-bounds read in `setVariances()` + reference bound to a moved temporary → aborts on construction. W6-B ported the documented-correct variance grid (`variance[i][j] = t[j]·vol[i][j-1]²` + Bilinear); the probe computes references inline with the corrected formula rather than driving the broken class. (A3-class finding; resolved by porting intended behavior + documenting the upstream bug.)
- **MoorePenrose** → `numpy.linalg.pinv` (wrapped-delegation; numpy's SVD pinv is ecosystem-superior to QuantLib's hand-rolled SVD). Validated A·A⁺·A = A.
- **ZigguratRng** reproduces the C++ MersenneTwister-backed stream — 49/50 draws (seed 42) bit-exact; the 1 tail draw routing through the Acklam inverse-normal differs by 1 ULP (~4.4e-16, documented).
- scipy-TRF vs C++ projected-LM optimizer divergence on Abcd/SABR fits (L10-C precedent); stochastic optimizers (PSO/Firefly/HSA) use fixed seeds + region-convergence contracts.

**Merge reconciliations:** add/add docstring-union on `experimental/volatility/__init__.py` (W6-A namespace-package had none; W6-B's governs + enriched with W6-A families) and `experimental/math/__init__.py` (W6-C+W6-D docstring union); CMakeLists stacking ×4; `docs/carve-outs.md` auto-merged (W6-C additions).

**W6 follow-ups (carve-outs):** PSO ClubsTopology + AdaptiveInertia + LevyFlightInertia variants; LatentModel FactorSampler random-sample specializations; GaussNonCentralChiSquared mpmath-multiprecision moment recurrence (double-precision shipped).

## W7 — experimental processes + commodities + inflation + variance-gamma

Mini-pilot topology: W7-B (commodity foundations) merged first to unblock W7-C (commodity instruments); W7-A + W7-D ran independently in parallel. +152 tests / 3468 → 3620. Tag `pquantlib-phase11-w7-complete` @ `6ae1cfd`. The ExtendedOU/ExtOUWithJumps/Kluge processes were already on main (W5-A) — W7-A ported only the 3 remaining experimental processes.

| Cluster | Tests | Scope |
|---|---|---|
| W7-A processes + variance-gamma | +33 | ExtendedBlackScholesMertonProcess + GemanRoncoroniProcess + VegaStressedBlackScholesProcess + VarianceGammaProcess/Model + AnalyticVarianceGammaEngine (Madan-Carr-Chang) + FFTEngine base + FFTVanillaEngine + FFTVarianceGammaEngine (Carr-Madan, numpy.fft) |
| W7-B commodity foundations (mini-pilot) | +62 | UnitOfMeasure + 7 petroleum UOM + UnitOfMeasureConversion(+Manager singleton) + CommodityType (flyweight) + Quantity (UOM-converting arithmetic) + CommoditySettings + CommodityUnitCost + ExchangeContract + PaymentTerm + DateInterval + PricingPeriod + Commodity base + CommodityPricingHelper + Money |
| W7-C commodity instruments | +30 | CommodityIndex + CommodityCurve + CommodityCashFlow + EnergyCommodity + EnergyFuture + EnergySwap + EnergyVanillaSwap + EnergyBasisSwap (canonical energy-swap NPV 1999.123 cross-validated) |
| W7-D experimental inflation | +27 | Polynomial2DSpline + GenericCPI/YYGenericCPI + Region.Generic + CPI/YoY cap-floor term price surface abstracts + Interpolated{CPI,YoY}CapFloorTermPriceSurface + InterpolatingCPICapFloorEngine + YoYOptionletHelper + YoYOptionletStripper + InterpolatedYoYOptionletStripper + KInterpolatedYoYOptionletVolatilitySurface + PiecewiseYoYOptionletVolatilityCurve + YoYInflationOptionletVolatilityStructure2 + make_yoy_inflation_cap_floor |

**Notable divergences:**
- VarianceGamma analytic engine delegates the split Gauss-Kronrod/Lobatto quadrature to `scipy.integrate.quad` (reproduces canonical 955.16/687.20/453.47); FFT engines use `numpy.fft.fft` (verified identical unnormalized-DFT convention).
- Two more C++ v1.42.1 quirks: `CommodityType` ctor arg-order disagreement between header and impl (replicated faithfully, registry keyed on code); `PricingPeriod` is 4-arg (no unit-cost), not 5 as initially specced — source-of-truth followed.
- `Polynomial2DSpline` ported the parabolic-y/cubic-x algorithm directly (scipy `RectBivariateSpline` doesn't match QuantLib's basis).
- Flyweight/singleton registries → Python module-level dicts + `Singleton` metaclass.

**W7 follow-ups (carve-outs):** cross-currency FX (`ExchangeRateManager` + `Money` cross-currency arithmetic + commodity `calculate_fx_conversion_factor`); YoY surface cubic-extrapolation above max strike (scipy BicubicSpline clamps).

## W8 — long-tail experimental (15 subdirs)

4-parallel-no-pilot. +143 tests / 3620 → 3763. Tag `pquantlib-phase11-w8-complete` @ `8573dd9`.

| Cluster | Tests | Scope |
|---|---|---|
| W8-A coupons + swaptions + basismodels | +31 | SwapSpreadIndex + CmsSpreadCoupon + ProxyIbor + BlackIborQuantoCouponPricer + IrregularSwap/Swaption + HaganIrregularSwaptionEngine + SwaptionCashFlows + TenorSwaptionVTS + TenorOptionletVTS |
| W8-B callable + cat bonds + risk | +23 | CallableFixedRate/ZeroCouponBond + CallableBondVolatilityStructure(+Constant) + DiscretizedCallableFixedRateBond + Black/TreeCallableFixedRateBondEngine + CatRisk/EventSet/BetaRisk + NotionalRisk + CatBond/FloatingCatBond + MonteCarloCatBondEngine + CreditRiskPlus + SensitivityAnalysis |
| W8-C mcbasket + Heston Asian/fwd + FX VannaVolga | +32 | DeltaVolQuote + BlackDeltaCalculator + **VannaVolga single/double barrier engines (closes W4-C)** + AnalyticCont/DiscreteGeometricAvgPriceAsianHestonEngine + AnalyticHestonForwardEuropeanEngine + ExtendedBinomialTree family + PathPayoff/AdaptedPathPayoff/PathMultiAssetOption + MCPathBasketEngine + ForwardVanillaOption |
| W8-D CLV + generalized short-rate + xccy helpers | +57 | NormalCLVModel + SquareRootCLVModel + GeneralizedHullWhite + GeneralizedOrnsteinUhlenbeckProcess + IborIbor/OvernightIborBasisSwapRateHelper + ConstNotional/MtM CrossCurrencyBasisSwapRateHelper + foundations (LagrangeInterpolation, GaussianQuadrature, SquareRootProcess, GBSMRNDCalculator, LinearFlatInterpolation) |

**Upstream-removed (carved out — nothing to port at v1.42.1):** `ArithmeticAverageOIS` + `ArithmeticOISRateHelper` + `MakeArithmeticAverageOIS` (emptied v1.41); `CreditRiskPlus` + `SensitivityAnalysis` (emptied v1.36 — W8-B ported full functionality from recovered pre-removal history `6f379f4e9~1` as bonus value, documented inline).

**Surfaced core gap (→ W12 audit target):** `CmsCoupon` / `CmsCouponPricer` / `CappedFlooredCoupon` / `DigitalCoupon` (core `ql/cashflows/`) were never ported in L2-D. This blocks 4 experimental CMS-spread classes (LognormalCmsSpreadPricer, CappedFlooredCmsSpreadCoupon, DigitalCmsSpreadCoupon, StrippedCappedFlooredCoupon). W12's coverage audit will catch these core headers and a gap-fill should port them (which then unblocks the experimental coupons).

**W8 follow-ups (carve-outs):** American multi-asset LS engines (`LongstaffSchwartzMultiPathPricer` + `MCLongstaffSchwartzPathEngine` + `MCAmericanPathEngine` — states/exercises plumbed, ~3hr); GeneralizedHullWhite tree-fitting + non-linear-mapping path (`dynamics()` raises, matching C++ QL_FAIL); W8-B `cluster_w8b/probe.cpp` lost (left untracked in the shared worktree by the subagent, cleaned pre-merge) — `references/cluster/w8b.json` is committed and tests pass; reconstruct the probe in W12 if reference regeneration is needed.

**Notable divergences:** Heston Asian/forward CF engines match C++ to machine precision (numpy.leggauss(128) == C++ GaussLegendreIntegration(128)); MtM cross-currency swap rate helper bit-exact vs C++; W8-C caught + fixed a real lazy-cache bug (FD `recalculate()` → `instrument.update()`); Tree callable-bond engine ~1e-8 (custom tolerance, TreeSwaptionEngine precedent).

## W9 — marketmodels CORE (LIBOR Market Model foundations)

Pilot (W9-A core spine) + 2-parallel (W9-B/C). +82 tests / 3763 → 3845. Tag `pquantlib-phase11-w9-complete` @ `4d09234`. First of the 3 marketmodels waves; corrected dependency order (core → models+evolvers → products+callability — original plan was inverted).

| Cluster | Tests | Scope |
|---|---|---|
| W9-A pilot core spine | +25 | EvolutionDescription + CurveState abstract + LMM/CMSwap/CoterminalSwap curve states + MarketModel/MarketModelEvolver/MarketModelMultiProduct/PathwiseMultiProduct/BrownianGenerator abstracts + MarketModelDiscounter/PathwiseDiscounter + ForwardForward/SwapForward mappings + PiecewiseConstantCorrelation + utilities |
| W9-B correlations + drift + historical | +38 | exponential_forward_correlation + ExponentialForwardCorrelation + TimeHomogeneousForwardCorrelation + CotSwapFromFwdCorrelation + LMM/LMMNormal/SMM/CMSMM drift calculators + historical rates analysis + **SequenceStatistics** (L1-B enabler) |
| W9-C Brownian generators + accounting | +19 | MTBrownianGenerator + SobolBrownianGenerator (+Ordering) + factories + AccountingEngine (BGM MC pricing loop) + PathwiseAccountingEngine (Delta) + ProxyGreekEngine + ConstrainedEvolver |

**Divergences:** MT uniform stream bit-exact, Gaussian variates TIGHT (L1 InverseCumulativeNormal Halley-refinement gap ~1e-16); Sobol scipy-Joe-Kuo vs C++-Jaeckel diverge beyond 2 dims → only stream-independent surfaces (ordering schema + Brownian-bridge algebra) cross-validated; W9-B caught + fixed a `np.diagonal`-view aliasing bug in SequenceStatistics.correlation().

**W9 follow-ups (carve-outs):** PathwiseVegasAccountingEngine + Burley2020SobolBrownianGenerator (need W10 evolver `browniansThisStep` + RatePseudoRootJacobian); EvolutionDescription.effective_stop_time (commented-out in v1.42.1 source); marketmodeldifferences free functions (need W10 concrete MarketModel).

## W10 — marketmodels models + evolvers

Pilot (W10-A vol models) + 2-parallel (W10-B evolvers, W10-C calibration). +64 tests / 3845 → 3909. Tag `pquantlib-phase11-w10-complete` @ `1d559de`.

| Cluster | Tests | Scope |
|---|---|---|
| W10-A pilot vol models | +33 | FlatVol + AbcdVol + PiecewiseConstant{,Abcd}Variance + AbcdFunction + pseudo_sqrt (rank-reduced spectral) + PseudoRootFacade + FwdToCotSwap/CotSwapToFwd/FwdPeriod adapters + VolatilityInterpolationSpecifier{,abcd} |
| W10-B evolvers | +16 | LogNormalFwdRate{Pc,Euler,EulerConstrained,Ipc,Balland,IBalland} + NormalFwdRatePc + LogNormalCotSwapRatePc + LogNormalCmSwapRatePc + SVDDFwdRatePc + MarketModelVolProcess + SquareRootAndersen |
| W10-C caplet-coterminal calibration | +15 | AlphaForm + AlphaFormInverseLinear/LinearHyperbolic + AlphaFinder + CTSMMCapletCalibration base + CTSMMCapletOriginal/MaxHomogeneity/AlphaForm calibration + periodic + Quadratic + SphereCylinderOptimizer + BasisIncompleteOrdered |

**Key divergence (pseudo-root eigenvector sign — load-bearing):** the spectral pseudo-root `A` (`numpy.linalg.eigh`) is unique only up to per-eigenvector sign/rotation, so `A@Aᵀ` (covariance) matched C++ but the sign-sensitive diffusion `A@Z` did not. W10-B's `align(pseudo_sqrt)` commit pins each eigenvector's first component non-negative — replicating C++ `SymmetricSchurDecomposition`'s convention — making `A` match C++ for distinct-eigenvalue covariances; evolvers cross-validated TIGHT against the bit-identical MT Gaussian stream (Sobol diverges >2 factors). Another C++ quirk replicated verbatim: FwdPeriodAdapter's never-reset cumulative-sum "average". The `PseudoRootFacade.from_calibrator` ↔ `CTSMMCapletCalibration` wiring (W10-A↔W10-C) is confirmed end-to-end; max-homogeneity caplet vols reprice within 1 bp.

**W10 follow-ups (carve-outs):** FlatVolFactory (curve-wiring convenience); rank_reduced_sqrt Higham salvaging arm (no consumer); OrthogonalProjections (matrices-test companion).

**Notable divergences:**
- ZabrInterpolation: scipy TRF finds strictly lower RMS (5.2e-5) than C++ projected-LM (5.8e-5) on the γ-fixed=1 slice. Fourth instance of "Python scipy more accurate" pattern (cf. L9-B `conventional_spread`, L10-C `AbcdInterpolation`, this).
- BootstrapError reframed as a typed Python exception (C++ has it as a deprecated callable functor in 1.40); subclasses `LibraryException`.
- CmsMarket reprice/weighted-errors raise `LibraryException("CmsCouponPricer not ported")` — depends on `CmsCouponPricer` family (deferred to long-tail).
- KahaleSmileSection.core_smile Halton multi-start (deeper repair iteration) deferred.

**Merge:** both W2 branches merged cleanly with no CMakeLists conflicts (probe entries placed at adjacent line ranges, git auto-merge succeeded). W2-A modified existing `sabr_swaption_volatility_cube.py` to subclass new XabrSwaptionVolatilityCube; all 12 original L9-C SABR-cube tests still pass.

**Notable divergences (per cluster summaries):**
- MarkovFunctional: numpy `hermgauss` weights already fold `exp(-x²)` whereas C++ `GaussHermiteIntegration` returns bare weights — divides by √π directly (fixed numeraire 1.5x off → 0.1% off C++ truth).
- Gaussian1d engines: C++ uses Lagrange-BC cubic; PQuantLib uses natural-BC (scipy.CubicSpline). Empirical deltas: cap NPV rel ~5e-7, FloatFloat swaption rel ~0.24%, NonStandard swaption rel ~3e-5.
- Bates: `BatesDoubleExp(lambda→0)` collapses to Heston (TIGHT); `BatesDetJump(thetaLambda==lambda)` collapses to base Bates (TIGHT).
- Heston SLV FDM scaffold returns unit leverage; full Fokker-Planck FDM solver scheduled for W5-C.

**Merge reconciliations:**
- 3 successive CMakeLists.txt conflicts as each branch appended its probe entry. Standard resolution: keep both sides, stacked W1-A → W1-B → W1-C → W1-D entries.
- `docs/carve-outs.md` modified by W1-D (clean merge — appended Specialty-Heston-variants status).
- 3 base-class vol files modified by W1-A (clean merge — backwards-compatible signature widenings).
- `gaussian1d_model.py` modified by W1-B (clean merge — added staticmethods).
