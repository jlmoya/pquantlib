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
| W5 | `pquantlib-phase11-w5-complete` @ `6e0f8d2` | +127 | **3299** | 2026-05-29 |

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
