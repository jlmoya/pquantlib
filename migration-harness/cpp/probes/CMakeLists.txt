cmake_minimum_required(VERSION 3.20)
project(pquantlib_probes CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED QUANTLIB_SRC_DIR)
    message(FATAL_ERROR "QUANTLIB_SRC_DIR must be passed via -D")
endif()
if(NOT DEFINED QUANTLIB_BUILD_DIR)
    message(FATAL_ERROR "QUANTLIB_BUILD_DIR must be passed via -D")
endif()

# QuantLib transitively #include's <boost/config.hpp> via ql/qldefines.hpp,
# so probe TUs need Boost on the include path even though we don't link
# against any Boost libraries directly.
find_package(Boost REQUIRED)

include_directories(${QUANTLIB_SRC_DIR} ${Boost_INCLUDE_DIRS})
link_directories(${QUANTLIB_BUILD_DIR}/ql)

# Find dynamic library (handles .dylib on macOS, .so on Linux)
find_library(QUANTLIB_LIBRARY
    NAMES QuantLib libQuantLib
    PATHS ${QUANTLIB_BUILD_DIR}/ql
    NO_DEFAULT_PATH)
if(NOT QUANTLIB_LIBRARY)
    message(FATAL_ERROR "Could not find QuantLib library in ${QUANTLIB_BUILD_DIR}/ql")
endif()

# Add one add_executable + target_link_libraries per probe.
# Convention: probe at probes/<topic>/<class>_probe.cpp → executable name <topic>_<class>_probe
# The harness/generate-references.sh script turns the executable name into the
# JSON output path by translating underscores to slashes:
#   harness_sentinel_probe → references/harness/sentinel.json

# === harness self-check ===
add_executable(harness_sentinel_probe harness/sentinel_probe.cpp)
target_link_libraries(harness_sentinel_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / enums ===
add_executable(time_enums_probe time/enums_probe.cpp)
target_link_libraries(time_enums_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / period ===
add_executable(time_period_probe time/period_probe.cpp)
target_link_libraries(time_period_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / date ===
add_executable(time_date_probe time/date_probe.cpp)
target_link_libraries(time_date_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / parsers ===
add_executable(time_parsers_probe time/parsers_probe.cpp)
target_link_libraries(time_parsers_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / calendar ===
add_executable(time_calendar_probe time/calendar_probe.cpp)
target_link_libraries(time_calendar_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / imm ===
add_executable(time_imm_probe time/imm_probe.cpp)
target_link_libraries(time_imm_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / schedule ===
add_executable(time_schedule_probe time/schedule_probe.cpp)
target_link_libraries(time_schedule_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / asx + ecb ===
add_executable(time_asx_ecb_probe time/asx_ecb_probe.cpp)
target_link_libraries(time_asx_ecb_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / timegrid + timeseries ===
add_executable(time_timegrid_probe time/timegrid_probe.cpp)
target_link_libraries(time_timegrid_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / one ===
add_executable(daycounters_one_day_counter_probe daycounters/one_day_counter_probe.cpp)
target_link_libraries(daycounters_one_day_counter_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / actual family (360 / 364 / 365 / 36525 / 366) ===
add_executable(daycounters_actual_family_probe daycounters/actual_family_probe.cpp)
target_link_libraries(daycounters_actual_family_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / thirty family (Thirty360 9 conventions + Thirty365) ===
add_executable(daycounters_thirty_family_probe daycounters/thirty_family_probe.cpp)
target_link_libraries(daycounters_thirty_family_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / simple + business252 ===
add_executable(daycounters_simple_business_probe daycounters/simple_business_probe.cpp)
target_link_libraries(daycounters_simple_business_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / actualactual ===
add_executable(daycounters_actualactual_probe daycounters/actualactual_probe.cpp)
target_link_libraries(daycounters_actualactual_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / calendars (all 41 sovereign/exchange, default-market) ===
add_executable(time_calendars_all_probe time/calendars_all_probe.cpp)
target_link_libraries(time_calendars_all_probe PRIVATE ${QUANTLIB_LIBRARY})

# === math / first batch (rounding, factorial, errorfunction, beta, ...) ===
add_executable(math_first_batch_probe math/first_batch_probe.cpp)
target_link_libraries(math_first_batch_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-B cluster: copulas + simple distributions + statistics + currencies ===
add_executable(cluster_b_probe cluster_b/probe.cpp)
target_link_libraries(cluster_b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-C cluster: Solver1D family + simple integrals ===
add_executable(cluster_c_probe cluster_c/probe.cpp)
target_link_libraries(cluster_c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-D cluster: deterministic RNGs ===
add_executable(cluster_d_probe cluster_d/probe.cpp)
target_link_libraries(cluster_d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-E cluster: simple interpolations + matrix + math root ===
add_executable(cluster_e_probe cluster_e/probe.cpp)
target_link_libraries(cluster_e_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L2-A: foundations (quotes — SimpleQuote / DerivedQuote / CompositeQuote) ===
add_executable(quotes_foundations_probe quotes/foundations_probe.cpp)
target_link_libraries(quotes_foundations_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L2-B cluster: yield term structures (FlatForward + InterpolatedZero/
#     Forward/Discount + 3 Spreaded + Implied) ===
add_executable(cluster_l2b_probe cluster_l2b/probe.cpp)
target_link_libraries(cluster_l2b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L2-C cluster: indexes (Euribor/USDLibor/Sofr/...) + rate helpers ===
add_executable(cluster_l2c_probe cluster_l2c/probe.cpp)
target_link_libraries(cluster_l2c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L2-D cluster: cashflows (InterestRate / coupons / legs / aggregator) ===
add_executable(cluster_l2d_probe cluster_l2d/probe.cpp)
target_link_libraries(cluster_l2d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L2-E cluster: volatility term structures (Black + Local, equity/FX) ===
add_executable(cluster_l2e_probe cluster_l2e/probe.cpp)
target_link_libraries(cluster_l2e_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L3-A pilot: foundations (Payoffs + BlackFormula family) ===
add_executable(l3a_foundations_probe l3a/foundations_probe.cpp)
target_link_libraries(l3a_foundations_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L3-B cluster: bond instruments + DiscountingBondEngine + BondForward ===
add_executable(cluster_l3b_probe cluster_l3b/probe.cpp)
target_link_libraries(cluster_l3b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L3-C cluster: swaps (Vanilla/OIS/ZeroCoupon) +
#     DiscountingSwapEngine + SwapRateHelper / OISRateHelper roundtrips ===
add_executable(cluster_l3c_probe cluster_l3c/probe.cpp)
target_link_libraries(cluster_l3c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L3-D cluster: equity options + processes (GBSM family +
#     EulerDiscretization + VanillaOption/EuropeanOption +
#     AnalyticEuropeanEngine + BinomialVanillaEngine) ===
add_executable(cluster_l3d_probe cluster_l3d/probe.cpp)
target_link_libraries(cluster_l3d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L3-E cluster: forwards + FRAs (Forward + FxForward +
#     ForwardRateAgreement + DiscountingFwdEngine + FraRateHelper indexed) ===
add_executable(cluster_l3e_probe cluster_l3e/probe.cpp)
target_link_libraries(cluster_l3e_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L4-A pilot: foundations (LevenbergMarquardt + Simplex closing
#     Phase 1 optimizer carry-overs, Parameter family round-trips, and
#     NoConstraint/PositiveConstraint transforms) ===
add_executable(l4a_foundations_probe l4a/foundations_probe.cpp)
target_link_libraries(l4a_foundations_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L4-B cluster: 1-factor short-rate models (Vasicek, HullWhite,
#     CoxIngersollRoss, ExtendedCoxIngersollRoss) over OrnsteinUhlenbeck
#     + CoxIngersollRoss processes, plus HullWhite::convexityBias. ===
add_executable(cluster_l4b_probe cluster_l4b/probe.cpp)
target_link_libraries(cluster_l4b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L4-C: HestonProcess + HestonModel + AnalyticHestonEngine
#     + BatesProcess + BatesModel + HestonModelHelper ===
add_executable(cluster_l4c_probe cluster_l4c/probe.cpp)
target_link_libraries(cluster_l4c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L4-D cluster: G2/G2Forward/HullWhiteForward/OU/CIR processes +
#     G2 two-additive-factor Gaussian short-rate model + G2.swaption ===
add_executable(cluster_l4d_probe cluster_l4d/probe.cpp)
target_link_libraries(cluster_l4d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L4-E cluster: swaption + capfloor instruments + helpers +
#     analytic engines (Black/Bachelier/Jamshidian/G2 swaption;
#     Black/Bachelier/AnalyticCapFloor capfloor) ===
add_executable(cluster_l4e_probe cluster_l4e/probe.cpp)
target_link_libraries(cluster_l4e_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L5-A pilot: foundations (Phase 1 carry-overs SobolRsg /
#     Burley2020SobolRsg / GammaFunction / AkimaCubicInterpolation,
#     plus the tree/lattice + DiscretizedAsset base hierarchies and
#     methods/protocols.py used by lattices and MC/FD engines.) ===
add_executable(l5a_foundations_probe l5a/foundations_probe.cpp)
target_link_libraries(l5a_foundations_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L5-B cluster: concrete trees (Binomial / Trinomial),
#     TreeLattice1D + BlackScholesLattice, DiscretizedSwap /
#     DiscretizedSwaption / DiscretizedCapFloor, TreeSwaptionEngine +
#     TreeCapFloorEngine, BlackKarasinski model + ShortRateModel.tree() ===
add_executable(cluster_l5b_probe cluster_l5b/probe.cpp)
target_link_libraries(cluster_l5b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L5-C cluster: Monte Carlo framework (Path / MultiPath / Brownian
#     bridge / PathGenerator / McSimulation) + simple MC engines
#     (MCEuropeanEngine + MCDiscreteArithmeticAPEngine with geometric
#     control variate via AnalyticDiscreteGeometricAveragePriceAsianEngine). ===
add_executable(cluster_l5c_probe cluster_l5c/probe.cpp)
target_link_libraries(cluster_l5c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L5-D cluster: 1-D Finite-Difference Black-Scholes framework +
#     FdBlackScholesVanillaEngine (closes the Phase 3 carve-out for
#     VanillaOption::impliedVolatility on American options). ===
add_executable(cluster_l5d_probe cluster_l5d/probe.cpp)
target_link_libraries(cluster_l5d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L5-E cluster: exotic instruments + analytic engines
#     (asian/barrier/basket/lookback closed-form pricing). ===
add_executable(cluster_l5e_probe cluster_l5e/probe.cpp)
target_link_libraries(cluster_l5e_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L6-A cluster: LongstaffSchwartz American MC closure
#     (LsmBasisSystem + LongstaffSchwartzPathPricer + MCAmericanEngine). ===
add_executable(cluster_l6a_probe cluster_l6a/probe.cpp)
target_link_libraries(cluster_l6a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L6-B cluster: BatesEngine (analytic Bates jump-diffusion engine
#     via the AnalyticHestonEngine add_on_term hook). ===
add_executable(cluster_l6b_probe cluster_l6b/probe.cpp)
target_link_libraries(cluster_l6b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L6-C cluster: DoubleBarrierOption + AnalyticDoubleBarrierEngine
#     (Ikeda-Kunitomo 1992 closed-form series for KnockIn/KnockOut). ===
add_executable(cluster_l6c_probe cluster_l6c/probe.cpp)
target_link_libraries(cluster_l6c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L7-A cluster: inflation foundations (region indexes + seasonality
#     + inflation-period helper for the Phase 7 inflation cluster). ===
add_executable(l7a_foundations_probe l7a/foundations_probe.cpp)
target_link_libraries(l7a_foundations_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L7-B cluster: inflation curves + bootstrap helpers (interpolated
#     zero/yoy curves + piecewise bootstrap roundtrip from ZCIIS helpers). ===
add_executable(cluster_l7b_probe cluster_l7b/probe.cpp)
target_link_libraries(cluster_l7b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L7-C cluster: inflation cashflows + pricers (InflationCoupon,
#     ZeroInflationCashFlow, CPICashFlow, CPICoupon, YoYInflationCoupon,
#     CappedFloored* variants, InflationCouponPricer, CPICouponPricer,
#     YoYInflationCouponPricer + Black/Bachelier variants). ===
add_executable(cluster_l7c_probe cluster_l7c/probe.cpp)
target_link_libraries(cluster_l7c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L7-D cluster: inflation instruments + vol surfaces + YoY cap/floor
#     engines (ZCIIS / YYIIS / CPISwap / YoYInflationCapFloor + CPI/YoY
#     vol surface abstracts + Constant variants + Bachelier/Black/UnitDisplaced
#     YoY cap/floor engines). ===
add_executable(cluster_l7d_probe cluster_l7d/probe.cpp)
target_link_libraries(cluster_l7d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L8-A cluster: piecewise inflation curves + IterativeBootstrap
#     (closes L7-Bb piecewise carve-out + L2-B bootstrap carve-out;
#     PiecewiseZeroInflationCurve via L7-B's existing reference;
#     PiecewiseYoYInflationCurve roundtrip + bootstrap-loop invariants). ===
add_executable(cluster_l8a_probe cluster_l8a/probe.cpp)
target_link_libraries(cluster_l8a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L8-B cluster: credit cluster — default-probability term structures
#     (FlatHazardRate + InterpolatedSurvivalProbability/HazardRate/DefaultDensity
#     curves + abstracts) + Claim + CreditDefaultSwap + MidPoint/Integral
#     CDS engines. ===
add_executable(cluster_l8b_probe cluster_l8b/probe.cpp)
target_link_libraries(cluster_l8b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L8-C cluster: capfloor + optionlet + swaption vol surfaces
#     (CapFloorTermVolatilityStructure family + OptionletVolatilityStructure
#     family + OptionletStripper1 + SwaptionVolatilityStructure family
#     + SwaptionVolatilityMatrix). ===
add_executable(cluster_l8c_probe cluster_l8c/probe.cpp)
target_link_libraries(cluster_l8c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L9-A cluster: cubic + bicubic interpolators
#     (CubicNaturalSpline + MonotonicCubicNaturalSpline + BicubicSpline —
#     closes the L1-E carve-out for cubic-family interpolators). ===
add_executable(cluster_l9a_probe cluster_l9a/probe.cpp)
target_link_libraries(cluster_l9a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L9-B cluster: post-L8 ergonomics — IsdaCdsEngine + implied_hazard_rate
#     + conventional_spread + PiecewiseYieldCurve + yield traits +
#     PiecewiseDefaultCurve bootstrap wiring + MakeCDS factory. ===
add_executable(cluster_l9b_probe cluster_l9b/probe.cpp)
target_link_libraries(cluster_l9b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L9-C cluster: SABR formula + smile sections + swaption vol cube
#     (sabrVolatility + SabrSmileSection + InterpolatedSmileSection +
#     SpreadedSmileSection + SabrSwaptionVolatilityCube). ===
add_executable(cluster_l9c_probe cluster_l9c/probe.cpp)
target_link_libraries(cluster_l9c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L10-A cluster: vol surface tail (Atm / AtmAdjusted / Kahale /
#     SabrInterpolated smile sections + OptionletStripper2 +
#     SabrInterpolation Halton multi-start). ===
add_executable(cluster_l10a_probe cluster_l10a/probe.cpp)
target_link_libraries(cluster_l10a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L10-B cluster: Gaussian1d short-rate cluster — Gaussian1dModel
#     abstract + GSR concrete + Gaussian1dSwaptionVolatility (model +
#     GsrProcess via gsrprocesscore + swaption-vol back-out via Newton
#     inversion of the Gaussian1dSwaptionEngine NPV). ===
add_executable(cluster_l10b_probe cluster_l10b/probe.cpp)
target_link_libraries(cluster_l10b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L10-C cluster: interpolator tail + ZABR family
#     (HymanFilteredCubic + ChebyshevInterpolation + MultiCubicSpline +
#     AbcdInterpolation + zabr_volatility + ZabrSmileSection). ===
add_executable(cluster_l10c_probe cluster_l10c/probe.cpp)
target_link_libraries(cluster_l10c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W1-B cluster: Gaussian1d engines (Gaussian1dCapFloorEngine +
#     Gaussian1dFloatFloatSwaptionEngine +
#     Gaussian1dNonstandardSwaptionEngine). ===
add_executable(cluster_w1b_probe cluster_w1b/probe.cpp)
target_link_libraries(cluster_w1b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W1-C cluster: Bates variant models + analytic engines
#     (BatesDetJumpModel + BatesDoubleExpModel + BatesDoubleExpDetJumpModel +
#     AnalyticBatesDetJumpEngine + AnalyticBatesDoubleExpEngine +
#     AnalyticBatesDoubleExpDetJumpEngine). ===
add_executable(cluster_w1c_probe cluster_w1c/probe.cpp)
target_link_libraries(cluster_w1c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === Phase 11 W1-D cluster: Heston SLV + GjrGarch + time-dep Heston
#     (GJRGARCHProcess + GJRGARCHModel + AnalyticGJRGARCHEngine +
#     PiecewiseTimeDependentHestonModel + AnalyticPTDHestonEngine).
#     Batch C HestonSlvFdmModel + batch D HestonSlvMcModel land as
#     structural scaffolds; their full Fokker-Planck FDM / bucketed-MC
#     calibration is documented as a Phase-11 W5 follow-up. ===
add_executable(cluster_w1d_probe cluster_w1d/probe.cpp)
target_link_libraries(cluster_w1d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === Phase 11 W2-A cluster: ZABR fitter + smile composition +
#     XABR/ZABR swaption volatility cube generalization
#     (ZabrInterpolation + ZabrInterpolatedSmileSection +
#     XabrSwaptionVolatilityCube + ZabrSwaptionVolatilityCube). ===
add_executable(cluster_w2a_probe cluster_w2a/probe.cpp)
target_link_libraries(cluster_w2a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W1-A cluster: MarkovFunctional model + adapter process
#     (Gaussian1dGsrProcess wrapper + MarkovFunctional swaption-strip
#     calibration + numeraire tabulation + zerobondImpl/numeraireImpl). ===
add_executable(cluster_w1a_probe cluster_w1a/probe.cpp)
target_link_libraries(cluster_w1a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W2-B cluster: ConvexMonotoneInterpolation + AbcdCalibration +
#     KahaleSmileSection.core_smile deep-iter (Phase 11 W2-B). ===
add_executable(cluster_w2b_probe cluster_w2b/probe.cpp)
target_link_libraries(cluster_w2b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W3-A pilot: experimental/credit foundation types — DefaultType +
#     AtomicDefault + Restructuring + DefaultEvent + DefaultProbKey +
#     Issuer + Pool + RecoveryRateQuote + ConstantRecoveryModel +
#     LossDist family + Distribution + Loss. Foundation for W3-B/C/D. ===
add_executable(cluster_w3a_probe cluster_w3a/probe.cpp)
target_link_libraries(cluster_w3a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W3-B cluster: copulas (OneFactorGaussian/StudentCopula) + correlation
#     (Correlation/Base/Compound structures) + latent models (Constant/
#     Spot/RandomDefault/RandomLoss) + DefaultLossModel + 5 concrete loss
#     models (GaussianLHP / Binomial / Recursive / Saddlepoint). ===
add_executable(cluster_w3b_probe cluster_w3b/probe.cpp)
target_link_libraries(cluster_w3b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W3-C cluster: hazard-rate specialty curves + CDO + tranche + Basket
#     (FactorSpreadedHazardRateCurve + SpreadedHazardRateCurve + Basket +
#     SyntheticCDO + IntegralCDOEngine + MidPointCDOEngine driven by
#     GaussianLHPLossModel). ===
add_executable(cluster_w3c_probe cluster_w3c/probe.cpp)
target_link_libraries(cluster_w3c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W3-D cluster: NthToDefault + IntegralNTDEngine + CdsOption +
#     BlackCDSOptionEngine + RandomDefaultModel + RiskyAssetSwap +
#     RiskyAssetSwapOption. NTD-basket + CDS-option + asset-swap-option
#     family. Depends on W3-A pilot (Pool/Issuer/DefaultProbKey),
#     and probes Basket/OneFactorGaussianCopula directly from C++. ===
add_executable(cluster_w3d_probe cluster_w3d/probe.cpp)
target_link_libraries(cluster_w3d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W4-A cluster: multi-asset exotic options — HimalayaOption /
#     EverestOption / PagodaOption with MC engines (PseudoRandom,
#     fixed seed + samples), plus TwoAssetBarrierOption + TwoAsset
#     CorrelationOption with their analytic engines (Heynen-Kat
#     barrier + Zhang correlation). ===
add_executable(cluster_w4a_probe cluster_w4a/probe.cpp)
target_link_libraries(cluster_w4a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W4-B cluster: compound + chooser + extensible options +
#     Kirk spread engine + AnalyticPdfHeston + continuous Asian
#     Levy/Vecer engines. Closed-form exotic options under BSM and
#     PDF-/PDE-based engines under Heston and Asian PDE. ===
add_executable(cluster_w4b_probe cluster_w4b/probe.cpp)
target_link_libraries(cluster_w4b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W4-C cluster: barrier specialties + variance options.
#     PartialTimeBarrierOption + AnalyticPartialTimeBarrierOptionEngine
#     (Heynen-Kat 1994); BinomialDoubleBarrierEngine[CRR] cross-checked
#     against AnalyticDoubleBarrierEngine (Ikeda-Kunitomo); SoftBarrier
#     option + AnalyticSoftBarrierEngine (Hart-Ross 1994); VarianceOption
#     + IntegralHestonVarianceOptionEngine (Bailey-Swarztrauber
#     oscillatory integral). ===
add_executable(cluster_w4c_probe cluster_w4c/probe.cpp)
target_link_libraries(cluster_w4c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W5-A cluster: ExtOU + Kluge process FD infrastructure.
#     Glued1dMesher; FdmExtendedOrnsteinUhlenbeckOp; FdmExtOUJumpOp;
#     FdmKlugeExtOUOp; FdmExtOUJumpSolver; FdmKlugeExtOUSolver;
#     FdmSimple2dExtOUSolver; FdmSimple3dExtOUJumpSolver;
#     FdmExpExtOUInnerValueCalculator; FdmSpreadPayoffInnerValue. ===
add_executable(cluster_w5a_probe cluster_w5a/probe.cpp)
target_link_libraries(cluster_w5a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W5-B cluster: VPP option + storage/swing FD engine scaffold.
#     VanillaVPPOption + FdmVPPStepCondition family +
#     FdmVPPStepConditionFactory + DynProgVPPIntrinsicValueEngine
#     (closed-form DP NPV against testVPPIntrinsicValue references).
#     FD storage/swing/VPP engines under Kluge+ExtOU are scaffolded
#     structurally (process side stubbed via Protocol) — full pricing
#     is deferred until the ExtOU/Kluge FD operator cluster lands. ===
add_executable(cluster_w5b_probe cluster_w5b/probe.cpp)
target_link_libraries(cluster_w5b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W5-C cluster: experimental FD operators + OU vanilla engine +
#     inner-value calculators. FdmZabrOp, FdmDupire1dOp,
#     FdmOrnsteinUhlenbeckOp, FdOrnsteinUhlenbeckVanillaEngine,
#     FdmExtOUJumpModelInnerValue. ===
add_executable(cluster_w5c_probe cluster_w5c/probe.cpp)
target_link_libraries(cluster_w5c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W6-A cluster: NoArbSABR (Doust 2012) + SVI (Gatheral) smile
#     families. SviSmileSection + sviTotalVariance; NoArbSabrModel
#     (absorption-probability no-arb SABR) + D0Interpolator;
#     NoArbSabrSmileSection. The probe also serializes the 1,209,600-
#     entry absorption-probability table (QuantLib::detail::sabrabsprob)
#     to a uint32 binary blob for the Python data asset. ===
add_executable(cluster_w6a_probe cluster_w6a/probe.cpp)
target_link_libraries(cluster_w6a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W6-B cluster: experimental volatility surfaces.
#     ExtendedBlackVarianceCurve / ExtendedBlackVarianceSurface
#     (Quote-backed Black variance term structures); AbcdAtmVolCurve
#     (Abcd-parametric ATM IR vol curve, k-adjusted fit reproduces
#     input ATM vols); SabrVolSurface (interpolated SABR vol surface
#     from market vol spreads over an InterestRateVolSurface base).
#     Abstract bases (BlackVolSurface, BlackAtmVolCurve,
#     EquityFXVolSurface, InterestRateVolSurface, VolatilityCube) are
#     smoke-tested via minimal Python concretes — no C++ probe.
#     NOTE: ExtendedBlackVarianceSurface aborts on construction in
#     v1.42.1 (OOB in setVariances()); its reference values are computed
#     inline with the documented-correct variance grid + Bilinear. ===
add_executable(cluster_w6b_probe cluster_w6b/probe.cpp)
target_link_libraries(cluster_w6b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W6-C cluster: experimental math foundations. ConvolvedStudentT,
#     GaussNonCentralChiSquaredPolynomial, moorePenroseInverse,
#     GaussianQuadMultidimIntegrator, MultidimIntegral, piecewise
#     function/integral, Gaussian/T copula policies, LatentModel,
#     laplaceInterpolation, PolarStudentTRng. ===
add_executable(cluster_w6c_probe cluster_w6c/probe.cpp)
target_link_libraries(cluster_w6c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W6-D cluster: experimental heuristic optimizers + RNG.
#     ZigguratRng (EXACT-tier first-N draw stream + LOOSE statistical
#     moments); LevyFlightDistribution (pdf + inverse-transform variate);
#     IsotropicRandomWalk (deterministic MT-driven spherical step);
#     ParticleSwarmOptimization / FireflyAlgorithm /
#     HybridSimulatedAnnealing (analytic test-function global optima,
#     LOOSE region-convergence contract). ===
add_executable(cluster_w6d_probe cluster_w6d/probe.cpp)
target_link_libraries(cluster_w6d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W7-B cluster: commodity foundation types + units of measure.
#     UnitOfMeasure + petroleum UOM concretes + UnitOfMeasureConversion +
#     UnitOfMeasureConversionManager (singleton conversion registry) +
#     CommodityType (flyweight) + Quantity (typed commodity amount with
#     UOM-converting arithmetic) + PricingPeriod + CommodityPricingHelper.
#     Probes known petroleum conversion factors, Quantity arithmetic
#     (same-UOM + cross-UOM AutomatedConversion), and accessor round-trips. ===
add_executable(cluster_w7b_probe cluster_w7b/probe.cpp)
target_link_libraries(cluster_w7b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W7-A cluster: experimental processes + variance-gamma.
#     ExtendedBlackScholesMertonProcess (selectable Euler/Milstein/
#     PredictorCorrector evol-discretization; Euler evolve matches the
#     plain GBSM Euler evolve); GemanRoncoroniProcess (energy spot-price
#     process: drift/diffusion/stdDeviation + deterministic jump-driven
#     evolve via supplied Array du); VarianceGammaProcess accessors;
#     VarianceGammaEngine (Madan-Carr-Chang integral) European call/put
#     NPV; FFTVanillaEngine (Carr-Madan FFT) vs AnalyticEuropeanEngine;
#     FFTVarianceGammaEngine vs the analytic VG engine. ===
add_executable(cluster_w7a_probe cluster_w7a/probe.cpp)
target_link_libraries(cluster_w7a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === cluster_w7c: commodity index/curve/cashflow + energy instruments —
#     CommodityCurve (forward-flat price curve) + CommodityIndex (history +
#     forward curve) + CommodityCashFlow (Money amount) + EnergyFuture +
#     EnergyVanillaSwap. Probes interpolated forward prices, past fixings,
#     and the canonical energy future / vanilla-swap NPVs. ===
add_executable(cluster_w7c_probe cluster_w7c/probe.cpp)
target_link_libraries(cluster_w7c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W7-D cluster: experimental inflation — YoY optionlet stripping +
#     CPI/YoY cap-floor price surfaces. Polynomial2DSpline;
#     Interpolated{CPI,YoY}CapFloorTermPriceSurface;
#     InterpolatingCPICapFloorEngine; InterpolatedYoYOptionletStripper +
#     KInterpolatedYoYOptionletVolatilitySurface. ===
add_executable(cluster_w7d_probe cluster_w7d/probe.cpp)
target_link_libraries(cluster_w7d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W8-A cluster: CMS-spread index + irregular swaptions + basis-model
#     (tenor-rescaled) volatility term structures. SwapSpreadIndex
#     (gearing1*fix1 + gearing2*fix2); IrregularSwap NPV (step-down notional);
#     HaganIrregularSwaptionEngine NPV (Hagan linear-TSR super-replication);
#     TenorSwaptionVTS + TenorOptionletVTS volatility rescaling;
#     SwaptionCashFlows annuity/float/fixed weight decomposition.
#     NOTE: LognormalCmsSpreadPricer + CappedFlooredCmsSpreadCoupon +
#     DigitalCmsSpreadCoupon + StrippedCappedFlooredCoupon are deferred
#     (they need the not-ported CmsCoupon/CmsCouponPricer + CappedFlooredCoupon
#     + DigitalCoupon foundations), so they are not probed. ===
add_executable(cluster_w8a_probe cluster_w8a/probe.cpp)
target_link_libraries(cluster_w8a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W8-C cluster: FX delta-vol + VannaVolga + Heston Asian/forward.
#     BlackDeltaCalculator (strikeFromDelta round-trip + atmStrike for
#     Fwd/Spot/PaSpot conventions); VannaVolgaBarrierEngine (canonical
#     FX single-barrier values, adaptVanDelta=true); VannaVolga double-
#     barrier engine (KnockOut/KnockIn over AnalyticDoubleBarrierEngine);
#     AnalyticContinuous/DiscreteGeometricAveragePriceAsianHestonEngine
#     (Kim-Wee CF integration); AnalyticHestonForwardEuropeanEngine
#     (Kruse forward-start). ===
add_executable(cluster_w8c_probe cluster_w8c/probe.cpp)
target_link_libraries(cluster_w8c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W8-D cluster: CLV models + generalized short-rate + basis/xccy
#     rate helpers. GeneralizedOrnsteinUhlenbeckProcess; GeneralizedHullWhite;
#     NormalCLVModel; SquareRootCLVModel; GBSMRNDCalculator;
#     IborIborBasisSwapRateHelper / OvernightIborBasisSwapRateHelper;
#     ConstNotional/MtM CrossCurrencyBasisSwapRateHelper +
#     ConstNotionalCrossCurrencySwapRateHelper. ===
add_executable(cluster_w8d_probe cluster_w8d/probe.cpp)
target_link_libraries(cluster_w8d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W9-A cluster (PILOT): MarketModels (LIBOR Market Model / BGM) core —
#     abstract spine + curve-state geometry + swap<->forward mappings +
#     numeraire discounters + grid utilities. mergeTimes / isInSubset;
#     EvolutionDescription (default evolutionTimes + firstAliveRate);
#     LMMCurveState / CMSwapCurveState / CoterminalSwapCurveState discounting
#     algebra; SwapForwardMappings::coterminalSwapForwardJacobian;
#     MarketModelDiscounter.numeraireBonds. ===
add_executable(cluster_w9a_probe cluster_w9a/probe.cpp)
target_link_libraries(cluster_w9a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W9-B cluster: MarketModels correlations + drift calculators +
#     historical-analysis enabler. exponentialCorrelations (closed-form);
#     TimeHomogeneousForwardCorrelation::evolvedMatrices (lower-right-shifted
#     family); ExponentialForwardCorrelation (gamma=1 evolved + gamma!=1
#     midpoint); CotSwapFromFwdCorrelation (Z C Z^T corr with expired-rate
#     zeroing). LMM / LMMNormal / SMM / CMSMM drift calculators (Joshi 2003
#     reduced-factor drift; pure linear algebra). SequenceStatistics
#     covariance/correlation (multi-dim moment + outer-product accumulation),
#     the enabler for historicalRatesAnalysis / HistoricalForwardRatesAnalysis. ===
add_executable(cluster_w9b_probe cluster_w9b/probe.cpp)
target_link_libraries(cluster_w9b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W9-C cluster: MarketModels Brownian generators + accounting/greek
#     engines. MTBrownianGenerator (nextPath/nextStep over the bit-identical
#     MT uniform stream + inverse-cumulative normal); SobolBrownianGenerator
#     orderedIndices (Factors/Steps/Diagonal fill schema) + the transform()
#     Brownian-bridge test interface on a fixed variate matrix;
#     AccountingEngine (discounted cash-flow accumulation over a deterministic
#     stub evolver + stub MultiProduct, terminal numeraire); ProxyGreekEngine
#     original-evolver leg (singlePathValues). PathwiseAccountingEngine is not
#     probed (its ctor needs a concrete LogNormalFwdRateEuler from W10). ===
add_executable(cluster_w9c_probe cluster_w9c/probe.cpp)
target_link_libraries(cluster_w9c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W10-A cluster: MarketModels concrete volatility models. FlatVol
#     (flat-per-rate vol; per-step covariance via flatVolCovariance integral
#     weighted by a PiecewiseConstantCorrelation, then rankReducedSqrt spectral
#     pseudo-sqrt); AbcdVol (Rebonato abcd vol via AbcdFunction.covariance);
#     PiecewiseConstantAbcdVariance (abcd variance integral per reset rate);
#     PseudoRootFacade (wrap pseudo-roots as a MarketModel);
#     FwdToCotSwapAdapter / CotSwapToFwdAdapter (Z-matrix swap<->forward
#     pseudo-root re-expression + round-trip). The covariance B@B^T is probed
#     (TIGHT) rather than the raw pseudo-root, which is only unique up to an
#     orthogonal rotation / sign in the spectral decomposition. ===
add_executable(cluster_w10a_probe cluster_w10a/probe.cpp)
target_link_libraries(cluster_w10a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W10-B cluster: MarketModels (BGM) forward-/swap-rate evolvers.
#     LMM forward-rate evolvers (Pc / Euler / Ipc / Balland / iBalland),
#     NormalFwdRatePc, LogNormalCotSwapRatePc, LogNormalCmSwapRatePc,
#     SVDDFwdRatePc, and the SquareRootAndersen QE vol process. Every evolver is
#     driven by MTBrownianGeneratorFactory, whose Gaussian stream pquantlib
#     ports bit-identically, so the evolved rates cross-validate TIGHT (the
#     full RNG -> drift -> diffusion pipeline, not just the algebra). MT is used
#     rather than Sobol because Sobol's stream diverges from C++ beyond ~2
#     factors (W9-C finding). ===
add_executable(cluster_w10b_probe cluster_w10b/probe.cpp)
target_link_libraries(cluster_w10b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W10-C cluster: MarketModels caplet / coterminal-swaption CALIBRATION.
#     AlphaForm concretes (AlphaFormInverseLinear 1/(1+a*t);
#     AlphaFormLinearHyperbolic sqrt(1+a*t*(atan(a*t)-pi/2))); AlphaFinder
#     (Joshi 2007 alpha-form root-find — Bisection on quadratic.valueAtTurningPoint
#     for solve(); golden-section Minimize on homogeneity-failure for
#     solveWithMaxHomogeneity()); SphereCylinderOptimizer (the inner max-
#     homogeneity solver — sphere ∩ vertical-cylinder closest-point via Brent
#     golden-section minimize); CTSMMCapletMaxHomogeneityCalibration.calibrate
#     (the canonical marketmodel.cpp coterminal-swap-market-model caplet
#     calibration — singleRateClosestPointFinder over BasisIncompleteOrdered +
#     SphereCylinderOptimizer per rate, iterated to match caplet vols while
#     keeping perfect swaption fit); CTSMMCapletOriginalCalibration.calibrate
#     (Joshi-original joint caplet+swaption alpha-rescaled quadratic per-rate
#     fit). PseudoRootFacade.from_calibrator consumes the CTSMMCapletCalibration
#     base. ===
add_executable(cluster_w10c_probe cluster_w10c/probe.cpp)
target_link_libraries(cluster_w10c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W11-A cluster: MarketModels (BGM) MultiProduct framework + core products.
#     The LAST marketmodels wave. Composite framework (MultiProductMultiStep/
#     OneStep, MarketModelComposite + MultiProductComposite, SingleProductComposite,
#     CallSpecifiedMultiProduct, ExerciseAdapter, MarketModelCashRebate) + core
#     concrete products (MultiStepNothing/Forwards/Optionlets/Swap/CoterminalSwaps/
#     CoterminalSwaptions, OneStepForwards/Optionlets). The CANONICAL BGM
#     end-to-end test reproduces marketmodel.cpp testMultiStepForwardsAndOptionlets:
#     AccountingEngine(LogNormalFwdRatePc(FlatVol)) over a MultiProductComposite
#     (forwards + optionlets) reproduces the closed-form Black FRA + caplet prices
#     (emitted here EXACT; the MC comparison is the C++ 2.5-standard-error band).
#     Per-product nextTimeStep cash-flow generation on a flat-forward LMMCurveState
#     and the CashRebate one-shot are probed TIGHT. ===
add_executable(cluster_w11a_probe cluster_w11a/probe.cpp)
target_link_libraries(cluster_w11a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W11-B cluster: MarketModels (BGM) remaining products + pathwise
#     (Greeks-aware) products. Remaining multistep/onestep products
#     (MultiStepCoinitialSwaps/Ratchet/InverseFloater/Tarn/Swaption/
#     PeriodCapletSwaptions, OneStepCoinitialSwaps/CoterminalSwaps) +
#     pathwise products (MarketModelPathwiseMultiCaplet/DeflatedCaplet,
#     MarketModelPathwiseSwap, MarketModelPathwiseCoterminalSwaptionsDeflated
#     + its numerical-FD twin, MarketModelPathwiseInverseFloater,
#     MarketModelPathwiseCashRebate, MultiProductPathwiseWrapper).
#     Per-product nextTimeStep cash-flow generation on a flat-forward
#     LMMCurveState is probed TIGHT; the Tarn/Ratchet path-dependent
#     accumulation is probed over a deterministic step sequence; pathwise
#     products' amount[] (value + per-rate derivatives) are probed TIGHT, with
#     the analytic swaption deltas cross-checked against the numerical-FD twin. ===
add_executable(cluster_w11b_probe cluster_w11b/probe.cpp)
target_link_libraries(cluster_w11b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W11-C: MarketModels (BGM) callability — exercise values + basis systems
#     + swap-rate trigger. NothingExerciseValue / BermudanSwaptionExerciseValue
#     .value, SwapBasisSystem / SwapForwardBasisSystem .values, and
#     SwapRateTrigger.exercise logic on a flat-forward LMMCurveState are probed
#     TIGHT. The canonical Bermudan-swaption MC test is structurally
#     self-validating (no reference). ===
add_executable(cluster_w11c_probe cluster_w11c/probe.cpp)
target_link_libraries(cluster_w11c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W11-D cluster: MarketModels (BGM) pathwise-greeks Jacobians.
#     RatePseudoRootJacobian / RatePseudoRootJacobianAllElements (the analytic
#     dRate/dPseudoRoot one-step log-Euler Jacobian — GG paper page 95 "B"),
#     cross-checked against RatePseudoRootJacobianNumerical (finite-difference
#     bump of the same one-step evolution); SwaptionPseudoDerivative
#     (d swaption variance/impliedVol / d pseudoRoot on a coterminal swaption);
#     CapPseudoDerivative (d cap price/impliedVol / d pseudoRoot, cap implied
#     vol via a Brent solve over sum-of-caplet Black prices); VegaBumpCluster /
#     VegaBumpCollection (deterministic pseudo-root-element clustering schema —
#     doesIntersect/isCompatible/isFull/isNonOverlapping/isSensible);
#     VolatilityBumpInstrumentJacobian (instrument-level vega-bump Jacobian +
#     0.01v/<v,v> one-percent bumps) + OrthogonalizedBumpFinder (the
#     orthogonalised pseudo-root bump directions consumed by
#     PathwiseVegasAccountingEngine). Deterministic FlatVol(6 rates, 3 factors,
#     semiannual, exponential correlation). Linear-algebra Jacobians TIGHT;
#     analytic-vs-numerical and orthogonalised-bump reconstruction LOOSE. ===
add_executable(cluster_w11d_probe cluster_w11d/probe.cpp)
target_link_libraries(cluster_w11d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W12-A cluster: CMS coupons + GFunction/Hagan (Conundrum) replication
#     pricers + LognormalCmsSpreadPricer. The core ql/cashflows/ CMS-coupon
#     family W8-A deferred — a genuine gap. GFunctionStandard closed-form
#     value/derivatives and the ConundrumIntegrand value are probed TIGHT;
#     AnalyticHaganPricer convexity-adjusted CMS coupon rates (Standard /
#     ExactYield / ParallelShifts / NonParallelShifts) on a flat 5% curve +
#     constant lognormal swaption vol, NumericHaganPricer-vs-Analytic
#     agreement (the canonical cms.cpp testFairRate 2e-4 equivalence), and the
#     LognormalCmsSpreadPricer CMS-spread coupon rate at known correlation are
#     probed LOOSE (static replication / bivariate-lognormal integration). ===
add_executable(cluster_w12a_probe cluster_w12a/probe.cpp)
target_link_libraries(cluster_w12a_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W12-C cluster: core ql/cashflows gap-fill — dividends + average-BMA
#     coupon + equity (quanto) cashflow + overnight-indexed coupon pricers.
#     FixedDividend / FractionalDividend amount() + amount(underlying) +
#     DividendVector (EXACT/TIGHT); AverageBMACoupon rate via the built-in
#     AverageBMACouponPricer over a fully-known weekly-Wednesday BMA fixing
#     history — calendar-day-weighted average (LOOSE); EquityQuantoCashFlowPricer
#     quanto-adjusted growth amount on a flat-curve flat-vol EquityIndex
#     (LOOSE); CompoundingOvernightIndexedCouponPricer and
#     ArithmeticAveragedOvernightIndexedCouponPricer coupon rate over a fully
#     known overnight (Sofr) fixing history — deterministic compounding /
#     averaging, no forward curve (TIGHT). ===
add_executable(cluster_w12c_probe cluster_w12c/probe.cpp)
target_link_libraries(cluster_w12c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W12-B cluster: core ql/cashflows gap-fill — capped/floored + digital
#     coupon families that L2-D deferred. CappedFlooredIborCoupon.rate() with a
#     cap / floor / collar (Black-vol-adjusted swaplet via BlackIborCouponPricer
#     + ConstantOptionletVolatility on the canonical digitalcoupon.cpp setup —
#     Euribor6M / flat 5% / fixingDays 2 / 10Y-forward coupon) is LOOSE;
#     effectiveCap/effectiveFloor ((strike-spread)/gearing) TIGHT. DigitalCoupon
#     callOptionRate/putOptionRate via call/put-spread replication at a small
#     gap (LOOSE — replication ≈ Cox-Rubinstein asset-or-nothing) plus
#     Central/Sub/Super replication-type rates and a cash-or-nothing call.
#     StrippedCappedFlooredCoupon.rate() = the extracted caplet/floorlet
#     optionality value, cross-checked against the underlying
#     CappedFlooredCoupon's swaplet - underlying.rate() identity (TIGHT).
#     CappedFlooredCmsCoupon uncapped degenerates to the AnalyticHaganPricer
#     CMS swaplet rate (LOOSE — Hagan convexity adjustment). ===
add_executable(cluster_w12b_probe cluster_w12b/probe.cpp)
target_link_libraries(cluster_w12b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W-S2 cluster (sibling-packages): SECONDARY economic-sanity reference for
#     the retired-API dividend-option compat layer (pquantlib-helpers).
#     DividendVanillaOption / BinomialDividendVanillaEngine /
#     BlackScholesDividendLattice do NOT exist in v1.42.1; this probe emits the
#     closed-form AnalyticDividendEuropeanEngine(process, DividendSchedule)
#     European value (+ no-dividend AnalyticEuropeanEngine context) as a LOOSE
#     ~1e-3 economic check for the compat European CRR-tree NPV. ===
add_executable(cluster_ws2_probe cluster_ws2/probe.cpp)
target_link_libraries(cluster_ws2_probe PRIVATE ${QUANTLIB_LIBRARY})

# === W-S4 cluster (sibling-packages): ConvertibleBond subsystem into pquantlib
#     CORE. PRIMARY same-method same-N cross-validation: ConvertibleFixedCouponBond
#     priced via BinomialConvertibleEngine<CoxRossRubinstein> (Tsiveriotis-Fernandes
#     credit-adjusted lattice) at a FIXED N=801. European / American /
#     American+SoftCallability+Put+FixedDividend NPVs are emitted for TIGHT
#     tree-vs-identical-tree cross-validation (no analytic convertible value
#     exists; the C++ same-engine value IS the reference). ===
add_executable(cluster_ws4_probe cluster_ws4/probe.cpp)
target_link_libraries(cluster_ws4_probe PRIVATE ${QUANTLIB_LIBRARY})

# === WS3-FD1 cluster: legacy FD linear-algebra core. TridiagonalOperator
#     (applyTo / solveFor / SOR / identity + operator algebra); BSMOperator
#     coefficient stencil rebuilt into a real TridiagonalOperator;
#     FiniteDifferenceModel + CrankNicolson<TridiagonalOperator> rollback of a
#     BSM operator over a payoff vector (deprecated old-FD-framework headers). ===
add_executable(cluster_ws3fd1_probe cluster_ws3fd1/probe.cpp)
target_link_libraries(cluster_ws3fd1_probe PRIVATE ${QUANTLIB_LIBRARY})
