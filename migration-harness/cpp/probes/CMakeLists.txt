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

# === W4-B cluster: compound + chooser + extensible options +
#     Kirk spread engine + AnalyticPdfHeston + continuous Asian
#     Levy/Vecer engines. Closed-form exotic options under BSM and
#     PDF-/PDE-based engines under Heston and Asian PDE. ===
add_executable(cluster_w4b_probe cluster_w4b/probe.cpp)
target_link_libraries(cluster_w4b_probe PRIVATE ${QUANTLIB_LIBRARY})
