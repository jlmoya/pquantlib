// migration-harness/cpp/probes/v143_ts_andreasenhuge/probe.cpp
//
// Pins QuantLib v1.43 AndreasenHugeVolatilityInterpl + its two adapters
// (ql/termstructures/volatility/equityfx/andreasenhuge{volatilityinterpl,
//  volatilityadapter,localvoladapter}.{hpp,cpp}).
//
// The class is a LazyObject whose performCalculations() runs a per-expiry
// least-squares calibration.  That splits the surface cleanly into two layers,
// and the probe emits them SEPARATELY because only one of them is
// optimiser-independent:
//
//   LAYER A ("noopt_*") — the FDM machinery with the calibration frozen.
//     A no-op OptimizationMethod leaves Problem::currentValue() at
//     CostFunction::initialValues() (a flat 0.25 vector,
//     andreasenhugevolatilityinterpl.cpp:148-150), so every emitted number is
//     a pure function of the Concentrating1dMesher, the FirstDerivative /
//     SecondDerivative operators, the TripleBandLinearOp splitting solve and
//     the interpolators.  No optimiser is involved: this is the TIGHT layer.
//
//   LAYER B ("lm_*") — the same quantities after a real LevenbergMarquardt
//     calibration.  MINPACK's iterates are not reproducible by a different
//     LM implementation, so these are emitted for the record and for a
//     calibration-QUALITY comparison, not for a tight parameter match.
//
// Also emitted: the mesher grid itself (so a port can be debugged one layer
// down), the standalone blackFormulaImpliedStdDevLiRS values that
// AndreasenHugeVolatilityAdapter::blackVarianceImpl depends on, and every
// inspector of both adapters.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/andreasenhuge.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/methods/finitedifferences/meshers/concentrating1dmesher.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/andreasenhugelocalvoladapter.hpp>
#include <ql/termstructures/volatility/equityfx/andreasenhugevolatilityadapter.hpp>
#include <ql/termstructures/volatility/equityfx/andreasenhugevolatilityinterpl.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// --- fixed market ---------------------------------------------------------

const Date kRefDate(15, June, 2023);
const Real kSpot = 100.0;
const Rate kRiskFree = 0.04;
const Rate kDividend = 0.02;
const Size kGridPoints = 60;

const Date kExpiries[] = {Date(15, September, 2023), Date(15, December, 2023),
                          Date(17, June, 2024)};
const Real kCalStrikes[] = {80.0, 90.0, 100.0, 110.0, 120.0};
// smile per expiry (rows = expiries, columns = strikes)
const Volatility kCalVols[3][5] = {{0.271, 0.234, 0.208, 0.196, 0.201},
                                   {0.258, 0.226, 0.204, 0.194, 0.198},
                                   {0.246, 0.219, 0.201, 0.193, 0.196}};

// Query grid.  0.10 sits before the first expiry (92/365 = 0.2521), 1.2 past
// the last (368/365 = 1.0082) — getExerciseTimeIdx clamps to the last slice.
const Time kQueryTimes[] = {0.10, 0.2520547945205479, 0.4, 0.7534246575342466, 1.0, 1.2};
const Real kQueryStrikes[] = {70.0, 85.0, 100.0, 115.0, 135.0};

// A frozen "optimizer": it runs no search, it just stamps a fixed, spelled-out
// sigma vector onto the Problem.  Everything downstream then depends only on
// the FDM machinery, never on an optimiser's iterates.
//
// The vector is deliberately NON-FLAT.  With a flat sigma the three
// InterpolationTypes (PiecewiseConstant / Linear / CubicSpline) all reproduce
// the same constant, so a flat freeze would leave the interpolation branch
// (andreasenhugevolatilityinterpl.cpp:83-104) completely uncovered.
class FrozenOptimizationMethod : public OptimizationMethod {
  public:
    static Real sigma(Size i) { return 0.30 - 0.02 * Real(i) + 0.01 * Real(i % 2); }

    EndCriteria::Type minimize(Problem& p, const EndCriteria&) override {
        Array x = p.currentValue();
        for (Size i = 0; i < x.size(); ++i)
            x[i] = sigma(i);
        p.setCurrentValue(x);
        return EndCriteria::None;
    }
};

Handle<Quote> spotHandle() {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(kSpot));
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kRefDate, r, Actual365Fixed()));
}

AndreasenHugeVolatilityInterpl::CalibrationSet calibrationSet() {
    AndreasenHugeVolatilityInterpl::CalibrationSet set;
    for (Size i = 0; i < 3; ++i) {
        const ext::shared_ptr<Exercise> exercise =
            ext::make_shared<EuropeanExercise>(kExpiries[i]);
        for (Size j = 0; j < 5; ++j) {
            const ext::shared_ptr<StrikedTypePayoff> payoff =
                ext::make_shared<PlainVanillaPayoff>(Option::Call, kCalStrikes[j]);
            set.emplace_back(ext::make_shared<VanillaOption>(payoff, exercise),
                             ext::make_shared<SimpleQuote>(kCalVols[i][j]));
        }
    }
    return set;
}

ext::shared_ptr<AndreasenHugeVolatilityInterpl>
makeInterpl(AndreasenHugeVolatilityInterpl::InterpolationType it,
            AndreasenHugeVolatilityInterpl::CalibrationType ct, bool frozen,
            Size nGridPoints = kGridPoints) {
    ext::shared_ptr<OptimizationMethod> om =
        frozen ? ext::shared_ptr<OptimizationMethod>(ext::make_shared<FrozenOptimizationMethod>())
               : ext::shared_ptr<OptimizationMethod>(ext::make_shared<LevenbergMarquardt>());
    return ext::make_shared<AndreasenHugeVolatilityInterpl>(
        calibrationSet(), spotHandle(), flatCurve(kRiskFree), flatCurve(kDividend), it, ct,
        nGridPoints, Null<Real>(), Null<Real>(), om,
        EndCriteria(500, 100, 1e-12, 1e-10, 1e-10));
}

// --- JSON helpers ---------------------------------------------------------

std::string tryRun(const std::function<void()>& f) {
    try {
        f();
        return "";
    } catch (const std::exception& e) {
        return std::string(e.what());
    }
}

void emitString(const std::string& name, const std::string& v, const char* indent,
                bool last = false) {
    std::cout << indent << "\"" << name << "\": \"";
    for (char c : v) {
        if (c == '"' || c == '\\')
            std::cout << '\\' << c;
        else if (c == '\n')
            std::cout << "\\n";
        else
            std::cout << c;
    }
    std::cout << "\"" << (last ? "\n" : ",\n");
}

void emitArray(const std::string& name, const std::vector<Real>& v, const char* indent,
               bool last = false) {
    std::cout << indent << "\"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << v[i];
    std::cout << "]" << (last ? "\n" : ",\n");
}

void emitScalar(const std::string& name, Real v, const char* indent, bool last = false) {
    std::cout << indent << "\"" << name << "\": " << v << (last ? "\n" : ",\n");
}

// Everything the interpolator itself exposes, over the query grid.
void emitInterplBlock(const std::string& name,
                      const ext::shared_ptr<AndreasenHugeVolatilityInterpl>& ah, bool last) {
    std::cout << "    \"" << name << "\": {\n";

    Real minErr, maxErr, avgErr;
    std::tie(minErr, maxErr, avgErr) = ah->calibrationError();
    emitScalar("calibration_min_error", minErr, "      ");
    emitScalar("calibration_max_error", maxErr, "      ");
    emitScalar("calibration_avg_error", avgErr, "      ");

    std::vector<Real> callPrices, putPrices, localVols, fwds;
    for (Time t : kQueryTimes) {
        fwds.push_back(ah->fwd(t));
        for (Real k : kQueryStrikes) {
            callPrices.push_back(ah->optionPrice(t, k, Option::Call));
            putPrices.push_back(ah->optionPrice(t, k, Option::Put));
            localVols.push_back(ah->localVol(t, k));
        }
    }
    emitArray("fwd", fwds, "      ");
    emitArray("call_price", callPrices, "      ");
    emitArray("put_price", putPrices, "      ");
    emitArray("local_vol", localVols, "      ");

    std::cout << "      \"max_date_serial\": " << ah->maxDate().serialNumber() << ",\n";
    emitScalar("min_strike", ah->minStrike(), "      ");
    emitScalar("max_strike", ah->maxStrike(), "      ", true);

    std::cout << "    }" << (last ? "\n" : ",\n");
}

// Both adapters over the same query grid.
void emitAdapterBlock(const std::string& name,
                      const ext::shared_ptr<AndreasenHugeVolatilityInterpl>& ah, bool last) {
    std::cout << "    \"" << name << "\": {\n";

    const ext::shared_ptr<AndreasenHugeVolatilityAdapter> volAdapter =
        ext::make_shared<AndreasenHugeVolatilityAdapter>(ah);
    const ext::shared_ptr<AndreasenHugeLocalVolAdapter> lvAdapter =
        ext::make_shared<AndreasenHugeLocalVolAdapter>(ah);

    std::vector<Real> blackVars, blackVols, atmLevels, localVols;
    for (Time t : kQueryTimes) {
        atmLevels.push_back(volAdapter->atmLevel(t));
        for (Real k : kQueryStrikes) {
            blackVars.push_back(volAdapter->blackVariance(t, k, true));
            blackVols.push_back(volAdapter->blackVol(t, k, true));
            localVols.push_back(lvAdapter->localVol(t, k, true));
        }
    }
    emitArray("vol_adapter_black_variance", blackVars, "      ");
    emitArray("vol_adapter_black_vol", blackVols, "      ");
    emitArray("vol_adapter_atm_level", atmLevels, "      ");
    emitArray("local_vol_adapter_local_vol", localVols, "      ");

    // Both adapters delegate calendar() / dayCounter() / settlementDays() /
    // referenceDate() to volInterpl_->riskFreeRate() (adapter cpp:57-68 and
    // localvoladapter cpp:51-62).  The curves here are FlatForwards built on
    // an explicit reference date, so C++ leaves calendar_ empty and
    // settlementDays_ Null — calling either THROWS.  That pass-through
    // behaviour is the thing being pinned, so it is recorded rather than
    // hidden behind a nicer curve.
    std::cout << "      \"vol_adapter_max_date_serial\": " << volAdapter->maxDate().serialNumber()
              << ",\n";
    emitScalar("vol_adapter_min_strike", volAdapter->minStrike(), "      ");
    emitScalar("vol_adapter_max_strike", volAdapter->maxStrike(), "      ");
    std::cout << "      \"vol_adapter_day_counter\": \"" << volAdapter->dayCounter().name()
              << "\",\n";
    std::cout << "      \"vol_adapter_calendar_empty\": "
              << (volAdapter->calendar().empty() ? "true" : "false") << ",\n";
    emitString("vol_adapter_settlement_days_error",
               tryRun([&]() { (void)volAdapter->settlementDays(); }), "      ");
    std::cout << "      \"vol_adapter_reference_date_serial\": "
              << volAdapter->referenceDate().serialNumber() << ",\n";

    std::cout << "      \"local_vol_adapter_max_date_serial\": "
              << lvAdapter->maxDate().serialNumber() << ",\n";
    emitScalar("local_vol_adapter_min_strike", lvAdapter->minStrike(), "      ");
    emitScalar("local_vol_adapter_max_strike", lvAdapter->maxStrike(), "      ");
    std::cout << "      \"local_vol_adapter_day_counter\": \"" << lvAdapter->dayCounter().name()
              << "\",\n";
    std::cout << "      \"local_vol_adapter_calendar_empty\": "
              << (lvAdapter->calendar().empty() ? "true" : "false") << ",\n";
    emitString("local_vol_adapter_settlement_days_error",
               tryRun([&]() { (void)lvAdapter->settlementDays(); }), "      ");
    std::cout << "      \"local_vol_adapter_reference_date_serial\": "
              << lvAdapter->referenceDate().serialNumber() << "\n";

    std::cout << "    }" << (last ? "\n" : ",\n");
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = kRefDate;

    std::cout << "{\n";

    // --- setup echo --------------------------------------------------------
    std::cout << "  \"setup\": {\n";
    std::cout << "    \"reference_date_serial\": " << kRefDate.serialNumber() << ",\n";
    emitScalar("spot", kSpot, "    ");
    emitScalar("risk_free_rate", kRiskFree, "    ");
    emitScalar("dividend_rate", kDividend, "    ");
    std::cout << "    \"n_grid_points\": " << kGridPoints << ",\n";
    std::cout << "    \"expiry_date_serials\": [";
    for (Size i = 0; i < 3; ++i)
        std::cout << (i ? ", " : "") << kExpiries[i].serialNumber();
    std::cout << "],\n";
    {
        std::vector<Real> ts, dts;
        Actual365Fixed dc;
        for (const Date& d : kExpiries)
            ts.push_back(dc.yearFraction(kRefDate, d));
        for (Size i = 0; i < ts.size(); ++i)
            dts.push_back(ts[i] - (i == 0 ? 0.0 : ts[i - 1]));
        emitArray("expiry_times", ts, "    ");
        emitArray("dT", dts, "    ");
        std::vector<Real> ks(std::begin(kCalStrikes), std::end(kCalStrikes));
        emitArray("calibration_strikes", ks, "    ");
        std::vector<Real> vols;
        for (Size i = 0; i < 3; ++i)
            for (Size j = 0; j < 5; ++j)
                vols.push_back(kCalVols[i][j]);
        emitArray("calibration_vols_row_major", vols, "    ");
        std::vector<Real> qt(std::begin(kQueryTimes), std::end(kQueryTimes));
        emitArray("query_times", qt, "    ");
        std::vector<Real> qk(std::begin(kQueryStrikes), std::end(kQueryStrikes));
        emitArray("query_strikes", qk, "    ");
    }
    // The mesher the surface builds internally (cpp:353-359). Reproduced here
    // so a port can pin the grid one layer below the surface.
    {
        const Real minStrike = kCalStrikes[0] / 8.0;
        const Real maxStrike = 8.0 * kCalStrikes[4];
        emitScalar("default_min_strike", minStrike, "    ");
        emitScalar("default_max_strike", maxStrike, "    ");
        Concentrating1dMesher mesher(std::log(minStrike / kSpot), std::log(maxStrike / kSpot),
                                     kGridPoints, std::pair<Real, Real>(0.0, 0.025));
        const std::vector<Real>& loc = mesher.locations();
        emitArray("mesher_locations", std::vector<Real>(loc.begin(), loc.end()), "    ");
        std::vector<Real> frozen;
        for (Size i = 0; i < 5; ++i)
            frozen.push_back(FrozenOptimizationMethod::sigma(i));
        emitArray("frozen_sigmas", frozen, "    ", true);
    }
    std::cout << "  },\n";

    // --- blackFormulaImpliedStdDevLiRS, standalone -------------------------
    //
    // AndreasenHugeVolatilityAdapter::blackVarianceImpl inverts the Black
    // formula with this solver (andreasenhugevolatilityadapter.cpp:41-44),
    // called with guess = Null<Real>() so the RS closed-form approximation
    // supplies the seed.  Pinned separately because the adapter's variance is
    // meaningless if the inversion is wrong.
    //
    // Input range note: the RS seed breaks down at deep-out-of-the-money /
    // near-zero prices — its C term is a difference of two nearly equal
    // squares (blackformula.cpp:293), goes negative on rounding, and
    // -M_PI_2*log(beta) then feeds NaN into the Maddock inverse normal.  That
    // is a live C++ fragility, recorded under "throws" below rather than
    // papered over; the grid here stays inside the well-conditioned region.
    std::cout << "  \"li_rs\": {\n";
    {
        const Real fwds[] = {95.0, 100.0, 105.0};
        const Real ks[] = {85.0, 95.0, 100.0, 105.0, 115.0};
        const Real stdDevs[] = {0.15, 0.30, 0.60};
        const Real dfs[] = {1.0, 0.97};
        std::vector<Real> approxRS, liRS, roundTripStdDev;
        std::vector<Real> inFwd, inStrike, inStdDev, inDf, inPrice;
        for (Real f : fwds)
            for (Real k : ks)
                for (Real sd : stdDevs)
                    for (Real df : dfs) {
                        const Option::Type ot = (f > k) ? Option::Put : Option::Call;
                        const Real price = blackFormula(ot, k, f, sd, df);
                        inFwd.push_back(f);
                        inStrike.push_back(k);
                        inStdDev.push_back(sd);
                        inDf.push_back(df);
                        inPrice.push_back(price);
                        approxRS.push_back(
                            blackFormulaImpliedStdDevApproximationRS(ot, k, f, price, df, 0.0));
                        const Real v = blackFormulaImpliedStdDevLiRS(
                            ot, k, f, price, df, 0.0, Null<Real>(), 1.0, 1e-6, 1000);
                        liRS.push_back(v);
                        roundTripStdDev.push_back(v - sd);
                    }
        emitArray("input_forward", inFwd, "    ");
        emitArray("input_strike", inStrike, "    ");
        emitArray("input_std_dev", inStdDev, "    ");
        emitArray("input_discount", inDf, "    ");
        emitArray("input_price", inPrice, "    ");
        emitArray("approximation_rs", approxRS, "    ");
        emitArray("li_rs_std_dev", liRS, "    ");
        emitArray("li_rs_minus_input", roundTripStdDev, "    ", true);
    }
    std::cout << "  },\n";

    // --- LAYER A: frozen calibration (optimiser-independent) ---------------
    std::cout << "  \"noopt\": {\n";
    {
        using IT = AndreasenHugeVolatilityInterpl::InterpolationType;
        using CT = AndreasenHugeVolatilityInterpl::CalibrationType;
        emitInterplBlock("cubic_call", makeInterpl(IT::CubicSpline, CT::Call, true), false);
        emitInterplBlock("cubic_put", makeInterpl(IT::CubicSpline, CT::Put, true), false);
        emitInterplBlock("cubic_callput", makeInterpl(IT::CubicSpline, CT::CallPut, true), false);
        emitInterplBlock("linear_call", makeInterpl(IT::Linear, CT::Call, true), false);
        emitInterplBlock("piecewise_constant_call",
                         makeInterpl(IT::PiecewiseConstant, CT::Call, true), false);
        emitInterplBlock("cubic_call_150pts",
                         makeInterpl(IT::CubicSpline, CT::Call, true, 150), true);
    }
    std::cout << "  },\n";

    std::cout << "  \"noopt_adapters\": {\n";
    {
        using IT = AndreasenHugeVolatilityInterpl::InterpolationType;
        using CT = AndreasenHugeVolatilityInterpl::CalibrationType;
        emitAdapterBlock("cubic_call", makeInterpl(IT::CubicSpline, CT::Call, true), false);
        emitAdapterBlock("cubic_put", makeInterpl(IT::CubicSpline, CT::Put, true), false);
        emitAdapterBlock("cubic_callput", makeInterpl(IT::CubicSpline, CT::CallPut, true), true);
    }
    std::cout << "  },\n";

    // --- LAYER B: real LevenbergMarquardt calibration ----------------------
    std::cout << "  \"lm\": {\n";
    {
        using IT = AndreasenHugeVolatilityInterpl::InterpolationType;
        using CT = AndreasenHugeVolatilityInterpl::CalibrationType;
        emitInterplBlock("cubic_call", makeInterpl(IT::CubicSpline, CT::Call, false), false);
        emitInterplBlock("cubic_callput", makeInterpl(IT::CubicSpline, CT::CallPut, false), false);
        emitInterplBlock("piecewise_constant_call",
                         makeInterpl(IT::PiecewiseConstant, CT::Call, false), true);
    }
    std::cout << "  },\n";

    std::cout << "  \"lm_adapters\": {\n";
    {
        using IT = AndreasenHugeVolatilityInterpl::InterpolationType;
        using CT = AndreasenHugeVolatilityInterpl::CalibrationType;
        emitAdapterBlock("cubic_call", makeInterpl(IT::CubicSpline, CT::Call, false), true);
    }
    std::cout << "  },\n";

    // --- constructor preconditions -----------------------------------------
    std::cout << "  \"throws\": {\n";
    {
        emitString("empty_calibration_set", tryRun([&]() {
                       AndreasenHugeVolatilityInterpl::CalibrationSet empty;
                       AndreasenHugeVolatilityInterpl(empty, spotHandle(), flatCurve(kRiskFree),
                                                      flatCurve(kDividend));
                   }),
                   "    ");
        emitString("too_few_grid_points", tryRun([&]() {
                       AndreasenHugeVolatilityInterpl(
                           calibrationSet(), spotHandle(), flatCurve(kRiskFree),
                           flatCurve(kDividend),
                           AndreasenHugeVolatilityInterpl::CubicSpline,
                           AndreasenHugeVolatilityInterpl::Call, 2);
                   }),
                   "    ");
        emitString("american_exercise_rejected", tryRun([&]() {
                       AndreasenHugeVolatilityInterpl::CalibrationSet set;
                       set.emplace_back(
                           ext::make_shared<VanillaOption>(
                               ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                               ext::make_shared<AmericanExercise>(kRefDate, kExpiries[0])),
                           ext::make_shared<SimpleQuote>(0.2));
                       AndreasenHugeVolatilityInterpl(set, spotHandle(), flatCurve(kRiskFree),
                                                      flatCurve(kDividend));
                   }),
                   "    ");
        emitString("min_strike_above_max_strike", tryRun([&]() {
                       AndreasenHugeVolatilityInterpl ah(
                           calibrationSet(), spotHandle(), flatCurve(kRiskFree),
                           flatCurve(kDividend), AndreasenHugeVolatilityInterpl::CubicSpline,
                           AndreasenHugeVolatilityInterpl::Call, kGridPoints, 200.0, 50.0);
                       ah.calibrationError();
                   }),
                   "    ");
        // The RS seed's C term (blackformula.cpp:293) is a difference of two
        // nearly equal squares; at a deep-OTM near-zero price it rounds
        // negative, log(beta) is NaN, and the Maddock inverse normal throws a
        // boost::math domain_error rather than a QuantLib::Error.  Recorded so
        // a port that reproduces the arithmetic reproduces the failure mode
        // too, instead of silently returning a plausible number.
        // f=105, K=70, stdDev=0.05 -> put price 1.3e-16.  The RS seed itself
        // survives; the LiRS iteration then feeds NaN into the Maddock inverse
        // normal and boost::math throws a domain_error, NOT a QuantLib::Error.
        emitString("li_rs_approximation_rs_deep_otm", tryRun([&]() {
                       const Real price = blackFormula(Option::Put, 70.0, 105.0, 0.05, 1.0);
                       (void)blackFormulaImpliedStdDevApproximationRS(Option::Put, 70.0, 105.0,
                                                                     price, 1.0, 0.0);
                   }),
                   "    ");
        emitString("li_rs_deep_otm_breaks_down", tryRun([&]() {
                       const Real price = blackFormula(Option::Put, 70.0, 105.0, 0.05, 1.0);
                       (void)blackFormulaImpliedStdDevLiRS(Option::Put, 70.0, 105.0, price, 1.0,
                                                           0.0, Null<Real>(), 1.0, 1e-6, 1000);
                   }),
                   "    ", true);
    }
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
