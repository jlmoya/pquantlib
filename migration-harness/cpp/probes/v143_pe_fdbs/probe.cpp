// migration-harness/cpp/probes/v143_pe_fdbs/probe.cpp
//
// Reference values for the *finite-difference Black-Scholes* pricing-engine
// cluster of C++ QuantLib v1.43:
//
//   * FdBlackScholesVanillaEngine / MakeFdBlackScholesVanillaEngine
//         ql/pricingengines/vanilla/fdblackscholesvanillaengine.{hpp,cpp}
//   * FdBlackScholesShoutEngine
//         ql/pricingengines/vanilla/fdblackscholesshoutengine.{hpp,cpp}
//   * FdCEVVanillaEngine
//         ql/pricingengines/vanilla/fdcevvanillaengine.{hpp,cpp}
//   * FdCIRVanillaEngine / MakeFdCIRVanillaEngine
//         ql/pricingengines/vanilla/fdcirvanillaengine.{hpp,cpp}
//   * FdSimpleBSSwingEngine
//         ql/pricingengines/vanilla/fdsimplebsswingengine.{hpp,cpp}
//   * FdBlackScholesAsianEngine
//         ql/pricingengines/asian/fdblackscholesasianengine.{hpp,cpp}
//   * FdBlackScholesBarrierEngine
//         ql/pricingengines/barrier/fdblackscholesbarrierengine.{hpp,cpp}
//   * FdBlackScholesRebateEngine
//         ql/pricingengines/barrier/fdblackscholesrebateengine.{hpp,cpp}
//   * Fd2dBlackScholesVanillaEngine
//         ql/pricingengines/basket/fd2dblackscholesvanillaengine.{hpp,cpp}
//   * FdndimBlackScholesVanillaEngine
//         ql/pricingengines/basket/fdndimblackscholesvanillaengine.{hpp,cpp}
//
// Why the values below are EXACT, not a confidence band
// -----------------------------------------------------
// Every engine here is a deterministic finite-difference rollback: a mesher
// with closed-form node placement, an operator splitting with fixed
// coefficients, a fixed number of time steps, and a cubic-spline read-out.
// No random numbers, no clock, no adaptive iteration whose step count could
// depend on the platform. Running the same engine twice returns bit-identical
// doubles, and a correct port reproduces them to ~1e-13 relative -- the only
// slack being the order of a handful of floating-point sums. A band test
// would pass with the wrong scheme, the wrong mesh, or a missing step
// condition, so nothing here is banded.
//
// What has to be pinned, and why
// ------------------------------
//  1. THE DEFAULT SCHEME. Every constructor of FdBlackScholesVanillaEngine,
//     FdBlackScholesShoutEngine, FdCEVVanillaEngine, FdSimpleBSSwingEngine,
//     FdBlackScholesAsianEngine, FdBlackScholesBarrierEngine,
//     FdBlackScholesRebateEngine and FdndimBlackScholesVanillaEngine defaults
//     `schemeDesc` to FdmSchemeDesc::Douglas(). Fd2dBlackScholesVanillaEngine
//     defaults to Hundsdorfer() and FdCIRVanillaEngine to
//     ModifiedHundsdorfer(). Those are three DIFFERENT defaults and they give
//     three different numbers. Careful: in ONE dimension Douglas and
//     Crank-Nicolson coincide -- FdmBackwardSolver's DouglasScheme with
//     theta = 0.5 and CrankNicolsonScheme with theta = 0.5 apply the same
//     operator when there is no splitting to do, and
//     `vanilla_european_call_crank_nicolson` is pinned to prove they agree BIT
//     FOR BIT. So the discriminating 1-D case is
//     `vanilla_european_call_implicit_euler`, which differs in the third
//     decimal. In two dimensions the defaults genuinely separate, which is why
//     the Fd2d and CIR cases take Hundsdorfer / ModifiedHundsdorfer.
//  2. GREEKS. Each engine fills a specific subset of results_:
//        - vanilla / shout / CEV / CIR / barrier / rebate / swing: value,
//          delta, gamma, theta.
//        - asian: value, delta, gamma -- NOT theta.
//        - Fd2d basket: value, delta (= deltaX + deltaY), gamma
//          (= gammaX + gammaY + 2 gammaXY), theta.
//        - Fdndim basket: value ONLY.
//     All of them are pinned, including their absence (`theta_throws` /
//     `delta_throws` carry the exact C++ "theta not provided" message).
//     `vanilla_local_vol` pins the localVol=true branch against a CONSTANT
//     Black vol, where the local vol surface collapses to the same number, so
//     the case proves the branch runs and agrees -- it cannot, on its own,
//     prove a non-flat local vol surface is handled.
//  3. CASH DIVIDEND MODEL. FdBlackScholesVanillaEngine::CashDividendModel is
//     Spot by default. Escrowed shifts the SPOT by
//     EscrowedDividendAdjustment::dividendAdjustment(t_settlement), swaps the
//     early-exercise calculator for FdmEscrowedLogInnerValueCalculator, and --
//     for non-European exercise only -- passes a ZERO-AMOUNT copy of the
//     dividend schedule to the step conditions so the dividend dates survive
//     as stopping times without moving the grid. The two models give
//     materially different American prices; both are pinned.
//  4. THE DIVIDEND SCHEDULE REACHES THE MESHER. FdmBlackScholesMesher's
//     forward walk inserts one (time, amount) point per dividend inside
//     [0, maturity] BEFORE the uniform points, then sorts. That moves the
//     mi/ma envelope and hence xMin/xMax. `*_dividends_*` cases exist so a
//     port that drops the schedule on the way to the mesher is caught even
//     when its dividend step condition is right.
//  5. IN-OUT PARITY IS NOT FREE. FdBlackScholesBarrierEngine prices a
//     knock-IN as vanilla + rebate - knock-OUT, where the vanilla leg is an
//     FdBlackScholesVanillaEngine with dampingSteps forced to 0 and the
//     rebate leg an FdBlackScholesRebateEngine on max(50, xGrid/5) points
//     with min(1, dampingSteps/2) damping steps. Those three constants are
//     load-bearing: `barrier_downin_damped` uses dampingSteps = 5 so the
//     min(Size(1), 5/2) == 1 clamp actually fires.
//  6. THE BARRIER MESH IS CLIPPED, THE REBATE MESH TOO. Both engines set
//     xMin = log(barrier) for Down* and xMax = log(barrier) for Up*, and pass
//     cPoint = (Null, Null) -- i.e. a UNIFORM helper mesh, unlike the vanilla
//     engine which concentrates at the strike. A port that reuses the vanilla
//     mesher call gets every barrier number wrong.
//  7. CEV BOUNDARIES. FdCEVVanillaEngine always installs an upper
//     FdmTimeDepDirichletBoundary priced by CEVCalculator, and installs a
//     lower FdmDiscountDirichletBoundary only when
//     delta = (1-2 beta)/(1-beta) < 2. beta = 0.6 gives delta = -0.5 < 2
//     (boundary present); beta = 1.4 gives delta = 4.5 > 2 (absent). Both
//     branches are pinned. The engine drives Fdm1DimSolver directly, so
//     delta/gamma are the RAW spline derivatives in forward space -- not
//     divided by f0 the way FdmBlackScholesSolver divides by the spot.
//  8. SWING. FdSimpleBSSwingEngine reads the value at (spot, 1) on a
//     Uniform1dMesher(0, maxExerciseRights, maxExerciseRights + 1) exercise
//     axis, and its delta/gamma use a bump of spot*0.01, not an analytic
//     derivative. `swing_rights_binding` sets minExerciseRights == number of
//     exercise dates so every right must be used, which activates the
//     `exercisesUsed + d <= minExercises_` branch of
//     FdmSimpleSwingCondition::applyTo.
//  9. ASIAN. The average axis bounds come from
//        xMin = min(log(avg) - 0.25 r, log(spot) - 1.5 r)
//        xMax = max(log(avg) + 0.25 r, log(spot) + 1.5 r)
//     with r = blackVol(T, K) sqrt(T) * N^{-1}(1 - 1e-4); the payoff is read
//     on direction 1 (the average), not 0. `asian_running` gives a non-zero
//     runningAccumulator with pastFixings > 0 so avg != spot and the two
//     branches of the xMin/xMax min/max separate.
// 10. NDIM. FdndimBlackScholesVanillaEngine solves in PCA space: each axis is
//     a Predefined1dMesher on 1.3 sqrt(l_i T) N^{-1}(eps + j h) with
//     eps = 1e-4, the value is read at the ORIGIN of every axis, and -- for a
//     EuropeanExercise only -- FdmWienerOp gets a NULL discount curve and the
//     result is multiplied by the interest-rate discount factor at the end.
//     For an AmericanExercise the curve goes into the operator instead and
//     there is no final multiplication. Both branches are pinned, at n = 2
//     and n = 3.
// 11. EXPECTED FAILURES. Barrier engines reject a triggered spot ("barrier
//     touched"), non-European exercise ("only european style option are
//     supported"), and a non-positive strike ("strike must be positive").
//     The escrowed vanilla engine rejects a quanto helper. The ndim engine
//     rejects more than PDE_MAX_SUPPORTED_DIM (4) underlyings and a
//     mis-shaped correlation matrix. All pinned as {"throws": true, "what"}.
//
// Emits JSON on stdout; nothing else may be printed.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/cashflows/dividend.hpp>
#include <ql/exercise.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/averagetype.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/instruments/vanillaswingoption.hpp>
#include <ql/math/matrix.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/fdmsimpleprocess1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmcirop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/pricingengines/asian/fdblackscholesasianengine.hpp>
#include <ql/pricingengines/barrier/fdblackscholesbarrierengine.hpp>
#include <ql/pricingengines/barrier/fdblackscholesrebateengine.hpp>
#include <ql/pricingengines/basket/fd2dblackscholesvanillaengine.hpp>
#include <ql/pricingengines/basket/fdndimblackscholesvanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesshoutengine.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdcevvanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdcirvanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdsimplebsswingengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/coxingersollrossprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// ---------------------------------------------------------------------------
std::string num(Real v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

std::string dateStr(const Date& d) {
    std::ostringstream o;
    o << d.year() << "-" << std::setfill('0') << std::setw(2)
      << static_cast<int>(d.month()) << "-" << std::setw(2) << d.dayOfMonth();
    return o.str();
}

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, "\"" + v + "\""); }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
    Obj& d(const std::string& k, const Date& v) { return s(k, dateStr(v)); }
    Obj& nv(const std::string& k, const std::vector<Real>& v) {
        std::string a = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            a += (j ? ", " : "") + num(v[j]);
        return put(k, a + "]");
    }
    Obj& iv(const std::string& k, const std::vector<Size>& v) {
        std::string a = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            a += (j ? ", " : "") + std::to_string(v[j]);
        return put(k, a + "]");
    }
    Obj& dv(const std::string& k, const std::vector<Date>& v) {
        std::string a = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            a += std::string(j ? ", " : "") + "\"" + dateStr(v[j]) + "\"";
        return put(k, a + "]");
    }
    Obj& mat(const std::string& k, const Matrix& m) {
        std::string a = "[";
        for (Size r = 0; r < m.rows(); ++r) {
            a += std::string(r ? ", " : "") + "[";
            for (Size c = 0; c < m.columns(); ++c)
                a += (c ? ", " : "") + num(m[r][c]);
            a += "]";
        }
        return put(k, a + "]");
    }
    std::string str() const { return "{" + body_ + "}"; }

  private:
    Obj& put(const std::string& k, const std::string& v) {
        if (!body_.empty())
            body_ += ", ";
        body_ += "\"" + k + "\": " + v;
        return *this;
    }
    std::string body_;
};

std::vector<std::pair<std::string, std::string>> gCases;

void addCase(const std::string& name, const Obj& inputs, const Obj& expected) {
    gCases.emplace_back(name, "{\"inputs\": " + inputs.str() +
                                  ", \"expected\": " + expected.str() + "}");
}

void emitDocument() {
    std::cout << "{\n";
    for (std::size_t i = 0; i < gCases.size(); ++i)
        std::cout << "  \"" << gCases[i].first << "\": " << gCases[i].second
                  << (i + 1 < gCases.size() ? "," : "") << "\n";
    std::cout << "}\n";
}

// Record a case that is expected to throw, with the exact C++ message.
template <class F>
void addThrowCase(const std::string& name, const Obj& inputs, F&& f) {
    Obj ex;
    try {
        f();
        ex.b("throws", false).s("what", "");
    } catch (const std::exception& e) {
        ex.b("throws", true).s("what", std::string(e.what()));
    }
    addCase(name, inputs, ex);
}

// ---------------------------------------------------------------------------
// Market. 15 May 2025 -> 15 May 2026 is exactly 365 days, so with
// Actual365Fixed the exercise time is exactly 1.0 and no day-count rounding
// noise leaks into any reference value.
// ---------------------------------------------------------------------------
const Date kToday(15, May, 2025);
const Date kMaturity(15, May, 2026);

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kToday, r, dayCounter()));
}

Handle<BlackVolTermStructure> flatVol(Volatility v) {
    return Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(
        kToday, NullCalendar(), v, dayCounter()));
}

ext::shared_ptr<GeneralizedBlackScholesProcess>
bsmProcess(Real spot, Rate r, Rate q, Volatility vol) {
    return ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(spot)), flatCurve(q),
        flatCurve(r), flatVol(vol));
}

// Every case carries its whole market description so the Python side rebuilds
// it rather than restating constants.
Obj marketInputs(Real spot, Rate r, Rate q, Volatility vol) {
    Obj in;
    in.d("evaluation_date", kToday)
        .s("day_counter", "Actual365Fixed")
        .s("calendar", "NullCalendar")
        .n("spot", spot)
        .n("risk_free_rate", r)
        .n("dividend_yield", q)
        .n("volatility", vol);
    return in;
}

void greeks(Obj& ex, const VanillaOption& o) {
    ex.n("npv", o.NPV())
        .n("delta", o.delta())
        .n("gamma", o.gamma())
        .n("theta", o.theta());
}

void greeks(Obj& ex, const BarrierOption& o) {
    ex.n("npv", o.NPV())
        .n("delta", o.delta())
        .n("gamma", o.gamma())
        .n("theta", o.theta());
}

// ---------------------------------------------------------------------------
// 1. FdBlackScholesVanillaEngine
// ---------------------------------------------------------------------------
void vanillaCase(const std::string& name, Option::Type type, Real strike,
                 const ext::shared_ptr<Exercise>& exercise,
                 const std::string& exerciseKind,
                 const std::vector<Date>& exerciseDates, Real spot, Rate r,
                 Rate q, Volatility vol, Size tGrid, Size xGrid,
                 Size dampingSteps, const FdmSchemeDesc& scheme,
                 const std::string& schemeName, bool localVol,
                 const DividendSchedule& dividends,
                 const std::vector<Date>& divDates,
                 const std::vector<Real>& divAmounts,
                 FdBlackScholesVanillaEngine::CashDividendModel model,
                 const std::string& modelName) {
    const auto process = bsmProcess(spot, r, q, vol);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);

    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
        process, dividends, tGrid, xGrid, dampingSteps, scheme, localVol,
        -Null<Real>(), model));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", exerciseKind)
        .dv("exercise_dates", exerciseDates)
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .s("scheme", schemeName)
        .b("local_vol", localVol)
        .dv("dividend_dates", divDates)
        .nv("dividend_amounts", divAmounts)
        .s("cash_dividend_model", modelName);

    Obj ex;
    greeks(ex, option);
    addCase(name, in, ex);
}

DividendSchedule divSchedule(const std::vector<Date>& d,
                             const std::vector<Real>& a) {
    return DividendVector(d, a);
}

void vanillaCases() {
    const auto european = ext::shared_ptr<Exercise>(
        ext::make_shared<EuropeanExercise>(kMaturity));
    const auto american = ext::shared_ptr<Exercise>(
        ext::make_shared<AmericanExercise>(kToday, kMaturity));
    const std::vector<Date> bermudanDates = {kToday + 91, kToday + 182,
                                             kToday + 273, kMaturity};
    const auto bermudan = ext::shared_ptr<Exercise>(
        ext::make_shared<BermudanExercise>(bermudanDates));

    const std::vector<Date> noDates;
    const std::vector<Real> noAmounts;
    const DividendSchedule noDiv;

    // Default scheme is Douglas -- taken explicitly so a port defaulting to
    // Crank-Nicolson is caught by a *different* number, not a missing case.
    vanillaCase("vanilla_european_call_douglas", Option::Call, 100.0, european,
                "European", {kMaturity}, 100.0, 0.05, 0.02, 0.20, 100, 100, 0,
                FdmSchemeDesc::Douglas(), "Douglas", false, noDiv, noDates,
                noAmounts, FdBlackScholesVanillaEngine::Spot, "Spot");
    vanillaCase("vanilla_european_put_douglas", Option::Put, 110.0, european,
                "European", {kMaturity}, 100.0, 0.05, 0.02, 0.20, 100, 100, 0,
                FdmSchemeDesc::Douglas(), "Douglas", false, noDiv, noDates,
                noAmounts, FdBlackScholesVanillaEngine::Spot, "Spot");
    vanillaCase("vanilla_european_call_crank_nicolson", Option::Call, 100.0,
                european, "European", {kMaturity}, 100.0, 0.05, 0.02, 0.20, 100,
                100, 0, FdmSchemeDesc::CrankNicolson(), "CrankNicolson", false,
                noDiv, noDates, noAmounts, FdBlackScholesVanillaEngine::Spot,
                "Spot");
    vanillaCase("vanilla_european_call_implicit_euler", Option::Call, 100.0,
                european, "European", {kMaturity}, 100.0, 0.05, 0.02, 0.20, 100,
                100, 0, FdmSchemeDesc::ImplicitEuler(), "ImplicitEuler", false,
                noDiv, noDates, noAmounts, FdBlackScholesVanillaEngine::Spot,
                "Spot");
    vanillaCase("vanilla_american_put_douglas", Option::Put, 110.0, american,
                "American", {kToday, kMaturity}, 100.0, 0.05, 0.02, 0.20, 100,
                100, 0, FdmSchemeDesc::Douglas(), "Douglas", false, noDiv,
                noDates, noAmounts, FdBlackScholesVanillaEngine::Spot, "Spot");
    vanillaCase("vanilla_american_call_damped", Option::Call, 90.0, american,
                "American", {kToday, kMaturity}, 100.0, 0.05, 0.06, 0.25, 60,
                80, 4, FdmSchemeDesc::Douglas(), "Douglas", false, noDiv,
                noDates, noAmounts, FdBlackScholesVanillaEngine::Spot, "Spot");
    vanillaCase("vanilla_bermudan_put", Option::Put, 105.0, bermudan,
                "Bermudan", bermudanDates, 100.0, 0.05, 0.02, 0.22, 80, 90, 0,
                FdmSchemeDesc::Douglas(), "Douglas", false, noDiv, noDates,
                noAmounts, FdBlackScholesVanillaEngine::Spot, "Spot");
    vanillaCase("vanilla_local_vol", Option::Call, 100.0, european, "European",
                {kMaturity}, 100.0, 0.05, 0.02, 0.20, 100, 100, 0,
                FdmSchemeDesc::Douglas(), "Douglas", true, noDiv, noDates,
                noAmounts, FdBlackScholesVanillaEngine::Spot, "Spot");

    // Cash dividends: Spot vs Escrowed, European and American.
    const std::vector<Date> divDates = {kToday + 120, kToday + 300};
    const std::vector<Real> divAmounts = {2.5, 3.0};
    const DividendSchedule divs = divSchedule(divDates, divAmounts);

    vanillaCase("vanilla_european_dividends_spot", Option::Call, 100.0,
                european, "European", {kMaturity}, 100.0, 0.05, 0.0, 0.20, 80,
                90, 0, FdmSchemeDesc::Douglas(), "Douglas", false, divs,
                divDates, divAmounts, FdBlackScholesVanillaEngine::Spot,
                "Spot");
    vanillaCase("vanilla_european_dividends_escrowed", Option::Call, 100.0,
                european, "European", {kMaturity}, 100.0, 0.05, 0.0, 0.20, 80,
                90, 0, FdmSchemeDesc::Douglas(), "Douglas", false, divs,
                divDates, divAmounts, FdBlackScholesVanillaEngine::Escrowed,
                "Escrowed");
    vanillaCase("vanilla_american_dividends_spot", Option::Put, 105.0, american,
                "American", {kToday, kMaturity}, 100.0, 0.05, 0.0, 0.20, 80, 90,
                0, FdmSchemeDesc::Douglas(), "Douglas", false, divs, divDates,
                divAmounts, FdBlackScholesVanillaEngine::Spot, "Spot");
    vanillaCase("vanilla_american_dividends_escrowed", Option::Put, 105.0,
                american, "American", {kToday, kMaturity}, 100.0, 0.05, 0.0,
                0.20, 80, 90, 0, FdmSchemeDesc::Douglas(), "Douglas", false,
                divs, divDates, divAmounts,
                FdBlackScholesVanillaEngine::Escrowed, "Escrowed");
}

// The quanto branch: the mesher swaps the dividend curve for a
// QuantoTermStructure and FdmBlackScholesOp adds the quanto drift.
void vanillaQuantoCase() {
    const Real spot = 100.0, strike = 100.0;
    const Rate r = 0.05, q = 0.02, fRate = 0.03;
    const Volatility vol = 0.20, fxVol = 0.15;
    const Real corr = 0.4, fxAtm = 1.25;

    const auto process = bsmProcess(spot, r, q, vol);
    const auto quanto = ext::make_shared<FdmQuantoHelper>(
        *flatCurve(r), *flatCurve(fRate), *flatVol(fxVol), corr, fxAtm);

    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
        process, quanto, Size(80), Size(90), Size(0), FdmSchemeDesc::Douglas()));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("option_type", "Call")
        .n("strike", strike)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .i("t_grid", 80)
        .i("x_grid", 90)
        .i("damping_steps", 0)
        .s("scheme", "Douglas")
        .n("foreign_rate", fRate)
        .n("fx_volatility", fxVol)
        .n("equity_fx_correlation", corr)
        .n("exch_rate_atm_level", fxAtm);

    Obj ex;
    greeks(ex, option);
    addCase("vanilla_quanto", in, ex);
}

void vanillaMakeCases() {
    const Real spot = 100.0, strike = 100.0;
    const Rate r = 0.05, q = 0.0;
    const Volatility vol = 0.20;
    const auto process = bsmProcess(spot, r, q, vol);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
    const auto exercise = ext::make_shared<AmericanExercise>(kToday, kMaturity);

    // Builder defaults: tGrid = 100, xGrid = 100, dampingSteps = 0,
    // schemeDesc = Douglas, localVol = false, cashDividendModel = Spot.
    {
        VanillaOption option(payoff, exercise);
        option.setPricingEngine(MakeFdBlackScholesVanillaEngine(process));
        Obj in = marketInputs(spot, r, q, vol);
        in.s("option_type", "Put")
            .n("strike", strike)
            .s("exercise", "American")
            .dv("exercise_dates", {kToday, kMaturity});
        Obj ex;
        greeks(ex, option);
        ex.i("default_t_grid", 100)
            .i("default_x_grid", 100)
            .i("default_damping_steps", 0)
            .s("default_scheme", "Douglas");
        addCase("vanilla_make_defaults", in, ex);
    }

    // ... and the same engine spelled out, which must agree bit for bit.
    {
        VanillaOption option(payoff, exercise);
        option.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
            process, Size(100), Size(100), Size(0), FdmSchemeDesc::Douglas()));
        Obj in = marketInputs(spot, r, q, vol);
        in.s("option_type", "Put")
            .n("strike", strike)
            .s("exercise", "American")
            .dv("exercise_dates", {kToday, kMaturity});
        Obj ex;
        greeks(ex, option);
        addCase("vanilla_make_defaults_explicit_twin", in, ex);
    }

    // Builder with every knob turned.
    {
        const std::vector<Date> divDates = {kToday + 150};
        const std::vector<Real> divAmounts = {4.0};
        VanillaOption option(payoff, exercise);
        option.setPricingEngine(MakeFdBlackScholesVanillaEngine(process)
                                    .withTGrid(60)
                                    .withXGrid(70)
                                    .withDampingSteps(2)
                                    .withFdmSchemeDesc(FdmSchemeDesc::CrankNicolson())
                                    .withCashDividends(divDates, divAmounts)
                                    .withCashDividendModel(
                                        FdBlackScholesVanillaEngine::Escrowed));
        Obj in = marketInputs(spot, r, q, vol);
        in.s("option_type", "Put")
            .n("strike", strike)
            .s("exercise", "American")
            .dv("exercise_dates", {kToday, kMaturity})
            .i("t_grid", 60)
            .i("x_grid", 70)
            .i("damping_steps", 2)
            .s("scheme", "CrankNicolson")
            .dv("dividend_dates", divDates)
            .nv("dividend_amounts", divAmounts)
            .s("cash_dividend_model", "Escrowed");
        Obj ex;
        greeks(ex, option);
        addCase("vanilla_make_all_knobs", in, ex);
    }
}

void vanillaThrowCases() {
    const auto process = bsmProcess(100.0, 0.05, 0.0, 0.20);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    const auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);
    const std::vector<Date> divDates = {kToday + 120};
    const std::vector<Real> divAmounts = {2.0};

    Obj in = marketInputs(100.0, 0.05, 0.0, 0.20);
    in.s("option_type", "Call")
        .n("strike", 100.0)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .dv("dividend_dates", divDates)
        .nv("dividend_amounts", divAmounts)
        .s("cash_dividend_model", "Escrowed")
        .b("with_quanto_helper", true);

    addThrowCase("vanilla_escrowed_quanto_rejected", in, [&] {
        const auto quanto = ext::make_shared<FdmQuantoHelper>(
            *flatCurve(0.05), *flatCurve(0.03), *flatVol(0.15), 0.4, 1.25);
        VanillaOption option(payoff, exercise);
        option.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
            process, divSchedule(divDates, divAmounts), quanto, Size(40),
            Size(40), Size(0), FdmSchemeDesc::Douglas(), false, -Null<Real>(),
            FdBlackScholesVanillaEngine::Escrowed));
        option.NPV();
    });

    // Escrowed with dividends that exceed the spot.
    Obj in2 = marketInputs(5.0, 0.05, 0.0, 0.20);
    const std::vector<Date> bigDivDates = {kToday + 30};
    const std::vector<Real> bigDivAmounts = {8.0};
    in2.s("option_type", "Call")
        .n("strike", 5.0)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .dv("dividend_dates", bigDivDates)
        .nv("dividend_amounts", bigDivAmounts)
        .s("cash_dividend_model", "Escrowed");
    addThrowCase("vanilla_escrowed_negative_spot", in2, [&] {
        const auto p = bsmProcess(5.0, 0.05, 0.0, 0.20);
        VanillaOption option(
            ext::make_shared<PlainVanillaPayoff>(Option::Call, 5.0), exercise);
        option.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
            p, divSchedule(bigDivDates, bigDivAmounts), Size(40), Size(40),
            Size(0), FdmSchemeDesc::Douglas(), false, -Null<Real>(),
            FdBlackScholesVanillaEngine::Escrowed));
        option.NPV();
    });
}

// ---------------------------------------------------------------------------
// 2. FdBlackScholesShoutEngine
// ---------------------------------------------------------------------------
void shoutCase(const std::string& name, Option::Type type, Real strike,
               const ext::shared_ptr<Exercise>& exercise,
               const std::string& exerciseKind,
               const std::vector<Date>& exerciseDates, Real spot, Rate r,
               Rate q, Volatility vol, Size tGrid, Size xGrid,
               Size dampingSteps, const std::vector<Date>& divDates,
               const std::vector<Real>& divAmounts) {
    const auto process = bsmProcess(spot, r, q, vol);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);

    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<FdBlackScholesShoutEngine>(
        process, divSchedule(divDates, divAmounts), tGrid, xGrid, dampingSteps,
        FdmSchemeDesc::Douglas()));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", exerciseKind)
        .dv("exercise_dates", exerciseDates)
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .s("scheme", "Douglas")
        .dv("dividend_dates", divDates)
        .nv("dividend_amounts", divAmounts);

    Obj ex;
    greeks(ex, option);
    addCase(name, in, ex);
}

void shoutCases() {
    const auto european = ext::shared_ptr<Exercise>(
        ext::make_shared<EuropeanExercise>(kMaturity));
    const auto american = ext::shared_ptr<Exercise>(
        ext::make_shared<AmericanExercise>(kToday, kMaturity));
    const std::vector<Date> noDates;
    const std::vector<Real> noAmounts;

    shoutCase("shout_european_call", Option::Call, 100.0, european, "European",
              {kMaturity}, 100.0, 0.05, 0.02, 0.25, 100, 100, 0, noDates,
              noAmounts);
    shoutCase("shout_european_put", Option::Put, 100.0, european, "European",
              {kMaturity}, 100.0, 0.05, 0.02, 0.25, 100, 100, 0, noDates,
              noAmounts);
    // The shout right actually bites: American exercise on a put, where the
    // holder can lock in intrinsic and keep the residual option.
    shoutCase("shout_american_put", Option::Put, 105.0, american, "American",
              {kToday, kMaturity}, 100.0, 0.07, 0.0, 0.30, 100, 100, 0, noDates,
              noAmounts);
    shoutCase("shout_american_put_dividends", Option::Put, 105.0, american,
              "American", {kToday, kMaturity}, 100.0, 0.07, 0.0, 0.30, 80, 90,
              0, {kToday + 120, kToday + 300}, {2.0, 2.5});
}

// ---------------------------------------------------------------------------
// 3. FdCEVVanillaEngine
// ---------------------------------------------------------------------------
void cevCase(const std::string& name, Option::Type type, Real strike, Real f0,
             Real alpha, Real beta, Rate r, Size tGrid, Size xGrid,
             Size dampingSteps, Real scalingFactor, Real eps,
             const ext::shared_ptr<Exercise>& exercise,
             const std::string& exerciseKind,
             const std::vector<Date>& exerciseDates) {
    const auto discount = flatCurve(r);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);

    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<FdCEVVanillaEngine>(
        f0, alpha, beta, discount, tGrid, xGrid, dampingSteps, scalingFactor,
        eps, FdmSchemeDesc::Douglas()));

    Obj in;
    in.d("evaluation_date", kToday)
        .s("day_counter", "Actual365Fixed")
        .n("f0", f0)
        .n("alpha", alpha)
        .n("beta", beta)
        .n("risk_free_rate", r)
        .s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", exerciseKind)
        .dv("exercise_dates", exerciseDates)
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .n("scaling_factor", scalingFactor)
        .n("eps", eps)
        .s("scheme", "Douglas")
        .n("cev_delta_exponent", (1 - 2 * beta) / (1 - beta))
        .b("lower_boundary_present", (1 - 2 * beta) / (1 - beta) < 2.0);

    Obj ex;
    greeks(ex, option);
    addCase(name, in, ex);
}

void cevCases() {
    const auto european = ext::shared_ptr<Exercise>(
        ext::make_shared<EuropeanExercise>(kMaturity));
    const auto american = ext::shared_ptr<Exercise>(
        ext::make_shared<AmericanExercise>(kToday, kMaturity));

    // beta = 0.6 -> delta = (1-1.2)/(1-0.6) = -0.5 < 2, lower boundary present.
    // Strike deliberately != f0: at K == f0 the read-out point sits exactly on
    // the payoff kink, where MonotonicCubicNaturalSpline's SECOND derivative is
    // discontinuous, and `gamma` there is a coin-flip between the two one-sided
    // limits rather than a price. Put-call parity on gamma is asserted in the
    // Python test to prove the choice is not hiding a defect.
    cevCase("cev_european_call_beta06", Option::Call, 90.0, 100.0, 2.0, 0.6,
            0.05, 50, 200, 0, 1.0, 1e-4, european, "European", {kMaturity});
    cevCase("cev_european_put_beta06", Option::Put, 110.0, 100.0, 2.0, 0.6,
            0.05, 50, 200, 0, 1.0, 1e-4, european, "European", {kMaturity});
    // beta = 1.4 -> delta = (1-2.8)/(1-1.4) = 4.5 > 2, no lower boundary.
    cevCase("cev_european_call_beta14", Option::Call, 90.0, 100.0, 0.15, 1.4,
            0.05, 50, 200, 0, 1.0, 1e-4, european, "European", {kMaturity});
    // beta = 1 would divide by zero in the delta formula, so the interesting
    // "beta != 1" cases above bracket it. American exercise exercises the
    // FdmAmericanStepCondition path on a non-log grid.
    cevCase("cev_american_put_beta06", Option::Put, 110.0, 100.0, 2.0, 0.6,
            0.05, 50, 200, 0, 1.0, 1e-4, american, "American",
            {kToday, kMaturity});
    cevCase("cev_european_call_scaled", Option::Call, 90.0, 100.0, 2.0, 0.6,
            0.05, 40, 150, 2, 2.5, 1e-3, european, "European", {kMaturity});
    // Same market, K == f0 == 100: pinned precisely BECAUSE gamma is fragile
    // there, so a port that silently substitutes a different spline is caught.
    cevCase("cev_european_call_at_kink", Option::Call, 100.0, 100.0, 2.0, 0.6,
            0.05, 50, 200, 0, 1.0, 1e-4, european, "European", {kMaturity});
}

// ---------------------------------------------------------------------------
// 4. FdCIRVanillaEngine / MakeFdCIRVanillaEngine
// ---------------------------------------------------------------------------
void cirCase(const std::string& name, Option::Type type, Real strike,
             const ext::shared_ptr<Exercise>& exercise,
             const std::string& exerciseKind,
             const std::vector<Date>& exerciseDates, Real spot, Rate r, Rate q,
             Volatility vol, Real speed, Real mean, Real sigma, Real x0,
             Real rho, Size tGrid, Size xGrid, Size rGrid, Size dampingSteps) {
    const auto bsProcess = bsmProcess(spot, r, q, vol);
    const auto cirProcess =
        ext::make_shared<CoxIngersollRossProcess>(speed, sigma, x0, mean);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);

    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<FdCIRVanillaEngine>(
        cirProcess, bsProcess, tGrid, xGrid, rGrid, dampingSteps, rho,
        FdmSchemeDesc::ModifiedHundsdorfer()));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", exerciseKind)
        .dv("exercise_dates", exerciseDates)
        .n("cir_speed", speed)
        .n("cir_mean", mean)
        .n("cir_sigma", sigma)
        .n("cir_x0", x0)
        .n("rho", rho)
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .i("r_grid", static_cast<long long>(rGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .s("scheme", "ModifiedHundsdorfer");

    Obj ex;
    greeks(ex, option);
    addCase(name, in, ex);
}

void cirCases() {
    const auto european = ext::shared_ptr<Exercise>(
        ext::make_shared<EuropeanExercise>(kMaturity));
    const auto american = ext::shared_ptr<Exercise>(
        ext::make_shared<AmericanExercise>(kToday, kMaturity));

    // Market taken from the upstream v1.43 test-suite (test-suite/fdcir.cpp):
    // speed = 1.2188 + 0.02438*(-0.5726) = 1.0792..., level rescaled likewise.
    // The CIR volatility there is DELIBERATELY small (0.02438): the 2-D ADI
    // splitting on a coarse short-rate axis is unstable for large cirSigma, and
    // a probe pinning an exploded value would pin amplified round-off rather
    // than a price. Only the grids and rho are ours.
    const Real speed = 1.0792;
    const Real level = 0.0240;
    const Real cirSigma = 0.02438;
    const Real r0 = 0.06;

    cirCase("cir_european_put_rho_small", Option::Put, 40.0, european,
            "European", {kMaturity}, 36.0, 0.06, 0.0, 0.20, speed, level,
            cirSigma, r0, 0.00789, 10, 60, 30, 0);
    cirCase("cir_european_call_rho0", Option::Call, 40.0, european, "European",
            {kMaturity}, 36.0, 0.06, 0.0, 0.20, speed, level, cirSigma, r0, 0.0,
            10, 60, 30, 0);
    // |rho| = 0.15 keeps the ADI splitting comfortably inside its stability
    // region on these grids; at 0.25 the amplification of round-off already
    // reaches ~3e-11 relative and at 0.40 the scheme diverges outright -- see
    // cirInstabilityCases() below for the evidence. The mixed term still
    // clearly bites: rho = -0.15 and +0.15 straddle the rho = 0 price by
    // ~0.035 each way, four orders of magnitude above any tolerance here.
    cirCase("cir_european_call_rho_neg", Option::Call, 40.0, european,
            "European", {kMaturity}, 36.0, 0.06, 0.0, 0.20, speed, level,
            cirSigma, r0, -0.15, 10, 60, 30, 0);
    cirCase("cir_european_put_rho_pos", Option::Put, 40.0, european, "European",
            {kMaturity}, 36.0, 0.06, 0.0, 0.20, speed, level, cirSigma, r0,
            0.15, 10, 60, 30, 0);
    cirCase("cir_american_put_rho_neg", Option::Put, 40.0, american, "American",
            {kToday, kMaturity}, 36.0, 0.06, 0.0, 0.20, speed, level, cirSigma,
            r0, -0.15, 10, 60, 30, 0);
}

// UPSTREAM FINDING, pinned deliberately.
//
// FdCIRVanillaEngine with its own default scheme (ModifiedHundsdorfer) is
// numerically UNSTABLE once the mixed derivative term is large, and REFINING
// the time grid makes it worse -- the signature of an unstable splitting, not
// of discretisation error. On the market above (36/40, r = 6%, vol = 20%,
// cirSigma = 0.02438) with xGrid = 60, rGrid = 30:
//
//     rho = +0.40, tGrid =  10  ->      4.2996...      (plausible)
//     rho = +0.40, tGrid =  25  ->   -626200.66...
//     rho = +0.40, tGrid =  50  ->      9.05e+19
//     rho = +0.40, tGrid = 100  ->      3.41e+37
//
// The consequence for cross-validation is concrete: near that edge a 1e-14
// perturbation in the operator is amplified to ~1e-9 in the price over ten
// steps, so a *correct* port cannot reproduce the tGrid = 10 value at TIGHT.
// The priced cases above therefore stay at |rho| <= 0.25, and the divergence
// itself is pinned here as an ORDER OF MAGNITUDE, not as a value: a port must
// blow up the same way, but no port should be asked to match 3.41e+37 bit for
// bit.
void cirInstabilityCases() {
    const Real speed = 1.0792, level = 0.0240, cirSigma = 0.02438, r0 = 0.06;
    const auto bsProcess = bsmProcess(36.0, 0.06, 0.0, 0.20);
    const auto cirProcess =
        ext::make_shared<CoxIngersollRossProcess>(speed, cirSigma, r0, level);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 40.0);
    const auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    Obj in = marketInputs(36.0, 0.06, 0.0, 0.20);
    in.s("option_type", "Put")
        .n("strike", 40.0)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .n("cir_speed", speed)
        .n("cir_mean", level)
        .n("cir_sigma", cirSigma)
        .n("cir_x0", r0)
        .n("rho", 0.40)
        .i("x_grid", 60)
        .i("r_grid", 30)
        .s("scheme", "ModifiedHundsdorfer");

    Obj ex;
    std::vector<Real> npvs;
    const std::vector<Size> tGrids = {10, 25, 50, 100};
    for (Size tg : tGrids) {
        VanillaOption option(payoff, exercise);
        option.setPricingEngine(ext::make_shared<FdCIRVanillaEngine>(
            cirProcess, bsProcess, tg, Size(60), Size(30), Size(0), 0.40,
            FdmSchemeDesc::ModifiedHundsdorfer()));
        npvs.push_back(option.NPV());
    }
    ex.iv("t_grids", tGrids).nv("npv_by_t_grid", npvs);
    ex.b("diverges_with_refinement", std::fabs(npvs.back()) > 1e30);
    addCase("cir_unstable_large_rho", in, ex);
}

void cirMakeCases() {
    const Real spot = 36.0, strike = 40.0;
    const Rate r = 0.06, q = 0.0;
    const Volatility vol = 0.20;
    const Real speed = 1.0792, level = 0.0240, cirSigma = 0.02438, r0 = 0.06;
    const Real rho = 0.00789;

    const auto bsProcess = bsmProcess(spot, r, q, vol);
    const auto cirProcess =
        ext::make_shared<CoxIngersollRossProcess>(speed, cirSigma, r0, level);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    // Builder defaults: tGrid = 10, xGrid = 100, rGrid = 100,
    // dampingSteps = 0, schemeDesc = ModifiedHundsdorfer. Only the grids are
    // narrowed, so the default scheme and tGrid stay under test.
    {
        VanillaOption option(payoff, exercise);
        option.setPricingEngine(MakeFdCIRVanillaEngine(cirProcess, bsProcess, rho)
                                    .withXGrid(60)
                                    .withRGrid(30));
        Obj in = marketInputs(spot, r, q, vol);
        in.s("option_type", "Put")
            .n("strike", strike)
            .s("exercise", "European")
            .dv("exercise_dates", {kMaturity})
            .n("cir_speed", speed)
            .n("cir_mean", level)
            .n("cir_sigma", cirSigma)
            .n("cir_x0", r0)
            .n("rho", rho)
            .i("x_grid", 60)
            .i("r_grid", 30);
        Obj ex;
        greeks(ex, option);
        ex.i("default_t_grid", 10)
            .i("default_x_grid", 100)
            .i("default_r_grid", 100)
            .i("default_damping_steps", 0)
            .s("default_scheme", "ModifiedHundsdorfer");
        addCase("cir_make_defaults", in, ex);
    }
}

// The CIR 2-D operator, opened up. `cir_*_rho_neg` is the only family whose
// value moved when the mixed derivative term was switched on, so the operator
// itself is pinned piece by piece: the two mesh axes, then apply / apply_mixed
// / apply_direction / solve_splitting on a deterministic probe vector. A port
// that gets the price wrong at rho != 0 but right at rho == 0 can be localised
// with these alone.
void cirOperatorDiagnostics() {
    const Real spot = 36.0, strike = 40.0;
    const Rate r = 0.06, q = 0.0;
    const Volatility vol = 0.20;
    const Real speed = 1.0792, level = 0.0240, cirSigma = 0.02438, r0 = 0.06;
    const Real rho = -0.40;
    const Size xGrid = 6, rGrid = 5, tGrid = 10;

    const auto bsProcess = bsmProcess(spot, r, q, vol);
    const auto cirProcess =
        ext::make_shared<CoxIngersollRossProcess>(speed, cirSigma, r0, level);

    const Time maturity = bsProcess->time(kMaturity);

    const ext::shared_ptr<Fdm1dMesher> shortRateMesher(
        new FdmSimpleProcess1dMesher(rGrid, cirProcess, maturity, tGrid));
    const ext::shared_ptr<Fdm1dMesher> equityMesher(new FdmBlackScholesMesher(
        xGrid, bsProcess, maturity, strike, Null<Real>(), Null<Real>(), 0.0001,
        1.5, std::pair<Real, Real>(strike, 0.1), DividendSchedule(),
        ext::shared_ptr<FdmQuantoHelper>(), 0.0));

    const ext::shared_ptr<FdmMesher> mesher(
        new FdmMesherComposite(equityMesher, shortRateMesher));

    FdmCIROp op(mesher, cirProcess, bsProcess, rho, strike);
    const Time t1 = 0.25, t2 = 0.35;
    op.setTime(t1, t2);

    const Size n = mesher->layout()->size();
    Array u(n);
    for (Size i = 0; i < n; ++i)
        u[i] = std::sin(0.7 * static_cast<Real>(i) + 0.3) + 1.5;

    Obj in = marketInputs(spot, r, q, vol);
    in.n("strike", strike)
        .n("cir_speed", speed)
        .n("cir_mean", level)
        .n("cir_sigma", cirSigma)
        .n("cir_x0", r0)
        .n("rho", rho)
        .i("x_grid", static_cast<long long>(xGrid))
        .i("r_grid", static_cast<long long>(rGrid))
        .i("t_grid", static_cast<long long>(tGrid))
        .d("maturity_date", kMaturity)
        .n("t1", t1)
        .n("t2", t2);

    Obj ex;
    std::vector<Real> ex0(equityMesher->locations().begin(),
                          equityMesher->locations().end());
    std::vector<Real> rx0(shortRateMesher->locations().begin(),
                          shortRateMesher->locations().end());
    ex.nv("equity_locations", ex0).nv("rate_locations", rx0);

    const Array au = op.apply(u);
    const Array am = op.apply_mixed(u);
    const Array a0 = op.apply_direction(0, u);
    const Array a1 = op.apply_direction(1, u);
    const Array s0 = op.solve_splitting(0, u, -0.1);
    const Array s1 = op.solve_splitting(1, u, -0.1);
    ex.nv("u", std::vector<Real>(u.begin(), u.end()))
        .nv("apply", std::vector<Real>(au.begin(), au.end()))
        .nv("apply_mixed", std::vector<Real>(am.begin(), am.end()))
        .nv("apply_direction0", std::vector<Real>(a0.begin(), a0.end()))
        .nv("apply_direction1", std::vector<Real>(a1.begin(), a1.end()))
        .nv("solve_splitting0", std::vector<Real>(s0.begin(), s0.end()))
        .nv("solve_splitting1", std::vector<Real>(s1.begin(), s1.end()));
    addCase("cir_operator_diagnostics", in, ex);
}

// ---------------------------------------------------------------------------
// 5. FdSimpleBSSwingEngine
// ---------------------------------------------------------------------------
void swingCase(const std::string& name, Option::Type type, Real strike,
               Real spot, Rate r, Rate q, Volatility vol,
               const std::vector<Date>& exerciseDates, Size minRights,
               Size maxRights, Size tGrid, Size xGrid) {
    const auto process = bsmProcess(spot, r, q, vol);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
    const auto exercise = ext::make_shared<SwingExercise>(exerciseDates);

    VanillaSwingOption option(payoff, exercise, minRights, maxRights);
    option.setPricingEngine(ext::make_shared<FdSimpleBSSwingEngine>(
        process, tGrid, xGrid, FdmSchemeDesc::Douglas()));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", "Swing")
        .dv("exercise_dates", exerciseDates)
        .i("min_exercise_rights", static_cast<long long>(minRights))
        .i("max_exercise_rights", static_cast<long long>(maxRights))
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .s("scheme", "Douglas");

    Obj ex;
    ex.n("npv", option.NPV());
    addCase(name, in, ex);
}

void swingCases() {
    const std::vector<Date> dates = {kToday + 73, kToday + 146, kToday + 219,
                                     kToday + 292, kMaturity};
    // Fewer rights than dates: the swing condition genuinely chooses.
    swingCase("swing_call_2of5", Option::Call, 100.0, 100.0, 0.05, 0.0, 0.30,
              dates, 0, 2, 50, 100);
    swingCase("swing_put_3of5", Option::Put, 100.0, 100.0, 0.05, 0.0, 0.30,
              dates, 0, 3, 50, 100);
    // minExerciseRights == number of dates: every right must be used, which
    // drives the `exercisesUsed + d <= minExercises_` branch.
    swingCase("swing_rights_binding", Option::Call, 100.0, 100.0, 0.05, 0.0,
              0.30, dates, 5, 5, 50, 100);
}

// ---------------------------------------------------------------------------
// 6. FdBlackScholesAsianEngine
// ---------------------------------------------------------------------------
void asianCase(const std::string& name, Option::Type type, Real strike,
               Real spot, Rate r, Rate q, Volatility vol,
               const std::vector<Date>& fixingDates, Real runningAccumulator,
               Size pastFixings, Size tGrid, Size xGrid, Size aGrid) {
    const auto process = bsmProcess(spot, r, q, vol);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    DiscreteAveragingAsianOption option(Average::Arithmetic, runningAccumulator,
                                        pastFixings, fixingDates, payoff,
                                        exercise);
    option.setPricingEngine(ext::make_shared<FdBlackScholesAsianEngine>(
        process, tGrid, xGrid, aGrid, FdmSchemeDesc::Douglas()));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .s("average_type", "Arithmetic")
        .dv("fixing_dates", fixingDates)
        .n("running_accumulator", runningAccumulator)
        .i("past_fixings", static_cast<long long>(pastFixings))
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .i("a_grid", static_cast<long long>(aGrid))
        .s("scheme", "Douglas");

    Obj ex;
    ex.n("npv", option.NPV()).n("delta", option.delta()).n("gamma", option.gamma());
    // C++ leaves theta at Null<Real>() for this engine, so `theta()` throws.
    bool thetaThrows = false;
    std::string thetaWhat;
    try {
        option.theta();
    } catch (const std::exception& e) {
        thetaThrows = true;
        thetaWhat = e.what();
    }
    ex.b("theta_throws", thetaThrows).s("theta_what", thetaWhat);
    addCase(name, in, ex);
}

void asianCases() {
    std::vector<Date> fixings;
    for (Size i = 1; i <= 4; ++i)
        fixings.push_back(kToday + static_cast<Integer>(91 * i));

    asianCase("asian_call_no_past", Option::Call, 100.0, 100.0, 0.05, 0.0, 0.25,
              fixings, 0.0, 0, 60, 60, 40);
    asianCase("asian_put_no_past", Option::Put, 105.0, 100.0, 0.05, 0.0, 0.25,
              fixings, 0.0, 0, 60, 60, 40);
    // Running average != spot separates the two branches of the xMin/xMax
    // min()/max() that place the average axis.
    asianCase("asian_call_running", Option::Call, 100.0, 100.0, 0.05, 0.0, 0.25,
              fixings, 190.0, 2, 60, 60, 40);
}

void asianThrowCases() {
    const auto process = bsmProcess(100.0, 0.05, 0.0, 0.25);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    const std::vector<Date> fixings = {kToday + 182, kMaturity};

    Obj in = marketInputs(100.0, 0.05, 0.0, 0.25);
    in.s("option_type", "Call")
        .n("strike", 100.0)
        .s("exercise", "American")
        .dv("exercise_dates", {kToday, kMaturity})
        .dv("fixing_dates", fixings);
    addThrowCase("asian_american_rejected", in, [&] {
        DiscreteAveragingAsianOption option(
            Average::Arithmetic, 0.0, 0, fixings, payoff,
            ext::make_shared<AmericanExercise>(kToday, kMaturity));
        option.setPricingEngine(
            ext::make_shared<FdBlackScholesAsianEngine>(process, 20, 20, 10));
        option.NPV();
    });

    Obj in2 = marketInputs(100.0, 0.05, 0.0, 0.25);
    in2.s("option_type", "Call")
        .n("strike", 100.0)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .dv("fixing_dates", fixings)
        .s("average_type", "Geometric");
    addThrowCase("asian_geometric_rejected", in2, [&] {
        DiscreteAveragingAsianOption option(
            Average::Geometric, 1.0, 0, fixings, payoff,
            ext::make_shared<EuropeanExercise>(kMaturity));
        option.setPricingEngine(
            ext::make_shared<FdBlackScholesAsianEngine>(process, 20, 20, 10));
        option.NPV();
    });
}

// ---------------------------------------------------------------------------
// 7. FdBlackScholesBarrierEngine + 8. FdBlackScholesRebateEngine
// ---------------------------------------------------------------------------
std::string barrierName(Barrier::Type t) {
    switch (t) {
      case Barrier::DownIn:
        return "DownIn";
      case Barrier::UpIn:
        return "UpIn";
      case Barrier::DownOut:
        return "DownOut";
      case Barrier::UpOut:
        return "UpOut";
    }
    return "?";
}

void barrierCase(const std::string& name, Barrier::Type barrierType,
                 Real barrier, Real rebate, Option::Type type, Real strike,
                 Real spot, Rate r, Rate q, Volatility vol, Size tGrid,
                 Size xGrid, Size dampingSteps,
                 const std::vector<Date>& divDates,
                 const std::vector<Real>& divAmounts) {
    const auto process = bsmProcess(spot, r, q, vol);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    BarrierOption option(barrierType, barrier, rebate, payoff, exercise);
    option.setPricingEngine(ext::make_shared<FdBlackScholesBarrierEngine>(
        process, divSchedule(divDates, divAmounts), tGrid, xGrid, dampingSteps,
        FdmSchemeDesc::Douglas()));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("barrier_type", barrierName(barrierType))
        .n("barrier", barrier)
        .n("rebate", rebate)
        .s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .s("scheme", "Douglas")
        .dv("dividend_dates", divDates)
        .nv("dividend_amounts", divAmounts);

    Obj ex;
    greeks(ex, option);
    addCase(name, in, ex);
}

void rebateCase(const std::string& name, Barrier::Type barrierType,
                Real barrier, Real rebate, Option::Type type, Real strike,
                Real spot, Rate r, Rate q, Volatility vol, Size tGrid,
                Size xGrid, Size dampingSteps) {
    const auto process = bsmProcess(spot, r, q, vol);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    BarrierOption option(barrierType, barrier, rebate, payoff, exercise);
    option.setPricingEngine(ext::make_shared<FdBlackScholesRebateEngine>(
        process, tGrid, xGrid, dampingSteps, FdmSchemeDesc::Douglas()));

    Obj in = marketInputs(spot, r, q, vol);
    in.s("barrier_type", barrierName(barrierType))
        .n("barrier", barrier)
        .n("rebate", rebate)
        .s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity})
        .i("t_grid", static_cast<long long>(tGrid))
        .i("x_grid", static_cast<long long>(xGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .s("scheme", "Douglas");

    Obj ex;
    greeks(ex, option);
    addCase(name, in, ex);
}

void barrierCases() {
    const std::vector<Date> noDates;
    const std::vector<Real> noAmounts;

    // Knock-outs: the barrier genuinely clips the mesh and the Dirichlet face
    // pins the rebate.
    barrierCase("barrier_downout_call", Barrier::DownOut, 90.0, 0.0,
                Option::Call, 100.0, 100.0, 0.05, 0.02, 0.25, 100, 100, 0,
                noDates, noAmounts);
    barrierCase("barrier_downout_call_rebate", Barrier::DownOut, 90.0, 3.0,
                Option::Call, 100.0, 100.0, 0.05, 0.02, 0.25, 100, 100, 0,
                noDates, noAmounts);
    barrierCase("barrier_upout_put_rebate", Barrier::UpOut, 120.0, 2.5,
                Option::Put, 100.0, 100.0, 0.05, 0.02, 0.25, 100, 100, 0,
                noDates, noAmounts);
    // Knock-ins: the in-out parity assembly, including the coarse rebate mesh.
    barrierCase("barrier_downin_call_rebate", Barrier::DownIn, 90.0, 3.0,
                Option::Call, 100.0, 100.0, 0.05, 0.02, 0.25, 100, 100, 0,
                noDates, noAmounts);
    // dampingSteps = 5 -> rebate leg gets min(Size(1), 5/2) == 1 damping step.
    barrierCase("barrier_downin_damped", Barrier::DownIn, 90.0, 3.0,
                Option::Call, 100.0, 100.0, 0.05, 0.02, 0.25, 100, 100, 5,
                noDates, noAmounts);
    barrierCase("barrier_upin_put_rebate", Barrier::UpIn, 120.0, 2.5,
                Option::Put, 100.0, 100.0, 0.05, 0.02, 0.25, 100, 100, 0,
                noDates, noAmounts);
    // xGrid/5 = 12 < 50, so the rebate leg is forced to the 50-point floor.
    barrierCase("barrier_downin_small_grid", Barrier::DownIn, 90.0, 3.0,
                Option::Call, 100.0, 100.0, 0.05, 0.02, 0.25, 60, 60, 0,
                noDates, noAmounts);
    barrierCase("barrier_downout_dividends", Barrier::DownOut, 85.0, 1.0,
                Option::Call, 100.0, 100.0, 0.05, 0.0, 0.25, 80, 90, 0,
                {kToday + 120, kToday + 300}, {2.0, 2.5});

    rebateCase("rebate_downout", Barrier::DownOut, 90.0, 5.0, Option::Call,
               100.0, 100.0, 0.05, 0.02, 0.25, 100, 100, 0);
    rebateCase("rebate_upout", Barrier::UpOut, 120.0, 5.0, Option::Put, 100.0,
               100.0, 0.05, 0.02, 0.25, 100, 100, 0);
    rebateCase("rebate_downin", Barrier::DownIn, 90.0, 5.0, Option::Call, 100.0,
               100.0, 0.05, 0.02, 0.25, 60, 50, 1);
}

void barrierThrowCases() {
    const auto process = bsmProcess(100.0, 0.05, 0.02, 0.25);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    const auto european = ext::make_shared<EuropeanExercise>(kMaturity);

    Obj in = marketInputs(100.0, 0.05, 0.02, 0.25);
    in.s("barrier_type", "DownOut")
        .n("barrier", 110.0)
        .n("rebate", 0.0)
        .n("strike", 100.0)
        .s("option_type", "Call")
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity});
    addThrowCase("barrier_triggered_rejected", in, [&] {
        BarrierOption option(Barrier::DownOut, 110.0, 0.0, payoff, european);
        option.setPricingEngine(
            ext::make_shared<FdBlackScholesBarrierEngine>(process, 20, 20));
        option.NPV();
    });

    Obj in2 = marketInputs(100.0, 0.05, 0.02, 0.25);
    in2.s("barrier_type", "DownOut")
        .n("barrier", 90.0)
        .n("rebate", 0.0)
        .n("strike", 100.0)
        .s("option_type", "Call")
        .s("exercise", "American")
        .dv("exercise_dates", {kToday, kMaturity});
    addThrowCase("barrier_american_rejected", in2, [&] {
        BarrierOption option(Barrier::DownOut, 90.0, 0.0, payoff,
                             ext::make_shared<AmericanExercise>(kToday, kMaturity));
        option.setPricingEngine(
            ext::make_shared<FdBlackScholesBarrierEngine>(process, 20, 20));
        option.NPV();
    });

    Obj in3 = marketInputs(100.0, 0.05, 0.02, 0.25);
    in3.s("barrier_type", "DownOut")
        .n("barrier", 90.0)
        .n("rebate", 0.0)
        .n("strike", 0.0)
        .s("option_type", "Call")
        .s("exercise", "European")
        .dv("exercise_dates", {kMaturity});
    addThrowCase("barrier_zero_strike_rejected", in3, [&] {
        BarrierOption option(
            Barrier::DownOut, 90.0, 0.0,
            ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0), european);
        option.setPricingEngine(
            ext::make_shared<FdBlackScholesBarrierEngine>(process, 20, 20));
        option.NPV();
    });

    Obj in4 = marketInputs(100.0, 0.05, 0.02, 0.25);
    in4.s("barrier_type", "DownOut")
        .n("barrier", 90.0)
        .n("rebate", 5.0)
        .n("strike", 100.0)
        .s("option_type", "Call")
        .s("exercise", "American")
        .dv("exercise_dates", {kToday, kMaturity});
    addThrowCase("rebate_american_rejected", in4, [&] {
        BarrierOption option(Barrier::DownOut, 90.0, 5.0, payoff,
                             ext::make_shared<AmericanExercise>(kToday, kMaturity));
        option.setPricingEngine(
            ext::make_shared<FdBlackScholesRebateEngine>(process, 20, 20));
        option.NPV();
    });
}

// ---------------------------------------------------------------------------
// 9. Fd2dBlackScholesVanillaEngine
// ---------------------------------------------------------------------------
ext::shared_ptr<BasketPayoff> basketPayoff(const std::string& kind,
                                           Option::Type type, Real strike,
                                           Size n) {
    const auto base = ext::make_shared<PlainVanillaPayoff>(type, strike);
    if (kind == "Max")
        return ext::make_shared<MaxBasketPayoff>(base);
    if (kind == "Min")
        return ext::make_shared<MinBasketPayoff>(base);
    if (kind == "Spread")
        return ext::make_shared<SpreadBasketPayoff>(base);
    return ext::make_shared<AverageBasketPayoff>(base, n);
}

void fd2dCase(const std::string& name, const std::string& payoffKind,
              Option::Type type, Real strike, Real s1, Real s2, Rate r,
              Rate q1, Rate q2, Volatility v1, Volatility v2, Real correlation,
              Size xGrid, Size yGrid, Size tGrid, Size dampingSteps,
              const ext::shared_ptr<Exercise>& exercise,
              const std::string& exerciseKind,
              const std::vector<Date>& exerciseDates) {
    const auto p1 = bsmProcess(s1, r, q1, v1);
    const auto p2 = bsmProcess(s2, r, q2, v2);
    const auto payoff = basketPayoff(payoffKind, type, strike, 2);

    BasketOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<Fd2dBlackScholesVanillaEngine>(
        p1, p2, correlation, xGrid, yGrid, tGrid, dampingSteps,
        FdmSchemeDesc::Hundsdorfer()));

    Obj in;
    in.d("evaluation_date", kToday)
        .s("day_counter", "Actual365Fixed")
        .s("calendar", "NullCalendar")
        .n("spot1", s1)
        .n("spot2", s2)
        .n("risk_free_rate", r)
        .n("dividend_yield1", q1)
        .n("dividend_yield2", q2)
        .n("volatility1", v1)
        .n("volatility2", v2)
        .n("correlation", correlation)
        .s("payoff_kind", payoffKind)
        .s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", exerciseKind)
        .dv("exercise_dates", exerciseDates)
        .i("x_grid", static_cast<long long>(xGrid))
        .i("y_grid", static_cast<long long>(yGrid))
        .i("t_grid", static_cast<long long>(tGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .s("scheme", "Hundsdorfer");

    Obj ex;
    ex.n("npv", option.NPV())
        .n("delta", option.delta())
        .n("gamma", option.gamma())
        .n("theta", option.theta());
    addCase(name, in, ex);
}

void fd2dCases() {
    const auto european = ext::shared_ptr<Exercise>(
        ext::make_shared<EuropeanExercise>(kMaturity));
    const auto american = ext::shared_ptr<Exercise>(
        ext::make_shared<AmericanExercise>(kToday, kMaturity));
    const std::vector<Date> bermudanDates = {kToday + 182, kMaturity};
    const auto bermudan = ext::shared_ptr<Exercise>(
        ext::make_shared<BermudanExercise>(bermudanDates));

    // rho = 0 and rho != 0 differ only through the mixed derivative term.
    fd2dCase("fd2d_max_call_rho0", "Max", Option::Call, 100.0, 100.0, 100.0,
             0.05, 0.0, 0.0, 0.20, 0.30, 0.0, 40, 40, 25, 0, european,
             "European", {kMaturity});
    fd2dCase("fd2d_max_call_rho_pos", "Max", Option::Call, 100.0, 100.0, 100.0,
             0.05, 0.0, 0.0, 0.20, 0.30, 0.60, 40, 40, 25, 0, european,
             "European", {kMaturity});
    fd2dCase("fd2d_min_put_rho_neg", "Min", Option::Put, 100.0, 100.0, 100.0,
             0.05, 0.0, 0.0, 0.20, 0.30, -0.50, 40, 40, 25, 0, european,
             "European", {kMaturity});
    fd2dCase("fd2d_spread_call", "Spread", Option::Call, 5.0, 100.0, 95.0, 0.05,
             0.02, 0.03, 0.20, 0.25, 0.35, 40, 40, 25, 0, european, "European",
             {kMaturity});
    fd2dCase("fd2d_american_max_put", "Max", Option::Put, 100.0, 100.0, 100.0,
             0.05, 0.0, 0.0, 0.20, 0.30, 0.30, 40, 40, 25, 0, american,
             "American", {kToday, kMaturity});
    fd2dCase("fd2d_bermudan_avg_call", "Average", Option::Call, 100.0, 100.0,
             100.0, 0.05, 0.0, 0.0, 0.20, 0.30, 0.30, 40, 40, 25, 2, bermudan,
             "Bermudan", bermudanDates);
}

// ---------------------------------------------------------------------------
// 10. FdndimBlackScholesVanillaEngine
// ---------------------------------------------------------------------------
void fdndimCase(const std::string& name, const std::string& payoffKind,
                Option::Type type, Real strike,
                const std::vector<Real>& spots, Rate r,
                const std::vector<Real>& divYields,
                const std::vector<Real>& vols, const Matrix& rho,
                const std::vector<Size>& xGrids, Size tGrid, Size dampingSteps,
                const ext::shared_ptr<Exercise>& exercise,
                const std::string& exerciseKind,
                const std::vector<Date>& exerciseDates, bool autoScale,
                Size autoXGrid) {
    std::vector<ext::shared_ptr<GeneralizedBlackScholesProcess>> processes;
    for (Size i = 0; i < spots.size(); ++i)
        processes.push_back(bsmProcess(spots[i], r, divYields[i], vols[i]));

    const auto payoff = basketPayoff(payoffKind, type, strike, spots.size());

    BasketOption option(payoff, exercise);
    if (autoScale)
        option.setPricingEngine(
            ext::make_shared<FdndimBlackScholesVanillaEngine>(
                processes, rho, autoXGrid, tGrid, dampingSteps,
                FdmSchemeDesc::Douglas()));
    else
        option.setPricingEngine(
            ext::make_shared<FdndimBlackScholesVanillaEngine>(
                processes, rho, xGrids, tGrid, dampingSteps,
                FdmSchemeDesc::Douglas()));

    Obj in;
    in.d("evaluation_date", kToday)
        .s("day_counter", "Actual365Fixed")
        .s("calendar", "NullCalendar")
        .nv("spots", spots)
        .n("risk_free_rate", r)
        .nv("dividend_yields", divYields)
        .nv("volatilities", vols)
        .mat("rho", rho)
        .s("payoff_kind", payoffKind)
        .s("option_type", type == Option::Call ? "Call" : "Put")
        .n("strike", strike)
        .s("exercise", exerciseKind)
        .dv("exercise_dates", exerciseDates)
        .b("auto_scale_grids", autoScale)
        .i("auto_x_grid", static_cast<long long>(autoXGrid))
        .iv("x_grids", xGrids)
        .i("t_grid", static_cast<long long>(tGrid))
        .i("damping_steps", static_cast<long long>(dampingSteps))
        .s("scheme", "Douglas");

    Obj ex;
    ex.n("npv", option.NPV());
    // C++ fills value only; delta/gamma/theta stay Null<Real>() and throw.
    bool deltaThrows = false;
    std::string deltaWhat;
    try {
        option.delta();
    } catch (const std::exception& e) {
        deltaThrows = true;
        deltaWhat = e.what();
    }
    ex.b("delta_throws", deltaThrows).s("delta_what", deltaWhat);
    addCase(name, in, ex);
}

Matrix corr2(Real rho) {
    Matrix m(2, 2);
    m[0][0] = m[1][1] = 1.0;
    m[0][1] = m[1][0] = rho;
    return m;
}

Matrix corr3(Real r01, Real r02, Real r12) {
    Matrix m(3, 3);
    m[0][0] = m[1][1] = m[2][2] = 1.0;
    m[0][1] = m[1][0] = r01;
    m[0][2] = m[2][0] = r02;
    m[1][2] = m[2][1] = r12;
    return m;
}

void fdndimCases() {
    const auto european = ext::shared_ptr<Exercise>(
        ext::make_shared<EuropeanExercise>(kMaturity));
    const auto american = ext::shared_ptr<Exercise>(
        ext::make_shared<AmericanExercise>(kToday, kMaturity));

    fdndimCase("fdndim_2d_max_call", "Max", Option::Call, 100.0, {100.0, 100.0},
               0.05, {0.0, 0.0}, {0.20, 0.30}, corr2(0.50), {20, 20}, 25, 0,
               european, "European", {kMaturity}, false, 0);
    fdndimCase("fdndim_2d_min_put", "Min", Option::Put, 100.0, {100.0, 95.0},
               0.05, {0.01, 0.02}, {0.20, 0.30}, corr2(-0.30), {20, 20}, 25, 0,
               european, "European", {kMaturity}, false, 0);
    // n = 3, the case the brief calls for.
    fdndimCase("fdndim_3d_avg_call", "Average", Option::Call, 100.0,
               {100.0, 100.0, 100.0}, 0.05, {0.0, 0.0, 0.0},
               {0.20, 0.25, 0.30}, corr3(0.4, 0.3, 0.2), {12, 12, 12}, 20, 0,
               european, "European", {kMaturity}, false, 0);
    // American: the discount curve goes into the operator and there is NO
    // final discount-factor multiplication.
    fdndimCase("fdndim_2d_american_max_put", "Max", Option::Put, 100.0,
               {100.0, 100.0}, 0.05, {0.0, 0.0}, {0.20, 0.30}, corr2(0.50),
               {20, 20}, 25, 0, american, "American", {kToday, kMaturity},
               false, 0);
    // Auto-scaled grids: xGrid_i = max(4, xGrid * (l_i/l_0)^0.1).
    fdndimCase("fdndim_2d_autoscaled", "Max", Option::Call, 100.0,
               {100.0, 100.0}, 0.05, {0.0, 0.0}, {0.20, 0.30}, corr2(0.50), {},
               25, 0, european, "European", {kMaturity}, true, 20);
}

void fdndimThrowCases() {
    std::vector<ext::shared_ptr<GeneralizedBlackScholesProcess>> five;
    for (Size i = 0; i < 5; ++i)
        five.push_back(bsmProcess(100.0, 0.05, 0.0, 0.20));
    Matrix rho5(5, 5, 0.2);
    for (Size i = 0; i < 5; ++i)
        rho5[i][i] = 1.0;

    Obj in;
    in.d("evaluation_date", kToday).i("n_processes", 5).i("max_supported_dim", 4);
    addThrowCase("fdndim_too_many_underlyings", in, [&] {
        BasketOption option(
            basketPayoff("Max", Option::Call, 100.0, 5),
            ext::make_shared<EuropeanExercise>(kMaturity));
        option.setPricingEngine(
            ext::make_shared<FdndimBlackScholesVanillaEngine>(
                five, rho5, Size(8), Size(5)));
        option.NPV();
    });

    Obj in2;
    in2.d("evaluation_date", kToday).i("n_processes", 2).i("rho_rows", 3);
    addThrowCase("fdndim_wrong_correlation_size", in2, [&] {
        std::vector<ext::shared_ptr<GeneralizedBlackScholesProcess>> two = {
            bsmProcess(100.0, 0.05, 0.0, 0.20),
            bsmProcess(100.0, 0.05, 0.0, 0.30)};
        Matrix rho3(3, 3, 0.0);
        FdndimBlackScholesVanillaEngine engine(two, rho3, Size(8), Size(5));
    });

    Obj in3;
    in3.d("evaluation_date", kToday).i("n_processes", 0);
    addThrowCase("fdndim_no_processes", in3, [&] {
        std::vector<ext::shared_ptr<GeneralizedBlackScholesProcess>> none;
        Matrix rho0(0, 0);
        FdndimBlackScholesVanillaEngine engine(none, rho0, Size(8), Size(5));
    });
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    vanillaCases();
    vanillaQuantoCase();
    vanillaMakeCases();
    vanillaThrowCases();
    shoutCases();
    cevCases();
    cirCases();
    cirMakeCases();
    cirInstabilityCases();
    cirOperatorDiagnostics();
    swingCases();
    asianCases();
    asianThrowCases();
    barrierCases();
    barrierThrowCases();
    fd2dCases();
    fdndimCases();
    fdndimThrowCases();

    emitDocument();
    return 0;
}
