// migration-harness/cpp/probes/v143_pe_fdheston/probe.cpp
//
// Reference values for the finite-difference Heston / SABR / short-rate
// pricing-engine cluster of C++ QuantLib v1.43:
//
//   * FdHestonVanillaEngine / MakeFdHestonVanillaEngine
//         ql/pricingengines/vanilla/fdhestonvanillaengine.{hpp,cpp}
//   * FdBatesVanillaEngine
//         ql/pricingengines/vanilla/fdbatesvanillaengine.{hpp,cpp}
//   * FdHestonHullWhiteVanillaEngine
//         ql/pricingengines/vanilla/fdhestonhullwhitevanillaengine.{hpp,cpp}
//   * FdSabrVanillaEngine
//         ql/pricingengines/vanilla/fdsabrvanillaengine.{hpp,cpp}
//   * FdHestonBarrierEngine
//         ql/pricingengines/barrier/fdhestonbarrierengine.{hpp,cpp}
//   * FdHestonDoubleBarrierEngine
//         ql/pricingengines/barrier/fdhestondoublebarrierengine.{hpp,cpp}
//   * FdHestonRebateEngine
//         ql/pricingengines/barrier/fdhestonrebateengine.{hpp,cpp}
//   * FdG2SwaptionEngine
//         ql/pricingengines/swaption/fdg2swaptionengine.{hpp,cpp}
//   * FdHullWhiteSwaptionEngine
//         ql/pricingengines/swaption/fdhullwhiteswaptionengine.{hpp,cpp}
//
// Why the values below are EXACT, not a band
// ------------------------------------------
// Every engine here is a *deterministic* backward finite-difference rollback.
// Given the same mesher locations, the same operator coefficients and the same
// ADI splitting, the answer is a fixed sequence of BLAS-level tridiagonal
// solves: no RNG, no clock, no iteration-to-tolerance anywhere in the path
// (the one root-finder involved, InverseNonCentralCumulativeChiSquare inside
// FdmHestonVarianceMesher, is itself deterministic and bounded). A correct
// port therefore reproduces every number to ~1e-14 relative, so the tier is
// TIGHT. A band would pass with the wrong mesher concentration, the wrong
// damping-step count, or a dropped boundary condition -- which is the whole
// failure mode this file exists to catch.
//
// What has to be pinned, and why
// ------------------------------
//  1. `FdHestonVanillaEngine::getSolverDesc` builds the equity mesh through
//     `FdmBlackScholesMesher::processHelper(s0, dividendYield, riskFreeRate,
//     avgVolaEstimate)`. Note the ARGUMENT ORDER: processHelper's own
//     parameters are named (rTS, qTS) but it forwards them to
//     GeneralizedBlackScholesProcess(s0, /*dividendTS=*/qTS, /*riskFreeTS=*/rTS,
//     ...). Combined with the call site passing (dividendYield, riskFreeRate)
//     the helper process ends up with dividendTS = the REAL risk-free curve and
//     riskFreeTS = the REAL dividend curve, i.e. the mesher's forward walk runs
//     the drift backwards (S*exp((q-r)T) instead of S*exp((r-q)T)). That is
//     v1.43 behaviour and the grid depends on it, so it is ground truth here.
//     The `heston_vanilla_*` cases use r != q precisely so a port that "fixes"
//     the swap gets a different mesh and a different number.
//  2. The scale factor. `calculate()` calls `getSolverDesc(1.5)` but
//     `getSolverDesc(Real)` IGNORES its argument and hard-codes scaleFactor
//     2.0 for the single-strike mesher and 1.5 for the multi-strike one, while
//     `FdBatesVanillaEngine::calculate` calls `getSolverDesc(2.0)` -- also
//     ignored. A port that threads the parameter through gets a wider mesh.
//  3. `FdHestonVanillaEngine` fills delta / gamma / theta off the 2-D solver
//     (`deltaAt`, `gammaAt`, `thetaAt`), not just the value. All four are
//     pinned for every Heston vanilla / barrier / rebate case.
//  4. `enableMultipleStrikesCaching`. With a non-empty strike vector the
//     equity mesher switches to FdmBlackScholesMultiStrikeMesher (different
//     grid => different value for the SAME strike), and after the first
//     `calculate()` the engine holds a per-strike result cache keyed on
//     (exercise type, exercise dates, PlainVanillaPayoff strike+type). The
//     cached entries are NOT re-solved: they are read off the same surface at
//     a moneyness-scaled spot, with
//         value = valueAt(spot*d, v0)/d,  delta = deltaAt(spot*d, v0),
//         gamma = gammaAt(spot*d, v0)*d,  theta = thetaAt(spot*d, v0)/d,
//         d     = payoff->strike()/strikes_[i].
//     `heston_ms_*` pins the direct solve AND the two cache hits, so a port
//     that silently re-solves per strike is caught by the scaling.
//  5. `MakeFdHestonVanillaEngine` has NO validation at all -- every with*
//     method just assigns and returns *this, and `operator shared_ptr` always
//     succeeds. The defaults are tGrid=100, xGrid=100, vGrid=50,
//     dampingSteps=0, schemeDesc=Hundsdorfer, no leverage, no quanto, no
//     dividends. `make_defaults` prices at those defaults; `make_all_withs`
//     sets every knob and must equal the directly-constructed engine.
//  6. `FdHestonBarrierEngine` installs an FdmDirichletBoundary at the barrier
//     (Lower for Down*, Upper for Up*) holding `arguments_.rebate`, and for
//     the *In* variants computes vanilla + rebate - knockout, where the rebate
//     leg is a separate `FdHestonRebateEngine` priced on a COARSER grid:
//         xGrid = max(20, xGrid/4), vGrid = max(10, vGrid/4),
//         dampingSteps = (dampingSteps > 0) ? min(Size(1), dampingSteps/2) : 0
//     -- note `min`, not `max`, so 4 damping steps become 1, not 2. All four
//     results (value/delta/gamma/theta) go through that in-out parity, so a
//     port that drops the rebate leg is caught on every one of them.
//  7. `FdHestonDoubleBarrierEngine` requires KnockOut and European exercise
//     and pins BOTH Dirichlet faces to `arguments_.rebate`. Its equity mesher
//     uses the SHORT FdmBlackScholesMesher overload (no eps / scaleFactor /
//     cPoint / dividends), i.e. eps=1e-4, scaleFactor=1.5, no concentration --
//     different from the single-barrier engine, which passes 0.0001/1.5 and
//     an explicitly null cPoint.
//  8. `FdHestonRebateEngine` prices a CashOrNothingPayoff(Call, 0.0, rebate)
//     -- strike 0, so it pays the rebate everywhere the barrier has not been
//     hit -- through FdmLogInnerValue, with the same Dirichlet face as the
//     barrier engine.
//  9. `FdHestonHullWhiteVanillaEngine` is 3-D (equity, variance, short rate)
//     and its greeks are BUMPED, not read off the spline:
//         deltaAt(spot, v0, 0, spot*0.01), gammaAt(..., spot*0.01).
//     With `controlVariate` on it adds AnalyticHestonEngine(model,164) minus a
//     2-D FdHestonVanillaEngine at the SAME tGrid/xGrid/vGrid/dampingSteps and
//     scheme; both settings are pinned so a port cannot ignore the flag.
// 10. `FdSabrVanillaEngine` validates `validateSabrParameters(alpha, 0.5, nu,
//     rho)` -- beta is passed as the constant 0.5, NOT the real beta -- and
//     then separately requires beta < 1.0. It reads the answer at
//     `interpolateAt(f0, log(alpha))` and installs two
//     FdmDiscountDirichletBoundary faces at the CEV mesh ends carrying the
//     UNDISCOUNTED payoff there. Both boundary values and the throw are
//     pinned.
// 11. `FdG2SwaptionEngine` / `FdHullWhiteSwaptionEngine` build a SECOND model
//     of the same class on the swap's forwarding curve, require that the
//     forwarding and discounting curves share day counter and reference date,
//     and read `valueAt(0,0)` / `valueAt(0)`. Payer, receiver and a Bermudan
//     schedule are pinned for each, plus both QL_REQUIRE messages.
//
// Grid sizes are deliberately small: the quantity being cross-validated is the
// engine wiring, not the discretisation error, and a small grid makes the
// Python side runnable inside the normal test suite. Every case that has a
// larger twin (`make_defaults`) exists to pin the DEFAULTS, not the accuracy.
//
// Emits JSON on stdout; nothing else may be printed.

#include <cstddef>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/cashflows/dividend.hpp>
#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/doublebarrieroption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/methods/finitedifferences/meshers/fdmhestonvariancemesher.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <ql/models/equity/batesmodel.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/pricingengines/barrier/fdhestonbarrierengine.hpp>
#include <ql/pricingengines/barrier/fdhestondoublebarrierengine.hpp>
#include <ql/pricingengines/barrier/fdhestonrebateengine.hpp>
#include <ql/pricingengines/swaption/fdg2swaptionengine.hpp>
#include <ql/pricingengines/swaption/fdhullwhiteswaptionengine.hpp>
#include <ql/pricingengines/vanilla/fdbatesvanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdhestonhullwhitevanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdhestonvanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdsabrvanillaengine.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/processes/hullwhiteprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

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

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, "\"" + v + "\""); }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
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

// Generic "does it throw, and with what message" probe.
void probeThrow(const std::string& name, const std::string& subject,
                const std::string& scenario, const std::function<void()>& f) {
    Obj in;
    in.s("subject", subject);
    in.s("scenario", scenario);
    Obj ex;
    bool threw = false;
    std::string what;
    try {
        f();
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    ex.b("throws", threw);
    ex.s("what", what);
    addCase(name, in, ex);
}

// ---------------------------------------------------------------------------
// Equity market. 15 May 2025 -> 15 May 2026 is exactly 365 days, so with
// Actual365Fixed the maturity is exactly 1.0 and no day-count rounding noise
// leaks into any reference value.
// ---------------------------------------------------------------------------
const Date kToday(15, May, 2025);
const int kExpiryDays = 365;

// Heston parameters, shared by every Heston-family case.
const Real kS0 = 100.0;
const Rate kR = 0.05;
const Rate kQ = 0.02;
const Real kV0 = 0.04;
const Real kKappa = 2.0;
const Real kTheta = 0.05;
const Real kSigma = 0.40;
const Real kRho = -0.50;

// Bates jump parameters (materially non-zero intensity).
const Real kJumpLambda = 0.20;
const Real kJumpNu = -0.10;
const Real kJumpDelta = 0.15;

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, r, dayCounter()));
}

ext::shared_ptr<HestonProcess> hestonProcess() {
    return ext::make_shared<HestonProcess>(
        flatCurve(kR), flatCurve(kQ),
        Handle<Quote>(ext::make_shared<SimpleQuote>(kS0)),
        kV0, kKappa, kTheta, kSigma, kRho);
}

ext::shared_ptr<HestonModel> hestonModel() {
    return ext::make_shared<HestonModel>(hestonProcess());
}

ext::shared_ptr<BatesModel> batesModel(Real lambda) {
    auto process = ext::make_shared<BatesProcess>(
        flatCurve(kR), flatCurve(kQ),
        Handle<Quote>(ext::make_shared<SimpleQuote>(kS0)),
        kV0, kKappa, kTheta, kSigma, kRho,
        lambda, kJumpNu, kJumpDelta);
    return ext::make_shared<BatesModel>(process);
}

ext::shared_ptr<Exercise> europeanExercise(int days = kExpiryDays) {
    return ext::make_shared<EuropeanExercise>(kToday + days);
}

// Describe the Heston market so the Python test reconstructs the case rather
// than restating constants.
void describeHeston(Obj& in) {
    in.n("s0", kS0);
    in.n("r", kR);
    in.n("q", kQ);
    in.n("v0", kV0);
    in.n("kappa", kKappa);
    in.n("theta", kTheta);
    in.n("sigma", kSigma);
    in.n("rho", kRho);
    in.i("expiry_days", kExpiryDays);
    in.s("day_counter", "Actual365Fixed");
    in.s("calendar", "NullCalendar");
    in.i("eval_year", kToday.year());
    in.i("eval_month", static_cast<int>(kToday.month()));
    in.i("eval_day", kToday.dayOfMonth());
}

void describeGrid(Obj& in, long long tGrid, long long xGrid, long long vGrid,
                  long long dampingSteps, const std::string& scheme) {
    in.i("t_grid", tGrid);
    in.i("x_grid", xGrid);
    in.i("v_grid", vGrid);
    in.i("damping_steps", dampingSteps);
    in.s("scheme", scheme);
}

const FdmSchemeDesc& schemeByName(const std::string& name) {
    static const FdmSchemeDesc hundsdorfer = FdmSchemeDesc::Hundsdorfer();
    static const FdmSchemeDesc douglas = FdmSchemeDesc::Douglas();
    static const FdmSchemeDesc craigSneyd = FdmSchemeDesc::CraigSneyd();
    if (name == "douglas")
        return douglas;
    if (name == "craigsneyd")
        return craigSneyd;
    return hundsdorfer;
}

// value + the three greeks the FD engines actually fill.
void recordGreeks(Obj& ex, const OneAssetOption& opt) {
    ex.n("npv", opt.NPV());
    ex.n("delta", opt.delta());
    ex.n("gamma", opt.gamma());
    ex.n("theta", opt.theta());
}

// ===========================================================================
// 1. FdHestonVanillaEngine
// ===========================================================================

struct HestonVanillaCfg {
    const char* name;
    Option::Type type;
    Real strike;
    const char* exercise; // "european" | "american" | "bermudan"
    long long tGrid, xGrid, vGrid, dampingSteps;
    const char* scheme;
    Real mixingFactor;
};

ext::shared_ptr<Exercise> makeExercise(const std::string& kind) {
    if (kind == "american")
        return ext::make_shared<AmericanExercise>(kToday, kToday + kExpiryDays);
    if (kind == "bermudan") {
        std::vector<Date> dates = {kToday + 120, kToday + 240, kToday + kExpiryDays};
        return ext::make_shared<BermudanExercise>(dates);
    }
    return europeanExercise();
}

void runHestonVanilla(const HestonVanillaCfg& c) {
    auto engine = ext::make_shared<FdHestonVanillaEngine>(
        hestonModel(), Size(c.tGrid), Size(c.xGrid), Size(c.vGrid),
        Size(c.dampingSteps), schemeByName(c.scheme),
        ext::shared_ptr<LocalVolTermStructure>(), c.mixingFactor);

    VanillaOption opt(ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                      makeExercise(c.exercise));
    opt.setPricingEngine(engine);

    Obj in;
    describeHeston(in);
    in.s("option_type", c.type == Option::Call ? "Call" : "Put");
    in.n("strike", c.strike);
    in.s("exercise", c.exercise);
    describeGrid(in, c.tGrid, c.xGrid, c.vGrid, c.dampingSteps, c.scheme);
    in.n("mixing_factor", c.mixingFactor);

    Obj ex;
    recordGreeks(ex, opt);
    addCase(c.name, in, ex);
}

void hestonVanillaCases() {
    const HestonVanillaCfg cfgs[] = {
        {"heston_vanilla_call_atm", Option::Call, 100.0, "european", 25, 30, 12, 0,
         "hundsdorfer", 1.0},
        {"heston_vanilla_put_atm", Option::Put, 100.0, "european", 25, 30, 12, 0,
         "hundsdorfer", 1.0},
        {"heston_vanilla_call_otm", Option::Call, 130.0, "european", 25, 30, 12, 0,
         "hundsdorfer", 1.0},
        {"heston_vanilla_put_itm", Option::Put, 130.0, "european", 25, 30, 12, 0,
         "hundsdorfer", 1.0},
        // Damping steps + a different ADI splitting.
        {"heston_vanilla_call_damped_douglas", Option::Call, 100.0, "european", 25, 30,
         12, 3, "douglas", 1.0},
        {"heston_vanilla_call_craigsneyd", Option::Call, 100.0, "european", 25, 30, 12,
         0, "craigsneyd", 1.0},
        // American: exercises FdmStepConditionComposite::vanillaComposite's
        // FdmAmericanStepCondition branch (exerciseStart = 0).
        {"heston_vanilla_put_american", Option::Put, 100.0, "american", 25, 30, 12, 0,
         "hundsdorfer", 1.0},
        {"heston_vanilla_call_american", Option::Call, 100.0, "american", 25, 30, 12, 0,
         "hundsdorfer", 1.0},
        // Bermudan: the vanillaComposite Bermudan branch pushes its exercise
        // times into the composite's stopping times, which changes the solver's
        // time grid as well as applying the floor.
        {"heston_vanilla_put_bermudan", Option::Put, 100.0, "bermudan", 25, 30, 12, 0,
         "hundsdorfer", 1.0},
        // mixingFactor scales sigma inside FdmHestonVarianceMesher AND inside
        // FdmHestonOp, so it moves both the mesh and the operator.
        {"heston_vanilla_call_mixing07", Option::Call, 100.0, "european", 25, 30, 12, 0,
         "hundsdorfer", 0.7},
    };
    for (const auto& c : cfgs)
        runHestonVanilla(c);
}

// Multiple-strike caching. The first NPV() solves once on the multi-strike
// mesh; the following two are pure cache hits with the moneyness scaling.
void hestonMultiStrikeCases() {
    const std::vector<Real> strikes = {80.0, 100.0, 120.0};
    const Size tGrid = 25, xGrid = 30, vGrid = 12;

    auto engine = ext::make_shared<FdHestonVanillaEngine>(
        hestonModel(), tGrid, xGrid, vGrid, 0, FdmSchemeDesc::Hundsdorfer());
    engine->enableMultipleStrikesCaching(strikes);

    auto exercise = europeanExercise();

    Obj in;
    describeHeston(in);
    in.s("option_type", "Call");
    in.s("exercise", "european");
    describeGrid(in, tGrid, xGrid, vGrid, 0, "hundsdorfer");
    in.n("strike_0", strikes[0]);
    in.n("strike_1", strikes[1]);
    in.n("strike_2", strikes[2]);
    in.n("pricing_strike", 100.0);

    // Direct solve at the middle strike -- on the MULTI-STRIKE mesh, so this
    // is deliberately different from heston_vanilla_call_atm.
    VanillaOption opt100(ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0), exercise);
    opt100.setPricingEngine(engine);
    Obj ex;
    ex.n("npv_direct_100", opt100.NPV());
    ex.n("delta_direct_100", opt100.delta());
    ex.n("gamma_direct_100", opt100.gamma());
    ex.n("theta_direct_100", opt100.theta());

    // Cache hits. Sharing `exercise` is what makes the cache key match.
    VanillaOption opt80(ext::make_shared<PlainVanillaPayoff>(Option::Call, 80.0), exercise);
    opt80.setPricingEngine(engine);
    ex.n("npv_cached_80", opt80.NPV());
    ex.n("delta_cached_80", opt80.delta());
    ex.n("gamma_cached_80", opt80.gamma());
    ex.n("theta_cached_80", opt80.theta());

    VanillaOption opt120(ext::make_shared<PlainVanillaPayoff>(Option::Call, 120.0), exercise);
    opt120.setPricingEngine(engine);
    ex.n("npv_cached_120", opt120.NPV());
    ex.n("delta_cached_120", opt120.delta());
    ex.n("gamma_cached_120", opt120.gamma());
    ex.n("theta_cached_120", opt120.theta());

    addCase("heston_ms_cache", in, ex);

    // A strike that is NOT in the cached vector falls through to a full solve
    // on the multi-strike mesh -- and, because calculate() re-fills the cache
    // from the freshly solved surface, the previously cached entries change
    // too. Pin the miss so a port that returns a nearest-cached value fails.
    VanillaOption opt90(ext::make_shared<PlainVanillaPayoff>(Option::Call, 90.0), exercise);
    opt90.setPricingEngine(engine);
    Obj in2;
    describeHeston(in2);
    in2.s("option_type", "Call");
    in2.s("exercise", "european");
    describeGrid(in2, tGrid, xGrid, vGrid, 0, "hundsdorfer");
    in2.n("strike_0", strikes[0]);
    in2.n("strike_1", strikes[1]);
    in2.n("strike_2", strikes[2]);
    in2.n("pricing_strike", 90.0);
    Obj ex2;
    ex2.n("npv_miss_90", opt90.NPV());
    ex2.n("delta_miss_90", opt90.delta());
    ex2.n("gamma_miss_90", opt90.gamma());
    ex2.n("theta_miss_90", opt90.theta());
    addCase("heston_ms_cache_miss", in2, ex2);
}

// ===========================================================================
// 2. MakeFdHestonVanillaEngine
// ===========================================================================

void makeFdHestonCases() {
    auto exercise = europeanExercise();
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);

    // 2.1 -- pure defaults: tGrid 100, xGrid 100, vGrid 50, damping 0,
    //        Hundsdorfer, no leverage / quanto / dividends.
    {
        VanillaOption opt(payoff, exercise);
        opt.setPricingEngine(MakeFdHestonVanillaEngine(hestonModel()));
        Obj in;
        describeHeston(in);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        in.s("exercise", "european");
        describeGrid(in, 100, 100, 50, 0, "hundsdorfer");
        Obj ex;
        recordGreeks(ex, opt);
        addCase("make_defaults", in, ex);
    }

    // 2.2 -- every with* knob set; must equal the direct engine below.
    {
        VanillaOption opt(payoff, exercise);
        opt.setPricingEngine(MakeFdHestonVanillaEngine(hestonModel())
                                 .withTGrid(25)
                                 .withXGrid(30)
                                 .withVGrid(12)
                                 .withDampingSteps(3)
                                 .withFdmSchemeDesc(FdmSchemeDesc::Douglas()));
        Obj in;
        describeHeston(in);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        in.s("exercise", "european");
        describeGrid(in, 25, 30, 12, 3, "douglas");
        Obj ex;
        recordGreeks(ex, opt);
        addCase("make_all_withs", in, ex);
    }
}

// ===========================================================================
// 2b. Discrete dividends and the quanto helper
// ===========================================================================
//
// Both reach the engine through `FdmBlackScholesMesher`:
//   * a DividendSchedule adds one intermediate step per dividend inside
//     [0, maturity] to the mesher's forward walk AND installs an
//     FdmDividendHandler in the step-condition composite -- with the
//     composite ALSO pushing a second copy of every dividend time shifted by
//     +1e-5 ("smoother convergence behavior") into the stopping times, which
//     changes the solver's time grid;
//   * an FdmQuantoHelper swaps the process' dividend curve for a
//     QuantoTermStructure inside the mesher AND is forwarded to
//     FdmHestonSolver, where FdmHestonOp adds the quanto drift adjustment.
// A port that wires only one of the two halves is caught here.

const int kDiv1Days = 90;
const int kDiv2Days = 270;
const Real kDiv1Amount = 1.5;
const Real kDiv2Amount = 2.0;

DividendSchedule cashDividends() {
    return DividendVector({kToday + kDiv1Days, kToday + kDiv2Days},
                          {kDiv1Amount, kDiv2Amount});
}

void describeDividends(Obj& in) {
    in.i("dividend_1_days", kDiv1Days);
    in.i("dividend_2_days", kDiv2Days);
    in.n("dividend_1_amount", kDiv1Amount);
    in.n("dividend_2_amount", kDiv2Amount);
}

const Rate kForeignRate = 0.01;
const Volatility kFxVol = 0.15;
const Real kEquityFxCorrelation = -0.30;
const Real kExchRateAtmLevel = 1.10;

void dividendAndQuantoCases() {
    auto exercise = europeanExercise();
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);

    // 2b.1 -- direct constructor with a DividendSchedule.
    {
        auto engine = ext::make_shared<FdHestonVanillaEngine>(
            hestonModel(), cashDividends(), 25, 30, 12, 0, FdmSchemeDesc::Hundsdorfer());
        VanillaOption opt(payoff, exercise);
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        describeDividends(in);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        in.s("exercise", "european");
        describeGrid(in, 25, 30, 12, 0, "hundsdorfer");
        Obj ex;
        recordGreeks(ex, opt);
        addCase("heston_vanilla_call_dividends", in, ex);
    }

    // 2b.2 -- the same thing through MakeFdHestonVanillaEngine::withCashDividends.
    {
        VanillaOption opt(payoff, exercise);
        opt.setPricingEngine(MakeFdHestonVanillaEngine(hestonModel())
                                 .withTGrid(25)
                                 .withXGrid(30)
                                 .withVGrid(12)
                                 .withCashDividends({kToday + kDiv1Days, kToday + kDiv2Days},
                                                    {kDiv1Amount, kDiv2Amount}));
        Obj in;
        describeHeston(in);
        describeDividends(in);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        in.s("exercise", "european");
        describeGrid(in, 25, 30, 12, 0, "hundsdorfer");
        Obj ex;
        recordGreeks(ex, opt);
        addCase("make_with_cash_dividends", in, ex);
    }

    // 2b.3 -- multiple-strike caching is incompatible with discrete dividends.
    probeThrow("heston_ms_throw_dividends", "FdHestonVanillaEngine",
               "enableMultipleStrikesCaching + DividendSchedule", [&] {
                   auto engine = ext::make_shared<FdHestonVanillaEngine>(
                       hestonModel(), cashDividends(), 25, 30, 12, 0,
                       FdmSchemeDesc::Hundsdorfer());
                   engine->enableMultipleStrikesCaching({80.0, 100.0, 120.0});
                   VanillaOption opt(payoff, exercise);
                   opt.setPricingEngine(engine);
                   opt.NPV();
               });

    // 2b.4 -- the barrier engine's own (hand-rolled) dividend composite.
    {
        auto engine = ext::make_shared<FdHestonBarrierEngine>(
            hestonModel(), cashDividends(), 25, 30, 12, 0, FdmSchemeDesc::Hundsdorfer());
        BarrierOption opt(Barrier::DownOut, 80.0, 3.0, payoff, exercise);
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        describeDividends(in);
        in.s("barrier_type", "DownOut");
        in.n("barrier", 80.0);
        in.n("rebate", 3.0);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        in.s("exercise", "european");
        describeGrid(in, 25, 30, 12, 0, "hundsdorfer");
        Obj ex;
        recordGreeks(ex, opt);
        addCase("barrier_downout_call_dividends", in, ex);
    }

    // 2b.5 -- the quanto helper.
    {
        auto quantoHelper = ext::make_shared<FdmQuantoHelper>(
            ext::make_shared<FlatForward>(kToday, kR, dayCounter()),
            ext::make_shared<FlatForward>(kToday, kForeignRate, dayCounter()),
            ext::make_shared<BlackConstantVol>(kToday, NullCalendar(), kFxVol, dayCounter()),
            kEquityFxCorrelation, kExchRateAtmLevel);

        auto engine = ext::make_shared<FdHestonVanillaEngine>(
            hestonModel(), quantoHelper, 25, 30, 12, 0, FdmSchemeDesc::Hundsdorfer());
        VanillaOption opt(payoff, exercise);
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        in.n("quanto_foreign_rate", kForeignRate);
        in.n("quanto_fx_vol", kFxVol);
        in.n("quanto_equity_fx_correlation", kEquityFxCorrelation);
        in.n("quanto_exch_rate_atm_level", kExchRateAtmLevel);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        in.s("exercise", "european");
        describeGrid(in, 25, 30, 12, 0, "hundsdorfer");
        Obj ex;
        recordGreeks(ex, opt);
        addCase("heston_vanilla_call_quanto", in, ex);
    }
}

// ===========================================================================
// 3. FdBatesVanillaEngine
// ===========================================================================

void batesCases() {
    struct Cfg {
        const char* name;
        Option::Type type;
        Real strike;
        Real lambda;
        long long tGrid, xGrid, vGrid, dampingSteps;
    };
    const Cfg cfgs[] = {
        {"bates_call_atm", Option::Call, 100.0, kJumpLambda, 25, 30, 12, 0},
        {"bates_put_atm", Option::Put, 100.0, kJumpLambda, 25, 30, 12, 0},
        // A much bigger intensity: if a port drops the jump integral entirely
        // the two lambdas give the same number, and this case separates them.
        {"bates_call_atm_lambda05", Option::Call, 100.0, 0.5, 25, 30, 12, 0},
        {"bates_call_otm_damped", Option::Call, 120.0, kJumpLambda, 25, 30, 12, 2},
    };
    for (const auto& c : cfgs) {
        auto engine = ext::make_shared<FdBatesVanillaEngine>(
            batesModel(c.lambda), Size(c.tGrid), Size(c.xGrid), Size(c.vGrid),
            Size(c.dampingSteps), FdmSchemeDesc::Hundsdorfer());
        VanillaOption opt(ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                          europeanExercise());
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        in.n("jump_lambda", c.lambda);
        in.n("jump_nu", kJumpNu);
        in.n("jump_delta", kJumpDelta);
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.s("exercise", "european");
        describeGrid(in, c.tGrid, c.xGrid, c.vGrid, c.dampingSteps, "hundsdorfer");

        Obj ex;
        recordGreeks(ex, opt);
        addCase(c.name, in, ex);
    }
}

// ===========================================================================
// 4. FdHestonHullWhiteVanillaEngine
// ===========================================================================

const Real kHwA = 0.05;
const Real kHwSigma = 0.01;

void hestonHullWhiteCases() {
    struct Cfg {
        const char* name;
        Option::Type type;
        Real strike;
        Real corr;
        bool controlVariate;
        long long tGrid, xGrid, vGrid, rGrid, dampingSteps;
    };
    const Cfg cfgs[] = {
        {"hhw_call_atm_cv", Option::Call, 100.0, -0.4, true, 12, 16, 6, 5, 0},
        {"hhw_call_atm_nocv", Option::Call, 100.0, -0.4, false, 12, 16, 6, 5, 0},
        {"hhw_put_atm_nocv", Option::Put, 100.0, -0.4, false, 12, 16, 6, 5, 0},
        {"hhw_call_otm_cv_damped", Option::Call, 120.0, 0.3, true, 12, 16, 6, 5, 2},
    };
    for (const auto& c : cfgs) {
        auto hwProcess = ext::make_shared<HullWhiteProcess>(flatCurve(kR), kHwA, kHwSigma);
        auto engine = ext::make_shared<FdHestonHullWhiteVanillaEngine>(
            hestonModel(), hwProcess, c.corr, Size(c.tGrid), Size(c.xGrid),
            Size(c.vGrid), Size(c.rGrid), Size(c.dampingSteps), c.controlVariate,
            FdmSchemeDesc::Hundsdorfer());
        VanillaOption opt(ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                          europeanExercise());
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        in.n("hw_a", kHwA);
        in.n("hw_sigma", kHwSigma);
        in.n("corr_equity_short_rate", c.corr);
        in.b("control_variate", c.controlVariate);
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.s("exercise", "european");
        describeGrid(in, c.tGrid, c.xGrid, c.vGrid, c.dampingSteps, "hundsdorfer");
        in.i("r_grid", c.rGrid);

        Obj ex;
        recordGreeks(ex, opt);
        addCase(c.name, in, ex);
    }
}

// ===========================================================================
// 5. FdSabrVanillaEngine
// ===========================================================================

const Real kSabrF0 = 100.0;
const Real kSabrAlpha = 0.35;
const Real kSabrBeta = 0.80;
const Real kSabrNu = 0.50;
const Real kSabrRho = -0.40;

void sabrCases() {
    struct Cfg {
        const char* name;
        Option::Type type;
        Real strike;
        long long tGrid, fGrid, xGrid, dampingSteps;
        Real scalingFactor, eps;
    };
    const Cfg cfgs[] = {
        {"sabr_call_atm", Option::Call, 100.0, 15, 60, 12, 0, 1.0, 1e-4},
        {"sabr_put_atm", Option::Put, 100.0, 15, 60, 12, 0, 1.0, 1e-4},
        {"sabr_call_otm_damped", Option::Call, 120.0, 15, 60, 12, 2, 1.0, 1e-4},
        {"sabr_call_atm_scaled", Option::Call, 100.0, 15, 60, 12, 0, 1.5, 1e-3},
    };
    for (const auto& c : cfgs) {
        auto engine = ext::make_shared<FdSabrVanillaEngine>(
            kSabrF0, kSabrAlpha, kSabrBeta, kSabrNu, kSabrRho, flatCurve(kR),
            Size(c.tGrid), Size(c.fGrid), Size(c.xGrid), Size(c.dampingSteps),
            c.scalingFactor, c.eps, FdmSchemeDesc::Hundsdorfer());
        VanillaOption opt(ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                          europeanExercise());
        opt.setPricingEngine(engine);

        Obj in;
        in.n("f0", kSabrF0);
        in.n("alpha", kSabrAlpha);
        in.n("beta", kSabrBeta);
        in.n("nu", kSabrNu);
        in.n("rho", kSabrRho);
        in.n("r", kR);
        in.i("expiry_days", kExpiryDays);
        in.s("day_counter", "Actual365Fixed");
        in.i("eval_year", kToday.year());
        in.i("eval_month", static_cast<int>(kToday.month()));
        in.i("eval_day", kToday.dayOfMonth());
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.s("exercise", "european");
        in.i("t_grid", c.tGrid);
        in.i("f_grid", c.fGrid);
        in.i("x_grid", c.xGrid);
        in.i("damping_steps", c.dampingSteps);
        in.n("scaling_factor", c.scalingFactor);
        in.n("eps", c.eps);
        in.s("scheme", "hundsdorfer");

        Obj ex;
        // Only `value` is filled by this engine -- the greeks stay Null.
        ex.n("npv", opt.NPV());
        addCase(c.name, in, ex);
    }

    probeThrow("sabr_throw_beta_one", "FdSabrVanillaEngine", "beta == 1.0", [] {
        FdSabrVanillaEngine(kSabrF0, kSabrAlpha, 1.0, kSabrNu, kSabrRho, flatCurve(kR));
    });
    probeThrow("sabr_throw_negative_alpha", "FdSabrVanillaEngine", "alpha < 0", [] {
        FdSabrVanillaEngine(kSabrF0, -0.1, kSabrBeta, kSabrNu, kSabrRho, flatCurve(kR));
    });
    probeThrow("sabr_throw_rho_one", "FdSabrVanillaEngine", "|rho| == 1", [] {
        FdSabrVanillaEngine(kSabrF0, kSabrAlpha, kSabrBeta, kSabrNu, 1.0, flatCurve(kR));
    });
}

// ===========================================================================
// 6. FdHestonBarrierEngine / FdHestonDoubleBarrierEngine / FdHestonRebateEngine
// ===========================================================================

void barrierCases() {
    struct Cfg {
        const char* name;
        Barrier::Type barrierType;
        Real barrier;
        Real rebate;
        Option::Type type;
        Real strike;
        long long tGrid, xGrid, vGrid, dampingSteps;
    };
    const Cfg cfgs[] = {
        {"barrier_downout_call", Barrier::DownOut, 80.0, 0.0, Option::Call, 100.0, 25,
         30, 12, 0},
        {"barrier_downout_call_rebate", Barrier::DownOut, 80.0, 3.0, Option::Call, 100.0,
         25, 30, 12, 0},
        {"barrier_upout_call_rebate", Barrier::UpOut, 130.0, 2.5, Option::Call, 100.0, 25,
         30, 12, 0},
        {"barrier_upout_put", Barrier::UpOut, 130.0, 0.0, Option::Put, 100.0, 25, 30, 12,
         0},
        // In-barriers: value = vanilla + rebate - knockout, with the rebate leg
        // on the coarser grid and the min()-not-max() damping quirk.
        {"barrier_downin_call_rebate", Barrier::DownIn, 80.0, 3.0, Option::Call, 100.0,
         25, 30, 12, 0},
        {"barrier_upin_call_rebate_damped", Barrier::UpIn, 130.0, 2.5, Option::Call,
         100.0, 25, 30, 12, 4},
        {"barrier_downin_put", Barrier::DownIn, 80.0, 0.0, Option::Put, 100.0, 25, 30, 12,
         0},
    };
    for (const auto& c : cfgs) {
        auto engine = ext::make_shared<FdHestonBarrierEngine>(
            hestonModel(), Size(c.tGrid), Size(c.xGrid), Size(c.vGrid),
            Size(c.dampingSteps), FdmSchemeDesc::Hundsdorfer());
        BarrierOption opt(c.barrierType, c.barrier, c.rebate,
                          ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                          europeanExercise());
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        in.s("barrier_type", c.barrierType == Barrier::DownIn    ? "DownIn"
                             : c.barrierType == Barrier::DownOut ? "DownOut"
                             : c.barrierType == Barrier::UpIn    ? "UpIn"
                                                                 : "UpOut");
        in.n("barrier", c.barrier);
        in.n("rebate", c.rebate);
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.s("exercise", "european");
        describeGrid(in, c.tGrid, c.xGrid, c.vGrid, c.dampingSteps, "hundsdorfer");

        Obj ex;
        recordGreeks(ex, opt);
        addCase(c.name, in, ex);
    }

    probeThrow("barrier_throw_american", "FdHestonBarrierEngine", "American exercise", [] {
        auto engine = ext::make_shared<FdHestonBarrierEngine>(hestonModel(), 25, 30, 12, 0,
                                                              FdmSchemeDesc::Hundsdorfer());
        BarrierOption opt(Barrier::DownOut, 80.0, 0.0,
                          ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                          ext::make_shared<AmericanExercise>(kToday, kToday + kExpiryDays));
        opt.setPricingEngine(engine);
        opt.NPV();
    });
}

void rebateCases() {
    struct Cfg {
        const char* name;
        Barrier::Type barrierType;
        Real barrier;
        Real rebate;
        long long tGrid, xGrid, vGrid, dampingSteps;
    };
    const Cfg cfgs[] = {
        {"rebate_downout", Barrier::DownOut, 80.0, 3.0, 25, 20, 10, 0},
        {"rebate_upout", Barrier::UpOut, 130.0, 2.5, 25, 20, 10, 0},
        {"rebate_downin_damped", Barrier::DownIn, 80.0, 5.0, 25, 20, 10, 1},
    };
    for (const auto& c : cfgs) {
        auto engine = ext::make_shared<FdHestonRebateEngine>(
            hestonModel(), Size(c.tGrid), Size(c.xGrid), Size(c.vGrid),
            Size(c.dampingSteps), FdmSchemeDesc::Hundsdorfer());
        BarrierOption opt(c.barrierType, c.barrier, c.rebate,
                          ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                          europeanExercise());
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        in.s("barrier_type", c.barrierType == Barrier::DownIn    ? "DownIn"
                             : c.barrierType == Barrier::DownOut ? "DownOut"
                             : c.barrierType == Barrier::UpIn    ? "UpIn"
                                                                 : "UpOut");
        in.n("barrier", c.barrier);
        in.n("rebate", c.rebate);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        in.s("exercise", "european");
        describeGrid(in, c.tGrid, c.xGrid, c.vGrid, c.dampingSteps, "hundsdorfer");

        Obj ex;
        recordGreeks(ex, opt);
        addCase(c.name, in, ex);
    }

    probeThrow("rebate_throw_american", "FdHestonRebateEngine", "American exercise", [] {
        auto engine = ext::make_shared<FdHestonRebateEngine>(hestonModel(), 25, 20, 10, 0,
                                                             FdmSchemeDesc::Hundsdorfer());
        BarrierOption opt(Barrier::DownOut, 80.0, 3.0,
                          ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                          ext::make_shared<AmericanExercise>(kToday, kToday + kExpiryDays));
        opt.setPricingEngine(engine);
        opt.NPV();
    });
}

void doubleBarrierCases() {
    struct Cfg {
        const char* name;
        Real lo, hi, rebate;
        Option::Type type;
        Real strike;
        long long tGrid, xGrid, vGrid, dampingSteps;
    };
    const Cfg cfgs[] = {
        {"double_barrier_ko_call", 80.0, 130.0, 0.0, Option::Call, 100.0, 25, 30, 12, 0},
        {"double_barrier_ko_call_rebate", 80.0, 130.0, 2.0, Option::Call, 100.0, 25, 30,
         12, 0},
        {"double_barrier_ko_put_damped", 75.0, 125.0, 1.5, Option::Put, 100.0, 25, 30, 12,
         2},
    };
    for (const auto& c : cfgs) {
        auto engine = ext::make_shared<FdHestonDoubleBarrierEngine>(
            hestonModel(), Size(c.tGrid), Size(c.xGrid), Size(c.vGrid),
            Size(c.dampingSteps), FdmSchemeDesc::Hundsdorfer());
        DoubleBarrierOption opt(DoubleBarrier::KnockOut, c.lo, c.hi, c.rebate,
                                ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                                europeanExercise());
        opt.setPricingEngine(engine);

        Obj in;
        describeHeston(in);
        in.s("barrier_type", "KnockOut");
        in.n("barrier_lo", c.lo);
        in.n("barrier_hi", c.hi);
        in.n("rebate", c.rebate);
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.s("exercise", "european");
        describeGrid(in, c.tGrid, c.xGrid, c.vGrid, c.dampingSteps, "hundsdorfer");

        Obj ex;
        recordGreeks(ex, opt);
        addCase(c.name, in, ex);
    }

    probeThrow("double_barrier_throw_knockin", "FdHestonDoubleBarrierEngine",
               "KnockIn barrier", [] {
                   auto engine = ext::make_shared<FdHestonDoubleBarrierEngine>(
                       hestonModel(), 25, 30, 12, 0, FdmSchemeDesc::Hundsdorfer());
                   DoubleBarrierOption opt(
                       DoubleBarrier::KnockIn, 80.0, 130.0, 0.0,
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       europeanExercise());
                   opt.setPricingEngine(engine);
                   opt.NPV();
               });
    probeThrow("double_barrier_throw_american", "FdHestonDoubleBarrierEngine",
               "American exercise", [] {
                   auto engine = ext::make_shared<FdHestonDoubleBarrierEngine>(
                       hestonModel(), 25, 30, 12, 0, FdmSchemeDesc::Hundsdorfer());
                   DoubleBarrierOption opt(
                       DoubleBarrier::KnockOut, 80.0, 130.0, 0.0,
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       ext::make_shared<AmericanExercise>(kToday, kToday + kExpiryDays));
                   opt.setPricingEngine(engine);
                   opt.NPV();
               });
}

// ===========================================================================
// 7. FdG2SwaptionEngine / FdHullWhiteSwaptionEngine
// ===========================================================================
//
// Rates market: same evaluation date, TARGET calendar, Euribor6M on a 3%
// forwarding curve, 2.5% discount curve. Both curves are Actual365Fixed with
// the same reference date so the engines' QL_REQUIREs pass.

const Rate kFwdRate = 0.03;
const Rate kDiscRate = 0.025;
const Real kG2A = 0.05;
const Real kG2Sigma = 0.008;
const Real kG2B = 0.10;
const Real kG2Eta = 0.006;
const Real kG2Rho = -0.70;

Handle<YieldTermStructure> ratesCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, r, dayCounter()));
}

Date swaptionExpiry() { return TARGET().advance(kToday, 2, Years); }
Date swapStart() { return TARGET().advance(swaptionExpiry(), 2, Days); }
Date swapEnd() { return TARGET().advance(swapStart(), 3, Years); }

Schedule swapSchedule(const Period& tenor) {
    return Schedule(swapStart(), swapEnd(), tenor, TARGET(),
                    ModifiedFollowing, ModifiedFollowing, DateGeneration::Backward, false);
}

ext::shared_ptr<VanillaSwap> makeSwap(const Handle<YieldTermStructure>& fwd,
                                      Swap::Type type, Rate fixedRate) {
    return ext::make_shared<VanillaSwap>(
        type, 1.0, swapSchedule(Period(1, Years)), fixedRate,
        Thirty360(Thirty360::BondBasis), swapSchedule(Period(6, Months)),
        ext::make_shared<Euribor>(Period(6, Months), fwd), 0.0, Actual360());
}

std::vector<Date> bermudanDates() {
    return {TARGET().advance(kToday, 1, Years), TARGET().advance(kToday, 18, Months),
            swaptionExpiry()};
}

void describeRates(Obj& in) {
    in.n("fwd_rate", kFwdRate);
    in.n("disc_rate", kDiscRate);
    in.s("day_counter", "Actual365Fixed");
    in.s("calendar", "TARGET");
    in.s("index", "Euribor6M");
    in.s("fixed_day_counter", "Thirty360BondBasis");
    in.s("float_day_counter", "Actual360");
    in.n("notional", 1.0);
    in.i("expiry_years", 2);
    in.i("swap_tenor_years", 3);
    in.i("eval_year", kToday.year());
    in.i("eval_month", static_cast<int>(kToday.month()));
    in.i("eval_day", kToday.dayOfMonth());
}

void hullWhiteSwaptionCases() {
    struct Cfg {
        const char* name;
        Swap::Type type;
        Rate fixedRate;
        const char* exercise;
        long long tGrid, xGrid, dampingSteps;
        Real invEps;
    };
    const Cfg cfgs[] = {
        {"hw_swaption_payer", Swap::Payer, 0.03, "european", 20, 21, 0, 1e-5},
        {"hw_swaption_receiver", Swap::Receiver, 0.03, "european", 20, 21, 0, 1e-5},
        {"hw_swaption_payer_otm", Swap::Payer, 0.045, "european", 20, 21, 0, 1e-5},
        {"hw_swaption_payer_bermudan", Swap::Payer, 0.03, "bermudan", 20, 21, 0, 1e-5},
        {"hw_swaption_payer_damped", Swap::Payer, 0.03, "european", 20, 21, 2, 1e-4},
    };
    for (const auto& c : cfgs) {
        auto fwd = ratesCurve(kFwdRate);
        auto disc = ratesCurve(kDiscRate);
        auto model = ext::make_shared<HullWhite>(disc, kHwA, kHwSigma);
        auto swap = makeSwap(fwd, c.type, c.fixedRate);
        ext::shared_ptr<Exercise> exercise;
        if (std::string(c.exercise) == "bermudan")
            exercise = ext::make_shared<BermudanExercise>(bermudanDates());
        else
            exercise = ext::make_shared<EuropeanExercise>(swaptionExpiry());

        Swaption swaption(swap, exercise);
        swaption.setPricingEngine(ext::make_shared<FdHullWhiteSwaptionEngine>(
            model, Size(c.tGrid), Size(c.xGrid), Size(c.dampingSteps), c.invEps,
            FdmSchemeDesc::Douglas()));

        Obj in;
        describeRates(in);
        in.n("hw_a", kHwA);
        in.n("hw_sigma", kHwSigma);
        in.s("swap_type", c.type == Swap::Payer ? "Payer" : "Receiver");
        in.n("fixed_rate", c.fixedRate);
        in.s("exercise", c.exercise);
        in.i("t_grid", c.tGrid);
        in.i("x_grid", c.xGrid);
        in.i("damping_steps", c.dampingSteps);
        in.n("inv_eps", c.invEps);
        in.s("scheme", "douglas");

        Obj ex;
        ex.n("npv", swaption.NPV());
        addCase(c.name, in, ex);
    }

    probeThrow("hw_swaption_throw_daycounter", "FdHullWhiteSwaptionEngine",
               "forward/discount day counters differ", [] {
                   auto fwd = Handle<YieldTermStructure>(
                       ext::make_shared<FlatForward>(kToday, kFwdRate, Actual360()));
                   auto disc = ratesCurve(kDiscRate);
                   auto model = ext::make_shared<HullWhite>(disc, kHwA, kHwSigma);
                   Swaption swaption(makeSwap(fwd, Swap::Payer, 0.03),
                                     ext::make_shared<EuropeanExercise>(swaptionExpiry()));
                   swaption.setPricingEngine(
                       ext::make_shared<FdHullWhiteSwaptionEngine>(model, 20, 21));
                   swaption.NPV();
               });
    probeThrow("hw_swaption_throw_refdate", "FdHullWhiteSwaptionEngine",
               "forward/discount reference dates differ", [] {
                   auto fwd = Handle<YieldTermStructure>(ext::make_shared<FlatForward>(
                       kToday + 1, kFwdRate, dayCounter()));
                   auto disc = ratesCurve(kDiscRate);
                   auto model = ext::make_shared<HullWhite>(disc, kHwA, kHwSigma);
                   Swaption swaption(makeSwap(fwd, Swap::Payer, 0.03),
                                     ext::make_shared<EuropeanExercise>(swaptionExpiry()));
                   swaption.setPricingEngine(
                       ext::make_shared<FdHullWhiteSwaptionEngine>(model, 20, 21));
                   swaption.NPV();
               });
}

void g2SwaptionCases() {
    struct Cfg {
        const char* name;
        Swap::Type type;
        Rate fixedRate;
        const char* exercise;
        long long tGrid, xGrid, yGrid, dampingSteps;
        Real invEps;
    };
    const Cfg cfgs[] = {
        {"g2_swaption_payer", Swap::Payer, 0.03, "european", 15, 11, 11, 0, 1e-5},
        {"g2_swaption_receiver", Swap::Receiver, 0.03, "european", 15, 11, 11, 0, 1e-5},
        {"g2_swaption_payer_otm", Swap::Payer, 0.045, "european", 15, 11, 11, 0, 1e-5},
        {"g2_swaption_payer_bermudan", Swap::Payer, 0.03, "bermudan", 15, 11, 11, 0, 1e-5},
        {"g2_swaption_payer_damped", Swap::Payer, 0.03, "european", 15, 11, 11, 2, 1e-4},
    };
    for (const auto& c : cfgs) {
        auto fwd = ratesCurve(kFwdRate);
        auto disc = ratesCurve(kDiscRate);
        auto model = ext::make_shared<G2>(disc, kG2A, kG2Sigma, kG2B, kG2Eta, kG2Rho);
        auto swap = makeSwap(fwd, c.type, c.fixedRate);
        ext::shared_ptr<Exercise> exercise;
        if (std::string(c.exercise) == "bermudan")
            exercise = ext::make_shared<BermudanExercise>(bermudanDates());
        else
            exercise = ext::make_shared<EuropeanExercise>(swaptionExpiry());

        Swaption swaption(swap, exercise);
        swaption.setPricingEngine(ext::make_shared<FdG2SwaptionEngine>(
            model, Size(c.tGrid), Size(c.xGrid), Size(c.yGrid), Size(c.dampingSteps),
            c.invEps, FdmSchemeDesc::Hundsdorfer()));

        Obj in;
        describeRates(in);
        in.n("g2_a", kG2A);
        in.n("g2_sigma", kG2Sigma);
        in.n("g2_b", kG2B);
        in.n("g2_eta", kG2Eta);
        in.n("g2_rho", kG2Rho);
        in.s("swap_type", c.type == Swap::Payer ? "Payer" : "Receiver");
        in.n("fixed_rate", c.fixedRate);
        in.s("exercise", c.exercise);
        in.i("t_grid", c.tGrid);
        in.i("x_grid", c.xGrid);
        in.i("y_grid", c.yGrid);
        in.i("damping_steps", c.dampingSteps);
        in.n("inv_eps", c.invEps);
        in.s("scheme", "hundsdorfer");

        Obj ex;
        ex.n("npv", swaption.NPV());
        addCase(c.name, in, ex);
    }

    probeThrow("g2_swaption_throw_daycounter", "FdG2SwaptionEngine",
               "forward/discount day counters differ", [] {
                   auto fwd = Handle<YieldTermStructure>(
                       ext::make_shared<FlatForward>(kToday, kFwdRate, Actual360()));
                   auto disc = ratesCurve(kDiscRate);
                   auto model =
                       ext::make_shared<G2>(disc, kG2A, kG2Sigma, kG2B, kG2Eta, kG2Rho);
                   Swaption swaption(makeSwap(fwd, Swap::Payer, 0.03),
                                     ext::make_shared<EuropeanExercise>(swaptionExpiry()));
                   swaption.setPricingEngine(
                       ext::make_shared<FdG2SwaptionEngine>(model, 15, 11, 11));
                   swaption.NPV();
               });
}

// A handful of intermediate quantities, so a Python failure can be localised
// to the mesher rather than the rollback.
void diagnosticsCases() {
    Obj in;
    describeHeston(in);
    describeGrid(in, 25, 30, 12, 0, "hundsdorfer");
    Obj ex;

    // FdmHestonLocalVolatilityVarianceMesher with no leverage function,
    // tAvgSteps = max(5, tGrid/50) = 5 for tGrid = 25.
    const ext::shared_ptr<FdmHestonLocalVolatilityVarianceMesher> vMesher =
        ext::make_shared<FdmHestonLocalVolatilityVarianceMesher>(
            12, hestonProcess(), ext::shared_ptr<LocalVolTermStructure>(), 1.0, 5, 0.0001,
            1.0);
    ex.n("vola_estimate", vMesher->volaEstimate());
    for (Size i = 0; i < 12; ++i)
        ex.n("v_loc" + std::to_string(i), vMesher->location(i));

    const ext::shared_ptr<FdmHestonLocalVolatilityVarianceMesher> vMesherMix =
        ext::make_shared<FdmHestonLocalVolatilityVarianceMesher>(
            12, hestonProcess(), ext::shared_ptr<LocalVolTermStructure>(), 1.0, 5, 0.0001,
            0.7);
    ex.n("vola_estimate_mix07", vMesherMix->volaEstimate());

    // The equity mesher of FdHestonVanillaEngine: scaleFactor 2.0 and the
    // strike-concentrated cPoint (strike, 0.1), through the swapped-curve
    // processHelper described at the top of this file.
    auto process = hestonProcess();
    auto helper = FdmBlackScholesMesher::processHelper(
        process->s0(), process->dividendYield(), process->riskFreeRate(),
        vMesher->volaEstimate());
    FdmBlackScholesMesher eq(30, helper, 1.0, 100.0, Null<Real>(), Null<Real>(), 0.0001,
                             2.0, std::pair<Real, Real>(100.0, 0.1));
    for (Size i = 0; i < 30; ++i)
        ex.n("x_loc" + std::to_string(i), eq.location(i));
    ex.n("helper_riskfree_discount_1y", helper->riskFreeRate()->discount(1.0));
    ex.n("helper_dividend_discount_1y", helper->dividendYield()->discount(1.0));

    addCase("diagnostics", in, ex);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    diagnosticsCases();
    hestonVanillaCases();
    hestonMultiStrikeCases();
    makeFdHestonCases();
    dividendAndQuantoCases();
    batesCases();
    hestonHullWhiteCases();
    sabrCases();
    barrierCases();
    rebateCases();
    doubleBarrierCases();
    hullWhiteSwaptionCases();
    g2SwaptionCases();

    emitDocument();
    return 0;
}
