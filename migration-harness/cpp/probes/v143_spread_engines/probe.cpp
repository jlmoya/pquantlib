// migration-harness/cpp/probes/v143_spread_engines/probe.cpp
//
// Reference values for the two spread-option pricing engines introduced in
// C++ QuantLib v1.43:
//
//   * PearsonSpreadEngine          (ql/pricingengines/basket/pearsonspreadengine.{hpp,cpp})
//       Pearson (1995) 1-D adaptive Gauss-Lobatto integration of a conditional
//       Black call/put over the standard normal driving asset 2.
//       Derives from SpreadBlackScholesVanillaEngine, so it only implements the
//       protected calculate(f1, f2, strike, type, variance1, variance2, df) hook;
//       the base class does the forward / variance / discount extraction.
//
//   * GaussianCopulaSpreadEngine   (ql/pricingengines/basket/gaussiancopulaspreadengine.{hpp,cpp})
//       Nested Gauss-Hermite quadrature over a Gaussian copula whose marginals
//       come from SmileSectionRNDCalculator, i.e. it is smile-aware. Derives
//       directly from BasketOption::engine and overrides calculate().
//
// What has to be pinned, and why
// ------------------------------
// Neither engine populates greeks or additionalResults: `calculate()` assigns
// only `results_.value`. MultiAssetOption::results carries a Greeks block, but
// both engines leave every field at Null<Real>(), so `option.delta()` throws
// "delta not provided". NPV is therefore the entire observable surface, and the
// `*_no_extra_results` cases pin that emptiness explicitly so a port does not
// invent greeks that C++ does not produce.
//
// Coverage is chosen so a port cannot pass by luck:
//
//  * strikes spanning deep-ITM / ATM (strike == f1 - f2) / deep-OTM, including
//    NEGATIVE strikes, which is the whole point of a spread option;
//  * both Option::Call and Option::Put at every interesting strike;
//  * correlations 0, +-0.9, +-1 and near-degenerate +-0.999999, which drive the
//    `std::max(1 - rho^2, 0)` guard in both engines to exactly zero;
//  * different volatilities per leg, and a Merton setup with different dividend
//    yields per leg so `fwd = spot * qDiscount / rDiscount` is actually exercised
//    (a BlackProcess has q == r and hides a wrong forward formula);
//  * a genuine SVI smile (via PiecewiseBlackVarianceSurface), which is the only
//    configuration where the copula engine differs from bivariate lognormal;
//  * the engines' own tuning knobs (Pearson: integrationTolerance / nStd;
//    copula: nPoints), so a port must expose them rather than hardcode defaults.
//
// Guard clauses / early returns that are deliberately hit
// -------------------------------------------------------
//  1. PearsonSpreadEngine::calculate() integrand:
//        if (effectiveStrike <= 0.0)
//            return phi(z) * max(0, f1*exp(rho*sigma1*z - 0.5*rho^2*var1) - effectiveStrike);
//     effectiveStrike = f2*exp(-0.5*var2 + sigma2*z) + strike, so this fires for
//     sufficiently negative strikes over the lower part of the z range.
//     NOTE: the branch ignores `optionType` and always returns the CALL intrinsic.
//     For a PUT with a deeply negative strike C++ therefore returns a non-zero
//     value where the mathematically correct answer is 0, and put-call parity
//     breaks. That is reproduced here on purpose (cases `pearson_a_put_km10_*`
//     and `pearson_a_put_km30_*`); a port MUST copy the branch verbatim.
//  2. sigma1Cond = sigma1 * sqrt(max(1 - rho^2, 0)) -> exactly 0 at rho = +-1,
//     so BlackCalculator is called with stdDev == 0 and returns intrinsic.
//  3. GaussianCopulaSpreadEngine ctor: QL_REQUIRE(correlation in [-1, 1])
//     (inclusive) and QL_REQUIRE that BOTH processes share the *same*
//     riskFreeRate().currentLink() -- not merely equal curves, the identical
//     shared_ptr. Every setup below uses one shared rTS handle for that reason.
//  4. GaussianCopulaSpreadEngine::calculate() clamps Phi(z) into
//     [QL_EPSILON, 1 - QL_EPSILON] before calling invcdf, because at extreme
//     Gauss-Hermite nodes Phi saturates to exactly 0 or 1 and
//     SmileSectionRNDCalculator::invcdf requires p strictly inside (0, 1).
//
// Quadrature convention (easy to get wrong in a port)
// ---------------------------------------------------
// QuantLib's GaussianQuadrature divides the classical weights by the weight
// function: w_i = mu_0 * ev[0][i]^2 / w(x_i), with w(x) = exp(-x^2) for
// GaussHermitePolynomial(0). So `sum_i w_i f(x_i)` approximates the *unweighted*
// integral of f. That is why GaussianCopulaSpreadEngine multiplies the kernel by
// exp(-x_i^2) exp(-x_j^2) again and normalises by 1/pi. A port whose
// GaussHermiteIntegration keeps the classical weights will be off by exp(x^2)
// factors and will not reproduce any copula case below.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/spread/engines.json. Nothing else may be printed.

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/experimental/volatility/svismilesection.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/pricingengines/basket/gaussiancopulaspreadengine.hpp>
#include <ql/pricingengines/basket/pearsonspreadengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/piecewiseblackvariancesurface.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/period.hpp>

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

// ---------------------------------------------------------------------------
// Market: 1 March 2025 -> 1 March 2026 is exactly 365 days, so with
// Actual365Fixed the exercise time is exactly 1.0 and no day-count rounding
// noise leaks into the reference values.
// ---------------------------------------------------------------------------
const Date kToday(1, March, 2025);
const Date kMaturity(1, March, 2026);
const Rate kRiskFree = 0.05;

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

// The one and only risk-free curve. GaussianCopulaSpreadEngine compares
// riskFreeRate().currentLink() pointers, so both legs must share this object.
const Handle<YieldTermStructure>& riskFreeCurve() {
    static const Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(kToday, kRiskFree, dayCounter()));
    return rTS;
}

const ext::shared_ptr<EuropeanExercise>& exercise() {
    static const ext::shared_ptr<EuropeanExercise> e =
        ext::make_shared<EuropeanExercise>(kMaturity);
    return e;
}

Handle<Quote> quote(Real v) {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(v));
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kToday, r, dayCounter()));
}

Handle<BlackVolTermStructure> flatVol(Volatility v) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(kToday, NullCalendar(), v, dayCounter()));
}

// PiecewiseBlackVarianceSurface with a single SVI tenor at the option maturity.
// The surface must live in a shared_ptr: BlackVolTermStructure::smileSectionImpl
// uses shared_from_this().
Handle<BlackVolTermStructure> sviVol(Real forward, const std::vector<Real>& params) {
    const Time t = dayCounter().yearFraction(kToday, kMaturity);
    const auto smile = ext::make_shared<SviSmileSection>(t, forward, params);
    return Handle<BlackVolTermStructure>(
        ext::make_shared<PiecewiseBlackVarianceSurface>(
            kToday, kMaturity, smile, dayCounter()));
}

using Process = ext::shared_ptr<GeneralizedBlackScholesProcess>;

struct Legs {
    Process p1, p2;
};

// SVI parameters { a, b, sigma, rho, m }, taken from the v1.43 test-suite case
// testGaussianCopulaSpreadEngineSVI in test-suite/basketoption.cpp.
const std::vector<Real> kSvi1 = {0.04, 0.10, 0.30, -0.40, 0.0};
const std::vector<Real> kSvi2 = {0.02, 0.08, 0.25, -0.30, 0.0};

// Setup A -- the v1.43 testPearsonSpreadEngine market.
// BlackProcess => dividend curve is the risk-free curve => forward == spot.
// f1 - f2 = -10, so strike -10 is the at-the-money spread strike.
Legs setupA() {
    return Legs{
        ext::make_shared<BlackProcess>(quote(100.0), riskFreeCurve(), flatVol(0.25)),
        ext::make_shared<BlackProcess>(quote(110.0), riskFreeCurve(), flatVol(0.35))};
}

// Setup B -- the v1.43 testGaussianCopulaSpreadEngineFlatVol market.
// f1 - f2 = 4.
Legs setupB() {
    return Legs{
        ext::make_shared<BlackProcess>(quote(100.0), riskFreeCurve(), flatVol(0.20)),
        ext::make_shared<BlackProcess>(quote(96.0), riskFreeCurve(), flatVol(0.25))};
}

// Setup C -- Merton legs with DIFFERENT dividend yields, so the engines' forward
// formula fwd = spot * qDiscount / rDiscount is genuinely exercised:
//   fwd1 = 100 * exp(-0.02) / exp(-0.05) = 100 * exp( 0.03)
//   fwd2 =  95 * exp(-0.06) / exp(-0.05) =  95 * exp(-0.01)
// Both legs still share the single risk-free curve object.
Legs setupC() {
    return Legs{
        ext::make_shared<BlackScholesMertonProcess>(
            quote(100.0), flatCurve(0.02), riskFreeCurve(), flatVol(0.30)),
        ext::make_shared<BlackScholesMertonProcess>(
            quote(95.0), flatCurve(0.06), riskFreeCurve(), flatVol(0.15))};
}

// Setup D -- the v1.43 testGaussianCopulaSpreadEngineSVI market, verbatim.
// Note the deliberate asymmetry upstream: the SVI sections are built at
// fwd_i = spot_i / df (105.127 and 100.922), but the legs are BlackProcess, so
// the engines compute fwd_i = spot_i (100 and 96). GaussianCopulaSpreadEngine
// then wraps the section in AtmSmileSection(section, 100) and OVERRIDES the
// SVI's own atmLevel. Reproducing that override is the point of setup D.
Legs setupD() {
    const DiscountFactor df = riskFreeCurve()->discount(kMaturity);
    return Legs{
        ext::make_shared<BlackProcess>(quote(100.0), riskFreeCurve(),
                                       sviVol(100.0 / df, kSvi1)),
        ext::make_shared<BlackProcess>(quote(96.0), riskFreeCurve(),
                                       sviVol(96.0 / df, kSvi2))};
}

// Setup E -- the same SVI shapes, but built at the forwards the engines will
// actually use, so AtmSmileSection's override is a no-op. Together with setup D
// this separates "used the process forward" from "used the smile's own atm".
Legs setupE() {
    return Legs{
        ext::make_shared<BlackProcess>(quote(100.0), riskFreeCurve(), sviVol(100.0, kSvi1)),
        ext::make_shared<BlackProcess>(quote(96.0), riskFreeCurve(), sviVol(96.0, kSvi2))};
}

ext::shared_ptr<BasketPayoff> spreadPayoff(Option::Type type, Real strike) {
    return ext::make_shared<SpreadBasketPayoff>(
        ext::make_shared<PlainVanillaPayoff>(type, strike));
}

Real pearsonNpv(const Legs& legs, Real rho, Option::Type type, Real strike,
                Real tolerance = 1e-10, Size maxIterations = 10000, Real nStd = 8.0) {
    BasketOption option(spreadPayoff(type, strike), exercise());
    option.setPricingEngine(ext::make_shared<PearsonSpreadEngine>(
        legs.p1, legs.p2, rho, tolerance, maxIterations, nStd));
    return option.NPV();
}

Real copulaNpv(const Legs& legs, Real rho, Option::Type type, Real strike,
               Size nPoints = 64) {
    BasketOption option(spreadPayoff(type, strike), exercise());
    option.setPricingEngine(ext::make_shared<GaussianCopulaSpreadEngine>(
        legs.p1, legs.p2, rho, nPoints));
    return option.NPV();
}

const char* typeName(Option::Type t) { return t == Option::Call ? "Call" : "Put"; }

Obj spreadInputs(const char* setup, Real rho, Option::Type type, Real strike) {
    Obj o;
    o.s("setup", setup)
        .s("today", "2025-03-01")
        .s("maturity", "2026-03-01")
        .s("day_counter", "Actual365Fixed")
        .n("exercise_time", 1.0)
        .n("risk_free_rate", kRiskFree)
        .n("correlation", rho)
        .s("option_type", typeName(type))
        .n("strike", strike);
    return o;
}

// ---------------------------------------------------------------------------
// Diagnostics: the inputs SpreadBlackScholesVanillaEngine::calculate() derives
// before delegating, and that GaussianCopulaSpreadEngine::calculate() derives
// independently. Pinning these first makes a failing NPV trivially bisectable
// into "wrong forward/variance/discount" vs "wrong quadrature".
// ---------------------------------------------------------------------------
void emitSetupDiagnostics(const char* setup, const Legs& legs) {
    const Real f1 = legs.p1->stateVariable()->value() *
                    legs.p1->dividendYield()->discount(kMaturity) /
                    legs.p1->riskFreeRate()->discount(kMaturity);
    const Real f2 = legs.p2->stateVariable()->value() *
                    legs.p2->dividendYield()->discount(kMaturity) /
                    legs.p2->riskFreeRate()->discount(kMaturity);
    const Real var1 = legs.p1->blackVolatility()->blackVariance(kMaturity, f1);
    const Real var2 = legs.p2->blackVolatility()->blackVariance(kMaturity, f2);
    const DiscountFactor df = legs.p1->riskFreeRate()->discount(kMaturity);
    const Time t1 = legs.p1->blackVolatility()->timeFromReference(kMaturity);
    const Time t2 = legs.p2->blackVolatility()->timeFromReference(kMaturity);

    Obj in;
    in.s("setup", setup)
        .s("today", "2025-03-01")
        .s("maturity", "2026-03-01")
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", kRiskFree);

    Obj ex;
    ex.n("forward1", f1)
        .n("forward2", f2)
        .n("variance1", var1)
        .n("variance2", var2)
        .n("discount", df)
        .n("time1", t1)
        .n("time2", t2);

    addCase(std::string(setup) + "_derived_inputs", in, ex);
}

struct Row {
    const char* name;
    Option::Type type;
    Real strike;
    Real rho;
};

void emitPearson(const char* setup, const Legs& legs, const std::vector<Row>& rows) {
    for (const Row& r : rows) {
        Obj ex;
        ex.n("npv", pearsonNpv(legs, r.rho, r.type, r.strike));
        addCase(r.name, spreadInputs(setup, r.rho, r.type, r.strike), ex);
    }
}

void emitCopula(const char* setup, const Legs& legs, const std::vector<Row>& rows,
                Size nPoints = 64) {
    for (const Row& r : rows) {
        Obj in = spreadInputs(setup, r.rho, r.type, r.strike);
        in.i("n_points", static_cast<long long>(nPoints));
        Obj ex;
        ex.n("npv", copulaNpv(legs, r.rho, r.type, r.strike, nPoints));
        addCase(r.name, in, ex);
    }
}

void emitPearsonKnobs(const char* name, const Legs& legs, Real rho, Option::Type type,
                      Real strike, Real tolerance, Size maxIterations, Real nStd) {
    Obj in = spreadInputs("a", rho, type, strike);
    in.n("integration_tolerance", tolerance)
        .i("max_integration_iterations", static_cast<long long>(maxIterations))
        .n("n_std", nStd);
    Obj ex;
    ex.n("npv", pearsonNpv(legs, rho, type, strike, tolerance, maxIterations, nStd));
    addCase(name, in, ex);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    const Legs a = setupA();
    const Legs b = setupB();
    const Legs c = setupC();
    const Legs d = setupD();
    const Legs e = setupE();

    emitSetupDiagnostics("a", a);
    emitSetupDiagnostics("b", b);
    emitSetupDiagnostics("c", c);
    emitSetupDiagnostics("d", d);
    emitSetupDiagnostics("e", e);

    // -----------------------------------------------------------------------
    // PearsonSpreadEngine
    // -----------------------------------------------------------------------

    // Setup A at the upstream correlation 0.75. f1 - f2 = -10, so:
    //   k = -30  deep ITM call / deep OTM put, effectiveStrike <= 0 over much of z
    //   k = -10  at-the-money spread, effectiveStrike <= 0 in the lower tail
    //   k =   0  exchange (Margrabe) option
    //   k =   5  the upstream put-call-parity strike; branch never fires
    //   k =  30  deep OTM call / deep ITM put
    emitPearson("a", a,
                {{"pearson_a_call_km30_rho075", Option::Call, -30.0, 0.75},
                 {"pearson_a_put_km30_rho075", Option::Put, -30.0, 0.75},
                 {"pearson_a_call_km10_rho075", Option::Call, -10.0, 0.75},
                 {"pearson_a_put_km10_rho075", Option::Put, -10.0, 0.75},
                 {"pearson_a_call_k0_rho075", Option::Call, 0.0, 0.75},
                 {"pearson_a_put_k0_rho075", Option::Put, 0.0, 0.75},
                 {"pearson_a_call_k5_rho075", Option::Call, 5.0, 0.75},
                 {"pearson_a_put_k5_rho075", Option::Put, 5.0, 0.75},
                 {"pearson_a_call_k30_rho075", Option::Call, 30.0, 0.75},
                 {"pearson_a_put_k30_rho075", Option::Put, 30.0, 0.75}});

    // Correlation sweep, including both signs of the degenerate rho = +-1
    // (sigma1Cond collapses to 0) and the near-degenerate +-0.999999.
    emitPearson("a", a,
                {{"pearson_a_call_k5_rho000", Option::Call, 5.0, 0.0},
                 {"pearson_a_put_k5_rho000", Option::Put, 5.0, 0.0},
                 {"pearson_a_call_k5_rho_m090", Option::Call, 5.0, -0.9},
                 {"pearson_a_put_k5_rho_m090", Option::Put, 5.0, -0.9},
                 {"pearson_a_call_k5_rho_p0999999", Option::Call, 5.0, 0.999999},
                 {"pearson_a_put_k5_rho_p0999999", Option::Put, 5.0, 0.999999},
                 {"pearson_a_call_k5_rho_m0999999", Option::Call, 5.0, -0.999999},
                 {"pearson_a_call_k5_rho_p1", Option::Call, 5.0, 1.0},
                 {"pearson_a_put_k5_rho_p1", Option::Put, 5.0, 1.0},
                 {"pearson_a_call_k5_rho_m1", Option::Call, 5.0, -1.0},
                 {"pearson_a_put_k5_rho_m1", Option::Put, 5.0, -1.0},
                 {"pearson_a_call_km10_rho_p1", Option::Call, -10.0, 1.0},
                 {"pearson_a_put_km10_rho_p1", Option::Put, -10.0, 1.0}});

    // Non-default integration knobs. nStd = 4 truncates the outer normal at
    // +-4 instead of +-8 and changes the answer materially; the coarse
    // tolerance changes it slightly. A port that hardcodes the defaults fails.
    emitPearsonKnobs("pearson_a_call_k5_rho075_nstd4", a, 0.75, Option::Call, 5.0,
                     1e-10, 10000, 4.0);
    emitPearsonKnobs("pearson_a_call_k5_rho075_nstd2", a, 0.75, Option::Call, 5.0,
                     1e-10, 10000, 2.0);
    emitPearsonKnobs("pearson_a_call_k5_rho075_tol1em6", a, 0.75, Option::Call, 5.0,
                     1e-6, 10000, 8.0);

    // Setup C: different dividend yields per leg. fwd1 - fwd2 ~ 8.99.
    emitPearson("c", c,
                {{"pearson_c_call_km15_rho050", Option::Call, -15.0, 0.5},
                 {"pearson_c_put_km15_rho050", Option::Put, -15.0, 0.5},
                 {"pearson_c_call_k0_rho050", Option::Call, 0.0, 0.5},
                 {"pearson_c_call_k5_rho050", Option::Call, 5.0, 0.5},
                 {"pearson_c_put_k5_rho050", Option::Put, 5.0, 0.5},
                 {"pearson_c_call_k25_rho050", Option::Call, 25.0, 0.5},
                 {"pearson_c_call_k5_rho_m090", Option::Call, 5.0, -0.9}});

    // Setup D: SVI smile. Pearson only sees blackVariance at the forward, so it
    // is smile-blind by construction -- pinning it here documents that, and
    // gives the copula cases a same-market baseline to be compared against.
    emitPearson("d", d,
                {{"pearson_d_call_km2_rho050", Option::Call, -2.0, 0.5},
                 {"pearson_d_call_k0_rho050", Option::Call, 0.0, 0.5},
                 {"pearson_d_call_k4_rho050", Option::Call, 4.0, 0.5},
                 {"pearson_d_put_k4_rho050", Option::Put, 4.0, 0.5},
                 {"pearson_d_call_k8_rho050", Option::Call, 8.0, 0.5}});

    // -----------------------------------------------------------------------
    // GaussianCopulaSpreadEngine
    // -----------------------------------------------------------------------

    // Setup B at the upstream correlation 0.5. f1 - f2 = 4.
    emitCopula("b", b,
               {{"copula_b_call_km10_rho050", Option::Call, -10.0, 0.5},
                {"copula_b_put_km10_rho050", Option::Put, -10.0, 0.5},
                {"copula_b_call_k0_rho050", Option::Call, 0.0, 0.5},
                {"copula_b_put_k0_rho050", Option::Put, 0.0, 0.5},
                {"copula_b_call_k3_rho050", Option::Call, 3.0, 0.5},
                {"copula_b_put_k3_rho050", Option::Put, 3.0, 0.5},
                {"copula_b_call_k4_rho050", Option::Call, 4.0, 0.5},
                {"copula_b_put_k4_rho050", Option::Put, 4.0, 0.5},
                {"copula_b_call_k20_rho050", Option::Call, 20.0, 0.5},
                {"copula_b_put_k20_rho050", Option::Put, 20.0, 0.5}});

    emitCopula("b", b,
               {{"copula_b_call_k3_rho000", Option::Call, 3.0, 0.0},
                {"copula_b_put_k3_rho000", Option::Put, 3.0, 0.0},
                {"copula_b_call_k3_rho_m090", Option::Call, 3.0, -0.9},
                {"copula_b_put_k3_rho_m090", Option::Put, 3.0, -0.9},
                {"copula_b_call_k3_rho_p0999999", Option::Call, 3.0, 0.999999},
                {"copula_b_call_k3_rho_m0999999", Option::Call, 3.0, -0.999999},
                {"copula_b_call_k3_rho_p1", Option::Call, 3.0, 1.0},
                {"copula_b_put_k3_rho_p1", Option::Put, 3.0, 1.0},
                {"copula_b_call_k3_rho_m1", Option::Call, 3.0, -1.0},
                {"copula_b_put_k3_rho_m1", Option::Put, 3.0, -1.0}});

    // nPoints sweep. The engine's cost is O(nPoints^2); the value converges but
    // is visibly different at 16 and 32, so the knob cannot be faked.
    emitCopula("b", b, {{"copula_b_call_k3_rho050_n16", Option::Call, 3.0, 0.5}}, 16);
    emitCopula("b", b, {{"copula_b_call_k3_rho050_n32", Option::Call, 3.0, 0.5}}, 32);
    emitCopula("b", b, {{"copula_b_call_k3_rho050_n128", Option::Call, 3.0, 0.5}}, 128);
    emitCopula("b", b, {{"copula_b_put_k3_rho050_n16", Option::Put, 3.0, 0.5}}, 16);

    // Setup A with the copula engine: same market as the Pearson block, so the
    // two engines can be compared directly on identical inputs.
    emitCopula("a", a,
               {{"copula_a_call_km10_rho075", Option::Call, -10.0, 0.75},
                {"copula_a_put_km10_rho075", Option::Put, -10.0, 0.75},
                {"copula_a_call_k0_rho075", Option::Call, 0.0, 0.75},
                {"copula_a_call_k5_rho075", Option::Call, 5.0, 0.75},
                {"copula_a_put_k5_rho075", Option::Put, 5.0, 0.75},
                {"copula_a_call_k30_rho075", Option::Call, 30.0, 0.75}});

    // Setup C: different dividend yields per leg -> forward formula exercised.
    emitCopula("c", c,
               {{"copula_c_call_k0_rho050", Option::Call, 0.0, 0.5},
                {"copula_c_call_k5_rho050", Option::Call, 5.0, 0.5},
                {"copula_c_put_k5_rho050", Option::Put, 5.0, 0.5},
                {"copula_c_call_km15_rho050", Option::Call, -15.0, 0.5},
                {"copula_c_call_k25_rho050", Option::Call, 25.0, 0.5}});

    // Setup D: the upstream SVI market. This is the configuration the engine
    // exists for, and the only one where SmileSectionRNDCalculator's marginals
    // differ from lognormal.
    emitCopula("d", d,
               {{"copula_d_call_km2_rho050", Option::Call, -2.0, 0.5},
                {"copula_d_call_k0_rho050", Option::Call, 0.0, 0.5},
                {"copula_d_call_k4_rho050", Option::Call, 4.0, 0.5},
                {"copula_d_put_k4_rho050", Option::Put, 4.0, 0.5},
                {"copula_d_call_k8_rho050", Option::Call, 8.0, 0.5},
                {"copula_d_call_k20_rho050", Option::Call, 20.0, 0.5},
                {"copula_d_call_km10_rho050", Option::Call, -10.0, 0.5},
                {"copula_d_call_k4_rho000", Option::Call, 4.0, 0.0},
                {"copula_d_call_k4_rho_m090", Option::Call, 4.0, -0.9},
                {"copula_d_call_k4_rho_p1", Option::Call, 4.0, 1.0}});

    // Setup E: identical SVI shapes anchored at the engine's own forwards, so
    // AtmSmileSection's atm override is a no-op. Differences between the D and
    // E values isolate the override behaviour.
    emitCopula("e", e,
               {{"copula_e_call_km2_rho050", Option::Call, -2.0, 0.5},
                {"copula_e_call_k4_rho050", Option::Call, 4.0, 0.5},
                {"copula_e_put_k4_rho050", Option::Put, 4.0, 0.5},
                {"copula_e_call_k8_rho050", Option::Call, 8.0, 0.5}});

    // -----------------------------------------------------------------------
    // Structural: both engines write ONLY results_.value.
    // -----------------------------------------------------------------------
    {
        BasketOption option(spreadPayoff(Option::Call, 5.0), exercise());
        option.setPricingEngine(
            ext::make_shared<PearsonSpreadEngine>(a.p1, a.p2, 0.75));
        const Real npv = option.NPV();
        bool deltaThrows = false;
        try {
            option.delta();
        } catch (const std::exception&) {
            deltaThrows = true;
        }
        Obj ex;
        ex.n("npv", npv)
            .i("additional_results_count",
               static_cast<long long>(option.additionalResults().size()))
            .b("delta_throws_not_provided", deltaThrows);
        addCase("pearson_no_extra_results",
                spreadInputs("a", 0.75, Option::Call, 5.0), ex);
    }
    {
        BasketOption option(spreadPayoff(Option::Call, 3.0), exercise());
        option.setPricingEngine(
            ext::make_shared<GaussianCopulaSpreadEngine>(b.p1, b.p2, 0.5));
        const Real npv = option.NPV();
        bool deltaThrows = false;
        try {
            option.delta();
        } catch (const std::exception&) {
            deltaThrows = true;
        }
        Obj ex;
        ex.n("npv", npv)
            .i("additional_results_count",
               static_cast<long long>(option.additionalResults().size()))
            .b("delta_throws_not_provided", deltaThrows);
        addCase("copula_no_extra_results",
                spreadInputs("b", 0.5, Option::Call, 3.0), ex);
    }

    // -----------------------------------------------------------------------
    // Guard clauses. GaussianCopulaSpreadEngine validates in its CONSTRUCTOR;
    // PearsonSpreadEngine has no constructor validation at all (rho outside
    // [-1, 1] is silently accepted and sqrt(max(1-rho^2, 0)) clamps it).
    // -----------------------------------------------------------------------
    {
        auto ctorThrows = [](Real rho, bool sameCurve) {
            try {
                const auto p1 = ext::make_shared<BlackProcess>(
                    quote(100.0), riskFreeCurve(), flatVol(0.20));
                const auto p2 = ext::make_shared<BlackProcess>(
                    quote(96.0),
                    sameCurve ? riskFreeCurve() : flatCurve(kRiskFree),
                    flatVol(0.25));
                const GaussianCopulaSpreadEngine engine(p1, p2, rho);
                static_cast<void>(engine);
                return false;
            } catch (const std::exception&) {
                return true;
            }
        };

        auto emitCtorCase = [&](const std::string& name, Real rho, bool sameCurve,
                                const std::string& note) {
            Obj in;
            in.n("correlation", rho).b("shared_risk_free_curve", sameCurve);
            if (!note.empty())
                in.s("note", note);
            Obj ex;
            ex.b("throws", ctorThrows(rho, sameCurve));
            addCase(name, in, ex);
        };

        emitCtorCase("copula_ctor_rejects_rho_above_one", 1.0000001, true, "");
        emitCtorCase("copula_ctor_rejects_rho_below_minus_one", -1.0000001, true, "");
        emitCtorCase("copula_ctor_accepts_rho_exactly_one", 1.0, true, "");
        emitCtorCase("copula_ctor_accepts_rho_exactly_minus_one", -1.0, true, "");
        // Two *equal but distinct* flat 5% curves: the check compares
        // currentLink() pointers, not curve values, so this must throw.
        emitCtorCase("copula_ctor_rejects_distinct_risk_free_curves", 0.5, false,
                     "both curves are FlatForward(2025-03-01, 0.05, Actual365Fixed)");
    }
    {
        // PearsonSpreadEngine does NOT validate the correlation.
        bool ctorThrows = false;
        try {
            const PearsonSpreadEngine engine(a.p1, a.p2, 1.5);
            static_cast<void>(engine);
        } catch (const std::exception&) {
            ctorThrows = true;
        }
        Obj in;
        in.n("correlation", 1.5);
        Obj ex;
        ex.b("throws", ctorThrows);
        addCase("pearson_ctor_does_not_validate_correlation", in, ex);
    }
    {
        // Both engines require a SpreadBasketPayoff wrapping a PlainVanillaPayoff
        // and a EuropeanExercise. A plain AverageBasketPayoff must be rejected.
        auto payoffRejected = [](bool useCopula) {
            BasketOption option(
                ext::make_shared<AverageBasketPayoff>(
                    ext::make_shared<PlainVanillaPayoff>(Option::Call, 5.0), Size(2)),
                exercise());
            const Legs legs = setupA();
            if (useCopula)
                option.setPricingEngine(ext::make_shared<GaussianCopulaSpreadEngine>(
                    legs.p1, legs.p2, 0.75));
            else
                option.setPricingEngine(
                    ext::make_shared<PearsonSpreadEngine>(legs.p1, legs.p2, 0.75));
            try {
                option.NPV();
                return false;
            } catch (const std::exception&) {
                return true;
            }
        };

        Obj in;
        in.s("payoff", "AverageBasketPayoff");
        Obj exPearson;
        exPearson.b("throws", payoffRejected(false));
        addCase("pearson_rejects_non_spread_payoff", in, exPearson);
        Obj exCopula;
        exCopula.b("throws", payoffRejected(true));
        addCase("copula_rejects_non_spread_payoff", in, exCopula);
    }

    emitDocument();
    return 0;
}
