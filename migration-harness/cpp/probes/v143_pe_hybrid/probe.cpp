// migration-harness/cpp/probes/v143_pe_hybrid/probe.cpp
//
// Reference values for the "hybrid / non-Black" analytic vanilla engines of
// C++ QuantLib v1.43, all living in ql/pricingengines/vanilla/:
//
//   * AnalyticBSMHullWhiteEngine   (analyticbsmhullwhiteengine.{hpp,cpp})
//       Black-Scholes with a *stochastic* Hull-White short rate and a constant
//       equity/short-rate correlation rho. Implemented by shifting the Black
//       variance by a closed-form offset and delegating to AnalyticEuropeanEngine.
//
//   * AnalyticBlackVasicekEngine   (analyticeuropeanvasicekengine.{hpp,cpp})
//       NOTE the header name does NOT match the class name. Black-Scholes with a
//       Vasicek short rate; the total variance upsilon is a Simpson integral.
//
//   * AnalyticHestonHullWhiteEngine(analytichestonhullwhiteengine.{hpp,cpp})
//       Derives from AnalyticHestonEngine and injects an addOnTerm into both
//       characteristic functions. Assumes dW_S dW_r = 0 (no equity/rate corr.).
//
//   * AnalyticH1HWEngine           (analytich1hwengine.{hpp,cpp})
//       Derives from AnalyticHestonHullWhiteEngine and ADDS the Grzelak-Oosterlee
//       H1-HW equity/rate correlation term on top of its parent's addOnTerm.
//
//   * CEVCalculator + AnalyticCEVEngine (analyticcevengine.{hpp,cpp})
//       Closed-form CEV (constant elasticity of variance) European prices via
//       non-central chi-squared CDFs; two structurally different formulae for
//       delta < 2 (beta < 1, absorbing boundary, true martingale) and
//       delta >= 2 (beta > 1, strict local martingale).
//
//   * AnalyticGJRGARCHEngine       (analyticgjrgarchengine.{hpp,cpp})
//       Edgeworth expansion of the GJR-GARCH(1,1) European call (Duan et al. 2006).
//
// What has to be pinned, and why
// ------------------------------
// 1. THE CORRELATION MUST MOVE THE PRICE. Three of these engines take an
//    equity/rate correlation as a plain constructor Real. An engine that accepts
//    and discards it looks ported and prices "reasonably". Every correlation
//    sweep below therefore prices the SAME market at several rho on purpose:
//      bsmhw_rho_*   rho in {-0.9,-0.75,-0.25,0,0.25,0.75,0.9}
//      vasicek_rho_* rho in {-0.9,-0.5,0,0.5,0.9}
//      h1hw_rhosr_*  rhoSr in {0,0.3,0.6,0.9}
//    and the reference values are all distinct.
//
// 2. DEGENERATE LIMITS. Each hybrid engine must collapse onto the corresponding
//    deterministic-rate engine when the short-rate vol goes to zero:
//      bsmhw_zero_hw_vol_matches_bs        -> AnalyticEuropeanEngine
//      vasicek_zero_rate_vol_matches_bs    -> AnalyticEuropeanEngine
//      hhw_zero_hwvol_collapses_to_heston  -> AnalyticHestonEngine
//      h1hw_rhosr_zero_matches_hhw         -> AnalyticHestonHullWhiteEngine
//    Each of those cases pins BOTH numbers, so the Python test asserts the
//    collapse rather than merely reproducing one value. Note the short-rate vol
//    cannot be set to exactly 0: HullWhite/Vasicek hold sigma in a
//    ConstantParameter with a PositiveConstraint and the ctor QL_REQUIREs it, so
//    the limit is taken at sigma = 1e-12 (m_ / varianceOffset then land at ~1e-24
//    and are below double resolution against the Heston/BS price).
//
// 3. THE LOW-`a` ALGEBRAIC BRANCH. Both Hull-White engines branch on
//        a*t > std::pow(QL_EPSILON, 0.25)      // == 2^-13 == 1.220703125e-4
//    and use a Taylor expansion below it. QL_EPSILON is 2^-52, so the threshold
//    is EXACTLY 2^-13 and can be straddled bit-for-bit. Cases
//    `bsmhw_a_at_threshold` / `bsmhw_a_just_above_threshold` (and the hhw_
//    equivalents) put a*t exactly ON the threshold (which takes the LOW branch,
//    because the test is a strict `>`) and one ULP above it. Every maturity in
//    this probe is a whole number of 365-day years under Actual365Fixed, so
//    t is exactly 1.0 / 5.0 / 10.0 / 20.0 and `a*t == a` for the t == 1 cases.
//
// 4. WHICH RESULT FIELDS EXIST.
//      - AnalyticBSMHullWhiteEngine assigns `results_ = *dynamic_cast<const
//        OneAssetOption::results*>(bsmEngine->getResults())`, i.e. it copies the
//        WHOLE AnalyticEuropeanEngine result block: every greek AND
//        additionalResults. `bsmhw_rho_*` pins delta/gamma/theta/vega/rho/
//        dividendRho/elasticity/deltaForward/strikeSensitivity/
//        itmCashProbability and the additionalResults, because a port that only
//        fills `value` is wrong in a way no NPV test can see. Note that vega and
//        the additionalResults "volatility" are taken against the SHIFTED vol
//        surface, not the input one.
//      - AnalyticBlackVasicekEngine, AnalyticHestonHullWhiteEngine,
//        AnalyticH1HWEngine, AnalyticCEVEngine and AnalyticGJRGARCHEngine assign
//        ONLY results_.value. `*_no_greeks` cases pin that emptiness explicitly
//        (option.delta() throws "delta not provided").
//
// 5. THE VASICEK ENGINE READS THE VOL AT t == 0, NOT AT MATURITY.
//        Real sigma_s = blackProcess_->blackVolatility()->blackVol(t, K);  // t = 0
//    That is analyticeuropeanvasicekengine.cpp:79 and it is not a typo we may
//    silently "fix": it is the engine's behaviour. With a flat vol it is
//    invisible, so `vasicek_vol_read_at_time_zero_*` uses a BlackVarianceCurve
//    whose 1y vol (0.15) and 5y vol (0.35) differ; a port that reads the vol at
//    maturity gets a completely different price. Note also that
//    BlackVarianceTermStructure::blackVolImpl substitutes t = 1e-5 for t == 0,
//    so the value actually used is the curve's short end, not a NaN.
//
// 6. THE CEV TWO-REGIME SPLIT. CEVCalculator caches delta_ = (1-2b)/(1-b), so
//    delta_ < 2 <=> beta < 1. The beta < 1 branch uses two non-central chi-squared
//    CDFs; the beta > 1 CALL branch additionally uses boost::math::gamma_p
//    (a regularized lower incomplete gamma) while the beta > 1 PUT branch does
//    not. All four (type x regime) combinations are pinned. The economics are
//    pinned too: for beta < 1 the CEV process is a true martingale, so
//    C - P == (f0 - K)*df to machine precision (`cev_parity_beta045`), while for
//    beta > 1 it is only a strict local martingale and put-call parity is
//    genuinely violated by a computable amount (`cev_parity_beta145`) -- a port
//    that "fixes" the beta > 1 branch to restore parity is wrong.
//    Both engines get their CDFs from BOOST (boost::math::non_central_chi_squared
//    and boost::math::gamma_p), NOT from QuantLib's own
//    NonCentralCumulativeChiSquareDistribution / CumulativeGammaDistribution;
//    those are different algorithms with different tail behaviour, so a port must
//    check which one it reproduces. `cev_calc_far_tail_*` sits deliberately deep
//    in the tail where the two disagree.
//
// 7. CONSTRUCTOR GUARDS, INCLUDING ONE ASYMMETRY.
//      - AnalyticH1HWEngine(model, hw, rhoSr, Size integrationOrder) QL_REQUIREs
//        rhoSr >= 0 ("Fourier integration is not stable if the equity interest
//        rate correlation is negative") but the
//        AnalyticH1HWEngine(model, hw, rhoSr, Real relTol, Size maxEval) overload
//        DOES NOT. Both are pinned (`h1hw_negative_rhosr_order_ctor_throws` /
//        `h1hw_negative_rhosr_tolerance_ctor_ok`).
//      - AnalyticHestonEngine::Integration::gaussLaguerre QL_REQUIREs
//        intOrder <= 192 (`hhw_integration_order_over_192_throws`).
//      - AnalyticBSMHullWhiteEngine QL_REQUIREs a non-null process and a
//        non-empty model handle, and calculate() QL_REQUIREs x0() > 0.
//      - AnalyticBSMHullWhiteEngine itself does NOT check the exercise type; the
//        AnalyticEuropeanEngine it delegates to does, so an American exercise
//        still throws, but with the *inner* engine's message.
//
// 8. INTEGRATION CONFIGURATION. AnalyticHestonHullWhiteEngine / AnalyticH1HWEngine
//    default to Integration::gaussLaguerre(144); the (relTolerance, maxEvaluations)
//    overload uses an adaptive GaussLobattoIntegral instead. PQuantLib's
//    AnalyticHestonEngine has no Integration surface at all (it always uses an
//    adaptive scipy QUADPACK QAGI), so the primary reference values below are
//    taken at a CONVERGED configuration -- gaussLobatto(1e-13, 1e6) -- which any
//    correct adaptive integrator must reproduce. `hhw_gausslaguerre144_vs_lobatto`
//    pins the 144-node Gauss-Laguerre value alongside the converged one so the
//    size of the quadrature gap is a recorded number rather than a guess.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/pe/hybrid.json. Nothing else may be printed.

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/models/equity/gjrgarchmodel.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/onefactormodels/vasicek.hpp>
#include <ql/pricingengines/vanilla/analyticbsmhullwhiteengine.hpp>
#include <ql/pricingengines/vanilla/analyticcevengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanvasicekengine.hpp>
#include <ql/pricingengines/vanilla/analyticgjrgarchengine.hpp>
#include <ql/pricingengines/vanilla/analytich1hwengine.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/pricingengines/vanilla/analytichestonhullwhiteengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/gjrgarchprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
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
// Market. Every maturity is a whole number of 365-day "years" measured in DAYS,
// so with Actual365Fixed the exercise time is exactly 1.0 / 5.0 / 10.0 / 20.0
// and no day-count rounding noise leaks into the branch thresholds.
// ---------------------------------------------------------------------------
const Date kToday(1, March, 2025);
const Integer kDays1Y = 365;
const Integer kDays5Y = 5 * 365;
const Integer kDays10Y = 10 * 365;
const Integer kDays20Y = 20 * 365;

// std::pow(QL_EPSILON, 0.25) == 2^-13 exactly; the Hull-White engines branch on
// `a*t > threshold` (strict).
const Real kLowABranchThreshold = std::pow(QL_EPSILON, 0.25);

const DayCounter& dc() {
    static const DayCounter d = Actual365Fixed();
    return d;
}

Date maturity(Integer days) { return kToday + days; }

Handle<Quote> quote(Real v) {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(v));
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, r, dc()));
}

Handle<BlackVolTermStructure> flatVol(Volatility v) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(kToday, NullCalendar(), v, dc()));
}

// Vol curve with a genuinely different short end and long end, used to prove
// which time the Vasicek engine samples the vol at.
Handle<BlackVolTermStructure> termVol() {
    const std::vector<Date> dates = {maturity(kDays1Y), maturity(kDays5Y),
                                     maturity(kDays10Y)};
    const std::vector<Volatility> vols = {0.15, 0.35, 0.40};
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackVarianceCurve>(kToday, dates, vols, dc(), true));
}

ext::shared_ptr<Exercise> european(Integer days) {
    return ext::make_shared<EuropeanExercise>(maturity(days));
}

ext::shared_ptr<StrikedTypePayoff> vanilla(Option::Type type, Real strike) {
    return ext::make_shared<PlainVanillaPayoff>(type, strike);
}

std::string typeName(Option::Type t) { return t == Option::Call ? "Call" : "Put"; }

// Report a greek, or the fact that the engine did not provide it.
template <class F>
void putGreek(Obj& o, const std::string& key, F f) {
    try {
        o.n(key, f());
    } catch (const std::exception&) {
        o.b(key + "_throws", true);
    }
}

bool throwsOnNPV(VanillaOption& option) {
    try {
        option.NPV();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

// ---------------------------------------------------------------------------
// 1. AnalyticBSMHullWhiteEngine
// ---------------------------------------------------------------------------
struct BsmMarket {
    Real spot = 100.0;
    Rate q = 0.04;
    Rate r = 0.0525;
    Volatility vol = 0.25;
};

ext::shared_ptr<GeneralizedBlackScholesProcess> bsmProcess(const BsmMarket& m) {
    return ext::make_shared<BlackScholesMertonProcess>(quote(m.spot), flatCurve(m.q),
                                                       flatCurve(m.r), flatVol(m.vol));
}

void putBsmMarket(Obj& in, const BsmMarket& m) {
    in.n("spot", m.spot);
    in.n("q", m.q);
    in.n("r", m.r);
    in.n("vol", m.vol);
    in.s("dayCounter", "Actual365Fixed");
    in.s("process", "BlackScholesMertonProcess");
}

void bsmHullWhiteCase(const std::string& name,
                      const BsmMarket& market,
                      Real rho,
                      Real hwA,
                      Real hwSigma,
                      Option::Type type,
                      Real strikeMultiple,  // strike = multiple * forward
                      Integer days,
                      bool withGreeks) {
    const auto process = bsmProcess(market);
    const auto hw = ext::make_shared<HullWhite>(flatCurve(market.r), hwA, hwSigma);
    const Date exDate = maturity(days);
    const Real fwd = market.spot * process->dividendYield()->discount(exDate) /
                     process->riskFreeRate()->discount(exDate);
    const Real strike = strikeMultiple * fwd;

    VanillaOption option(vanilla(type, strike), european(days));
    option.setPricingEngine(ext::make_shared<AnalyticBSMHullWhiteEngine>(rho, process, hw));

    Obj in;
    putBsmMarket(in, market);
    in.n("rho", rho);
    in.n("hwA", hwA);
    in.n("hwSigma", hwSigma);
    in.s("optionType", typeName(type));
    in.n("strikeMultipleOfForward", strikeMultiple);
    in.n("strike", strike);
    in.n("forward", fwd);
    in.i("maturityDays", days);

    Obj ex;
    ex.n("npv", option.NPV());
    if (withGreeks) {
        putGreek(ex, "delta", [&] { return option.delta(); });
        putGreek(ex, "gamma", [&] { return option.gamma(); });
        putGreek(ex, "theta", [&] { return option.theta(); });
        putGreek(ex, "vega", [&] { return option.vega(); });
        putGreek(ex, "rho", [&] { return option.rho(); });
        putGreek(ex, "dividendRho", [&] { return option.dividendRho(); });
        putGreek(ex, "elasticity", [&] { return option.elasticity(); });
        putGreek(ex, "deltaForward", [&] { return option.deltaForward(); });
        putGreek(ex, "strikeSensitivity", [&] { return option.strikeSensitivity(); });
        putGreek(ex, "itmCashProbability", [&] { return option.itmCashProbability(); });
        putGreek(ex, "thetaPerDay", [&] { return option.thetaPerDay(); });
        // AnalyticEuropeanEngine's additionalResults survive the results_ copy.
        const auto& extra = option.additionalResults();
        for (const char* key : {"spot", "dividendDiscount", "riskFreeDiscount", "forward",
                                "strike", "volatility", "timeToExpiry"}) {
            const auto it = extra.find(key);
            if (it != extra.end())
                ex.n(std::string("additional_") + key, ext::any_cast<Real>(it->second));
        }
    }
    addCase(name, in, ex);
}

void emitBsmHullWhite() {
    const BsmMarket market;

    // -- rho sweep on the v1.43 testBsmHullWhiteEngine market, 20y ATMF call --
    const std::pair<const char*, Real> rhos[] = {
        {"m090", -0.90}, {"m075", -0.75}, {"m025", -0.25}, {"000", 0.0},
        {"025", 0.25},   {"075", 0.75},   {"090", 0.90}};
    for (const auto& [tag, rho] : rhos)
        bsmHullWhiteCase(std::string("bsmhw_rho_") + tag, market, rho, 0.00883, 0.00526,
                         Option::Call, 1.0, kDays20Y, true);

    // Puts at the extremes of the same sweep -- the correlation must move the put too.
    bsmHullWhiteCase("bsmhw_put_rho_m090", market, -0.90, 0.00883, 0.00526, Option::Put,
                     1.0, kDays20Y, true);
    bsmHullWhiteCase("bsmhw_put_rho_000", market, 0.0, 0.00883, 0.00526, Option::Put, 1.0,
                     kDays20Y, true);
    bsmHullWhiteCase("bsmhw_put_rho_090", market, 0.90, 0.00883, 0.00526, Option::Put, 1.0,
                     kDays20Y, true);

    // ITM / OTM at a short maturity, both types.
    bsmHullWhiteCase("bsmhw_call_itm_1y", market, 0.5, 0.05, 0.01, Option::Call, 0.7,
                     kDays1Y, false);
    bsmHullWhiteCase("bsmhw_call_otm_1y", market, 0.5, 0.05, 0.01, Option::Call, 1.4,
                     kDays1Y, false);
    bsmHullWhiteCase("bsmhw_put_itm_1y", market, -0.5, 0.05, 0.01, Option::Put, 1.4,
                     kDays1Y, false);
    bsmHullWhiteCase("bsmhw_put_otm_1y", market, -0.5, 0.05, 0.01, Option::Put, 0.7,
                     kDays1Y, false);

    // -- the a -> 0 algebraic branch, straddled bit-for-bit at t == 1.0 --
    bsmHullWhiteCase("bsmhw_low_a_branch", market, 0.6, 1.0e-8, 0.01, Option::Call, 1.0,
                     kDays1Y, false);
    bsmHullWhiteCase("bsmhw_a_at_threshold", market, 0.6, kLowABranchThreshold, 0.01,
                     Option::Call, 1.0, kDays1Y, false);
    bsmHullWhiteCase("bsmhw_a_just_above_threshold", market, 0.6,
                     std::nextafter(kLowABranchThreshold, 1.0), 0.01, Option::Call, 1.0,
                     kDays1Y, false);

    // -- degenerate limit: hw sigma -> 0 must reproduce AnalyticEuropeanEngine --
    {
        const Real tinySigma = 1.0e-12;
        const auto process = bsmProcess(market);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(market.r), 0.05, tinySigma);
        const Date exDate = maturity(kDays5Y);
        const Real fwd = market.spot * process->dividendYield()->discount(exDate) /
                         process->riskFreeRate()->discount(exDate);

        VanillaOption hybrid(vanilla(Option::Call, fwd), european(kDays5Y));
        hybrid.setPricingEngine(
            ext::make_shared<AnalyticBSMHullWhiteEngine>(0.9, process, hw));

        VanillaOption plain(vanilla(Option::Call, fwd), european(kDays5Y));
        plain.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

        Obj in;
        putBsmMarket(in, market);
        in.n("rho", 0.9);
        in.n("hwA", 0.05);
        in.n("hwSigma", tinySigma);
        in.s("optionType", "Call");
        in.n("strike", fwd);
        in.i("maturityDays", kDays5Y);
        Obj ex;
        ex.n("npv", hybrid.NPV());
        ex.n("analytic_european_npv", plain.NPV());
        addCase("bsmhw_zero_hw_vol_matches_bs", in, ex);
    }

    // -- guards --
    {
        // calculate() QL_REQUIREs x0() > 0.
        const auto process = ext::make_shared<BlackScholesMertonProcess>(
            quote(0.0), flatCurve(market.q), flatCurve(market.r), flatVol(market.vol));
        const auto hw = ext::make_shared<HullWhite>(flatCurve(market.r), 0.05, 0.01);
        VanillaOption option(vanilla(Option::Call, 100.0), european(kDays1Y));
        option.setPricingEngine(
            ext::make_shared<AnalyticBSMHullWhiteEngine>(0.0, process, hw));
        Obj in;
        in.n("spot", 0.0);
        in.s("guard", "process->x0() > 0.0");
        Obj ex;
        ex.b("throws", throwsOnNPV(option));
        addCase("bsmhw_zero_spot_throws", in, ex);
    }
    {
        // No exercise-type check of its own -- the inner AnalyticEuropeanEngine
        // rejects the American exercise.
        const auto process = bsmProcess(market);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(market.r), 0.05, 0.01);
        VanillaOption option(
            vanilla(Option::Call, 100.0),
            ext::make_shared<AmericanExercise>(kToday, maturity(kDays1Y)));
        option.setPricingEngine(
            ext::make_shared<AnalyticBSMHullWhiteEngine>(0.0, process, hw));
        Obj in;
        in.s("exercise", "AmericanExercise");
        in.s("guard", "delegated AnalyticEuropeanEngine: not a European option");
        Obj ex;
        ex.b("throws", throwsOnNPV(option));
        addCase("bsmhw_american_exercise_throws", in, ex);
    }
    {
        // ctor QL_REQUIREs a process and a non-empty model handle.
        bool nullProcessThrows = false;
        try {
            AnalyticBSMHullWhiteEngine(
                0.0, ext::shared_ptr<GeneralizedBlackScholesProcess>(),
                ext::make_shared<HullWhite>(flatCurve(market.r), 0.05, 0.01));
        } catch (const std::exception&) {
            nullProcessThrows = true;
        }
        bool nullModelThrows = false;
        try {
            AnalyticBSMHullWhiteEngine(0.0, bsmProcess(market),
                                       ext::shared_ptr<HullWhite>());
        } catch (const std::exception&) {
            nullModelThrows = true;
        }
        Obj in;
        in.s("guard", "ctor QL_REQUIREs process and non-empty model");
        Obj ex;
        ex.b("null_process_throws", nullProcessThrows);
        ex.b("null_model_throws", nullModelThrows);
        addCase("bsmhw_ctor_guards", in, ex);
    }
}

// ---------------------------------------------------------------------------
// 2. AnalyticBlackVasicekEngine
// ---------------------------------------------------------------------------
struct VasicekParams {
    Rate r0 = 0.05;
    Real a = 0.10;
    Real b = 0.05;
    Real sigma = 0.01;
    Real lambda = 0.0;
};

void putVasicekParams(Obj& in, const VasicekParams& v) {
    in.n("vasicekR0", v.r0);
    in.n("vasicekA", v.a);
    in.n("vasicekB", v.b);
    in.n("vasicekSigma", v.sigma);
    in.n("vasicekLambda", v.lambda);
}

void vasicekCase(const std::string& name,
                 const BsmMarket& market,
                 const VasicekParams& vp,
                 Real correlation,
                 Option::Type type,
                 Real strike,
                 Integer days,
                 bool useTermVol) {
    const auto process = ext::make_shared<BlackScholesMertonProcess>(
        quote(market.spot), flatCurve(market.q), flatCurve(market.r),
        useTermVol ? termVol() : flatVol(market.vol));
    const auto vasicek =
        ext::make_shared<Vasicek>(vp.r0, vp.a, vp.b, vp.sigma, vp.lambda);

    VanillaOption option(vanilla(type, strike), european(days));
    option.setPricingEngine(
        ext::make_shared<AnalyticBlackVasicekEngine>(process, vasicek, correlation));

    Obj in;
    putBsmMarket(in, market);
    putVasicekParams(in, vp);
    in.n("correlation", correlation);
    in.s("optionType", typeName(type));
    in.n("strike", strike);
    in.i("maturityDays", days);
    in.s("volTermStructure", useTermVol ? "BlackVarianceCurve" : "BlackConstantVol");

    Obj ex;
    ex.n("npv", option.NPV());
    addCase(name, in, ex);
}

void emitVasicek() {
    const BsmMarket market;
    const VasicekParams vp;

    const std::pair<const char*, Real> rhos[] = {
        {"m090", -0.90}, {"m050", -0.50}, {"000", 0.0}, {"050", 0.50}, {"090", 0.90}};
    for (const auto& [tag, rho] : rhos) {
        vasicekCase(std::string("vasicek_rho_") + tag, market, vp, rho, Option::Call,
                    100.0, kDays5Y, false);
        vasicekCase(std::string("vasicek_put_rho_") + tag, market, vp, rho, Option::Put,
                    100.0, kDays5Y, false);
    }

    // strike sweep -- ITM / ATM / OTM at fixed correlation
    for (const auto& [tag, k] : {std::pair<const char*, Real>{"k70", 70.0},
                                 {"k100", 100.0},
                                 {"k140", 140.0}}) {
        vasicekCase(std::string("vasicek_call_") + tag, market, vp, 0.4, Option::Call, k,
                    kDays1Y, false);
        vasicekCase(std::string("vasicek_put_") + tag, market, vp, 0.4, Option::Put, k,
                    kDays1Y, false);
    }

    // The vol is sampled at t == 0, not at maturity. Same market, two maturities:
    // if the port read the vol at maturity these two would use 0.15 and 0.35; C++
    // uses the curve's short end for both.
    vasicekCase("vasicek_vol_read_at_time_zero_1y", market, vp, 0.3, Option::Call, 100.0,
                kDays1Y, true);
    vasicekCase("vasicek_vol_read_at_time_zero_5y", market, vp, 0.3, Option::Call, 100.0,
                kDays5Y, true);

    // Small mean reversion -- Vasicek's own B(t,T) low-a branch (a < sqrt(QL_EPSILON)).
    {
        VasicekParams small = vp;
        small.a = 1.0e-9;
        small.b = 0.05;
        vasicekCase("vasicek_low_a", market, small, 0.4, Option::Call, 100.0, kDays5Y,
                    false);
    }

    // -- degenerate limit: rate vol -> 0 with b == r0 makes the Vasicek zero-coupon
    //    bond exactly exp(-r0*T), so the engine collapses onto Black-Scholes with
    //    ZERO dividend yield and discount curve exp(-r0*T).
    {
        VasicekParams deg;
        deg.r0 = 0.05;
        deg.a = 0.10;
        deg.b = 0.05;
        deg.sigma = 1.0e-12;
        deg.lambda = 0.0;

        BsmMarket degMarket;
        degMarket.q = 0.0;
        degMarket.r = 0.05;

        const auto process = ext::make_shared<BlackScholesMertonProcess>(
            quote(degMarket.spot), flatCurve(degMarket.q), flatCurve(degMarket.r),
            flatVol(degMarket.vol));
        const auto vasicek =
            ext::make_shared<Vasicek>(deg.r0, deg.a, deg.b, deg.sigma, deg.lambda);

        VanillaOption hybrid(vanilla(Option::Call, 100.0), european(kDays5Y));
        hybrid.setPricingEngine(
            ext::make_shared<AnalyticBlackVasicekEngine>(process, vasicek, 0.9));

        VanillaOption plain(vanilla(Option::Call, 100.0), european(kDays5Y));
        plain.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

        Obj in;
        putBsmMarket(in, degMarket);
        putVasicekParams(in, deg);
        in.n("correlation", 0.9);
        in.s("optionType", "Call");
        in.n("strike", 100.0);
        in.i("maturityDays", kDays5Y);
        Obj ex;
        ex.n("npv", hybrid.NPV());
        ex.n("analytic_european_npv", plain.NPV());
        addCase("vasicek_zero_rate_vol_matches_bs", in, ex);
    }

    // -- the engine fills only results_.value --
    {
        const auto process = bsmProcess(market);
        const auto vasicek = ext::make_shared<Vasicek>(vp.r0, vp.a, vp.b, vp.sigma,
                                                       vp.lambda);
        VanillaOption option(vanilla(Option::Call, 100.0), european(kDays1Y));
        option.setPricingEngine(
            ext::make_shared<AnalyticBlackVasicekEngine>(process, vasicek, 0.4));
        option.NPV();
        Obj in;
        in.s("note", "AnalyticBlackVasicekEngine::calculate assigns only results_.value");
        Obj ex;
        putGreek(ex, "delta", [&] { return option.delta(); });
        putGreek(ex, "gamma", [&] { return option.gamma(); });
        putGreek(ex, "vega", [&] { return option.vega(); });
        addCase("vasicek_no_greeks", in, ex);
    }

    // -- guard: European only --
    {
        const auto process = bsmProcess(market);
        const auto vasicek = ext::make_shared<Vasicek>(vp.r0, vp.a, vp.b, vp.sigma,
                                                       vp.lambda);
        VanillaOption option(
            vanilla(Option::Call, 100.0),
            ext::make_shared<AmericanExercise>(kToday, maturity(kDays1Y)));
        option.setPricingEngine(
            ext::make_shared<AnalyticBlackVasicekEngine>(process, vasicek, 0.4));
        Obj in;
        in.s("exercise", "AmericanExercise");
        in.s("guard", "not an European option");
        Obj ex;
        ex.b("throws", throwsOnNPV(option));
        addCase("vasicek_american_exercise_throws", in, ex);
    }
}

// ---------------------------------------------------------------------------
// 3 + 4. AnalyticHestonHullWhiteEngine / AnalyticH1HWEngine
// ---------------------------------------------------------------------------
struct HestonParams {
    Real spot = 100.0;
    Rate r = 0.03;
    Rate q = 0.02;
    Real v0 = 0.08;
    Real kappa = 1.5;
    Real theta = 0.0625;
    Real sigma = 0.5;
    Real rho = -0.8;
};

// The converged adaptive configuration -- see header note 8.
const Real kRelTol = 1.0e-13;
const Size kMaxEval = 1000000;

ext::shared_ptr<HestonModel> hestonModel(const HestonParams& p) {
    return ext::make_shared<HestonModel>(ext::make_shared<HestonProcess>(
        flatCurve(p.r), flatCurve(p.q), quote(p.spot), p.v0, p.kappa, p.theta, p.sigma,
        p.rho));
}

void putHestonParams(Obj& in, const HestonParams& p) {
    in.n("spot", p.spot);
    in.n("r", p.r);
    in.n("q", p.q);
    in.n("hestonV0", p.v0);
    in.n("hestonKappa", p.kappa);
    in.n("hestonTheta", p.theta);
    in.n("hestonSigma", p.sigma);
    in.n("hestonRho", p.rho);
    in.s("dayCounter", "Actual365Fixed");
    in.n("relTolerance", kRelTol);
    in.i("maxEvaluations", static_cast<long long>(kMaxEval));
}

void hhwCase(const std::string& name,
             const HestonParams& p,
             Real hwA,
             Real hwSigma,
             Option::Type type,
             Real strike,
             Integer days) {
    const auto model = hestonModel(p);
    const auto hw = ext::make_shared<HullWhite>(flatCurve(p.r), hwA, hwSigma);

    VanillaOption option(vanilla(type, strike), european(days));
    option.setPricingEngine(ext::make_shared<AnalyticHestonHullWhiteEngine>(
        model, hw, kRelTol, kMaxEval));

    Obj in;
    putHestonParams(in, p);
    in.n("hwA", hwA);
    in.n("hwSigma", hwSigma);
    in.s("optionType", typeName(type));
    in.n("strike", strike);
    in.i("maturityDays", days);

    Obj ex;
    ex.n("npv", option.NPV());
    addCase(name, in, ex);
}

void emitHestonHullWhite() {
    const HestonParams p;

    for (const auto& [tag, k] : {std::pair<const char*, Real>{"k80", 80.0},
                                 {"k100", 100.0},
                                 {"k120", 120.0}}) {
        hhwCase(std::string("hhw_call_") + tag + "_5y", p, 0.01, 0.01, Option::Call, k,
                kDays5Y);
        hhwCase(std::string("hhw_put_") + tag + "_5y", p, 0.01, 0.01, Option::Put, k,
                kDays5Y);
    }

    // The Hull-White volatility must move the price.
    for (const auto& [tag, s] : {std::pair<const char*, Real>{"tiny", 1.0e-12},
                                 {"0005", 0.005},
                                 {"001", 0.01},
                                 {"002", 0.02},
                                 {"004", 0.04}})
        hhwCase(std::string("hhw_hwvol_") + tag, p, 0.01, s, Option::Call, 100.0,
                kDays5Y);

    // The a -> 0 algebraic branch of AnalyticHestonHullWhiteEngine::calculate,
    // straddled at t == 1.0 (Actual365Fixed over exactly 365 days).
    hhwCase("hhw_low_a_branch", p, 1.0e-8, 0.01, Option::Call, 100.0, kDays1Y);
    hhwCase("hhw_a_at_threshold", p, kLowABranchThreshold, 0.01, Option::Call, 100.0,
            kDays1Y);
    hhwCase("hhw_a_just_above_threshold", p, std::nextafter(kLowABranchThreshold, 1.0),
            0.01, Option::Call, 100.0, kDays1Y);

    // -- degenerate limit: hw sigma -> 0 must reproduce the plain Heston price --
    {
        const Real tinySigma = 1.0e-12;
        const auto model = hestonModel(p);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(p.r), 0.01, tinySigma);

        VanillaOption hybrid(vanilla(Option::Call, 100.0), european(kDays5Y));
        hybrid.setPricingEngine(ext::make_shared<AnalyticHestonHullWhiteEngine>(
            model, hw, kRelTol, kMaxEval));

        VanillaOption plain(vanilla(Option::Call, 100.0), european(kDays5Y));
        plain.setPricingEngine(ext::make_shared<AnalyticHestonEngine>(
            model, AnalyticHestonEngine::Gatheral,
            AnalyticHestonEngine::Integration::gaussLobatto(kRelTol, Null<Real>(),
                                                            kMaxEval)));

        Obj in;
        putHestonParams(in, p);
        in.n("hwA", 0.01);
        in.n("hwSigma", tinySigma);
        in.s("optionType", "Call");
        in.n("strike", 100.0);
        in.i("maturityDays", kDays5Y);
        Obj ex;
        ex.n("npv", hybrid.NPV());
        ex.n("analytic_heston_npv", plain.NPV());
        addCase("hhw_zero_hwvol_collapses_to_heston", in, ex);
    }

    // -- the size of the quadrature gap between the two C++ constructors --
    {
        const auto model = hestonModel(p);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(p.r), 0.01, 0.01);

        VanillaOption laguerre(vanilla(Option::Call, 100.0), european(kDays5Y));
        laguerre.setPricingEngine(
            ext::make_shared<AnalyticHestonHullWhiteEngine>(model, hw, Size(144)));

        VanillaOption lobatto(vanilla(Option::Call, 100.0), european(kDays5Y));
        lobatto.setPricingEngine(ext::make_shared<AnalyticHestonHullWhiteEngine>(
            model, hw, kRelTol, kMaxEval));

        Obj in;
        putHestonParams(in, p);
        in.n("hwA", 0.01);
        in.n("hwSigma", 0.01);
        in.s("optionType", "Call");
        in.n("strike", 100.0);
        in.i("maturityDays", kDays5Y);
        in.i("integrationOrder", 144);
        Obj ex;
        ex.n("gauss_laguerre_144_npv", laguerre.NPV());
        ex.n("gauss_lobatto_converged_npv", lobatto.NPV());
        addCase("hhw_gausslaguerre144_vs_lobatto", in, ex);
    }

    // -- Integration::gaussLaguerre QL_REQUIREs intOrder <= 192 --
    {
        const auto model = hestonModel(p);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(p.r), 0.01, 0.01);
        bool throws = false;
        try {
            AnalyticHestonHullWhiteEngine(model, hw, Size(1024));
        } catch (const std::exception&) {
            throws = true;
        }
        Obj in;
        in.i("integrationOrder", 1024);
        in.s("guard", "maximum integraton order (192) exceeded");
        Obj ex;
        ex.b("throws", throws);
        addCase("hhw_integration_order_over_192_throws", in, ex);
    }

    // -- no greeks, European only --
    {
        const auto model = hestonModel(p);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(p.r), 0.01, 0.01);
        VanillaOption option(vanilla(Option::Call, 100.0), european(kDays1Y));
        option.setPricingEngine(ext::make_shared<AnalyticHestonHullWhiteEngine>(
            model, hw, kRelTol, kMaxEval));
        option.NPV();
        Obj in;
        in.s("note", "inherits AnalyticHestonEngine::calculate -- value only");
        Obj ex;
        putGreek(ex, "delta", [&] { return option.delta(); });
        putGreek(ex, "vega", [&] { return option.vega(); });
        addCase("hhw_no_greeks", in, ex);
    }
    {
        const auto model = hestonModel(p);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(p.r), 0.01, 0.01);
        VanillaOption option(
            vanilla(Option::Call, 100.0),
            ext::make_shared<AmericanExercise>(kToday, maturity(kDays1Y)));
        option.setPricingEngine(ext::make_shared<AnalyticHestonHullWhiteEngine>(
            model, hw, kRelTol, kMaxEval));
        Obj in;
        in.s("exercise", "AmericanExercise");
        in.s("guard", "not an European option");
        Obj ex;
        ex.b("throws", throwsOnNPV(option));
        addCase("hhw_american_exercise_throws", in, ex);
    }
}

// ---- H1-HW ---------------------------------------------------------------
// Grzelak / Oosterlee parameter set, re-anchored onto this probe's evaluation
// date. The two Heston vol-of-vols select the two branches of
// AnalyticH1HWEngine::Fj_Helper::operator(): 8*kappa*theta/gamma^2 is
// 8*0.3*0.05/0.09 = 1.333 > 1 for sigma_v = 0.3 (closed form for a, b, c) and
// 8*0.3*0.05/0.36 = 0.333 <= 1 for sigma_v = 0.6 (which instead calls the
// truncated hypergeometric series Lambda(1/kappa)).
struct H1hwParams {
    Real spot = 100.0;
    Rate r = 0.02;
    Rate q = 0.00;
    Real v0 = 0.05;
    Real theta = 0.05;
    Real kappa = 0.3;
    Real sigma = 0.3;
    Real rhoSv = -0.30;
    Real kappaR = 0.01;
    Real sigmaR = 0.01;
};

ext::shared_ptr<HestonModel> h1hwHestonModel(const H1hwParams& p) {
    return ext::make_shared<HestonModel>(ext::make_shared<HestonProcess>(
        flatCurve(p.r), flatCurve(p.q), quote(p.spot), p.v0, p.kappa, p.theta, p.sigma,
        p.rhoSv));
}

void putH1hwParams(Obj& in, const H1hwParams& p) {
    in.n("spot", p.spot);
    in.n("r", p.r);
    in.n("q", p.q);
    in.n("hestonV0", p.v0);
    in.n("hestonKappa", p.kappa);
    in.n("hestonTheta", p.theta);
    in.n("hestonSigma", p.sigma);
    in.n("hestonRho", p.rhoSv);
    in.n("hwA", p.kappaR);
    in.n("hwSigma", p.sigmaR);
    in.s("dayCounter", "Actual365Fixed");
    in.n("relTolerance", kRelTol);
    in.i("maxEvaluations", static_cast<long long>(kMaxEval));
    in.n("fellerRatio", 8.0 * p.kappa * p.theta / (p.sigma * p.sigma));
}

void h1hwCase(const std::string& name,
              const H1hwParams& p,
              Real rhoSr,
              Option::Type type,
              Real strike,
              Integer days) {
    const auto model = h1hwHestonModel(p);
    const auto hw = ext::make_shared<HullWhite>(flatCurve(p.r), p.kappaR, p.sigmaR);

    VanillaOption option(vanilla(type, strike), european(days));
    option.setPricingEngine(
        ext::make_shared<AnalyticH1HWEngine>(model, hw, rhoSr, kRelTol, kMaxEval));

    Obj in;
    putH1hwParams(in, p);
    in.n("rhoSr", rhoSr);
    in.s("optionType", typeName(type));
    in.n("strike", strike);
    in.i("maturityDays", days);

    Obj ex;
    ex.n("npv", option.NPV());
    addCase(name, in, ex);
}

void emitH1hw() {
    H1hwParams lowVolOfVol;          // sigma_v = 0.3 -> first Fj_Helper branch
    H1hwParams highVolOfVol;
    highVolOfVol.sigma = 0.6;        // sigma_v = 0.6 -> Lambda() series branch

    const Real strikes[] = {40.0, 80.0, 100.0, 120.0, 180.0};
    for (Real k : strikes) {
        std::ostringstream tag;
        tag << "k" << static_cast<int>(k);
        h1hwCase("h1hw_sv030_" + tag.str(), lowVolOfVol, 0.6, Option::Call, k, kDays10Y);
        h1hwCase("h1hw_sv060_" + tag.str(), highVolOfVol, 0.6, Option::Call, k, kDays10Y);
    }
    h1hwCase("h1hw_sv030_k100_put", lowVolOfVol, 0.6, Option::Put, 100.0, kDays10Y);

    // rhoSr must move the price.
    for (const auto& [tag, rho] : {std::pair<const char*, Real>{"000", 0.0},
                                   {"030", 0.3},
                                   {"060", 0.6},
                                   {"090", 0.9}})
        h1hwCase(std::string("h1hw_rhosr_") + tag, lowVolOfVol, rho, Option::Call, 100.0,
                 kDays10Y);

    // -- degenerate limit: rhoSr == 0 kills the whole Fj_Helper term (it returns
    //    eta*rhoSr*I4), so H1HW must equal its AnalyticHestonHullWhiteEngine parent.
    {
        const auto model = h1hwHestonModel(lowVolOfVol);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(lowVolOfVol.r),
                                                    lowVolOfVol.kappaR,
                                                    lowVolOfVol.sigmaR);
        VanillaOption h1hw(vanilla(Option::Call, 100.0), european(kDays10Y));
        h1hw.setPricingEngine(
            ext::make_shared<AnalyticH1HWEngine>(model, hw, 0.0, kRelTol, kMaxEval));

        VanillaOption hhw(vanilla(Option::Call, 100.0), european(kDays10Y));
        hhw.setPricingEngine(ext::make_shared<AnalyticHestonHullWhiteEngine>(
            model, hw, kRelTol, kMaxEval));

        Obj in;
        putH1hwParams(in, lowVolOfVol);
        in.n("rhoSr", 0.0);
        in.s("optionType", "Call");
        in.n("strike", 100.0);
        in.i("maturityDays", kDays10Y);
        Obj ex;
        ex.n("npv", h1hw.NPV());
        ex.n("heston_hull_white_npv", hhw.NPV());
        addCase("h1hw_rhosr_zero_matches_hhw", in, ex);
    }

    // -- ctor asymmetry: only the integrationOrder overload rejects rhoSr < 0 --
    {
        const auto model = h1hwHestonModel(lowVolOfVol);
        const auto hw = ext::make_shared<HullWhite>(flatCurve(lowVolOfVol.r),
                                                    lowVolOfVol.kappaR,
                                                    lowVolOfVol.sigmaR);
        bool orderCtorThrows = false;
        try {
            AnalyticH1HWEngine(model, hw, -0.5, Size(144));
        } catch (const std::exception&) {
            orderCtorThrows = true;
        }
        bool tolCtorThrows = false;
        Real negRhoNpv = 0.0;
        try {
            VanillaOption option(vanilla(Option::Call, 100.0), european(kDays10Y));
            option.setPricingEngine(
                ext::make_shared<AnalyticH1HWEngine>(model, hw, -0.5, kRelTol, kMaxEval));
            negRhoNpv = option.NPV();
        } catch (const std::exception&) {
            tolCtorThrows = true;
        }
        Obj in;
        putH1hwParams(in, lowVolOfVol);
        in.n("rhoSr", -0.5);
        in.s("guard",
             "integrationOrder ctor QL_REQUIREs rhoSr >= 0; relTolerance ctor does not");
        in.s("note",
             "the value the relTolerance ctor then produces is the instability the "
             "other ctor's QL_REQUIRE is warning about: it is not an option price and "
             "it is not reproducible across integrators, so only its absurdity is "
             "pinned, never its digits");
        Obj ex;
        ex.b("order_ctor_throws", orderCtorThrows);
        ex.b("tolerance_ctor_throws", tolCtorThrows);
        // Deliberately NOT the number: |NPV| here is ~3e16 on a spot of 100.
        ex.b("tolerance_ctor_npv_exceeds_spot",
             !tolCtorThrows && std::fabs(negRhoNpv) > lowVolOfVol.spot);
        addCase("h1hw_negative_rhosr_ctor_asymmetry", in, ex);
    }
}

// ---------------------------------------------------------------------------
// 5. CEVCalculator + AnalyticCEVEngine
// ---------------------------------------------------------------------------
void emitCev() {
    // fdcev.cpp testFdmCevOp market.
    const Real f0 = 2.1;
    const Real alpha = 0.75;
    const Rate cevRate = 0.15;
    const Time t = 1.0;

    const Real betas[] = {-2.0, -0.5, 0.45, 0.6, 0.9, 0.99, 1.01, 1.45, 2.0};
    const Real strikes[] = {1.5, 2.1, 2.3, 3.0};

    for (Real beta : betas) {
        const CEVCalculator calc(f0, alpha, beta);
        std::ostringstream btag;
        btag << "b" << std::fixed << std::setprecision(2) << beta;
        std::string tag = btag.str();
        // "-" and "." are awkward in JSON keys; normalise.
        for (char& ch : tag) {
            if (ch == '-')
                ch = 'm';
            if (ch == '.')
                ch = '_';
        }

        for (Real k : strikes) {
            std::ostringstream ktag;
            ktag << "k" << std::fixed << std::setprecision(1) << k;
            std::string kt = ktag.str();
            for (char& ch : kt)
                if (ch == '.')
                    ch = '_';

            for (Option::Type type : {Option::Call, Option::Put}) {
                Obj in;
                in.n("f0", f0);
                in.n("alpha", alpha);
                in.n("beta", beta);
                in.n("strike", k);
                in.n("t", t);
                in.s("optionType", typeName(type));
                in.n("delta", (1.0 - 2.0 * beta) / (1.0 - beta));
                Obj ex;
                ex.n("value", calc.value(type, k, t));
                addCase("cev_calc_" + tag + "_" + kt + "_" +
                            (type == Option::Call ? "call" : "put"),
                        in, ex);
            }
        }

        // Inspectors -- f0()/alpha()/beta() are the whole public read surface.
        Obj in;
        in.n("f0", f0);
        in.n("alpha", alpha);
        in.n("beta", beta);
        Obj ex;
        ex.n("f0", calc.f0());
        ex.n("alpha", calc.alpha());
        ex.n("beta", calc.beta());
        addCase("cev_calc_inspectors_" + tag, in, ex);
    }

    // Deliberately deep in the tail. Both terms of the beta < 1 call formula are
    // evaluated where the non-central chi-squared CDF is far from 1/2, which is
    // exactly where boost's continued fraction and QuantLib's own truncated
    // series (NonCentralCumulativeChiSquareDistribution, whose stopping rule is
    // an ABSOLUTE 1e-12 bound) part company. A port that reaches for the wrong
    // routine reproduces the at-the-money cases and fails here. The (5.0, 0.25)
    // point still returns a small nonzero value; the (40.0, 0.05) point
    // underflows to exactly 0.0 in C++ and that zero is pinned too.
    {
        const Real farBeta = 0.45;
        const CEVCalculator calc(f0, alpha, farBeta);
        const std::pair<Real, Time> points[] = {
            {5.0, 0.25}, {5.0, 1.0}, {8.0, 1.0}, {4.0, 0.1}, {40.0, 0.05}};
        int idx = 0;
        for (const auto& [k, tt] : points) {
            Obj in;
            in.n("f0", f0);
            in.n("alpha", alpha);
            in.n("beta", farBeta);
            in.n("strike", k);
            in.n("t", tt);
            in.s("optionType", "Call");
            in.s("note", "deep OTM -- boost cdf vs QuantLib's own series diverge here");
            Obj ex;
            ex.n("value", calc.value(Option::Call, k, tt));
            addCase("cev_calc_far_tail_call_" + std::to_string(idx++), in, ex);
        }
    }

    // -- engine wrapping the calculator --
    for (Real beta : {-2.0, 0.45, 0.9, 1.45}) {
        for (Option::Type type : {Option::Call, Option::Put}) {
            std::ostringstream btag;
            btag << "b" << std::fixed << std::setprecision(2) << beta;
            std::string tag = btag.str();
            for (char& ch : tag) {
                if (ch == '-')
                    ch = 'm';
                if (ch == '.')
                    ch = '_';
            }

            const auto rTS = flatCurve(cevRate);
            VanillaOption option(vanilla(type, 2.3), european(kDays1Y));
            option.setPricingEngine(
                ext::make_shared<AnalyticCEVEngine>(f0, alpha, beta, rTS));

            Obj in;
            in.n("f0", f0);
            in.n("alpha", alpha);
            in.n("beta", beta);
            in.n("strike", 2.3);
            in.n("discountRate", cevRate);
            in.i("maturityDays", kDays1Y);
            in.s("optionType", typeName(type));
            Obj ex;
            ex.n("npv", option.NPV());
            ex.n("discount", rTS->discount(maturity(kDays1Y)));
            addCase("cev_engine_" + tag + "_" +
                        (type == Option::Call ? "call" : "put"),
                    in, ex);
        }
    }

    // -- put-call parity holds for beta < 1 (true martingale) and FAILS for
    //    beta > 1 (strict local martingale). Both are pinned. --
    for (const auto& [tag, beta] : {std::pair<const char*, Real>{"beta045", 0.45},
                                    {"beta145", 1.45}}) {
        const auto rTS = flatCurve(cevRate);
        const Real df = rTS->discount(maturity(kDays1Y));
        VanillaOption call(vanilla(Option::Call, 2.3), european(kDays1Y));
        call.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(f0, alpha, beta, rTS));
        VanillaOption put(vanilla(Option::Put, 2.3), european(kDays1Y));
        put.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(f0, alpha, beta, rTS));

        Obj in;
        in.n("f0", f0);
        in.n("alpha", alpha);
        in.n("beta", beta);
        in.n("strike", 2.3);
        in.n("discount", df);
        Obj ex;
        ex.n("call_npv", call.NPV());
        ex.n("put_npv", put.NPV());
        ex.n("parity_residual", call.NPV() - put.NPV() - (f0 - 2.3) * df);
        addCase(std::string("cev_parity_") + tag, in, ex);
    }

    // -- guards --
    {
        VanillaOption option(vanilla(Option::Call, 2.3),
                             ext::make_shared<AmericanExercise>(kToday,
                                                                maturity(kDays1Y)));
        option.setPricingEngine(
            ext::make_shared<AnalyticCEVEngine>(f0, alpha, 0.45, flatCurve(cevRate)));
        Obj in;
        in.s("exercise", "AmericanExercise");
        in.s("guard", "not an European option");
        Obj ex;
        ex.b("throws", throwsOnNPV(option));
        addCase("cev_engine_american_exercise_throws", in, ex);
    }
    {
        VanillaOption option(vanilla(Option::Call, 2.3), european(kDays1Y));
        option.setPricingEngine(
            ext::make_shared<AnalyticCEVEngine>(f0, alpha, 0.45, flatCurve(cevRate)));
        option.NPV();
        Obj in;
        in.s("note", "AnalyticCEVEngine::calculate assigns only results_.value");
        Obj ex;
        putGreek(ex, "delta", [&] { return option.delta(); });
        putGreek(ex, "vega", [&] { return option.vega(); });
        addCase("cev_engine_no_greeks", in, ex);
    }
}

// ---------------------------------------------------------------------------
// 6. AnalyticGJRGARCHEngine
// ---------------------------------------------------------------------------
struct GjrMarket {
    Real spot;
    Rate r;
    Rate q;
    Real v0;
    Real omega;
    Real alpha;
    Real beta;
    Real gamma;
    Real lambda;
    Real daysPerYear;
};

ext::shared_ptr<GJRGARCHModel> gjrModel(const GjrMarket& m) {
    return ext::make_shared<GJRGARCHModel>(ext::make_shared<GJRGARCHProcess>(
        flatCurve(m.r), flatCurve(m.q), quote(m.spot), m.v0, m.omega, m.alpha, m.beta,
        m.gamma, m.lambda, m.daysPerYear));
}

void putGjrMarket(Obj& in, const GjrMarket& m) {
    in.n("spot", m.spot);
    in.n("r", m.r);
    in.n("q", m.q);
    in.n("v0", m.v0);
    in.n("omega", m.omega);
    in.n("alpha", m.alpha);
    in.n("beta", m.beta);
    in.n("gamma", m.gamma);
    in.n("lambda", m.lambda);
    in.n("daysPerYear", m.daysPerYear);
    in.s("dayCounter", "Actual365Fixed");
}

void emitGjrGarch() {
    // -- the Duan/Gauthier/Simonato/Sasseville (2006) grid, exactly as the v1.43
    //    test-suite gjrgarchmodel.cpp testEngines sets it up. v0 = omega/(1-m1)
    //    with m1 the stationary first moment, which is what the C++ test computes
    //    inline; it is spelled out in `inputs` so the port reconstructs it. --
    const Real omega = 2.0e-6;
    const Real alpha = 0.024;
    const Real beta = 0.93;
    const Real gamma = 0.059;
    const Real lambdas[] = {0.0, 0.1, 0.2};
    const Integer maturities[] = {90, 180};
    const Real strikes[] = {35.0, 40.0, 45.0, 50.0, 55.0, 60.0};

    for (Real lambda : lambdas) {
        const Real nCdf = CumulativeNormalDistribution()(lambda);
        const Real nPdf = std::exp(-lambda * lambda / 2.0) / std::sqrt(2.0 * M_PI);
        const Real m1 = beta + (alpha + gamma * nCdf) * (1.0 + lambda * lambda) +
                        gamma * lambda * nPdf;
        const Real v0 = omega / (1.0 - m1);

        GjrMarket m{50.0, 0.05, 0.0, v0, omega, alpha, beta, gamma, lambda, 365.0};
        const auto model = gjrModel(m);

        std::ostringstream ltag;
        ltag << "lam" << static_cast<int>(std::lround(lambda * 100));

        for (Integer days : maturities) {
            for (Real k : strikes) {
                for (Option::Type type : {Option::Call, Option::Put}) {
                    VanillaOption option(vanilla(type, k), european(days));
                    option.setPricingEngine(
                        ext::make_shared<AnalyticGJRGARCHEngine>(model));

                    Obj in;
                    putGjrMarket(in, m);
                    in.n("m1", m1);
                    in.n("strike", k);
                    in.i("maturityDays", days);
                    in.s("optionType", typeName(type));
                    Obj ex;
                    ex.n("npv", option.NPV());

                    std::ostringstream name;
                    name << "gjrgarch_" << ltag.str() << "_d" << days << "_k"
                         << static_cast<int>(k) << "_"
                         << (type == Option::Call ? "call" : "put");
                    addCase(name.str(), in, ex);
                }
            }
        }
    }

    // -- the market the existing PQuantLib test already used (via the
    //    phase-11 cluster/w1d probe), re-anchored onto this probe's evaluation
    //    date. Flat curves and a 365-day tenor make it identical arithmetic, so
    //    these two values must equal the w1d ones bit-for-bit. --
    {
        GjrMarket m{100.0, 0.05, 0.0, 0.000160, 0.000002, 0.024, 0.93, 0.059, 0.2, 252.0};
        const auto model = gjrModel(m);
        for (Option::Type type : {Option::Call, Option::Put}) {
            VanillaOption option(vanilla(type, 100.0), european(kDays1Y));
            option.setPricingEngine(ext::make_shared<AnalyticGJRGARCHEngine>(model));
            Obj in;
            putGjrMarket(in, m);
            in.n("strike", 100.0);
            in.i("maturityDays", kDays1Y);
            in.s("optionType", typeName(type));
            Obj ex;
            ex.n("npv", option.NPV());
            addCase(std::string("gjrgarch_atm_1y_") +
                        (type == Option::Call ? "call" : "put"),
                    in, ex);
        }
    }

    // -- guards --
    {
        GjrMarket m{50.0, 0.05, 0.0, 0.0001, 2.0e-6, 0.024, 0.93, 0.059, 0.2, 365.0};
        const auto model = gjrModel(m);
        VanillaOption option(
            vanilla(Option::Call, 50.0),
            ext::make_shared<AmericanExercise>(kToday, maturity(90)));
        option.setPricingEngine(ext::make_shared<AnalyticGJRGARCHEngine>(model));
        Obj in;
        in.s("exercise", "AmericanExercise");
        in.s("guard", "not an European option");
        Obj ex;
        ex.b("throws", throwsOnNPV(option));
        addCase("gjrgarch_american_exercise_throws", in, ex);
    }
    {
        GjrMarket m{50.0, 0.05, 0.0, 0.0001, 2.0e-6, 0.024, 0.93, 0.059, 0.2, 365.0};
        const auto model = gjrModel(m);
        VanillaOption option(vanilla(Option::Call, 50.0), european(90));
        option.setPricingEngine(ext::make_shared<AnalyticGJRGARCHEngine>(model));
        option.NPV();
        Obj in;
        in.s("note", "AnalyticGJRGARCHEngine::calculate assigns only results_.value");
        Obj ex;
        putGreek(ex, "delta", [&] { return option.delta(); });
        putGreek(ex, "vega", [&] { return option.vega(); });
        addCase("gjrgarch_no_greeks", in, ex);
    }
    {
        // calculate() QL_REQUIREs spotPrice > 0.
        GjrMarket m{0.0, 0.05, 0.0, 0.0001, 2.0e-6, 0.024, 0.93, 0.059, 0.2, 365.0};
        const auto model = gjrModel(m);
        VanillaOption option(vanilla(Option::Call, 50.0), european(90));
        option.setPricingEngine(ext::make_shared<AnalyticGJRGARCHEngine>(model));
        Obj in;
        in.n("spot", 0.0);
        in.s("guard", "negative or null underlying given");
        Obj ex;
        ex.b("throws", throwsOnNPV(option));
        addCase("gjrgarch_zero_spot_throws", in, ex);
    }
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    Obj meta;
    meta.s("evaluationDate", "2025-03-01");
    meta.i("evaluationDateSerial", static_cast<long long>(kToday.serialNumber()));
    meta.s("dayCounter", "Actual365Fixed");
    meta.n("lowABranchThreshold", kLowABranchThreshold);
    Obj metaExpected;
    metaExpected.b("pinned", true);
    addCase("_meta", meta, metaExpected);

    emitBsmHullWhite();
    emitVasicek();
    emitHestonHullWhite();
    emitH1hw();
    emitCev();
    emitGjrGarch();

    emitDocument();
    return 0;
}
