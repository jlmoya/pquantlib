// migration-harness/cpp/probes/v143_pe_heston/probe.cpp
//
// Reference values for the Heston/Bates family of vanilla pricing engines in
// C++ QuantLib v1.43 that PQuantLib had either not ported at all or had ported
// under an invented class name:
//
//   * COSHestonEngine                 (ql/pricingengines/vanilla/coshestonengine.{hpp,cpp})
//   * ExponentialFittingHestonEngine  (ql/pricingengines/vanilla/exponentialfittinghestonengine.{hpp,cpp})
//   * AnalyticPTDHestonEngine         (ql/pricingengines/vanilla/analyticptdhestonengine.{hpp,cpp})
//   * HestonExpansion (abstract) + LPP2HestonExpansion + LPP3HestonExpansion
//     + FordeHestonExpansion + HestonExpansionEngine
//                                     (ql/pricingengines/vanilla/hestonexpansionengine.{hpp,cpp})
//   * BatesDetJumpEngine, BatesDoubleExpEngine, BatesDoubleExpDetJumpEngine
//                                     (ql/pricingengines/vanilla/batesengine.{hpp,cpp})
//
// What has to be pinned, and why
// ------------------------------
// Every one of these engines writes only `results_.value`. VanillaOption's
// results carry a Greeks block (delta/gamma/theta/vega/rho/dividendRho) plus
// OneAssetOption's `additionalResults`, and NONE of these engines fills any of
// them: `option.delta()` throws "delta not provided". NPV is therefore the whole
// observable surface, and the `*_no_greeks` cases pin that emptiness explicitly
// so a port does not invent greeks C++ does not produce.
//
//  1. COSHestonEngine is a Fourier-cosine (Fang-Oosterlee) pricer whose accuracy
//     is entirely governed by two knobs, `L` (truncation width in standard
//     deviations) and `N` (number of cosine terms). Both are varied below; a
//     port that hardcodes the defaults (L=16, N=200) reproduces only the
//     `_L16_N200` cases. Beyond NPV the engine exposes a PUBLIC cumulant /
//     moment surface -- chF, c1..c4, mu, var, skew, kurtosis -- and those are
//     pinned directly. They are the single easiest place for a port to go
//     wrong invisibly: the c2/c3/c4 closed forms are enormous Mathematica-
//     generated expressions, a port that re-derives them instead of
//     transliterating gets NPVs that still look plausible because c1/c2 only
//     set the integration window.
//     NOTE (easy to miss): COSHestonEngine::muT() -- the drift log(qDiscount /
//     rDiscount) -- is PRIVATE and never called anywhere in v1.43. c1() is the
//     *driftless* first cumulant and mu() == c1(). A port that "fixes" this by
//     folding the drift into c1 will fail every cumulant case here.
//     The cumulants also depend ONLY on (v0, kappa, theta, sigma, rho, t) --
//     not on rates, spot or strike -- which is why the cumulant cases carry no
//     market at all.
//
//  2. ExponentialFittingHestonEngine's `ControlVariate` is a typedef of
//     AnalyticHestonEngine::ComplexLogFormula, and the constructor argument
//     genuinely changes the answer: it selects between the Andersen-Piterbarg
//     Black control variate, the asymptotic-chF control variate, and (for
//     OptimalCV) a runtime choice between AsymptoticChF and AngledContour.
//     All of them are pinned, at parameter sets that resolve OptimalCV BOTH
//     ways -- and `optimalControlVariate()` itself is pinned as a string per
//     (parameter set, t) so a port cannot fake the selector. A port that
//     accepts the enum and ignores it still returns plausible prices; the
//     `expfit_cv_*` cases at the same strike/maturity differ in the 5th-10th
//     digit, which is exactly the signature such a bug hides behind.
//     Gatheral and BranchCorrection are rejected by an explicit QL_REQUIRE;
//     those two cases pin `{"throws": true}`.
//
//  3. HestonExpansionEngine's `HestonExpansionFormula` (LPP2 / LPP3 / Forde) is
//     pinned for all three values, and each HestonExpansion subclass's
//     `impliedVolatility(strike, forward)` is pinned DIRECTLY as well, not only
//     through the engine -- the subclasses are public API used stand-alone
//     during calibration (one expansion per expiry, many strikes). Forde's
//     expansion is a genuinely small-time asymptotic and "breaks down for long
//     maturities" per the C++ test-suite's own comment, so terms span 1 week to
//     10 years to exercise both regimes. The `max(1e-8, ...)` clamp inside
//     LPP2/LPP3::impliedVolatility (and `max(1e-8, var)` inside Forde's) is a
//     real branch at extreme strikes; the `wing` cases exist to hit it.
//     NOTE: the engine prices via blackFormula(payoff, forward, vol*sqrt(term),
//     riskFreeDiscount, 0) -- displacement 0, and the DISCOUNT is the risk-free
//     discount while the forward already contains the dividend discount.
//
//  4. The three Bates engines differ from BatesEngine only through the virtual
//     `addOnTerm(phi, t, j)` hook, and their models are distinct parameter
//     hierarchies (BatesDetJumpModel adds kappaLambda/thetaLambda,
//     BatesDoubleExpModel replaces the lognormal jump law with Kou's asymmetric
//     double exponential, BatesDoubleExpDetJumpModel does both). Non-default
//     model parameters are used everywhere so a port cannot pass by leaving a
//     parameter at its C++ default. `i = (j == 1) ? 1.0 : 0.0` and the complex
//     ctor order `complex<Real> g(i, phi)` are the classic transliteration
//     traps, so both j-branches are exercised at every case (they always are:
//     the engine integrates both).
//     The integration ORDER is varied down to deliberately-too-small values
//     (1, 2, 4, 8) because those are the only cases that can detect an engine
//     which accepts `integrationOrder` and silently discards it -- at order 144
//     every sane quadrature agrees to 1e-13. Order 1024 is pinned as
//     `{"throws": true}`: GaussLaguerreIntegration refuses orders above 192.
//
//  5. AnalyticPTDHestonEngine is pinned on a genuinely piecewise model: THREE
//     segments with four different (kappa, theta, sigma, rho) each. A
//     single-segment model degenerates to plain Heston and hides every
//     piecewise bug, so the degenerate case appears only once, as an explicit
//     consistency anchor. Both ComplexLogFormula values (Gatheral,
//     AndersenPiterbarg) are pinned at identical market/strike/maturity so the
//     two must differ; so are both integration flavours (Gauss-Laguerre of a
//     given order, adaptive Gauss-Lobatto) and `numberOfEvaluations()`.
//     chF/lnChF are pinned directly.
//     NOTE: AnalyticPTDHestonEngine::ComplexLogFormula is its OWN two-valued
//     enum { Gatheral, AndersenPiterbarg }, NOT AnalyticHestonEngine's
//     eight-valued one; only `Integration` is a typedef of the latter's.
//     NOTE: the Gatheral branch's Fj_Helper clamps phi to
//     `max(numeric_limits<float>::epsilon(), phi)` -- FLOAT epsilon
//     (1.1920929e-07), not double epsilon. A port using double epsilon changes
//     the integrand at the left endpoint.
//     NOTE: the PTD engine takes the spot/rates off the MODEL
//     (model_->s0(), model_->riskFreeRate()), not off a process, and the
//     maturity is `riskFreeRate()->dayCounter().yearFraction(referenceDate,
//     lastDate)` -- not `process->time()`.
//
// Feller condition
// ----------------
// Every parameter set below records `feller = 2*kappa*theta - sigma^2` in its
// inputs. Two of the four sets violate it (negative value), because each engine
// takes a different route through the near-zero-variance regime: the COS
// cumulants stay analytic, the Gatheral complex log needs its branch tracking,
// and the LPP/Forde expansions lose accuracy. A port validated only on
// Feller-satisfying parameters passes for the wrong reason.
//
// Evaluation date
// ---------------
// Settings::instance().evaluationDate() = Date(1, March, 2025) -- pinned once,
// below, and the Python test MUST pin the same date. 1-Mar-2025 -> 1-Mar-2026 is
// exactly 365 days, so with Actual365Fixed the 1y exercise time is exactly 1.0
// and no day-count rounding noise leaks into the reference values.
//
// Emits JSON on stdout; nothing else may be printed.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/models/equity/batesmodel.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/models/equity/piecewisetimedependenthestonmodel.hpp>
#include <ql/models/parameter.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/pricingengines/vanilla/analyticptdhestonengine.hpp>
#include <ql/pricingengines/vanilla/batesengine.hpp>
#include <ql/pricingengines/vanilla/coshestonengine.hpp>
#include <ql/pricingengines/vanilla/exponentialfittinghestonengine.hpp>
#include <ql/pricingengines/vanilla/hestonexpansionengine.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>

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
    Obj& a(const std::string& k, const std::vector<Real>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j != 0U ? ", " : "") + num(v[j]);
        return put(k, body + "]");
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

// ---------------------------------------------------------------------------
// Market fixtures.
// ---------------------------------------------------------------------------
const Date kToday(1, March, 2025);

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kToday, r, dayCounter()));
}

Handle<Quote> quote(Real v) {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(v));
}

// Two markets, because a single market with r == q (or q == 0) hides a wrong
// forward: every engine here divides by `dividendYield()->discount(T)` at least
// once, and with q == 0 that factor is 1 and the mistake is invisible.
struct Market {
    std::string name;
    Real s0;
    Rate r;
    Rate q;
};

const Market kMarketA{"A", 100.0, 0.05, 0.02};
const Market kMarketB{"B", 100.0, 0.35, 0.17};

void describe(Obj& in, const Market& m) {
    in.s("market", m.name);
    in.n("s0", m.s0);
    in.n("r", m.r);
    in.n("q", m.q);
    in.s("day_counter", "Actual365Fixed");
    in.s("evaluation_date", "2025-03-01");
}

// Heston parameter sets. `feller` = 2*kappa*theta - sigma^2; negative means the
// Feller condition is violated and the variance process can touch zero.
struct HParams {
    std::string name;
    Real v0, kappa, theta, sigma, rho;
};

// Alan Lewis reference parameters (test-suite/hestonmodel.cpp:1428-1432).
// Feller: 2*4*0.25 - 1 = +1.0  -> SATISFIED.
const HParams kLewis{"lewis", 0.04, 4.00, 0.25, 1.00, -0.50};
// Forde reference-vol parameters (test-suite/hestonmodel.cpp:1503-1508).
// Feller: 2*1.15*0.04 - 0.04 = +0.052 -> SATISFIED. Low vol-of-vol.
const HParams kForde{"forde", 0.04, 1.15, 0.04, 0.20, -0.40};
// COS-engine / chF testbed (test-suite/hestonmodel.cpp:1999-2003).
// Feller: 2*2*0.15 - 0.64 = -0.04 -> VIOLATED (barely). Strongly negative rho.
const HParams kFellerBad{"fellerbad", 0.10, 2.00, 0.15, 0.80, -0.85};
// Andersen-Piterbarg testbed (test-suite/hestonmodel.cpp:2065-2069).
// Feller: 2*1*0.1 - 0.5625 = -0.3625 -> VIOLATED. POSITIVE rho.
const HParams kPosCorr{"poscorr", 0.10, 1.00, 0.10, 0.75, 0.80};

void describe(Obj& in, const HParams& p) {
    in.s("params", p.name);
    in.n("v0", p.v0);
    in.n("kappa", p.kappa);
    in.n("theta", p.theta);
    in.n("sigma", p.sigma);
    in.n("rho", p.rho);
    in.n("feller", 2 * p.kappa * p.theta - p.sigma * p.sigma);
}

ext::shared_ptr<HestonProcess> hestonProcess(const Market& m, const HParams& p) {
    return ext::make_shared<HestonProcess>(flatCurve(m.r), flatCurve(m.q), quote(m.s0),
                                           p.v0, p.kappa, p.theta, p.sigma, p.rho);
}

ext::shared_ptr<HestonModel> hestonModel(const Market& m, const HParams& p) {
    return ext::make_shared<HestonModel>(hestonProcess(m, p));
}

// Maturities, expressed as day offsets so the Python test reconstructs the exact
// same Date rather than a year fraction.
struct Maturity {
    std::string name;
    int days;
};

const Maturity kT1w{"1w", 7};      // 0.019178082191780823 -- Forde's small-time regime
const Maturity kT3m{"3m", 91};     // 0.24931506849315068
const Maturity kT1y{"1y", 365};    // exactly 1.0
const Maturity kT5y{"5y", 1826};   // 5.002739726027397 (includes one leap day)

Date maturityDate(const Maturity& mat) { return kToday + mat.days * Days; }

void describe(Obj& in, const Maturity& mat) {
    in.s("maturity", mat.name);
    in.i("maturity_days", mat.days);
    in.n("t", dayCounter().yearFraction(kToday, maturityDate(mat)));
}

const char* typeName(Option::Type t) { return t == Option::Call ? "Call" : "Put"; }

ext::shared_ptr<EuropeanExercise> exercise(const Maturity& mat) {
    return ext::make_shared<EuropeanExercise>(maturityDate(mat));
}

ext::shared_ptr<PlainVanillaPayoff> payoff(Option::Type t, Real k) {
    return ext::make_shared<PlainVanillaPayoff>(t, k);
}

// Every engine here fills `results_.value` and nothing else. `hasGreeks` probes
// that: a port that fills greeks would flip these to true.
bool hasAnyGreek(VanillaOption& option) {
    auto probe = [&](Real (VanillaOption::*g)() const) {
        try {
            (void)(option.*g)();
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    return probe(&VanillaOption::delta) || probe(&VanillaOption::gamma) ||
           probe(&VanillaOption::theta) || probe(&VanillaOption::vega) ||
           probe(&VanillaOption::rho) || probe(&VanillaOption::dividendRho);
}

}  // namespace

namespace {

// ===========================================================================
// COSHestonEngine
// ===========================================================================
void emitCosCases() {
    // --- cumulant / moment surface -----------------------------------------
    // c1..c4 and mu/var/skew/kurtosis depend ONLY on (v0, kappa, theta, sigma,
    // rho, t): no market, no strike. mu(t) == c1(t) and var(t) == c2(t)
    // identically; skew = c3/c2^1.5, kurtosis = c4/c2^2. The engine still needs
    // *a* model to exist, so market A is used but is irrelevant to the answer --
    // which is itself worth pinning: the same cumulants are emitted for both
    // markets in the `cos_cumulants_marketB_*` case below.
    const HParams sets[] = {kLewis, kForde, kFellerBad, kPosCorr};
    const Real ts[] = {0.02, 0.5, 1.0, 5.0};

    for (const auto& p : sets) {
        const COSHestonEngine engine(hestonModel(kMarketA, p));
        for (const Real t : ts) {
            Obj in;
            describe(in, p);
            in.n("t", t);
            in.i("L", 16);
            in.i("N", 200);

            Obj ex;
            ex.n("c1", engine.c1(t));
            ex.n("c2", engine.c2(t));
            ex.n("c3", engine.c3(t));
            ex.n("c4", engine.c4(t));
            ex.n("mu", engine.mu(t));
            ex.n("var", engine.var(t));
            ex.n("skew", engine.skew(t));
            ex.n("kurtosis", engine.kurtosis(t));
            addCase("cos_cumulants_" + p.name + "_t" + num(t), in, ex);
        }
    }
    {
        // Same parameters, different market: proves the cumulants ignore rates
        // and spot entirely (muT() is private and never called in v1.43).
        const COSHestonEngine engine(hestonModel(kMarketB, kFellerBad));
        Obj in;
        describe(in, kMarketB);
        describe(in, kFellerBad);
        in.n("t", 1.0);
        Obj ex;
        ex.n("c1", engine.c1(1.0));
        ex.n("c2", engine.c2(1.0));
        ex.n("c3", engine.c3(1.0));
        ex.n("c4", engine.c4(1.0));
        addCase("cos_cumulants_marketB_fellerbad_t1", in, ex);
    }

    // --- normalized characteristic function --------------------------------
    // chF(u, t) takes a REAL u (unlike AnalyticHestonEngine::chF, which takes a
    // complex z) and is the driftless Heston chF. Pinned at the u/t grid the C++
    // test-suite itself uses (hestonmodel.cpp:2012-2013).
    {
        const Real us[] = {0.45, 1.0, 3.0, 4.0};
        const Real ts2[] = {0.01, 3.2, 23.2};
        for (const auto& p : {kLewis, kFellerBad}) {
            const COSHestonEngine engine(hestonModel(kMarketA, p));
            for (const Real u : us) {
                for (const Real t : ts2) {
                    Obj in;
                    describe(in, p);
                    in.n("u", u);
                    in.n("t", t);
                    const std::complex<Real> c = engine.chF(u, t);
                    Obj ex;
                    ex.n("chf_real", c.real());
                    ex.n("chf_imag", c.imag());
                    addCase("cos_chf_" + p.name + "_u" + num(u) + "_t" + num(t), in, ex);
                }
            }
        }
    }

    // --- NPV across the parameter ladder, both knobs varied ----------------
    // (L, N) pairs: the default, the two used by the C++ test-suite
    // (12/75 and 20/400, hestonmodel.cpp:344 and :1327), a high-resolution one,
    // and a deliberately tiny N that a port hardcoding N=200 cannot reproduce.
    struct Knobs {
        Real L;
        Size N;
    };
    const Knobs knobs[] = {{16, 200}, {12, 75}, {20, 400}, {25, 600}, {16, 25}};
    const Real strikes[] = {60.0, 80.0, 100.0, 120.0, 160.0};
    const Option::Type types[] = {Option::Call, Option::Put};

    for (const auto& m : {kMarketA, kMarketB}) {
        for (const auto& p : {kLewis, kFellerBad, kPosCorr}) {
            const auto model = hestonModel(m, p);
            for (const auto& mat : {kT1w, kT3m, kT1y, kT5y}) {
                for (const auto& kn : knobs) {
                    // Restrict the knob sweep to the 1y maturity + market A so the
                    // case count stays readable; every other maturity/market uses
                    // the default knobs.
                    if (!(kn.L == 16 && kn.N == 200) &&
                        !(mat.name == "1y" && m.name == "A"))
                        continue;
                    for (const Real k : strikes) {
                        for (const Option::Type type : types) {
                            VanillaOption option(payoff(type, k), exercise(mat));
                            option.setPricingEngine(
                                ext::make_shared<COSHestonEngine>(model, kn.L, kn.N));

                            Obj in;
                            describe(in, m);
                            describe(in, p);
                            describe(in, mat);
                            in.s("type", typeName(type));
                            in.n("strike", k);
                            in.n("L", kn.L);
                            in.i("N", static_cast<long long>(kn.N));

                            Obj ex;
                            ex.n("npv", option.NPV());
                            addCase("cos_npv_" + m.name + "_" + p.name + "_" + mat.name +
                                        "_" + typeName(type) + "_k" + num(k) + "_L" +
                                        num(kn.L) + "_N" + std::to_string(kn.N),
                                    in, ex);
                        }
                    }
                }
            }
        }
    }

    // --- the truncation-bound early return ---------------------------------
    // calculate() computes a = x + c1 - L*w, b = x + c1 + L*w with
    // w = sqrt(|c2|), x = log(fwd/K); when `x >= b/2 || x <= a/2` it abandons the
    // cosine series and returns the no-arbitrage BOUND, max(spot*qf - K*df, 0)
    // for a call and max(K*df - spot*qf, 0) for a put -- i.e. the discounted
    // intrinsic, NOT a price. A very short maturity plus a far strike makes
    // L*w tiny compared with |x| and fires the branch. The `_bound` cases below
    // are pinned together with the raw bound so the Python test can assert the
    // engine took the branch rather than merely landed near intrinsic.
    {
        const auto model = hestonModel(kMarketA, kForde);
        const Real farStrikes[] = {20.0, 400.0};
        for (const Real k : farStrikes) {
            for (const Option::Type type : types) {
                VanillaOption option(payoff(type, k), exercise(kT1w));
                option.setPricingEngine(ext::make_shared<COSHestonEngine>(model, 16.0, 200));

                const Date md = maturityDate(kT1w);
                const Real df = flatCurve(kMarketA.r)->discount(md);
                const Real qf = flatCurve(kMarketA.q)->discount(md);
                const Real bound = (type == Option::Call)
                                       ? std::max(kMarketA.s0 * qf - k * df, 0.0)
                                       : std::max(k * df - kMarketA.s0 * qf, 0.0);

                Obj in;
                describe(in, kMarketA);
                describe(in, kForde);
                describe(in, kT1w);
                in.s("type", typeName(type));
                in.n("strike", k);
                in.n("L", 16.0);
                in.i("N", 200);

                Obj ex;
                ex.n("npv", option.NPV());
                ex.n("no_arbitrage_bound", bound);
                ex.b("hit_truncation_branch", true);
                addCase("cos_bound_" + std::string(typeName(type)) + "_k" + num(k), in, ex);
            }
        }
    }

    // --- guards -------------------------------------------------------------
    {
        VanillaOption option(payoff(Option::Call, 100.0), exercise(kT1y));
        option.setPricingEngine(
            ext::make_shared<COSHestonEngine>(hestonModel(kMarketA, kLewis)));
        (void)option.NPV();

        Obj in;
        describe(in, kMarketA);
        describe(in, kLewis);
        describe(in, kT1y);
        in.s("type", "Call");
        in.n("strike", 100.0);
        Obj ex;
        ex.b("has_any_greek", hasAnyGreek(option));
        addCase("cos_no_greeks", in, ex);
    }
    {
        // Not a plain-vanilla payoff -> QL_REQUIRE(payoff, "non plain vanilla
        // payoff given"). A CashOrNothingPayoff is a StrikedTypePayoff, so it
        // constructs fine and only the engine rejects it.
        VanillaOption option(
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0), exercise(kT1y));
        option.setPricingEngine(
            ext::make_shared<COSHestonEngine>(hestonModel(kMarketA, kLewis)));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("payoff", "CashOrNothingPayoff");
        Obj ex;
        ex.b("throws", threw);
        addCase("cos_rejects_non_plain_vanilla_payoff", in, ex);
    }
    {
        // American exercise -> QL_REQUIRE(exercise->type() == European).
        VanillaOption option(
            payoff(Option::Call, 100.0),
            ext::make_shared<AmericanExercise>(kToday, maturityDate(kT1y)));
        option.setPricingEngine(
            ext::make_shared<COSHestonEngine>(hestonModel(kMarketA, kLewis)));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("exercise", "AmericanExercise");
        Obj ex;
        ex.b("throws", threw);
        addCase("cos_rejects_non_european_exercise", in, ex);
    }
}

}  // namespace

namespace {

// ===========================================================================
// HestonExpansion / LPP2 / LPP3 / Forde / HestonExpansionEngine
// ===========================================================================
const char* expansionName(HestonExpansionEngine::HestonExpansionFormula f) {
    switch (f) {
        case HestonExpansionEngine::LPP2:
            return "LPP2";
        case HestonExpansionEngine::LPP3:
            return "LPP3";
        case HestonExpansionEngine::Forde:
            return "Forde";
    }
    return "?";
}

void emitExpansionCases() {
    // --- HestonExpansion::impliedVolatility, called DIRECTLY ----------------
    // This is the abstract base's only method and the way the expansions are
    // used during calibration: construct once per expiry, call per strike.
    // `forward` is an explicit argument, so no market is involved at all --
    // these cases carry only (kappa, theta, sigma, v0, rho, term).
    //
    // terms include 1/52 (Forde's small-time asymptotic regime, where it is the
    // most accurate of the three) and 10.0 (where the C++ test-suite's own
    // comment says "forde breaks down for long maturities" and allows a 100%
    // tolerance). Both are pinned exactly: a port must reproduce the breakdown,
    // not repair it.
    const Real forward = 100.0;
    const Real terms[] = {1.0 / 52.0, 0.1, 1.0, 5.0, 10.0};
    const Real strikes[] = {60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0};

    for (const auto& p : {kForde, kLewis, kFellerBad}) {
        for (const Real term : terms) {
            const LPP2HestonExpansion lpp2(p.kappa, p.theta, p.sigma, p.v0, p.rho, term);
            const LPP3HestonExpansion lpp3(p.kappa, p.theta, p.sigma, p.v0, p.rho, term);
            const FordeHestonExpansion forde(p.kappa, p.theta, p.sigma, p.v0, p.rho, term);

            std::vector<Real> v2, v3, vf;
            for (const Real k : strikes) {
                v2.push_back(lpp2.impliedVolatility(k, forward));
                v3.push_back(lpp3.impliedVolatility(k, forward));
                vf.push_back(forde.impliedVolatility(k, forward));
            }

            Obj in;
            describe(in, p);
            in.n("term", term);
            in.n("forward", forward);
            in.a("strikes", std::vector<Real>(std::begin(strikes), std::end(strikes)));

            Obj ex;
            ex.a("lpp2_vols", v2);
            ex.a("lpp3_vols", v3);
            ex.a("forde_vols", vf);
            addCase("expansion_iv_" + p.name + "_term" + num(term), in, ex);
        }
    }

    // --- the max(1e-8, ...) clamp ------------------------------------------
    // LPP2/LPP3::impliedVolatility return max(1e-8, vol) where vol is a
    // polynomial in x = log(K/F); Forde returns sqrt(max(1e-8, var)) so its
    // floor is 1e-4, not 1e-8. Far wings at a long term with a large vol-of-vol
    // drive the polynomials negative and pin the clamp. NOTE the asymmetry: a
    // port that clamps Forde at 1e-8 *after* the sqrt gets 1e-8 where C++ gets
    // 1e-4.
    {
        const Real wings[] = {1.0, 5.0, 500.0, 5000.0};
        for (const auto& p : {kLewis, kFellerBad}) {
            for (const Real term : {5.0, 10.0}) {
                const LPP2HestonExpansion lpp2(p.kappa, p.theta, p.sigma, p.v0, p.rho, term);
                const LPP3HestonExpansion lpp3(p.kappa, p.theta, p.sigma, p.v0, p.rho, term);
                const FordeHestonExpansion forde(p.kappa, p.theta, p.sigma, p.v0, p.rho, term);
                std::vector<Real> v2, v3, vf;
                for (const Real k : wings) {
                    v2.push_back(lpp2.impliedVolatility(k, forward));
                    v3.push_back(lpp3.impliedVolatility(k, forward));
                    vf.push_back(forde.impliedVolatility(k, forward));
                }
                Obj in;
                describe(in, p);
                in.n("term", term);
                in.n("forward", forward);
                in.a("strikes", std::vector<Real>(std::begin(wings), std::end(wings)));
                Obj ex;
                ex.a("lpp2_vols", v2);
                ex.a("lpp3_vols", v3);
                ex.a("forde_vols", vf);
                addCase("expansion_iv_wing_" + p.name + "_term" + num(term), in, ex);
            }
        }
    }

    // --- HestonExpansionEngine NPV, all three formulas ----------------------
    {
        const HestonExpansionEngine::HestonExpansionFormula formulas[] = {
            HestonExpansionEngine::LPP2, HestonExpansionEngine::LPP3,
            HestonExpansionEngine::Forde};
        const Real npvStrikes[] = {60.0, 80.0, 100.0, 120.0, 160.0};
        const Option::Type types[] = {Option::Call, Option::Put};

        for (const auto& m : {kMarketA, kMarketB}) {
            for (const auto& p : {kLewis, kForde, kFellerBad}) {
                const auto model = hestonModel(m, p);
                for (const auto& mat : {kT1w, kT3m, kT1y, kT5y}) {
                    if (m.name == "B" && mat.name != "1y")
                        continue;
                    for (const auto f : formulas) {
                        for (const Real k : npvStrikes) {
                            for (const Option::Type type : types) {
                                VanillaOption option(payoff(type, k), exercise(mat));
                                option.setPricingEngine(
                                    ext::make_shared<HestonExpansionEngine>(model, f));

                                Obj in;
                                describe(in, m);
                                describe(in, p);
                                describe(in, mat);
                                in.s("formula", expansionName(f));
                                in.s("type", typeName(type));
                                in.n("strike", k);

                                Obj ex;
                                ex.n("npv", option.NPV());
                                addCase("expansion_npv_" + m.name + "_" + p.name + "_" +
                                            mat.name + "_" + expansionName(f) + "_" +
                                            typeName(type) + "_k" + num(k),
                                        in, ex);
                            }
                        }
                    }
                }
            }
        }
    }

    // --- guards -------------------------------------------------------------
    {
        VanillaOption option(payoff(Option::Call, 100.0), exercise(kT1y));
        option.setPricingEngine(ext::make_shared<HestonExpansionEngine>(
            hestonModel(kMarketA, kLewis), HestonExpansionEngine::LPP2));
        (void)option.NPV();
        Obj in;
        describe(in, kMarketA);
        describe(in, kLewis);
        describe(in, kT1y);
        in.s("formula", "LPP2");
        Obj ex;
        ex.b("has_any_greek", hasAnyGreek(option));
        addCase("expansion_no_greeks", in, ex);
    }
    {
        VanillaOption option(
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0), exercise(kT1y));
        option.setPricingEngine(ext::make_shared<HestonExpansionEngine>(
            hestonModel(kMarketA, kLewis), HestonExpansionEngine::LPP3));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("payoff", "CashOrNothingPayoff");
        Obj ex;
        ex.b("throws", threw);
        addCase("expansion_rejects_non_plain_vanilla_payoff", in, ex);
    }
    {
        VanillaOption option(
            payoff(Option::Call, 100.0),
            ext::make_shared<AmericanExercise>(kToday, maturityDate(kT1y)));
        option.setPricingEngine(ext::make_shared<HestonExpansionEngine>(
            hestonModel(kMarketA, kLewis), HestonExpansionEngine::Forde));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("exercise", "AmericanExercise");
        Obj ex;
        ex.b("throws", threw);
        addCase("expansion_rejects_non_european_exercise", in, ex);
    }
}

}  // namespace

namespace {

// ===========================================================================
// BatesDetJumpEngine / BatesDoubleExpEngine / BatesDoubleExpDetJumpEngine
// ===========================================================================
//
// All three derive (directly or via BatesEngine) from AnalyticHestonEngine and
// are constructed with `AnalyticHestonEngine::Gatheral` plus
// `Integration::gaussLaguerre(integrationOrder)` -- NOT the OptimalCV /
// control-variate default that AnalyticHestonEngine(model, order) itself uses.
// A port that routes these through an AnalyticHestonEngine whose default cpxLog
// is OptimalCV will be close but not equal.
//
// Jump parameters are deliberately away from every C++ default:
//   lambda = 0.7 (default 0.1), nu = -0.15, delta = 0.25,
//   nuUp = 0.22, nuDown = 0.13, p = 0.35 (default 0.5),
//   kappaLambda = 2.4 (default 1.0), thetaLambda = 0.32 (default 0.1).
// A port that mis-orders the (lambda, nuUp, nuDown, p) constructor arguments --
// easy, because addOnTerm reads them back as (p, nuDown, nuUp, lambda) --
// produces a different price for every case below.

struct JumpParams {
    Real lambda, nu, delta;          // lognormal jump law (BatesModel)
    Real nuUp, nuDown, p;            // double-exponential jump law
    Real kappaLambda, thetaLambda;   // deterministic OU jump intensity
};

const JumpParams kJumps{0.7, -0.15, 0.25, 0.22, 0.13, 0.35, 2.4, 0.32};

void describe(Obj& in, const JumpParams& j) {
    in.n("lambda", j.lambda);
    in.n("nu", j.nu);
    in.n("delta", j.delta);
    in.n("nu_up", j.nuUp);
    in.n("nu_down", j.nuDown);
    in.n("p", j.p);
    in.n("kappa_lambda", j.kappaLambda);
    in.n("theta_lambda", j.thetaLambda);
}

ext::shared_ptr<BatesProcess> batesProcess(const Market& m, const HParams& p,
                                           const JumpParams& j) {
    return ext::make_shared<BatesProcess>(flatCurve(m.r), flatCurve(m.q), quote(m.s0), p.v0,
                                          p.kappa, p.theta, p.sigma, p.rho, j.lambda, j.nu,
                                          j.delta);
}

void emitBatesCases() {
    const Real strikes[] = {60.0, 80.0, 100.0, 120.0, 160.0};
    const Option::Type types[] = {Option::Call, Option::Put};
    // Order 144 is the C++ default; 64 is what the C++ test-suite uses; 1/2/4/8
    // are far too small for convergence and are the ONLY orders at which an
    // engine that discards `integrationOrder` can be caught.
    const Size orders[] = {1, 2, 4, 8, 64, 144};

    for (const auto& m : {kMarketA, kMarketB}) {
        for (const auto& p : {kLewis, kFellerBad, kPosCorr}) {
            const auto bProcess = batesProcess(m, p, kJumps);
            const auto hProcess = hestonProcess(m, p);

            const auto detJumpModel = ext::make_shared<BatesDetJumpModel>(
                bProcess, kJumps.kappaLambda, kJumps.thetaLambda);
            const auto dblExpModel = ext::make_shared<BatesDoubleExpModel>(
                hProcess, kJumps.lambda, kJumps.nuUp, kJumps.nuDown, kJumps.p);
            const auto dblExpDetJumpModel = ext::make_shared<BatesDoubleExpDetJumpModel>(
                hProcess, kJumps.lambda, kJumps.nuUp, kJumps.nuDown, kJumps.p,
                kJumps.kappaLambda, kJumps.thetaLambda);

            for (const auto& mat : {kT1w, kT3m, kT1y, kT5y}) {
                for (const Size order : orders) {
                    // Sweep the order only on the 1y/market-A slice; everything
                    // else uses the C++ default order of 144.
                    if (order != 144 && !(mat.name == "1y" && m.name == "A"))
                        continue;
                    for (const Real k : strikes) {
                        for (const Option::Type type : types) {
                            Obj in;
                            describe(in, m);
                            describe(in, p);
                            describe(in, kJumps);
                            describe(in, mat);
                            in.s("type", typeName(type));
                            in.n("strike", k);
                            in.i("integration_order", static_cast<long long>(order));

                            const std::string suffix =
                                m.name + "_" + p.name + "_" + mat.name + "_" +
                                typeName(type) + "_k" + num(k) + "_o" +
                                std::to_string(order);

                            {
                                VanillaOption option(payoff(type, k), exercise(mat));
                                option.setPricingEngine(
                                    ext::make_shared<BatesDetJumpEngine>(detJumpModel, order));
                                Obj ex;
                                ex.n("npv", option.NPV());
                                addCase("bates_detjump_" + suffix, in, ex);
                            }
                            {
                                VanillaOption option(payoff(type, k), exercise(mat));
                                option.setPricingEngine(
                                    ext::make_shared<BatesDoubleExpEngine>(dblExpModel, order));
                                Obj ex;
                                ex.n("npv", option.NPV());
                                addCase("bates_dblexp_" + suffix, in, ex);
                            }
                            {
                                VanillaOption option(payoff(type, k), exercise(mat));
                                option.setPricingEngine(
                                    ext::make_shared<BatesDoubleExpDetJumpEngine>(
                                        dblExpDetJumpModel, order));
                                Obj ex;
                                ex.n("npv", option.NPV());
                                addCase("bates_dblexpdetjump_" + suffix, in, ex);
                            }
                        }
                    }
                }
            }
        }
    }

    // --- the (relTolerance, maxEvaluations) constructor ---------------------
    // Second overload of each engine: Gatheral + adaptive
    // Integration::gaussLobatto(relTolerance, Null<Real>(), maxEvaluations),
    // which integrates the transformed integrand on [0, 1] rather than the raw
    // one on [0, inf). Same answer to ~1e-10, different code path.
    {
        const auto bProcess = batesProcess(kMarketA, kFellerBad, kJumps);
        const auto hProcess = hestonProcess(kMarketA, kFellerBad);
        const auto detJumpModel = ext::make_shared<BatesDetJumpModel>(
            bProcess, kJumps.kappaLambda, kJumps.thetaLambda);
        const auto dblExpModel = ext::make_shared<BatesDoubleExpModel>(
            hProcess, kJumps.lambda, kJumps.nuUp, kJumps.nuDown, kJumps.p);
        const auto dblExpDetJumpModel = ext::make_shared<BatesDoubleExpDetJumpModel>(
            hProcess, kJumps.lambda, kJumps.nuUp, kJumps.nuDown, kJumps.p,
            kJumps.kappaLambda, kJumps.thetaLambda);

        const Real relTolerance = 1e-10;
        const Size maxEvaluations = 10000;

        for (const Real k : {80.0, 100.0, 120.0}) {
            for (const Option::Type type : types) {
                Obj in;
                describe(in, kMarketA);
                describe(in, kFellerBad);
                describe(in, kJumps);
                describe(in, kT1y);
                in.s("type", typeName(type));
                in.n("strike", k);
                in.n("rel_tolerance", relTolerance);
                in.i("max_evaluations", static_cast<long long>(maxEvaluations));

                const std::string suffix =
                    std::string(typeName(type)) + "_k" + num(k) + "_lobatto";
                {
                    VanillaOption option(payoff(type, k), exercise(kT1y));
                    option.setPricingEngine(ext::make_shared<BatesDetJumpEngine>(
                        detJumpModel, relTolerance, maxEvaluations));
                    Obj ex;
                    ex.n("npv", option.NPV());
                    addCase("bates_detjump_" + suffix, in, ex);
                }
                {
                    VanillaOption option(payoff(type, k), exercise(kT1y));
                    option.setPricingEngine(ext::make_shared<BatesDoubleExpEngine>(
                        dblExpModel, relTolerance, maxEvaluations));
                    Obj ex;
                    ex.n("npv", option.NPV());
                    addCase("bates_dblexp_" + suffix, in, ex);
                }
                {
                    VanillaOption option(payoff(type, k), exercise(kT1y));
                    option.setPricingEngine(ext::make_shared<BatesDoubleExpDetJumpEngine>(
                        dblExpDetJumpModel, relTolerance, maxEvaluations));
                    Obj ex;
                    ex.n("npv", option.NPV());
                    addCase("bates_dblexpdetjump_" + suffix, in, ex);
                }
            }
        }
    }

    // --- guards -------------------------------------------------------------
    {
        // analytichestonengine.cpp:926 --
        //   QL_REQUIRE(intOrder <= 192, "maximum integraton order (192) exceeded")
        // fires in Integration::gaussLaguerre, i.e. in the ENGINE CONSTRUCTOR,
        // before any pricing happens.
        const auto detJumpModel = ext::make_shared<BatesDetJumpModel>(
            batesProcess(kMarketA, kLewis, kJumps), kJumps.kappaLambda, kJumps.thetaLambda);
        bool threw = false;
        try {
            const auto engine = ext::make_shared<BatesDetJumpEngine>(detJumpModel, 1024);
            (void)engine;
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.i("integration_order", 1024);
        Obj ex;
        ex.b("throws", threw);
        addCase("bates_rejects_integration_order_above_192", in, ex);
    }
    {
        // The largest order that is still accepted.
        const auto detJumpModel = ext::make_shared<BatesDetJumpModel>(
            batesProcess(kMarketA, kLewis, kJumps), kJumps.kappaLambda, kJumps.thetaLambda);
        VanillaOption option(payoff(Option::Call, 100.0), exercise(kT1y));
        option.setPricingEngine(ext::make_shared<BatesDetJumpEngine>(detJumpModel, 192));
        Obj in;
        describe(in, kMarketA);
        describe(in, kLewis);
        describe(in, kJumps);
        describe(in, kT1y);
        in.s("type", "Call");
        in.n("strike", 100.0);
        in.i("integration_order", 192);
        Obj ex;
        ex.n("npv", option.NPV());
        ex.b("has_any_greek", hasAnyGreek(option));
        addCase("bates_detjump_order192", in, ex);
    }
}

}  // namespace

namespace {

// ===========================================================================
// AnalyticPTDHestonEngine
// ===========================================================================
//
// The model is genuinely piecewise: THREE segments, each with its own
// (theta, kappa, sigma, rho). Segment 2 violates the Feller condition
// (2*2.5*0.09 = 0.45 < 0.81 = sigma^2) while segments 1 and 3 satisfy it, and
// rho changes SIGN between segments -- so a port that accidentally uses the
// first segment's parameters everywhere, or that walks the grid forwards
// instead of backwards, gets a different number on every case.
//
// Time grid: mandatory {0.5, 1.0, 2.0} -> times {0, 0.5, 1.0, 2.0}.
// Parameter break times {0.5, 1.0} -> 3 piecewise-constant values each.
// The model's parameter at time t is params[first i with t < breaks[i]], so
// segment k is [grid[k], grid[k+1]) and the engine samples it at the MIDPOINT
// 0.5*(begin+end) of each interval it visits -- never at an endpoint.

struct PtdSegments {
    std::vector<Real> theta, kappa, sigma, rho;
    Real v0;
};

const PtdSegments kPtd{{0.06, 0.09, 0.14},
                       {1.50, 2.50, 0.80},
                       {0.40, 0.90, 0.25},
                       {-0.75, 0.35, -0.20},
                       0.07};

const std::vector<Real>& ptdBreaks() {
    static const std::vector<Real> b{0.5, 1.0};
    return b;
}

const std::vector<Real>& ptdGridTimes() {
    static const std::vector<Real> g{0.5, 1.0, 2.0};
    return g;
}

Parameter piecewise(const std::vector<Real>& values, const Constraint& c) {
    PiecewiseConstantParameter p(ptdBreaks(), c);
    for (Size i = 0; i < values.size(); ++i)
        p.setParam(i, values[i]);
    return p;
}

ext::shared_ptr<PiecewiseTimeDependentHestonModel> ptdModel(const Market& m,
                                                            const PtdSegments& s) {
    const TimeGrid grid(ptdGridTimes().begin(), ptdGridTimes().end());
    return ext::make_shared<PiecewiseTimeDependentHestonModel>(
        flatCurve(m.r), flatCurve(m.q), quote(m.s0), s.v0,
        piecewise(s.theta, PositiveConstraint()), piecewise(s.kappa, PositiveConstraint()),
        piecewise(s.sigma, PositiveConstraint()),
        piecewise(s.rho, BoundaryConstraint(-1.0, 1.0)), grid);
}

void describe(Obj& in, const PtdSegments& s) {
    in.a("ptd_theta", s.theta);
    in.a("ptd_kappa", s.kappa);
    in.a("ptd_sigma", s.sigma);
    in.a("ptd_rho", s.rho);
    in.n("v0", s.v0);
    in.a("ptd_breaks", ptdBreaks());
    in.a("ptd_grid_times", ptdGridTimes());
}

const char* ptdCpxLogName(AnalyticPTDHestonEngine::ComplexLogFormula f) {
    return f == AnalyticPTDHestonEngine::Gatheral ? "Gatheral" : "AndersenPiterbarg";
}

void emitPtdCases() {
    // Maturities INSIDE the time grid. 91d spans one segment, 274d spans two,
    // 730d (= exactly 2.0y with Actual365Fixed) spans all three and lands
    // exactly on the grid's last point (the close_enough branch of the
    // "maturity is too large" guard).
    const Maturity mats[] = {{"91d", 91}, {"274d", 274}, {"730d", 730}};
    const Real strikes[] = {60.0, 80.0, 100.0, 120.0, 160.0};
    const Option::Type types[] = {Option::Call, Option::Put};

    for (const auto& m : {kMarketA, kMarketB}) {
        const auto model = ptdModel(m, kPtd);
        for (const auto& mat : mats) {
            for (const Real k : strikes) {
                for (const Option::Type type : types) {
                    Obj in;
                    describe(in, m);
                    describe(in, kPtd);
                    describe(in, mat);
                    in.s("type", typeName(type));
                    in.n("strike", k);

                    const std::string suffix = m.name + "_" + mat.name + "_" +
                                               typeName(type) + "_k" + num(k);

                    // Gatheral, Gauss-Laguerre order 144 (the `integrationOrder`
                    // constructor's default).
                    {
                        VanillaOption option(payoff(type, k), exercise(mat));
                        const auto engine =
                            ext::make_shared<AnalyticPTDHestonEngine>(model, 144);
                        option.setPricingEngine(engine);
                        Obj ex;
                        ex.n("npv", option.NPV());
                        ex.i("number_of_evaluations",
                             static_cast<long long>(engine->numberOfEvaluations()));
                        addCase("ptd_gatheral_laguerre144_" + suffix, in, ex);
                    }
                    // AndersenPiterbarg control variate, SAME market, SAME
                    // strike, SAME maturity, SAME quadrature. If a port accepts
                    // `cpxLog` and discards it these two cases are identical --
                    // that is exactly the defect this pair exists to catch.
                    {
                        VanillaOption option(payoff(type, k), exercise(mat));
                        const auto engine = ext::make_shared<AnalyticPTDHestonEngine>(
                            model, AnalyticPTDHestonEngine::AndersenPiterbarg,
                            AnalyticHestonEngine::Integration::gaussLaguerre(144), 1e-8);
                        option.setPricingEngine(engine);
                        Obj ex;
                        ex.n("npv", option.NPV());
                        ex.i("number_of_evaluations",
                             static_cast<long long>(engine->numberOfEvaluations()));
                        addCase("ptd_andersenpiterbarg_laguerre144_" + suffix, in, ex);
                    }
                }
            }
        }
    }

    // --- integration order actually honoured ---------------------------------
    {
        const auto model = ptdModel(kMarketA, kPtd);
        const Size orders[] = {1, 2, 4, 8, 32, 64, 192};
        for (const Size order : orders) {
            for (const auto f : {AnalyticPTDHestonEngine::Gatheral,
                                 AnalyticPTDHestonEngine::AndersenPiterbarg}) {
                VanillaOption option(payoff(Option::Call, 100.0), exercise(mats[2]));
                const auto engine = ext::make_shared<AnalyticPTDHestonEngine>(
                    model, f, AnalyticHestonEngine::Integration::gaussLaguerre(order), 1e-8);
                option.setPricingEngine(engine);

                Obj in;
                describe(in, kMarketA);
                describe(in, kPtd);
                describe(in, mats[2]);
                in.s("type", "Call");
                in.n("strike", 100.0);
                in.s("cpx_log", ptdCpxLogName(f));
                in.s("integration", "gaussLaguerre");
                in.i("integration_order", static_cast<long long>(order));
                in.n("andersen_piterbarg_epsilon", 1e-8);

                Obj ex;
                ex.n("npv", option.NPV());
                ex.i("number_of_evaluations",
                     static_cast<long long>(engine->numberOfEvaluations()));
                addCase("ptd_order_" + std::string(ptdCpxLogName(f)) + "_o" +
                            std::to_string(order),
                        in, ex);
            }
        }
    }

    // --- the adaptive (relTolerance, maxEvaluations) constructor -------------
    // Gatheral + Integration::gaussLobatto(relTolerance, Null<Real>(),
    // maxEvaluations). The Gatheral branch passes NO maxBound, so Gauss-Lobatto
    // integrates the TRANSFORMED integrand on [0, 1] (integrand2), not the raw
    // one on [0, inf). numberOfEvaluations() is therefore data-dependent and is
    // pinned so a port cannot silently substitute a fixed-order rule.
    {
        const auto model = ptdModel(kMarketA, kPtd);
        for (const Real k : {80.0, 100.0, 120.0}) {
            for (const Option::Type type : types) {
                VanillaOption option(payoff(type, k), exercise(mats[2]));
                const auto engine =
                    ext::make_shared<AnalyticPTDHestonEngine>(model, 1e-10, 10000);
                option.setPricingEngine(engine);

                Obj in;
                describe(in, kMarketA);
                describe(in, kPtd);
                describe(in, mats[2]);
                in.s("type", typeName(type));
                in.n("strike", k);
                in.s("cpx_log", "Gatheral");
                in.s("integration", "gaussLobatto");
                in.n("rel_tolerance", 1e-10);
                in.i("max_evaluations", 10000);

                Obj ex;
                ex.n("npv", option.NPV());
                addCase("ptd_lobatto_" + std::string(typeName(type)) + "_k" + num(k), in, ex);
            }
        }
    }
    // AndersenPiterbarg + Gauss-Lobatto: this is the ONLY configuration in which
    // Integration::andersenPiterbargIntegrationLimit() is actually consulted
    // (Gauss-Laguerre ignores `maxBound` entirely), so it exercises the Brent
    // root-finds inside it.
    {
        const auto model = ptdModel(kMarketA, kPtd);
        for (const Real k : {80.0, 100.0, 120.0}) {
            for (const Option::Type type : types) {
                VanillaOption option(payoff(type, k), exercise(mats[2]));
                const auto engine = ext::make_shared<AnalyticPTDHestonEngine>(
                    model, AnalyticPTDHestonEngine::AndersenPiterbarg,
                    AnalyticHestonEngine::Integration::gaussLobatto(1e-10, Null<Real>(), 10000),
                    1e-8);
                option.setPricingEngine(engine);

                Obj in;
                describe(in, kMarketA);
                describe(in, kPtd);
                describe(in, mats[2]);
                in.s("type", typeName(type));
                in.n("strike", k);
                in.s("cpx_log", "AndersenPiterbarg");
                in.s("integration", "gaussLobatto");
                in.n("rel_tolerance", 1e-10);
                in.i("max_evaluations", 10000);
                in.n("andersen_piterbarg_epsilon", 1e-8);

                Obj ex;
                ex.n("npv", option.NPV());
                addCase("ptd_ap_lobatto_" + std::string(typeName(type)) + "_k" + num(k), in,
                        ex);
            }
        }
    }

    // --- chF / lnChF ---------------------------------------------------------
    // Public members. lnChF walks the SAME backwards recursion the pricer uses,
    // so pinning it isolates a Riccati-recursion bug from an integration bug.
    {
        const auto model = ptdModel(kMarketA, kPtd);
        const auto engine = ext::make_shared<AnalyticPTDHestonEngine>(model, 144);
        const Real us[] = {0.0, 0.25, 1.0, 4.0};
        const Real vs[] = {0.0, -0.5, 0.75};
        const Real ts[] = {0.25, 0.75, 1.5, 2.0};
        for (const Real u : us) {
            for (const Real v : vs) {
                for (const Real t : ts) {
                    const std::complex<Real> z(u, v);
                    const std::complex<Real> ln = engine->lnChF(z, t);
                    const std::complex<Real> ch = engine->chF(z, t);

                    Obj in;
                    describe(in, kMarketA);
                    describe(in, kPtd);
                    in.n("z_real", u);
                    in.n("z_imag", v);
                    in.n("t", t);
                    Obj ex;
                    ex.n("lnchf_real", ln.real());
                    ex.n("lnchf_imag", ln.imag());
                    ex.n("chf_real", ch.real());
                    ex.n("chf_imag", ch.imag());
                    addCase("ptd_chf_u" + num(u) + "_v" + num(v) + "_t" + num(t), in, ex);
                }
            }
        }
    }

    // --- degenerate: all three segments identical ---------------------------
    // The ONE case where the PTD engine must reproduce plain Heston. Pinned
    // together with the AnalyticHestonEngine (Gatheral, Gauss-Laguerre 144)
    // value it has to match, so the Python test can assert the identity rather
    // than merely the number.
    {
        const PtdSegments flat{{kFellerBad.theta, kFellerBad.theta, kFellerBad.theta},
                               {kFellerBad.kappa, kFellerBad.kappa, kFellerBad.kappa},
                               {kFellerBad.sigma, kFellerBad.sigma, kFellerBad.sigma},
                               {kFellerBad.rho, kFellerBad.rho, kFellerBad.rho},
                               kFellerBad.v0};
        const auto model = ptdModel(kMarketA, flat);
        const auto plain = hestonModel(kMarketA, kFellerBad);
        for (const Real k : {80.0, 100.0, 120.0}) {
            for (const Option::Type type : types) {
                VanillaOption ptdOption(payoff(type, k), exercise(mats[2]));
                ptdOption.setPricingEngine(
                    ext::make_shared<AnalyticPTDHestonEngine>(model, 144));
                VanillaOption plainOption(payoff(type, k), exercise(mats[2]));
                plainOption.setPricingEngine(ext::make_shared<AnalyticHestonEngine>(
                    plain, AnalyticHestonEngine::Gatheral,
                    AnalyticHestonEngine::Integration::gaussLaguerre(144)));

                Obj in;
                describe(in, kMarketA);
                describe(in, flat);
                describe(in, mats[2]);
                in.s("type", typeName(type));
                in.n("strike", k);
                Obj ex;
                ex.n("npv", ptdOption.NPV());
                ex.n("plain_heston_npv", plainOption.NPV());
                addCase("ptd_degenerate_" + std::string(typeName(type)) + "_k" + num(k), in,
                        ex);
            }
        }
    }

    // --- guards -------------------------------------------------------------
    {
        const auto model = ptdModel(kMarketA, kPtd);
        VanillaOption option(payoff(Option::Call, 100.0), exercise(mats[2]));
        option.setPricingEngine(ext::make_shared<AnalyticPTDHestonEngine>(model, 144));
        (void)option.NPV();
        Obj in;
        describe(in, kMarketA);
        describe(in, kPtd);
        Obj ex;
        ex.b("has_any_greek", hasAnyGreek(option));
        addCase("ptd_no_greeks", in, ex);
    }
    {
        // Maturity beyond the time grid's last point (2.0y): the engine's own
        // QL_REQUIRE("maturity ... is too large, time grid is bounded by ...").
        const auto model = ptdModel(kMarketA, kPtd);
        VanillaOption option(payoff(Option::Call, 100.0),
                             ext::make_shared<EuropeanExercise>(kToday + 1096 * Days));
        option.setPricingEngine(ext::make_shared<AnalyticPTDHestonEngine>(model, 144));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.i("maturity_days", 1096);
        in.n("time_grid_back", 2.0);
        Obj ex;
        ex.b("throws", threw);
        addCase("ptd_rejects_maturity_past_time_grid", in, ex);
    }
    {
        const auto model = ptdModel(kMarketA, kPtd);
        VanillaOption option(
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0), exercise(mats[2]));
        option.setPricingEngine(ext::make_shared<AnalyticPTDHestonEngine>(model, 144));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("payoff", "CashOrNothingPayoff");
        Obj ex;
        ex.b("throws", threw);
        addCase("ptd_rejects_non_plain_vanilla_payoff", in, ex);
    }
    {
        const auto model = ptdModel(kMarketA, kPtd);
        VanillaOption option(payoff(Option::Call, 100.0),
                             ext::make_shared<AmericanExercise>(kToday, maturityDate(mats[2])));
        option.setPricingEngine(ext::make_shared<AnalyticPTDHestonEngine>(model, 144));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("exercise", "AmericanExercise");
        Obj ex;
        ex.b("throws", threw);
        addCase("ptd_rejects_non_european_exercise", in, ex);
    }
}

}  // namespace

namespace {

// ===========================================================================
// ExponentialFittingHestonEngine
// ===========================================================================
//
// A 64-node exponentially-fitted Gauss-Laguerre rule (Conte-Ixaru-Paternoster-
// Santomauro) applied to AnalyticHestonEngine::AP_Helper. The node/weight table
// `values4` is a 148 x 129 array selected by MONEYNESS: column 0 of each row is
// the fitted omega, and the engine picks the row whose omega is nearest to
// |scalingFactor * freq|, then rescales u = |omega / freq|. Two branches:
//
//   * |freq| < 0.1  ->  n = 0, u = scalingFactor          (the ATM branch)
//   * otherwise     ->  lower_bound on the moneyness column, then step BACK one
//                       row if the previous omega is closer. Note this is a
//                       nearest-neighbour search built out of lower_bound, and
//                       the tie-break is `>` (strictly greater), so on an exact
//                       tie the HIGHER row wins.
//
// freq = log(spot) - log(rd/dd) - log(strike) = log(fwd/strike), so with market
// A at 1y (fwd = 100*e^0.03) strike 100 gives freq = +0.03 -> ATM branch, and
// strike 120 gives freq = -0.152 -> lookup branch. Both are covered.
//
// scalingFactor, when `scaling` is Null<Real>():
//     analyticCV != AsymptoticChF ? max(0.25, min(1000, 0.25/sqrt(0.5*vAvg*t)))
//                                 : 1.0
// -- i.e. the DEFAULT scaling depends on which control variate was selected,
// including on the runtime OptimalCV resolution. A port that computes the
// scaling before resolving OptimalCV gets a different quadrature abscissa set.

const char* cvName(AnalyticHestonEngine::ComplexLogFormula f) {
    switch (f) {
        case AnalyticHestonEngine::Gatheral:
            return "Gatheral";
        case AnalyticHestonEngine::BranchCorrection:
            return "BranchCorrection";
        case AnalyticHestonEngine::AndersenPiterbarg:
            return "AndersenPiterbarg";
        case AnalyticHestonEngine::AndersenPiterbargOptCV:
            return "AndersenPiterbargOptCV";
        case AnalyticHestonEngine::AsymptoticChF:
            return "AsymptoticChF";
        case AnalyticHestonEngine::AngledContour:
            return "AngledContour";
        case AnalyticHestonEngine::AngledContourNoCV:
            return "AngledContourNoCV";
        case AnalyticHestonEngine::OptimalCV:
            return "OptimalCV";
    }
    return "?";
}

void emitExpFitCases() {
    // --- optimalControlVariate(), pinned as a string ------------------------
    // Static; OptimalCV resolves through it at every calculate(). It returns
    // AsymptoticChF only when ALL THREE of
    //     t > 0.15,
    //     (v0 + t*kappa*theta)/sigma*sqrt(1-rho^2) < 0.15,
    //     ((kappa - 0.5*rho*sigma)*(v0 + t*kappa*theta)
    //          + kappa*theta*log(4*(1-rho^2)))/sigma^2 < 0.1
    // hold, and AngledContour otherwise. The parameter sets below resolve it
    // BOTH ways, and a port that hardcodes either answer fails half of them.
    {
        const Real ts[] = {0.019178082191780823, 0.1, 0.24931506849315068, 1.0,
                           5.002739726027397};
        for (const auto& p : {kLewis, kForde, kFellerBad, kPosCorr}) {
            for (const Real t : ts) {
                Obj in;
                describe(in, p);
                in.n("t", t);
                Obj ex;
                ex.s("optimal_control_variate",
                     cvName(AnalyticHestonEngine::optimalControlVariate(
                         t, p.v0, p.kappa, p.theta, p.sigma, p.rho)));
                addCase("expfit_optimal_cv_" + p.name + "_t" + num(t), in, ex);
            }
        }
    }

    // --- NPV for every legal ControlVariate ---------------------------------
    const AnalyticHestonEngine::ComplexLogFormula cvs[] = {
        AnalyticHestonEngine::AndersenPiterbarg,
        AnalyticHestonEngine::AndersenPiterbargOptCV,
        AnalyticHestonEngine::AsymptoticChF,
        AnalyticHestonEngine::AngledContour,
        AnalyticHestonEngine::AngledContourNoCV,
        AnalyticHestonEngine::OptimalCV};
    const Real strikes[] = {60.0, 80.0, 100.0, 120.0, 160.0};
    const Option::Type types[] = {Option::Call, Option::Put};

    for (const auto& m : {kMarketA, kMarketB}) {
        for (const auto& p : {kLewis, kForde, kFellerBad, kPosCorr}) {
            const auto model = hestonModel(m, p);
            for (const auto& mat : {kT1w, kT3m, kT1y, kT5y}) {
                for (const auto cv : cvs) {
                    // Full CV sweep on the 1y/market-A slice; elsewhere only the
                    // default (OptimalCV) to keep the case count sane.
                    if (cv != AnalyticHestonEngine::OptimalCV &&
                        !(mat.name == "1y" && m.name == "A"))
                        continue;
                    for (const Real k : strikes) {
                        for (const Option::Type type : types) {
                            VanillaOption option(payoff(type, k), exercise(mat));
                            option.setPricingEngine(
                                ext::make_shared<ExponentialFittingHestonEngine>(model, cv));

                            Obj in;
                            describe(in, m);
                            describe(in, p);
                            describe(in, mat);
                            in.s("type", typeName(type));
                            in.n("strike", k);
                            in.s("control_variate", cvName(cv));
                            in.s("scaling", "Null");
                            in.n("alpha", -0.5);

                            Obj ex;
                            ex.n("npv", option.NPV());
                            addCase("expfit_npv_" + m.name + "_" + p.name + "_" + mat.name +
                                        "_" + cvName(cv) + "_" + typeName(type) + "_k" +
                                        num(k),
                                    in, ex);
                        }
                    }
                }
            }
        }
    }

    // --- explicit `scaling` (bypasses the CV-dependent default) -------------
    {
        const auto model = hestonModel(kMarketA, kFellerBad);
        for (const Real scaling : {0.25, 1.0, 4.0}) {
            for (const auto cv : {AnalyticHestonEngine::AndersenPiterbarg,
                                  AnalyticHestonEngine::AsymptoticChF,
                                  AnalyticHestonEngine::OptimalCV}) {
                for (const Real k : {90.0, 100.0, 130.0}) {
                    VanillaOption option(payoff(Option::Call, k), exercise(kT1y));
                    option.setPricingEngine(
                        ext::make_shared<ExponentialFittingHestonEngine>(model, cv, scaling));

                    Obj in;
                    describe(in, kMarketA);
                    describe(in, kFellerBad);
                    describe(in, kT1y);
                    in.s("type", "Call");
                    in.n("strike", k);
                    in.s("control_variate", cvName(cv));
                    in.n("scaling", scaling);
                    in.n("alpha", -0.5);
                    Obj ex;
                    ex.n("npv", option.NPV());
                    addCase("expfit_scaling_" + std::string(cvName(cv)) + "_s" +
                                num(scaling) + "_k" + num(k),
                            in, ex);
                }
            }
        }
    }

    // --- explicit `alpha` ----------------------------------------------------
    // alpha shifts the integration contour: AP_Helper uses z = (u, -alpha) and
    // multiplies by s_alpha = exp(alpha*freq). AsymptoticChF's
    // controlVariateValue() has QL_REQUIRE(alpha_ == -0.5), so a non-default
    // alpha with AsymptoticChF THROWS -- pinned below.
    {
        const auto model = hestonModel(kMarketA, kFellerBad);
        for (const Real alpha : {-0.5, -0.75, -1.5, 0.25}) {
            for (const Real k : {90.0, 100.0, 130.0}) {
                VanillaOption option(payoff(Option::Call, k), exercise(kT1y));
                option.setPricingEngine(ext::make_shared<ExponentialFittingHestonEngine>(
                    model, AnalyticHestonEngine::AndersenPiterbarg, Null<Real>(), alpha));

                Obj in;
                describe(in, kMarketA);
                describe(in, kFellerBad);
                describe(in, kT1y);
                in.s("type", "Call");
                in.n("strike", k);
                in.s("control_variate", "AndersenPiterbarg");
                in.s("scaling", "Null");
                in.n("alpha", alpha);
                Obj ex;
                ex.n("npv", option.NPV());
                addCase("expfit_alpha_a" + num(alpha) + "_k" + num(k), in, ex);
            }
        }
    }
    {
        const auto model = hestonModel(kMarketA, kFellerBad);
        VanillaOption option(payoff(Option::Call, 100.0), exercise(kT1y));
        option.setPricingEngine(ext::make_shared<ExponentialFittingHestonEngine>(
            model, AnalyticHestonEngine::AsymptoticChF, Null<Real>(), -0.75));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("control_variate", "AsymptoticChF");
        in.n("alpha", -0.75);
        Obj ex;
        ex.b("throws", threw);
        addCase("expfit_asymptotic_requires_alpha_minus_half", in, ex);
    }

    // --- guards -------------------------------------------------------------
    {
        // exponentialfittinghestonengine.cpp:241-242 --
        //   QL_REQUIRE(cv_ != Gatheral && cv_ != BranchCorrection, ...)
        // raised in calculate(), NOT in the constructor.
        for (const auto cv : {AnalyticHestonEngine::Gatheral,
                              AnalyticHestonEngine::BranchCorrection}) {
            const auto model = hestonModel(kMarketA, kLewis);
            VanillaOption option(payoff(Option::Call, 100.0), exercise(kT1y));
            option.setPricingEngine(
                ext::make_shared<ExponentialFittingHestonEngine>(model, cv));
            bool threw = false;
            try {
                (void)option.NPV();
            } catch (const std::exception&) {
                threw = true;
            }
            Obj in;
            in.s("control_variate", cvName(cv));
            Obj ex;
            ex.b("throws", threw);
            addCase("expfit_rejects_" + std::string(cvName(cv)), in, ex);
        }
    }
    {
        const auto model = hestonModel(kMarketA, kLewis);
        VanillaOption option(payoff(Option::Call, 100.0), exercise(kT1y));
        option.setPricingEngine(ext::make_shared<ExponentialFittingHestonEngine>(model));
        (void)option.NPV();
        Obj in;
        describe(in, kMarketA);
        describe(in, kLewis);
        describe(in, kT1y);
        Obj ex;
        ex.b("has_any_greek", hasAnyGreek(option));
        addCase("expfit_no_greeks", in, ex);
    }
    {
        const auto model = hestonModel(kMarketA, kLewis);
        VanillaOption option(
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0), exercise(kT1y));
        option.setPricingEngine(ext::make_shared<ExponentialFittingHestonEngine>(model));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("payoff", "CashOrNothingPayoff");
        Obj ex;
        ex.b("throws", threw);
        addCase("expfit_rejects_non_plain_vanilla_payoff", in, ex);
    }
    {
        const auto model = hestonModel(kMarketA, kLewis);
        VanillaOption option(payoff(Option::Call, 100.0),
                             ext::make_shared<AmericanExercise>(kToday, maturityDate(kT1y)));
        option.setPricingEngine(ext::make_shared<ExponentialFittingHestonEngine>(model));
        bool threw = false;
        try {
            (void)option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.s("exercise", "AmericanExercise");
        Obj ex;
        ex.b("throws", threw);
        addCase("expfit_rejects_non_european_exercise", in, ex);
    }
}

}  // namespace

int main() {
    try {
        Settings::instance().evaluationDate() = kToday;

        emitCosCases();
        emitExpansionCases();
        emitBatesCases();
        emitPtdCases();
        emitExpFitCases();

        emitDocument();
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
