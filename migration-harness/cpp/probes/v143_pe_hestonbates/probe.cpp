// migration-harness/cpp/probes/v143_pe_hestonbates/probe.cpp
//
// Reference values for the branches of the Heston/Bates Fourier engines that
// v143_pe_heston/probe.cpp leaves unpinned. It is a COMPLEMENT to that probe,
// not a second copy of it: every case below exercises a code path no case in
// references/v143/pe/heston.json reaches. Classes covered:
//
//   * BatesEngine                     (ql/pricingengines/vanilla/batesengine.{hpp,cpp})
//   * ExponentialFittingHestonEngine  (ql/pricingengines/vanilla/exponentialfittinghestonengine.{hpp,cpp})
//   * AnalyticHestonEngine::optimalControlVariate
//                                     (ql/pricingengines/vanilla/analytichestonengine.{hpp,cpp})
//   * AnalyticPTDHestonEngine         (ql/pricingengines/vanilla/analyticptdhestonengine.{hpp,cpp})
//
// Why the values are EXACT, not banded
// ------------------------------------
// Every engine here is a deterministic quadrature over a closed-form
// characteristic function: fixed Gauss-Laguerre nodes, a tabulated 64-node
// exponentially-fitted rule, or an adaptive Gauss-Lobatto whose subdivision is
// itself a deterministic function of the integrand. No random numbers, no
// clock, no locale. A correct port reproduces every number below to within the
// double-precision rounding of the cancellation each formula performs.
//
// What each section adds over v143_pe_heston, and why
// ---------------------------------------------------
//  1. optimalControlVariate() resolving to AsymptoticChF.
//     v143_pe_heston pins 20 (parameter set, t) selections and -- despite the
//     comment there claiming the sets "resolve it BOTH ways" -- ALL TWENTY
//     return AngledContour, and so does every OptimalCV pricing case in that
//     file. The AsymptoticChF arm of the selector is therefore completely
//     unpinned upstream. The three conditions are
//         (a) t > 0.15
//         (b) (v0 + t*kappa*theta)/sigma*sqrt(1-rho^2) < 0.15
//         (c) ((kappa - 0.5*rho*sigma)*(v0 + t*kappa*theta)
//                  + kappa*theta*log(4*(1-rho^2)))/sigma^2 < 0.1
//     and ALL THREE must hold. `kAsymp` satisfies all three at t >= 1; the
//     cases below sweep t across (a)'s threshold, and `kNearB` / `kNearC` sit
//     just either side of (b) and (c) so a port with a flipped comparison or a
//     dropped term lands on the wrong arm. A port that hardcodes AngledContour
//     passes every upstream case and fails here.
//
//  2. ExponentialFittingHestonEngine's scaling default under a RESOLVED
//     AsymptoticChF. exponentialfittinghestonengine.cpp:293-298 reads
//         scalingFactor = (scaling == Null)
//             ? (analyticCV != AsymptoticChF
//                   ? max(0.25, min(1000, 0.25/sqrt(0.5*vAvg*t)))
//                   : 1.0)
//             : scaling
//     -- the test is on `analyticCV`, the value AFTER OptimalCV has been
//     resolved at runtime, not on the constructor argument. Upstream never
//     resolves OptimalCV to AsymptoticChF, so the 1.0 arm is only ever reached
//     with an explicitly-named control variate and the ordering bug (compute
//     the scaling, then resolve) is invisible there. Here `expfit_optcv_*` and
//     `expfit_explicit_asymp_*` price the SAME market with OptimalCV and with
//     AsymptoticChF spelled out; they must be bit-identical, and a port that
//     computes the scaling before resolving gets a different abscissa set and
//     therefore a different price for the first of the pair.
//
//  3. ExponentialFittingHestonEngine's moneyness-row clamp. Row selection is
//         n = min(moneyness.size()-1, lower_bound(moneyness, |scaling*freq|))
//     followed by a step back when the previous omega is nearer. The table's
//     last omega is 49.484168050663322; upstream's strikes (60..160 against a
//     ~103 forward) give |scaling*freq| below 2, so neither the min() clamp nor
//     the upper half of the 147-row table is ever touched. The `expfit_wing_*`
//     cases drive |freq| up to ~4.6 by strike alone and, with explicit
//     scaling up to 1000, past the end of the table so the clamp fires. Note
//     these are options worth ~0 or ~fwd: what is being pinned is the row
//     selection, and a wrong row changes the abscissae by orders of magnitude.
//
//  4. BatesEngine's own two constructors. v143_pe_heston pins the three
//     DERIVED engines and never the base, and the upstream
//     tests/.../test_bates_engine.py cross-validates against a v1.42.1-era
//     reference at LOOSE tolerance. Both constructors are pinned here:
//     (model, integrationOrder) with the order swept down to 1, and
//     (model, relTolerance, maxEvaluations), which routes through
//     Integration::gaussLobatto(relTolerance, Null<Real>(), maxEvaluations) and
//     therefore integrates the TRANSFORMED integrand on [0, 1] instead of the
//     raw one on [0, inf). Same answer to ~1e-10, entirely different code path.
//     Jump parameters are away from every C++ default (lambda default is 0.1,
//     nu 0.0, delta 0.0) so a port that leaves one at its default is caught.
//
//  5. AnalyticPTDHestonEngine::lnChF's own guard. analyticptdhestonengine.cpp:166-168
//         QL_REQUIRE(T <= lastModelTime, "maturity ... is too large ...")
//     is UNREACHABLE through calculate(), whose own guard (.cpp:272-275) fires
//     first on a shorter fuse. It is reachable only by calling the public
//     lnChF/chF directly, which is what a calibration routine does.
//
//  6. AnalyticPTDHestonEngine::lnChF at T landing EXACTLY on an interior grid
//     point. `lastI = lower_bound(timeGrid, T)` returns the index OF that point
//     when T equals it, so the segment starting there is NOT walked. Upstream
//     pins t in {0.25, 0.75, 1.5, 2.0}; only 2.0 is a grid point and it is the
//     last one, where the off-by-one cannot show. t = 0.5 and t = 1.0 are the
//     interior grid points, and a port using upper_bound instead of lower_bound
//     walks one segment too many at exactly those two values.
//
// Evaluation date
// ---------------
// Settings::instance().evaluationDate() = Date(1, March, 2025) -- the same date
// v143_pe_heston pins, so a Python test may share one fixture across both.
// 1-Mar-2025 -> 1-Mar-2026 is exactly 365 days, so with Actual365Fixed the 1y
// exercise time is exactly 1.0 and no day-count rounding leaks into a value.
//
// Emits JSON on stdout; nothing else may be printed.

#include <complex>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
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
#include <ql/pricingengines/vanilla/exponentialfittinghestonengine.hpp>
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
// Market fixtures. Same shape as v143_pe_heston so a Python test can share the
// reconstruction helpers.
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

struct Market {
    std::string name;
    Real s0;
    Rate r;
    Rate q;
};

// r != q so a wrong forward cannot hide: every engine here divides by the
// dividend discount at least once.
const Market kMarketA{"A", 100.0, 0.05, 0.02};

void describe(Obj& in, const Market& m) {
    in.s("market", m.name);
    in.n("s0", m.s0);
    in.n("r", m.r);
    in.n("q", m.q);
    in.s("day_counter", "Actual365Fixed");
    in.s("evaluation_date", "2025-03-01");
}

struct HParams {
    std::string name;
    Real v0, kappa, theta, sigma, rho;
};

// Chosen so optimalControlVariate() returns AsymptoticChF -- the arm no case in
// references/v143/pe/heston.json reaches. Feller: 2*0.1*0.01 - 1 = -0.998,
// violated, which is the regime the asymptotic expansion is FOR.
const HParams kAsymp{"asymp", 0.01, 0.10, 0.01, 1.00, -0.90};
// Straddles condition (b): (v0 + t*kappa*theta)/sigma*sqrt(1-rho^2) against
// 0.15. At t = 1 this evaluates to just under 0.15, and the `_t5` sweep pushes
// it over, flipping the answer with (a) and (c) both still satisfied.
const HParams kNearB{"nearB", 0.02, 0.50, 0.05, 1.00, -0.80};
// Straddles condition (c): the log(4*(1-rho^2)) term is NEGATIVE for
// |rho| > sqrt(3)/2, and dropping it flips this set to the wrong arm.
const HParams kNearC{"nearC", 0.05, 1.20, 0.09, 1.30, -0.95};
// Alan Lewis reference parameters (test-suite/hestonmodel.cpp:1428-1432), kept
// so the Bates section below shares a parameter set with v143_pe_heston and a
// port cannot pass one probe by tuning to the other.
const HParams kLewis{"lewis", 0.04, 4.00, 0.25, 1.00, -0.50};

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

struct Maturity {
    std::string name;
    int days;
};

const Maturity kT1m{"1m", 30};
const Maturity kT1y{"1y", 365};
const Maturity kT5y{"5y", 1826};

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

const char* cvName(AnalyticHestonEngine::ComplexLogFormula f) {
    switch (f) {
        case AnalyticHestonEngine::Gatheral: return "Gatheral";
        case AnalyticHestonEngine::BranchCorrection: return "BranchCorrection";
        case AnalyticHestonEngine::AndersenPiterbarg: return "AndersenPiterbarg";
        case AnalyticHestonEngine::AndersenPiterbargOptCV: return "AndersenPiterbargOptCV";
        case AnalyticHestonEngine::AsymptoticChF: return "AsymptoticChF";
        case AnalyticHestonEngine::AngledContour: return "AngledContour";
        case AnalyticHestonEngine::AngledContourNoCV: return "AngledContourNoCV";
        case AnalyticHestonEngine::OptimalCV: return "OptimalCV";
    }
    return "?";
}

// ===========================================================================
// 1. optimalControlVariate() -- the AsymptoticChF arm
// ===========================================================================
void emitOptimalControlVariateCases() {
    // The three conditions are evaluated at these t so each of (a), (b) and (c)
    // is the binding one somewhere in the sweep. 0.15 is (a)'s exact threshold
    // and the comparison is strict, so t = 0.15 must return AngledContour.
    const Real ts[] = {0.05, 0.15, 0.2, 1.0, 2.0, 5.0};

    for (const auto& p : {kAsymp, kNearB, kNearC, kLewis}) {
        for (const Real t : ts) {
            const auto chosen =
                AnalyticHestonEngine::optimalControlVariate(t, p.v0, p.kappa, p.theta,
                                                            p.sigma, p.rho);
            Obj in;
            describe(in, p);
            in.n("t", t);
            // The three condition values, so a failing port can be diagnosed to
            // the individual comparison rather than only to the verdict.
            in.n("cond_a", t);
            in.n("cond_b", (p.v0 + t * p.kappa * p.theta) / p.sigma *
                               std::sqrt(1 - p.rho * p.rho));
            in.n("cond_c", ((p.kappa - 0.5 * p.rho * p.sigma) *
                                (p.v0 + t * p.kappa * p.theta) +
                            p.kappa * p.theta * std::log(4 * (1 - p.rho * p.rho))) /
                               (p.sigma * p.sigma));
            Obj ex;
            ex.s("optimal_control_variate", cvName(chosen));
            addCase("optcv_" + p.name + "_t" + num(t), in, ex);
        }
    }
}

// ===========================================================================
// 2 + 3. ExponentialFittingHestonEngine -- resolved-OptimalCV scaling, and the
//        moneyness-row clamp
// ===========================================================================
void emitExpFitCases() {
    const Option::Type types[] = {Option::Call, Option::Put};

    // --- OptimalCV that RESOLVES to AsymptoticChF ---------------------------
    // Paired with the same market priced under an explicit AsymptoticChF. The
    // two must be bit-identical: `analyticCV` is the only thing the scaling
    // default looks at, so if a port computes the scaling from `cv_` instead
    // the first of each pair moves and the second does not.
    for (const auto& p : {kAsymp, kNearC}) {
        const auto model = hestonModel(kMarketA, p);
        for (const auto& mat : {kT1y, kT5y}) {
            for (const Real k : {70.0, 100.0, 140.0}) {
                for (const Option::Type type : types) {
                    const std::string suffix = p.name + "_" + mat.name + "_" +
                                               typeName(type) + "_k" + num(k);
                    Obj in;
                    describe(in, kMarketA);
                    describe(in, p);
                    describe(in, mat);
                    in.s("type", typeName(type));
                    in.n("strike", k);
                    in.s("scaling", "Null");
                    in.n("alpha", -0.5);
                    in.s("resolved_control_variate",
                         cvName(AnalyticHestonEngine::optimalControlVariate(
                             dayCounter().yearFraction(kToday, maturityDate(mat)), p.v0,
                             p.kappa, p.theta, p.sigma, p.rho)));
                    {
                        Obj inA = in;
                        inA.s("control_variate", "OptimalCV");
                        VanillaOption option(payoff(type, k), exercise(mat));
                        option.setPricingEngine(
                            ext::make_shared<ExponentialFittingHestonEngine>(
                                model, AnalyticHestonEngine::OptimalCV));
                        Obj ex;
                        ex.n("npv", option.NPV());
                        addCase("expfit_optcv_" + suffix, inA, ex);
                    }
                    {
                        Obj inB = in;
                        inB.s("control_variate", "AsymptoticChF");
                        VanillaOption option(payoff(type, k), exercise(mat));
                        option.setPricingEngine(
                            ext::make_shared<ExponentialFittingHestonEngine>(
                                model, AnalyticHestonEngine::AsymptoticChF));
                        Obj ex;
                        ex.n("npv", option.NPV());
                        addCase("expfit_explicit_asymp_" + suffix, inB, ex);
                    }
                }
            }
        }
    }

    // --- moneyness-row selection driven deep into the table ------------------
    // freq = log(fwd/strike); at market A / 1y the forward is 100*exp(0.03), so
    // strike 1 gives freq ~ +4.64 and strike 10000 gives freq ~ -4.57. Combined
    // with explicit scalings up to 1000 this sweeps |scaling*freq| from ~1 to
    // ~4600, i.e. from the low rows of the 147-row table, through its last
    // omega (49.484168050663322), and past it so the min() clamp fires.
    {
        const auto model = hestonModel(kMarketA, kLewis);
        const Real strikes[] = {1.0, 10.0, 50.0, 200.0, 1000.0, 10000.0};
        const Real scalings[] = {1.0, 10.0, 1000.0};
        for (const Real k : strikes) {
            for (const Real scaling : scalings) {
                for (const Option::Type type : types) {
                    VanillaOption option(payoff(type, k), exercise(kT1y));
                    option.setPricingEngine(
                        ext::make_shared<ExponentialFittingHestonEngine>(
                            model, AnalyticHestonEngine::AndersenPiterbarg, scaling));
                    Obj in;
                    describe(in, kMarketA);
                    describe(in, kLewis);
                    describe(in, kT1y);
                    in.s("type", typeName(type));
                    in.n("strike", k);
                    in.s("control_variate", "AndersenPiterbarg");
                    in.n("scaling", scaling);
                    in.n("alpha", -0.5);
                    Obj ex;
                    ex.n("npv", option.NPV());
                    addCase("expfit_wing_" + std::string(typeName(type)) + "_k" + num(k) +
                                "_s" + num(scaling),
                            in, ex);
                }
            }
        }
    }
}

// ===========================================================================
// 4. BatesEngine -- the base class, both constructors
// ===========================================================================
struct JumpParams {
    Real lambda, nu, delta;
};

// Away from every C++ default (lambda 0.1, nu 0.0, delta 0.0). A negative nu
// with a fat delta is the empirically usual calibration and makes the
// compensator `exp(nu + delta^2/2) - 1` differ from zero in both terms.
const JumpParams kJumps{0.7, -0.15, 0.25};

void describe(Obj& in, const JumpParams& j) {
    in.n("lambda", j.lambda);
    in.n("nu", j.nu);
    in.n("delta", j.delta);
}

ext::shared_ptr<BatesProcess> batesProcess(const Market& m, const HParams& p,
                                           const JumpParams& j) {
    return ext::make_shared<BatesProcess>(flatCurve(m.r), flatCurve(m.q), quote(m.s0), p.v0,
                                          p.kappa, p.theta, p.sigma, p.rho, j.lambda, j.nu,
                                          j.delta);
}

void emitBatesEngineCases() {
    const Real strikes[] = {60.0, 100.0, 160.0};
    const Option::Type types[] = {Option::Call, Option::Put};
    // 1/2/4/8 are far too small to converge and are the ONLY orders at which an
    // engine that accepts `integrationOrder` and discards it can be caught.
    const Size orders[] = {1, 2, 4, 8, 64, 144};

    const auto model = ext::make_shared<BatesModel>(
        batesProcess(kMarketA, kLewis, kJumps));

    for (const auto& mat : {kT1m, kT1y}) {
        for (const Size order : orders) {
            if (order != 144 && mat.name != "1y")
                continue;
            for (const Real k : strikes) {
                for (const Option::Type type : types) {
                    VanillaOption option(payoff(type, k), exercise(mat));
                    option.setPricingEngine(ext::make_shared<BatesEngine>(model, order));
                    Obj in;
                    describe(in, kMarketA);
                    describe(in, kLewis);
                    describe(in, kJumps);
                    describe(in, mat);
                    in.s("type", typeName(type));
                    in.n("strike", k);
                    in.i("integration_order", static_cast<long long>(order));
                    Obj ex;
                    ex.n("npv", option.NPV());
                    addCase("batesbase_" + mat.name + "_" + typeName(type) + "_k" + num(k) +
                                "_o" + std::to_string(order),
                            in, ex);
                }
            }
        }
    }

    // --- the (relTolerance, maxEvaluations) constructor ----------------------
    // Gatheral + Integration::gaussLobatto(relTolerance, Null<Real>(),
    // maxEvaluations): the Gatheral branch passes no maxBound, so Lobatto
    // integrates the TRANSFORMED integrand on [0, 1] rather than the raw one on
    // [0, inf). Nothing in references/v143/pe/heston.json exercises this
    // constructor of the BASE class.
    {
        const Real relTolerance = 1e-10;
        const Size maxEvaluations = 10000;
        for (const Real k : strikes) {
            for (const Option::Type type : types) {
                VanillaOption option(payoff(type, k), exercise(kT1y));
                option.setPricingEngine(
                    ext::make_shared<BatesEngine>(model, relTolerance, maxEvaluations));
                Obj in;
                describe(in, kMarketA);
                describe(in, kLewis);
                describe(in, kJumps);
                describe(in, kT1y);
                in.s("type", typeName(type));
                in.n("strike", k);
                in.n("rel_tolerance", relTolerance);
                in.i("max_evaluations", static_cast<long long>(maxEvaluations));
                Obj ex;
                ex.n("npv", option.NPV());
                addCase("batesbase_lobatto_" + std::string(typeName(type)) + "_k" + num(k),
                        in, ex);
            }
        }
    }

    // --- addOnTerm, approached through the vanishing-jump limit --------------
    // The hook is protected in C++, so it is pinned through the only public
    // consequence available: as lambda -> 0 the whole term vanishes and the
    // Bates price must collapse onto the plain-Heston Gatheral price. Both
    // numbers are emitted for each case so a port has to reproduce the pair,
    // not merely land near one of them. A port whose add-on leaks a spurious
    // real part -- the classic `complex<Real> g(i, phi)` argument-order bug --
    // does not collapse.
    //
    // lambda cannot be set to exactly 0: BatesModel's ctor wraps it in a
    // ConstantParameter(PositiveConstraint) (batesmodel.cpp:47), whose test is
    // a STRICT x > 0, so BatesModel(process) with lambda == 0 throws
    // "0: invalid value" before any pricing happens. That rejection is itself
    // pinned below -- it is the reason the limit has to be approached rather
    // than taken.
    {
        const JumpParams tiny{1e-13, -0.15, 0.25};
        const auto tinyJumpModel =
            ext::make_shared<BatesModel>(batesProcess(kMarketA, kLewis, tiny));
        const auto plain = hestonModel(kMarketA, kLewis);
        for (const Real k : strikes) {
            for (const Option::Type type : types) {
                VanillaOption bates(payoff(type, k), exercise(kT1y));
                bates.setPricingEngine(ext::make_shared<BatesEngine>(tinyJumpModel, 144));
                VanillaOption heston(payoff(type, k), exercise(kT1y));
                heston.setPricingEngine(ext::make_shared<AnalyticHestonEngine>(
                    plain, AnalyticHestonEngine::Gatheral,
                    AnalyticHestonEngine::Integration::gaussLaguerre(144)));
                Obj in;
                describe(in, kMarketA);
                describe(in, kLewis);
                describe(in, tiny);
                describe(in, kT1y);
                in.s("type", typeName(type));
                in.n("strike", k);
                in.i("integration_order", 144);
                Obj ex;
                ex.n("npv", bates.NPV());
                ex.n("plain_heston_npv", heston.NPV());
                addCase("batesbase_tinyjump_" + std::string(typeName(type)) + "_k" + num(k),
                        in, ex);
            }
        }
    }

    // --- BatesModel rejects a zero jump intensity ----------------------------
    // batesmodel.cpp:45-47 -- nu is NoConstraint, but delta and lambda are both
    // ConstantParameter(..., PositiveConstraint()) and PositiveConstraint::test
    // is `x > 0.0`. Zero is therefore rejected for BOTH, in the MODEL ctor, not
    // in the engine. A port that used a non-negative constraint (or none)
    // silently accepts a degenerate model.
    {
        const std::pair<std::string, JumpParams> bad[] = {
            {"lambda", JumpParams{0.0, -0.15, 0.25}},
            {"delta", JumpParams{0.7, -0.15, 0.0}},
        };
        for (const auto& [which, jumps] : bad) {
            bool threw = false;
            try {
                const auto m = ext::make_shared<BatesModel>(
                    batesProcess(kMarketA, kLewis, jumps));
                (void)m;
            } catch (const std::exception&) {
                threw = true;
            }
            Obj in;
            describe(in, jumps);
            in.s("zero_parameter", which);
            Obj ex;
            ex.b("throws", threw);
            addCase("batesbase_rejects_zero_" + which, in, ex);
        }
    }
}

// ===========================================================================
// 5 + 6. AnalyticPTDHestonEngine -- lnChF's own guard and the lower_bound edge
// ===========================================================================
struct PtdSegments {
    std::vector<Real> theta, kappa, sigma, rho;
    Real v0;
};

// Same three-segment model v143_pe_heston uses, so the two probes agree on the
// model and disagree only on what they ask of it.
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

void emitPtdCases() {
    const auto model = ptdModel(kMarketA, kPtd);
    const auto engine = ext::make_shared<AnalyticPTDHestonEngine>(model, 144);

    // --- lnChF/chF exactly ON the interior grid points -----------------------
    // lastI = lower_bound(timeGrid, T). At T == 0.5 that is index 1, so exactly
    // one segment is walked; upper_bound would return 2 and walk two. The
    // difference is invisible at any T strictly between grid points, which is
    // where every upstream chF case sits (0.25, 0.75, 1.5) except the final
    // 2.0, where there is no following segment to walk by mistake.
    {
        const Real us[] = {0.0, 0.6, 2.5};
        const Real vs[] = {0.0, -0.5, 1.25};
        const Real ts[] = {0.5, 1.0};
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
                    addCase("ptdgrid_chf_u" + num(u) + "_v" + num(v) + "_t" + num(t), in, ex);
                }
            }
        }
    }

    // --- lnChF's own "maturity is too large" guard ---------------------------
    // Unreachable through calculate(); reachable by calling lnChF directly.
    {
        for (const Real t : {2.0000000001, 3.0}) {
            bool threw = false;
            try {
                (void)engine->lnChF(std::complex<Real>(1.0, -0.5), t);
            } catch (const std::exception&) {
                threw = true;
            }
            Obj in;
            in.n("t", t);
            in.n("time_grid_back", ptdGridTimes().back());
            Obj ex;
            ex.b("throws", threw);
            addCase("ptdgrid_lnchf_rejects_t" + num(t), in, ex);
        }
        // Positive control at exactly the last grid point: the comparison is
        // `T <= lastModelTime`, so 2.0 must NOT throw.
        bool threw = false;
        try {
            (void)engine->lnChF(std::complex<Real>(1.0, -0.5), 2.0);
        } catch (const std::exception&) {
            threw = true;
        }
        Obj in;
        in.n("t", 2.0);
        in.n("time_grid_back", ptdGridTimes().back());
        Obj ex;
        ex.b("throws", threw);
        addCase("ptdgrid_lnchf_accepts_t2", in, ex);
    }
}

}  // namespace

int main() {
    try {
        Settings::instance().evaluationDate() = kToday;

        emitOptimalControlVariateCases();
        emitExpFitCases();
        emitBatesEngineCases();
        emitPtdCases();

        emitDocument();
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
