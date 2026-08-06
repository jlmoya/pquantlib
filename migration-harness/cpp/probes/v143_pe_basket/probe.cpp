// migration-harness/cpp/probes/v143_pe_basket/probe.cpp
//
// Reference values for the C++ QuantLib v1.43 basket pricing engines that had
// no Python counterpart:
//
//   * detail::VectorBsmProcessExtractor  (basket/vectorbsmprocessextractor.{hpp,cpp})
//   * detail::SumExponentialsRootSolver  (basket/singlefactorbsmbasketengine.{hpp,cpp})
//   * SingleFactorBsmBasketEngine        (basket/singlefactorbsmbasketengine.{hpp,cpp})
//   * BjerksundStenslandSpreadEngine     (basket/bjerksundstenslandspreadengine.{hpp,cpp})
//   * OperatorSplittingSpreadEngine      (basket/operatorsplittingspreadengine.{hpp,cpp})
//   * ChoiBasketEngine                   (basket/choibasketengine.{hpp,cpp})
//   * DengLiZhouBasketEngine             (basket/denglizhoubasketengine.{hpp,cpp})
//   * MCEuropeanBasketEngine / EuropeanMultiPathPricer / MakeMCEuropeanBasketEngine
//   * MCAmericanBasketEngine / AmericanBasketPathPricer / MakeMCAmericanBasketEngine
//   * ChoiAsianEngine                    (asian/choiasianengine.{hpp,cpp})
//
// What has to be pinned, and why
// ------------------------------
// None of these engines fills a Greeks block: BasketOption's results carry
// Greeks fields that stay Null<Real>(), so option.delta() throws. The whole
// observable surface is NPV plus, for two engines, additionalResults:
//   * SingleFactorBsmBasketEngine writes additionalResults["d"] -- and ONLY on
//     the non-degenerate branch. When every stdDev is close_enough(0) it takes
//     the intrinsic branch and writes NO "d" at all. Both are pinned.
//   * ChoiBasketEngine writes additionalResults["forwardDelta 0"] ..
//     ["forwardDelta n-1"], and only when calcFwdDelta_ is set. Note that
//     calcFwdDelta_ = (calcfwdDelta || controlVariate) -- turning ON the
//     control variate silently turns ON the deltas too. Pinned explicitly.
//
// Behaviour a port would plausibly get wrong, and which cases catch it
// -------------------------------------------------------------------
//  1. VectorBsmProcessExtractor::getInterestRateDf compares the per-process
//     DISCOUNT FACTORS with close_enough(), not the curve pointers and not the
//     curve objects. Two *distinct* FlatForward objects at the same rate are
//     accepted; two different rates throw. Cases vx_distinct_but_equal_curves_ok
//     and vx_different_rates_throws pin both directions. (This differs from
//     GaussianCopulaSpreadEngine, which really does compare currentLink()
//     pointers -- do not copy that check into this class.)
//  2. getBlackStdDev returns blackVol * sqrt(t) -- SIGNED. getBlackVariance
//     returns blackVariance -- always non-negative. A BlackConstantVol built
//     with a NEGATIVE volatility therefore yields a negative std-dev but a
//     positive variance, and SingleFactorBsmBasketEngine relies on exactly that
//     to admit negative-weight legs (its a*sig >= 0 guard). The upstream
//     test-suite market {200, 50, -125} x {0.4, 0.3, -0.5} does this on
//     purpose. Case vx_signed_std_dev pins the sign split; a port that returns
//     sqrt(variance) from getBlackStdDev fails it and then fails every
//     three-asset SingleFactorBsm case.
//  3. SumExponentialsRootSolver::getRoot is NOT a black-box root find. It
//     validates a*sig >= 0 elementwise, rejects K <= 0 when every a_i > 0,
//     builds a specific linear-approximation start point
//         xInit = clamp((K - sum a) / sum(a*sig), -10, 10),
//     falling back to 0 when |sum(a*sig)| <= 1000*QL_EPSILON, and then calls
//     QuantLib's Brent/Newton/Ridder/Halley with step 1.0. The evaluation
//     counters (getFCtr / getDerivativeCtr / getSecondDerivativeCtr) are pinned
//     for every strategy: they are the only observable that proves the port
//     runs the same iteration rather than delegating to scipy.
//  4. OperatorSplittingSpreadEngine::calculate takes a completely different
//     second-order branch when rs = (rho*vol1 - sig2)^2 < QL_EPSILON^0.625.
//     Case block osse_degenerate_* is constructed so that rs is EXACTLY zero:
//     f2 = 100, k = 25 => f2/(f2+k) = 0.8 exactly; vol2 = 0.25 => sig2 = 0.2
//     exactly; vol1 = 0.4, rho = 0.5 => rho*vol1 = 0.2 exactly. A port that
//     only transcribes the generic branch divides by e2 = 0 and returns NaN.
//     Also: the Put value is NOT computed from a put formula -- it is
//     callPrice - df*(f1-f2-k), i.e. exact put-call parity by construction.
//  5. ChoiBasketEngine flips the sign of a component of vStar1 whenever
//     sign(g[i])*vStar1[i] < tol*stdDev[i] (tol = 100*sqrt(QL_EPSILON)), and
//     then solves a triangular system by hand instead of using q1 = C^T g. The
//     four-asset golden market with weights {1, -2, -1, 4} exercises the flip.
//     It also RESCALES lambda by 0.9 in a do/while until the product of the
//     per-dimension quadrature orders fits maxNrIntegrationSteps -- so the
//     lambda and maxNrIntegrationSteps knobs interact; both are swept.
//  6. DengLiZhouBasketEngine sorts (weight, index, spot, dq, variance) tuples
//     with std::greater<> -- a LEXICOGRAPHIC descending sort on the whole
//     tuple, not on the weight alone -- and requires at least one strictly
//     positive and one non-positive weight. For a NEGATIVE strike it appends a
//     synthetic asset (1.0, n, -K, dr0, 0.0) with zero correlation to
//     everything and then prices with K = 0. Both weight-count guards and the
//     negative-strike branch are pinned.
//  7. MCEuropeanBasketEngine passes brownianBridge_ straight into
//     MultiPathGenerator, which QL_FAILs with "Brownian bridge not supported"
//     for multi-variate paths. withBrownianBridge(true) therefore THROWS at
//     pricing time rather than doing anything. Pinned.
//  8. AmericanBasketPathPricer regresses on the per-asset states
//     path[i][t] * scalingValue_, where scalingValue_ = 1/strike when the
//     basket payoff wraps a StrikedTypePayoff -- and appends the payoff itself
//     as an extra basis function, so the basis size is
//     multiPathBasisSystem(assets, order, type).size() + 1. It is a multi-asset
//     basis over the SCALED asset vector, not a single-variable basis over the
//     max of the basket.
//  9. ChoiAsianEngine folds a fixing at t == 0 into the past fixings (it
//     mutates pastFixings and runningAccumulator), and its effective strike is
//     payoff->strike() - runningAccumulator/(pastFixings + futureFixings) --
//     divided by the TOTAL fixing count, not by pastFixings. It then has three
//     branches: futureFixings == 0 (pure intrinsic on the running average),
//     == 1 (blackFormula), and > 1 (ChoiBasketEngine with weights
//     1/(future+past)). All three are pinned, plus the
//     "effective strike should to be positive" throw.
//
// Monte Carlo
// -----------
// Every MC case fixes a NON-ZERO seed and pins the EXACT NPV and the EXACT
// error estimate, not a statistical band: PseudoRandom == MersenneTwisterUniformRng
// + InverseCumulativeNormal is fully reproducible. mc_rng_* pins the raw
// Gaussian stream and the first generated MultiPath so that a mismatch is
// bisectable into "wrong RNG" / "wrong path" / "wrong pricer" instead of a bare
// NPV disagreement.
//
// Emits JSON on stdout. Nothing else may be printed.

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/math/array.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/methods/montecarlo/multipathgenerator.hpp>
#include <ql/pricingengines/asian/choiasianengine.hpp>
#include <ql/pricingengines/basket/bjerksundstenslandspreadengine.hpp>
#include <ql/pricingengines/basket/choibasketengine.hpp>
#include <ql/pricingengines/basket/denglizhoubasketengine.hpp>
#include <ql/pricingengines/basket/mcamericanbasketengine.hpp>
#include <ql/pricingengines/basket/mceuropeanbasketengine.hpp>
#include <ql/pricingengines/basket/operatorsplittingspreadengine.hpp>
#include <ql/pricingengines/basket/singlefactorbsmbasketengine.hpp>
#include <ql/pricingengines/basket/vectorbsmprocessextractor.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/stochasticprocessarray.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
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

    Obj& a(const std::string& k, const std::vector<Real>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j ? ", " : "") + num(v[j]);
        return put(k, body + "]");
    }
    Obj& a(const std::string& k, const Array& v) {
        return a(k, std::vector<Real>(v.begin(), v.end()));
    }
    Obj& ai(const std::string& k, const std::vector<long long>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j ? ", " : "") + std::to_string(v[j]);
        return put(k, body + "]");
    }
    Obj& as(const std::string& k, const std::vector<std::string>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j ? ", " : "") + ("\"" + v[j] + "\"");
        return put(k, body + "]");
    }
    Obj& m(const std::string& k, const Matrix& mat) {
        std::string body = "[";
        for (Size r = 0; r < mat.rows(); ++r) {
            body += (r ? ", [" : "[");
            for (Size c = 0; c < mat.columns(); ++c)
                body += (c ? ", " : "") + num(mat[r][c]);
            body += "]";
        }
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
    for (std::size_t j = 0; j < gCases.size(); ++j)
        std::cout << "  \"" << gCases[j].first << "\": " << gCases[j].second
                  << (j + 1 < gCases.size() ? "," : "") << "\n";
    std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// Market helpers. Every curve/vol is built at an explicit reference date, so
// Settings::evaluationDate() only matters for Instrument::isExpired(); it is
// still set per section (and echoed into every case's inputs) because
// BasketOption::isExpired() short-circuits NPV to 0 past maturity.
// ---------------------------------------------------------------------------
const DayCounter& dc() {
    static const DayCounter d = Actual365Fixed();
    return d;
}

std::string isoDate(const Date& d) {
    std::ostringstream o;
    o << std::setw(4) << std::setfill('0') << int(d.year()) << "-" << std::setw(2)
      << std::setfill('0') << int(d.month()) << "-" << std::setw(2) << std::setfill('0')
      << int(d.dayOfMonth());
    return o.str();
}

Handle<Quote> quote(Real v) { return Handle<Quote>(ext::make_shared<SimpleQuote>(v)); }

Handle<YieldTermStructure> flatCurve(const Date& ref, Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(ref, r, dc()));
}

Handle<BlackVolTermStructure> flatVol(const Date& ref, Volatility v) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(ref, NullCalendar(), v, dc()));
}

using Process = ext::shared_ptr<GeneralizedBlackScholesProcess>;

// Merton leg: spot, own dividend curve, SHARED risk-free curve, flat vol.
Process merton(const Date& ref, Real spot, Rate q, Volatility vol,
               const Handle<YieldTermStructure>& rTS) {
    return ext::make_shared<BlackScholesMertonProcess>(quote(spot), flatCurve(ref, q), rTS,
                                                       flatVol(ref, vol));
}

// Black (futures) leg: q == r, so the forward is the spot.
Process black(const Date& ref, Real fwd, Volatility vol,
              const Handle<YieldTermStructure>& rTS) {
    return ext::make_shared<BlackProcess>(quote(fwd), rTS, flatVol(ref, vol));
}

const char* typeName(Option::Type t) { return t == Option::Call ? "Call" : "Put"; }

Option::Type typeOf(const std::string& s) {
    return s == "Call" ? Option::Call : Option::Put;
}

bool throwsDuring(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// ===========================================================================
// Section 1 -- detail::VectorBsmProcessExtractor
// ===========================================================================
// Market "vx": 3 Merton legs sharing one risk-free curve. Leg 2 carries a
// NEGATIVE volatility on purpose so getBlackStdDev (signed) and
// getBlackVariance (non-negative) can be told apart.
const Date kVxToday(1, March, 2025);
const Date kVxMaturity(1, March, 2026); // exactly 365 days => t = 1 exactly
const Rate kVxRate = 0.05;

const std::vector<Real> kVxSpots = {100.0, 50.0, -125.0};
const std::vector<Rate> kVxQ = {0.03, 0.075, 0.04};
const std::vector<Volatility> kVxVols = {0.4, 0.3, -0.5};

std::vector<Process> vxProcesses(const Handle<YieldTermStructure>& rTS) {
    std::vector<Process> p;
    for (Size k = 0; k < kVxSpots.size(); ++k)
        p.push_back(merton(kVxToday, kVxSpots[k], kVxQ[k], kVxVols[k], rTS));
    return p;
}

Obj vxInputs() {
    Obj o;
    o.s("market", "vx")
        .s("today", isoDate(kVxToday))
        .s("maturity", isoDate(kVxMaturity))
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", kVxRate)
        .a("spots", kVxSpots)
        .a("dividend_yields", std::vector<Real>(kVxQ.begin(), kVxQ.end()))
        .a("volatilities", std::vector<Real>(kVxVols.begin(), kVxVols.end()));
    return o;
}

void emitVectorBsmProcessExtractor() {
    Settings::instance().evaluationDate() = kVxToday;

    const Handle<YieldTermStructure> rTS = flatCurve(kVxToday, kVxRate);
    const detail::VectorBsmProcessExtractor px(vxProcesses(rTS));

    {
        Obj ex;
        ex.a("spot", px.getSpot())
            .a("dividend_yield_df", px.getDividendYieldDf(kVxMaturity))
            .n("interest_rate_df", px.getInterestRateDf(kVxMaturity))
            .a("black_variance", px.getBlackVariance(kVxMaturity))
            .a("black_std_dev", px.getBlackStdDev(kVxMaturity));
        addCase("vx_extract", vxInputs(), ex);
    }

    // getBlackStdDev is blackVol*sqrt(t) and therefore SIGNED; getBlackVariance
    // is vol^2*t and therefore not. Leg 2 has vol = -0.5, so the two disagree
    // in sign while agreeing in magnitude.
    {
        const Array sd = px.getBlackStdDev(kVxMaturity);
        const Array var = px.getBlackVariance(kVxMaturity);
        Obj ex;
        ex.n("std_dev_leg2", sd[2])
            .n("variance_leg2", var[2])
            .n("sqrt_variance_leg2", std::sqrt(var[2]))
            .b("std_dev_leg2_is_negative", sd[2] < 0.0);
        Obj in = vxInputs();
        in.s("note", "leg 2 volatility is -0.5");
        addCase("vx_signed_std_dev", in, ex);
    }

    // The interest-rate check is close_enough() on the DISCOUNT FACTORS, not
    // pointer identity of the curves.
    {
        auto build = [](Rate r2) {
            const Handle<YieldTermStructure> a = flatCurve(kVxToday, kVxRate);
            const Handle<YieldTermStructure> b = flatCurve(kVxToday, r2);
            std::vector<Process> p = {merton(kVxToday, 100.0, 0.0, 0.2, a),
                                      merton(kVxToday, 50.0, 0.0, 0.3, b)};
            detail::VectorBsmProcessExtractor(p).getInterestRateDf(kVxMaturity);
        };

        Obj in;
        in.s("market", "vx_two_curves")
            .s("today", isoDate(kVxToday))
            .s("maturity", isoDate(kVxMaturity))
            .n("rate_1", kVxRate)
            .n("rate_2", kVxRate)
            .s("note", "two DISTINCT FlatForward objects at the same rate");
        Obj ex;
        ex.b("throws", throwsDuring([&] { build(kVxRate); }));
        addCase("vx_distinct_but_equal_curves_ok", in, ex);

        Obj in2;
        in2.s("market", "vx_two_curves")
            .s("today", isoDate(kVxToday))
            .s("maturity", isoDate(kVxMaturity))
            .n("rate_1", kVxRate)
            .n("rate_2", 0.06);
        Obj ex2;
        ex2.b("throws", throwsDuring([&] { build(0.06); }));
        addCase("vx_different_rates_throws", in2, ex2);
    }
}

// ===========================================================================
// Section 2 -- detail::SumExponentialsRootSolver
// ===========================================================================
struct SumExpCase {
    const char* name;
    std::vector<Real> a;
    std::vector<Real> sig;
    Real k;
};

// Strategy enum in v1.43 is exactly {Ridder, Newton, Brent, Halley}: four
// values, in that declaration order. There is no SuperHalley.
const std::vector<std::pair<std::string, detail::SumExponentialsRootSolver::Strategy>>&
sumExpStrategies() {
    static const std::vector<
        std::pair<std::string, detail::SumExponentialsRootSolver::Strategy>>
        v = {{"Ridder", detail::SumExponentialsRootSolver::Ridder},
             {"Newton", detail::SumExponentialsRootSolver::Newton},
             {"Brent", detail::SumExponentialsRootSolver::Brent},
             {"Halley", detail::SumExponentialsRootSolver::Halley}};
    return v;
}

void emitSumExponentialsRootSolver() {
    const std::vector<SumExpCase> cases = {
        // all a > 0, all sig > 0 -- the plain basket configuration
        {"pos", {2.0, 3.0, 4.0}, {0.2, 0.4, 0.1}, 12.0},
        // deep root: K far below sum(a) pushes xInit to the -10 clamp
        {"pos_low_k", {2.0, 3.0, 4.0}, {0.2, 0.4, 0.1}, 0.5},
        // K above sum(a): root on the positive side
        {"pos_high_k", {2.0, 3.0, 4.0}, {0.2, 0.4, 0.1}, 40.0},
        // mixed signs with a_i*sig_i >= 0 -- the spread configuration; K may be
        // non-positive because not every a_i is positive.
        {"mixed", {5.0, -3.0}, {0.3, -0.25}, 0.0},
        {"mixed_negk", {5.0, -3.0}, {0.3, -0.25}, -1.0},
        {"mixed3", {4.0, -1.5, 2.0}, {0.35, -0.2, 0.15}, 3.0},
        // single term: sum(a*sig) is fine, root is analytic
        {"single", {7.0}, {0.4}, 9.0},
        // sum(a*sig) == 0 exactly => the |denom| > 1000*QL_EPSILON guard fails
        // and xInit falls back to 0.0.
        {"zero_denominator", {2.0, -2.0}, {0.5, -0.5}, 0.0},
    };

    for (const SumExpCase& c : cases) {
        const Array a(c.a.begin(), c.a.end());
        const Array sig(c.sig.begin(), c.sig.end());

        Obj in;
        in.s("case", c.name).a("a", c.a).a("sig", c.sig).n("k", c.k);

        // Function / derivative / second-derivative values at fixed abscissae,
        // independent of any solver.
        {
            const detail::SumExponentialsRootSolver f(a, sig, c.k);
            std::vector<Real> xs = {-2.0, -0.5, 0.0, 0.5, 2.0};
            std::vector<Real> fv, dv, d2v;
            for (Real x : xs) {
                fv.push_back(f(x));
                dv.push_back(f.derivative(x));
                d2v.push_back(f.secondDerivative(x));
            }
            Obj ex;
            ex.a("x", xs)
                .a("f", fv)
                .a("derivative", dv)
                .a("second_derivative", d2v)
                .i("f_ctr", static_cast<long long>(f.getFCtr()))
                .i("derivative_ctr", static_cast<long long>(f.getDerivativeCtr()))
                .i("second_derivative_ctr",
                   static_cast<long long>(f.getSecondDerivativeCtr()));
            addCase(std::string("sumexp_values_") + c.name, in, ex);
        }

        // Root + evaluation counters, per strategy, at the default xTol.
        for (const auto& st : sumExpStrategies()) {
            const detail::SumExponentialsRootSolver solver(a, sig, c.k);
            const Real root = solver.getRoot(1e6 * QL_EPSILON, st.second);
            Obj inS = in;
            inS.s("strategy", st.first).n("x_tol", 1e6 * QL_EPSILON);
            Obj ex;
            ex.n("root", root)
                .n("residual", detail::SumExponentialsRootSolver(a, sig, c.k)(root))
                .i("f_ctr", static_cast<long long>(solver.getFCtr()))
                .i("derivative_ctr", static_cast<long long>(solver.getDerivativeCtr()))
                .i("second_derivative_ctr",
                   static_cast<long long>(solver.getSecondDerivativeCtr()));
            addCase(std::string("sumexp_root_") + c.name + "_" + st.first, inS, ex);
        }
    }

    // Tighter tolerance changes the iteration count, so a port that hardcodes
    // the default fails here.
    {
        const Array a({2.0, 3.0, 4.0}), sig({0.2, 0.4, 0.1});
        for (Real tol : {1e-4, 1e-10, 1e-14}) {
            const detail::SumExponentialsRootSolver solver(a, sig, 12.0);
            const Real root =
                solver.getRoot(tol, detail::SumExponentialsRootSolver::Brent);
            Obj in;
            in.s("case", "pos")
                .a("a", std::vector<Real>({2.0, 3.0, 4.0}))
                .a("sig", std::vector<Real>({0.2, 0.4, 0.1}))
                .n("k", 12.0)
                .s("strategy", "Brent")
                .n("x_tol", tol);
            Obj ex;
            ex.n("root", root)
                .i("f_ctr", static_cast<long long>(solver.getFCtr()))
                .i("derivative_ctr", static_cast<long long>(solver.getDerivativeCtr()))
                .i("second_derivative_ctr",
                   static_cast<long long>(solver.getSecondDerivativeCtr()));
            std::ostringstream nm;
            nm << "sumexp_root_pos_Brent_tol" << std::scientific << std::setprecision(0)
               << tol;
            addCase(nm.str(), in, ex);
        }
    }

    // Guards. Both come straight from the upstream test-suite case
    // testRootOfSumExponentials.
    {
        struct Guard {
            const char* name;
            std::vector<Real> a;
            std::vector<Real> sig;
            Real k;
            const char* why;
        };
        const std::vector<Guard> guards = {
            {"sumexp_rejects_negative_a_times_sig",
             {2.0, 3.0, 4.0},
             {0.2, 0.4, -0.1},
             0.0,
             "a*sig should not be negative"},
            {"sumexp_rejects_negative_a_times_sig_2",
             {2.0, -3.0, 4.0},
             {0.2, -0.4, -0.1},
             0.0,
             "a*sig should not be negative"},
            {"sumexp_rejects_non_positive_strike_when_all_a_positive",
             {2.0, 3.0},
             {0.2, 0.4},
             0.0,
             "non-positive strikes only allowed for spread options"},
            {"sumexp_rejects_negative_strike_when_all_a_positive",
             {2.0, 3.0},
             {0.2, 0.4},
             -5.0,
             "non-positive strikes only allowed for spread options"},
        };
        for (const Guard& g : guards) {
            Obj in;
            in.a("a", g.a).a("sig", g.sig).n("k", g.k).s("why", g.why);
            Obj ex;
            ex.b("throws", throwsDuring([&] {
                   const Array a(g.a.begin(), g.a.end());
                   const Array sig(g.sig.begin(), g.sig.end());
                   detail::SumExponentialsRootSolver(a, sig, g.k).getRoot();
               }));
            addCase(g.name, in, ex);
        }

        // Mismatched array sizes are rejected in the CONSTRUCTOR.
        Obj in;
        in.a("a", std::vector<Real>({1.0, 2.0}))
            .a("sig", std::vector<Real>({0.1}))
            .n("k", 1.0)
            .s("why", "Arrays must have the same size");
        Obj ex;
        ex.b("throws", throwsDuring([] {
               detail::SumExponentialsRootSolver(Array({1.0, 2.0}), Array({0.1}), 1.0);
           }));
        addCase("sumexp_ctor_rejects_size_mismatch", in, ex);
    }
}

// ===========================================================================
// Section 3 -- SingleFactorBsmBasketEngine
// ===========================================================================
// Market "sf": the upstream testSingleFactorBsmBasketEngine market, verbatim.
// Note the NEGATIVE spot (-125) and NEGATIVE volatilities: the engine works on
// the SIGNED std-dev, so a negative vol flips the sign of sig_i and lets a
// negative-weight leg satisfy the a*sig >= 0 guard.
const Date kSfToday(3, July, 2024);
const Date kSfMaturity = Date(3, July, 2024) + Period(18, Months);

struct SfCase {
    const char* name;
    std::vector<Real> underlyings;
    std::vector<Volatility> volatilities;
    std::vector<Rate> q;
    Rate r;
    std::vector<Real> weights;
    Option::Type type;
};

const std::vector<SfCase>& sfCases() {
    static const std::vector<SfCase> v = {
        {"3asset_call", {200, 50, -125}, {0.4, 0.3, -0.5}, {0.03, 0.075, 0.04}, 0.05,
         {0.5, 0.25, 1.0}, Option::Call},
        {"3asset_put", {200, 50, -125}, {0.4, 0.3, -0.5}, {0.03, 0.075, 0.04}, 0.05,
         {0.5, 0.25, 1.0}, Option::Put},
        {"2asset_put", {100, 50}, {0.4, -0.3}, {0.03, 0.075}, 0.025, {1.0, -2.0},
         Option::Put},
        {"2asset_call", {100, 50}, {0.4, -0.3}, {0.03, 0.075}, 0.025, {1.0, -2.0},
         Option::Call},
        {"1asset_call", {100}, {0.4}, {0.03}, 0.045, {1.0}, Option::Call},
        {"4asset_call", {100, 50, 100, 150}, {0.4, 0.0, 0.2, 0.1},
         {0.03, 0.05, 0.02, 0.0}, 0.045, {1.0, 2.0, 1.0, 1.0}, Option::Call},
        // every vol is zero -> the close_enough(stdDev, 0) branch, which
        // returns the DISCOUNTED INTRINSIC and writes NO additionalResults["d"].
        {"2asset_zero_vol_call", {100, 50}, {0.0, 0.0}, {0.03, 0.05}, 0.055,
         {1.0, 1.95}, Option::Call},
    };
    return v;
}

std::vector<Process> sfProcesses(const SfCase& c,
                                 const Handle<YieldTermStructure>& rTS) {
    std::vector<Process> p;
    for (Size k = 0; k < c.underlyings.size(); ++k)
        p.push_back(
            merton(kSfToday, c.underlyings[k], c.q[k], c.volatilities[k], rTS));
    return p;
}

Real sfStrike(const SfCase& c) {
    Real s = 0.0;
    for (Size k = 0; k < c.weights.size(); ++k)
        s += c.weights[k] * c.underlyings[k];
    return s;
}

Obj sfInputs(const SfCase& c) {
    Obj o;
    o.s("market", "sf")
        .s("case", c.name)
        .s("today", isoDate(kSfToday))
        .s("maturity", isoDate(kSfMaturity))
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", c.r)
        .a("spots", c.underlyings)
        .a("dividend_yields", std::vector<Real>(c.q.begin(), c.q.end()))
        .a("volatilities", std::vector<Real>(c.volatilities.begin(), c.volatilities.end()))
        .a("weights", c.weights)
        .n("strike", sfStrike(c))
        .s("option_type", typeName(c.type));
    return o;
}

void emitSingleFactorBsmBasketEngine() {
    Settings::instance().evaluationDate() = kSfToday;

    for (const SfCase& c : sfCases()) {
        const Handle<YieldTermStructure> rTS = flatCurve(kSfToday, c.r);
        const std::vector<Process> processes = sfProcesses(c, rTS);

        BasketOption option(
            ext::make_shared<AverageBasketPayoff>(
                ext::make_shared<PlainVanillaPayoff>(c.type, sfStrike(c)),
                Array(c.weights.begin(), c.weights.end())),
            ext::make_shared<EuropeanExercise>(kSfMaturity));
        option.setPricingEngine(
            ext::make_shared<SingleFactorBsmBasketEngine>(processes));

        const Real npv = option.NPV();
        const auto& extra = option.additionalResults();
        const bool hasD = extra.find("d") != extra.end();

        Obj ex;
        ex.n("npv", npv).b("has_d", hasD);
        if (hasD)
            ex.n("d", ext::any_cast<Real>(extra.at("d")));
        ex.i("additional_results_count", static_cast<long long>(extra.size()));
        addCase(std::string("sf_") + c.name, sfInputs(c), ex);
    }

    // xTol knob: a coarse tolerance moves the root and hence the price.
    {
        const SfCase& c = sfCases()[0];
        for (Real xTol : {1e4 * QL_EPSILON, 1e-6, 1e-3}) {
            const Handle<YieldTermStructure> rTS = flatCurve(kSfToday, c.r);
            BasketOption option(
                ext::make_shared<AverageBasketPayoff>(
                    ext::make_shared<PlainVanillaPayoff>(c.type, sfStrike(c)),
                    Array(c.weights.begin(), c.weights.end())),
                ext::make_shared<EuropeanExercise>(kSfMaturity));
            option.setPricingEngine(ext::make_shared<SingleFactorBsmBasketEngine>(
                sfProcesses(c, rTS), xTol));
            Obj in = sfInputs(c);
            in.n("x_tol", xTol);
            Obj ex;
            ex.n("npv", option.NPV())
                .n("d", ext::any_cast<Real>(option.additionalResults().at("d")));
            std::ostringstream nm;
            nm << "sf_3asset_call_xtol" << std::scientific << std::setprecision(0)
               << xTol;
            addCase(nm.str(), in, ex);
        }
    }

    // Guards.
    {
        const SfCase& c = sfCases()[0];
        const Handle<YieldTermStructure> rTS = flatCurve(kSfToday, c.r);
        const std::vector<Process> processes = sfProcesses(c, rTS);

        auto priceWith = [&](const ext::shared_ptr<BasketPayoff>& payoff,
                             const ext::shared_ptr<Exercise>& exercise) {
            BasketOption o(payoff, exercise);
            o.setPricingEngine(
                ext::make_shared<SingleFactorBsmBasketEngine>(processes));
            o.NPV();
        };

        struct G {
            const char* name;
            const char* why;
            std::function<void()> f;
        };
        const std::vector<G> guards = {
            {"sf_rejects_non_average_payoff", "average basket payoff expected",
             [&] {
                 priceWith(ext::make_shared<MaxBasketPayoff>(
                               ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0)),
                           ext::make_shared<EuropeanExercise>(kSfMaturity));
             }},
            {"sf_rejects_wrong_weight_count",
             "wrong number of weights arguments in payoff",
             [&] {
                 priceWith(ext::make_shared<AverageBasketPayoff>(
                               ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                               Array({1.0, 1.0})),
                           ext::make_shared<EuropeanExercise>(kSfMaturity));
             }},
            {"sf_rejects_american_exercise", "not an European exercise",
             [&] {
                 priceWith(ext::make_shared<AverageBasketPayoff>(
                               ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                               Array(3, 1.0)),
                           ext::make_shared<AmericanExercise>(kSfToday, kSfMaturity));
             }},
        };
        for (const G& g : guards) {
            Obj in;
            in.s("market", "sf").s("why", g.why);
            Obj ex;
            ex.b("throws", throwsDuring(g.f));
            addCase(g.name, in, ex);
        }
    }
}

// ===========================================================================
// Section 4 -- BjerksundStenslandSpreadEngine
// ===========================================================================
// Market "bs": the upstream testBjerksundStenslandSpreadEngine market.
// BlackProcess legs, so f1 = 100 and f2 = 110 exactly (q == r).
const Date kBsToday(1, March, 2024);
const Date kBsMaturity = Date(1, March, 2024) + Period(12, Months);
const Rate kBsRate = 0.05;

struct SpreadRow {
    const char* name;
    Option::Type type;
    Real strike;
    Real rho;
};

Obj spreadInputs(const char* market, const Date& today, const Date& maturity,
                 Rate r, Real f1, Volatility v1, Real f2, Volatility v2,
                 Option::Type type, Real strike, Real rho) {
    Obj o;
    o.s("market", market)
        .s("today", isoDate(today))
        .s("maturity", isoDate(maturity))
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", r)
        .n("forward1", f1)
        .n("volatility1", v1)
        .n("forward2", f2)
        .n("volatility2", v2)
        .s("option_type", typeName(type))
        .n("strike", strike)
        .n("correlation", rho);
    return o;
}

void emitBjerksundStenslandSpreadEngine() {
    Settings::instance().evaluationDate() = kBsToday;

    const Real f1 = 100.0, f2 = 110.0;
    const Volatility v1 = 0.25, v2 = 0.35;
    const Handle<YieldTermStructure> rTS = flatCurve(kBsToday, kBsRate);

    // f1 - f2 = -10, so strike -10 is the at-the-money spread strike.
    const std::vector<SpreadRow> rows = {
        {"bs_call_k5_rho075", Option::Call, 5.0, 0.75},
        {"bs_put_k5_rho075", Option::Put, 5.0, 0.75},
        {"bs_call_km10_rho075", Option::Call, -10.0, 0.75},
        {"bs_put_km10_rho075", Option::Put, -10.0, 0.75},
        {"bs_call_km30_rho075", Option::Call, -30.0, 0.75},
        {"bs_put_km30_rho075", Option::Put, -30.0, 0.75},
        {"bs_call_k0_rho075", Option::Call, 0.0, 0.75},
        {"bs_put_k0_rho075", Option::Put, 0.0, 0.75},
        {"bs_call_k40_rho075", Option::Call, 40.0, 0.75},
        {"bs_put_k40_rho075", Option::Put, 40.0, 0.75},
        {"bs_call_k5_rho000", Option::Call, 5.0, 0.0},
        {"bs_put_k5_rho000", Option::Put, 5.0, 0.0},
        {"bs_call_k5_rho_m090", Option::Call, 5.0, -0.9},
        {"bs_put_k5_rho_m090", Option::Put, 5.0, -0.9},
        {"bs_call_k5_rho_p1", Option::Call, 5.0, 1.0},
        {"bs_put_k5_rho_p1", Option::Put, 5.0, 1.0},
        {"bs_call_k5_rho_m1", Option::Call, 5.0, -1.0},
        {"bs_put_k5_rho_m1", Option::Put, 5.0, -1.0},
    };

    for (const SpreadRow& r : rows) {
        const Process p1 = black(kBsToday, f1, v1, rTS);
        const Process p2 = black(kBsToday, f2, v2, rTS);
        BasketOption option(
            ext::make_shared<SpreadBasketPayoff>(
                ext::make_shared<PlainVanillaPayoff>(r.type, r.strike)),
            ext::make_shared<EuropeanExercise>(kBsMaturity));
        option.setPricingEngine(
            ext::make_shared<BjerksundStenslandSpreadEngine>(p1, p2, r.rho));

        Obj ex;
        ex.n("npv", option.NPV())
            .i("additional_results_count",
               static_cast<long long>(option.additionalResults().size()));
        addCase(r.name, spreadInputs("bs", kBsToday, kBsMaturity, kBsRate, f1, v1,
                                     f2, v2, r.type, r.strike, r.rho),
                ex);
    }

    // Different dividend yields per leg, so fwd = spot*qDF/rDF is exercised.
    {
        const Process p1 = merton(kBsToday, 100.0, 0.02, 0.30, rTS);
        const Process p2 = merton(kBsToday, 95.0, 0.06, 0.15, rTS);
        for (const SpreadRow& r : std::vector<SpreadRow>{
                 {"bs_merton_call_k5_rho050", Option::Call, 5.0, 0.5},
                 {"bs_merton_put_k5_rho050", Option::Put, 5.0, 0.5},
                 {"bs_merton_call_km15_rho050", Option::Call, -15.0, 0.5}}) {
            BasketOption option(
                ext::make_shared<SpreadBasketPayoff>(
                    ext::make_shared<PlainVanillaPayoff>(r.type, r.strike)),
                ext::make_shared<EuropeanExercise>(kBsMaturity));
            option.setPricingEngine(
                ext::make_shared<BjerksundStenslandSpreadEngine>(p1, p2, r.rho));
            Obj in;
            in.s("market", "bs_merton")
                .s("today", isoDate(kBsToday))
                .s("maturity", isoDate(kBsMaturity))
                .n("risk_free_rate", kBsRate)
                .a("spots", std::vector<Real>({100.0, 95.0}))
                .a("dividend_yields", std::vector<Real>({0.02, 0.06}))
                .a("volatilities", std::vector<Real>({0.30, 0.15}))
                .s("option_type", typeName(r.type))
                .n("strike", r.strike)
                .n("correlation", r.rho);
            Obj ex;
            ex.n("npv", option.NPV());
            addCase(r.name, in, ex);
        }
    }
}

// ===========================================================================
// Section 5 -- OperatorSplittingSpreadEngine
// ===========================================================================
// Market "os": the Chi-Fai Lo paper market used upstream. BlackProcess legs
// whose spots are already the forwards
//     f1 = 110 * dq(3%) / df(5%),  f2 = 90 * dq(2%) / df(5%)
// so the Python side must reconstruct the same two numbers, not 110 and 90.
const Date kOsToday(1, March, 2025);
const Date kOsMaturity = Date(1, March, 2025) + 365; // Act/365F => t = 1 exactly
const Rate kOsRate = 0.05;

void emitOperatorSplittingSpreadEngine() {
    Settings::instance().evaluationDate() = kOsToday;

    const Handle<YieldTermStructure> rTS = flatCurve(kOsToday, kOsRate);
    const DiscountFactor df = rTS->discount(kOsMaturity);
    const DiscountFactor dq1 = flatCurve(kOsToday, 0.03)->discount(kOsMaturity);
    const DiscountFactor dq2 = flatCurve(kOsToday, 0.02)->discount(kOsMaturity);
    const Real f1 = 110 * dq1 / df, f2 = 90 * dq2 / df;
    const Volatility v1 = 0.3, v2 = 0.2;

    const std::vector<Real> rhos = {-0.9, -0.7, -0.5, -0.3, -0.1, 0.0,
                                    0.1,  0.3,  0.5,  0.7,  0.9};
    const std::vector<std::pair<std::string, OperatorSplittingSpreadEngine::Order>>
        orders = {{"First", OperatorSplittingSpreadEngine::First},
                  {"Second", OperatorSplittingSpreadEngine::Second}};

    auto emitOne = [&](const std::string& name, Real strike, Option::Type type,
                       Real rho, const std::string& orderName,
                       OperatorSplittingSpreadEngine::Order order, Real fwd1,
                       Volatility vol1, Real fwd2, Volatility vol2,
                       const char* market) {
        const Process p1 = black(kOsToday, fwd1, vol1, rTS);
        const Process p2 = black(kOsToday, fwd2, vol2, rTS);
        BasketOption option(ext::make_shared<SpreadBasketPayoff>(
                                ext::make_shared<PlainVanillaPayoff>(type, strike)),
                            ext::make_shared<EuropeanExercise>(kOsMaturity));
        option.setPricingEngine(ext::make_shared<OperatorSplittingSpreadEngine>(
            p1, p2, rho, order));
        Obj in = spreadInputs(market, kOsToday, kOsMaturity, kOsRate, fwd1, vol1,
                              fwd2, vol2, type, strike, rho);
        in.s("order", orderName);
        Obj ex;
        ex.n("npv", option.NPV())
            .i("additional_results_count",
               static_cast<long long>(option.additionalResults().size()));
        addCase(name, in, ex);
    };

    // Upstream rho sweep, both orders, K = 20 Call.
    for (Real rho : rhos)
        for (const auto& o : orders) {
            std::ostringstream nm;
            nm << "os_call_k20_rho" << std::fixed << std::setprecision(2) << rho
               << "_" << o.first;
            emitOne(nm.str(), 20.0, Option::Call, rho, o.first, o.second, f1, v1, f2,
                    v2, "os");
        }

    // Puts: the Put value is call - df*(f1 - f2 - k) by construction, so this
    // pins the parity branch of the callPutParityPrice lambda.
    for (Real rho : {-0.5, 0.0, 0.5})
        for (const auto& o : orders) {
            std::ostringstream nm;
            nm << "os_put_k20_rho" << std::fixed << std::setprecision(2) << rho << "_"
               << o.first;
            emitOne(nm.str(), 20.0, Option::Put, rho, o.first, o.second, f1, v1, f2,
                    v2, "os");
        }

    // Strike sweep including a NEGATIVE strike (f2 + k must stay positive).
    for (Real k : {-20.0, 0.0, 5.0, 50.0})
        for (const auto& o : orders) {
            std::ostringstream nm;
            nm << "os_call_k" << std::fixed << std::setprecision(0) << k << "_rho050_"
               << o.first;
            emitOne(nm.str(), k, Option::Call, 0.5, o.first, o.second, f1, v1, f2, v2,
                    "os");
        }

    // ---- the rs == 0 degenerate second-order branch -----------------------
    // BlackProcess => f2 = 100 exactly, k = 25 => f2/(f2+k) = 0.8 exactly;
    // vol2 = 0.25 => sig2 = 0.2 exactly; vol1 = 0.4, rho = 0.5 => rho*vol1 = 0.2.
    // Therefore rs = (rho*vol1 - sig2)^2 == 0 < QL_EPSILON^0.625 and the engine
    // takes the closed-form `ooPlt` branch instead of the generic one.
    {
        const Real gf1 = 120.0, gf2 = 100.0, gk = 25.0;
        const Volatility gv1 = 0.4, gv2 = 0.25;
        const Real grho = 0.5;
        for (const auto& o : orders)
            for (Option::Type t : {Option::Call, Option::Put}) {
                std::ostringstream nm;
                nm << "os_degenerate_" << (t == Option::Call ? "call" : "put") << "_"
                   << o.first;
                emitOne(nm.str(), gk, t, grho, o.first, o.second, gf1, gv1, gf2, gv2,
                        "os_degenerate");
            }

        // A nearby, NON-degenerate correlation so the two branches can be
        // compared: rs is tiny but above the QL_EPSILON^0.625 threshold.
        for (const auto& o : orders) {
            std::ostringstream nm;
            nm << "os_near_degenerate_call_" << o.first;
            emitOne(nm.str(), gk, Option::Call, 0.5000001, o.first, o.second, gf1, gv1,
                    gf2, gv2, "os_degenerate");
        }
    }

    // Unknown order values cannot be constructed from the enum, but the
    // engine's default order is Second -- pin that so a port cannot default to
    // First and pass the Second-order sweep by accident.
    {
        const Process p1 = black(kOsToday, f1, v1, rTS);
        const Process p2 = black(kOsToday, f2, v2, rTS);
        BasketOption option(
            ext::make_shared<SpreadBasketPayoff>(
                ext::make_shared<PlainVanillaPayoff>(Option::Call, 20.0)),
            ext::make_shared<EuropeanExercise>(kOsMaturity));
        option.setPricingEngine(
            ext::make_shared<OperatorSplittingSpreadEngine>(p1, p2, 0.5));
        Obj in = spreadInputs("os", kOsToday, kOsMaturity, kOsRate, f1, v1, f2, v2,
                              Option::Call, 20.0, 0.5);
        in.s("order", "default");
        Obj ex;
        ex.n("npv", option.NPV());
        addCase("os_default_order_is_second", in, ex);
    }
}

// ===========================================================================
// Section 6 -- ChoiBasketEngine
// ===========================================================================
// Market "choi": the upstream testGoldenChoiBasketEngineExample market.
const Date kChoiToday(26, September, 2024);
const Date kChoiMaturity = Date(26, September, 2024) + Period(18, Months);
const Rate kChoiRate = 0.05;

const std::vector<Real> kChoiSpots = {100.0, 50.0, 75.0, 25.0};
const std::vector<Rate> kChoiQ = {0.075, 0.035, 0.08, 0.02};
const std::vector<Volatility> kChoiVols = {0.45, 0.4, 0.35, 0.2};

Matrix choiRho(Size n) {
    static const Real full[4][4] = {{1.0, 0.2, 0.3, 0.0},
                                    {0.2, 1.0, -0.3, 0.1},
                                    {0.3, -0.3, 1.0, 0.7},
                                    {0.0, 0.1, 0.7, 1.0}};
    Matrix m(n, n);
    for (Size r = 0; r < n; ++r)
        for (Size c = 0; c < n; ++c)
            m[r][c] = full[r][c];
    return m;
}

std::vector<Process> choiProcesses(Size n, const Handle<YieldTermStructure>& rTS) {
    std::vector<Process> p;
    for (Size k = 0; k < n; ++k)
        p.push_back(merton(kChoiToday, kChoiSpots[k], kChoiQ[k], kChoiVols[k], rTS));
    return p;
}

Obj choiInputs(Size n, const std::vector<Real>& weights, Option::Type type,
               Real strike, Real lambda, long long maxSteps, bool calcFwdDelta,
               bool controlVariate, const char* payoffKind) {
    Obj o;
    o.s("market", "choi")
        .s("today", isoDate(kChoiToday))
        .s("maturity", isoDate(kChoiMaturity))
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", kChoiRate)
        .a("spots", std::vector<Real>(kChoiSpots.begin(), kChoiSpots.begin() + n))
        .a("dividend_yields", std::vector<Real>(kChoiQ.begin(), kChoiQ.begin() + n))
        .a("volatilities",
           std::vector<Real>(kChoiVols.begin(), kChoiVols.begin() + n))
        .m("rho", choiRho(n))
        .a("weights", weights)
        .s("payoff_kind", payoffKind)
        .s("option_type", typeName(type))
        .n("strike", strike)
        .n("lambda", lambda)
        .i("max_nr_integration_steps", maxSteps)
        .b("calc_fwd_delta", calcFwdDelta)
        .b("control_variate", controlVariate);
    return o;
}

// maxNrIntegrationSteps defaults to numeric_limits<Size>::max(); JSON cannot
// carry that faithfully, so "unbounded" is emitted as -1 and the Python side
// maps it back to the default.
const long long kUnbounded = -1;

void emitChoiCase(const std::string& name, Size n, const std::vector<Real>& weights,
                  Option::Type type, Real strike, Real lambda, long long maxSteps,
                  bool calcFwdDelta, bool controlVariate,
                  const char* payoffKind = "average") {
    const Handle<YieldTermStructure> rTS = flatCurve(kChoiToday, kChoiRate);
    const std::vector<Process> processes = choiProcesses(n, rTS);

    const ext::shared_ptr<PlainVanillaPayoff> base =
        ext::make_shared<PlainVanillaPayoff>(type, strike);
    const ext::shared_ptr<BasketPayoff> payoff =
        (std::string(payoffKind) == "spread")
            ? ext::static_pointer_cast<BasketPayoff>(
                  ext::make_shared<SpreadBasketPayoff>(base))
            : ext::static_pointer_cast<BasketPayoff>(
                  ext::make_shared<AverageBasketPayoff>(
                      base, Array(weights.begin(), weights.end())));

    BasketOption option(payoff, ext::make_shared<EuropeanExercise>(kChoiMaturity));
    option.setPricingEngine(ext::make_shared<ChoiBasketEngine>(
        processes, choiRho(n), lambda,
        maxSteps < 0 ? std::numeric_limits<Size>::max() : Size(maxSteps),
        calcFwdDelta, controlVariate));

    const Real npv = option.NPV();
    const auto& extra = option.additionalResults();

    Obj ex;
    ex.n("npv", npv)
        .i("additional_results_count", static_cast<long long>(extra.size()));
    if (!extra.empty()) {
        std::vector<Real> deltas;
        for (Size k = 0; k < n; ++k)
            deltas.push_back(
                ext::any_cast<Real>(extra.at("forwardDelta " + std::to_string(k))));
        ex.a("forward_deltas", deltas);
    }
    addCase(name, choiInputs(n, weights, type, strike, lambda, maxSteps,
                             calcFwdDelta, controlVariate, payoffKind),
            ex);
}

void emitChoiBasketEngine() {
    Settings::instance().evaluationDate() = kChoiToday;

    const std::vector<Real> w4 = {1.0, -2.0, -1.0, 4.0};

    // The golden example: lambda 7, 10000 steps, deltas AND control variate on.
    emitChoiCase("choi_golden_put", 4, w4, Option::Put, 20.0, 7.0, 10000, true, true);
    emitChoiCase("choi_golden_call", 4, w4, Option::Call, 20.0, 7.0, 10000, true,
                 true);

    // The four (calcfwdDelta, controlVariate) combinations. calcFwdDelta_ is
    // (calcfwdDelta || controlVariate), so (false, true) still produces deltas
    // AND applies the control-variate correction to the value.
    emitChoiCase("choi_knobs_ff", 4, w4, Option::Call, 20.0, 7.0, 10000, false,
                 false);
    emitChoiCase("choi_knobs_tf", 4, w4, Option::Call, 20.0, 7.0, 10000, true, false);
    emitChoiCase("choi_knobs_ft", 4, w4, Option::Call, 20.0, 7.0, 10000, false, true);
    emitChoiCase("choi_knobs_tt", 4, w4, Option::Call, 20.0, 7.0, 10000, true, true);

    // lambda sweep at unbounded maxNrIntegrationSteps: lambda alone sets the
    // per-dimension quadrature orders, so every value must move the answer.
    for (Real lambda : {3.0, 5.0, 7.0, 10.0, 15.0}) {
        std::ostringstream nm;
        nm << "choi_lambda" << std::fixed << std::setprecision(0) << lambda;
        emitChoiCase(nm.str(), 4, w4, Option::Call, 20.0, lambda, kUnbounded, false,
                     false);
    }

    // maxNrIntegrationSteps sweep at lambda 20: the do/while rescales lambda by
    // 0.9 until the product of the orders fits, so small caps change the price.
    for (long long steps : {50LL, 200LL, 1000LL, 100000LL}) {
        std::ostringstream nm;
        nm << "choi_maxsteps" << steps;
        emitChoiCase(nm.str(), 4, w4, Option::Call, 20.0, 20.0, steps, false, false);
    }

    // Smaller baskets, and an all-positive-weight basket (no sign flip).
    emitChoiCase("choi_2asset_call", 2, {1.0, -1.0}, Option::Call, 20.0, 15.0,
                 kUnbounded, false, false);
    emitChoiCase("choi_2asset_put", 2, {1.0, -1.0}, Option::Put, 20.0, 15.0,
                 kUnbounded, false, false);
    emitChoiCase("choi_3asset_call", 3, {1.0, 1.0, 1.0}, Option::Call, 200.0, 15.0,
                 kUnbounded, false, false);
    emitChoiCase("choi_3asset_positive_weights_put", 3, {0.5, 0.25, 1.0}, Option::Put,
                 150.0, 15.0, kUnbounded, false, false);
    emitChoiCase("choi_4asset_positive_weights_call", 4, {1.0, 1.0, 1.0, 1.0},
                 Option::Call, 250.0, 10.0, kUnbounded, true, false);

    // A SpreadBasketPayoff is silently rewritten as AverageBasketPayoff{1, -1}.
    // Priced with the same two legs, it must equal the explicit {1, -1} average
    // basket above at the same strike -- pinned separately so the rewrite is
    // observable.
    emitChoiCase("choi_spread_payoff_call", 2, {1.0, -1.0}, Option::Call, 20.0, 15.0,
                 kUnbounded, false, false, "spread");
    emitChoiCase("choi_spread_payoff_put", 2, {1.0, -1.0}, Option::Put, 20.0, 15.0,
                 kUnbounded, false, false, "spread");

    // Guards.
    {
        const Handle<YieldTermStructure> rTS = flatCurve(kChoiToday, kChoiRate);
        const std::vector<Process> p4 = choiProcesses(4, rTS);

        struct G {
            const char* name;
            const char* why;
            std::function<void()> f;
        };
        const std::vector<G> guards = {
            {"choi_ctor_rejects_empty_processes", "No Black-Scholes process is given.",
             [] {
                 ChoiBasketEngine(std::vector<Process>(), Matrix(0, 0));
             }},
            {"choi_ctor_rejects_rho_size_mismatch",
             "process and correlation matrix must have the same size.",
             [&] { ChoiBasketEngine(p4, choiRho(3)); }},
            {"choi_ctor_rejects_zero_lambda", "lambda must be positive",
             [&] { ChoiBasketEngine(p4, choiRho(4), 0.0); }},
            {"choi_ctor_rejects_negative_lambda", "lambda must be positive",
             [&] { ChoiBasketEngine(p4, choiRho(4), -1.0); }},
            {"choi_rejects_single_asset", "wrong number of weights arguments in payoff",
             [&] {
                 const std::vector<Process> p1 = choiProcesses(1, rTS);
                 BasketOption o(ext::make_shared<AverageBasketPayoff>(
                                    ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                         100.0),
                                    Array(1, 1.0)),
                                ext::make_shared<EuropeanExercise>(kChoiMaturity));
                 o.setPricingEngine(
                     ext::make_shared<ChoiBasketEngine>(p1, choiRho(1)));
                 o.NPV();
             }},
            {"choi_rejects_wrong_weight_count",
             "wrong number of weights arguments in payoff",
             [&] {
                 BasketOption o(ext::make_shared<AverageBasketPayoff>(
                                    ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                         20.0),
                                    Array(3, 1.0)),
                                ext::make_shared<EuropeanExercise>(kChoiMaturity));
                 o.setPricingEngine(
                     ext::make_shared<ChoiBasketEngine>(p4, choiRho(4)));
                 o.NPV();
             }},
            {"choi_rejects_min_basket_payoff",
             "average or spread basket payoff expected",
             [&] {
                 BasketOption o(ext::make_shared<MinBasketPayoff>(
                                    ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                         20.0)),
                                ext::make_shared<EuropeanExercise>(kChoiMaturity));
                 o.setPricingEngine(
                     ext::make_shared<ChoiBasketEngine>(p4, choiRho(4)));
                 o.NPV();
             }},
            {"choi_rejects_american_exercise", "not an European exercise",
             [&] {
                 BasketOption o(ext::make_shared<AverageBasketPayoff>(
                                    ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                         20.0),
                                    Array({1.0, -2.0, -1.0, 4.0})),
                                ext::make_shared<AmericanExercise>(kChoiToday,
                                                                   kChoiMaturity));
                 o.setPricingEngine(
                     ext::make_shared<ChoiBasketEngine>(p4, choiRho(4)));
                 o.NPV();
             }},
            {"choi_rejects_unfittable_max_integration_steps",
             "can not rescale lambda to fit max integration order",
             [&] {
                 BasketOption o(ext::make_shared<AverageBasketPayoff>(
                                    ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                         20.0),
                                    Array({1.0, -2.0, -1.0, 4.0})),
                                ext::make_shared<EuropeanExercise>(kChoiMaturity));
                 // three integration dimensions => the product is at least 1,
                 // but maxNrIntegrationSteps 0 can never be met.
                 o.setPricingEngine(ext::make_shared<ChoiBasketEngine>(
                     p4, choiRho(4), 10.0, Size(0)));
                 o.NPV();
             }},
        };
        for (const G& g : guards) {
            Obj in;
            in.s("market", "choi").s("why", g.why);
            Obj ex;
            ex.b("throws", throwsDuring(g.f));
            addCase(g.name, in, ex);
        }
    }
}

// ===========================================================================
// Section 7 -- DengLiZhouBasketEngine
// ===========================================================================
// Market "dlz": the upstream testDengLiZhouVsPDE market.
const Date kDlzToday(25, March, 2024);
const Date kDlzMaturity = Date(25, March, 2024) + Period(6, Months);
const Rate kDlzRate = 0.05;

const std::vector<Real> kDlzSpots = {50.0, 11.0, 55.0, 200.0};
const std::vector<Volatility> kDlzVols = {0.2, 0.6, 0.4, 0.3};
const std::vector<Rate> kDlzQ = {0.075, 0.05, 0.08, 0.04};

Matrix dlzRho(Size n) {
    Matrix m(n, n);
    for (Size r = 0; r < n; ++r)
        for (Size c = r; c < n; ++c)
            m[r][c] = m[c][r] =
                std::exp(-0.5 * std::abs(Real(r) - Real(c)) -
                         ((r != c) ? 0.02 * Real(r + c) : Real(0.0)));
    return m;
}

std::vector<Process> dlzProcesses(Size n, const Handle<YieldTermStructure>& rTS) {
    std::vector<Process> p;
    for (Size k = 0; k < n; ++k)
        p.push_back(merton(kDlzToday, kDlzSpots[k], kDlzQ[k], kDlzVols[k], rTS));
    return p;
}

void emitDlzCase(const std::string& name, Size n, const std::vector<Real>& weights,
                 Option::Type type, Real strike, const char* payoffKind = "average") {
    const Handle<YieldTermStructure> rTS = flatCurve(kDlzToday, kDlzRate);
    const std::vector<Process> processes = dlzProcesses(n, rTS);

    const ext::shared_ptr<PlainVanillaPayoff> base =
        ext::make_shared<PlainVanillaPayoff>(type, strike);
    const ext::shared_ptr<BasketPayoff> payoff =
        (std::string(payoffKind) == "spread")
            ? ext::static_pointer_cast<BasketPayoff>(
                  ext::make_shared<SpreadBasketPayoff>(base))
            : ext::static_pointer_cast<BasketPayoff>(
                  ext::make_shared<AverageBasketPayoff>(
                      base, Array(weights.begin(), weights.end())));

    BasketOption option(payoff, ext::make_shared<EuropeanExercise>(kDlzMaturity));
    option.setPricingEngine(
        ext::make_shared<DengLiZhouBasketEngine>(processes, dlzRho(n)));

    Obj in;
    in.s("market", "dlz")
        .s("today", isoDate(kDlzToday))
        .s("maturity", isoDate(kDlzMaturity))
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", kDlzRate)
        .a("spots", std::vector<Real>(kDlzSpots.begin(), kDlzSpots.begin() + n))
        .a("dividend_yields", std::vector<Real>(kDlzQ.begin(), kDlzQ.begin() + n))
        .a("volatilities", std::vector<Real>(kDlzVols.begin(), kDlzVols.begin() + n))
        .m("rho", dlzRho(n))
        .a("weights", weights)
        .s("payoff_kind", payoffKind)
        .s("option_type", typeName(type))
        .n("strike", strike);
    Obj ex;
    ex.n("npv", option.NPV())
        .i("additional_results_count",
           static_cast<long long>(option.additionalResults().size()));
    addCase(name, in, ex);
}

void emitDengLiZhouBasketEngine() {
    Settings::instance().evaluationDate() = kDlzToday;

    // Upstream basket: three negative weights, one positive => M == 1, which
    // takes the `else` branch that never builds the Lo-2013 lognormal proxy.
    emitDlzCase("dlz_upstream_put", 4, {-1.0, -5.0, -2.0, 1.0}, Option::Put, 5.0);
    emitDlzCase("dlz_upstream_call", 4, {-1.0, -5.0, -2.0, 1.0}, Option::Call, 5.0);

    // M > 1: two positive weights, so the engine collapses them onto one
    // lognormal proxy (S0, dq_S0, v_s) via the WKB approximation and rebuilds
    // the correlation row. This is a completely different code path.
    emitDlzCase("dlz_m2_call", 4, {1.0, 2.0, -1.0, -1.0}, Option::Call, 5.0);
    emitDlzCase("dlz_m2_put", 4, {1.0, 2.0, -1.0, -1.0}, Option::Put, 5.0);
    emitDlzCase("dlz_m3_call", 4, {1.0, 2.0, 0.5, -1.0}, Option::Call, 5.0);

    // Genuine 2-asset spread, weights +1 / -1, both payoff spellings.
    emitDlzCase("dlz_spread_avg_call", 2, {1.0, -1.0}, Option::Call, 5.0);
    emitDlzCase("dlz_spread_avg_put", 2, {1.0, -1.0}, Option::Put, 5.0);
    emitDlzCase("dlz_spread_payoff_call", 2, {1.0, -1.0}, Option::Call, 5.0, "spread");
    emitDlzCase("dlz_spread_payoff_put", 2, {1.0, -1.0}, Option::Put, 5.0, "spread");

    // Strike sweep, including the at-the-money-ish and deep-OTM ends.
    for (Real k : {0.0, 1.0, 20.0, 60.0})
        for (Option::Type t : {Option::Call, Option::Put}) {
            std::ostringstream nm;
            nm << "dlz_upstream_" << (t == Option::Call ? "call" : "put") << "_k"
               << std::fixed << std::setprecision(0) << k;
            emitDlzCase(nm.str(), 4, {-1.0, -5.0, -2.0, 1.0}, t, k);
        }

    // NEGATIVE strike: the engine appends a synthetic asset
    // (1.0, n_, -K, dr0, 0.0) -- weight 1, spot -K, dividend discount dr0, zero
    // variance -- with zero correlation to everything, enlarges rho to
    // (n+1)x(n+1), and then prices with strike max(0, K) == 0.
    for (Real k : {-1.0, -10.0})
        for (Option::Type t : {Option::Call, Option::Put}) {
            std::ostringstream nm;
            nm << "dlz_negative_strike_" << (t == Option::Call ? "call" : "put")
               << "_km" << std::fixed << std::setprecision(0) << -k;
            emitDlzCase(nm.str(), 4, {-1.0, -5.0, -2.0, 1.0}, t, k);
        }
    emitDlzCase("dlz_negative_strike_spread_call", 2, {1.0, -1.0}, Option::Call, -5.0);

    // 3-asset mixed-sign basket.
    emitDlzCase("dlz_3asset_call", 3, {2.0, -1.0, -0.5}, Option::Call, 10.0);
    emitDlzCase("dlz_3asset_put", 3, {2.0, -1.0, -0.5}, Option::Put, 10.0);

    // Guards.
    {
        const Handle<YieldTermStructure> rTS = flatCurve(kDlzToday, kDlzRate);
        const std::vector<Process> p4 = dlzProcesses(4, rTS);

        auto priceWeights = [&](const std::vector<Real>& w, Real strike) {
            BasketOption o(ext::make_shared<AverageBasketPayoff>(
                               ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                    strike),
                               Array(w.begin(), w.end())),
                           ext::make_shared<EuropeanExercise>(kDlzMaturity));
            o.setPricingEngine(
                ext::make_shared<DengLiZhouBasketEngine>(p4, dlzRho(4)));
            o.NPV();
        };

        struct G {
            const char* name;
            const char* why;
            std::function<void()> f;
        };
        const std::vector<G> guards = {
            {"dlz_ctor_rejects_empty_processes", "No Black-Scholes process is given.",
             [] { DengLiZhouBasketEngine(std::vector<Process>(), Matrix(0, 0)); }},
            {"dlz_ctor_rejects_rho_size_mismatch",
             "process and correlation matrix must have the same size.",
             [&] { DengLiZhouBasketEngine(p4, dlzRho(3)); }},
            {"dlz_rejects_all_positive_weights",
             "at least one negative asset weight must be given",
             [&] { priceWeights({1.0, 2.0, 3.0, 4.0}, 5.0); }},
            {"dlz_rejects_all_negative_weights",
             "at least one positive asset weight must be given",
             [&] { priceWeights({-1.0, -2.0, -3.0, -4.0}, 5.0); }},
            // A weight of exactly 0 counts as NON-positive (lower_bound uses
            // `> 0`), so {1, 0, 0, 0} still has M == 1 and N == 3 and prices.
            {"dlz_rejects_wrong_weight_count",
             "wrong number of weights arguments in payoff",
             [&] { priceWeights({1.0, -1.0}, 5.0); }},
            {"dlz_rejects_min_basket_payoff",
             "average or spread basket payoff expected",
             [&] {
                 BasketOption o(ext::make_shared<MaxBasketPayoff>(
                                    ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                         5.0)),
                                ext::make_shared<EuropeanExercise>(kDlzMaturity));
                 o.setPricingEngine(
                     ext::make_shared<DengLiZhouBasketEngine>(p4, dlzRho(4)));
                 o.NPV();
             }},
            {"dlz_rejects_american_exercise", "not an European exercise",
             [&] {
                 BasketOption o(ext::make_shared<AverageBasketPayoff>(
                                    ext::make_shared<PlainVanillaPayoff>(Option::Call,
                                                                         5.0),
                                    Array({-1.0, -5.0, -2.0, 1.0})),
                                ext::make_shared<AmericanExercise>(kDlzToday,
                                                                   kDlzMaturity));
                 o.setPricingEngine(
                     ext::make_shared<DengLiZhouBasketEngine>(p4, dlzRho(4)));
                 o.NPV();
             }},
        };
        for (const G& g : guards) {
            Obj in;
            in.s("market", "dlz").s("why", g.why);
            Obj ex;
            ex.b("throws", throwsDuring(g.f));
            addCase(g.name, in, ex);
        }

        // Zero weights are NOT positive, so this is a legal M == 1 basket.
        emitDlzCase("dlz_zero_weights_call", 4, {1.0, 0.0, 0.0, 0.0}, Option::Call,
                    5.0);
    }
}

// ===========================================================================
// Sections 8 & 9 -- Monte Carlo basket engines
// ===========================================================================
// Market "mc": three Merton legs sharing one risk-free curve, constant
// correlation. Every case fixes a NON-ZERO seed (MersenneTwisterUniformRng
// treats 0 as "draw a seed from SeedGenerator", which is not reproducible) and
// pins the EXACT mean and error estimate.
const Date kMcToday(1, March, 2025);
const Date kMcMaturity = Date(1, March, 2025) + 365; // t = 1 exactly
const Rate kMcRate = 0.05;

const std::vector<Real> kMcSpots = {100.0, 105.0, 95.0};
const std::vector<Rate> kMcQ = {0.02, 0.03, 0.0};
const std::vector<Volatility> kMcVols = {0.20, 0.25, 0.30};

Matrix mcCorrelation(Size n, Real rho) {
    Matrix m(n, n, rho);
    for (Size k = 0; k < n; ++k)
        m[k][k] = 1.0;
    return m;
}

ext::shared_ptr<StochasticProcessArray> mcProcessArray(Size n, Real rho) {
    const Handle<YieldTermStructure> rTS = flatCurve(kMcToday, kMcRate);
    std::vector<ext::shared_ptr<StochasticProcess1D>> procs;
    for (Size k = 0; k < n; ++k)
        procs.push_back(merton(kMcToday, kMcSpots[k], kMcQ[k], kMcVols[k], rTS));
    return ext::make_shared<StochasticProcessArray>(procs, mcCorrelation(n, rho));
}

Obj mcInputs(Size n, Real rho) {
    Obj o;
    o.s("market", "mc")
        .s("today", isoDate(kMcToday))
        .s("maturity", isoDate(kMcMaturity))
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", kMcRate)
        .a("spots", std::vector<Real>(kMcSpots.begin(), kMcSpots.begin() + n))
        .a("dividend_yields", std::vector<Real>(kMcQ.begin(), kMcQ.begin() + n))
        .a("volatilities", std::vector<Real>(kMcVols.begin(), kMcVols.begin() + n))
        .n("correlation", rho)
        .i("n_assets", static_cast<long long>(n));
    return o;
}

ext::shared_ptr<BasketPayoff> mcPayoff(const std::string& kind, Option::Type type,
                                       Real strike, const std::vector<Real>& weights) {
    const ext::shared_ptr<PlainVanillaPayoff> base =
        ext::make_shared<PlainVanillaPayoff>(type, strike);
    if (kind == "max")
        return ext::make_shared<MaxBasketPayoff>(base);
    if (kind == "min")
        return ext::make_shared<MinBasketPayoff>(base);
    if (kind == "spread")
        return ext::make_shared<SpreadBasketPayoff>(base);
    return ext::make_shared<AverageBasketPayoff>(
        base, Array(weights.begin(), weights.end()));
}

// --- RNG / path diagnostics -------------------------------------------------
void emitMcDiagnostics() {
    Settings::instance().evaluationDate() = kMcToday;

    // Raw PseudoRandom stream: MersenneTwisterUniformRng(seed) mapped through
    // InverseCumulativeNormal, three at a time.
    {
        auto gen = PseudoRandom::make_sequence_generator(3, 42);
        std::vector<Real> flat;
        for (int draw = 0; draw < 3; ++draw) {
            const auto& seq = gen.nextSequence();
            flat.insert(flat.end(), seq.value.begin(), seq.value.end());
        }
        Obj in;
        in.i("dimension", 3).i("seed", 42).i("draws", 3);
        Obj ex;
        ex.a("gaussians", flat);
        addCase("mc_rng_pseudo_random_dim3_seed42", in, ex);
    }
    {
        auto gen = PseudoRandom::make_sequence_generator(6, 7);
        const auto& seq = gen.nextSequence();
        Obj in;
        in.i("dimension", 6).i("seed", 7).i("draws", 1);
        Obj ex;
        ex.a("gaussians", std::vector<Real>(seq.value.begin(), seq.value.end()));
        addCase("mc_rng_pseudo_random_dim6_seed7", in, ex);
    }

    // StochasticProcessArray::evolve with a fixed dw -- this is where the
    // spectral pseudo-square-root of the correlation matrix enters.
    {
        const auto pa = mcProcessArray(3, 0.5);
        const Array x0 = pa->initialValues();
        const Array dw({0.3, -1.2, 0.7});
        const Array evolved = pa->evolve(0.0, x0, 1.0, dw);
        Obj in = mcInputs(3, 0.5);
        in.a("dw", std::vector<Real>({0.3, -1.2, 0.7})).n("dt", 1.0).n("t", 0.0);
        Obj ex;
        ex.a("x0", x0).a("evolved", evolved).m("correlation", pa->correlation());
        addCase("mc_process_array_evolve", in, ex);
    }

    // The first MultiPath the engine's own generator produces: 3 assets,
    // TimeGrid(1.0, 2) => 2 steps => dimension 3*2 = 6.
    {
        const auto pa = mcProcessArray(3, 0.5);
        const TimeGrid grid(1.0, 2);
        auto gen = PseudoRandom::make_sequence_generator(3 * (grid.size() - 1), 42);
        MultiPathGenerator<PseudoRandom::rsg_type> pg(pa, grid, gen, false);
        const MultiPath& path = pg.next().value;

        std::vector<Real> flat;
        for (Size j = 0; j < path.assetNumber(); ++j)
            for (Size i = 0; i < path.pathSize(); ++i)
                flat.push_back(path[j][i]);

        Obj in = mcInputs(3, 0.5);
        in.i("seed", 42).i("time_steps", 2).n("horizon", 1.0);
        Obj ex;
        ex.i("asset_number", static_cast<long long>(path.assetNumber()))
            .i("path_size", static_cast<long long>(path.pathSize()))
            .a("values_row_major", flat);
        addCase("mc_multipath_first_draw", in, ex);
    }
}

// --- MCEuropeanBasketEngine -------------------------------------------------
struct McEuroRow {
    const char* name;
    Size n;
    Real rho;
    const char* payoffKind;
    Option::Type type;
    Real strike;
    std::vector<Real> weights;
    long long steps;         // -1 => unset
    long long stepsPerYear;  // -1 => unset
    long long samples;
    bool antithetic;
    long long seed;
};

void emitMcEuropeanBasketEngine() {
    Settings::instance().evaluationDate() = kMcToday;

    const std::vector<McEuroRow> rows = {
        {"mceb_max_call_2a", 2, 0.5, "max", Option::Call, 100.0, {}, -1, 1, 4096, false,
         42},
        {"mceb_max_put_2a", 2, 0.5, "max", Option::Put, 100.0, {}, -1, 1, 4096, false,
         42},
        {"mceb_min_call_2a", 2, 0.5, "min", Option::Call, 100.0, {}, -1, 1, 4096, false,
         42},
        {"mceb_spread_call_2a", 2, 0.5, "spread", Option::Call, 5.0, {}, -1, 1, 4096,
         false, 42},
        {"mceb_avg_call_3a", 3, 0.5, "average", Option::Call, 100.0, {1.0, 1.0, 1.0},
         -1, 1, 4096, false, 42},
        {"mceb_avg_call_3a_atm", 3, 0.5, "average", Option::Call, 300.0,
         {1.0, 1.0, 1.0}, -1, 1, 4096, false, 42},
        {"mceb_avg_put_3a_atm", 3, 0.5, "average", Option::Put, 300.0,
         {1.0, 1.0, 1.0}, -1, 1, 4096, false, 42},
        {"mceb_avg_call_3a_weighted", 3, 0.5, "average", Option::Call, 50.0,
         {0.5, -0.25, 0.75}, -1, 1, 4096, false, 42},
        {"mceb_max_call_3a", 3, 0.5, "max", Option::Call, 100.0, {}, -1, 1, 4096, false,
         42},
        // knobs: fixed step count, several steps, antithetic, other seeds,
        // other sample counts, zero and negative correlation
        {"mceb_max_call_2a_steps1", 2, 0.5, "max", Option::Call, 100.0, {}, 1, -1, 4096,
         false, 42},
        {"mceb_max_call_2a_steps4", 2, 0.5, "max", Option::Call, 100.0, {}, 4, -1, 4096,
         false, 42},
        {"mceb_max_call_2a_antithetic", 2, 0.5, "max", Option::Call, 100.0, {}, -1, 1,
         4096, true, 42},
        {"mceb_max_call_2a_seed7", 2, 0.5, "max", Option::Call, 100.0, {}, -1, 1, 4096,
         false, 7},
        {"mceb_max_call_2a_samples1023", 2, 0.5, "max", Option::Call, 100.0, {}, -1, 1,
         1023, false, 42},
        {"mceb_max_call_2a_rho0", 2, 0.0, "max", Option::Call, 100.0, {}, -1, 1, 4096,
         false, 42},
        {"mceb_max_call_2a_rho_m05", 2, -0.5, "max", Option::Call, 100.0, {}, -1, 1,
         4096, false, 42},
    };

    for (const McEuroRow& r : rows) {
        const auto pa = mcProcessArray(r.n, r.rho);
        MakeMCEuropeanBasketEngine<PseudoRandom, Statistics> maker(pa);
        if (r.steps >= 0)
            maker.withSteps(Size(r.steps));
        if (r.stepsPerYear >= 0)
            maker.withStepsPerYear(Size(r.stepsPerYear));
        maker.withSamples(Size(r.samples)).withSeed(BigNatural(r.seed));
        if (r.antithetic)
            maker.withAntitheticVariate();

        BasketOption option(mcPayoff(r.payoffKind, r.type, r.strike, r.weights),
                            ext::make_shared<EuropeanExercise>(kMcMaturity));
        option.setPricingEngine(maker);

        Obj in = mcInputs(r.n, r.rho);
        in.s("payoff_kind", r.payoffKind)
            .s("option_type", typeName(r.type))
            .n("strike", r.strike)
            .a("weights", r.weights)
            .i("steps", r.steps)
            .i("steps_per_year", r.stepsPerYear)
            .i("samples", r.samples)
            .b("antithetic", r.antithetic)
            .i("seed", r.seed);
        Obj ex;
        ex.n("npv", option.NPV()).n("error_estimate", option.errorEstimate());
        addCase(r.name, in, ex);
    }

    // Guards.
    {
        const auto pa = mcProcessArray(2, 0.5);
        struct G {
            const char* name;
            const char* why;
            std::function<void()> f;
        };
        const std::vector<G> guards = {
            {"mceb_rejects_brownian_bridge", "Brownian bridge not supported",
             [&] {
                 BasketOption o(mcPayoff("max", Option::Call, 100.0, {}),
                                ext::make_shared<EuropeanExercise>(kMcMaturity));
                 o.setPricingEngine(MakeMCEuropeanBasketEngine<PseudoRandom, Statistics>(pa)
                                        .withStepsPerYear(1)
                                        .withSamples(1023)
                                        .withBrownianBridge()
                                        .withSeed(42));
                 o.NPV();
             }},
            {"mceb_rejects_no_steps", "number of steps not given",
             [&] {
                 ext::shared_ptr<PricingEngine> e =
                     MakeMCEuropeanBasketEngine<PseudoRandom, Statistics>(pa)
                         .withSamples(1023)
                         .withSeed(42);
             }},
            {"mceb_rejects_overspecified_steps", "number of steps overspecified",
             [&] {
                 ext::shared_ptr<PricingEngine> e =
                     MakeMCEuropeanBasketEngine<PseudoRandom, Statistics>(pa)
                         .withSteps(2)
                         .withStepsPerYear(1)
                         .withSamples(1023)
                         .withSeed(42);
             }},
            {"mceb_rejects_samples_and_tolerance", "tolerance already set",
             [&] {
                 MakeMCEuropeanBasketEngine<PseudoRandom, Statistics>(pa)
                     .withStepsPerYear(1)
                     .withAbsoluteTolerance(0.02)
                     .withSamples(1023);
             }},
            {"mceb_rejects_tolerance_and_samples", "number of samples already set",
             [&] {
                 MakeMCEuropeanBasketEngine<PseudoRandom, Statistics>(pa)
                     .withStepsPerYear(1)
                     .withSamples(1023)
                     .withAbsoluteTolerance(0.02);
             }},
            {"mceb_rejects_non_basket_payoff", "non-basket payoff given",
             [&] {
                 BasketOption o(mcPayoff("max", Option::Call, 100.0, {}),
                                ext::make_shared<EuropeanExercise>(kMcMaturity));
                 // A BasketOption always carries a BasketPayoff, so the guard is
                 // unreachable through the public API; pinned as "does not throw"
                 // for the happy path instead.
                 o.setPricingEngine(MakeMCEuropeanBasketEngine<PseudoRandom, Statistics>(pa)
                                        .withStepsPerYear(1)
                                        .withSamples(1023)
                                        .withSeed(42));
                 o.NPV();
             }},
        };
        for (const G& g : guards) {
            Obj in;
            in.s("market", "mc").s("why", g.why);
            Obj ex;
            ex.b("throws", throwsDuring(g.f));
            addCase(g.name, in, ex);
        }
    }

    // withAbsoluteTolerance: the adaptive sampling loop, pinned exactly.
    {
        const auto pa = mcProcessArray(2, 0.5);
        BasketOption o(mcPayoff("max", Option::Call, 100.0, {}),
                       ext::make_shared<EuropeanExercise>(kMcMaturity));
        o.setPricingEngine(MakeMCEuropeanBasketEngine<PseudoRandom, Statistics>(pa)
                               .withStepsPerYear(1)
                               .withAbsoluteTolerance(0.5)
                               .withMaxSamples(1 << 20)
                               .withSeed(42));
        Obj in = mcInputs(2, 0.5);
        in.s("payoff_kind", "max")
            .s("option_type", "Call")
            .n("strike", 100.0)
            .i("steps_per_year", 1)
            .n("absolute_tolerance", 0.5)
            .i("max_samples", 1 << 20)
            .i("seed", 42);
        Obj ex;
        ex.n("npv", o.NPV()).n("error_estimate", o.errorEstimate());
        addCase("mceb_absolute_tolerance", in, ex);
    }
}

// --- MCAmericanBasketEngine -------------------------------------------------
struct McAmRow {
    const char* name;
    Size n;
    Real rho;
    const char* payoffKind;
    Option::Type type;
    Real strike;
    std::vector<Real> weights;
    long long steps;
    long long samples;
    long long calibrationSamples;
    long long polynomialOrder;
    const char* polynomialType;
    bool antithetic;
    long long seed;
};

LsmBasisSystem::PolynomialType polyTypeOf(const std::string& s) {
    if (s == "Monomial")
        return LsmBasisSystem::Monomial;
    if (s == "Laguerre")
        return LsmBasisSystem::Laguerre;
    if (s == "Hermite")
        return LsmBasisSystem::Hermite;
    if (s == "Hyperbolic")
        return LsmBasisSystem::Hyperbolic;
    if (s == "Chebyshev2nd")
        return LsmBasisSystem::Chebyshev2nd;
    QL_FAIL("unknown polynomial type");
}

void emitMcAmericanBasketEngine() {
    Settings::instance().evaluationDate() = kMcToday;

    const std::vector<McAmRow> rows = {
        {"mcab_max_put_2a", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048, 512, 2,
         "Monomial", false, 1},
        {"mcab_max_call_2a", 2, 0.5, "max", Option::Call, 100.0, {}, 8, 2048, 512, 2,
         "Monomial", false, 1},
        {"mcab_min_put_2a", 2, 0.5, "min", Option::Put, 100.0, {}, 8, 2048, 512, 2,
         "Monomial", false, 1},
        {"mcab_avg_put_3a", 3, 0.5, "average", Option::Put, 300.0, {1.0, 1.0, 1.0}, 6,
         2048, 512, 2, "Monomial", false, 1},
        {"mcab_max_put_2a_order3", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048, 512,
         3, "Monomial", false, 1},
        {"mcab_max_put_2a_order1", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048, 512,
         1, "Monomial", false, 1},
        {"mcab_max_put_2a_laguerre", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048,
         512, 2, "Laguerre", false, 1},
        {"mcab_max_put_2a_hermite", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048, 512,
         2, "Hermite", false, 1},
        {"mcab_max_put_2a_chebyshev2nd", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048,
         512, 2, "Chebyshev2nd", false, 1},
        {"mcab_max_put_2a_antithetic", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048,
         512, 2, "Monomial", true, 1},
        {"mcab_max_put_2a_seed42", 2, 0.5, "max", Option::Put, 100.0, {}, 8, 2048, 512,
         2, "Monomial", false, 42},
        {"mcab_max_put_2a_steps16", 2, 0.5, "max", Option::Put, 100.0, {}, 16, 2048,
         512, 2, "Monomial", false, 1},
        {"mcab_spread_put_2a", 2, 0.5, "spread", Option::Put, 5.0, {}, 8, 2048, 512, 2,
         "Monomial", false, 1},
    };

    for (const McAmRow& r : rows) {
        const auto pa = mcProcessArray(r.n, r.rho);
        MakeMCAmericanBasketEngine<PseudoRandom> maker(pa);
        maker.withSteps(Size(r.steps))
            .withSamples(Size(r.samples))
            .withCalibrationSamples(Size(r.calibrationSamples))
            .withPolynomialOrder(Size(r.polynomialOrder))
            .withBasisSystem(polyTypeOf(r.polynomialType))
            .withSeed(BigNatural(r.seed));
        if (r.antithetic)
            maker.withAntitheticVariate();

        BasketOption option(mcPayoff(r.payoffKind, r.type, r.strike, r.weights),
                            ext::make_shared<AmericanExercise>(kMcToday, kMcMaturity));
        option.setPricingEngine(maker);

        Obj in = mcInputs(r.n, r.rho);
        in.s("payoff_kind", r.payoffKind)
            .s("option_type", typeName(r.type))
            .n("strike", r.strike)
            .a("weights", r.weights)
            .i("steps", r.steps)
            .i("samples", r.samples)
            .i("calibration_samples", r.calibrationSamples)
            .i("polynomial_order", r.polynomialOrder)
            .s("polynomial_type", r.polynomialType)
            .b("antithetic", r.antithetic)
            .i("seed", r.seed);
        Obj ex;
        ex.n("npv", option.NPV()).n("error_estimate", option.errorEstimate());
        const auto& extra = option.additionalResults();
        ex.i("additional_results_count", static_cast<long long>(extra.size()));
        if (extra.find("exerciseProbability") != extra.end())
            ex.n("exercise_probability",
                 ext::any_cast<Real>(extra.at("exerciseProbability")));
        addCase(r.name, in, ex);
    }

    // AmericanBasketPathPricer's own surface: the basis system size and the
    // scaled state/payoff, taken directly rather than through the engine.
    {
        const ext::shared_ptr<Payoff> payoff =
            ext::make_shared<MaxBasketPayoff>(
                ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0));
        for (long long order : {1LL, 2LL, 3LL}) {
            const AmericanBasketPathPricer pricer(2, payoff, Size(order),
                                                  LsmBasisSystem::Monomial);
            const Array state({1.10, 0.95}); // already scaled by 1/strike
            std::vector<Real> basisValues;
            for (const auto& f : pricer.basisSystem())
                basisValues.push_back(f(state));

            Obj in;
            in.s("market", "mc")
                .i("asset_number", 2)
                .s("payoff_kind", "max")
                .s("option_type", "Put")
                .n("strike", 100.0)
                .i("polynomial_order", order)
                .s("polynomial_type", "Monomial")
                .a("state", std::vector<Real>({1.10, 0.95}));
            Obj ex;
            ex.i("basis_size", static_cast<long long>(pricer.basisSystem().size()))
                .a("basis_values", basisValues);
            std::ostringstream nm;
            nm << "mcab_path_pricer_basis_order" << order;
            addCase(nm.str(), in, ex);
        }

        // The pricer rejects polynomial types outside the allowed set.
        Obj in;
        in.s("polynomial_type", "Legendre").s("why", "insufficient polynomial type");
        Obj ex;
        ex.b("throws", throwsDuring([&] {
               AmericanBasketPathPricer(2, payoff, 2, LsmBasisSystem::Legendre);
           }));
        addCase("mcab_path_pricer_rejects_legendre", in, ex);

        // ... and a payoff that is not a BasketPayoff at all.
        Obj in2;
        in2.s("payoff_kind", "plain").s("why", "payoff not a basket payoff");
        Obj ex2;
        ex2.b("throws", throwsDuring([] {
                AmericanBasketPathPricer(
                    2, ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0), 2,
                    LsmBasisSystem::Monomial);
            }));
        addCase("mcab_path_pricer_rejects_non_basket_payoff", in2, ex2);
    }

    // Guards on the engine itself.
    {
        const auto pa = mcProcessArray(2, 0.5);
        Obj in;
        in.s("market", "mc").s("why", "wrong exercise given");
        Obj ex;
        ex.b("throws", throwsDuring([&] {
               BasketOption o(mcPayoff("max", Option::Put, 100.0, {}),
                              ext::make_shared<EuropeanExercise>(kMcMaturity));
               o.setPricingEngine(MakeMCAmericanBasketEngine<PseudoRandom>(pa)
                                      .withSteps(4)
                                      .withSamples(1023)
                                      .withCalibrationSamples(256)
                                      .withSeed(1));
               o.NPV();
           }));
        addCase("mcab_rejects_european_exercise", in, ex);
    }
}

// ===========================================================================
// Section 10 -- ChoiAsianEngine
// ===========================================================================
// Market "ca": one Merton leg. The engine maps the discrete arithmetic average
// onto a ChoiBasketEngine basket whose "assets" are the individual fixings.
const Date kCaToday(1, March, 2025);
const Rate kCaRate = 0.05;
const Rate kCaQ = 0.02;
const Real kCaSpot = 100.0;
const Volatility kCaVol = 0.30;

Process caProcess() {
    const Handle<YieldTermStructure> rTS = flatCurve(kCaToday, kCaRate);
    return merton(kCaToday, kCaSpot, kCaQ, kCaVol, rTS);
}

// n monthly fixings, the first `startMonth` months out.
std::vector<Date> caFixings(int startMonth, int n) {
    std::vector<Date> d;
    for (int k = 0; k < n; ++k)
        d.push_back(kCaToday + Period(startMonth + k, Months));
    return d;
}

std::vector<std::string> isoDates(const std::vector<Date>& v) {
    std::vector<std::string> out;
    for (const Date& d : v)
        out.push_back(isoDate(d));
    return out;
}

void emitCaCase(const std::string& name, const std::vector<Date>& fixings,
                const Date& exerciseDate, Option::Type type, Real strike,
                Real runningAccumulator, Size pastFixings, Real lambda,
                long long maxSteps) {
    const Process p = caProcess();
    DiscreteAveragingAsianOption option(
        Average::Arithmetic, runningAccumulator, pastFixings, fixings,
        ext::make_shared<PlainVanillaPayoff>(type, strike),
        ext::make_shared<EuropeanExercise>(exerciseDate));
    option.setPricingEngine(ext::make_shared<ChoiAsianEngine>(
        p, lambda, maxSteps < 0 ? Size(2 << 21) : Size(maxSteps)));

    Obj in;
    in.s("market", "ca")
        .s("today", isoDate(kCaToday))
        .s("day_counter", "Actual365Fixed")
        .n("risk_free_rate", kCaRate)
        .n("dividend_yield", kCaQ)
        .n("spot", kCaSpot)
        .n("volatility", kCaVol)
        .as("fixing_dates", isoDates(fixings))
        .s("exercise_date", isoDate(exerciseDate))
        .s("average_type", "Arithmetic")
        .s("option_type", typeName(type))
        .n("strike", strike)
        .n("running_accumulator", runningAccumulator)
        .i("past_fixings", static_cast<long long>(pastFixings))
        .n("lambda", lambda)
        .i("max_nr_integration_steps", maxSteps);
    Obj ex;
    ex.n("npv", option.NPV())
        .i("additional_results_count",
           static_cast<long long>(option.additionalResults().size()));
    addCase(name, in, ex);
}

void emitChoiAsianEngine() {
    Settings::instance().evaluationDate() = kCaToday;

    const long long kFast = 2 << 12; // 8192, the knob the upstream tests use

    // 6 monthly future fixings, no past fixings. Exercise on the last fixing.
    {
        const std::vector<Date> f = caFixings(1, 6);
        emitCaCase("ca_6fix_call_atm", f, f.back(), Option::Call, 100.0, 0.0, 0, 10.0,
                   kFast);
        emitCaCase("ca_6fix_put_atm", f, f.back(), Option::Put, 100.0, 0.0, 0, 10.0,
                   kFast);
        emitCaCase("ca_6fix_call_itm", f, f.back(), Option::Call, 80.0, 0.0, 0, 10.0,
                   kFast);
        emitCaCase("ca_6fix_call_otm", f, f.back(), Option::Call, 130.0, 0.0, 0, 10.0,
                   kFast);
        emitCaCase("ca_6fix_put_itm", f, f.back(), Option::Put, 130.0, 0.0, 0, 10.0,
                   kFast);
        // Exercise strictly AFTER the last fixing: the payoff is discounted from
        // the exercise date, not from the last fixing date.
        emitCaCase("ca_6fix_call_late_exercise", f, kCaToday + Period(12, Months),
                   Option::Call, 100.0, 0.0, 0, 10.0, kFast);
        // lambda / maxNrIntegrationSteps knobs.
        emitCaCase("ca_6fix_call_lambda5", f, f.back(), Option::Call, 100.0, 0.0, 0,
                   5.0, kFast);
        emitCaCase("ca_6fix_call_lambda20", f, f.back(), Option::Call, 100.0, 0.0, 0,
                   20.0, kFast);
        emitCaCase("ca_6fix_call_maxsteps256", f, f.back(), Option::Call, 100.0, 0.0, 0,
                   20.0, 256);
    }

    // 3 fixings at the engine's DEFAULT knobs (lambda 15, 2<<21 steps).
    {
        const std::vector<Date> f = caFixings(2, 3);
        emitCaCase("ca_3fix_call_defaults", f, f.back(), Option::Call, 100.0, 0.0, 0,
                   15.0, -1);
        emitCaCase("ca_3fix_put_defaults", f, f.back(), Option::Put, 100.0, 0.0, 0,
                   15.0, -1);
    }

    // PAST FIXINGS. The effective strike is
    //     payoff.strike - runningAccumulator/(pastFixings + futureFixings)
    // -- divided by the TOTAL count -- and the basket weights are
    // 1/(futureFixings + pastFixings).
    {
        const std::vector<Date> f = caFixings(1, 6);
        emitCaCase("ca_past6_future6_call", f, f.back(), Option::Call, 100.0, 6 * 102.0,
                   6, 10.0, kFast);
        emitCaCase("ca_past6_future6_put", f, f.back(), Option::Put, 100.0, 6 * 102.0,
                   6, 10.0, kFast);
        emitCaCase("ca_past3_future6_call", f, f.back(), Option::Call, 100.0, 3 * 95.0,
                   3, 10.0, kFast);
        emitCaCase("ca_past9_future6_call", f, f.back(), Option::Call, 120.0, 9 * 98.0,
                   9, 10.0, kFast);
    }

    // A fixing at t == 0 is folded into the past fixings: futureFixings drops by
    // one, pastFixings rises by one and runningAccumulator picks up x0.
    {
        std::vector<Date> f = caFixings(0, 5); // f[0] == today
        emitCaCase("ca_today_fixing_folded_call", f, f.back(), Option::Call, 100.0, 0.0,
                   0, 10.0, kFast);
        emitCaCase("ca_today_fixing_folded_with_past_call", f, f.back(), Option::Call,
                   100.0, 2 * 97.0, 2, 10.0, kFast);
        // Fixing dates are SORTED by the engine, so an out-of-order vector must
        // give the same answer as the sorted one.
        std::vector<Date> shuffled = {f[3], f[0], f[4], f[1], f[2]};
        emitCaCase("ca_today_fixing_folded_call_unsorted", shuffled, f.back(),
                   Option::Call, 100.0, 0.0, 0, 10.0, kFast);
    }

    // Exactly ONE future fixing -> the blackFormula branch, no basket at all.
    {
        const std::vector<Date> f = caFixings(6, 1);
        emitCaCase("ca_1fix_call", f, f.back(), Option::Call, 100.0, 0.0, 0, 10.0,
                   kFast);
        emitCaCase("ca_1fix_put", f, f.back(), Option::Put, 100.0, 0.0, 0, 10.0, kFast);
        emitCaCase("ca_1fix_call_with_past", f, f.back(), Option::Call, 100.0,
                   3 * 101.0, 3, 10.0, kFast);
        emitCaCase("ca_1fix_call_late_exercise", f, kCaToday + Period(12, Months),
                   Option::Call, 100.0, 0.0, 0, 10.0, kFast);
        // Today's single fixing gets folded into the past, leaving ZERO future
        // fixings -> the pure-intrinsic branch.
        const std::vector<Date> today1 = caFixings(0, 1);
        emitCaCase("ca_only_today_fixing_call", today1, kCaToday + Period(6, Months),
                   Option::Call, 90.0, 0.0, 0, 10.0, kFast);
        emitCaCase("ca_only_today_fixing_put", today1, kCaToday + Period(6, Months),
                   Option::Put, 110.0, 0.0, 0, 10.0, kFast);
    }

    // Zero future fixings supplied directly -> intrinsic on runningAccumulator.
    {
        Obj in;
        in.s("market", "ca")
            .s("today", isoDate(kCaToday))
            .n("risk_free_rate", kCaRate)
            .n("dividend_yield", kCaQ)
            .n("spot", kCaSpot)
            .n("volatility", kCaVol)
            .as("fixing_dates", std::vector<std::string>())
            .s("exercise_date", isoDate(kCaToday + Period(6, Months)))
            .s("option_type", "Call")
            .n("strike", 95.0)
            .n("running_accumulator", 4 * 103.0)
            .i("past_fixings", 4)
            .n("lambda", 10.0)
            .i("max_nr_integration_steps", kFast);
        Obj ex;
        // DiscreteAveragingAsianOption::arguments::validate() requires at least
        // one fixing date, so the "futureFixings == 0" branch is only reachable
        // through the today-fixing fold above. Recorded here for the record.
        ex.b("reachable_directly", false);
        addCase("ca_zero_fixings_is_unreachable", in, ex);
    }

    // Guards.
    {
        const Process p = caProcess();
        auto price = [&](Average::Type avg,
                         const ext::shared_ptr<StrikedTypePayoff>& payoff,
                         const ext::shared_ptr<Exercise>& exercise,
                         const std::vector<Date>& fixings, Real acc, Size past) {
            DiscreteAveragingAsianOption o(avg, acc, past, fixings, payoff, exercise);
            o.setPricingEngine(ext::make_shared<ChoiAsianEngine>(p, 10.0, Size(2 << 12)));
            o.NPV();
        };
        const std::vector<Date> f = caFixings(1, 4);

        struct G {
            const char* name;
            const char* why;
            std::function<void()> f;
        };
        const std::vector<G> guards = {
            {"ca_rejects_geometric_average", "must be Average::Type Arithmetic ",
             [&] {
                 price(Average::Geometric,
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       ext::make_shared<EuropeanExercise>(f.back()), f, 1.0, 0);
             }},
            {"ca_rejects_american_exercise", "not a European Option",
             [&] {
                 price(Average::Arithmetic,
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       ext::make_shared<AmericanExercise>(kCaToday, f.back()), f, 0.0,
                       0);
             }},
            {"ca_rejects_non_plain_payoff", "non plain vanilla payoff given",
             [&] {
                 price(Average::Arithmetic,
                       ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0),
                       ext::make_shared<EuropeanExercise>(f.back()), f, 0.0, 0);
             }},
            {"ca_rejects_negative_effective_strike",
             "effective strike should to be positive",
             [&] {
                 // runningAccumulator/(past + future) = 4*300/8 = 150 > strike 100
                 price(Average::Arithmetic,
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       ext::make_shared<EuropeanExercise>(f.back()), f, 4 * 300.0, 4);
             }},
            {"ca_rejects_fixing_after_exercise",
             "last fixing date must be before exercise date",
             [&] {
                 price(Average::Arithmetic,
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       ext::make_shared<EuropeanExercise>(kCaToday + Period(2, Months)),
                       f, 0.0, 0);
             }},
            {"ca_rejects_duplicate_fixing_dates", "two fixing dates are the same",
             [&] {
                 std::vector<Date> dup = {f[0], f[1], f[1], f[2]};
                 price(Average::Arithmetic,
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       ext::make_shared<EuropeanExercise>(f.back()), dup, 0.0, 0);
             }},
        };
        for (const G& g : guards) {
            Obj in;
            in.s("market", "ca").s("why", g.why);
            Obj ex;
            ex.b("throws", throwsDuring(g.f));
            addCase(g.name, in, ex);
        }
    }
}

} // namespace

int main() {
    emitVectorBsmProcessExtractor();
    emitSumExponentialsRootSolver();
    emitSingleFactorBsmBasketEngine();
    emitBjerksundStenslandSpreadEngine();
    emitOperatorSplittingSpreadEngine();
    emitChoiBasketEngine();
    emitDengLiZhouBasketEngine();
    emitMcDiagnostics();
    emitMcEuropeanBasketEngine();
    emitMcAmericanBasketEngine();
    emitChoiAsianEngine();

    emitDocument();
    return 0;
}
