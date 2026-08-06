// migration-harness/cpp/probes/v143_pe_qdfp/probe.cpp
//
// Reference values for the Andersen-Lake(-Offengenden) family of American
// option engines in C++ QuantLib v1.43:
//
//   ql/pricingengines/vanilla/qdplusamericanengine.{hpp,cpp}
//     * detail::QdPutCallParityEngine  -- abstract VanillaOption::engine base.
//                                         Reads (S, K, r, q, vol, T) off the
//                                         process, dispatches a CALL as a PUT
//                                         with (S,K,r,q) -> (K,S,q,r) (the
//                                         McDonald-Schroder parity), and owns
//                                         every degenerate-input early return.
//     * detail::QdPlusAddOnValue       -- the early-exercise add-on integrand
//                                         d/dz of the American premium, in the
//                                         z = sqrt(t) variable.
//     * QdPlusAmericanEngine           -- Li (2009) QD+ approximation of the
//                                         exercise boundary; also the initial
//                                         guess for the fixed-point engine.
//   ql/pricingengines/vanilla/qdfpamericanengine.{hpp,cpp}
//     * QdFpIterationScheme            -- abstract iteration-scheme interface.
//     * QdFpLegendreScheme             -- (l,m,n)-p Gauss-Legendre scheme.
//     * QdFpTanhSinhIterationScheme    -- (m,n)-eps tanh-sinh scheme.
//     * QdFpLegendreTanhSinhScheme     -- (l,m,n)-eps hybrid scheme.
//     * QdFpAmericanEngine             -- Andersen/Lake/Offengenden fixed-point
//                                         engine (2015, 2021).
//
// What has to be pinned, and why
// ------------------------------
// 1. THE SOLVER KNOB IS REAL. QdPlusAmericanEngine takes
//    (process, interpolationPoints, solverType, eps, maxIter) with
//    SolverType in {Brent, Newton, Ridder, Halley, SuperHalley}. Brent /
//    Newton / Ridder go through QuantLib's Solver1D via buildInSolver();
//    Halley and SuperHalley are hand-rolled in putExerciseBoundaryAtTau and
//    differ only in the step
//        Halley:      step = 1/(1 - lf/2) * f/f'
//        SuperHalley: step = (1 + lf/(2(1-lf))) * f/f',   lf = f f'' / f'^2
//    with a Brent fallback (10*maxIter) if the iteration has not converged.
//    maxIter also defaults differently per solver: 100 for Brent/Newton/Ridder,
//    10 for Halley/SuperHalley. Every solver is pinned at the *boundary* level
//    (value AND the evaluation count the engine reports) and at the NPV level,
//    so a port that hardcodes one solver cannot pass.
// 2. THE FIXED-POINT EQUATION KNOB IS REAL. QdFpAmericanEngine takes
//    (process, iterationScheme, fpEquation) with fpEquation in
//    {FP_A, FP_B, Auto}. FP-A and FP-B are genuinely different equations
//    (DqFpEquation_A integrates over m = tau/4 (1+y)^2 with a Phi + phi
//    kernel, DqFpEquation_B over u in [0, tau] with a pure Phi kernel) and
//    give visibly different premia. Auto picks
//        (std::abs(r - q) < 0.001) ? FP_A : FP_B
//    -- pinned on BOTH sides of that switch, with FP_A / FP_B / Auto all
//    priced for the same market so the port's Auto must coincide with the
//    right one.
// 3. THE QUADRATURE ORDER CANNOT BE SUBSTITUTED SILENTLY. For every scheme
//    (the three public classes plus the three static factories
//    fastScheme() = QdFpLegendreScheme(7,2,7,27),
//    accurateScheme() = QdFpLegendreTanhSinhScheme(25,5,13,1e-8),
//    highPrecisionScheme() = QdFpTanhSinhIterationScheme(10,30,1e-10))
//    we pin getNumberOfChebyshevInterpolationNodes(),
//    getNumberOfNaiveFixedPointSteps() (== m-1),
//    getNumberOfJacobiNewtonFixedPointSteps() (== 1, always), *and* the value
//    each of getFixedPointIntegrator() / getExerciseBoundaryToPriceIntegrator()
//    produces on three fixed integrands. An l-point Gauss-Legendre rule is
//    exact for polynomials of degree < 2l but badly wrong on 1/(1+x^2) and on
//    sqrt(x), so those two integrals fingerprint the order; the normal pdf on
//    [-10,10] is the upstream test's own check. A port that substitutes a
//    different order, or reuses one integrator where C++ has two, fails here
//    before any option is priced.
//    NOTE the C++ spelling: ...FixedPointSteps, not ...FixedPointIterations.
// 4. THE BOUNDARY ITSELF, NOT ONLY THE PRICE. putExerciseBoundaryAtTau is
//    protected/public surface that QdFpAmericanEngine consumes; a boundary
//    that is wrong in the middle of [0, T] can still give a nearly-right price
//    at one strike. So we pin the boundary at several taus, the whole
//    Chebyshev interpolation getPutExerciseBoundary() produces (evaluated at
//    its own nodes and between them), and QdPlusAddOnValue at several z.
//
// Guard clauses / early returns that the cases deliberately hit
// ------------------------------------------------------------
//  a. QdPutCallParityEngine::calculate
//       QL_REQUIRE(exercise->type() == American)      -> throws_european_exercise
//       QL_REQUIRE(dynamic_pointer_cast<StrikedTypePayoff>) -> throws_non_striked_payoff
//         (note: the requirement is *striked*, not *plain vanilla*; a
//          CashOrNothingPayoff is accepted and priced as if plain vanilla,
//          because the engine only ever reads strike() and optionType().
//          Case `qdplus_cash_or_nothing_payoff_is_accepted` pins that.)
//       QL_REQUIRE(spot >= 0.0)                       -> throws_negative_spot
//       Call branch: calculatePutWithEdgeCases(K, S, q, r, vol, T).
//  b. QdPutCallParityEngine::calculatePutWithEdgeCases
//       close(K, 0)            -> 0                       (K = 0, and K = 1e-8 which is NOT close)
//       close(S, 0)            -> max(K, K exp(-rT))      (S = 0; both r > 0 and r < 0)
//       r <= 0 && r <= q       -> European put, floored at 0
//       close(vol, 0)          -> max over {t=0, t=T, t=extremT} of the
//                                 discounted intrinsic, where
//                                 extremT = log(rK/(qS))/(r-q), or QL_MAX_REAL
//                                 when close_enough(r, q). Cases cover
//                                 extremT inside (0,T), outside, and the r==q
//                                 degenerate branch.
//  c. QdPlusAmericanEngine::xMax -- all eight branches of Table 2 of
//     Andersen/Lake (2021) including the two that return 0 (European case)
//     and the "internal error" fall-through is unreachable by construction.
//  d. QdPlusAmericanEngine::calculatePut and QdFpAmericanEngine::calculatePut
//       QL_FAIL when r < 0 && q < r (double-boundary put). Pinned as a throw.
//       Because the CALL branch feeds (K, S, q, r) into the same function, a
//       call throws exactly when q < 0 && r < q -- so the parity image of the
//       throwing put throws too (pinned), while a call with q < 0 <= r, or
//       q < 0 and r >= q, early-returns the European value instead
//       (qdplus_neg_div_call_european). Those two are the call-side halves of
//       the `r <= 0 && r <= q` guard and of the QL_FAIL, and they are what a
//       port that forgets the (S,K,r,q) -> (K,S,q,r) swap gets wrong.
//  e. QdPlusAmericanEngine::putExerciseBoundaryAtTau
//       tau < QL_EPSILON -> (0 evaluations, xMax(K, r, q)) -- pinned at tau = 0.
//  f. QdPlusAddOnValue::operator()
//       v >= QL_EPSILON && b_t > QL_EPSILON  -> the two-Phi expression
//       v <  QL_EPSILON                      -> the three close_enough / >
//                                               branches; hit at z = 0 exactly.
//
// Absence of results is pinned too
// --------------------------------
// Neither engine fills a single greek: calculate() assigns results_.value and
// nothing else, and GenericEngine::reset() nulls the whole Greeks block first.
// `greeks_not_provided` pins that option.delta() throws.
//
// Instrument-level short circuit deliberately NOT pinned
// -----------------------------------------------------
// An American exercise whose last date is the evaluation date makes
// OneAssetOption::isExpired() true, so Instrument::calculate() returns 0
// without ever calling the engine. That is instrument behaviour, not engine
// behaviour, and the engine would in fact fail (T = 0 makes
// r = -log(1)/0 = NaN, and xMax() then falls through to "internal error").
// No case here relies on it.
//
// Market convention for every priced case
// ---------------------------------------
// evaluationDate = 1 June 2022, Actual365Fixed, NullCalendar, flat r / q / vol.
// Maturity is given in days, so T = maturity_days / 365 exactly and the engine
// recovers r and q exactly from -log(discount)/T.
//
// Emits JSON on stdout and nothing else.

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/oneassetoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/integrals/integral.hpp>
#include <ql/math/interpolations/chebyshevinterpolation.hpp>
#include <ql/pricingengines/vanilla/qdfpamericanengine.hpp>
#include <ql/pricingengines/vanilla/qdplusamericanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
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
    if (std::isnan(v))
        return "\"nan\"";
    if (std::isinf(v))
        return v > 0 ? "\"inf\"" : "\"-inf\"";
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
        body += "]";
        return put(k, body);
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

std::vector<std::pair<std::string, std::string> > gCases;

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
// Market
// ---------------------------------------------------------------------------
const Date kToday(1, June, 2022);

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

struct Spec {
    Option::Type type;
    Real spot;
    Real strike;
    Size maturityDays;
    Volatility vol;
    Rate r;
    Rate q;
};

ext::shared_ptr<GeneralizedBlackScholesProcess> makeProcess(const Spec& s) {
    return ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(s.spot)),
        Handle<YieldTermStructure>(
            ext::make_shared<FlatForward>(kToday, s.q, dayCounter())),
        Handle<YieldTermStructure>(
            ext::make_shared<FlatForward>(kToday, s.r, dayCounter())),
        Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(
            kToday, NullCalendar(), s.vol, dayCounter())));
}

Obj specInputs(const Spec& s) {
    Obj o;
    o.s("option_type", s.type == Option::Call ? "Call" : "Put")
        .n("spot", s.spot)
        .n("strike", s.strike)
        .i("maturity_days", static_cast<long long>(s.maturityDays))
        .n("volatility", s.vol)
        .n("r", s.r)
        .n("q", s.q)
        .s("evaluation_date", "2022-06-01")
        .s("day_counter", "Actual365Fixed");
    return o;
}

ext::shared_ptr<VanillaOption> makeAmericanOption(const Spec& s) {
    const Date maturity = kToday + Period(static_cast<Integer>(s.maturityDays), Days);
    return ext::make_shared<VanillaOption>(
        ext::make_shared<PlainVanillaPayoff>(s.type, s.strike),
        ext::make_shared<AmericanExercise>(kToday, maturity));
}

// ---------------------------------------------------------------------------
// Solver-type <-> name
// ---------------------------------------------------------------------------
struct SolverSpec {
    QdPlusAmericanEngine::SolverType type;
    const char* name;
};

const SolverSpec kSolvers[] = {{QdPlusAmericanEngine::Brent, "Brent"},
                               {QdPlusAmericanEngine::Newton, "Newton"},
                               {QdPlusAmericanEngine::Ridder, "Ridder"},
                               {QdPlusAmericanEngine::Halley, "Halley"},
                               {QdPlusAmericanEngine::SuperHalley, "SuperHalley"}};

// ---------------------------------------------------------------------------
// 1. QdPlusAmericanEngine::xMax -- Table 2 of Andersen/Lake (2021)
// ---------------------------------------------------------------------------
void emitXMax() {
    struct XCase {
        const char* name;
        Real K;
        Rate r;
        Rate q;
    };
    const XCase cases[] = {
        {"xmax_r_pos_q_pos_r_lt_q", 120.0, 0.03, 0.10},   // K*r/q
        {"xmax_r_pos_q_pos_r_gt_q", 120.0, 0.10, 0.03},   // K*min(1, r/q) = K
        {"xmax_r_pos_q_pos_r_eq_q", 120.0, 0.05, 0.05},   // K
        {"xmax_r_pos_q_zero", 120.0, 0.10, 0.0},          // K
        {"xmax_r_pos_q_neg", 120.0, 0.10, -0.02},         // K
        {"xmax_r_zero_q_neg", 120.0, 0.0, -0.02},         // K
        {"xmax_r_zero_q_zero", 120.0, 0.0, 0.0},          // 0 (European)
        {"xmax_r_zero_q_pos", 120.0, 0.0, 0.03},          // 0 (European)
        {"xmax_r_neg_q_zero", 120.0, -0.05, 0.0},         // 0 (European)
        {"xmax_r_neg_q_pos", 120.0, -0.05, 0.03},         // 0 (European)
        {"xmax_r_neg_q_lt_r", 120.0, -0.05, -0.10},       // K (double boundary)
        {"xmax_r_neg_q_between", 120.0, -0.10, -0.05},    // 0 (European)
        {"xmax_r_neg_q_eq_r", 120.0, -0.05, -0.05},       // 0 (European)
    };

    for (const auto& c : cases) {
        Obj in;
        in.n("strike", c.K).n("r", c.r).n("q", c.q);
        Obj out;
        out.n("x_max", QdPlusAmericanEngine::xMax(c.K, c.r, c.q));
        addCase(c.name, in, out);
    }
}

// ---------------------------------------------------------------------------
// 2. putExerciseBoundaryAtTau -- every solver, several taus, two markets
// ---------------------------------------------------------------------------
struct BoundaryMarket {
    const char* tag;
    Real S, K;
    Rate r, q;
    Volatility vol;
    Time T;
    std::vector<Time> taus;
};

void emitBoundaries() {
    const std::vector<BoundaryMarket> markets = {
        // The upstream testQdPlusBoundaryValues market (S=100, K=120, r=0.1,
        // q=0.03, sigma=0.25, T=5). r > q > 0 so xMax = K.
        {"m1", 100.0, 120.0, 0.10, 0.03, 0.25, 5.0, {0.0, 0.1, 1.0, 2.5, 4.0, 4.9}},
        // r << q > 0 so xMax = K*r/q -- a completely different scale for the
        // upper bracket, and the r/(1-dr) small-r series in the evaluator.
        {"m2", 100.0, 120.0, 0.0001, 0.03, 0.25, 10.0, {0.05, 2.0, 9.0}},
        // q == 0 exactly: omega = 2r/sigma^2, xMax = K.
        {"m3", 100.0, 30.0, 0.03, 0.0, 0.25, 10.0, {0.1, 5.0}},
        // Near-degenerate volatility (1e-6, i.e. NOT close(vol, 0) so the full
        // machinery runs) over 40 years: v = sigma sqrt(tau) ~ 6e-6, so
        // dp = log(...)/v is amplified by 1e6 and the boundary equation is at
        // its worst conditioning. This is the parameter set upstream's
        // testQdAmericanEngines gives a precision of 1e-4.
        {"m4", 100.0, 120.0, 0.01, 0.50, 1e-6, 40.0, {0.5, 10.0, 25.0, 39.5}},
    };

    for (const auto& m : markets) {
        for (const auto& solver : kSolvers) {
            // eps = 1e-8 as in testQdPlusBoundaryConvergence; maxIter left at
            // the per-solver default (100 for Brent/Newton/Ridder, 10 for the
            // Halley pair) by passing Null<Size>().
            const QdPlusAmericanEngine engine(
                ext::shared_ptr<GeneralizedBlackScholesProcess>(), 10,
                solver.type, 1e-8);

            for (std::size_t k = 0; k < m.taus.size(); ++k) {
                const Time tau = m.taus[k];
                const std::pair<Size, Real> res = engine.putExerciseBoundaryAtTau(
                    m.S, m.K, m.r, m.q, m.vol, m.T, tau);

                Obj in;
                in.s("market", m.tag)
                    .n("spot", m.S)
                    .n("strike", m.K)
                    .n("r", m.r)
                    .n("q", m.q)
                    .n("volatility", m.vol)
                    .n("T", m.T)
                    .n("tau", tau)
                    .s("solver", solver.name)
                    .i("interpolation_points", 10)
                    .n("eps", 1e-8);
                Obj out;
                out.n("boundary", res.second)
                    .i("evaluations", static_cast<long long>(res.first));

                addCase(std::string("boundary_") + m.tag + "_" + solver.name +
                            "_tau" + std::to_string(k),
                        in, out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. getPutExerciseBoundary -- the whole Chebyshev interpolation, and
//    QdPlusAddOnValue evaluated against it.
// ---------------------------------------------------------------------------
void emitBoundaryCurveAndAddOn(const std::string& tag,
                               Real S,
                               Real K,
                               Rate r,
                               Rate q,
                               Volatility vol,
                               Time T,
                               const std::vector<Size>& interpolationPoints) {
    const std::vector<Real> zs = {-1.0,  -0.9, -0.5, -0.25, 0.0,
                                  0.125, 0.5,  0.75, 1.0};

    for (Size np : interpolationPoints) {
        const QdPlusAmericanEngine engine(
            ext::shared_ptr<GeneralizedBlackScholesProcess>(), np,
            QdPlusAmericanEngine::Halley, 1e-8);

        const ext::shared_ptr<ChebyshevInterpolation> interp =
            engine.getPutExerciseBoundary(S, K, r, q, vol, T);

        std::vector<Real> nodes;
        for (Real z : interp->nodes())
            nodes.push_back(z);

        std::vector<Real> values;
        for (Real z : zs)
            values.push_back((*interp)(z, true));

        std::vector<Real> nodeValues;
        for (Real z : nodes)
            nodeValues.push_back((*interp)(z, true));

        Obj in;
        in.s("market", tag)
            .n("spot", S)
            .n("strike", K)
            .n("r", r)
            .n("q", q)
            .n("volatility", vol)
            .n("T", T)
            .i("interpolation_points", static_cast<long long>(np))
            .s("solver", "Halley")
            .n("eps", 1e-8)
            .a("z", zs);
        Obj out;
        out.a("nodes", nodes).a("node_values", nodeValues).a("values", values);
        addCase("put_exercise_boundary_" + tag + "_n" + std::to_string(np), in, out);

        // QdPlusAddOnValue on top of that boundary. z in [0, sqrt(T)];
        // z = 0 exercises the v < QL_EPSILON branch exactly.
        const Real xmax = QdPlusAmericanEngine::xMax(K, r, q);
        const detail::QdPlusAddOnValue aov(T, S, K, r, q, vol, xmax, interp);

        const std::vector<Real> addOnZ = {0.0,
                                          1e-8,
                                          0.05,
                                          0.5,
                                          1.0,
                                          std::sqrt(T) * 0.5,
                                          std::sqrt(T) * 0.99,
                                          std::sqrt(T)};
        std::vector<Real> addOnV;
        for (Real z : addOnZ)
            addOnV.push_back(aov(z));

        Obj ain;
        ain.s("market", tag)
            .n("spot", S)
            .n("strike", K)
            .n("r", r)
            .n("q", q)
            .n("volatility", vol)
            .n("T", T)
            .n("x_max", xmax)
            .i("interpolation_points", static_cast<long long>(np))
            .s("solver", "Halley")
            .n("eps", 1e-8)
            .a("z", addOnZ);
        Obj aout;
        aout.a("values", addOnV);
        addCase("add_on_value_" + tag + "_n" + std::to_string(np), ain, aout);
    }
}

// ---------------------------------------------------------------------------
// 4. QdPlusAmericanEngine NPVs
// ---------------------------------------------------------------------------
Obj qdPlusEngineInputs(Size np, const char* solver, Real eps) {
    Obj o;
    o.s("engine", "QdPlusAmericanEngine")
        .i("interpolation_points", static_cast<long long>(np))
        .s("solver", solver)
        .n("eps", eps);
    return o;
}

void addQdPlusCase(const std::string& name,
                   const Spec& spec,
                   Size np,
                   const SolverSpec& solver,
                   Real eps) {
    const auto process = makeProcess(spec);
    const auto option = makeAmericanOption(spec);
    option->setPricingEngine(ext::make_shared<QdPlusAmericanEngine>(
        process, np, solver.type, eps));

    Obj in = specInputs(spec);
    in.s("engine", "QdPlusAmericanEngine")
        .i("interpolation_points", static_cast<long long>(np))
        .s("solver", solver.name)
        .n("eps", eps);

    Obj out;
    try {
        out.n("npv", option->NPV()).b("throws", false);
    } catch (const std::exception&) {
        out = Obj();
        out.b("throws", true);
    }
    addCase(name, in, out);
}

void emitQdPlusNpvs() {
    // (a) The standard put, priced under every solver. They must agree with
    //     each other to the solver tolerance -- pinned individually so a port
    //     that wires them all to the same code path is visible in the diff.
    const Spec standardPut = {Option::Put, 100.0, 120.0, 3650, 0.25, 0.10, 0.03};
    for (const auto& solver : kSolvers)
        addQdPlusCase(std::string("qdplus_standard_put_") + solver.name,
                      standardPut, 8, solver, 1e-10);

    // (b) The put/call parity image of the same option: a CALL with
    //     (S,K,r,q) = (120,100,0.03,0.10) must price to the same number.
    const Spec parityCall = {Option::Call, 120.0, 100.0, 3650, 0.25, 0.03, 0.10};
    for (const auto& solver : kSolvers)
        addQdPlusCase(std::string("qdplus_parity_call_") + solver.name,
                      parityCall, 8, solver, 1e-10);

    // (c) interpolationPoints and eps sweeps (Halley), same option.
    const Size nps[] = {4, 5, 8, 16};
    for (Size np : nps)
        addQdPlusCase("qdplus_standard_put_n" + std::to_string(np), standardPut,
                      np, kSolvers[3], 1e-10);
    const Real epsValues[] = {1e-4, 1e-6, 1e-8, 1e-10};
    for (std::size_t j = 0; j < 4; ++j)
        addQdPlusCase("qdplus_standard_put_eps" + std::to_string(j), standardPut,
                      8, kSolvers[3], epsValues[j]);

    // (d) Degenerate-input branches of calculatePutWithEdgeCases.
    struct Edge {
        const char* name;
        Spec spec;
    };
    const Edge edges[] = {
        // close(K, 0) -> 0
        {"qdplus_zero_strike_put", {Option::Put, 100.0, 0.0, 365, 0.25, 0.02, 0.02}},
        // K = 1e-8 is NOT close(K, 0) (close() is relative) -> full machinery
        {"qdplus_tiny_strike_put", {Option::Put, 100.0, 1e-8, 365, 0.25, 0.02, 0.02}},
        // close(S, 0) after the call parity swap -> max(K, K exp(-rT))
        {"qdplus_zero_strike_call", {Option::Call, 100.0, 0.0, 365, 0.25, 0.05, 0.01}},
        {"qdplus_tiny_strike_call", {Option::Call, 100.0, 1e-7, 365, 0.25, 0.05, 0.01}},
        // close(S, 0) directly, with r > 0 so max(K, K exp(-rT)) = K
        {"qdplus_zero_spot_put_r_pos", {Option::Put, 0.0, 120.0, 365, 0.25, 0.075, 0.05}},
        // ... and with r < 0 so the discounted branch wins
        {"qdplus_zero_spot_put_r_neg", {Option::Put, 0.0, 120.0, 365, 0.25, -0.075, 0.05}},
        {"qdplus_tiny_spot_put_r_neg", {Option::Put, 1e-6, 120.0, 365, 0.25, -0.075, 0.05}},
        {"qdplus_zero_spot_call", {Option::Call, 0.0, 120.0, 365, 0.25, 0.075, 0.05}},
        // r <= 0 && r <= q -> European put
        {"qdplus_neg_rate_put_european", {Option::Put, 100.0, 120.0, 365, 0.25, -0.02, 0.01}},
        {"qdplus_zero_rate_put_european", {Option::Put, 100.0, 120.0, 365, 0.25, 0.0, 0.05}},
        // r == 0, q == 0 -> r <= 0 && r <= q -> European put
        {"qdplus_zero_rate_zero_div_put", {Option::Put, 100.0, 120.0, 365, 0.25, 0.0, 0.0}},
        // close(vol, 0): extremT = log(rK/(qS))/(r-q) inside (0, T)
        {"qdplus_zero_vol_put_extremT_inside",
         {Option::Put, 100.0, 120.0, 14600, 0.0, 0.01, 0.50}},
        {"qdplus_tiny_vol_put_extremT_inside",
         {Option::Put, 100.0, 120.0, 14600, 1e-6, 0.01, 0.50}},
        // close(vol, 0): extremT outside (0, T)
        {"qdplus_zero_vol_put_extremT_outside",
         {Option::Put, 100.0, 120.0, 365, 0.0, 0.05, 0.01}},
        {"qdplus_zero_vol_call", {Option::Call, 100.0, 50.0, 365, 0.0, 0.05, 0.01}},
        {"qdplus_tiny_vol_call", {Option::Call, 100.0, 50.0, 365, 1e-8, 0.05, 0.01}},
        // close(vol, 0) with r == q -> close_enough(r,q) -> extremT = QL_MAX_REAL
        {"qdplus_zero_vol_put_r_eq_q", {Option::Put, 100.0, 120.0, 365, 0.0, 0.05, 0.05}},
        // zero everything
        {"qdplus_zero_everything", {Option::Put, 0.0, 0.0, 365, 0.0, 0.0, 0.0}},
        // r = 0 call (parity -> r' = q = 0.025 > 0)
        {"qdplus_zero_rate_call", {Option::Call, 100.0, 100.0, 365, 0.25, 0.0, 0.025}},
        // q = 0 call
        {"qdplus_zero_div_call", {Option::Call, 100.0, 100.0, 365, 0.25, 0.05, 0.0}},
        // large / small parameter corners
        {"qdplus_extreme_spot_call", {Option::Call, 1e10, 100.0, 365, 0.25, 0.01, 0.05}},
        {"qdplus_extreme_strike_call", {Option::Call, 100.0, 1e10, 365, 0.25, 0.01, 0.05}},
        {"qdplus_extreme_vol_call", {Option::Call, 100.0, 100.0, 365, 100.0, 0.01, 0.05}},
        {"qdplus_extreme_div_call", {Option::Call, 100.0, 100.0, 365, 0.25, 0.10, 10.0}},
        {"qdplus_extreme_maturity_call", {Option::Call, 100.0, 100.0, 62050, 0.25, 0.01, 0.002}},
        {"qdplus_one_day_put", {Option::Put, 100.0, 120.0, 1, 0.25, 0.05, 0.0}},
        // ordinary in/at/out of the money, 1y
        {"qdplus_itm_put", {Option::Put, 90.0, 100.0, 365, 0.20, 0.05, 0.02}},
        {"qdplus_atm_put", {Option::Put, 100.0, 100.0, 365, 0.20, 0.05, 0.02}},
        {"qdplus_otm_put", {Option::Put, 115.0, 100.0, 365, 0.20, 0.05, 0.02}},
        {"qdplus_itm_call", {Option::Call, 110.0, 100.0, 365, 0.20, 0.02, 0.05}},
        {"qdplus_atm_call", {Option::Call, 100.0, 100.0, 365, 0.20, 0.02, 0.05}},
        {"qdplus_otm_call", {Option::Call, 85.0, 100.0, 365, 0.20, 0.02, 0.05}},
        // the double-boundary put: r < 0 && q < r -> QL_FAIL
        {"qdplus_double_boundary_put", {Option::Put, 100.0, 120.0, 365, 0.25, -0.05, -0.10}},
        // its parity image: a call with q < 0 and r < q hits the same QL_FAIL
        // after the (S,K,r,q) -> (K,S,q,r) swap
        {"qdplus_double_boundary_call", {Option::Call, 120.0, 100.0, 365, 0.25, -0.10, -0.05}},
        // a call with q < 0 <= r instead early-returns the European value,
        // because after the swap r' = q <= 0 and r' <= q' = r
        {"qdplus_neg_div_call_european", {Option::Call, 100.0, 100.0, 365, 0.25, 0.03, -0.05}},
        // ... and so does a call with r < 0 but r >= q
        {"qdplus_neg_rate_call_european", {Option::Call, 100.0, 100.0, 365, 0.25, -0.02, -0.05}},
    };

    for (const auto& e : edges)
        addQdPlusCase(e.name, e.spec, 8, kSolvers[3], 1e-10);
}

// ---------------------------------------------------------------------------
// 5. Iteration schemes
// ---------------------------------------------------------------------------
Real integrandNormalPdf(Real x) {
    static const NormalDistribution nd;
    return nd(x);
}
Real integrandLorentz(Real x) { return 1.0 / (1.0 + x * x); }
Real integrandSqrt(Real x) { return std::sqrt(x); }

void emitSchemeCase(const std::string& name,
                    const Obj& ctorInputs,
                    const ext::shared_ptr<QdFpIterationScheme>& scheme) {
    const auto fp = scheme->getFixedPointIntegrator();
    const auto eb = scheme->getExerciseBoundaryToPriceIntegrator();

    Obj out;
    out.i("chebyshev_nodes",
          static_cast<long long>(scheme->getNumberOfChebyshevInterpolationNodes()))
        .i("naive_fixed_point_steps",
           static_cast<long long>(scheme->getNumberOfNaiveFixedPointSteps()))
        .i("jacobi_newton_fixed_point_steps",
           static_cast<long long>(scheme->getNumberOfJacobiNewtonFixedPointSteps()))
        .n("fp_normal_pdf_m10_10", (*fp)(integrandNormalPdf, -10.0, 10.0))
        .n("fp_lorentz_m1_1", (*fp)(integrandLorentz, -1.0, 1.0))
        .n("fp_sqrt_0_1", (*fp)(integrandSqrt, 0.0, 1.0))
        .n("eb_normal_pdf_m10_10", (*eb)(integrandNormalPdf, -10.0, 10.0))
        .n("eb_lorentz_m1_1", (*eb)(integrandLorentz, -1.0, 1.0))
        .n("eb_sqrt_0_1", (*eb)(integrandSqrt, 0.0, 1.0));

    addCase(name, ctorInputs, out);
}

void emitSchemes() {
    const Size l = 32, m = 6, n = 18, p = 36;
    const Real tol = 1e-8;

    {
        Obj in;
        in.s("scheme", "QdFpLegendreScheme")
            .i("l", static_cast<long long>(l))
            .i("m", static_cast<long long>(m))
            .i("n", static_cast<long long>(n))
            .i("p", static_cast<long long>(p));
        emitSchemeCase("scheme_legendre", in,
                       ext::make_shared<QdFpLegendreScheme>(l, m, n, p));
    }
    {
        Obj in;
        in.s("scheme", "QdFpLegendreTanhSinhScheme")
            .i("l", static_cast<long long>(l))
            .i("m", static_cast<long long>(m))
            .i("n", static_cast<long long>(n))
            .n("eps", tol);
        emitSchemeCase("scheme_legendre_tanh_sinh", in,
                       ext::make_shared<QdFpLegendreTanhSinhScheme>(l, m, n, tol));
    }
    {
        Obj in;
        in.s("scheme", "QdFpTanhSinhIterationScheme")
            .i("m", static_cast<long long>(m))
            .i("n", static_cast<long long>(n))
            .n("eps", tol);
        emitSchemeCase("scheme_tanh_sinh", in,
                       ext::make_shared<QdFpTanhSinhIterationScheme>(m, n, tol));
    }
    // A second Legendre scheme with a small order, so the l-dependence of the
    // integrator fingerprints is unmistakable.
    {
        Obj in;
        in.s("scheme", "QdFpLegendreScheme").i("l", 7).i("m", 2).i("n", 7).i("p", 27);
        emitSchemeCase("scheme_legendre_small", in,
                       ext::make_shared<QdFpLegendreScheme>(7, 2, 7, 27));
    }
    // The three static factories. Their (l,m,n,p)/eps are compiled in, so the
    // pinned numbers are what proves a port picked the same ones.
    {
        Obj in;
        in.s("scheme", "QdFpAmericanEngine::fastScheme");
        emitSchemeCase("scheme_fast", in, QdFpAmericanEngine::fastScheme());
    }
    {
        Obj in;
        in.s("scheme", "QdFpAmericanEngine::accurateScheme");
        emitSchemeCase("scheme_accurate", in, QdFpAmericanEngine::accurateScheme());
    }
    {
        Obj in;
        in.s("scheme", "QdFpAmericanEngine::highPrecisionScheme");
        emitSchemeCase("scheme_high_precision", in,
                       QdFpAmericanEngine::highPrecisionScheme());
    }
}

// ---------------------------------------------------------------------------
// 6. QdFpAmericanEngine NPVs
// ---------------------------------------------------------------------------
struct SchemeSpec {
    std::string kind;  // "legendre" | "legendre_tanh_sinh" | "tanh_sinh" |
                       // "fast" | "accurate" | "high_precision"
    Size l, m, n, p;
    Real eps;
};

ext::shared_ptr<QdFpIterationScheme> buildScheme(const SchemeSpec& s) {
    if (s.kind == "legendre")
        return ext::make_shared<QdFpLegendreScheme>(s.l, s.m, s.n, s.p);
    if (s.kind == "legendre_tanh_sinh")
        return ext::make_shared<QdFpLegendreTanhSinhScheme>(s.l, s.m, s.n, s.eps);
    if (s.kind == "tanh_sinh")
        return ext::make_shared<QdFpTanhSinhIterationScheme>(s.m, s.n, s.eps);
    if (s.kind == "fast")
        return QdFpAmericanEngine::fastScheme();
    if (s.kind == "accurate")
        return QdFpAmericanEngine::accurateScheme();
    if (s.kind == "high_precision")
        return QdFpAmericanEngine::highPrecisionScheme();
    QL_FAIL("unknown scheme kind");
}

const char* fpName(QdFpAmericanEngine::FixedPointEquation e) {
    switch (e) {
      case QdFpAmericanEngine::FP_A:
        return "FP_A";
      case QdFpAmericanEngine::FP_B:
        return "FP_B";
      default:
        return "Auto";
    }
}

void addQdFpCase(const std::string& name,
                 const Spec& spec,
                 const SchemeSpec& schemeSpec,
                 QdFpAmericanEngine::FixedPointEquation fpEquation) {
    const auto process = makeProcess(spec);
    const auto option = makeAmericanOption(spec);
    option->setPricingEngine(ext::make_shared<QdFpAmericanEngine>(
        process, buildScheme(schemeSpec), fpEquation));

    Obj in = specInputs(spec);
    in.s("engine", "QdFpAmericanEngine").s("fp_equation", fpName(fpEquation));
    {
        // Splice the scheme description in (Obj has no merge; restate).
        in.s("scheme_kind", schemeSpec.kind);
        if (schemeSpec.kind == "legendre")
            in.i("l", static_cast<long long>(schemeSpec.l))
                .i("m", static_cast<long long>(schemeSpec.m))
                .i("n", static_cast<long long>(schemeSpec.n))
                .i("p", static_cast<long long>(schemeSpec.p));
        else if (schemeSpec.kind == "legendre_tanh_sinh")
            in.i("l", static_cast<long long>(schemeSpec.l))
                .i("m", static_cast<long long>(schemeSpec.m))
                .i("n", static_cast<long long>(schemeSpec.n))
                .n("eps", schemeSpec.eps);
        else if (schemeSpec.kind == "tanh_sinh")
            in.i("m", static_cast<long long>(schemeSpec.m))
                .i("n", static_cast<long long>(schemeSpec.n))
                .n("eps", schemeSpec.eps);
    }

    Obj out;
    try {
        out.n("npv", option->NPV()).b("throws", false);
    } catch (const std::exception&) {
        out = Obj();
        out.b("throws", true);
    }
    addCase(name, in, out);
}

void emitQdFpNpvs() {
    const QdFpAmericanEngine::FixedPointEquation fps[] = {
        QdFpAmericanEngine::FP_A, QdFpAmericanEngine::FP_B,
        QdFpAmericanEngine::Auto};

    // (a) testQdEngineStandardExample: S=100, K=95, r=0.075, q=0.05,
    //     sigma=0.25, T=1y, QdFpLegendreScheme(32, 2, 15, 48).
    //     |r - q| = 0.025 >= 0.001, so Auto == FP_B.
    const Spec standard = {Option::Put, 100.0, 95.0, 365, 0.25, 0.075, 0.05};
    const SchemeSpec legendre32 = {"legendre", 32, 2, 15, 48, 0.0};
    for (const auto& fp : fps)
        addQdFpCase(std::string("qdfp_standard_example_") + fpName(fp), standard,
                    legendre32, fp);

    // (b) Auto on the OTHER side of the |r - q| < 0.001 switch: r = 0.0505,
    //     q = 0.05 gives |r - q| = 5e-4, so Auto must equal FP_A.
    const Spec autoA = {Option::Put, 100.0, 95.0, 365, 0.25, 0.0505, 0.05};
    for (const auto& fp : fps)
        addQdFpCase(std::string("qdfp_auto_picks_a_") + fpName(fp), autoA,
                    legendre32, fp);
    // and just outside it: |r - q| = 2e-3 -> Auto must equal FP_B.
    const Spec autoB = {Option::Put, 100.0, 95.0, 365, 0.25, 0.052, 0.05};
    for (const auto& fp : fps)
        addQdFpCase(std::string("qdfp_auto_picks_b_") + fpName(fp), autoB,
                    legendre32, fp);

    // (c) testAndersenLakeHighPrecisionExample: S = K = 100, q = 0.05,
    //     sigma = 0.25, T = 1y, QdFpLegendreTanhSinhScheme(l, m, n, tol),
    //     under two rates and five (l, m, n) triples.
    struct AlSpec {
        Size l, m, n;
        Rate r;
        Real tol;
    };
    const AlSpec alCases[] = {
        {24, 3, 9, 0.05, 1e-6},   {5, 1, 4, 0.05, 1e-3},
        {11, 2, 5, 0.05, 1e-4},   {35, 8, 16, 0.05, 1e-9},
        {65, 8, 32, 0.05, 1e-11}, {5, 1, 4, 0.075, 1e-3},
        {11, 2, 5, 0.075, 1e-4},  {35, 8, 16, 0.075, 1e-9},
        {65, 8, 32, 0.075, 1e-11}};
    int alIndex = 0;
    for (const auto& al : alCases) {
        const Spec spec = {Option::Put, 100.0, 100.0, 365, 0.25, al.r, 0.05};
        const SchemeSpec sc = {"legendre_tanh_sinh", al.l, al.m, al.n, 0, al.tol};
        for (const auto& fp : {QdFpAmericanEngine::FP_A, QdFpAmericanEngine::FP_B})
            addQdFpCase("qdfp_andersen_lake_" + std::to_string(alIndex) + "_" +
                            fpName(fp),
                        spec, sc, fp);
        ++alIndex;
    }

    // (d) The three static factory schemes on a plain put and its parity call.
    const Spec put1y = {Option::Put, 100.0, 110.0, 365, 0.30, 0.06, 0.02};
    const Spec call1y = {Option::Call, 110.0, 100.0, 365, 0.30, 0.02, 0.06};
    const SchemeSpec factories[] = {{"fast", 0, 0, 0, 0, 0.0},
                                    {"accurate", 0, 0, 0, 0, 0.0},
                                    {"high_precision", 0, 0, 0, 0, 0.0}};
    for (const auto& sc : factories) {
        for (const auto& fp : fps) {
            addQdFpCase("qdfp_" + sc.kind + "_put_" + fpName(fp), put1y, sc, fp);
            addQdFpCase("qdfp_" + sc.kind + "_call_" + fpName(fp), call1y, sc, fp);
        }
    }

    // (e) The pure tanh-sinh scheme spelled out explicitly (not via the
    //     factory) so a port cannot pass by hardcoding the factory's numbers.
    const SchemeSpec ts = {"tanh_sinh", 0, 4, 9, 0, 1e-8};
    for (const auto& fp : {QdFpAmericanEngine::FP_A, QdFpAmericanEngine::FP_B})
        addQdFpCase(std::string("qdfp_tanh_sinh_put_") + fpName(fp), put1y, ts, fp);

    // (f) The same degenerate-input battery as QdPlus, through the default
    //     engine (accurateScheme + Auto), because every one of those branches
    //     lives in the shared QdPutCallParityEngine base.
    const SchemeSpec accurate = {"accurate", 0, 0, 0, 0, 0.0};
    struct Edge {
        const char* name;
        Spec spec;
    };
    const Edge edges[] = {
        {"qdfp_zero_strike_put", {Option::Put, 100.0, 0.0, 365, 0.25, 0.02, 0.02}},
        {"qdfp_tiny_strike_put", {Option::Put, 100.0, 1e-8, 365, 0.25, 0.02, 0.02}},
        {"qdfp_zero_strike_call", {Option::Call, 100.0, 0.0, 365, 0.25, 0.05, 0.01}},
        {"qdfp_tiny_strike_call", {Option::Call, 100.0, 1e-7, 365, 0.25, 0.05, 0.01}},
        {"qdfp_zero_spot_put_r_pos", {Option::Put, 0.0, 120.0, 365, 0.25, 0.075, 0.05}},
        {"qdfp_zero_spot_put_r_neg", {Option::Put, 0.0, 120.0, 365, 0.25, -0.075, 0.05}},
        {"qdfp_zero_spot_call", {Option::Call, 0.0, 120.0, 365, 0.25, 0.075, 0.05}},
        {"qdfp_neg_rate_put_european", {Option::Put, 100.0, 120.0, 365, 0.25, -0.02, 0.01}},
        {"qdfp_zero_rate_put_european", {Option::Put, 100.0, 120.0, 365, 0.25, 0.0, 0.05}},
        {"qdfp_zero_vol_put_extremT_inside",
         {Option::Put, 100.0, 120.0, 14600, 0.0, 0.01, 0.50}},
        {"qdfp_zero_vol_put_extremT_outside",
         {Option::Put, 100.0, 120.0, 365, 0.0, 0.05, 0.01}},
        {"qdfp_zero_vol_call", {Option::Call, 100.0, 50.0, 365, 0.0, 0.05, 0.01}},
        {"qdfp_zero_everything", {Option::Put, 0.0, 0.0, 365, 0.0, 0.0, 0.0}},
        {"qdfp_zero_rate_call", {Option::Call, 100.0, 100.0, 365, 0.25, 0.0, 0.025}},
        {"qdfp_zero_div_call", {Option::Call, 100.0, 100.0, 365, 0.25, 0.05, 0.0}},
        {"qdfp_extreme_spot_call", {Option::Call, 1e10, 100.0, 365, 0.25, 0.01, 0.05}},
        {"qdfp_extreme_strike_call", {Option::Call, 100.0, 1e10, 365, 0.25, 0.01, 0.05}},
        {"qdfp_extreme_vol_call", {Option::Call, 100.0, 100.0, 365, 100.0, 0.01, 0.05}},
        {"qdfp_extreme_div_call", {Option::Call, 100.0, 100.0, 365, 0.25, 0.10, 10.0}},
        {"qdfp_one_day_put", {Option::Put, 100.0, 120.0, 1, 0.25, 0.05, 0.0}},
        {"qdfp_itm_put", {Option::Put, 90.0, 100.0, 365, 0.20, 0.05, 0.02}},
        {"qdfp_atm_put", {Option::Put, 100.0, 100.0, 365, 0.20, 0.05, 0.02}},
        {"qdfp_otm_put", {Option::Put, 115.0, 100.0, 365, 0.20, 0.05, 0.02}},
        {"qdfp_itm_call", {Option::Call, 110.0, 100.0, 365, 0.20, 0.02, 0.05}},
        {"qdfp_long_dated_put", {Option::Put, 100.0, 120.0, 3650, 0.25, 0.10, 0.03}},
        {"qdfp_double_boundary_put", {Option::Put, 100.0, 120.0, 365, 0.25, -0.05, -0.10}},
        {"qdfp_double_boundary_call", {Option::Call, 120.0, 100.0, 365, 0.25, -0.10, -0.05}},
    };
    for (const auto& e : edges)
        addQdFpCase(e.name, e.spec, accurate, QdFpAmericanEngine::Auto);
}

// ---------------------------------------------------------------------------
// 7. Throws and the absent greeks
// ---------------------------------------------------------------------------
void emitThrowsAndGreeks() {
    const Spec spec = {Option::Put, 100.0, 110.0, 365, 0.30, 0.06, 0.02};
    const auto process = makeProcess(spec);
    const Date maturity = kToday + Period(365, Days);

    // (a) European exercise -> "not an American option", for both engines.
    {
        const auto option = ext::make_shared<VanillaOption>(
            ext::make_shared<PlainVanillaPayoff>(Option::Put, spec.strike),
            ext::make_shared<EuropeanExercise>(maturity));
        option->setPricingEngine(
            ext::make_shared<QdPlusAmericanEngine>(process));
        Obj in = specInputs(spec);
        in.s("engine", "QdPlusAmericanEngine").s("exercise", "European");
        Obj out;
        try {
            option->NPV();
            out.b("throws", false);
        } catch (const std::exception&) {
            out.b("throws", true);
        }
        addCase("throws_qdplus_european_exercise", in, out);
    }
    {
        const auto option = ext::make_shared<VanillaOption>(
            ext::make_shared<PlainVanillaPayoff>(Option::Put, spec.strike),
            ext::make_shared<EuropeanExercise>(maturity));
        option->setPricingEngine(ext::make_shared<QdFpAmericanEngine>(process));
        Obj in = specInputs(spec);
        in.s("engine", "QdFpAmericanEngine").s("exercise", "European");
        Obj out;
        try {
            option->NPV();
            out.b("throws", false);
        } catch (const std::exception&) {
            out.b("throws", true);
        }
        addCase("throws_qdfp_european_exercise", in, out);
    }

    // (b) Non-striked payoff -> "non-striked payoff given". VanillaOption's
    //     constructor only accepts a StrikedTypePayoff, so the bare
    //     OneAssetOption (which takes a Payoff) is what reaches the guard.
    {
        const auto option = ext::make_shared<OneAssetOption>(
            ext::make_shared<NullPayoff>(),
            ext::make_shared<AmericanExercise>(kToday, maturity));
        option->setPricingEngine(
            ext::make_shared<QdPlusAmericanEngine>(process));
        Obj in = specInputs(spec);
        in.s("engine", "QdPlusAmericanEngine").s("payoff", "NullPayoff");
        Obj out;
        try {
            option->NPV();
            out.b("throws", false);
        } catch (const std::exception&) {
            out.b("throws", true);
        }
        addCase("throws_qdplus_non_striked_payoff", in, out);
    }
    {
        const auto option = ext::make_shared<OneAssetOption>(
            ext::make_shared<NullPayoff>(),
            ext::make_shared<AmericanExercise>(kToday, maturity));
        option->setPricingEngine(ext::make_shared<QdFpAmericanEngine>(process));
        Obj in = specInputs(spec);
        in.s("engine", "QdFpAmericanEngine").s("payoff", "NullPayoff");
        Obj out;
        try {
            option->NPV();
            out.b("throws", false);
        } catch (const std::exception&) {
            out.b("throws", true);
        }
        addCase("throws_qdfp_non_striked_payoff", in, out);
    }

    // (c) A CashOrNothingPayoff IS a StrikedTypePayoff, so it is accepted and
    //     priced exactly as the plain-vanilla payoff with the same strike and
    //     type: the engine only reads strike() and optionType(). Pinned so a
    //     port does not tighten the guard to "plain vanilla only".
    {
        const auto cashOption = ext::make_shared<VanillaOption>(
            ext::make_shared<CashOrNothingPayoff>(Option::Put, spec.strike, 1.0),
            ext::make_shared<AmericanExercise>(kToday, maturity));
        cashOption->setPricingEngine(ext::make_shared<QdPlusAmericanEngine>(
            process, 8, QdPlusAmericanEngine::Halley, 1e-10));
        const auto plainOption = makeAmericanOption(spec);
        plainOption->setPricingEngine(ext::make_shared<QdPlusAmericanEngine>(
            process, 8, QdPlusAmericanEngine::Halley, 1e-10));

        Obj in = specInputs(spec);
        in.s("engine", "QdPlusAmericanEngine")
            .s("payoff", "CashOrNothingPayoff")
            .n("cash_payoff", 1.0)
            .i("interpolation_points", 8)
            .s("solver", "Halley")
            .n("eps", 1e-10);
        Obj out;
        out.b("throws", false)
            .n("npv", cashOption->NPV())
            .n("plain_vanilla_npv", plainOption->NPV());
        addCase("qdplus_cash_or_nothing_payoff_is_accepted", in, out);
    }

    // (d) Negative underlying -> "negative underlying given".
    {
        const Spec neg = {Option::Put, -1.0, 110.0, 365, 0.30, 0.06, 0.02};
        const auto option = makeAmericanOption(neg);
        option->setPricingEngine(
            ext::make_shared<QdPlusAmericanEngine>(makeProcess(neg)));
        Obj in = specInputs(neg);
        in.s("engine", "QdPlusAmericanEngine");
        Obj out;
        try {
            option->NPV();
            out.b("throws", false);
        } catch (const std::exception&) {
            out.b("throws", true);
        }
        addCase("throws_qdplus_negative_spot", in, out);
    }

    // (e) No greeks: calculate() assigns results_.value only.
    {
        const auto option = makeAmericanOption(spec);
        option->setPricingEngine(ext::make_shared<QdFpAmericanEngine>(process));
        const Real npv = option->NPV();
        Obj in = specInputs(spec);
        in.s("engine", "QdFpAmericanEngine")
            .s("scheme_kind", "accurate")
            .s("fp_equation", "Auto");
        Obj out;
        out.n("npv", npv);
        bool deltaThrows = false, gammaThrows = false, thetaThrows = false,
             vegaThrows = false, rhoThrows = false;
        try {
            option->delta();
        } catch (const std::exception&) {
            deltaThrows = true;
        }
        try {
            option->gamma();
        } catch (const std::exception&) {
            gammaThrows = true;
        }
        try {
            option->theta();
        } catch (const std::exception&) {
            thetaThrows = true;
        }
        try {
            option->vega();
        } catch (const std::exception&) {
            vegaThrows = true;
        }
        try {
            option->rho();
        } catch (const std::exception&) {
            rhoThrows = true;
        }
        out.b("delta_throws", deltaThrows)
            .b("gamma_throws", gammaThrows)
            .b("theta_throws", thetaThrows)
            .b("vega_throws", vegaThrows)
            .b("rho_throws", rhoThrows);
        addCase("greeks_not_provided", in, out);
    }
}

}  // namespace

int main() {
    try {
        Settings::instance().evaluationDate() = kToday;

        emitXMax();
        emitBoundaries();
        // Well-conditioned: sigma = 0.25.
        emitBoundaryCurveAndAddOn("m1", 100.0, 120.0, 0.10, 0.03, 0.25, 5.0, {8, 13});
        // Near-degenerate: sigma = 1e-6 over 40 years. Pins the boundary and the
        // add-on integrand separately from the price, so a disagreement in the
        // NPV can be attributed to the boundary, the integrand or the quadrature.
        emitBoundaryCurveAndAddOn("m4", 100.0, 120.0, 0.01, 0.50, 1e-6, 40.0, {8});
        emitQdPlusNpvs();
        emitSchemes();
        emitQdFpNpvs();
        emitThrowsAndGreeks();

        emitDocument();
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
