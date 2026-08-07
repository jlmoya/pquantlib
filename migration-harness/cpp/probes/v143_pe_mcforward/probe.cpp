// migration-harness/cpp/probes/v143_pe_mcforward/probe.cpp
//
// Reference values for the Monte Carlo forward-start / variance-swap pricing
// engines of C++ QuantLib v1.43:
//
//   * MCForwardVanillaEngine        (ql/pricingengines/forward/mcforwardvanillaengine.hpp)
//   * MCForwardEuropeanBSEngine     (ql/pricingengines/forward/mcforwardeuropeanbsengine.{hpp,cpp})
//   * ForwardEuropeanBSPathPricer   (same)
//   * MakeMCForwardEuropeanBSEngine (same)
//   * MCForwardEuropeanHestonEngine     (ql/pricingengines/forward/mcforwardeuropeanhestonengine.{hpp,cpp})
//   * ForwardEuropeanHestonPathPricer   (same)
//   * MakeMCForwardEuropeanHestonEngine (same)
//   * MCVarianceSwapEngine          (ql/pricingengines/forward/mcvarianceswapengine.hpp)
//   * VariancePathPricer            (same)
//   * detail::Integrand             (same)
//   * MakeMCVarianceSwapEngine      (same)
//
// Why every case below exists
// ---------------------------
// A Monte Carlo engine is trivially "passable" against a statistical band: a
// port with the wrong RNG, the wrong path construction, the wrong antithetic
// pairing or the wrong time grid still lands inside two standard errors. So
// nothing here is a band. Every sample count is FIXED, every seed is FIXED,
// and both the NPV and the errorEstimate are emitted at 17 significant digits.
// A correct port reproduces them to ~1e-14 relative; an incorrect one does not
// reproduce them at all. That is the entire point of this file.
//
// What a port has to get right, and which case forces it
// ------------------------------------------------------
//  1. TIME GRID. MCForwardVanillaEngine::timeGrid() is NOT a uniform grid over
//     [0, T]. It is
//         TimeGrid(fixingTimes.begin(), fixingTimes.end(), totalSteps)
//     with fixingTimes = {t(resetDate), t(exerciseDate)} — i.e. a *mandatory
//     points* grid, which inserts steps between 0 and t1 and between t1 and t2
//     so that the reset time is exactly on a node. A port that builds
//     TimeGrid(t2, steps) gets a different grid, a different resetIndex, and a
//     different price with the same seed. The `*_steps_*` vs
//     `*_stepsperyear_*` pairs and the deliberately awkward reset dates below
//     make that visible.
//     NOTE the asymmetry: `totalSteps` is derived from timeStepsPerYear_*t2
//     (the *exercise* time), not from the reset time.
//  2. resetIndex = timeGrid.closestIndex(resetTime) — closest, not floor, not
//     lower_bound. `fwdbs_reset_offgrid_*` puts the reset between two nodes.
//  3. ForwardEuropeanBSPathPricer: strike = path[resetIndex] * moneyness, and
//     the payoff is evaluated at path.back(), discounted by
//     riskFreeRate()->discount(timeGrid.back()) — the discount is taken at the
//     grid's last TIME, not at the exercise DATE. Those differ whenever the
//     day-counter year fraction and the grid disagree; `fwdbs_moneyness_*`
//     covers the strike, and the discount at the exercise time is pinned
//     separately in the `market` case as `discountToExercise`.
//  4. TWO moneyness guards, and the stricter one wins.
//     ForwardEuropeanBSPathPricer's ctor has
//       QL_REQUIRE(moneyness >= 0.0, "moneyness less than zero not allowed")
//     but ForwardOptionArguments::validate() has
//       QL_REQUIRE(moneyness > 0.0, "negative or zero moneyness given")
//     and runs first, so moneyness == 0 throws and the path pricer's `>= 0.0`
//     boundary is unreachable through the instrument. Both -0.5 and 0.0 are
//     pinned as throws; a port that only copies the path pricer's guard will
//     price a zero-moneyness option instead of rejecting it.
//  5. CONTROL VARIATE. MCForwardVanillaEngine::controlVariateValue() prices a
//     *plain* vanilla with strike = moneyness * spot (spot = initialValues()[0])
//     through the control pricing engine, and McSimulation applies
//     mean(pathValue) - mean(controlPathValue) + controlVariateValue.
//     `fwdbs_cv_on` vs `fwdbs_cv_off` differ; an engine that accepts
//     `controlVariate` and ignores it returns the same number twice, which is
//     exactly the defect shape this port has been bitten by before.
//     MCForwardEuropeanBSEngine's own constructor does NOT expose it (it always
//     passes the base default false) while MCForwardEuropeanHestonEngine's DOES
//     — that asymmetry is real and is pinned.
//  6. ANTITHETIC. antitheticVariate flips the sample count semantics: with
//     antithetics on, `requiredSamples` counts *pairs* consumed differently, so
//     the same seed and same requiredSamples give a different answer. Pinned on
//     and off for every engine.
//  7. BROWNIAN BRIDGE. brownianBridge=true reorders how the Gaussian draws are
//     mapped onto the path, so it changes the answer at a fixed seed even
//     though it does not change the distribution. Pinned on and off.
//  8. LowDiscrepancy vs PseudoRandom. Sobol has allowsErrorEstimate == false,
//     so `errorEstimate` is Null<Real>() and results_.errorEstimate is never
//     assigned. The `*_ld_*` cases pin `has_error_estimate: false` explicitly so
//     a port cannot invent one. Also, withAbsoluteTolerance() must THROW for
//     LowDiscrepancy ("chosen random generator policy does not allow an error
//     estimate") — pinned.
//  9. MakeMC* validation. withSamples then withAbsoluteTolerance must throw
//     ("tolerance already set" / "number of samples already set"); neither steps
//     nor stepsPerYear must throw ("number of steps not given"); both must throw
//     ("number of steps overspecified"). All four pinned per builder.
// 10. VariancePathPricer integrates the SQUARED DIFFUSION along the path:
//         SegmentIntegral integrator(Size(t/dt));
//         detail::Integrand f(path, process);   // f(u) = sigma(u, path[i])^2
//                                               // with i = Size(u/dt)  <-- floor
//         return integrator(f, t0, t) / t;
//     Two things a port gets wrong here. First, the integrand indexes the path
//     by *truncation* of u/dt, so it is a piecewise-constant left-endpoint
//     sample of the path, not an interpolation. Second, SegmentIntegral with
//     n = Size(t/dt) intervals is a midpoint rule over [t0, t], so it evaluates
//     the integrand at cell CENTRES — which, combined with the truncating
//     index, reads path[i] at i = floor((k+0.5)*dt/dt) = k. The composition is
//     easy to reproduce and easy to get subtly wrong; `vs_*` pins it, and
//     `vs_pathpricer_direct*` pins the path pricer on a hand-built
//     deterministic path so a failure localises to the pricer rather than to
//     the RNG.
//     CAVEAT, verified rather than assumed: the truncating PATH INDEX is not
//     pinned by anything in this file. Both volatility surfaces used here are
//     strike-independent (BlackConstantVol trivially; BlackVarianceCurve
//     because its Dupire local vol is a function of t alone), so
//     localVol(u, path[k]) is the same number for every k. Pinning the index
//     needs a strike-dependent local-vol surface and is left open.
// 11. MCVarianceSwapEngine fills results_.variance, results_.value AND
//         value = position_multiplier * riskFreeDiscount * notional * (variance - strike)
//     with multiplier = -1 for a SHORT position. Both positions pinned; a port
//     that returns |value| passes a long-only test and fails here.
//     MCVarianceSwapEngine passes `false` for controlVariate unconditionally
//     (see its ctor initialiser list) and MakeMCVarianceSwapEngine has NO
//     withControlVariate — pinned as `vs_make_has_no_control_variate`.
// 12. MCVarianceSwapEngine::timeGrid() is TimeGrid(t, steps) — a plain uniform
//     grid, UNLIKE the forward engines. And for stepsPerYear it is
//     max(Size(stepsPerYear*t), 1), so a sub-year swap with a small
//     stepsPerYear collapses to a single step: `vs_stepsperyear_collapses`.
//
// Evaluation date
// ---------------
// Settings::instance().evaluationDate() is set ONCE, below, to
// Date(15, June, 2023). Every reference here depends on it. The Python test
// MUST pin the same date in a fixture and restore it in teardown.
//
// Emits JSON on stdout; nothing else may be printed.

#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/forwardvanillaoption.hpp>
#include <ql/instruments/varianceswap.hpp>
#include <ql/methods/montecarlo/path.hpp>
#include <ql/pricingengines/forward/mcforwardeuropeanbsengine.hpp>
#include <ql/pricingengines/forward/mcforwardeuropeanhestonengine.hpp>
#include <ql/pricingengines/forward/mcvarianceswapengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// ---------------------------------------------------------------------------
std::string num(Real v) {
    if (v == Null<Real>())
        return "null";
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
    Obj& arr(const std::string& k, const std::vector<Real>& v) {
        std::string body = "[";
        for (Size j = 0; j < v.size(); ++j) {
            if (j != 0)
                body += ", ";
            body += num(v[j]);
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
    gCases.emplace_back(name,
                        "{\"inputs\": " + inputs.str() + ", \"expected\": " + expected.str() + "}");
}

void emitDocument() {
    std::cout << "{\n";
    for (Size j = 0; j < gCases.size(); ++j) {
        std::cout << "  \"" << gCases[j].first << "\": " << gCases[j].second;
        if (j + 1 != gCases.size())
            std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// Market. One shared setup so the Python test reconstructs it once.
// ---------------------------------------------------------------------------
const Date TODAY(15, June, 2023);
const Rate RISK_FREE = 0.04;
const Rate DIVIDEND = 0.015;
const Volatility VOL = 0.22;
const Real SPOT = 95.0;

// Reset 4 months out, exercise 15 months out — chosen so that the reset time
// (0.336...) is NOT a node of any uniform grid over [0, 1.2575...], which is
// what forces a port to honour TimeGrid's mandatory-point construction.
const Date RESET(16, October, 2023);
const Date EXERCISE(18, September, 2024);

DayCounter dc() { return Actual365Fixed(); }

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(TODAY, r, dc()));
}

ext::shared_ptr<GeneralizedBlackScholesProcess> bsmProcess() {
    return ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(SPOT)), flatCurve(DIVIDEND), flatCurve(RISK_FREE),
        Handle<BlackVolTermStructure>(
            ext::make_shared<BlackConstantVol>(TODAY, NullCalendar(), VOL, dc())));
}

// A term structure of volatility, so that GeneralizedBlackScholesProcess::
// diffusion(t, x) = localVolatility()->localVol(t, x) actually VARIES along the
// path. With BlackConstantVol the integrand of VariancePathPricer is the
// constant sigma^2 and the whole SegmentIntegral / detail::Integrand machinery
// is unobservable -- every quadrature rule and every path index gives sigma^2.
// These cases are what make the integration itself testable.
ext::shared_ptr<GeneralizedBlackScholesProcess> bsmProcessTermVol() {
    const std::vector<Date> dates{TODAY + Period(3, Months), TODAY + Period(1, Years),
                                  TODAY + Period(2, Years)};
    const std::vector<Volatility> vols{0.15, 0.25, 0.30};
    return ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(SPOT)), flatCurve(DIVIDEND), flatCurve(RISK_FREE),
        Handle<BlackVolTermStructure>(
            ext::make_shared<BlackVarianceCurve>(TODAY, dates, vols, dc(), false)));
}

ext::shared_ptr<HestonProcess> hestonProcess() {
    return ext::make_shared<HestonProcess>(flatCurve(RISK_FREE), flatCurve(DIVIDEND),
                                           Handle<Quote>(ext::make_shared<SimpleQuote>(SPOT)),
                                           /*v0*/ 0.0484, /*kappa*/ 1.5, /*theta*/ 0.0576,
                                           /*sigma*/ 0.4, /*rho*/ -0.65);
}

ext::shared_ptr<ForwardVanillaOption> fwdOption(Option::Type type, Real moneyness) {
    return ext::make_shared<ForwardVanillaOption>(
        moneyness, RESET, ext::make_shared<PlainVanillaPayoff>(type, 0.0),
        ext::make_shared<EuropeanExercise>(EXERCISE));
}

// Every case carries the whole configuration so the Python side rebuilds it.
Obj mcInputs(const std::string& engine,
             const std::string& rng,
             Option::Type type,
             Real moneyness,
             long long steps,
             long long stepsPerYear,
             bool brownianBridge,
             bool antithetic,
             long long samples,
             long long seed,
             bool controlVariate) {
    Obj in;
    in.s("engine", engine);
    in.s("rng", rng);
    in.s("optionType", type == Option::Call ? "Call" : "Put");
    in.n("moneyness", moneyness);
    if (steps > 0)
        in.i("steps", steps);
    if (stepsPerYear > 0)
        in.i("stepsPerYear", stepsPerYear);
    in.b("brownianBridge", brownianBridge);
    in.b("antitheticVariate", antithetic);
    in.i("samples", samples);
    in.i("seed", seed);
    in.b("controlVariate", controlVariate);
    return in;
}

// Instrument::errorEstimate() THROWS ("error estimate not provided") when the
// engine left it Null -- which is exactly what LowDiscrepancy does, since
// Sobol has allowsErrorEstimate == false and the engines guard the assignment
// with `if constexpr (RNG::allowsErrorEstimate)`. So the absence of an error
// estimate is observable only as an exception, and that is what gets pinned.
Real errorEstimateOrNull(const Instrument& inst) {
    try {
        return inst.errorEstimate();
    } catch (const std::exception&) {
        return Null<Real>();
    }
}

// --- forward-start BS engine ------------------------------------------------

template <class RNG>
void runFwdBs(const std::string& name,
              const std::string& rngName,
              Option::Type type,
              Real moneyness,
              Size steps,
              Size stepsPerYear,
              bool brownianBridge,
              bool antithetic,
              Size samples,
              BigNatural seed) {
    auto process = bsmProcess();
    auto option = fwdOption(type, moneyness);

    auto builder = MakeMCForwardEuropeanBSEngine<RNG>(process)
                       .withSamples(samples)
                       .withSeed(seed)
                       .withBrownianBridge(brownianBridge)
                       .withAntitheticVariate(antithetic);
    if (steps != Null<Size>())
        builder.withSteps(steps);
    else
        builder.withStepsPerYear(stepsPerYear);

    option->setPricingEngine(builder);

    Obj in = mcInputs("MCForwardEuropeanBSEngine", rngName, type, moneyness,
                      steps == Null<Size>() ? 0 : (long long)steps,
                      stepsPerYear == Null<Size>() ? 0 : (long long)stepsPerYear, brownianBridge,
                      antithetic, (long long)samples, (long long)seed, false);
    Obj ex;
    ex.n("npv", option->NPV());
    const Real err = errorEstimateOrNull(*option);
    ex.b("hasErrorEstimate", err != Null<Real>());
    ex.n("errorEstimate", err);
    addCase(name, in, ex);
}

// --- forward-start Heston engine -------------------------------------------

template <class RNG>
void runFwdHeston(const std::string& name,
                  const std::string& rngName,
                  Option::Type type,
                  Real moneyness,
                  Size steps,
                  Size stepsPerYear,
                  bool antithetic,
                  Size samples,
                  BigNatural seed,
                  bool controlVariate) {
    auto process = hestonProcess();
    auto option = fwdOption(type, moneyness);

    auto builder = MakeMCForwardEuropeanHestonEngine<RNG>(process)
                       .withSamples(samples)
                       .withSeed(seed)
                       .withAntitheticVariate(antithetic)
                       .withControlVariate(controlVariate);
    if (steps != Null<Size>())
        builder.withSteps(steps);
    else
        builder.withStepsPerYear(stepsPerYear);

    option->setPricingEngine(builder);

    Obj in = mcInputs("MCForwardEuropeanHestonEngine", rngName, type, moneyness,
                      steps == Null<Size>() ? 0 : (long long)steps,
                      stepsPerYear == Null<Size>() ? 0 : (long long)stepsPerYear,
                      /*brownianBridge*/ false, antithetic, (long long)samples, (long long)seed,
                      controlVariate);
    Obj ex;
    ex.n("npv", option->NPV());
    const Real err = errorEstimateOrNull(*option);
    ex.b("hasErrorEstimate", err != Null<Real>());
    ex.n("errorEstimate", err);
    addCase(name, in, ex);
}

// --- variance swap ----------------------------------------------------------

template <class RNG>
void runVarSwap(const std::string& name,
                const std::string& rngName,
                const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
                const std::string& volKind,
                Position::Type position,
                Real strike,
                Real notional,
                Size steps,
                Size stepsPerYear,
                bool brownianBridge,
                bool antithetic,
                Size samples,
                BigNatural seed) {
    VarianceSwap swap(position, strike, notional, TODAY, EXERCISE);

    auto builder = MakeMCVarianceSwapEngine<RNG>(process)
                       .withSamples(samples)
                       .withSeed(seed)
                       .withBrownianBridge(brownianBridge)
                       .withAntitheticVariate(antithetic);
    if (steps != Null<Size>())
        builder.withSteps(steps);
    else
        builder.withStepsPerYear(stepsPerYear);

    swap.setPricingEngine(builder);

    Obj in;
    in.s("engine", "MCVarianceSwapEngine");
    in.s("rng", rngName);
    in.s("volKind", volKind);
    in.s("position", position == Position::Long ? "Long" : "Short");
    in.n("strike", strike);
    in.n("notional", notional);
    if (steps != Null<Size>())
        in.i("steps", (long long)steps);
    else
        in.i("stepsPerYear", (long long)stepsPerYear);
    in.b("brownianBridge", brownianBridge);
    in.b("antitheticVariate", antithetic);
    in.i("samples", (long long)samples);
    in.i("seed", (long long)seed);

    Obj ex;
    ex.n("variance", swap.variance());
    ex.n("npv", swap.NPV());
    const Real err = errorEstimateOrNull(swap);
    ex.b("hasErrorEstimate", err != Null<Real>());
    ex.n("errorEstimate", err);
    addCase(name, in, ex);
}

bool throws(const std::function<void()>& f) {
    try {
        f();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = TODAY;

    // -----------------------------------------------------------------------
    // 0. The market itself, and the derived quantities the engines use, so a
    //    Python failure localises to "wrong market" vs "wrong engine".
    // -----------------------------------------------------------------------
    {
        auto process = bsmProcess();
        Obj in;
        in.i("evaluationDateSerial", (long long)TODAY.serialNumber());
        in.i("resetDateSerial", (long long)RESET.serialNumber());
        in.i("exerciseDateSerial", (long long)EXERCISE.serialNumber());
        in.n("spot", SPOT);
        in.n("riskFreeRate", RISK_FREE);
        in.n("dividendYield", DIVIDEND);
        in.n("volatility", VOL);
        Obj ex;
        ex.n("resetTime", process->time(RESET));
        ex.n("exerciseTime", process->time(EXERCISE));
        ex.n("discountToExercise", process->riskFreeRate()->discount(process->time(EXERCISE)));
        addCase("market", in, ex);
    }

    // The volatility term structure used by the vs_termvol_* cases, pinned as
    // both Black vols and LOCAL vols, because VariancePathPricer reads
    // localVolatility() and the Dupire step from one to the other is exactly
    // where a port diverges without the price ever looking obviously wrong.
    {
        auto tv = bsmProcessTermVol();
        std::vector<Real> times, blackVols, localVols;
        for (Time u = 0.1; u < 1.3; u += 0.1) {
            times.push_back(u);
            blackVols.push_back(tv->blackVolatility()->blackVol(u, SPOT, true));
            localVols.push_back(tv->localVolatility()->localVol(u, SPOT, true));
        }
        Obj in;
        in.s("curve", "BlackVarianceCurve");
        in.i("pillar1DaysMonths", 3);
        in.i("pillar2Years", 1);
        in.i("pillar3Years", 2);
        in.n("vol1", 0.15);
        in.n("vol2", 0.25);
        in.n("vol3", 0.30);
        in.b("forceMonotoneVariance", false);
        Obj ex;
        ex.arr("times", times);
        ex.arr("blackVols", blackVols);
        ex.arr("localVols", localVols);
        addCase("termvol_curve", in, ex);
    }

    // -----------------------------------------------------------------------
    // 1. The mandatory-point time grid. Pinned directly, because every forward
    //    engine result below is downstream of it and a port that builds a
    //    uniform grid fails everything at once with no clue why.
    // -----------------------------------------------------------------------
    {
        auto process = bsmProcess();
        const Time t1 = process->time(RESET);
        const Time t2 = process->time(EXERCISE);
        for (Size totalSteps : {Size(4), Size(12), Size(30)}) {
            std::vector<Time> fixingTimes{t1, t2};
            TimeGrid grid(fixingTimes.begin(), fixingTimes.end(), totalSteps);
            std::vector<Real> times(grid.begin(), grid.end());
            Obj in;
            in.n("t1", t1);
            in.n("t2", t2);
            in.i("totalSteps", (long long)totalSteps);
            Obj ex;
            ex.i("size", (long long)grid.size());
            ex.arr("times", times);
            ex.i("resetIndex", (long long)grid.closestIndex(t1));
            addCase("fwd_timegrid_steps" + std::to_string(totalSteps), in, ex);
        }
    }

    // -----------------------------------------------------------------------
    // 2. MCForwardEuropeanBSEngine — the sweep.
    // -----------------------------------------------------------------------
    runFwdBs<PseudoRandom>("fwdbs_call_m1_steps12_pr", "PseudoRandom", Option::Call, 1.0, 12,
                           Null<Size>(), false, false, 4095, 42);
    runFwdBs<PseudoRandom>("fwdbs_put_m1_steps12_pr", "PseudoRandom", Option::Put, 1.0, 12,
                           Null<Size>(), false, false, 4095, 42);
    // Same everything, different seed: proves the seed reaches the generator.
    runFwdBs<PseudoRandom>("fwdbs_call_m1_steps12_pr_seed7", "PseudoRandom", Option::Call, 1.0, 12,
                           Null<Size>(), false, false, 4095, 7);
    // Antithetic on: same seed, different answer.
    runFwdBs<PseudoRandom>("fwdbs_call_m1_steps12_pr_anti", "PseudoRandom", Option::Call, 1.0, 12,
                           Null<Size>(), false, true, 4095, 42);
    // Brownian bridge on: same seed, different answer.
    runFwdBs<PseudoRandom>("fwdbs_call_m1_steps12_pr_bb", "PseudoRandom", Option::Call, 1.0, 12,
                           Null<Size>(), true, false, 4095, 42);
    // stepsPerYear instead of steps — totalSteps = Size(stepsPerYear * t2).
    runFwdBs<PseudoRandom>("fwdbs_call_m1_stepsperyear10_pr", "PseudoRandom", Option::Call, 1.0,
                           Null<Size>(), 10, false, false, 4095, 42);
    // Moneyness ladder: ITM / ATM / OTM forward-start.
    // NOTE on the boundary: ForwardEuropeanBSPathPricer's own guard is
    // `QL_REQUIRE(moneyness >= 0.0)`, i.e. it admits zero — but
    // ForwardOptionArguments::validate() is stricter
    // (`QL_REQUIRE(moneyness > 0.0, "negative or zero moneyness given")`), so
    // moneyness == 0 is unreachable THROUGH the instrument and throws before
    // the path pricer is ever constructed. Pinned as a throw below
    // (`fwdbs_zero_moneyness_throws`) rather than as a price; a port that
    // relaxes the arguments check to match the path pricer would diverge.
    runFwdBs<PseudoRandom>("fwdbs_call_m085_steps12_pr", "PseudoRandom", Option::Call, 0.85, 12,
                           Null<Size>(), false, false, 4095, 42);
    runFwdBs<PseudoRandom>("fwdbs_call_m125_steps12_pr", "PseudoRandom", Option::Call, 1.25, 12,
                           Null<Size>(), false, false, 4095, 42);
    // Coarse grid: reset lands off-node, so closestIndex actually matters.
    runFwdBs<PseudoRandom>("fwdbs_reset_offgrid_steps3_pr", "PseudoRandom", Option::Call, 1.0, 3,
                           Null<Size>(), false, false, 4095, 42);
    // LowDiscrepancy: no error estimate at all.
    runFwdBs<LowDiscrepancy>("fwdbs_call_m1_steps12_ld", "LowDiscrepancy", Option::Call, 1.0, 12,
                             Null<Size>(), false, false, 4096, 42);
    runFwdBs<LowDiscrepancy>("fwdbs_call_m1_steps12_ld_anti", "LowDiscrepancy", Option::Call, 1.0,
                             12, Null<Size>(), false, true, 4096, 42);

    // -----------------------------------------------------------------------
    // 3. MCForwardEuropeanHestonEngine — including the control variate, which
    //    this engine DOES expose and the BS one does not.
    // -----------------------------------------------------------------------
    runFwdHeston<PseudoRandom>("fwdheston_call_m1_steps24_pr", "PseudoRandom", Option::Call, 1.0, 24,
                               Null<Size>(), false, 2047, 42, false);
    runFwdHeston<PseudoRandom>("fwdheston_put_m1_steps24_pr", "PseudoRandom", Option::Put, 1.0, 24,
                               Null<Size>(), false, 2047, 42, false);
    runFwdHeston<PseudoRandom>("fwdheston_call_m1_steps24_pr_anti", "PseudoRandom", Option::Call,
                               1.0, 24, Null<Size>(), true, 2047, 42, false);
    runFwdHeston<PseudoRandom>("fwdheston_call_m1_steps24_pr_cv", "PseudoRandom", Option::Call, 1.0,
                               24, Null<Size>(), false, 2047, 42, true);
    runFwdHeston<PseudoRandom>("fwdheston_call_m085_steps24_pr", "PseudoRandom", Option::Call, 0.85,
                               24, Null<Size>(), false, 2047, 42, false);
    runFwdHeston<PseudoRandom>("fwdheston_call_m125_steps24_pr", "PseudoRandom", Option::Call, 1.25,
                               24, Null<Size>(), false, 2047, 42, false);
    runFwdHeston<PseudoRandom>("fwdheston_call_stepsperyear20_pr", "PseudoRandom", Option::Call, 1.0,
                               Null<Size>(), 20, false, 2047, 42, false);
    runFwdHeston<LowDiscrepancy>("fwdheston_call_m1_steps24_ld", "LowDiscrepancy", Option::Call, 1.0,
                                 24, Null<Size>(), false, 2048, 42, false);

    // -----------------------------------------------------------------------
    // 4. MCVarianceSwapEngine.
    // -----------------------------------------------------------------------
    runVarSwap<PseudoRandom>("vs_long_steps52_pr", "PseudoRandom", bsmProcess(), "constant", Position::Long, 0.04, 1000000.0,
                             52, Null<Size>(), false, false, 1023, 42);
    // Short position: multiplier = -1. A port returning |value| fails here only.
    runVarSwap<PseudoRandom>("vs_short_steps52_pr", "PseudoRandom", bsmProcess(), "constant", Position::Short, 0.04, 1000000.0,
                             52, Null<Size>(), false, false, 1023, 42);
    runVarSwap<PseudoRandom>("vs_long_steps52_pr_anti", "PseudoRandom", bsmProcess(), "constant", Position::Long, 0.04,
                             1000000.0, 52, Null<Size>(), false, true, 1023, 42);
    runVarSwap<PseudoRandom>("vs_long_steps52_pr_bb", "PseudoRandom", bsmProcess(), "constant", Position::Long, 0.04,
                             1000000.0, 52, Null<Size>(), true, false, 1023, 42);
    // Strike above the realised variance -> negative NPV for a long.
    runVarSwap<PseudoRandom>("vs_long_highstrike_steps52_pr", "PseudoRandom", bsmProcess(), "constant", Position::Long, 0.09,
                             1000000.0, 52, Null<Size>(), false, false, 1023, 42);
    runVarSwap<PseudoRandom>("vs_long_stepsperyear12_pr", "PseudoRandom", bsmProcess(), "constant", Position::Long, 0.04,
                             1000000.0, Null<Size>(), 12, false, false, 1023, 42);
    // stepsPerYear = 0 would give Size(0*t) = 0 -> max(...,1) = 1 step. The
    // ctor rejects timeStepsPerYear == 0, so use 1 with t < 1 instead, which
    // is the smallest configuration that reaches the max(steps,1) clamp.
    runVarSwap<PseudoRandom>("vs_stepsperyear_collapses", "PseudoRandom", bsmProcess(), "constant", Position::Long, 0.04,
                             1000000.0, Null<Size>(), 1, false, false, 1023, 42);
    runVarSwap<LowDiscrepancy>("vs_long_steps52_ld", "LowDiscrepancy", bsmProcess(), "constant", Position::Long, 0.04,
                               1000000.0, 52, Null<Size>(), false, false, 1024, 42);

    // Term structure of volatility: now sigma(u, .) varies with TIME, so the
    // SegmentIntegral midpoint rule and the Size(t/dt) interval count become
    // observable -- under BlackConstantVol they are not, because the integrand
    // is the constant sigma^2 and any quadrature reproduces it.
    //
    // What these cases still do NOT pin: detail::Integrand's truncating path
    // index u -> path[Size(u/dt)]. BlackVarianceCurve is strike-independent,
    // so its Dupire local vol is a function of t alone and localVol(u, path[k])
    // does not depend on k at all. Discriminating floor from round needs a
    // strike-dependent local-vol surface; that is an open gap in this probe,
    // recorded here rather than left to be assumed covered.
    runVarSwap<PseudoRandom>("vs_termvol_long_steps52_pr", "PseudoRandom", bsmProcessTermVol(),
                             "termStructure", Position::Long, 0.04, 1000000.0, 52, Null<Size>(),
                             false, false, 1023, 42);
    runVarSwap<PseudoRandom>("vs_termvol_long_steps13_pr", "PseudoRandom", bsmProcessTermVol(),
                             "termStructure", Position::Long, 0.04, 1000000.0, 13, Null<Size>(),
                             false, false, 1023, 42);
    runVarSwap<PseudoRandom>("vs_termvol_short_steps52_pr", "PseudoRandom", bsmProcessTermVol(),
                             "termStructure", Position::Short, 0.04, 1000000.0, 52, Null<Size>(),
                             false, false, 1023, 42);
    runVarSwap<PseudoRandom>("vs_termvol_long_stepsperyear20_pr", "PseudoRandom",
                             bsmProcessTermVol(), "termStructure", Position::Long, 0.04, 1000000.0,
                             Null<Size>(), 20, false, false, 1023, 42);

    // -----------------------------------------------------------------------
    // 5. VariancePathPricer + detail::Integrand, directly, on a hand-built
    //    deterministic path. This localises a failure to the pricer rather
    //    than to the RNG, and it is the only way to pin detail::Integrand's
    //    truncating index and SegmentIntegral's midpoint rule.
    // -----------------------------------------------------------------------
    {
        auto process = bsmProcess();
        const Time t = 1.0;
        const Size n = 8;
        TimeGrid grid(t, n);
        Path path(grid);
        // A deliberately non-monotone path so an off-by-one index shows up.
        const Real levels[] = {95.0, 101.0, 88.0, 93.5, 110.0, 84.0, 99.0, 105.5, 91.0};
        for (Size k = 0; k <= n; ++k)
            path[k] = levels[k];

        std::vector<Real> pathVals(levels, levels + n + 1);

        // Under constant vol the answer is exactly sigma^2 whatever the path —
        // pinned anyway, because a port that gets the 1/t normalisation or the
        // integration bounds wrong fails even here.
        {
            VariancePathPricer pricer(process);
            Obj in;
            in.s("volKind", "constant");
            in.n("t", t);
            in.i("n", (long long)n);
            in.arr("path", pathVals);
            Obj ex;
            ex.n("value", pricer(path));
            addCase("vs_pathpricer_direct", in, ex);
        }
        // Under a term structure of vol the answer depends on WHERE the
        // integrand is sampled, so this is the case that pins
        // SegmentIntegral's midpoint rule composed with detail::Integrand's
        // truncating index.
        {
            VariancePathPricer pricer(bsmProcessTermVol());
            Obj in;
            in.s("volKind", "termStructure");
            in.n("t", t);
            in.i("n", (long long)n);
            in.arr("path", pathVals);
            Obj ex;
            ex.n("value", pricer(path));
            addCase("vs_pathpricer_direct_termvol", in, ex);
        }
    }

    // -----------------------------------------------------------------------
    // 6. Guard clauses and builder validation.
    // -----------------------------------------------------------------------
    {
        auto process = bsmProcess();

        // moneyness < 0 and moneyness == 0 both throw, but from
        // ForwardOptionArguments::validate() ("negative or zero moneyness
        // given"), NOT from ForwardEuropeanBSPathPricer's own
        // `QL_REQUIRE(moneyness >= 0.0)`. The arguments check is the stricter
        // of the two and runs first, so the path pricer's `>= 0.0` boundary is
        // dead code through the instrument. A port must reproduce the STRICT
        // check, or moneyness == 0 will price instead of throwing.
        for (Real m : {-0.5, 0.0}) {
            Obj in1;
            in1.n("moneyness", m);
            Obj ex1;
            ex1.b("throws", throws([&] {
                      auto option = fwdOption(Option::Call, m);
                      option->setPricingEngine(MakeMCForwardEuropeanBSEngine<PseudoRandom>(process)
                                                   .withSteps(8)
                                                   .withSamples(255)
                                                   .withSeed(42));
                      option->NPV();
                  }));
            addCase(m < 0.0 ? "fwdbs_negative_moneyness_throws" : "fwdbs_zero_moneyness_throws", in1,
                    ex1);
        }

        // Neither steps nor stepsPerYear -> "number of steps not given".
        Obj in2;
        in2.s("configuration", "no steps");
        Obj ex2;
        ex2.b("throws", throws([&] {
                  auto option = fwdOption(Option::Call, 1.0);
                  option->setPricingEngine(MakeMCForwardEuropeanBSEngine<PseudoRandom>(process)
                                               .withSamples(255)
                                               .withSeed(42));
              }));
        addCase("fwdbs_make_no_steps_throws", in2, ex2);

        // Both -> "number of steps overspecified".
        Obj in3;
        in3.s("configuration", "steps and stepsPerYear");
        Obj ex3;
        ex3.b("throws", throws([&] {
                  auto option = fwdOption(Option::Call, 1.0);
                  option->setPricingEngine(MakeMCForwardEuropeanBSEngine<PseudoRandom>(process)
                                               .withSteps(8)
                                               .withStepsPerYear(4)
                                               .withSamples(255)
                                               .withSeed(42));
              }));
        addCase("fwdbs_make_both_steps_throws", in3, ex3);

        // withSamples then withAbsoluteTolerance -> "number of samples already set".
        Obj in4;
        in4.s("configuration", "samples then tolerance");
        Obj ex4;
        ex4.b("throws", throws([&] {
                  MakeMCForwardEuropeanBSEngine<PseudoRandom>(process)
                      .withSteps(8)
                      .withSamples(255)
                      .withAbsoluteTolerance(0.01);
              }));
        addCase("fwdbs_make_samples_then_tolerance_throws", in4, ex4);

        // withAbsoluteTolerance then withSamples -> "tolerance already set".
        Obj in5;
        in5.s("configuration", "tolerance then samples");
        Obj ex5;
        ex5.b("throws", throws([&] {
                  MakeMCForwardEuropeanBSEngine<PseudoRandom>(process)
                      .withSteps(8)
                      .withAbsoluteTolerance(0.01)
                      .withSamples(255);
              }));
        addCase("fwdbs_make_tolerance_then_samples_throws", in5, ex5);

        // LowDiscrepancy + withAbsoluteTolerance -> "does not allow an error estimate".
        Obj in6;
        in6.s("configuration", "LowDiscrepancy + absolute tolerance");
        Obj ex6;
        ex6.b("throws", throws([&] {
                  MakeMCForwardEuropeanBSEngine<LowDiscrepancy>(process).withSteps(8).withAbsoluteTolerance(
                      0.01);
              }));
        addCase("fwdbs_make_ld_tolerance_throws", in6, ex6);

        // Same four for the variance-swap builder.
        Obj in7;
        in7.s("configuration", "no steps");
        Obj ex7;
        ex7.b("throws", throws([&] {
                  VarianceSwap swap(Position::Long, 0.04, 1000000.0, TODAY, EXERCISE);
                  swap.setPricingEngine(
                      MakeMCVarianceSwapEngine<PseudoRandom>(process).withSamples(255).withSeed(42));
              }));
        addCase("vs_make_no_steps_throws", in7, ex7);

        Obj in8;
        in8.s("configuration", "steps and stepsPerYear");
        Obj ex8;
        ex8.b("throws", throws([&] {
                  VarianceSwap swap(Position::Long, 0.04, 1000000.0, TODAY, EXERCISE);
                  swap.setPricingEngine(MakeMCVarianceSwapEngine<PseudoRandom>(process)
                                            .withSteps(8)
                                            .withStepsPerYear(4)
                                            .withSamples(255)
                                            .withSeed(42));
              }));
        addCase("vs_make_both_steps_throws", in8, ex8);

        // MCVarianceSwapEngine hardwires controlVariate = false and the builder
        // exposes no withControlVariate at all. Pinned as documentation so a
        // port does not invent the knob.
        Obj in9;
        in9.s("api", "MakeMCVarianceSwapEngine");
        Obj ex9;
        ex9.b("hasWithControlVariate", false);
        addCase("vs_make_has_no_control_variate", in9, ex9);

        // MCForwardEuropeanBSEngine's own ctor takes no controlVariate either;
        // only MCForwardVanillaEngine (the base) and the Heston subclass do.
        Obj in10;
        in10.s("api", "MakeMCForwardEuropeanBSEngine");
        Obj ex10;
        ex10.b("hasWithControlVariate", false);
        addCase("fwdbs_make_has_no_control_variate", in10, ex10);
    }

    // -----------------------------------------------------------------------
    // 7. The control-variate value itself, computed the way
    //    MCForwardVanillaEngine::controlVariateValue() does it: a plain vanilla
    //    with strike = moneyness * spot priced by AnalyticEuropeanEngine.
    //    Pinned separately so a control-variate failure localises.
    // -----------------------------------------------------------------------
    {
        auto process = bsmProcess();
        for (Real moneyness : {0.85, 1.0, 1.25}) {
            VanillaOption control(
                ext::make_shared<PlainVanillaPayoff>(Option::Call, moneyness * SPOT),
                ext::make_shared<EuropeanExercise>(EXERCISE));
            control.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));
            Obj in;
            in.n("moneyness", moneyness);
            in.n("spot", SPOT);
            in.n("strike", moneyness * SPOT);
            Obj ex;
            ex.n("controlVariateValue", control.NPV());
            std::ostringstream nm;
            nm << "fwd_control_variate_value_m" << std::setprecision(3) << moneyness;
            addCase(nm.str(), in, ex);
        }
    }

    emitDocument();
    return 0;
}
