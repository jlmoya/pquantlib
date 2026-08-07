// migration-harness/cpp/probes/v143_pe_mcfwdlb/probe.cpp
//
// Reference values for the Monte Carlo LOOKBACK and PERFORMANCE (cliquet)
// pricing engines of C++ QuantLib v1.43, plus the parts of the MC
// forward-start cluster that v143_pe_mcforward/probe.cpp does NOT already pin:
//
//   * MCLookbackEngine                  (ql/pricingengines/lookback/mclookbackengine.hpp)
//   * MakeMCLookbackEngine              (same)
//   * detail::mc_lookback_path_pricer   (same; 4 overloads)
//   * LookbackFixedPathPricer           (ql/pricingengines/lookback/mclookbackengine.cpp)
//   * LookbackPartialFixedPathPricer    (same)
//   * LookbackFloatingPathPricer        (same)
//   * LookbackPartialFloatingPathPricer (same)
//   * MCPerformanceEngine               (ql/pricingengines/cliquet/mcperformanceengine.hpp)
//   * PerformanceOptionPathPricer       (ql/pricingengines/cliquet/mcperformanceengine.{hpp,cpp})
//   * MakeMCPerformanceEngine           (same)
//   * MCForwardVanillaEngine            (ql/pricingengines/forward/mcforwardvanillaengine.hpp)
//         -- constructor guards only; the priced sweep lives in mcforward.json
//   * ForwardEuropeanBSPathPricer       (ql/pricingengines/forward/mcforwardeuropeanbsengine.{hpp,cpp})
//         -- direct, on a hand-built Path, and its own moneyness guard
//   * ForwardEuropeanHestonPathPricer   (ql/pricingengines/forward/mcforwardeuropeanhestonengine.{hpp,cpp})
//         -- direct, on a hand-built MultiPath, and its own moneyness guard
//   * MakeMCForwardEuropeanHestonEngine (same) -- builder validation
//
// Deliberately NOT duplicated here: everything already pinned by
// migration-harness/references/v143/pe/mcforward.json -- the priced sweeps of
// MCForwardEuropeanBSEngine / MCForwardEuropeanHestonEngine / MCVarianceSwapEngine,
// the mandatory-point forward time grid, VariancePathPricer + detail::Integrand,
// MakeMCForwardEuropeanBSEngine / MakeMCVarianceSwapEngine validation, and the
// control-variate reference values. This probe uses the SAME evaluation date
// (15 June 2023) and the SAME market as that one so the two reference files can
// be consumed by one test module with one evaluation-date fixture.
//
// Why the values below are EXACT, not a confidence band
// -----------------------------------------------------
// Every engine here is deterministic once the seed is fixed:
//   PseudoRandom   = InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
//                                         InverseCumulativeNormal>
//   LowDiscrepancy = InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>
// Neither consults the clock while `seed != 0` (seed 0 routes through
// SeedGenerator, which IS clock-derived; no priced case below uses seed 0).
// A correct port reproduces both the NPV and the error estimate to ~1e-14
// relative. A statistical band would pass with the wrong RNG, the wrong path
// construction, the wrong time grid or the wrong antithetic pairing, so nothing
// here is banded.
//
// What a port gets wrong, and which case forces it
// ------------------------------------------------
//  1. MCLookbackEngine::timeGrid() is a PLAIN UNIFORM grid over [0, T]:
//         TimeGrid(residualTime, timeSteps)
//     or  TimeGrid(residualTime, max<Size>(Size(timeStepsPerYear*residualTime), 1)).
//     Unlike the forward-start engines there are NO mandatory points -- in
//     particular the partial-lookback window boundary is NOT forced onto a node.
//     `lb_timegrid_*` pins the grid and the closestIndex of both window
//     boundaries so a failure localises before any price is compared.
//  2. Every lookback path pricer scans `path.begin()+1`, i.e. it EXCLUDES the
//     t=0 spot from the running extremum. A port that includes path[0] prices a
//     different (and, for a floating call, systematically cheaper) option. The
//     `lb_pathpricer_*` cases run on a hand-built path whose FIRST point is the
//     global extremum, so including it changes the answer by a wide margin.
//  3. LookbackPartialFixedPathPricer scans `[begin+startIndex+1, end)` with
//     startIndex = timeGrid.closestIndex(lookbackStart) -- closest, not floor.
//     LookbackPartialFloatingPathPricer scans `[begin+1, begin+endIndex+1)`,
//     i.e. INCLUSIVE of the node at endIndex. The two are not mirror images of
//     each other and a port that makes them symmetric fails one of them.
//  4. The MC lookback engines IGNORE two instrument fields that the analytic
//     engines use: `minmax` (the seasoned running extremum) and, for the partial
//     floating variant, `lambda`. That is real C++ behaviour, not an oversight of
//     this probe: `LookbackFixedPathPricer` / `LookbackFloatingPathPricer` never
//     see minmax, and `LookbackPartialFloatingPathPricer` never sees lambda.
//     `lb_*_ignored` cases pin two different values of each producing the SAME
//     price, so a port that "helpfully" wires them in is caught.
//  5. Degenerate corners: a partial-fixed lookback whose window starts at t=0
//     equals the full-window fixed lookback, and a partial-floating lookback
//     whose window ends at expiry equals the full-window floating lookback.
//     Pinned as equalities between named cases.
//  6. MCLookbackEngine::calculate() checks `spot > 0` BEFORE running the
//     simulation ("negative or null underlying given").
//  7. PerformanceOptionPathPricer sums `discounts_[i-1] * payoff(path[i]/path[i-1])`
//     for i in [2, n) -- note it starts at TWO, so the first period (grid node 0
//     to grid node 1) is NOT paid, and each period is discounted at the period's
//     END date. It also requires `path.length() == discounts.size()+1` exactly.
//  8. MCPerformanceEngine::timeGrid() is TimeGrid(fixingTimes.begin(), end) with
//     fixingTimes = resetDates + exercise date and NO step count: the grid is the
//     fixing schedule itself, plus a leading 0 inserted only when the first
//     fixing time is > 0. So a schedule whose FIRST reset is the evaluation date
//     produces a grid one point SHORT of what the discount vector expects, and
//     the engine throws "discounts/options mismatch" instead of pricing.
//     `perf_first_reset_today_throws` pins that.
//  9. MakeMCPerformanceEngine has NO steps knobs at all and its
//     `operator shared_ptr<PricingEngine>()` performs NO validation -- unlike
//     every other MakeMC* builder in this wave. Pinned as
//     `perf_make_no_steps_knobs`.
// 10. MCForwardVanillaEngine's constructor carries the four steps guards
//     (neither / both / zero / zero-per-year); MCForwardEuropeanBSEngine and
//     MCForwardEuropeanHestonEngine both inherit them.
//     MakeMCForwardEuropeanHestonEngine's validation is the same shape as the BS
//     one, which mcforward.json pins only for BS.
// 11. ForwardEuropeanBSPathPricer / ForwardEuropeanHestonPathPricer both carry
//     `QL_REQUIRE(moneyness >= 0.0)`. Through the instrument that guard is dead
//     code (ForwardOptionArguments::validate is stricter: `> 0.0`), which
//     mcforward.json pins -- but constructed DIRECTLY the guard is live, and
//     `moneyness == 0` is then ACCEPTED. Both pinned here.
//
// Evaluation date
// ---------------
// Settings::instance().evaluationDate() is set ONCE, in main(), to
// Date(15, June, 2023). Every reference here depends on it. The Python test
// MUST pin the same date in a fixture and restore it in teardown.
//
// Emits JSON on stdout; nothing else may be printed.

#include <algorithm>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/cliquetoption.hpp>
#include <ql/instruments/forwardvanillaoption.hpp>
#include <ql/instruments/lookbackoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/math/statistics/generalstatistics.hpp>
#include <ql/methods/montecarlo/multipath.hpp>
#include <ql/methods/montecarlo/path.hpp>
#include <ql/methods/montecarlo/pathgenerator.hpp>
#include <ql/pricingengines/cliquet/mcperformanceengine.hpp>
#include <ql/pricingengines/forward/mcforwardeuropeanbsengine.hpp>
#include <ql/pricingengines/forward/mcforwardeuropeanhestonengine.hpp>
#include <ql/pricingengines/forward/mcvarianceswapengine.hpp>
#include <ql/pricingengines/lookback/mclookbackengine.hpp>
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

std::string quote(const std::string& v) {
    std::string out = "\"";
    for (char c : v) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out + "\"";
}

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, quote(v)); }
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
    Obj& iarr(const std::string& k, const std::vector<long long>& v) {
        std::string body = "[";
        for (Size j = 0; j < v.size(); ++j) {
            if (j != 0)
                body += ", ";
            body += std::to_string(v[j]);
        }
        return put(k, body + "]");
    }
    std::string str() const { return "{" + body_ + "}"; }

  private:
    Obj& put(const std::string& k, const std::string& v) {
        if (!body_.empty())
            body_ += ", ";
        body_ += quote(k) + ": " + v;
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
// Market. Identical to v143_pe_mcforward/probe.cpp so both reference files can
// live under one evaluation-date fixture on the Python side.
// ---------------------------------------------------------------------------
const Date TODAY(15, June, 2023);
const Rate RISK_FREE = 0.04;
const Rate DIVIDEND = 0.015;
const Volatility VOL = 0.22;
const Real SPOT = 95.0;

const Date RESET(16, October, 2023);      // forward-start strike reset
const Date EXERCISE(18, September, 2024); // expiry for every instrument here

// Partial-lookback window boundaries, both deliberately OFF any uniform node of
// the 52-step grid so `closestIndex` is exercised rather than trivially exact.
const Date LB_START(15, December, 2023); // partial-FIXED window starts here
const Date LB_END(15, March, 2024);      // partial-FLOATING window ends here

DayCounter dc() { return Actual365Fixed(); }

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(TODAY, r, dc()));
}

ext::shared_ptr<GeneralizedBlackScholesProcess> bsmProcess(Real spot = SPOT) {
    return ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(spot)), flatCurve(DIVIDEND), flatCurve(RISK_FREE),
        Handle<BlackVolTermStructure>(
            ext::make_shared<BlackConstantVol>(TODAY, NullCalendar(), VOL, dc())));
}

// The term structure of volatility used by mcforward.json's vs_termvol_* cases,
// repeated verbatim here so the sample-spread diagnostic below runs on exactly
// the process those cases price with.
// v143_pe_mcforward/probe.cpp:250-258.
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

// Instrument::errorEstimate() THROWS ("error estimate not provided") when the
// engine left it Null -- which is what LowDiscrepancy does, since Sobol has
// allowsErrorEstimate == false and the engines guard the assignment with
// `if constexpr (RNG::allowsErrorEstimate)`. The absence is only observable as
// an exception, so that is what gets pinned.
Real errorEstimateOrNull(const Instrument& inst) {
    try {
        return inst.errorEstimate();
    } catch (const std::exception&) {
        return Null<Real>();
    }
}

bool throws(const std::function<void()>& f) {
    try {
        f();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

std::string whatOf(const std::function<void()>& f) {
    try {
        f();
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

// Every throwing case carries both the boolean and the message, so a port
// cannot satisfy the test by throwing for an unrelated reason.
void addThrowCase(const std::string& name,
                  const std::string& scenario,
                  const std::function<void()>& f) {
    Obj in;
    in.s("scenario", scenario);
    Obj ex;
    const std::string what = whatOf(f);
    ex.b("throws", !what.empty());
    ex.s("what", what);
    addCase(name, in, ex);
}

void describeMarket(Obj& in) {
    in.n("spot", SPOT);
    in.n("riskFreeRate", RISK_FREE);
    in.n("dividendYield", DIVIDEND);
    in.n("volatility", VOL);
    in.s("dayCounter", "Actual365Fixed");
    in.s("calendar", "NullCalendar");
    in.i("evaluationDateSerial", (long long)TODAY.serialNumber());
    in.i("exerciseDateSerial", (long long)EXERCISE.serialNumber());
}

void describeEngine(Obj& in,
                    const std::string& rng,
                    long long steps,
                    long long stepsPerYear,
                    bool brownianBridge,
                    bool antithetic,
                    long long samples,
                    long long seed) {
    in.s("rng", rng);
    in.i("steps", steps);                 // -1 == Null<Size>()
    in.i("stepsPerYear", stepsPerYear);   // -1 == Null<Size>()
    in.b("brownianBridge", brownianBridge);
    in.b("antitheticVariate", antithetic);
    in.i("samples", samples);
    in.i("seed", seed);
}

// ---------------------------------------------------------------------------
// Lookback runners. One per instrument type, because MCLookbackEngine is
// templated on the INSTRUMENT and each carries different arguments.
// ---------------------------------------------------------------------------

template <class I, class RNG>
ext::shared_ptr<PricingEngine> lookbackEngine(
        const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
        Size steps, Size stepsPerYear, bool brownianBridge, bool antithetic,
        Size samples, BigNatural seed) {
    MakeMCLookbackEngine<I, RNG> make(process);
    if (steps != Null<Size>())
        make.withSteps(steps);
    else
        make.withStepsPerYear(stepsPerYear);
    make.withBrownianBridge(brownianBridge)
        .withAntitheticVariate(antithetic)
        .withSamples(samples)
        .withSeed(seed);
    return make;
}

void recordOption(Obj& ex, const Instrument& inst) {
    ex.n("npv", inst.NPV());
    const Real err = errorEstimateOrNull(inst);
    ex.b("hasErrorEstimate", err != Null<Real>());
    ex.n("errorEstimate", err);
}

template <class RNG>
void runFixedLookback(const std::string& name, const std::string& rngName, Option::Type type,
                      Real strike, Real minmax, Size steps, Size stepsPerYear,
                      bool brownianBridge, bool antithetic, Size samples, BigNatural seed) {
    auto process = bsmProcess();
    ContinuousFixedLookbackOption option(
        minmax, ext::make_shared<PlainVanillaPayoff>(type, strike),
        ext::make_shared<EuropeanExercise>(EXERCISE));
    option.setPricingEngine(
        lookbackEngine<ContinuousFixedLookbackOption, RNG>(process, steps, stepsPerYear,
                                                           brownianBridge, antithetic, samples,
                                                           seed));
    Obj in;
    in.s("instrument", "ContinuousFixedLookbackOption");
    in.s("engine", "MCLookbackEngine");
    describeMarket(in);
    in.s("optionType", type == Option::Call ? "Call" : "Put");
    in.s("payoff", "PlainVanilla");
    in.n("strike", strike);
    in.n("minmax", minmax);
    describeEngine(in, rngName, steps == Null<Size>() ? -1 : (long long)steps,
                   stepsPerYear == Null<Size>() ? -1 : (long long)stepsPerYear, brownianBridge,
                   antithetic, (long long)samples, (long long)seed);
    Obj ex;
    recordOption(ex, option);
    addCase(name, in, ex);
}

template <class RNG>
void runFloatingLookback(const std::string& name, const std::string& rngName, Option::Type type,
                         Real minmax, Size steps, Size stepsPerYear, bool brownianBridge,
                         bool antithetic, Size samples, BigNatural seed) {
    auto process = bsmProcess();
    ContinuousFloatingLookbackOption option(minmax, ext::make_shared<FloatingTypePayoff>(type),
                                            ext::make_shared<EuropeanExercise>(EXERCISE));
    option.setPricingEngine(
        lookbackEngine<ContinuousFloatingLookbackOption, RNG>(process, steps, stepsPerYear,
                                                              brownianBridge, antithetic, samples,
                                                              seed));
    Obj in;
    in.s("instrument", "ContinuousFloatingLookbackOption");
    in.s("engine", "MCLookbackEngine");
    describeMarket(in);
    in.s("optionType", type == Option::Call ? "Call" : "Put");
    in.s("payoff", "FloatingType");
    in.n("minmax", minmax);
    describeEngine(in, rngName, steps == Null<Size>() ? -1 : (long long)steps,
                   stepsPerYear == Null<Size>() ? -1 : (long long)stepsPerYear, brownianBridge,
                   antithetic, (long long)samples, (long long)seed);
    Obj ex;
    recordOption(ex, option);
    addCase(name, in, ex);
}

template <class RNG>
void runPartialFixedLookback(const std::string& name, const std::string& rngName,
                             Option::Type type, Real strike, const Date& lookbackStart, Size steps,
                             Size stepsPerYear, bool brownianBridge, bool antithetic, Size samples,
                             BigNatural seed) {
    auto process = bsmProcess();
    ContinuousPartialFixedLookbackOption option(
        lookbackStart, ext::make_shared<PlainVanillaPayoff>(type, strike),
        ext::make_shared<EuropeanExercise>(EXERCISE));
    option.setPricingEngine(
        lookbackEngine<ContinuousPartialFixedLookbackOption, RNG>(process, steps, stepsPerYear,
                                                                  brownianBridge, antithetic,
                                                                  samples, seed));
    Obj in;
    in.s("instrument", "ContinuousPartialFixedLookbackOption");
    in.s("engine", "MCLookbackEngine");
    describeMarket(in);
    in.s("optionType", type == Option::Call ? "Call" : "Put");
    in.s("payoff", "PlainVanilla");
    in.n("strike", strike);
    in.i("lookbackPeriodStartSerial", (long long)lookbackStart.serialNumber());
    describeEngine(in, rngName, steps == Null<Size>() ? -1 : (long long)steps,
                   stepsPerYear == Null<Size>() ? -1 : (long long)stepsPerYear, brownianBridge,
                   antithetic, (long long)samples, (long long)seed);
    Obj ex;
    recordOption(ex, option);
    addCase(name, in, ex);
}

template <class RNG>
void runPartialFloatingLookback(const std::string& name, const std::string& rngName,
                                Option::Type type, Real minmax, Real lambda,
                                const Date& lookbackEnd, Size steps, Size stepsPerYear,
                                bool brownianBridge, bool antithetic, Size samples,
                                BigNatural seed) {
    auto process = bsmProcess();
    ContinuousPartialFloatingLookbackOption option(minmax, lambda, lookbackEnd,
                                                   ext::make_shared<FloatingTypePayoff>(type),
                                                   ext::make_shared<EuropeanExercise>(EXERCISE));
    option.setPricingEngine(
        lookbackEngine<ContinuousPartialFloatingLookbackOption, RNG>(process, steps, stepsPerYear,
                                                                     brownianBridge, antithetic,
                                                                     samples, seed));
    Obj in;
    in.s("instrument", "ContinuousPartialFloatingLookbackOption");
    in.s("engine", "MCLookbackEngine");
    describeMarket(in);
    in.s("optionType", type == Option::Call ? "Call" : "Put");
    in.s("payoff", "FloatingType");
    in.n("minmax", minmax);
    in.n("lambda", lambda);
    in.i("lookbackPeriodEndSerial", (long long)lookbackEnd.serialNumber());
    describeEngine(in, rngName, steps == Null<Size>() ? -1 : (long long)steps,
                   stepsPerYear == Null<Size>() ? -1 : (long long)stepsPerYear, brownianBridge,
                   antithetic, (long long)samples, (long long)seed);
    Obj ex;
    recordOption(ex, option);
    addCase(name, in, ex);
}

// ---------------------------------------------------------------------------
// Performance (cliquet) runner.
// ---------------------------------------------------------------------------

std::vector<Date> resetSchedule() {
    // Four quarterly resets, none of them the evaluation date (see
    // perf_first_reset_today_throws for why that matters).
    return {Date(15, September, 2023), Date(15, December, 2023), Date(15, March, 2024),
            Date(17, June, 2024)};
}

template <class RNG>
void runPerformance(const std::string& name, const std::string& rngName, Option::Type type,
                    Real moneyness, const std::vector<Date>& resets, bool brownianBridge,
                    bool antithetic, Size samples, BigNatural seed) {
    auto process = bsmProcess();
    CliquetOption option(ext::make_shared<PercentageStrikePayoff>(type, moneyness),
                         ext::make_shared<EuropeanExercise>(EXERCISE), resets);
    option.setPricingEngine(MakeMCPerformanceEngine<RNG>(process)
                                .withBrownianBridge(brownianBridge)
                                .withAntitheticVariate(antithetic)
                                .withSamples(samples)
                                .withSeed(seed));
    Obj in;
    in.s("instrument", "CliquetOption");
    in.s("engine", "MCPerformanceEngine");
    describeMarket(in);
    in.s("optionType", type == Option::Call ? "Call" : "Put");
    in.s("payoff", "PercentageStrike");
    in.n("moneyness", moneyness);
    std::vector<long long> serials;
    serials.reserve(resets.size());
    for (const auto& d : resets)
        serials.push_back((long long)d.serialNumber());
    in.iarr("resetDateSerials", serials);
    describeEngine(in, rngName, -1, -1, brownianBridge, antithetic, (long long)samples,
                   (long long)seed);
    Obj ex;
    recordOption(ex, option);
    addCase(name, in, ex);
}

// A deliberately non-monotone path whose FIRST point is both the global maximum
// and (after the first point) far from the running extremum, so that a port
// which scans from path.begin() instead of path.begin()+1 gets a different
// answer for every lookback pricer below.
const Real kPathLevels[] = {150.0, 101.0, 88.0, 93.5, 130.0, 74.0, 99.0, 118.5, 91.0};
const Size kPathSteps = 8;

Path buildPath() {
    TimeGrid grid(1.0, kPathSteps);
    Path path(grid);
    for (Size k = 0; k <= kPathSteps; ++k)
        path[k] = kPathLevels[k];
    return path;
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = TODAY;

    // =======================================================================
    // 0. Market + derived times, so a Python failure localises to "wrong
    //    market" rather than "wrong engine".
    // =======================================================================
    {
        auto process = bsmProcess();
        Obj in;
        describeMarket(in);
        in.i("resetDateSerial", (long long)RESET.serialNumber());
        in.i("lookbackPeriodStartSerial", (long long)LB_START.serialNumber());
        in.i("lookbackPeriodEndSerial", (long long)LB_END.serialNumber());
        Obj ex;
        ex.n("exerciseTime", process->time(EXERCISE));
        ex.n("resetTime", process->time(RESET));
        ex.n("lookbackStartTime", process->time(LB_START));
        ex.n("lookbackEndTime", process->time(LB_END));
        ex.n("discountToExerciseTime",
             process->riskFreeRate()->discount(process->time(EXERCISE)));
        ex.n("x0", process->x0());
        addCase("market", in, ex);
    }

    // =======================================================================
    // 1. MCLookbackEngine::timeGrid() -- a PLAIN uniform grid, with no
    //    mandatory points. Pinned together with the closestIndex of both
    //    partial-window boundaries, which is what selects the scan range.
    // =======================================================================
    {
        auto process = bsmProcess();
        const Time residual = process->time(EXERCISE);
        const Time tStart = process->time(LB_START);
        const Time tEnd = process->time(LB_END);
        struct GridCfg {
            const char* name;
            long long steps;
            long long stepsPerYear;
        };
        const GridCfg cfgs[] = {{"lb_timegrid_steps52", 52, -1},
                                {"lb_timegrid_steps13", 13, -1},
                                {"lb_timegrid_stepsperyear20", -1, 20},
                                // Size(1 * 1.2575...) == 1 -> the max(steps,1)
                                // clamp is one step away from firing.
                                {"lb_timegrid_stepsperyear1", -1, 1}};
        for (const auto& c : cfgs) {
            TimeGrid grid = c.steps > 0
                                ? TimeGrid(residual, Size(c.steps))
                                : TimeGrid(residual, std::max<Size>(
                                                         Size(c.stepsPerYear * residual), 1));
            std::vector<Real> times(grid.begin(), grid.end());
            Obj in;
            in.n("residualTime", residual);
            in.i("steps", c.steps);
            in.i("stepsPerYear", c.stepsPerYear);
            Obj ex;
            ex.i("size", (long long)grid.size());
            ex.arr("times", times);
            ex.n("back", grid.back());
            ex.i("lookbackStartClosestIndex", (long long)grid.closestIndex(tStart));
            ex.i("lookbackEndClosestIndex", (long long)grid.closestIndex(tEnd));
            ex.n("discountAtBack",
                 bsmProcess()->riskFreeRate()->discount(grid.back()));
            addCase(c.name, in, ex);
        }
    }

    // =======================================================================
    // 2. MCLookbackEngine over ContinuousFixedLookbackOption.
    // =======================================================================
    runFixedLookback<PseudoRandom>("lb_fixed_call_steps52_pr", "PseudoRandom", Option::Call, 95.0,
                                   95.0, 52, Null<Size>(), false, false, 1023, 42);
    runFixedLookback<PseudoRandom>("lb_fixed_put_steps52_pr", "PseudoRandom", Option::Put, 95.0,
                                   95.0, 52, Null<Size>(), false, false, 1023, 42);
    // Same everything, different seed: proves the seed reaches the generator.
    runFixedLookback<PseudoRandom>("lb_fixed_call_steps52_pr_seed7", "PseudoRandom", Option::Call,
                                   95.0, 95.0, 52, Null<Size>(), false, false, 1023, 7);
    runFixedLookback<PseudoRandom>("lb_fixed_call_steps52_pr_anti", "PseudoRandom", Option::Call,
                                   95.0, 95.0, 52, Null<Size>(), false, true, 1023, 42);
    runFixedLookback<PseudoRandom>("lb_fixed_call_steps52_pr_bb", "PseudoRandom", Option::Call,
                                   95.0, 95.0, 52, Null<Size>(), true, false, 1023, 42);
    runFixedLookback<PseudoRandom>("lb_fixed_call_stepsperyear20_pr", "PseudoRandom", Option::Call,
                                   95.0, 95.0, Null<Size>(), 20, false, false, 1023, 42);
    runFixedLookback<PseudoRandom>("lb_fixed_call_otm_steps52_pr", "PseudoRandom", Option::Call,
                                   130.0, 95.0, 52, Null<Size>(), false, false, 1023, 42);
    // minmax = 400 instead of 95: the MC engine never reads it, so this MUST
    // equal lb_fixed_call_steps52_pr to the last bit.
    runFixedLookback<PseudoRandom>("lb_fixed_call_steps52_pr_minmax_ignored", "PseudoRandom",
                                   Option::Call, 95.0, 400.0, 52, Null<Size>(), false, false, 1023,
                                   42);
    runFixedLookback<LowDiscrepancy>("lb_fixed_call_steps52_ld", "LowDiscrepancy", Option::Call,
                                     95.0, 95.0, 52, Null<Size>(), false, false, 1024, 42);

    // =======================================================================
    // 3. MCLookbackEngine over ContinuousFloatingLookbackOption.
    // =======================================================================
    runFloatingLookback<PseudoRandom>("lb_floating_call_steps52_pr", "PseudoRandom", Option::Call,
                                      95.0, 52, Null<Size>(), false, false, 1023, 42);
    runFloatingLookback<PseudoRandom>("lb_floating_put_steps52_pr", "PseudoRandom", Option::Put,
                                      95.0, 52, Null<Size>(), false, false, 1023, 42);
    runFloatingLookback<PseudoRandom>("lb_floating_call_steps52_pr_anti", "PseudoRandom",
                                      Option::Call, 95.0, 52, Null<Size>(), false, true, 1023, 42);
    runFloatingLookback<PseudoRandom>("lb_floating_call_steps52_pr_bb", "PseudoRandom",
                                      Option::Call, 95.0, 52, Null<Size>(), true, false, 1023, 42);
    // Again: minmax is not read by the MC pricer.
    runFloatingLookback<PseudoRandom>("lb_floating_call_steps52_pr_minmax_ignored", "PseudoRandom",
                                      Option::Call, 400.0, 52, Null<Size>(), false, false, 1023,
                                      42);
    runFloatingLookback<LowDiscrepancy>("lb_floating_call_steps52_ld", "LowDiscrepancy",
                                        Option::Call, 95.0, 52, Null<Size>(), false, false, 1024,
                                        42);

    // =======================================================================
    // 4. MCLookbackEngine over ContinuousPartialFixedLookbackOption.
    // =======================================================================
    runPartialFixedLookback<PseudoRandom>("lb_partfixed_call_steps52_pr", "PseudoRandom",
                                          Option::Call, 95.0, LB_START, 52, Null<Size>(), false,
                                          false, 1023, 42);
    runPartialFixedLookback<PseudoRandom>("lb_partfixed_put_steps52_pr", "PseudoRandom",
                                          Option::Put, 95.0, LB_START, 52, Null<Size>(), false,
                                          false, 1023, 42);
    runPartialFixedLookback<PseudoRandom>("lb_partfixed_call_steps52_pr_anti", "PseudoRandom",
                                          Option::Call, 95.0, LB_START, 52, Null<Size>(), false,
                                          true, 1023, 42);
    // Window starting at t=0 -> startIndex == 0 -> scan range identical to the
    // full-window fixed lookback. MUST equal lb_fixed_call_steps52_pr exactly.
    runPartialFixedLookback<PseudoRandom>("lb_partfixed_call_start_today_pr", "PseudoRandom",
                                          Option::Call, 95.0, TODAY, 52, Null<Size>(), false,
                                          false, 1023, 42);
    runPartialFixedLookback<LowDiscrepancy>("lb_partfixed_call_steps52_ld", "LowDiscrepancy",
                                            Option::Call, 95.0, LB_START, 52, Null<Size>(), false,
                                            false, 1024, 42);

    // =======================================================================
    // 5. MCLookbackEngine over ContinuousPartialFloatingLookbackOption.
    // =======================================================================
    runPartialFloatingLookback<PseudoRandom>("lb_partfloating_call_steps52_pr", "PseudoRandom",
                                             Option::Call, 95.0, 1.0, LB_END, 52, Null<Size>(),
                                             false, false, 1023, 42);
    runPartialFloatingLookback<PseudoRandom>("lb_partfloating_put_steps52_pr", "PseudoRandom",
                                             Option::Put, 95.0, 1.0, LB_END, 52, Null<Size>(),
                                             false, false, 1023, 42);
    runPartialFloatingLookback<PseudoRandom>("lb_partfloating_call_steps52_pr_anti", "PseudoRandom",
                                             Option::Call, 95.0, 1.0, LB_END, 52, Null<Size>(),
                                             false, true, 1023, 42);
    // lambda = 1.35 instead of 1.0: never read by the MC pricer, so this MUST
    // equal lb_partfloating_call_steps52_pr exactly.
    runPartialFloatingLookback<PseudoRandom>("lb_partfloating_call_lambda_ignored", "PseudoRandom",
                                             Option::Call, 95.0, 1.35, LB_END, 52, Null<Size>(),
                                             false, false, 1023, 42);
    // Window ending at expiry -> endIndex == last -> scan range identical to the
    // full-window floating lookback. MUST equal lb_floating_call_steps52_pr.
    runPartialFloatingLookback<PseudoRandom>("lb_partfloating_call_end_at_expiry_pr", "PseudoRandom",
                                             Option::Call, 95.0, 1.0, EXERCISE, 52, Null<Size>(),
                                             false, false, 1023, 42);
    runPartialFloatingLookback<LowDiscrepancy>("lb_partfloating_call_steps52_ld", "LowDiscrepancy",
                                               Option::Call, 95.0, 1.0, LB_END, 52, Null<Size>(),
                                               false, false, 1024, 42);

    // =======================================================================
    // 6. The four lookback path pricers, DIRECTLY, on a hand-built path.
    //    They are private to mclookbackengine.cpp, so they are reached through
    //    the four detail::mc_lookback_path_pricer overloads declared in the
    //    header -- which also pins the overload set itself.
    // =======================================================================
    {
        auto process = bsmProcess();
        const Path path = buildPath();
        const DiscountFactor discount = 0.9375;  // arbitrary, pinned in inputs
        std::vector<Real> pathVals(kPathLevels, kPathLevels + kPathSteps + 1);
        auto exercise = ext::make_shared<EuropeanExercise>(EXERCISE);

        auto describePath = [&](Obj& in) {
            in.arr("path", pathVals);
            in.n("t", 1.0);
            in.i("n", (long long)kPathSteps);
            in.n("discount", discount);
        };

        // --- fixed --------------------------------------------------------
        for (Option::Type type : {Option::Call, Option::Put}) {
            ContinuousFixedLookbackOption::arguments args;
            args.payoff = ext::make_shared<PlainVanillaPayoff>(type, 100.0);
            args.exercise = exercise;
            args.minmax = 95.0;
            auto pricer = detail::mc_lookback_path_pricer(args, *process, discount);
            Obj in;
            in.s("pricer", "LookbackFixedPathPricer");
            in.s("optionType", type == Option::Call ? "Call" : "Put");
            in.n("strike", 100.0);
            in.n("minmax", 95.0);
            describePath(in);
            Obj ex;
            ex.n("value", (*pricer)(path));
            addCase(type == Option::Call ? "lb_pathpricer_fixed_call"
                                         : "lb_pathpricer_fixed_put",
                    in, ex);
        }

        // --- partial fixed: window start at 0.5 (closestIndex == 4 on this
        //     8-step unit grid), so the scan starts at index 5.
        for (Option::Type type : {Option::Call, Option::Put}) {
            ContinuousPartialFixedLookbackOption::arguments args;
            args.payoff = ext::make_shared<PlainVanillaPayoff>(type, 100.0);
            args.exercise = exercise;
            args.minmax = 0.0;
            args.lookbackPeriodStart = LB_START;
            auto pricer = detail::mc_lookback_path_pricer(args, *process, discount);
            Obj in;
            in.s("pricer", "LookbackPartialFixedPathPricer");
            in.s("optionType", type == Option::Call ? "Call" : "Put");
            in.n("strike", 100.0);
            in.i("lookbackPeriodStartSerial", (long long)LB_START.serialNumber());
            in.n("lookbackStartTime", process->time(LB_START));
            describePath(in);
            Obj ex;
            ex.i("startIndex", (long long)path.timeGrid().closestIndex(process->time(LB_START)));
            ex.n("value", (*pricer)(path));
            addCase(type == Option::Call ? "lb_pathpricer_partfixed_call"
                                         : "lb_pathpricer_partfixed_put",
                    in, ex);
        }

        // --- floating -----------------------------------------------------
        for (Option::Type type : {Option::Call, Option::Put}) {
            ContinuousFloatingLookbackOption::arguments args;
            args.payoff = ext::make_shared<FloatingTypePayoff>(type);
            args.exercise = exercise;
            args.minmax = 95.0;
            auto pricer = detail::mc_lookback_path_pricer(args, *process, discount);
            Obj in;
            in.s("pricer", "LookbackFloatingPathPricer");
            in.s("optionType", type == Option::Call ? "Call" : "Put");
            in.n("minmax", 95.0);
            describePath(in);
            Obj ex;
            ex.n("value", (*pricer)(path));
            addCase(type == Option::Call ? "lb_pathpricer_floating_call"
                                         : "lb_pathpricer_floating_put",
                    in, ex);
        }

        // --- partial floating: scan is [1, endIndex] INCLUSIVE, unlike the
        //     partial-fixed one which starts at startIndex+1.
        for (Option::Type type : {Option::Call, Option::Put}) {
            ContinuousPartialFloatingLookbackOption::arguments args;
            args.payoff = ext::make_shared<FloatingTypePayoff>(type);
            args.exercise = exercise;
            args.minmax = 95.0;
            args.lambda = type == Option::Call ? 1.0 : 1.0;
            args.lookbackPeriodEnd = LB_END;
            auto pricer = detail::mc_lookback_path_pricer(args, *process, discount);
            Obj in;
            in.s("pricer", "LookbackPartialFloatingPathPricer");
            in.s("optionType", type == Option::Call ? "Call" : "Put");
            in.n("minmax", 95.0);
            in.n("lambda", args.lambda);
            in.i("lookbackPeriodEndSerial", (long long)LB_END.serialNumber());
            in.n("lookbackEndTime", process->time(LB_END));
            describePath(in);
            Obj ex;
            ex.i("endIndex", (long long)path.timeGrid().closestIndex(process->time(LB_END)));
            ex.n("value", (*pricer)(path));
            addCase(type == Option::Call ? "lb_pathpricer_partfloating_call"
                                         : "lb_pathpricer_partfloating_put",
                    in, ex);
        }

        // Wrong payoff kinds are rejected by the factory overloads.
        addThrowCase("lb_pathpricer_fixed_non_plain_payoff",
                     "detail::mc_lookback_path_pricer(fixed args) with FloatingTypePayoff", [&] {
                         ContinuousFixedLookbackOption::arguments args;
                         args.payoff = ext::make_shared<FloatingTypePayoff>(Option::Call);
                         args.exercise = exercise;
                         args.minmax = 95.0;
                         auto p = detail::mc_lookback_path_pricer(args, *process, discount);
                     });
        addThrowCase("lb_pathpricer_floating_non_floating_payoff",
                     "detail::mc_lookback_path_pricer(floating args) with PlainVanillaPayoff", [&] {
                         ContinuousFloatingLookbackOption::arguments args;
                         args.payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
                         args.exercise = exercise;
                         args.minmax = 95.0;
                         auto p = detail::mc_lookback_path_pricer(args, *process, discount);
                     });
        // A negative fixed strike trips LookbackFixedPathPricer's own guard.
        addThrowCase("lb_pathpricer_fixed_negative_strike", "fixed lookback with strike < 0", [&] {
            ContinuousFixedLookbackOption::arguments args;
            args.payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, -1.0);
            args.exercise = exercise;
            args.minmax = 95.0;
            auto p = detail::mc_lookback_path_pricer(args, *process, discount);
        });
    }

    // =======================================================================
    // 7. MakeMCLookbackEngine / MCLookbackEngine validation.
    // =======================================================================
    {
        auto process = bsmProcess();
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 95.0);
        auto exercise = ext::make_shared<EuropeanExercise>(EXERCISE);

        addThrowCase("lb_make_no_steps_throws", "neither withSteps nor withStepsPerYear", [&] {
            ext::shared_ptr<PricingEngine> e =
                MakeMCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>(process)
                    .withSamples(255)
                    .withSeed(42);
        });
        addThrowCase("lb_make_both_steps_throws", "withSteps and withStepsPerYear", [&] {
            ext::shared_ptr<PricingEngine> e =
                MakeMCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>(process)
                    .withSteps(8)
                    .withStepsPerYear(4)
                    .withSamples(255)
                    .withSeed(42);
        });
        addThrowCase("lb_make_samples_then_tolerance_throws", "withSamples then tolerance", [&] {
            MakeMCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>(process)
                .withSteps(8)
                .withSamples(255)
                .withAbsoluteTolerance(0.01);
        });
        addThrowCase("lb_make_tolerance_then_samples_throws", "tolerance then withSamples", [&] {
            MakeMCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>(process)
                .withSteps(8)
                .withAbsoluteTolerance(0.01)
                .withSamples(255);
        });
        addThrowCase("lb_make_ld_tolerance_throws", "LowDiscrepancy + withAbsoluteTolerance", [&] {
            MakeMCLookbackEngine<ContinuousFixedLookbackOption, LowDiscrepancy>(process)
                .withSteps(8)
                .withAbsoluteTolerance(0.01);
        });
        addThrowCase("lb_engine_zero_steps_throws", "MCLookbackEngine(timeSteps = 0)", [&] {
            auto e = ext::make_shared<
                MCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>>(
                process, 0, Null<Size>(), false, false, 255, Null<Real>(), Null<Size>(), 42);
        });
        addThrowCase("lb_engine_zero_steps_per_year_throws",
                     "MCLookbackEngine(timeStepsPerYear = 0)", [&] {
                         auto e = ext::make_shared<
                             MCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>>(
                             process, Null<Size>(), 0, false, false, 255, Null<Real>(),
                             Null<Size>(), 42);
                     });
        addThrowCase("lb_engine_no_steps_throws", "MCLookbackEngine(both Null)", [&] {
            auto e = ext::make_shared<
                MCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>>(
                process, Null<Size>(), Null<Size>(), false, false, 255, Null<Real>(), Null<Size>(),
                42);
        });
        addThrowCase("lb_engine_both_steps_throws", "MCLookbackEngine(steps AND stepsPerYear)",
                     [&] {
                         auto e = ext::make_shared<
                             MCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>>(
                             process, 8, 4, false, false, 255, Null<Real>(), Null<Size>(), 42);
                     });
        // calculate() checks the spot BEFORE simulating.
        addThrowCase("lb_engine_zero_spot_throws", "spot == 0 -> negative or null underlying", [&] {
            ContinuousFixedLookbackOption option(95.0, payoff, exercise);
            option.setPricingEngine(
                lookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>(
                    bsmProcess(0.0), 8, Null<Size>(), false, false, 255, 42));
            option.NPV();
        });
        // Neither samples nor tolerance -> McSimulation::calculate QL_REQUIRE.
        addThrowCase("lb_engine_no_samples_no_tolerance_throws",
                     "requiredSamples and requiredTolerance both Null", [&] {
                         ContinuousFixedLookbackOption option(95.0, payoff, exercise);
                         option.setPricingEngine(ext::make_shared<
                             MCLookbackEngine<ContinuousFixedLookbackOption, PseudoRandom>>(
                             process, 8, Null<Size>(), false, false, Null<Size>(), Null<Real>(),
                             Null<Size>(), 42));
                         option.NPV();
                     });
    }

    // =======================================================================
    // 8. MCPerformanceEngine::timeGrid() -- the fixing schedule itself.
    // =======================================================================
    {
        auto process = bsmProcess();
        const std::vector<Date> resets = resetSchedule();
        std::vector<Time> fixingTimes;
        for (const auto& d : resets)
            fixingTimes.push_back(process->time(d));
        fixingTimes.push_back(process->time(EXERCISE));
        TimeGrid grid(fixingTimes.begin(), fixingTimes.end());
        std::vector<Real> times(grid.begin(), grid.end());
        std::vector<Real> discounts;
        for (const auto& d : resets)
            discounts.push_back(process->riskFreeRate()->discount(d));
        discounts.push_back(process->riskFreeRate()->discount(EXERCISE));

        Obj in;
        std::vector<long long> serials;
        for (const auto& d : resets)
            serials.push_back((long long)d.serialNumber());
        in.iarr("resetDateSerials", serials);
        in.i("exerciseDateSerial", (long long)EXERCISE.serialNumber());
        Obj ex;
        ex.i("size", (long long)grid.size());
        ex.arr("times", times);
        ex.arr("fixingTimes", fixingTimes);
        // NOTE: the discounts are taken at the reset DATES and at the exercise
        // DATE, not at grid times -- unlike every other engine in this wave.
        ex.arr("discounts", discounts);
        addCase("perf_timegrid", in, ex);
    }

    // =======================================================================
    // 9. MCPerformanceEngine sweep.
    // =======================================================================
    runPerformance<PseudoRandom>("perf_call_m1_pr", "PseudoRandom", Option::Call, 1.0,
                                 resetSchedule(), false, false, 1023, 42);
    runPerformance<PseudoRandom>("perf_put_m1_pr", "PseudoRandom", Option::Put, 1.0,
                                 resetSchedule(), false, false, 1023, 42);
    runPerformance<PseudoRandom>("perf_call_m1_pr_seed7", "PseudoRandom", Option::Call, 1.0,
                                 resetSchedule(), false, false, 1023, 7);
    runPerformance<PseudoRandom>("perf_call_m1_pr_anti", "PseudoRandom", Option::Call, 1.0,
                                 resetSchedule(), false, true, 1023, 42);
    runPerformance<PseudoRandom>("perf_call_m1_pr_bb", "PseudoRandom", Option::Call, 1.0,
                                 resetSchedule(), true, false, 1023, 42);
    runPerformance<PseudoRandom>("perf_call_m105_pr", "PseudoRandom", Option::Call, 1.05,
                                 resetSchedule(), false, false, 1023, 42);
    runPerformance<PseudoRandom>("perf_call_m095_pr", "PseudoRandom", Option::Call, 0.95,
                                 resetSchedule(), false, false, 1023, 42);
    // A single reset: the grid is {0, t(reset), t(expiry)} and the pricer's
    // i-from-2 loop pays exactly ONE period.
    runPerformance<PseudoRandom>("perf_call_single_reset_pr", "PseudoRandom", Option::Call, 1.0,
                                 {Date(15, December, 2023)}, false, false, 1023, 42);
    runPerformance<LowDiscrepancy>("perf_call_m1_ld", "LowDiscrepancy", Option::Call, 1.0,
                                   resetSchedule(), false, false, 1024, 42);

    // =======================================================================
    // 10. PerformanceOptionPathPricer, directly, plus its guards, plus the
    //     first-reset-today grid collapse.
    // =======================================================================
    {
        const Path path = buildPath();
        std::vector<Real> pathVals(kPathLevels, kPathLevels + kPathSteps + 1);
        // path.length() == 9, so exactly 8 discounts are required.
        const std::vector<DiscountFactor> discounts = {0.99, 0.98, 0.97, 0.96,
                                                       0.95, 0.94, 0.93, 0.92};
        for (Option::Type type : {Option::Call, Option::Put}) {
            for (Real strike : {1.0, 1.05}) {
                PerformanceOptionPathPricer pricer(type, strike, discounts);
                Obj in;
                in.s("pricer", "PerformanceOptionPathPricer");
                in.s("optionType", type == Option::Call ? "Call" : "Put");
                in.n("strike", strike);
                in.arr("path", pathVals);
                in.arr("discounts", std::vector<Real>(discounts.begin(), discounts.end()));
                Obj ex;
                ex.n("value", pricer(path));
                std::ostringstream nm;
                nm << "perf_pathpricer_" << (type == Option::Call ? "call" : "put") << "_k"
                   << (strike == 1.0 ? "1" : "105");
                addCase(nm.str(), in, ex);
            }
        }
        addThrowCase("perf_pathpricer_discount_mismatch_throws",
                     "len(discounts) + 1 != path.length()", [&] {
                         std::vector<DiscountFactor> shortDiscounts(discounts.begin(),
                                                                    discounts.end() - 1);
                         PerformanceOptionPathPricer pricer(Option::Call, 1.0, shortDiscounts);
                         (void)pricer(buildPath());
                     });
    }

    // =======================================================================
    // 11. MakeMCPerformanceEngine / MCPerformanceEngine validation.
    // =======================================================================
    {
        auto process = bsmProcess();
        const std::vector<Date> resets = resetSchedule();

        addThrowCase("perf_make_samples_then_tolerance_throws", "withSamples then tolerance", [&] {
            MakeMCPerformanceEngine<PseudoRandom>(process).withSamples(255).withAbsoluteTolerance(
                0.01);
        });
        addThrowCase("perf_make_tolerance_then_samples_throws", "tolerance then withSamples", [&] {
            MakeMCPerformanceEngine<PseudoRandom>(process).withAbsoluteTolerance(0.01).withSamples(
                255);
        });
        addThrowCase("perf_make_ld_tolerance_throws", "LowDiscrepancy + withAbsoluteTolerance",
                     [&] {
                         MakeMCPerformanceEngine<LowDiscrepancy>(process).withAbsoluteTolerance(
                             0.01);
                     });
        // Unlike every other MakeMC* builder here, this one has no steps knobs
        // and its conversion operator validates NOTHING: a builder with no
        // samples and no tolerance converts fine and only fails at NPV() time.
        {
            Obj in;
            in.s("api", "MakeMCPerformanceEngine");
            Obj ex;
            ex.b("hasWithSteps", false);
            ex.b("hasWithStepsPerYear", false);
            ex.b("conversionValidates", false);
            ex.b("conversionThrowsWithNoSamples",
                 throws([&] {
                     ext::shared_ptr<PricingEngine> e =
                         MakeMCPerformanceEngine<PseudoRandom>(process).withSeed(42);
                 }));
            addCase("perf_make_no_steps_knobs", in, ex);
        }
        addThrowCase("perf_engine_no_samples_no_tolerance_throws",
                     "neither samples nor tolerance -> fails at NPV()", [&] {
                         CliquetOption option(
                             ext::make_shared<PercentageStrikePayoff>(Option::Call, 1.0),
                             ext::make_shared<EuropeanExercise>(EXERCISE), resets);
                         option.setPricingEngine(
                             MakeMCPerformanceEngine<PseudoRandom>(process).withSeed(42));
                         option.NPV();
                     });
        // The FIRST reset being the evaluation date makes t(reset0) == 0, so
        // TimeGrid does not prepend its own 0 and the grid ends up one point
        // shorter than the discount vector -> "discounts/options mismatch".
        addThrowCase("perf_first_reset_today_throws",
                     "first reset date == evaluation date collapses the grid", [&] {
                         std::vector<Date> withToday = {TODAY, Date(15, December, 2023),
                                                        Date(15, March, 2024)};
                         CliquetOption option(
                             ext::make_shared<PercentageStrikePayoff>(Option::Call, 1.0),
                             ext::make_shared<EuropeanExercise>(EXERCISE), withToday);
                         option.setPricingEngine(MakeMCPerformanceEngine<PseudoRandom>(process)
                                                     .withSamples(255)
                                                     .withSeed(42));
                         option.NPV();
                     });
    }

    // =======================================================================
    // 12. MCForwardVanillaEngine constructor guards, through the concrete BS
    //     subclass. mcforward.json pins the BUILDER guards; these are the
    //     ENGINE ones, which a port can easily leave out because the builder
    //     appears to cover them.
    // =======================================================================
    {
        auto process = bsmProcess();
        addThrowCase("fwdvanilla_engine_zero_steps_throws",
                     "MCForwardEuropeanBSEngine(timeSteps = 0)", [&] {
                         auto e = ext::make_shared<MCForwardEuropeanBSEngine<PseudoRandom>>(
                             process, 0, Null<Size>(), false, false, 255, Null<Real>(),
                             Null<Size>(), 42);
                     });
        addThrowCase("fwdvanilla_engine_zero_steps_per_year_throws",
                     "MCForwardEuropeanBSEngine(timeStepsPerYear = 0)", [&] {
                         auto e = ext::make_shared<MCForwardEuropeanBSEngine<PseudoRandom>>(
                             process, Null<Size>(), 0, false, false, 255, Null<Real>(),
                             Null<Size>(), 42);
                     });
        addThrowCase("fwdvanilla_engine_no_steps_throws",
                     "MCForwardEuropeanBSEngine(both Null)", [&] {
                         auto e = ext::make_shared<MCForwardEuropeanBSEngine<PseudoRandom>>(
                             process, Null<Size>(), Null<Size>(), false, false, 255, Null<Real>(),
                             Null<Size>(), 42);
                     });
        addThrowCase("fwdvanilla_engine_both_steps_throws",
                     "MCForwardEuropeanBSEngine(steps AND stepsPerYear)", [&] {
                         auto e = ext::make_shared<MCForwardEuropeanBSEngine<PseudoRandom>>(
                             process, 8, 4, false, false, 255, Null<Real>(), Null<Size>(), 42);
                     });
        // pathPricer() rejects a non-plain payoff and a non-European exercise.
        addThrowCase("fwdbs_engine_non_plain_payoff_throws",
                     "CashOrNothingPayoff on MCForwardEuropeanBSEngine", [&] {
                         ForwardVanillaOption option(
                             1.0, RESET,
                             ext::make_shared<CashOrNothingPayoff>(Option::Call, 0.0, 10.0),
                             ext::make_shared<EuropeanExercise>(EXERCISE));
                         option.setPricingEngine(
                             MakeMCForwardEuropeanBSEngine<PseudoRandom>(process)
                                 .withSteps(8)
                                 .withSamples(255)
                                 .withSeed(42));
                         option.NPV();
                     });
        addThrowCase("fwdbs_engine_american_exercise_throws",
                     "AmericanExercise on MCForwardEuropeanBSEngine", [&] {
                         ForwardVanillaOption option(
                             1.0, RESET, ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0),
                             ext::make_shared<AmericanExercise>(TODAY, EXERCISE));
                         option.setPricingEngine(
                             MakeMCForwardEuropeanBSEngine<PseudoRandom>(process)
                                 .withSteps(8)
                                 .withSamples(255)
                                 .withSeed(42));
                         option.NPV();
                     });
    }

    // =======================================================================
    // 13. MakeMCForwardEuropeanHestonEngine validation. mcforward.json pins the
    //     BS builder only; this builder is a separate class with the same
    //     shape, and (unlike MakeMCEuropeanHestonEngine in the vanilla wave) it
    //     checks BOTH halves of the steps rule at conversion time, not early.
    // =======================================================================
    {
        auto process = hestonProcess();
        addThrowCase("fwdheston_make_no_steps_throws", "neither withSteps nor withStepsPerYear",
                     [&] {
                         ext::shared_ptr<PricingEngine> e =
                             MakeMCForwardEuropeanHestonEngine<PseudoRandom>(process)
                                 .withSamples(255)
                                 .withSeed(42);
                     });
        addThrowCase("fwdheston_make_both_steps_throws", "withSteps and withStepsPerYear", [&] {
            ext::shared_ptr<PricingEngine> e =
                MakeMCForwardEuropeanHestonEngine<PseudoRandom>(process)
                    .withSteps(8)
                    .withStepsPerYear(4)
                    .withSamples(255)
                    .withSeed(42);
        });
        addThrowCase("fwdheston_make_samples_then_tolerance_throws", "withSamples then tolerance",
                     [&] {
                         MakeMCForwardEuropeanHestonEngine<PseudoRandom>(process)
                             .withSteps(8)
                             .withSamples(255)
                             .withAbsoluteTolerance(0.01);
                     });
        addThrowCase("fwdheston_make_tolerance_then_samples_throws", "tolerance then withSamples",
                     [&] {
                         MakeMCForwardEuropeanHestonEngine<PseudoRandom>(process)
                             .withSteps(8)
                             .withAbsoluteTolerance(0.01)
                             .withSamples(255);
                     });
        addThrowCase("fwdheston_make_ld_tolerance_throws",
                     "LowDiscrepancy + withAbsoluteTolerance", [&] {
                         MakeMCForwardEuropeanHestonEngine<LowDiscrepancy>(process)
                             .withSteps(8)
                             .withAbsoluteTolerance(0.01);
                     });
        // withSteps/withStepsPerYear must NOT throw early -- the asymmetry that
        // bit the vanilla wave (MakeMCEuropeanHestonEngine rejects early).
        {
            Obj in;
            in.s("api", "MakeMCForwardEuropeanHestonEngine");
            Obj ex;
            ex.b("withStepsPerYearAfterStepsThrowsEarly", throws([&] {
                     MakeMCForwardEuropeanHestonEngine<PseudoRandom>(process)
                         .withSteps(8)
                         .withStepsPerYear(4);
                 }));
            ex.b("hasWithControlVariate", true);
            ex.b("hasWithBrownianBridge", false);
            addCase("fwdheston_make_shape", in, ex);
        }
    }

    // =======================================================================
    // 14. ForwardEuropeanBSPathPricer / ForwardEuropeanHestonPathPricer,
    //     directly. Through the instrument the moneyness guard is dead code
    //     (ForwardOptionArguments::validate is stricter); constructed directly
    //     it is live, and moneyness == 0 is ACCEPTED here even though the
    //     instrument rejects it.
    // =======================================================================
    {
        const Path path = buildPath();
        std::vector<Real> pathVals(kPathLevels, kPathLevels + kPathSteps + 1);
        const DiscountFactor discount = 0.9375;

        struct FwdCfg {
            const char* name;
            Option::Type type;
            Real moneyness;
            long long resetIndex;
        };
        const FwdCfg cfgs[] = {
            {"fwdbs_pathpricer_call_m1_reset0", Option::Call, 1.0, 0},
            {"fwdbs_pathpricer_call_m1_reset3", Option::Call, 1.0, 3},
            {"fwdbs_pathpricer_put_m1_reset3", Option::Put, 1.0, 3},
            {"fwdbs_pathpricer_call_m085_reset3", Option::Call, 0.85, 3},
            {"fwdbs_pathpricer_call_m125_reset3", Option::Call, 1.25, 3},
            // moneyness == 0 -> strike 0 -> a call pays the terminal spot.
            {"fwdbs_pathpricer_call_m0_reset3", Option::Call, 0.0, 3},
        };
        for (const auto& c : cfgs) {
            ForwardEuropeanBSPathPricer pricer(c.type, c.moneyness, Size(c.resetIndex), discount);
            Obj in;
            in.s("pricer", "ForwardEuropeanBSPathPricer");
            in.s("optionType", c.type == Option::Call ? "Call" : "Put");
            in.n("moneyness", c.moneyness);
            in.i("resetIndex", c.resetIndex);
            in.n("discount", discount);
            in.arr("path", pathVals);
            Obj ex;
            ex.n("value", pricer(path));
            addCase(c.name, in, ex);
        }

        // The Heston pricer reads multiPath[0] and ignores the variance leg
        // entirely; a second, wildly different asset path is supplied as
        // multiPath[1] to prove it is never touched.
        {
            std::vector<Path> legs;
            legs.push_back(buildPath());
            Path variance(TimeGrid(1.0, kPathSteps));
            for (Size k = 0; k <= kPathSteps; ++k)
                variance[k] = 0.01 + 0.002 * Real(k);
            legs.push_back(variance);
            MultiPath multiPath(legs);
            std::vector<Real> varianceVals;
            for (Size k = 0; k <= kPathSteps; ++k)
                varianceVals.push_back(variance[k]);

            const FwdCfg hCfgs[] = {
                {"fwdheston_pathpricer_call_m1_reset0", Option::Call, 1.0, 0},
                {"fwdheston_pathpricer_call_m1_reset3", Option::Call, 1.0, 3},
                {"fwdheston_pathpricer_put_m1_reset3", Option::Put, 1.0, 3},
                {"fwdheston_pathpricer_call_m085_reset3", Option::Call, 0.85, 3},
            };
            for (const auto& c : hCfgs) {
                ForwardEuropeanHestonPathPricer pricer(c.type, c.moneyness, Size(c.resetIndex),
                                                       discount);
                Obj in;
                in.s("pricer", "ForwardEuropeanHestonPathPricer");
                in.s("optionType", c.type == Option::Call ? "Call" : "Put");
                in.n("moneyness", c.moneyness);
                in.i("resetIndex", c.resetIndex);
                in.n("discount", discount);
                in.arr("assetPath", pathVals);
                in.arr("variancePath", varianceVals);
                Obj ex;
                ex.n("value", pricer(multiPath));
                addCase(c.name, in, ex);
            }
        }

        addThrowCase("fwdbs_pathpricer_negative_moneyness_throws",
                     "ForwardEuropeanBSPathPricer(moneyness = -0.5)",
                     [&] { ForwardEuropeanBSPathPricer p(Option::Call, -0.5, 3, discount); });
        addThrowCase("fwdheston_pathpricer_negative_moneyness_throws",
                     "ForwardEuropeanHestonPathPricer(moneyness = -0.5)",
                     [&] { ForwardEuropeanHestonPathPricer p(Option::Call, -0.5, 3, discount); });
    }

    // =======================================================================
    // 15. DIAGNOSTIC, and it is load-bearing: the raw per-path sample spread of
    //     MCVarianceSwapEngine's estimator.
    //
    //     MCVarianceSwapEngine reports errorEstimate = multiplier *
    //     sampleAccumulator().errorEstimate(), with multiplier = +-discount *
    //     notional (1e6 in the mcforward cases). For a Black-Scholes process
    //     whose local volatility is a function of TIME ONLY -- which both
    //     BlackConstantVol and BlackVarianceCurve are, since
    //     GeneralizedBlackScholesProcess::localVolatility() routes a
    //     BlackVarianceCurve to LocalVolCurve, whose localVolImpl ignores the
    //     level argument entirely -- VariancePathPricer returns the SAME number
    //     for every path. The sample variance is then not a statistic at all: it
    //     is the floating-point residue of subtracting a mean from values equal
    //     to it, scaled up by the notional.
    //
    //     min/max/standardDeviation are pinned here so a port can prove that,
    //     rather than assume it, and so the Python test can bound the error
    //     estimate by ULPs instead of pretending it is reproducible.
    // =======================================================================
    {
        struct SpreadCfg {
            const char* name;
            const char* volKind;
            long long steps;
            long long samples;
            long long seed;
        };
        const SpreadCfg cfgs[] = {
            {"vs_sample_spread_constant_steps52", "constant", 52, 1023, 42},
            {"vs_sample_spread_termvol_steps52", "termStructure", 52, 1023, 42},
            {"vs_sample_spread_termvol_steps13", "termStructure", 13, 1023, 42},
        };
        for (const auto& c : cfgs) {
            auto process = std::string(c.volKind) == "termStructure" ? bsmProcessTermVol()
                                                                     : bsmProcess();
            const Time t = process->time(EXERCISE);
            TimeGrid grid(t, Size(c.steps));
            PseudoRandom::rsg_type gen = PseudoRandom::make_sequence_generator(
                process->factors() * (grid.size() - 1), BigNatural(c.seed));
            PathGenerator<PseudoRandom::rsg_type> pathGen(process, grid, gen, false);
            VariancePathPricer pricer(process);
            GeneralStatistics stats;
            for (long long k = 0; k < c.samples; ++k) {
                const auto& sample = pathGen.next();
                stats.add(pricer(sample.value), sample.weight);
            }
            Obj in;
            in.s("volKind", c.volKind);
            in.i("steps", c.steps);
            in.i("samples", c.samples);
            in.i("seed", c.seed);
            Obj ex;
            ex.n("mean", stats.mean());
            ex.n("min", stats.min());
            ex.n("max", stats.max());
            ex.n("spread", stats.max() - stats.min());
            ex.b("allSamplesIdentical", stats.min() == stats.max());
            ex.n("standardDeviation", stats.standardDeviation());
            ex.n("errorEstimate", stats.errorEstimate());
            addCase(c.name, in, ex);
        }

        // And the reason the spread is what it is: the local volatility of a
        // BlackVarianceCurve does not depend on the level. Pinned at three very
        // different levels; mcforward.json's `termvol_curve` pins only spot.
        {
            auto tv = bsmProcessTermVol();
            std::vector<Real> levels{50.0, 95.0, 200.0};
            std::vector<Real> times{0.1, 0.5, 1.0};
            std::vector<Real> localVols;
            for (Real u : times)
                for (Real level : levels)
                    localVols.push_back(tv->localVolatility()->localVol(u, level, true));
            Obj in;
            in.s("curve", "BlackVarianceCurve");
            in.arr("times", times);
            in.arr("levels", levels);
            Obj ex;
            ex.arr("localVols", localVols);
            ex.b("strikeIndependent",
                 localVols[0] == localVols[1] && localVols[1] == localVols[2] &&
                     localVols[3] == localVols[4] && localVols[4] == localVols[5] &&
                     localVols[6] == localVols[7] && localVols[7] == localVols[8]);
            addCase("termvol_localvol_is_level_independent", in, ex);
        }
    }

    emitDocument();
    return 0;
}
