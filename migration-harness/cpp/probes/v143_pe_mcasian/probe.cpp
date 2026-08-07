// migration-harness/cpp/probes/v143_pe_mcasian/probe.cpp
//
// Reference values for the Monte Carlo *discrete-average Asian* pricing-engine
// cluster of C++ QuantLib v1.43:
//
//   * MCDiscreteAveragingAsianEngineBase / detail::PastFixingsOnly
//         ql/pricingengines/asian/mcdiscreteasianenginebase.hpp
//   * MCDiscreteArithmeticAPEngine / ArithmeticAPOPathPricer /
//     MakeMCDiscreteArithmeticAPEngine
//         ql/pricingengines/asian/mc_discr_arith_av_price.{hpp,cpp}
//   * MCDiscreteGeometricAPEngine / GeometricAPOPathPricer /
//     MakeMCDiscreteGeometricAPEngine
//         ql/pricingengines/asian/mc_discr_geom_av_price.{hpp,cpp}
//   * MCDiscreteArithmeticASEngine / ArithmeticASOPathPricer /
//     MakeMCDiscreteArithmeticASEngine
//         ql/pricingengines/asian/mc_discr_arith_av_strike.{hpp,cpp}
//   * MCDiscreteArithmeticAPHestonEngine / ArithmeticAPOHestonPathPricer /
//     MakeMCDiscreteArithmeticAPHestonEngine
//         ql/pricingengines/asian/mc_discr_arith_av_price_heston.{hpp,cpp}
//   * MCDiscreteGeometricAPHestonEngine / GeometricAPOHestonPathPricer /
//     MakeMCDiscreteGeometricAPHestonEngine
//         ql/pricingengines/asian/mc_discr_geom_av_price_heston.{hpp,cpp}
//
// Why the values below are EXACT, not a confidence band
// -----------------------------------------------------
// Every engine here is deterministic once the seed is fixed:
//   PseudoRandom   = InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
//                                         InverseCumulativeNormal>
//   LowDiscrepancy = InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>
// Neither consults the clock as long as `seed != 0` (seed 0 routes through
// SeedGenerator, which IS clock-derived).  EVERY case below pins an explicit
// nonzero seed, so a correct port reproduces the NPV *and* the error estimate
// to ~1e-14.  A statistical band would pass with the wrong RNG, the wrong
// fixing-count rule, or the wrong antithetic pairing, so nothing is banded.
//
// What has to be pinned, and why
// ------------------------------
//  1. `MCDiscreteAveragingAsianEngineBase::timeGrid()`
//     (mcdiscreteasianenginebase.hpp:154-185) is where most ports go wrong:
//       - fixing times come from `process_->time(date)`, i.e. the RISK-FREE
//         curve's day counter and reference date -- NOT the vol curve's;
//       - only `t >= 0` fixings survive;
//       - if nothing survives, or the only survivor is t == 0, it throws
//         `detail::PastFixingsOnly` ("all fixings are in the past") -- a
//         distinct exception type, pinned by the `throw_past_fixings_*` cases;
//       - `includeExerciseDate_` (true ONLY for MCDiscreteArithmeticASEngine)
//         appends the exercise time when it is strictly after the last fixing;
//       - `timeSteps_` / `timeStepsPerYear_` select the mandatory+steps
//         TimeGrid constructor, and `Size(timeStepsPerYear_*t)` TRUNCATES.
//     Every priced case therefore also emits the realised grid
//     (`grid_size`, `grid_t<i>`), read back out of
//     `results_.additionalResults["TimeGrid"]` -- which the base class fills
//     unconditionally (mcdiscreteasianenginebase.hpp:104).
//  2. The fixing count in the path pricers depends on
//     `path.timeGrid().mandatoryTimes()[0] == 0.0`:
//       - if the first mandatory time is 0 (a fixing ON the evaluation date),
//         path[0] JOINS the average and `fixings = pastFixings + n`;
//       - otherwise path[0] is the t=0 anchor, it is skipped, and
//         `fixings = pastFixings + n - 1`.
//     `*_t0fix` cases drive the first branch, everything else the second.
//     Note ArithmeticAPOPathPricer uses `n = path.length()` while
//     GeometricAPOPathPricer uses `n = path.length() - 1`; the two spell the
//     same count differently and a port that copies one into the other is
//     wrong by one fixing.
//  3. Seasoning. `runningAccumulator` is a running SUM for the arithmetic
//     engines and a running PRODUCT for the geometric one; `pastFixings`
//     enlarges the denominator.  `*_seasoned` cases carry both.  NB the
//     control-variate path pricer built by
//     `MCDiscreteArithmeticAPEngine::controlPathPricer` deliberately drops the
//     seasoning (mc_discr_arith_av_price.hpp:178-184) and the analytic control
//     engine's non-Geometric branch does the same (`runningLog = 1.0`,
//     `pastFixings = 0`), so a seasoned CV case is NOT a seasoned geometric
//     price -- that asymmetry is pinned, not smoothed over.
//  4. Control variate. `MCDiscreteArithmeticAPEngine` uses
//     `AnalyticDiscreteGeometricAveragePriceAsianEngine` +
//     `GeometricAPOPathPricer` (discounted at `timeGrid().back()`, NOT at the
//     exercise date -- the two differ once a grid step lands off the exercise
//     date).  `MCDiscreteArithmeticAPHestonEngine` uses
//     `AnalyticDiscreteGeometricAveragePriceAsianHestonEngine` +
//     `GeometricAPOHestonPathPricer` (discounted AT the exercise date).
//     `MCDiscreteAveragingAsianEngineBase::calculate` then clamps
//     `results_.value = max(0.0, value)` when controlVariate_ is on; the
//     `arithap_cv_clamped` case searches seeds until the clamp actually fires
//     and pins the seed it found, so the clamp is exercised rather than
//     asserted about.
//  5. `if constexpr (RNG::allowsErrorEstimate)`.  LowDiscrepancy has
//     allowsErrorEstimate == 0, so `option.errorEstimate()` THROWS for every
//     `*_ld_*` case; that absence is pinned ("has_error_estimate": false) so a
//     port cannot invent a number C++ never produces.  A port that instead
//     gates on `samples() > 1` fills it and fails here.
//  6. The Heston engines are MultiVariate: MultiPathGenerator, dimension =
//     factors*(grid.size()-1), Gaussians consumed 2 at a time per step,
//     Brownian bridge unconditionally unavailable (the base is constructed
//     with brownianBridge = false).  Their path pricers read `multiPath[0]`
//     (the spot leg) at the FIXING INDICES only --
//     `timeGrid.closestIndex(mandatoryTime)` -- so when extra steps are
//     inserted between fixings the non-fixing points must NOT enter the
//     average.  `*_steps24` / `*_stepsperyear24` drive exactly that.
//     `HestonProcess`'s default discretization is
//     QuadraticExponentialMartingale, not Euler.
//  7. `MakeMC*` builder validation.  The three Black-Scholes builders have NO
//     steps knobs at all; the two Heston builders reject over-specified steps
//     EARLY, inside withSteps/withStepsPerYear ("number of steps per year
//     already set" / "number of steps already set"), and their
//     `operator shared_ptr` performs no check whatsoever -- there is no
//     "number of steps not given" for this family, because Null steps are
//     legal (one step per fixing).  All five builders share
//     "tolerance already set" / "number of samples already set" /
//     "chosen random generator policy does not allow an error estimate".
//     Leaving both samples and tolerance unset is legal at build time and
//     fails later inside McSimulation::calculate.
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

#include <ql/exercise.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/averagetype.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/pricingengines/asian/mc_discr_arith_av_price.hpp>
#include <ql/pricingengines/asian/mc_discr_arith_av_price_heston.hpp>
#include <ql/pricingengines/asian/mc_discr_arith_av_strike.hpp>
#include <ql/pricingengines/asian/mc_discr_geom_av_price.hpp>
#include <ql/pricingengines/asian/mc_discr_geom_av_price_heston.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
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
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

std::string esc(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        if (c == '\n')
            out += "\\n";
        else
            out += c;
    }
    return out;
}

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) {
        return put(k, "\"" + esc(v) + "\"");
    }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
    Obj& arr(const std::string& k, const std::vector<long long>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j ? ", " : "") + std::to_string(v[j]);
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
// Market.  15 May 2025 + 30k days for k = 1..12 keeps every fixing time an
// exact multiple of 30/365 under Actual365Fixed, so no day-count rounding
// noise leaks into any reference value.
// ---------------------------------------------------------------------------
const Date kToday(15, May, 2025);

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
bsm(Real s0, Rate r, Rate q, Volatility v) {
    return ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(s0)), flatCurve(q),
        flatCurve(r), flatVol(v));
}

struct HestonParams {
    Real s0, r, q, v0, kappa, theta, sigma, rho;
};

const HestonParams kHeston = {100.0, 0.05, 0.0, 0.09, 1.20, 0.10, 0.40, -0.30};

ext::shared_ptr<HestonProcess> heston(const HestonParams& p) {
    return ext::make_shared<HestonProcess>(
        flatCurve(p.r), flatCurve(p.q),
        Handle<Quote>(ext::make_shared<SimpleQuote>(p.s0)), p.v0, p.kappa,
        p.theta, p.sigma, p.rho);
}

// --- fixing schedules -------------------------------------------------------
// Day offsets from kToday.  Negative offsets are genuinely past fixings.
const std::vector<long long> kFix12 = {30,  60,  90,  120, 150, 180,
                                       210, 240, 270, 300, 330, 360};
// A fixing exactly ON the evaluation date -> mandatoryTimes()[0] == 0.0.
const std::vector<long long> kFix12WithT0 = {0,   30,  60,  90,  120, 150,
                                             180, 210, 240, 270, 300, 330};
// Six remaining future fixings, used together with pastFixings = 6.
const std::vector<long long> kFix6 = {210, 240, 270, 300, 330, 360};
const std::vector<long long> kFixAllPast = {-180, -150, -120, -90, -60, -30};
const std::vector<long long> kFixOnlyT0 = {0};

std::vector<Date> dates(const std::vector<long long>& offsets) {
    std::vector<Date> out;
    out.reserve(offsets.size());
    for (long long o : offsets)
        out.push_back(kToday + static_cast<Date::serial_type>(o));
    return out;
}

// --- shared case description ------------------------------------------------

void describeCommon(Obj& in, const std::string& engine, const std::string& rng,
                    const std::vector<long long>& fixings, long long expiryDays,
                    Option::Type type, Real strike,
                    const std::string& averageType, Real runningAccumulator,
                    long long pastFixings) {
    in.s("engine", engine);
    in.s("rng", rng);
    in.arr("fixing_offsets", fixings);
    in.i("expiry_days", expiryDays);
    in.s("payoff", "PlainVanilla");
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.s("average_type", averageType);
    in.n("running_accumulator", runningAccumulator);
    in.i("past_fixings", pastFixings);
    in.s("day_counter", "Actual365Fixed");
    in.s("calendar", "NullCalendar");
    in.i("eval_year", kToday.year());
    in.i("eval_month", static_cast<int>(kToday.month()));
    in.i("eval_day", kToday.dayOfMonth());
}

void describeBsmMarket(Obj& in, Real s0, Rate r, Rate q, Volatility v) {
    in.s("process", "bsm");
    in.n("s0", s0);
    in.n("r", r);
    in.n("q", q);
    in.n("vol", v);
}

void describeHestonMarket(Obj& in, const HestonParams& p) {
    in.s("process", "heston");
    in.n("s0", p.s0);
    in.n("r", p.r);
    in.n("q", p.q);
    in.n("v0", p.v0);
    in.n("kappa", p.kappa);
    in.n("theta", p.theta);
    in.n("sigma", p.sigma);
    in.n("rho", p.rho);
    in.s("discretization", "QuadraticExponentialMartingale");
}

void describeMc(Obj& in, bool brownianBridge, bool antithetic,
                bool controlVariate, long long samples, Real tolerance,
                long long maxSamples, long long seed, long long steps,
                long long stepsPerYear) {
    in.b("brownian_bridge", brownianBridge);
    in.b("antithetic", antithetic);
    in.b("control_variate", controlVariate);
    in.i("samples", samples);        // -1 == Null<Size>()
    in.n("tolerance", tolerance);    // -1 == Null<Real>()
    in.i("max_samples", maxSamples); // -1 == Null<Size>()
    in.i("seed", seed);
    in.i("steps", steps);            // -1 == Null<Size>()
    in.i("steps_per_year", stepsPerYear);
}

// Record NPV, the (optionally absent) error estimate, the realised sample
// count and the realised time grid.
void recordResults(Obj& ex, DiscreteAveragingAsianOption& opt, Real npv) {
    ex.n("npv", npv);
    try {
        ex.n("error_estimate", opt.errorEstimate());
        ex.b("has_error_estimate", true);
    } catch (const std::exception&) {
        ex.b("has_error_estimate", false);
    }
    const auto& grid = opt.result<TimeGrid>("TimeGrid");
    ex.i("grid_size", static_cast<long long>(grid.size()));
    for (Size i = 0; i < grid.size(); ++i)
        ex.n("grid_t" + std::to_string(i), grid[i]);
    ex.i("mandatory_size", static_cast<long long>(grid.mandatoryTimes().size()));
    for (Size i = 0; i < grid.mandatoryTimes().size(); ++i)
        ex.n("mandatory_t" + std::to_string(i), grid.mandatoryTimes()[i]);
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

// ===========================================================================
// 1. MCDiscreteArithmeticAPEngine (Black-Scholes, SingleVariate)
// ===========================================================================

struct BsCfg {
    const char* name;
    const char* rng; // "pseudo" | "lowdiscrepancy"
    Real s0, r, q, vol;
    const std::vector<long long>* fixings;
    long long expiryDays;
    Option::Type type;
    Real strike;
    Real runningAccumulator;
    long long pastFixings;
    bool brownianBridge, antithetic, controlVariate;
    long long samples;
    Real tolerance;
    long long maxSamples;
    long long seed;
};

template <class RNG>
void runArithmeticAP(const BsCfg& c) {
    auto process = bsm(c.s0, c.r, c.q, c.vol);
    MakeMCDiscreteArithmeticAPEngine<RNG> make(process);
    make.withBrownianBridge(c.brownianBridge)
        .withAntitheticVariate(c.antithetic)
        .withControlVariate(c.controlVariate)
        .withSeed(BigNatural(c.seed));
    if (c.samples > 0)
        make.withSamples(Size(c.samples));
    else
        make.withAbsoluteTolerance(c.tolerance);
    if (c.maxSamples > 0)
        make.withMaxSamples(Size(c.maxSamples));

    DiscreteAveragingAsianOption opt(
        Average::Arithmetic, c.runningAccumulator, Size(c.pastFixings),
        dates(*c.fixings),
        ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
        ext::make_shared<EuropeanExercise>(
            kToday + static_cast<Date::serial_type>(c.expiryDays)));
    opt.setPricingEngine(make);

    Obj in;
    describeBsmMarket(in, c.s0, c.r, c.q, c.vol);
    describeCommon(in, "MCDiscreteArithmeticAPEngine", c.rng, *c.fixings,
                   c.expiryDays, c.type, c.strike, "Arithmetic",
                   c.runningAccumulator, c.pastFixings);
    describeMc(in, c.brownianBridge, c.antithetic, c.controlVariate, c.samples,
               c.tolerance, c.maxSamples, c.seed, -1, -1);

    Obj ex;
    recordResults(ex, opt, opt.NPV());
    addCase(c.name, in, ex);
}

void arithmeticAPCases() {
    const BsCfg cfgs[] = {
        // --- vanilla 12 monthly future fixings, exercise == last fixing ----
        {"arithap_ps_call_atm", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, false, false, false, 8191, -1.0, -1, 42},
        {"arithap_ps_put_atm", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Put, 100.0, 0.0, 0, false, false, false, 8191, -1.0, -1, 42},
        // A second seed on an otherwise identical case proves the seed
        // actually reaches the generator.
        {"arithap_ps_call_atm_seed7", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 0.0, 0, false, false, false, 8191, -1.0, -1,
         7},
        {"arithap_ps_call_itm", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 80.0, 0.0, 0, false, false, false, 8191, -1.0, -1, 42},
        {"arithap_ps_call_otm", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 130.0, 0.0, 0, false, false, false, 8191, -1.0, -1, 42},
        // --- a fixing ON the evaluation date -> path[0] joins the average ---
        {"arithap_ps_call_t0fix", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12WithT0,
         330, Option::Call, 100.0, 0.0, 0, false, false, false, 8191, -1.0, -1,
         42},
        {"arithap_ps_put_t0fix", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12WithT0,
         330, Option::Put, 100.0, 0.0, 0, false, false, false, 8191, -1.0, -1,
         42},
        // --- seasoned: running SUM over 6 past fixings ----------------------
        {"arithap_ps_call_seasoned", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix6,
         360, Option::Call, 100.0, 618.0, 6, false, false, false, 8191, -1.0,
         -1, 42},
        {"arithap_ps_put_seasoned", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix6,
         360, Option::Put, 100.0, 618.0, 6, false, false, false, 8191, -1.0, -1,
         42},
        // --- variance reduction --------------------------------------------
        {"arithap_ps_call_antithetic", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 0.0, 0, false, true, false, 8191, -1.0, -1,
         42},
        {"arithap_ps_call_bb", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, true, false, false, 8191, -1.0, -1, 42},
        {"arithap_ps_call_cv", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, false, false, true, 8191, -1.0, -1, 42},
        {"arithap_ps_call_cv_antithetic", "pseudo", 100.0, 0.05, 0.0, 0.20,
         &kFix12, 360, Option::Call, 100.0, 0.0, 0, false, true, true, 8191,
         -1.0, -1, 42},
        {"arithap_ps_call_cv_bb", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, true, false, true, 8191, -1.0, -1, 42},
        // CV on a seasoned option: the CV path pricer and the analytic control
        // engine BOTH drop the seasoning, the pricing path pricer does not.
        {"arithap_ps_call_cv_seasoned", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix6,
         360, Option::Call, 100.0, 618.0, 6, false, false, true, 8191, -1.0, -1,
         42},
        // --- exercise strictly after the last fixing: the AP engine does NOT
        //     include the exercise date in the grid (includeExerciseDate is
        //     false), so the grid still ends at the last fixing while the
        //     discount is taken at the exercise date.
        {"arithap_ps_call_exercise_after_fixings", "pseudo", 100.0, 0.05, 0.0,
         0.20, &kFix12, 400, Option::Call, 100.0, 0.0, 0, false, false, false,
         8191, -1.0, -1, 42},
        // --- dividend yield != 0 and a non-ATM strike ----------------------
        {"arithap_ps_call_with_dividend", "pseudo", 100.0, 0.06, 0.03, 0.25,
         &kFix12, 360, Option::Call, 105.0, 0.0, 0, false, false, false, 4095,
         -1.0, -1, 123456789},
        // --- zero vol: every path is the forward -> MC == intrinsic --------
        {"arithap_ps_call_zero_vol", "pseudo", 100.0, 0.05, 0.0, 0.0, &kFix12,
         360, Option::Call, 100.0, 0.0, 0, false, false, false, 1023, -1.0, -1,
         42},
        // --- tolerance-driven termination ----------------------------------
        {"arithap_ps_call_tol_0p20", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 0.0, 0, false, false, false, -1, 0.20, -1,
         42},
        {"arithap_ps_call_tol_0p10_cv", "pseudo", 100.0, 0.05, 0.0, 0.20,
         &kFix12, 360, Option::Call, 100.0, 0.0, 0, false, false, true, -1, 0.10,
         -1, 42},
        {"arithap_ps_call_tol_0p20_maxsamples", "pseudo", 100.0, 0.05, 0.0, 0.20,
         &kFix12, 360, Option::Call, 100.0, 0.0, 0, false, false, false, -1,
         0.20, 1000000, 42},
    };
    for (const auto& c : cfgs)
        runArithmeticAP<PseudoRandom>(c);

    const BsCfg ldCfgs[] = {
        {"arithap_ld_call_atm", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20,
         &kFix12, 360, Option::Call, 100.0, 0.0, 0, false, false, false, 8191,
         -1.0, -1, 42},
        {"arithap_ld_call_t0fix", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20,
         &kFix12WithT0, 330, Option::Call, 100.0, 0.0, 0, false, false, false,
         8191, -1.0, -1, 42},
        {"arithap_ld_call_cv", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 0.0, 0, false, false, true, 8191, -1.0, -1,
         42},
    };
    for (const auto& c : ldCfgs)
        runArithmeticAP<LowDiscrepancy>(c);
}

// The control-variate clamp: `results_.value = max(0.0, value)` fires only
// when the CV-adjusted mean is negative.  For an arithmetic CALL the estimator
// is bounded below by the analytic geometric value (arithmetic >= geometric
// pathwise), so the clamp can NEVER fire there.  It CAN fire for a seasoned
// PUT whose running sum pushes the arithmetic average out of the money while
// the (unseasoned) geometric control still pays: the estimator is then pure
// sampling noise about zero.  Search seeds until it does, and pin the seed.
//
// This case deliberately leaves `withBrownianBridge` unset, so it also pins
// the BUILDER's default -- `bool brownianBridge_ = true`, the opposite of the
// engine constructor's default.  The recorded input says `true` accordingly.
void arithmeticAPClampCase() {
    for (long long seed = 1; seed <= 60; ++seed) {
        auto process = bsm(100.0, 0.05, 0.0, 0.30);
        DiscreteAveragingAsianOption opt(
            Average::Arithmetic, 780.0, 6, dates(kFix6),
            ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
            ext::make_shared<EuropeanExercise>(kToday + 360));
        opt.setPricingEngine(MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                                 .withControlVariate(true)
                                 .withSamples(1023)
                                 .withSeed(BigNatural(seed)));
        const Real npv = opt.NPV();
        if (npv != 0.0)
            continue;

        Obj in;
        describeBsmMarket(in, 100.0, 0.05, 0.0, 0.30);
        describeCommon(in, "MCDiscreteArithmeticAPEngine", "pseudo", kFix6, 360,
                       Option::Put, 100.0, "Arithmetic", 780.0, 6);
        // brownian_bridge = true: the builder default, left untouched above.
        describeMc(in, true, false, true, 1023, -1.0, -1, seed, -1, -1);
        in.s("scenario",
             "seasoned OTM put with CV: raw estimator is negative and "
             "calculate() clamps results_.value to max(0.0, value)");
        Obj ex;
        recordResults(ex, opt, npv);
        ex.b("clamped", true);
        addCase("arithap_ps_put_cv_clamped", in, ex);
        return;
    }
    Obj in;
    in.s("scenario", "no seed in 1..60 produced a negative CV estimator");
    Obj ex;
    ex.b("clamped", false);
    addCase("arithap_ps_put_cv_clamped", in, ex);
}

// ===========================================================================
// 2. MCDiscreteGeometricAPEngine (Black-Scholes, SingleVariate)
// ===========================================================================

template <class RNG>
void runGeometricAP(const BsCfg& c) {
    auto process = bsm(c.s0, c.r, c.q, c.vol);
    MakeMCDiscreteGeometricAPEngine<RNG> make(process);
    make.withBrownianBridge(c.brownianBridge)
        .withAntitheticVariate(c.antithetic)
        .withSeed(BigNatural(c.seed));
    if (c.samples > 0)
        make.withSamples(Size(c.samples));
    else
        make.withAbsoluteTolerance(c.tolerance);
    if (c.maxSamples > 0)
        make.withMaxSamples(Size(c.maxSamples));

    DiscreteAveragingAsianOption opt(
        Average::Geometric, c.runningAccumulator, Size(c.pastFixings),
        dates(*c.fixings),
        ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
        ext::make_shared<EuropeanExercise>(
            kToday + static_cast<Date::serial_type>(c.expiryDays)));
    opt.setPricingEngine(make);

    Obj in;
    describeBsmMarket(in, c.s0, c.r, c.q, c.vol);
    describeCommon(in, "MCDiscreteGeometricAPEngine", c.rng, *c.fixings,
                   c.expiryDays, c.type, c.strike, "Geometric",
                   c.runningAccumulator, c.pastFixings);
    describeMc(in, c.brownianBridge, c.antithetic, false, c.samples,
               c.tolerance, c.maxSamples, c.seed, -1, -1);

    Obj ex;
    recordResults(ex, opt, opt.NPV());
    addCase(c.name, in, ex);
}

void geometricAPCases() {
    // running PRODUCT of six past fixings at 103 each: 103^6.
    const Real kRunningProduct = 103.0 * 103.0 * 103.0 * 103.0 * 103.0 * 103.0;
    const BsCfg cfgs[] = {
        {"geomap_ps_call_atm", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 100.0, 1.0, 0, false, false, false, 8191, -1.0, -1, 42},
        {"geomap_ps_put_atm", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Put, 100.0, 1.0, 0, false, false, false, 8191, -1.0, -1, 42},
        {"geomap_ps_call_atm_seed7", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 1.0, 0, false, false, false, 8191, -1.0, -1,
         7},
        {"geomap_ps_call_otm", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 130.0, 1.0, 0, false, false, false, 8191, -1.0, -1, 42},
        // path[0] joins the geometric product when the first mandatory time is 0
        {"geomap_ps_call_t0fix", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12WithT0,
         330, Option::Call, 100.0, 1.0, 0, false, false, false, 8191, -1.0, -1,
         42},
        // seasoned: running PRODUCT, not sum
        {"geomap_ps_call_seasoned", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix6,
         360, Option::Call, 100.0, kRunningProduct, 6, false, false, false,
         8191, -1.0, -1, 42},
        {"geomap_ps_put_seasoned", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix6, 360,
         Option::Put, 100.0, kRunningProduct, 6, false, false, false, 8191,
         -1.0, -1, 42},
        {"geomap_ps_call_antithetic", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 1.0, 0, false, true, false, 8191, -1.0, -1,
         42},
        {"geomap_ps_call_bb", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 100.0, 1.0, 0, true, false, false, 8191, -1.0, -1, 42},
        {"geomap_ps_call_exercise_after_fixings", "pseudo", 100.0, 0.05, 0.0,
         0.20, &kFix12, 400, Option::Call, 100.0, 1.0, 0, false, false, false,
         8191, -1.0, -1, 42},
        {"geomap_ps_call_tol_0p20", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 1.0, 0, false, false, false, -1, 0.20, -1,
         42},
        {"geomap_ps_call_zero_vol", "pseudo", 100.0, 0.05, 0.0, 0.0, &kFix12,
         360, Option::Call, 100.0, 1.0, 0, false, false, false, 1023, -1.0, -1,
         42},
    };
    for (const auto& c : cfgs)
        runGeometricAP<PseudoRandom>(c);

    const BsCfg ldCfgs[] = {
        {"geomap_ld_call_atm", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 1.0, 0, false, false, false, 8191, -1.0, -1,
         42},
        {"geomap_ld_call_bb", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 100.0, 1.0, 0, true, false, false, 8191, -1.0, -1,
         42},
    };
    for (const auto& c : ldCfgs)
        runGeometricAP<LowDiscrepancy>(c);
}

// ===========================================================================
// 3. MCDiscreteArithmeticASEngine (Black-Scholes, SingleVariate)
//    includeExerciseDate = true -> the grid may carry one extra, non-fixing
//    point at the end, and ArithmeticASOPathPricer must exclude it.
// ===========================================================================

template <class RNG>
void runArithmeticAS(const BsCfg& c) {
    auto process = bsm(c.s0, c.r, c.q, c.vol);
    MakeMCDiscreteArithmeticASEngine<RNG> make(process);
    make.withBrownianBridge(c.brownianBridge)
        .withAntitheticVariate(c.antithetic)
        .withSeed(BigNatural(c.seed));
    if (c.samples > 0)
        make.withSamples(Size(c.samples));
    else
        make.withAbsoluteTolerance(c.tolerance);
    if (c.maxSamples > 0)
        make.withMaxSamples(Size(c.maxSamples));

    DiscreteAveragingAsianOption opt(
        Average::Arithmetic, c.runningAccumulator, Size(c.pastFixings),
        dates(*c.fixings),
        // The strike of an average-strike option is ignored by the path
        // pricer (the average IS the strike); keep it at 0 to make that
        // explicit.
        ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
        ext::make_shared<EuropeanExercise>(
            kToday + static_cast<Date::serial_type>(c.expiryDays)));
    opt.setPricingEngine(make);

    Obj in;
    describeBsmMarket(in, c.s0, c.r, c.q, c.vol);
    describeCommon(in, "MCDiscreteArithmeticASEngine", c.rng, *c.fixings,
                   c.expiryDays, c.type, c.strike, "Arithmetic",
                   c.runningAccumulator, c.pastFixings);
    describeMc(in, c.brownianBridge, c.antithetic, false, c.samples,
               c.tolerance, c.maxSamples, c.seed, -1, -1);

    Obj ex;
    recordResults(ex, opt, opt.NPV());
    addCase(c.name, in, ex);
}

void arithmeticASCases() {
    const BsCfg cfgs[] = {
        // exercise == last fixing: nothing appended, fixingCount stays Null
        {"arithas_ps_call_atexpiry", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 0.0, 0.0, 0, false, false, false, 8191, -1.0, -1,
         42},
        {"arithas_ps_put_atexpiry", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Put, 0.0, 0.0, 0, false, false, false, 8191, -1.0, -1, 42},
        {"arithas_ps_call_atexpiry_seed7", "pseudo", 100.0, 0.05, 0.0, 0.20,
         &kFix12, 360, Option::Call, 0.0, 0.0, 0, false, false, false, 8191,
         -1.0, -1, 7},
        // exercise AFTER the last fixing: the exercise time is appended to the
        // grid and excluded from the average (fixingCount = grid.size()-1)
        {"arithas_ps_call_exercise_after_fixings", "pseudo", 100.0, 0.05, 0.0,
         0.20, &kFix12, 400, Option::Call, 0.0, 0.0, 0, false, false, false,
         8191, -1.0, -1, 42},
        {"arithas_ps_put_exercise_after_fixings", "pseudo", 100.0, 0.05, 0.0,
         0.20, &kFix12, 400, Option::Put, 0.0, 0.0, 0, false, false, false, 8191,
         -1.0, -1, 42},
        // a fixing on the evaluation date
        {"arithas_ps_call_t0fix", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12WithT0,
         330, Option::Call, 0.0, 0.0, 0, false, false, false, 8191, -1.0, -1,
         42},
        {"arithas_ps_call_t0fix_exercise_after_fixings", "pseudo", 100.0, 0.05,
         0.0, 0.20, &kFix12WithT0, 400, Option::Call, 0.0, 0.0, 0, false, false,
         false, 8191, -1.0, -1, 42},
        // seasoned average strike
        {"arithas_ps_call_seasoned", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix6,
         360, Option::Call, 0.0, 618.0, 6, false, false, false, 8191, -1.0, -1,
         42},
        {"arithas_ps_call_seasoned_exercise_after_fixings", "pseudo", 100.0,
         0.05, 0.0, 0.20, &kFix6, 400, Option::Call, 0.0, 618.0, 6, false, false,
         false, 8191, -1.0, -1, 42},
        {"arithas_ps_call_antithetic", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 0.0, 0.0, 0, false, true, false, 8191, -1.0, -1, 42},
        {"arithas_ps_call_bb", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12, 360,
         Option::Call, 0.0, 0.0, 0, true, false, false, 8191, -1.0, -1, 42},
        {"arithas_ps_call_tol_0p20", "pseudo", 100.0, 0.05, 0.0, 0.20, &kFix12,
         360, Option::Call, 0.0, 0.0, 0, false, false, false, -1, 0.20, -1, 42},
    };
    for (const auto& c : cfgs)
        runArithmeticAS<PseudoRandom>(c);

    const BsCfg ldCfgs[] = {
        {"arithas_ld_call_atexpiry", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20,
         &kFix12, 360, Option::Call, 0.0, 0.0, 0, false, false, false, 8191,
         -1.0, -1, 42},
        {"arithas_ld_call_exercise_after_fixings", "lowdiscrepancy", 100.0, 0.05,
         0.0, 0.20, &kFix12, 400, Option::Call, 0.0, 0.0, 0, false, false, false,
         8191, -1.0, -1, 42},
    };
    for (const auto& c : ldCfgs)
        runArithmeticAS<LowDiscrepancy>(c);
}

// ===========================================================================
// 4/5. Heston MC engines (MultiVariate)
// ===========================================================================

struct HestonCfg {
    const char* name;
    const char* rng;
    const std::vector<long long>* fixings;
    long long expiryDays;
    Option::Type type;
    Real strike;
    Real runningAccumulator;
    long long pastFixings;
    bool antithetic, controlVariate;
    long long samples;
    Real tolerance;
    long long maxSamples;
    long long seed;
    long long steps, stepsPerYear; // -1 == Null<Size>()
};

template <class RNG>
void runArithmeticAPHeston(const HestonCfg& c) {
    auto process = heston(kHeston);
    MakeMCDiscreteArithmeticAPHestonEngine<RNG> make(process);
    make.withAntitheticVariate(c.antithetic)
        .withControlVariate(c.controlVariate)
        .withSeed(BigNatural(c.seed));
    if (c.steps > 0)
        make.withSteps(Size(c.steps));
    if (c.stepsPerYear > 0)
        make.withStepsPerYear(Size(c.stepsPerYear));
    if (c.samples > 0)
        make.withSamples(Size(c.samples));
    else
        make.withAbsoluteTolerance(c.tolerance);
    if (c.maxSamples > 0)
        make.withMaxSamples(Size(c.maxSamples));

    DiscreteAveragingAsianOption opt(
        Average::Arithmetic, c.runningAccumulator, Size(c.pastFixings),
        dates(*c.fixings),
        ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
        ext::make_shared<EuropeanExercise>(
            kToday + static_cast<Date::serial_type>(c.expiryDays)));
    opt.setPricingEngine(make);

    Obj in;
    describeHestonMarket(in, kHeston);
    describeCommon(in, "MCDiscreteArithmeticAPHestonEngine", c.rng, *c.fixings,
                   c.expiryDays, c.type, c.strike, "Arithmetic",
                   c.runningAccumulator, c.pastFixings);
    describeMc(in, false, c.antithetic, c.controlVariate, c.samples,
               c.tolerance, c.maxSamples, c.seed, c.steps, c.stepsPerYear);

    Obj ex;
    recordResults(ex, opt, opt.NPV());
    addCase(c.name, in, ex);
}

template <class RNG>
void runGeometricAPHeston(const HestonCfg& c) {
    auto process = heston(kHeston);
    MakeMCDiscreteGeometricAPHestonEngine<RNG> make(process);
    make.withAntitheticVariate(c.antithetic).withSeed(BigNatural(c.seed));
    if (c.steps > 0)
        make.withSteps(Size(c.steps));
    if (c.stepsPerYear > 0)
        make.withStepsPerYear(Size(c.stepsPerYear));
    if (c.samples > 0)
        make.withSamples(Size(c.samples));
    else
        make.withAbsoluteTolerance(c.tolerance);
    if (c.maxSamples > 0)
        make.withMaxSamples(Size(c.maxSamples));

    DiscreteAveragingAsianOption opt(
        Average::Geometric, c.runningAccumulator, Size(c.pastFixings),
        dates(*c.fixings),
        ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
        ext::make_shared<EuropeanExercise>(
            kToday + static_cast<Date::serial_type>(c.expiryDays)));
    opt.setPricingEngine(make);

    Obj in;
    describeHestonMarket(in, kHeston);
    describeCommon(in, "MCDiscreteGeometricAPHestonEngine", c.rng, *c.fixings,
                   c.expiryDays, c.type, c.strike, "Geometric",
                   c.runningAccumulator, c.pastFixings);
    describeMc(in, false, c.antithetic, false, c.samples, c.tolerance,
               c.maxSamples, c.seed, c.steps, c.stepsPerYear);

    Obj ex;
    recordResults(ex, opt, opt.NPV());
    addCase(c.name, in, ex);
}

void hestonCases() {
    const Real kRunningProduct = 103.0 * 103.0 * 103.0 * 103.0 * 103.0 * 103.0;

    const HestonCfg arithCfgs[] = {
        {"arithapheston_ps_call_atm", "pseudo", &kFix12, 360, Option::Call,
         100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"arithapheston_ps_put_atm", "pseudo", &kFix12, 360, Option::Put, 100.0,
         0.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"arithapheston_ps_call_atm_seed7", "pseudo", &kFix12, 360, Option::Call,
         100.0, 0.0, 0, false, false, 4095, -1.0, -1, 7, -1, -1},
        {"arithapheston_ps_call_t0fix", "pseudo", &kFix12WithT0, 330,
         Option::Call, 100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"arithapheston_ps_call_seasoned", "pseudo", &kFix6, 360, Option::Call,
         100.0, 618.0, 6, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"arithapheston_ps_call_antithetic", "pseudo", &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, true, false, 4095, -1.0, -1, 42, -1, -1},
        // extra time steps between fixings: the non-fixing grid points must
        // NOT enter the average
        {"arithapheston_ps_call_steps24", "pseudo", &kFix12, 360, Option::Call,
         100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, 24, -1},
        {"arithapheston_ps_call_steps36", "pseudo", &kFix12, 360, Option::Call,
         100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, 36, -1},
        {"arithapheston_ps_call_stepsperyear24", "pseudo", &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, -1, 24},
        {"arithapheston_ps_call_stepsperyear52", "pseudo", &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, -1, 52},
        // exercise strictly after the last fixing (includeExerciseDate is
        // false here, so the grid still ends at the last fixing)
        {"arithapheston_ps_call_exercise_after_fixings", "pseudo", &kFix12, 400,
         Option::Call, 100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        // control variate: analytic discrete geometric Heston + geometric
        // Heston path pricer
        {"arithapheston_ps_call_cv", "pseudo", &kFix12, 360, Option::Call, 100.0,
         0.0, 0, false, true, 2047, -1.0, -1, 42, -1, -1},
        {"arithapheston_ps_call_cv_steps24", "pseudo", &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, false, true, 2047, -1.0, -1, 42, 24, -1},
        {"arithapheston_ps_put_cv", "pseudo", &kFix12, 360, Option::Put, 100.0,
         0.0, 0, false, true, 2047, -1.0, -1, 42, -1, -1},
        {"arithapheston_ps_call_tol_0p30", "pseudo", &kFix12, 360, Option::Call,
         100.0, 0.0, 0, false, false, -1, 0.30, -1, 42, -1, -1},
    };
    for (const auto& c : arithCfgs)
        runArithmeticAPHeston<PseudoRandom>(c);

    const HestonCfg arithLdCfgs[] = {
        {"arithapheston_ld_call_atm", "lowdiscrepancy", &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"arithapheston_ld_call_steps24", "lowdiscrepancy", &kFix12, 360,
         Option::Call, 100.0, 0.0, 0, false, false, 4095, -1.0, -1, 42, 24, -1},
    };
    for (const auto& c : arithLdCfgs)
        runArithmeticAPHeston<LowDiscrepancy>(c);

    const HestonCfg geomCfgs[] = {
        {"geomapheston_ps_call_atm", "pseudo", &kFix12, 360, Option::Call, 100.0,
         1.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"geomapheston_ps_put_atm", "pseudo", &kFix12, 360, Option::Put, 100.0,
         1.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"geomapheston_ps_call_t0fix", "pseudo", &kFix12WithT0, 330,
         Option::Call, 100.0, 1.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"geomapheston_ps_call_seasoned", "pseudo", &kFix6, 360, Option::Call,
         100.0, kRunningProduct, 6, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"geomapheston_ps_call_antithetic", "pseudo", &kFix12, 360, Option::Call,
         100.0, 1.0, 0, true, false, 4095, -1.0, -1, 42, -1, -1},
        {"geomapheston_ps_call_steps24", "pseudo", &kFix12, 360, Option::Call,
         100.0, 1.0, 0, false, false, 4095, -1.0, -1, 42, 24, -1},
        {"geomapheston_ps_call_stepsperyear24", "pseudo", &kFix12, 360,
         Option::Call, 100.0, 1.0, 0, false, false, 4095, -1.0, -1, 42, -1, 24},
        {"geomapheston_ps_call_exercise_after_fixings", "pseudo", &kFix12, 400,
         Option::Call, 100.0, 1.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
        {"geomapheston_ps_call_tol_0p30", "pseudo", &kFix12, 360, Option::Call,
         100.0, 1.0, 0, false, false, -1, 0.30, -1, 42, -1, -1},
    };
    for (const auto& c : geomCfgs)
        runGeometricAPHeston<PseudoRandom>(c);

    const HestonCfg geomLdCfgs[] = {
        {"geomapheston_ld_call_atm", "lowdiscrepancy", &kFix12, 360,
         Option::Call, 100.0, 1.0, 0, false, false, 4095, -1.0, -1, 42, -1, -1},
    };
    for (const auto& c : geomLdCfgs)
        runGeometricAPHeston<LowDiscrepancy>(c);
}

// ===========================================================================
// 6. detail::PastFixingsOnly and the other engine-level guards
// ===========================================================================

ext::shared_ptr<DiscreteAveragingAsianOption>
asianOption(Average::Type type, Real runningAccumulator, Size pastFixings,
            const std::vector<long long>& fixings, Option::Type optionType,
            Real strike, long long expiryDays) {
    return ext::make_shared<DiscreteAveragingAsianOption>(
        type, runningAccumulator, pastFixings, dates(fixings),
        ext::make_shared<PlainVanillaPayoff>(optionType, strike),
        ext::make_shared<EuropeanExercise>(
            kToday + static_cast<Date::serial_type>(expiryDays)));
}

void guardCases() {
    auto process = bsm(100.0, 0.05, 0.0, 0.20);
    auto hprocess = heston(kHeston);

    // --- detail::PastFixingsOnly -------------------------------------------
    // Every fixing strictly in the past -> fixingTimes is empty.
    probeThrow("throw_past_fixings_only_all_past",
               "MCDiscreteAveragingAsianEngineBase::timeGrid",
               "all fixing dates before the evaluation date", [&] {
                   auto opt = asianOption(Average::Arithmetic, 600.0, 6,
                                          kFixAllPast, Option::Call, 100.0, 360);
                   opt->setPricingEngine(
                       MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt->NPV();
               });
    // Exactly one surviving fixing, and it sits at t == 0.
    probeThrow("throw_past_fixings_only_single_t0",
               "MCDiscreteAveragingAsianEngineBase::timeGrid",
               "the only fixing is on the evaluation date", [&] {
                   auto opt = asianOption(Average::Arithmetic, 0.0, 0,
                                          kFixOnlyT0, Option::Call, 100.0, 360);
                   opt->setPricingEngine(
                       MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt->NPV();
               });
    // Same guard through the geometric and average-strike engines.
    probeThrow("throw_past_fixings_only_geometric",
               "MCDiscreteGeometricAPEngine", "all fixings in the past", [&] {
                   auto opt = asianOption(Average::Geometric, 1.0, 0,
                                          kFixAllPast, Option::Call, 100.0, 360);
                   opt->setPricingEngine(
                       MakeMCDiscreteGeometricAPEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt->NPV();
               });
    probeThrow("throw_past_fixings_only_averagestrike",
               "MCDiscreteArithmeticASEngine", "all fixings in the past", [&] {
                   auto opt = asianOption(Average::Arithmetic, 600.0, 6,
                                          kFixAllPast, Option::Call, 0.0, 360);
                   opt->setPricingEngine(
                       MakeMCDiscreteArithmeticASEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt->NPV();
               });
    probeThrow("throw_past_fixings_only_heston",
               "MCDiscreteArithmeticAPHestonEngine", "all fixings in the past",
               [&] {
                   auto opt = asianOption(Average::Arithmetic, 600.0, 6,
                                          kFixAllPast, Option::Call, 100.0, 360);
                   opt->setPricingEngine(
                       MakeMCDiscreteArithmeticAPHestonEngine<PseudoRandom>(
                           hprocess)
                           .withSamples(1023)
                           .withSeed(42));
                   opt->NPV();
               });

    // --- neither samples nor tolerance --------------------------------------
    probeThrow("throw_no_samples_no_tolerance", "McSimulation::calculate",
               "requiredSamples and requiredTolerance both Null", [&] {
                   auto opt = asianOption(Average::Arithmetic, 0.0, 0, kFix12,
                                          Option::Call, 100.0, 360);
                   opt->setPricingEngine(
                       MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                           .withSeed(42));
                   opt->NPV();
               });
    // --- maxSamples reached before tolerance --------------------------------
    probeThrow("throw_tolerance_maxsamples_exceeded", "McSimulation::value",
               "maxSamples reached while error is still above tolerance", [&] {
                   auto opt = asianOption(Average::Arithmetic, 0.0, 0, kFix12,
                                          Option::Call, 100.0, 360);
                   opt->setPricingEngine(
                       MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                           .withAbsoluteTolerance(1e-4)
                           .withMaxSamples(2000)
                           .withSeed(42));
                   opt->NPV();
               });

    // --- payoff / exercise guards ------------------------------------------
    probeThrow("throw_arithap_non_plain_payoff",
               "MCDiscreteArithmeticAPEngine::pathPricer", "CashOrNothingPayoff",
               [&] {
                   DiscreteAveragingAsianOption opt(
                       Average::Arithmetic, 0.0, 0, dates(kFix12),
                       ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0,
                                                             10.0),
                       ext::make_shared<EuropeanExercise>(kToday + 360));
                   opt.setPricingEngine(
                       MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt.NPV();
               });
    probeThrow("throw_arithap_american_exercise",
               "MCDiscreteArithmeticAPEngine::pathPricer", "AmericanExercise",
               [&] {
                   DiscreteAveragingAsianOption opt(
                       Average::Arithmetic, 0.0, 0, dates(kFix12),
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                       ext::make_shared<AmericanExercise>(kToday,
                                                          kToday + 360));
                   opt.setPricingEngine(
                       MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt.NPV();
               });
    probeThrow("throw_geomap_non_plain_payoff",
               "MCDiscreteGeometricAPEngine::pathPricer", "CashOrNothingPayoff",
               [&] {
                   DiscreteAveragingAsianOption opt(
                       Average::Geometric, 1.0, 0, dates(kFix12),
                       ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0,
                                                             10.0),
                       ext::make_shared<EuropeanExercise>(kToday + 360));
                   opt.setPricingEngine(
                       MakeMCDiscreteGeometricAPEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt.NPV();
               });
    probeThrow("throw_arithas_american_exercise",
               "MCDiscreteArithmeticASEngine::pathPricer", "AmericanExercise",
               [&] {
                   DiscreteAveragingAsianOption opt(
                       Average::Arithmetic, 0.0, 0, dates(kFix12),
                       ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0),
                       ext::make_shared<AmericanExercise>(kToday,
                                                          kToday + 400));
                   opt.setPricingEngine(
                       MakeMCDiscreteArithmeticASEngine<PseudoRandom>(process)
                           .withSamples(1023)
                           .withSeed(42));
                   opt.NPV();
               });
    probeThrow("throw_arithapheston_non_plain_payoff",
               "MCDiscreteArithmeticAPHestonEngine::pathPricer",
               "CashOrNothingPayoff", [&] {
                   DiscreteAveragingAsianOption opt(
                       Average::Arithmetic, 0.0, 0, dates(kFix12),
                       ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0,
                                                             10.0),
                       ext::make_shared<EuropeanExercise>(kToday + 360));
                   opt.setPricingEngine(
                       MakeMCDiscreteArithmeticAPHestonEngine<PseudoRandom>(
                           hprocess)
                           .withSamples(1023)
                           .withSeed(42));
                   opt.NPV();
               });

    // --- path-pricer constructor guards -------------------------------------
    probeThrow("throw_arithap_pathpricer_negative_strike",
               "ArithmeticAPOPathPricer", "strike < 0", [&] {
                   ArithmeticAPOPathPricer p(Option::Call, -1.0, 0.95);
               });
    probeThrow("throw_geomap_pathpricer_negative_strike",
               "GeometricAPOPathPricer", "strike < 0", [&] {
                   GeometricAPOPathPricer p(Option::Call, -1.0, 0.95);
               });
    probeThrow("throw_arithapheston_pathpricer_negative_strike",
               "ArithmeticAPOHestonPathPricer", "strike < 0", [&] {
                   ArithmeticAPOHestonPathPricer p(Option::Call, -1.0, 0.95,
                                                   std::vector<Size>{1, 2});
               });
    probeThrow("throw_geomapheston_pathpricer_negative_strike",
               "GeometricAPOHestonPathPricer", "strike < 0", [&] {
                   GeometricAPOHestonPathPricer p(Option::Call, -1.0, 0.95,
                                                  std::vector<Size>{1, 2});
               });

    // --- engine-level steps guard (unreachable through the builder) ---------
    probeThrow("throw_arithapheston_engine_both_steps",
               "MCDiscreteArithmeticAPHestonEngine ctor",
               "timeSteps and timeStepsPerYear both given", [&] {
                   auto e = ext::make_shared<
                       MCDiscreteArithmeticAPHestonEngine<PseudoRandom>>(
                       hprocess, false, 1023, Null<Real>(), Null<Size>(), 42, 24,
                       24);
               });
    probeThrow("throw_geomapheston_engine_both_steps",
               "MCDiscreteGeometricAPHestonEngine ctor",
               "timeSteps and timeStepsPerYear both given", [&] {
                   auto e = ext::make_shared<
                       MCDiscreteGeometricAPHestonEngine<PseudoRandom>>(
                       hprocess, false, 1023, Null<Real>(), Null<Size>(), 42, 24,
                       24);
               });

    // NB: `MCDiscreteAveragingAsianEngineBase::controlVariateValue`'s
    // "engine does not provide control variation pricing engine" guard is
    // UNREACHABLE for every concrete engine in this cluster: the two engines
    // that do not override `controlPricingEngine` (MCDiscreteGeometricAPEngine,
    // MCDiscreteArithmeticASEngine, MCDiscreteGeometricAPHestonEngine)
    // hard-wire `controlVariate = false` in their constructors, so
    // `McSimulation::calculate` never calls it. Nothing to pin.
}

// ===========================================================================
// 7. MakeMC* builder validation
// ===========================================================================

void builderCases() {
    auto process = bsm(100.0, 0.05, 0.0, 0.20);
    auto hprocess = heston(kHeston);

    // --- the three Black-Scholes builders -----------------------------------
    probeThrow("make_arithap_samples_after_tolerance",
               "MakeMCDiscreteArithmeticAPEngine", "withSamples after tolerance",
               [&] {
                   MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                       .withAbsoluteTolerance(0.02)
                       .withSamples(1023);
               });
    probeThrow("make_arithap_tolerance_after_samples",
               "MakeMCDiscreteArithmeticAPEngine", "tolerance after withSamples",
               [&] {
                   MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                       .withSamples(1023)
                       .withAbsoluteTolerance(0.02);
               });
    probeThrow("make_arithap_tolerance_lowdiscrepancy",
               "MakeMCDiscreteArithmeticAPEngine",
               "withAbsoluteTolerance on LowDiscrepancy", [&] {
                   MakeMCDiscreteArithmeticAPEngine<LowDiscrepancy>(process)
                       .withAbsoluteTolerance(0.02);
               });
    probeThrow("make_arithap_defaults_ok", "MakeMCDiscreteArithmeticAPEngine",
               "withSamples only (must not throw)", [&] {
                   ext::shared_ptr<PricingEngine> e =
                       MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                           .withSamples(1023);
               });

    probeThrow("make_geomap_samples_after_tolerance",
               "MakeMCDiscreteGeometricAPEngine", "withSamples after tolerance",
               [&] {
                   MakeMCDiscreteGeometricAPEngine<PseudoRandom>(process)
                       .withAbsoluteTolerance(0.02)
                       .withSamples(1023);
               });
    probeThrow("make_geomap_tolerance_after_samples",
               "MakeMCDiscreteGeometricAPEngine", "tolerance after withSamples",
               [&] {
                   MakeMCDiscreteGeometricAPEngine<PseudoRandom>(process)
                       .withSamples(1023)
                       .withAbsoluteTolerance(0.02);
               });
    probeThrow("make_geomap_tolerance_lowdiscrepancy",
               "MakeMCDiscreteGeometricAPEngine",
               "withAbsoluteTolerance on LowDiscrepancy", [&] {
                   MakeMCDiscreteGeometricAPEngine<LowDiscrepancy>(process)
                       .withAbsoluteTolerance(0.02);
               });

    probeThrow("make_arithas_samples_after_tolerance",
               "MakeMCDiscreteArithmeticASEngine", "withSamples after tolerance",
               [&] {
                   MakeMCDiscreteArithmeticASEngine<PseudoRandom>(process)
                       .withAbsoluteTolerance(0.02)
                       .withSamples(1023);
               });
    probeThrow("make_arithas_tolerance_after_samples",
               "MakeMCDiscreteArithmeticASEngine", "tolerance after withSamples",
               [&] {
                   MakeMCDiscreteArithmeticASEngine<PseudoRandom>(process)
                       .withSamples(1023)
                       .withAbsoluteTolerance(0.02);
               });
    probeThrow("make_arithas_tolerance_lowdiscrepancy",
               "MakeMCDiscreteArithmeticASEngine",
               "withAbsoluteTolerance on LowDiscrepancy", [&] {
                   MakeMCDiscreteArithmeticASEngine<LowDiscrepancy>(process)
                       .withAbsoluteTolerance(0.02);
               });

    // --- the two Heston builders: steps guards fire in the SETTERS ---------
    probeThrow("make_arithapheston_steps_after_stepsperyear",
               "MakeMCDiscreteArithmeticAPHestonEngine",
               "withSteps after withStepsPerYear", [&] {
                   MakeMCDiscreteArithmeticAPHestonEngine<PseudoRandom>(hprocess)
                       .withStepsPerYear(12)
                       .withSteps(24);
               });
    probeThrow("make_arithapheston_stepsperyear_after_steps",
               "MakeMCDiscreteArithmeticAPHestonEngine",
               "withStepsPerYear after withSteps", [&] {
                   MakeMCDiscreteArithmeticAPHestonEngine<PseudoRandom>(hprocess)
                       .withSteps(24)
                       .withStepsPerYear(12);
               });
    probeThrow("make_arithapheston_samples_after_tolerance",
               "MakeMCDiscreteArithmeticAPHestonEngine",
               "withSamples after tolerance", [&] {
                   MakeMCDiscreteArithmeticAPHestonEngine<PseudoRandom>(hprocess)
                       .withAbsoluteTolerance(0.02)
                       .withSamples(1023);
               });
    probeThrow("make_arithapheston_tolerance_after_samples",
               "MakeMCDiscreteArithmeticAPHestonEngine",
               "tolerance after withSamples", [&] {
                   MakeMCDiscreteArithmeticAPHestonEngine<PseudoRandom>(hprocess)
                       .withSamples(1023)
                       .withAbsoluteTolerance(0.02);
               });
    probeThrow("make_arithapheston_tolerance_lowdiscrepancy",
               "MakeMCDiscreteArithmeticAPHestonEngine",
               "withAbsoluteTolerance on LowDiscrepancy", [&] {
                   MakeMCDiscreteArithmeticAPHestonEngine<LowDiscrepancy>(
                       hprocess)
                       .withAbsoluteTolerance(0.02);
               });
    // No steps at all is LEGAL for this family (one step per fixing).
    probeThrow("make_arithapheston_no_steps_ok",
               "MakeMCDiscreteArithmeticAPHestonEngine",
               "neither withSteps nor withStepsPerYear (must not throw)", [&] {
                   ext::shared_ptr<PricingEngine> e =
                       MakeMCDiscreteArithmeticAPHestonEngine<PseudoRandom>(
                           hprocess)
                           .withSamples(1023);
               });

    probeThrow("make_geomapheston_steps_after_stepsperyear",
               "MakeMCDiscreteGeometricAPHestonEngine",
               "withSteps after withStepsPerYear", [&] {
                   MakeMCDiscreteGeometricAPHestonEngine<PseudoRandom>(hprocess)
                       .withStepsPerYear(12)
                       .withSteps(24);
               });
    probeThrow("make_geomapheston_stepsperyear_after_steps",
               "MakeMCDiscreteGeometricAPHestonEngine",
               "withStepsPerYear after withSteps", [&] {
                   MakeMCDiscreteGeometricAPHestonEngine<PseudoRandom>(hprocess)
                       .withSteps(24)
                       .withStepsPerYear(12);
               });
    probeThrow("make_geomapheston_samples_after_tolerance",
               "MakeMCDiscreteGeometricAPHestonEngine",
               "withSamples after tolerance", [&] {
                   MakeMCDiscreteGeometricAPHestonEngine<PseudoRandom>(hprocess)
                       .withAbsoluteTolerance(0.02)
                       .withSamples(1023);
               });
    probeThrow("make_geomapheston_tolerance_after_samples",
               "MakeMCDiscreteGeometricAPHestonEngine",
               "tolerance after withSamples", [&] {
                   MakeMCDiscreteGeometricAPHestonEngine<PseudoRandom>(hprocess)
                       .withSamples(1023)
                       .withAbsoluteTolerance(0.02);
               });
    probeThrow("make_geomapheston_tolerance_lowdiscrepancy",
               "MakeMCDiscreteGeometricAPHestonEngine",
               "withAbsoluteTolerance on LowDiscrepancy", [&] {
                   MakeMCDiscreteGeometricAPHestonEngine<LowDiscrepancy>(
                       hprocess)
                       .withAbsoluteTolerance(0.02);
               });
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    arithmeticAPCases();
    arithmeticAPClampCase();
    geometricAPCases();
    arithmeticASCases();
    hestonCases();
    guardCases();
    builderCases();

    emitDocument();
    return 0;
}
