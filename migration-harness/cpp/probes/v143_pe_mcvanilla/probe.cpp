// migration-harness/cpp/probes/v143_pe_mcvanilla/probe.cpp
//
// Reference values for the Monte Carlo *vanilla* pricing-engine cluster of
// C++ QuantLib v1.43:
//
//   * MCEuropeanEngine / EuropeanPathPricer / MakeMCEuropeanEngine
//         ql/pricingengines/vanilla/mceuropeanengine.hpp
//   * MCAmericanEngine / AmericanPathPricer / MakeMCAmericanEngine
//         ql/pricingengines/vanilla/mcamericanengine.{hpp,cpp}
//   * MCDigitalEngine / DigitalPathPricer / MakeMCDigitalEngine
//         ql/pricingengines/vanilla/mcdigitalengine.{hpp,cpp}
//   * MCEuropeanHestonEngine / EuropeanHestonPathPricer / MakeMCEuropeanHestonEngine
//         ql/pricingengines/vanilla/mceuropeanhestonengine.hpp
//   * MCEuropeanGJRGARCHEngine / EuropeanGJRGARCHPathPricer /
//     MakeMCEuropeanGJRGARCHEngine
//         ql/pricingengines/vanilla/mceuropeangjrgarchengine.hpp
//   * MCHestonHullWhiteEngine / HestonHullWhitePathPricer /
//     MakeMCHestonHullWhiteEngine
//         ql/pricingengines/vanilla/mchestonhullwhiteengine.{hpp,cpp}
//     (and, transitively, HybridHestonHullWhiteProcess)
//
// Why the values below are EXACT, not a confidence band
// -----------------------------------------------------
// Every engine here is deterministic once the seed is fixed:
//   PseudoRandom  = InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
//                                        InverseCumulativeNormal>
//   LowDiscrepancy = InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>
// Neither consults the clock as long as `seed != 0` (seed 0 routes through
// SeedGenerator, which IS clock-derived -- no case below uses seed 0 for
// anything whose value is pinned). A correct port therefore reproduces the
// NPV and the error estimate to ~1e-14 relative. A statistical band would
// pass with the wrong RNG, the wrong path construction, or the wrong
// antithetic pairing, so nothing here is banded.
//
// What has to be pinned, and why
// ------------------------------
//  1. `MCVanillaEngine::calculate()` fills `results_.value` unconditionally
//     but `results_.errorEstimate` only `if constexpr (RNG::allowsErrorEstimate)`.
//     LowDiscrepancy has allowsErrorEstimate == 0, so `option.errorEstimate()`
//     THROWS for every LD case. That absence is pinned explicitly
//     ("has_error_estimate": false) so a port does not invent a number the
//     C++ never produces.
//  2. Antithetic pairing. `MonteCarloModel::addSamples` draws `next()`, then
//     -- only if antithetic -- `antithetic()`, which re-uses the generator's
//     *last* sequence with every Gaussian negated, and books
//     `(price + price2)/2` as ONE sample with `path.weight`. So antithetic
//     halves the sample count of underlying draws, it does not double it;
//     `samples()` still equals `requiredSamples`. A port that books two
//     samples, or that draws a fresh sequence for the antithetic leg, gets a
//     different number for every antithetic case below.
//  3. `MCVanillaEngine::timeGrid()`: `TimeGrid(t, timeSteps)` when steps are
//     given, else `TimeGrid(t, max<Size>(Size(timeStepsPerYear*t), 1))`. The
//     `Size(...)` is a truncation, and the `max(...,1)` guard is driven to
//     exactly 1 by the `stepsperyear_truncates_to_one` case (T = 1/12 y with
//     stepsPerYear = 4 gives Size(0.333) == 0).
//  4. `MCEuropeanEngine::pathPricer()` discounts with
//     `riskFreeRate()->discount(this->timeGrid().back())` -- the *grid* end
//     point `(t/steps)*steps`, NOT `discount(exerciseDate)`. Those differ in
//     the last bits, and with a non-flat (zero-curve) term structure the
//     interpolation amplifies the difference. Cases `*_zerocurve_*` use a
//     ZeroCurve precisely so a port that discounts by date instead of by grid
//     time is caught.
//  5. `McSimulation::value(tolerance, maxSamples, minSamples=1023)`: the
//     tolerance loop starts by forcing the sample count up to 1023, then
//     grows batches by
//        order    = error^2/tolerance^2
//        nextBatch= max(Size(n*order*0.8 - n), minSamples)  [capped at maxSamples-n]
//     so the terminal sample count is a deterministic function of the seed.
//     The `*_tol_*` cases pin `samples` as well as the NPV, so a port with a
//     different growth rule is caught even when it happens to converge to a
//     similar mean. `european_tol_maxsamples_exceeded` drives the
//     QL_REQUIRE(sampleNumber < maxSamples) failure.
//  6. `MakeMC*Engine` validation. These builders are not sugar:
//        - MakeMCEuropean/Digital/American: `operator shared_ptr<PricingEngine>()`
//          requires steps XOR stepsPerYear ("number of steps not given" /
//          "number of steps overspecified"). Note the ASYMMETRY that a port
//          will get wrong: MakeMCEuropeanHestonEngine and
//          MakeMCEuropeanGJRGARCHEngine reject the over-specification EARLY,
//          inside withSteps/withStepsPerYear ("number of steps per year
//          already set" / "number of steps already set"), and their
//          `operator shared_ptr` checks ONLY the "not given" half. Both
//          spellings are pinned.
//        - `withSamples` after `withAbsoluteTolerance` throws "tolerance
//          already set"; `withAbsoluteTolerance` after `withSamples` throws
//          "number of samples already set"; `withAbsoluteTolerance` on a
//          LowDiscrepancy builder throws "chosen random generator policy does
//          not allow an error estimate" -- a compile-time-constant test in
//          C++ (QL_REQUIRE(RNG::allowsErrorEstimate)) that a port must keep
//          as a runtime one.
//        - MakeMCAmericanEngine defaults: calibrationSamples_ = 2048,
//          polynomialOrder_ = 2, polynomialType_ = Monomial, seed_ = 0,
//          antitheticCalibration_ = nullopt, seedCalibration_ = Null<Size>().
//          Those defaults are exercised by `american_make_defaults`, whose
//          value must equal the explicitly-parameterised twin.
//  7. `MCLongstaffSchwartzEngine` calibration wiring:
//        seedCalibration = (seed == 0 ? 0 : seed + 1768237423)
//        antitheticVariateCalibration = antitheticVariate  (if not given)
//        nCalibrationSamples          = 2048               (if Null)
//     and the calibration pass runs BEFORE the pricing pass on its own
//     MonteCarloModel, so the pricing RNG stream starts fresh. A port that
//     shares one generator between calibration and pricing gets every
//     American number wrong. `american_seedcal_explicit` pins an explicitly
//     different calibration seed to prove the parameter is actually used.
//     `exerciseProbability` is pinned from results_.additionalResults.
//  8. `MCAmericanEngine::calculate()` clamps `results_.value` to
//     `max(0.0, value)` ONLY when controlVariate_ is on.
//     `american_cv_deep_otm_clamped` is chosen so the raw CV estimator is
//     negative and the clamp actually fires (expected value exactly 0).
//  9. `DigitalPathPricer` (mcdigitalengine.cpp:36-109) carries the
//     Beaglehole-Dybvig-Zhou Brownian-bridge hit correction:
//        x   = log(path[i+1]/path[i])
//        vol = diffProcess->diffusion(timeGrid[i+1], exp(log_asset_price))
//        y   = log_asset_price + 0.5*(x +/- sqrt(x*x - 2*vol*vol*dt*log(u)))
//     with u drawn from a SEPARATE, hard-coded generator
//        PseudoRandom::ursg_type(grid.size()-1, PseudoRandom::urng_type(76))
//     -- seed 76, *uniforms* (not Gaussians), one fresh sequence per path,
//     independent of the engine seed. Getting that generator wrong is
//     invisible in a band test and fatal here. Note the Call branch uses
//     `log(1-u[i])` and the Put branch `log(u[i])`: NOT symmetric.
//     The vol is evaluated at `timeGrid[i+1]` with the *initial* asset value
//     of the step (the commented-out alternative in the C++ source is not
//     used). On a hit the payoff discounts at `timeGrid[i+1]` unless
//     `exercise->payoffAtExpiry()`, in which case it discounts at
//     `timeGrid.back()`. `digital_*_coarse_*` uses 4 steps over a year so the
//     bridge correction dominates; `digital_*_fine_*` uses 90 steps/yr, the
//     upstream test-suite setting, where it is a small correction. Both are
//     pinned so a port cannot tune one and break the other.
// 10. `MCEuropeanHestonEngine` and `MCEuropeanGJRGARCHEngine` are
//     MultiVariate: MultiPathGenerator, dimension = factors*(grid.size()-1),
//     Gaussians consumed `factors` at a time per step, and Brownian bridge is
//     unconditionally unavailable (MultiPathGenerator QL_FAILs). Their path
//     pricers read `multiPath[0].back()` -- asset only, variance ignored.
//     The answer depends on the process's `evolve()` override:
//     HestonProcess defaults to QuadraticExponentialMartingale (NOT the
//     generic apply(expectation, stdDeviation*dw)); GJRGARCHProcess defaults
//     to FullTruncation. Every non-default discretization that changes the
//     answer is pinned too.
// 11. `MCHestonHullWhiteEngine`:
//        - pathPricer requires European exercise and prices
//          payoff(states[0]) / process->numeraire(exerciseTime, states)
//          -- a STOCHASTIC discount, so the terminal short rate matters;
//        - controlPathGenerator() builds a SECOND HybridHestonHullWhiteProcess
//          with corrEquityShortRate = 0 and the SAME seed as the pricing
//          generator, i.e. the control paths are a different process driven
//          by the same numbers. A port that re-uses the pricing generator, or
//          that re-seeds, will not reproduce any control-variate case;
//        - controlVariateValue() comes from AnalyticHestonHullWhiteEngine
//          (integrationOrder 144);
//        - calculate() clamps to max(0, value) when controlVariate_ is on;
//        - HybridHestonHullWhiteProcess::Discretization (Euler vs
//          BSMHullWhite, the default) changes the answer -- both pinned.
//
// Emits JSON on stdout; nothing else may be printed.

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
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/methods/montecarlo/longstaffschwartzpathpricer.hpp>
#include <ql/methods/montecarlo/lsmbasissystem.hpp>
#include <ql/methods/montecarlo/montecarlomodel.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/mcamericanengine.hpp>
#include <ql/pricingengines/vanilla/mcdigitalengine.hpp>
#include <ql/pricingengines/vanilla/mceuropeanengine.hpp>
#include <ql/pricingengines/vanilla/mceuropeangjrgarchengine.hpp>
#include <ql/pricingengines/vanilla/mceuropeanhestonengine.hpp>
#include <ql/pricingengines/vanilla/mchestonhullwhiteengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/gjrgarchprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/processes/hullwhiteprocess.hpp>
#include <ql/processes/hybridhestonhullwhiteprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
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
// Market. 15 May 2025 -> 15 May 2026 is exactly 365 days, so with
// Actual365Fixed the exercise time is exactly 1.0 and no day-count rounding
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

// A genuinely non-flat curve: linear-in-zero-rate interpolation, so
// discount(t) is NOT exp(-r t) and the difference between discounting at
// timeGrid().back() and at the exercise date is observable.
Handle<YieldTermStructure> zeroCurve() {
    std::vector<Date> dates = {kToday, kToday + 90, kToday + 180, kToday + 365,
                               kToday + 730};
    std::vector<Rate> rates = {0.02, 0.028, 0.035, 0.045, 0.05};
    return Handle<YieldTermStructure>(
        ext::make_shared<ZeroCurve>(dates, rates, dayCounter()));
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

ext::shared_ptr<GeneralizedBlackScholesProcess>
bsmZeroCurve(Real s0, Rate q, Volatility v) {
    return ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(s0)), flatCurve(q),
        zeroCurve(), flatVol(v));
}

// Describe the market in `inputs` so the Python test reconstructs the case
// rather than restating constants.
void describeBsm(Obj& in, Real s0, Rate r, Rate q, Volatility v, int days,
                 const std::string& curve = "flat") {
    in.n("s0", s0);
    in.n("r", r);
    in.n("q", q);
    in.n("vol", v);
    in.i("expiry_days", days);
    in.s("curve", curve);
    in.s("day_counter", "Actual365Fixed");
    in.s("calendar", "NullCalendar");
    in.i("eval_year", kToday.year());
    in.i("eval_month", static_cast<int>(kToday.month()));
    in.i("eval_day", kToday.dayOfMonth());
}

void describeEngine(Obj& in, const std::string& rng, long long steps,
                    long long stepsPerYear, bool brownianBridge, bool antithetic,
                    long long samples, Real tolerance, long long maxSamples,
                    long long seed) {
    in.s("rng", rng);
    in.i("steps", steps);            // -1 == Null<Size>()
    in.i("steps_per_year", stepsPerYear);
    in.b("brownian_bridge", brownianBridge);
    in.b("antithetic", antithetic);
    in.i("samples", samples);        // -1 == Null<Size>()
    in.n("tolerance", tolerance);    // NaN-free sentinel: -1 == Null<Real>()
    in.i("max_samples", maxSamples); // -1 == Null<Size>()
    in.i("seed", seed);
}

// Record NPV + (optionally absent) error estimate + realised sample count.
void recordResults(Obj& ex, VanillaOption& opt, Real npv) {
    ex.n("npv", npv);
    try {
        ex.n("error_estimate", opt.errorEstimate());
        ex.b("has_error_estimate", true);
    } catch (const std::exception&) {
        ex.b("has_error_estimate", false);
    }
}

// ===========================================================================
// 1. MCEuropeanEngine / EuropeanPathPricer / MakeMCEuropeanEngine
// ===========================================================================

struct EuroCfg {
    const char* name;
    const char* rng;    // "pseudo" | "lowdiscrepancy"
    Real s0, r, q, vol;
    int days;
    Option::Type type;
    Real strike;
    long long steps, stepsPerYear;
    bool brownianBridge, antithetic;
    long long samples;
    long long seed;
    const char* curve; // "flat" | "zerocurve"
};

template <class RNG>
void runEuropean(const EuroCfg& c) {
    auto process = std::string(c.curve) == "zerocurve"
                       ? bsmZeroCurve(c.s0, c.q, c.vol)
                       : bsm(c.s0, c.r, c.q, c.vol);

    MakeMCEuropeanEngine<RNG> make(process);
    if (c.steps > 0)
        make.withSteps(Size(c.steps));
    if (c.stepsPerYear > 0)
        make.withStepsPerYear(Size(c.stepsPerYear));
    make.withBrownianBridge(c.brownianBridge)
        .withAntitheticVariate(c.antithetic)
        .withSamples(Size(c.samples))
        .withSeed(BigNatural(c.seed));

    VanillaOption opt(ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                      ext::make_shared<EuropeanExercise>(kToday + c.days));
    opt.setPricingEngine(make);

    Obj in;
    describeBsm(in, c.s0, c.r, c.q, c.vol, c.days, c.curve);
    in.s("payoff", "PlainVanilla");
    in.s("option_type", c.type == Option::Call ? "Call" : "Put");
    in.n("strike", c.strike);
    describeEngine(in, c.rng, c.steps, c.stepsPerYear, c.brownianBridge,
                   c.antithetic, c.samples, -1.0, -1, c.seed);

    Obj ex;
    const Real npv = opt.NPV();
    recordResults(ex, opt, npv);
    addCase(c.name, in, ex);
}

void europeanCases() {
    // Textbook BSM: S=K=100, r=5%, q=0, sigma=20%, T=1y.
    const EuroCfg cfgs[] = {
        // --- PseudoRandom, one step, no variance reduction --------------
        {"european_ps_call_atm", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 100.0, 1, -1, false, false, 8191, 42, "flat"},
        {"european_ps_put_atm", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Put, 100.0, 1, -1, false, false, 8191, 42, "flat"},
        // Different seed on an otherwise identical case: proves the seed
        // reaches the generator rather than being swallowed.
        {"european_ps_call_atm_seed7", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 100.0, 1, -1, false, false, 8191, 7, "flat"},
        // --- deep ITM / deep OTM ---------------------------------------
        {"european_ps_call_deep_itm", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 40.0, 1, -1, false, false, 8191, 42, "flat"},
        {"european_ps_call_deep_otm", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 250.0, 1, -1, false, false, 8191, 42, "flat"},
        {"european_ps_put_deep_otm", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Put, 25.0, 1, -1, false, false, 8191, 42, "flat"},
        // --- antithetic on ---------------------------------------------
        {"european_ps_call_antithetic", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 100.0, 1, -1, false, true, 8191, 42, "flat"},
        // --- multi-step, and multi-step + brownian bridge ---------------
        {"european_ps_call_12steps", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 100.0, 12, -1, false, false, 2047, 42, "flat"},
        {"european_ps_call_12steps_bb", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 100.0, 12, -1, true, false, 2047, 42, "flat"},
        {"european_ps_call_12steps_bb_antithetic", "pseudo", 100.0, 0.05, 0.0,
         0.20, 365, Option::Call, 100.0, 12, -1, true, true, 2047, 42, "flat"},
        // --- stepsPerYear instead of steps ------------------------------
        {"european_ps_call_stepsperyear", "pseudo", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 100.0, -1, 12, false, false, 2047, 42, "flat"},
        // Size(4 * 0.084931...) == 0 -> max(0,1) == 1 step. Guard driven to
        // exactly its clamp value.
        {"european_ps_call_stepsperyear_truncates_to_one", "pseudo", 100.0,
         0.05, 0.0, 0.20, 31, Option::Call, 100.0, -1, 4, false, false, 2047,
         42, "flat"},
        // --- non-flat curve: discount at timeGrid().back(), not at date --
        {"european_ps_call_zerocurve_7steps", "pseudo", 100.0, 0.0, 0.01, 0.25,
         365, Option::Call, 100.0, 7, -1, false, false, 2047, 42, "zerocurve"},
        {"european_ps_put_zerocurve_7steps", "pseudo", 100.0, 0.0, 0.01, 0.25,
         365, Option::Put, 100.0, 7, -1, false, false, 2047, 42, "zerocurve"},
        // --- dividend yield != 0 (a wrong drift hides when q == 0) -------
        {"european_ps_call_with_dividend", "pseudo", 100.0, 0.05, 0.03, 0.20,
         365, Option::Call, 100.0, 4, -1, false, false, 2047, 123456789,
         "flat"},
        // --- zero vol: every path is the forward, MC == intrinsic --------
        {"european_ps_call_zero_vol", "pseudo", 100.0, 0.05, 0.0, 0.0, 365,
         Option::Call, 100.0, 1, -1, false, false, 1023, 42, "flat"},
    };
    for (const auto& c : cfgs)
        runEuropean<PseudoRandom>(c);

    // --- LowDiscrepancy: no error estimate at all ------------------------
    const EuroCfg ldCfgs[] = {
        {"european_ld_call_atm", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Call, 100.0, 1, -1, false, false, 8191, 42, "flat"},
        {"european_ld_put_atm", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Put, 100.0, 1, -1, false, false, 8191, 42, "flat"},
        {"european_ld_call_atm_seed0", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20,
         365, Option::Call, 100.0, 1, -1, false, false, 8191, 0, "flat"},
        {"european_ld_call_12steps", "lowdiscrepancy", 100.0, 0.05, 0.0, 0.20,
         365, Option::Call, 100.0, 12, -1, false, false, 2047, 42, "flat"},
        {"european_ld_call_12steps_bb", "lowdiscrepancy", 100.0, 0.05, 0.0,
         0.20, 365, Option::Call, 100.0, 12, -1, true, false, 2047, 42, "flat"},
        {"european_ld_call_antithetic", "lowdiscrepancy", 100.0, 0.05, 0.0,
         0.20, 365, Option::Call, 100.0, 1, -1, false, true, 8191, 42, "flat"},
    };
    for (const auto& c : ldCfgs)
        runEuropean<LowDiscrepancy>(c);
}

// Tolerance-driven termination: pins the realised sample count too.
void europeanToleranceCases() {
    struct TolCfg {
        const char* name;
        Real tolerance;
        long long maxSamples; // -1 == Null<Size>()
        bool antithetic;
        long long seed;
    };
    // Tolerances are chosen so the loop terminates in tens of thousands of
    // samples rather than millions: what is being pinned is the *growth rule*
    // and the terminal sample count, not the asymptotics.
    const TolCfg cfgs[] = {
        {"european_tol_0p50", 0.50, -1, false, 42},
        {"european_tol_0p25", 0.25, -1, false, 42},
        {"european_tol_0p25_antithetic", 0.25, -1, true, 42},
        {"european_tol_0p25_maxsamples", 0.25, 1000000, false, 42},
        {"european_tol_0p10_seed7", 0.10, -1, false, 7},
    };
    for (const auto& c : cfgs) {
        auto process = bsm(100.0, 0.05, 0.0, 0.20);
        MakeMCEuropeanEngine<PseudoRandom> make(process);
        make.withSteps(1)
            .withAntitheticVariate(c.antithetic)
            .withAbsoluteTolerance(c.tolerance)
            .withSeed(BigNatural(c.seed));
        if (c.maxSamples > 0)
            make.withMaxSamples(Size(c.maxSamples));

        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(make);

        Obj in;
        describeBsm(in, 100.0, 0.05, 0.0, 0.20, 365);
        in.s("payoff", "PlainVanilla");
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        describeEngine(in, "pseudo", 1, -1, false, c.antithetic, -1,
                       c.tolerance, c.maxSamples, c.seed);

        Obj ex;
        const Real npv = opt.NPV();
        recordResults(ex, opt, npv);
        addCase(c.name, in, ex);
    }

    // maxSamples reached before tolerance -> QL_REQUIRE fires.
    {
        auto process = bsm(100.0, 0.05, 0.0, 0.20);
        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(MakeMCEuropeanEngine<PseudoRandom>(process)
                                 .withSteps(1)
                                 .withAbsoluteTolerance(1e-4)
                                 .withMaxSamples(2000)
                                 .withSeed(42));
        Obj in;
        describeBsm(in, 100.0, 0.05, 0.0, 0.20, 365);
        in.s("option_type", "Call");
        in.n("strike", 100.0);
        describeEngine(in, "pseudo", 1, -1, false, false, -1, 1e-4, 2000, 42);
        Obj ex;
        bool threw = false;
        try {
            opt.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase("european_tol_maxsamples_exceeded", in, ex);
    }
}

// MakeMCEuropeanEngine validation.
void europeanBuilderCases() {
    auto process = bsm(100.0, 0.05, 0.0, 0.20);
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("builder", "MakeMCEuropeanEngine");
        in.s("scenario", what);
        Obj ex;
        bool threw = false;
        try {
            f();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };

    probe("european_make_no_steps", "neither withSteps nor withStepsPerYear",
          [&] {
              ext::shared_ptr<PricingEngine> e =
                  MakeMCEuropeanEngine<PseudoRandom>(process).withSamples(1023);
          });
    probe("european_make_both_steps", "withSteps and withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCEuropeanEngine<PseudoRandom>(process)
                .withSteps(4)
                .withStepsPerYear(12)
                .withSamples(1023);
    });
    probe("european_make_samples_after_tolerance", "withSamples after tolerance",
          [&] {
              MakeMCEuropeanEngine<PseudoRandom>(process)
                  .withAbsoluteTolerance(0.02)
                  .withSamples(1023);
          });
    probe("european_make_tolerance_after_samples", "tolerance after withSamples",
          [&] {
              MakeMCEuropeanEngine<PseudoRandom>(process)
                  .withSamples(1023)
                  .withAbsoluteTolerance(0.02);
          });
    probe("european_make_tolerance_lowdiscrepancy",
          "withAbsoluteTolerance on LowDiscrepancy", [&] {
              MakeMCEuropeanEngine<LowDiscrepancy>(process)
                  .withAbsoluteTolerance(0.02);
          });
    // Positive control: the same chain WITHOUT the conflict must succeed.
    probe("european_make_steps_only_ok", "withSteps only (must not throw)",
          [&] {
              ext::shared_ptr<PricingEngine> e =
                  MakeMCEuropeanEngine<PseudoRandom>(process)
                      .withSteps(4)
                      .withSamples(1023);
          });

    // MCVanillaEngine constructor guards, reached directly.
    probe("european_engine_zero_steps", "timeSteps == 0", [&] {
        auto e = ext::make_shared<MCEuropeanEngine<PseudoRandom>>(
            process, 0, Null<Size>(), false, false, 1023, Null<Real>(),
            Null<Size>(), 42);
    });
    probe("european_engine_zero_steps_per_year", "timeStepsPerYear == 0", [&] {
        auto e = ext::make_shared<MCEuropeanEngine<PseudoRandom>>(
            process, Null<Size>(), 0, false, false, 1023, Null<Real>(),
            Null<Size>(), 42);
    });
    // Neither tolerance nor samples -> McSimulation::calculate QL_REQUIRE.
    probe("european_engine_no_samples_no_tolerance",
          "requiredSamples and requiredTolerance both Null", [&] {
              VanillaOption opt(
                  ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                  ext::make_shared<EuropeanExercise>(kToday + 365));
              opt.setPricingEngine(ext::make_shared<MCEuropeanEngine<PseudoRandom>>(
                  process, 1, Null<Size>(), false, false, Null<Size>(),
                  Null<Real>(), Null<Size>(), 42));
              opt.NPV();
          });
    // EuropeanPathPricer rejects a negative strike.
    probe("european_pathpricer_negative_strike", "strike < 0", [&] {
        EuropeanPathPricer p(Option::Call, -1.0, 0.95);
    });
    // MCEuropeanEngine::pathPricer requires a *plain vanilla* payoff.
    probe("european_engine_non_plain_payoff", "CashOrNothingPayoff", [&] {
        VanillaOption opt(
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(MakeMCEuropeanEngine<PseudoRandom>(process)
                                 .withSteps(1)
                                 .withSamples(1023)
                                 .withSeed(42));
        opt.NPV();
    });
}

// ===========================================================================
// 2. MCAmericanEngine / AmericanPathPricer / MakeMCAmericanEngine
// ===========================================================================

struct AmerCfg {
    const char* name;
    Real s0, r, q, vol;
    int days;
    Option::Type type;
    Real strike;
    long long steps;
    bool antithetic, controlVariate;
    long long samples;
    long long seed;
    long long polynomialOrder;
    LsmBasisSystem::PolynomialType polynomialType;
    long long calibrationSamples;
    long long seedCalibration; // -1 == Null<Size>() (use the default rule)
};

const char* polyName(LsmBasisSystem::PolynomialType t) {
    switch (t) {
        case LsmBasisSystem::Monomial: return "Monomial";
        case LsmBasisSystem::Laguerre: return "Laguerre";
        case LsmBasisSystem::Hermite: return "Hermite";
        case LsmBasisSystem::Hyperbolic: return "Hyperbolic";
        case LsmBasisSystem::Legendre: return "Legendre";
        case LsmBasisSystem::Chebyshev: return "Chebyshev";
        case LsmBasisSystem::Chebyshev2nd: return "Chebyshev2nd";
    }
    return "?";
}

void americanCases() {
    const AmerCfg cfgs[] = {
        {"american_ps_put_atm", 100.0, 0.05, 0.0, 0.20, 365, Option::Put, 100.0,
         10, false, false, 2047, 42, 2, LsmBasisSystem::Monomial, 2048, -1},
        {"american_ps_put_itm", 90.0, 0.05, 0.0, 0.20, 365, Option::Put, 100.0,
         10, false, false, 2047, 42, 2, LsmBasisSystem::Monomial, 2048, -1},
        {"american_ps_call_otm", 100.0, 0.05, 0.06, 0.20, 365, Option::Call,
         120.0, 10, false, false, 2047, 42, 2, LsmBasisSystem::Monomial, 2048,
         -1},
        {"american_ps_put_atm_seed7", 100.0, 0.05, 0.0, 0.20, 365, Option::Put,
         100.0, 10, false, false, 2047, 7, 2, LsmBasisSystem::Monomial, 2048,
         -1},
        {"american_ps_put_antithetic", 100.0, 0.05, 0.0, 0.20, 365, Option::Put,
         100.0, 10, true, false, 2047, 42, 2, LsmBasisSystem::Monomial, 2048,
         -1},
        {"american_ps_put_cv", 100.0, 0.05, 0.0, 0.20, 365, Option::Put, 100.0,
         10, false, true, 2047, 42, 2, LsmBasisSystem::Monomial, 2048, -1},
        {"american_ps_put_cv_antithetic", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Put, 100.0, 10, true, true, 2047, 42, 2,
         LsmBasisSystem::Monomial, 2048, -1},
        {"american_ps_put_order4", 100.0, 0.05, 0.0, 0.20, 365, Option::Put,
         100.0, 10, false, false, 2047, 42, 4, LsmBasisSystem::Monomial, 2048,
         -1},
        {"american_ps_put_laguerre", 100.0, 0.05, 0.0, 0.20, 365, Option::Put,
         100.0, 10, false, false, 2047, 42, 3, LsmBasisSystem::Laguerre, 2048,
         -1},
        {"american_ps_put_hermite", 100.0, 0.05, 0.0, 0.20, 365, Option::Put,
         100.0, 10, false, false, 2047, 42, 3, LsmBasisSystem::Hermite, 2048,
         -1},
        {"american_ps_put_chebyshev2nd", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Put, 100.0, 10, false, false, 2047, 42, 3,
         LsmBasisSystem::Chebyshev2nd, 2048, -1},
        {"american_ps_put_calibration4096", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Put, 100.0, 10, false, false, 2047, 42, 2,
         LsmBasisSystem::Monomial, 4096, -1},
        {"american_ps_put_seedcal_explicit", 100.0, 0.05, 0.0, 0.20, 365,
         Option::Put, 100.0, 10, false, false, 2047, 42, 2,
         LsmBasisSystem::Monomial, 2048, 999},
        // Deep-OTM put with CV: the CV-adjusted mean goes negative and
        // MCAmericanEngine::calculate() clamps it to exactly 0.
        {"american_cv_deep_otm_clamped", 100.0, 0.05, 0.0, 0.10, 365,
         Option::Put, 20.0, 10, false, true, 1023, 42, 2,
         LsmBasisSystem::Monomial, 2048, -1},
    };

    for (const auto& c : cfgs) {
        auto process = bsm(c.s0, c.r, c.q, c.vol);
        MakeMCAmericanEngine<PseudoRandom> make(process);
        make.withSteps(Size(c.steps))
            .withAntitheticVariate(c.antithetic)
            .withControlVariate(c.controlVariate)
            .withSamples(Size(c.samples))
            .withSeed(BigNatural(c.seed))
            .withPolynomialOrder(Size(c.polynomialOrder))
            .withBasisSystem(c.polynomialType)
            .withCalibrationSamples(Size(c.calibrationSamples));
        if (c.seedCalibration >= 0)
            make.withSeedCalibration(BigNatural(c.seedCalibration));

        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
            ext::make_shared<AmericanExercise>(kToday, kToday + c.days));
        opt.setPricingEngine(make);

        Obj in;
        describeBsm(in, c.s0, c.r, c.q, c.vol, c.days);
        in.s("payoff", "PlainVanilla");
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        describeEngine(in, "pseudo", c.steps, -1, false, c.antithetic,
                       c.samples, -1.0, -1, c.seed);
        in.b("control_variate", c.controlVariate);
        in.i("polynomial_order", c.polynomialOrder);
        in.s("polynomial_type", polyName(c.polynomialType));
        in.i("calibration_samples", c.calibrationSamples);
        in.i("seed_calibration", c.seedCalibration);

        Obj ex;
        const Real npv = opt.NPV();
        recordResults(ex, opt, npv);
        ex.n("exercise_probability",
             opt.result<Real>("exerciseProbability"));
        addCase(c.name, in, ex);
    }

    // MakeMCAmericanEngine defaults must equal the explicit twin above
    // (order 2, Monomial, 2048 calibration samples, no antithetic/CV).
    {
        auto process = bsm(100.0, 0.05, 0.0, 0.20);
        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
            ext::make_shared<AmericanExercise>(kToday, kToday + 365));
        opt.setPricingEngine(MakeMCAmericanEngine<PseudoRandom>(process)
                                 .withSteps(10)
                                 .withSamples(2047)
                                 .withSeed(42));
        Obj in;
        describeBsm(in, 100.0, 0.05, 0.0, 0.20, 365);
        in.s("option_type", "Put");
        in.n("strike", 100.0);
        in.s("scenario", "all Make* knobs left at their defaults");
        describeEngine(in, "pseudo", 10, -1, false, false, 2047, -1.0, -1, 42);
        Obj ex;
        recordResults(ex, opt, opt.NPV());
        ex.n("exercise_probability", opt.result<Real>("exerciseProbability"));
        addCase("american_make_defaults", in, ex);
    }

    // Builder validation.
    auto process = bsm(100.0, 0.05, 0.0, 0.20);
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("builder", "MakeMCAmericanEngine");
        in.s("scenario", what);
        Obj ex;
        bool threw = false;
        try {
            f();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    probe("american_make_no_steps", "neither withSteps nor withStepsPerYear",
          [&] {
              ext::shared_ptr<PricingEngine> e =
                  MakeMCAmericanEngine<PseudoRandom>(process).withSamples(1023);
          });
    probe("american_make_both_steps", "withSteps and withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCAmericanEngine<PseudoRandom>(process)
                .withSteps(4)
                .withStepsPerYear(12)
                .withSamples(1023);
    });
    probe("american_make_samples_after_tolerance", "withSamples after tolerance",
          [&] {
              MakeMCAmericanEngine<PseudoRandom>(process)
                  .withAbsoluteTolerance(0.02)
                  .withSamples(1023);
          });
    probe("american_make_tolerance_after_samples", "tolerance after withSamples",
          [&] {
              MakeMCAmericanEngine<PseudoRandom>(process)
                  .withSamples(1023)
                  .withAbsoluteTolerance(0.02);
          });
    probe("american_make_tolerance_lowdiscrepancy",
          "withAbsoluteTolerance on LowDiscrepancy", [&] {
              MakeMCAmericanEngine<LowDiscrepancy>(process)
                  .withAbsoluteTolerance(0.02);
          });
    // AmericanPathPricer rejects the polynomial types LsmBasisSystem cannot
    // build a path basis for.
    probe("american_pathpricer_legendre_rejected",
          "AmericanPathPricer with Legendre", [&] {
              AmericanPathPricer p(
                  ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0), 2,
                  LsmBasisSystem::Legendre);
          });
    // European exercise on the American engine -> "wrong exercise given".
    probe("american_engine_european_exercise", "EuropeanExercise", [&] {
        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(MakeMCAmericanEngine<PseudoRandom>(process)
                                 .withSteps(10)
                                 .withSamples(1023)
                                 .withSeed(42));
        opt.NPV();
    });
    // payoffAtExpiry() -> "payoff at expiry not handled".
    probe("american_engine_payoff_at_expiry", "AmericanExercise(payoffAtExpiry)",
          [&] {
              VanillaOption opt(
                  ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
                  ext::make_shared<AmericanExercise>(kToday, kToday + 365,
                                                     true));
              opt.setPricingEngine(MakeMCAmericanEngine<PseudoRandom>(process)
                                       .withSteps(10)
                                       .withSamples(1023)
                                       .withSeed(42));
              opt.NPV();
          });
}

// ===========================================================================
// 3. MCDigitalEngine / DigitalPathPricer / MakeMCDigitalEngine
// ===========================================================================

struct DigitalCfg {
    const char* name;
    const char* rng;
    Real s0, r, q, vol;
    int days;
    Option::Type type;
    Real strike, cash;
    long long steps, stepsPerYear;
    bool brownianBridge, antithetic, payoffAtExpiry;
    long long samples;
    long long seed;
};

template <class RNG>
void runDigital(const DigitalCfg& c) {
    auto process = bsm(c.s0, c.r, c.q, c.vol);
    MakeMCDigitalEngine<RNG> make(process);
    if (c.steps > 0)
        make.withSteps(Size(c.steps));
    if (c.stepsPerYear > 0)
        make.withStepsPerYear(Size(c.stepsPerYear));
    make.withBrownianBridge(c.brownianBridge)
        .withAntitheticVariate(c.antithetic)
        .withSamples(Size(c.samples))
        .withSeed(BigNatural(c.seed));

    VanillaOption opt(
        ext::make_shared<CashOrNothingPayoff>(c.type, c.strike, c.cash),
        ext::make_shared<AmericanExercise>(kToday, kToday + c.days,
                                           c.payoffAtExpiry));
    opt.setPricingEngine(make);

    Obj in;
    describeBsm(in, c.s0, c.r, c.q, c.vol, c.days);
    in.s("payoff", "CashOrNothing");
    in.s("option_type", c.type == Option::Call ? "Call" : "Put");
    in.n("strike", c.strike);
    in.n("cash_payoff", c.cash);
    in.s("exercise", "American");
    in.b("payoff_at_expiry", c.payoffAtExpiry);
    describeEngine(in, c.rng, c.steps, c.stepsPerYear, c.brownianBridge,
                   c.antithetic, c.samples, -1.0, -1, c.seed);
    in.i("bridge_uniform_seed", 76);

    Obj ex;
    recordResults(ex, opt, opt.NPV());
    addCase(c.name, in, ex);
}

void digitalCases() {
    // Coarse grids first: 4 steps over a year makes the Brownian-bridge hit
    // correction dominate the answer, so a port that drops it is off by a
    // large margin rather than by a rounding.
    const DigitalCfg cfgs[] = {
        {"digital_ps_call_coarse4", "pseudo", 100.0, 0.05, 0.04, 0.20, 365,
         Option::Call, 110.0, 15.0, 4, -1, false, false, false, 2047, 1},
        {"digital_ps_put_coarse4", "pseudo", 100.0, 0.05, 0.04, 0.20, 365,
         Option::Put, 90.0, 15.0, 4, -1, false, false, false, 2047, 1},
        {"digital_ps_call_coarse2", "pseudo", 100.0, 0.05, 0.04, 0.20, 365,
         Option::Call, 110.0, 15.0, 2, -1, false, false, false, 2047, 1},
        // payoffAtExpiry flips the discount from timeGrid[i+1] to
        // timeGrid.back(); same paths, strictly smaller value.
        {"digital_ps_call_coarse4_payoff_at_expiry", "pseudo", 100.0, 0.05,
         0.04, 0.20, 365, Option::Call, 110.0, 15.0, 4, -1, false, false, true,
         2047, 1},
        {"digital_ps_put_coarse4_payoff_at_expiry", "pseudo", 100.0, 0.05, 0.04,
         0.20, 365, Option::Put, 90.0, 15.0, 4, -1, false, false, true, 2047,
         1},
        // Antithetic + brownian bridge on the coarse grid.
        {"digital_ps_call_coarse4_antithetic", "pseudo", 100.0, 0.05, 0.04,
         0.20, 365, Option::Call, 110.0, 15.0, 4, -1, false, true, false, 2047,
         1},
        {"digital_ps_call_coarse4_bb", "pseudo", 100.0, 0.05, 0.04, 0.20, 365,
         Option::Call, 110.0, 15.0, 4, -1, true, false, false, 2047, 1},
        // Already-in-the-money at t0: the very first step hits, so the
        // discount is timeGrid[1] for every path.
        {"digital_ps_call_already_itm", "pseudo", 120.0, 0.05, 0.04, 0.20, 365,
         Option::Call, 100.0, 15.0, 4, -1, false, false, false, 1023, 1},
        // Upstream test-suite geometry (90 steps/yr, seed 1), where the
        // bridge correction is small.
        {"digital_ps_call_fine90", "pseudo", 100.0, 0.04, 0.00, 0.20, 182,
         Option::Call, 100.0, 15.0, -1, 90, false, false, false, 2047, 1},
        {"digital_ps_put_fine90", "pseudo", 100.0, 0.04, 0.00, 0.20, 182,
         Option::Put, 100.0, 15.0, -1, 90, false, false, false, 2047, 1},
        {"digital_ps_call_seed7_coarse4", "pseudo", 100.0, 0.05, 0.04, 0.20,
         365, Option::Call, 110.0, 15.0, 4, -1, false, false, false, 2047, 7},
    };
    for (const auto& c : cfgs)
        runDigital<PseudoRandom>(c);

    const DigitalCfg ldCfgs[] = {
        {"digital_ld_call_coarse4", "lowdiscrepancy", 100.0, 0.05, 0.04, 0.20,
         365, Option::Call, 110.0, 15.0, 4, -1, false, false, false, 2047, 1},
        {"digital_ld_call_fine90_bb", "lowdiscrepancy", 100.0, 0.04, 0.00, 0.20,
         182, Option::Call, 100.0, 15.0, -1, 90, true, false, false, 2047, 1},
    };
    for (const auto& c : ldCfgs)
        runDigital<LowDiscrepancy>(c);

    // Builder + engine validation.
    auto process = bsm(100.0, 0.05, 0.04, 0.20);
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("builder", "MakeMCDigitalEngine");
        in.s("scenario", what);
        Obj ex;
        bool threw = false;
        try {
            f();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    probe("digital_make_no_steps", "neither withSteps nor withStepsPerYear",
          [&] {
              ext::shared_ptr<PricingEngine> e =
                  MakeMCDigitalEngine<PseudoRandom>(process).withSamples(1023);
          });
    probe("digital_make_both_steps", "withSteps and withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCDigitalEngine<PseudoRandom>(process)
                .withSteps(4)
                .withStepsPerYear(12)
                .withSamples(1023);
    });
    probe("digital_make_tolerance_lowdiscrepancy",
          "withAbsoluteTolerance on LowDiscrepancy", [&] {
              MakeMCDigitalEngine<LowDiscrepancy>(process)
                  .withAbsoluteTolerance(0.02);
          });
    probe("digital_engine_plain_payoff", "PlainVanillaPayoff -> wrong payoff",
          [&] {
              VanillaOption opt(
                  ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                  ext::make_shared<AmericanExercise>(kToday, kToday + 365));
              opt.setPricingEngine(MakeMCDigitalEngine<PseudoRandom>(process)
                                       .withSteps(4)
                                       .withSamples(1023)
                                       .withSeed(1));
              opt.NPV();
          });
    probe("digital_engine_european_exercise", "EuropeanExercise -> wrong exercise",
          [&] {
              VanillaOption opt(
                  ext::make_shared<CashOrNothingPayoff>(Option::Call, 110.0,
                                                        15.0),
                  ext::make_shared<EuropeanExercise>(kToday + 365));
              opt.setPricingEngine(MakeMCDigitalEngine<PseudoRandom>(process)
                                       .withSteps(4)
                                       .withSamples(1023)
                                       .withSeed(1));
              opt.NPV();
          });
}

// ===========================================================================
// 4. MCEuropeanHestonEngine / EuropeanHestonPathPricer /
//    MakeMCEuropeanHestonEngine
// ===========================================================================

const char* hestonDiscName(HestonProcess::Discretization d) {
    switch (d) {
        case HestonProcess::PartialTruncation: return "PartialTruncation";
        case HestonProcess::FullTruncation: return "FullTruncation";
        case HestonProcess::Reflection: return "Reflection";
        case HestonProcess::NonCentralChiSquareVariance:
            return "NonCentralChiSquareVariance";
        case HestonProcess::QuadraticExponential: return "QuadraticExponential";
        case HestonProcess::QuadraticExponentialMartingale:
            return "QuadraticExponentialMartingale";
        case HestonProcess::BroadieKayaExactSchemeLobatto:
            return "BroadieKayaExactSchemeLobatto";
        case HestonProcess::BroadieKayaExactSchemeLaguerre:
            return "BroadieKayaExactSchemeLaguerre";
        case HestonProcess::BroadieKayaExactSchemeTrapezoidal:
            return "BroadieKayaExactSchemeTrapezoidal";
    }
    return "?";
}

struct HestonCfg {
    const char* name;
    Real s0, r, q;
    Real v0, kappa, theta, sigma, rho;
    int days;
    Option::Type type;
    Real strike;
    long long steps, stepsPerYear;
    bool antithetic;
    long long samples;
    long long seed;
    HestonProcess::Discretization disc;
};

void hestonCases() {
    const HestonCfg cfgs[] = {
        // Default discretization: QuadraticExponentialMartingale.
        {"heston_ps_call_atm", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04, 0.30, -0.7,
         365, Option::Call, 100.0, 12, -1, false, 1023, 42,
         HestonProcess::QuadraticExponentialMartingale},
        {"heston_ps_put_atm", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04, 0.30, -0.7,
         365, Option::Put, 100.0, 12, -1, false, 1023, 42,
         HestonProcess::QuadraticExponentialMartingale},
        {"heston_ps_call_otm", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04, 0.30, -0.7,
         365, Option::Call, 140.0, 12, -1, false, 1023, 42,
         HestonProcess::QuadraticExponentialMartingale},
        {"heston_ps_call_atm_seed7", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04, 0.30,
         -0.7, 365, Option::Call, 100.0, 12, -1, false, 1023, 7,
         HestonProcess::QuadraticExponentialMartingale},
        {"heston_ps_call_antithetic", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04, 0.30,
         -0.7, 365, Option::Call, 100.0, 12, -1, true, 1023, 42,
         HestonProcess::QuadraticExponentialMartingale},
        {"heston_ps_call_stepsperyear", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04,
         0.30, -0.7, 365, Option::Call, 100.0, -1, 24, false, 1023, 42,
         HestonProcess::QuadraticExponentialMartingale},
        // Feller condition violated (2 kappa theta < sigma^2): the variance
        // path hits zero, so the truncation branch is actually exercised.
        {"heston_ps_call_feller_violated", 100.0, 0.05, 0.0, 0.04, 0.5, 0.04,
         0.90, -0.5, 365, Option::Call, 100.0, 24, -1, false, 1023, 42,
         HestonProcess::QuadraticExponentialMartingale},
        // Non-default discretizations: each changes the answer.
        {"heston_ps_call_full_truncation", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04,
         0.30, -0.7, 365, Option::Call, 100.0, 12, -1, false, 1023, 42,
         HestonProcess::FullTruncation},
        {"heston_ps_call_partial_truncation", 100.0, 0.05, 0.02, 0.04, 1.0,
         0.04, 0.30, -0.7, 365, Option::Call, 100.0, 12, -1, false, 1023, 42,
         HestonProcess::PartialTruncation},
        {"heston_ps_call_reflection", 100.0, 0.05, 0.02, 0.04, 1.0, 0.04, 0.30,
         -0.7, 365, Option::Call, 100.0, 12, -1, false, 1023, 42,
         HestonProcess::Reflection},
        {"heston_ps_call_quadratic_exponential", 100.0, 0.05, 0.02, 0.04, 1.0,
         0.04, 0.30, -0.7, 365, Option::Call, 100.0, 12, -1, false, 1023, 42,
         HestonProcess::QuadraticExponential},
        // rho == 0 removes the sqrt(1-rho^2) mixing entirely.
        {"heston_ps_call_zero_rho", 100.0, 0.05, 0.0, 0.04, 1.0, 0.04, 0.30,
         0.0, 365, Option::Call, 100.0, 12, -1, false, 1023, 42,
         HestonProcess::QuadraticExponentialMartingale},
    };
    for (const auto& c : cfgs) {
        auto heston = ext::make_shared<HestonProcess>(
            flatCurve(c.r), flatCurve(c.q),
            Handle<Quote>(ext::make_shared<SimpleQuote>(c.s0)), c.v0, c.kappa,
            c.theta, c.sigma, c.rho, c.disc);

        MakeMCEuropeanHestonEngine<PseudoRandom> make(heston);
        if (c.steps > 0)
            make.withSteps(Size(c.steps));
        if (c.stepsPerYear > 0)
            make.withStepsPerYear(Size(c.stepsPerYear));
        make.withAntitheticVariate(c.antithetic)
            .withSamples(Size(c.samples))
            .withSeed(BigNatural(c.seed));

        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
            ext::make_shared<EuropeanExercise>(kToday + c.days));
        opt.setPricingEngine(make);

        Obj in;
        in.n("s0", c.s0);
        in.n("r", c.r);
        in.n("q", c.q);
        in.n("v0", c.v0);
        in.n("kappa", c.kappa);
        in.n("theta", c.theta);
        in.n("sigma", c.sigma);
        in.n("rho", c.rho);
        in.i("expiry_days", c.days);
        in.s("day_counter", "Actual365Fixed");
        in.i("eval_year", kToday.year());
        in.i("eval_month", static_cast<int>(kToday.month()));
        in.i("eval_day", kToday.dayOfMonth());
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.s("discretization", hestonDiscName(c.disc));
        describeEngine(in, "pseudo", c.steps, c.stepsPerYear, false,
                       c.antithetic, c.samples, -1.0, -1, c.seed);

        Obj ex;
        recordResults(ex, opt, opt.NPV());
        addCase(c.name, in, ex);
    }

    // Builder validation. NOTE the asymmetry vs MakeMCEuropeanEngine: the
    // over-specification is rejected EARLY, inside withSteps/withStepsPerYear.
    auto heston = ext::make_shared<HestonProcess>(
        flatCurve(0.05), flatCurve(0.02),
        Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)), 0.04, 1.0, 0.04,
        0.30, -0.7);
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("builder", "MakeMCEuropeanHestonEngine");
        in.s("scenario", what);
        Obj ex;
        bool threw = false;
        try {
            f();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    probe("heston_make_no_steps", "neither withSteps nor withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCEuropeanHestonEngine<PseudoRandom>(heston).withSamples(1023);
    });
    probe("heston_make_steps_then_stepsperyear",
          "withStepsPerYear after withSteps (throws inside the setter)", [&] {
              MakeMCEuropeanHestonEngine<PseudoRandom>(heston)
                  .withSteps(4)
                  .withStepsPerYear(12);
          });
    probe("heston_make_stepsperyear_then_steps",
          "withSteps after withStepsPerYear (throws inside the setter)", [&] {
              MakeMCEuropeanHestonEngine<PseudoRandom>(heston)
                  .withStepsPerYear(12)
                  .withSteps(4);
          });
    probe("heston_make_samples_after_tolerance", "withSamples after tolerance",
          [&] {
              MakeMCEuropeanHestonEngine<PseudoRandom>(heston)
                  .withAbsoluteTolerance(0.02)
                  .withSamples(1023);
          });
    probe("heston_make_tolerance_lowdiscrepancy",
          "withAbsoluteTolerance on LowDiscrepancy", [&] {
              MakeMCEuropeanHestonEngine<LowDiscrepancy>(heston)
                  .withAbsoluteTolerance(0.02);
          });
    probe("heston_engine_non_plain_payoff", "CashOrNothingPayoff", [&] {
        VanillaOption opt(
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(MakeMCEuropeanHestonEngine<PseudoRandom>(heston)
                                 .withSteps(4)
                                 .withSamples(1023)
                                 .withSeed(42));
        opt.NPV();
    });
}

// ===========================================================================
// 5. MCEuropeanGJRGARCHEngine / EuropeanGJRGARCHPathPricer /
//    MakeMCEuropeanGJRGARCHEngine
// ===========================================================================

const char* gjrDiscName(GJRGARCHProcess::Discretization d) {
    switch (d) {
        case GJRGARCHProcess::PartialTruncation: return "PartialTruncation";
        case GJRGARCHProcess::FullTruncation: return "FullTruncation";
        case GJRGARCHProcess::Reflection: return "Reflection";
    }
    return "?";
}

struct GjrCfg {
    const char* name;
    Real s0, r, q;
    Real v0, omega, alpha, beta, gamma, lambda, daysPerYear;
    int days;
    Option::Type type;
    Real strike;
    long long steps, stepsPerYear;
    bool antithetic;
    long long samples;
    long long seed;
    GJRGARCHProcess::Discretization disc;
};

void gjrGarchCases() {
    // Parameters from the C++ test-suite (gjrgarchmodel.cpp): daily-scaled
    // omega/alpha/beta/gamma with daysPerYear = 252.
    const GjrCfg cfgs[] = {
        {"gjrgarch_ps_call_atm", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6, 0.024, 0.93,
         0.059, 0.1, 252.0, 365, Option::Call, 100.0, 12, -1, false, 1023, 42,
         GJRGARCHProcess::FullTruncation},
        {"gjrgarch_ps_put_atm", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6, 0.024, 0.93,
         0.059, 0.1, 252.0, 365, Option::Put, 100.0, 12, -1, false, 1023, 42,
         GJRGARCHProcess::FullTruncation},
        {"gjrgarch_ps_call_otm", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6, 0.024, 0.93,
         0.059, 0.1, 252.0, 365, Option::Call, 130.0, 12, -1, false, 1023, 42,
         GJRGARCHProcess::FullTruncation},
        {"gjrgarch_ps_call_atm_seed7", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6, 0.024,
         0.93, 0.059, 0.1, 252.0, 365, Option::Call, 100.0, 12, -1, false, 1023,
         7, GJRGARCHProcess::FullTruncation},
        {"gjrgarch_ps_call_antithetic", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6, 0.024,
         0.93, 0.059, 0.1, 252.0, 365, Option::Call, 100.0, 12, -1, true, 1023,
         42, GJRGARCHProcess::FullTruncation},
        {"gjrgarch_ps_call_stepsperyear", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6,
         0.024, 0.93, 0.059, 0.1, 252.0, 365, Option::Call, 100.0, -1, 24, false,
         1023, 42, GJRGARCHProcess::FullTruncation},
        {"gjrgarch_ps_call_partial_truncation", 100.0, 0.05, 0.0, 2.0e-6,
         2.0e-6, 0.024, 0.93, 0.059, 0.1, 252.0, 365, Option::Call, 100.0, 12,
         -1, false, 1023, 42, GJRGARCHProcess::PartialTruncation},
        {"gjrgarch_ps_call_reflection", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6, 0.024,
         0.93, 0.059, 0.1, 252.0, 365, Option::Call, 100.0, 12, -1, false, 1023,
         42, GJRGARCHProcess::Reflection},
        // lambda == 0 zeroes sigma12/sigma13 asymmetry terms.
        {"gjrgarch_ps_call_zero_lambda", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6,
         0.024, 0.93, 0.059, 0.0, 252.0, 365, Option::Call, 100.0, 12, -1, false,
         1023, 42, GJRGARCHProcess::FullTruncation},
        // gamma == 0 collapses GJR-GARCH to plain GARCH(1,1).
        {"gjrgarch_ps_call_zero_gamma", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6, 0.024,
         0.93, 0.0, 0.1, 252.0, 365, Option::Call, 100.0, 12, -1, false, 1023,
         42, GJRGARCHProcess::FullTruncation},
        // daysPerYear = 365 rescales every drift/diffusion term.
        {"gjrgarch_ps_call_365_days_per_year", 100.0, 0.05, 0.0, 2.0e-6, 2.0e-6,
         0.024, 0.93, 0.059, 0.1, 365.0, 365, Option::Call, 100.0, 12, -1, false,
         1023, 42, GJRGARCHProcess::FullTruncation},
    };
    for (const auto& c : cfgs) {
        auto process = ext::make_shared<GJRGARCHProcess>(
            flatCurve(c.r), flatCurve(c.q),
            Handle<Quote>(ext::make_shared<SimpleQuote>(c.s0)), c.v0, c.omega,
            c.alpha, c.beta, c.gamma, c.lambda, c.daysPerYear, c.disc);

        MakeMCEuropeanGJRGARCHEngine<PseudoRandom> make(process);
        if (c.steps > 0)
            make.withSteps(Size(c.steps));
        if (c.stepsPerYear > 0)
            make.withStepsPerYear(Size(c.stepsPerYear));
        make.withAntitheticVariate(c.antithetic)
            .withSamples(Size(c.samples))
            .withSeed(BigNatural(c.seed));

        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
            ext::make_shared<EuropeanExercise>(kToday + c.days));
        opt.setPricingEngine(make);

        Obj in;
        in.n("s0", c.s0);
        in.n("r", c.r);
        in.n("q", c.q);
        in.n("v0", c.v0);
        in.n("omega", c.omega);
        in.n("alpha", c.alpha);
        in.n("beta", c.beta);
        in.n("gamma", c.gamma);
        in.n("lambda", c.lambda);
        in.n("days_per_year", c.daysPerYear);
        in.i("expiry_days", c.days);
        in.s("day_counter", "Actual365Fixed");
        in.i("eval_year", kToday.year());
        in.i("eval_month", static_cast<int>(kToday.month()));
        in.i("eval_day", kToday.dayOfMonth());
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.s("discretization", gjrDiscName(c.disc));
        describeEngine(in, "pseudo", c.steps, c.stepsPerYear, false,
                       c.antithetic, c.samples, -1.0, -1, c.seed);

        Obj ex;
        recordResults(ex, opt, opt.NPV());
        addCase(c.name, in, ex);
    }

    auto process = ext::make_shared<GJRGARCHProcess>(
        flatCurve(0.05), flatCurve(0.0),
        Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)), 2.0e-6, 2.0e-6,
        0.024, 0.93, 0.059, 0.1);
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("builder", "MakeMCEuropeanGJRGARCHEngine");
        in.s("scenario", what);
        Obj ex;
        bool threw = false;
        try {
            f();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    probe("gjrgarch_make_no_steps", "neither withSteps nor withStepsPerYear",
          [&] {
              ext::shared_ptr<PricingEngine> e =
                  MakeMCEuropeanGJRGARCHEngine<PseudoRandom>(process)
                      .withSamples(1023);
          });
    probe("gjrgarch_make_steps_then_stepsperyear",
          "withStepsPerYear after withSteps (throws inside the setter)", [&] {
              MakeMCEuropeanGJRGARCHEngine<PseudoRandom>(process)
                  .withSteps(4)
                  .withStepsPerYear(12);
          });
    probe("gjrgarch_make_stepsperyear_then_steps",
          "withSteps after withStepsPerYear (throws inside the setter)", [&] {
              MakeMCEuropeanGJRGARCHEngine<PseudoRandom>(process)
                  .withStepsPerYear(12)
                  .withSteps(4);
          });
    probe("gjrgarch_make_tolerance_lowdiscrepancy",
          "withAbsoluteTolerance on LowDiscrepancy", [&] {
              MakeMCEuropeanGJRGARCHEngine<LowDiscrepancy>(process)
                  .withAbsoluteTolerance(0.02);
          });
    probe("gjrgarch_engine_non_plain_payoff", "CashOrNothingPayoff", [&] {
        VanillaOption opt(
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(
            MakeMCEuropeanGJRGARCHEngine<PseudoRandom>(process)
                .withSteps(4)
                .withSamples(1023)
                .withSeed(42));
        opt.NPV();
    });
}

// ===========================================================================
// 6. MCHestonHullWhiteEngine / HestonHullWhitePathPricer /
//    MakeMCHestonHullWhiteEngine  (+ HybridHestonHullWhiteProcess)
// ===========================================================================

struct HhwCfg {
    const char* name;
    Real s0, r, q;
    Real v0, kappa, theta, sigma, rho;   // Heston
    Real hwA, hwSigma;                   // Hull-White
    Real corrEquityShortRate;
    HybridHestonHullWhiteProcess::Discretization disc;
    int days;
    Option::Type type;
    Real strike;
    long long steps;
    bool antithetic, controlVariate;
    long long samples;
    long long seed;
};

void hestonHullWhiteCases() {
    const HhwCfg cfgs[] = {
        // BSMHullWhite is the C++ default discretization.
        {"hhw_bsmhw_put_atm", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5, 0.3,
         0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::BSMHullWhite, 365,
         Option::Put, 100.0, 4, false, false, 1023, 42},
        {"hhw_bsmhw_call_atm", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5,
         0.3, 0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::BSMHullWhite, 365,
         Option::Call, 100.0, 4, false, false, 1023, 42},
        {"hhw_bsmhw_put_otm", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5, 0.3,
         0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::BSMHullWhite, 365,
         Option::Put, 75.0, 4, false, false, 1023, 42},
        {"hhw_bsmhw_put_atm_seed7", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625,
         1e-5, 0.3, 0.01, 0.01, -0.4,
         HybridHestonHullWhiteProcess::BSMHullWhite, 365, Option::Put, 100.0, 4,
         false, false, 1023, 7},
        {"hhw_bsmhw_put_antithetic", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625,
         1e-5, 0.3, 0.01, 0.01, -0.4,
         HybridHestonHullWhiteProcess::BSMHullWhite, 365, Option::Put, 100.0, 4,
         true, false, 1023, 42},
        // corr = 0 removes the equity/short-rate coupling entirely.
        {"hhw_bsmhw_put_zero_corr", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625,
         1e-5, 0.3, 0.01, 0.01, 0.0,
         HybridHestonHullWhiteProcess::BSMHullWhite, 365, Option::Put, 100.0, 4,
         false, false, 1023, 42},
        {"hhw_bsmhw_put_pos_corr", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5,
         0.3, 0.01, 0.01, 0.45, HybridHestonHullWhiteProcess::BSMHullWhite, 365,
         Option::Put, 100.0, 4, false, false, 1023, 42},
        // Euler discretization: a different answer on identical inputs.
        {"hhw_euler_put_atm", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5, 0.3,
         0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::Euler, 365,
         Option::Put, 100.0, 4, false, false, 1023, 42},
        {"hhw_euler_call_atm", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5,
         0.3, 0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::Euler, 365,
         Option::Call, 100.0, 4, false, false, 1023, 42},
        {"hhw_euler_put_antithetic", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625,
         1e-5, 0.3, 0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::Euler, 365,
         Option::Put, 100.0, 4, true, false, 1023, 42},
        // Control variate on: exercises controlPathGenerator (a SECOND
        // hybrid process with corr = 0 on the SAME seed) plus
        // AnalyticHestonHullWhiteEngine as the CV value.
        {"hhw_bsmhw_put_cv", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5, 0.3,
         0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::BSMHullWhite, 365,
         Option::Put, 100.0, 4, false, true, 1023, 42},
        {"hhw_bsmhw_put_cv_antithetic", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625,
         1e-5, 0.3, 0.01, 0.01, -0.4,
         HybridHestonHullWhiteProcess::BSMHullWhite, 365, Option::Put, 100.0, 4,
         true, true, 1023, 42},
        {"hhw_euler_put_cv", 100.0, 0.04, 0.03, 0.0625, 0.5, 0.0625, 1e-5, 0.3,
         0.01, 0.01, -0.4, HybridHestonHullWhiteProcess::Euler, 365,
         Option::Put, 100.0, 4, false, true, 1023, 42},
    };

    for (const auto& c : cfgs) {
        auto rTS = flatCurve(c.r);
        auto qTS = flatCurve(c.q);
        auto heston = ext::make_shared<HestonProcess>(
            rTS, qTS, Handle<Quote>(ext::make_shared<SimpleQuote>(c.s0)), c.v0,
            c.kappa, c.theta, c.sigma, c.rho);
        auto hw = ext::make_shared<HullWhiteForwardProcess>(rTS, c.hwA,
                                                           c.hwSigma);
        const Time maturityTime =
            dayCounter().yearFraction(kToday, kToday + c.days);
        hw->setForwardMeasureTime(maturityTime);

        auto joint = ext::make_shared<HybridHestonHullWhiteProcess>(
            heston, hw, c.corrEquityShortRate, c.disc);

        MakeMCHestonHullWhiteEngine<PseudoRandom> make(joint);
        make.withSteps(Size(c.steps))
            .withAntitheticVariate(c.antithetic)
            .withControlVariate(c.controlVariate)
            .withSamples(Size(c.samples))
            .withSeed(BigNatural(c.seed));

        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
            ext::make_shared<EuropeanExercise>(kToday + c.days));
        opt.setPricingEngine(make);

        Obj in;
        in.n("s0", c.s0);
        in.n("r", c.r);
        in.n("q", c.q);
        in.n("v0", c.v0);
        in.n("kappa", c.kappa);
        in.n("theta", c.theta);
        in.n("heston_sigma", c.sigma);
        in.n("rho", c.rho);
        in.n("hw_a", c.hwA);
        in.n("hw_sigma", c.hwSigma);
        in.n("corr_equity_short_rate", c.corrEquityShortRate);
        in.s("hhw_discretization",
             c.disc == HybridHestonHullWhiteProcess::Euler ? "Euler"
                                                           : "BSMHullWhite");
        in.n("forward_measure_time", maturityTime);
        in.i("expiry_days", c.days);
        in.s("day_counter", "Actual365Fixed");
        in.i("eval_year", kToday.year());
        in.i("eval_month", static_cast<int>(kToday.month()));
        in.i("eval_day", kToday.dayOfMonth());
        in.s("option_type", c.type == Option::Call ? "Call" : "Put");
        in.n("strike", c.strike);
        in.b("control_variate", c.controlVariate);
        describeEngine(in, "pseudo", c.steps, -1, false, c.antithetic,
                       c.samples, -1.0, -1, c.seed);

        Obj ex;
        recordResults(ex, opt, opt.NPV());
        addCase(c.name, in, ex);
    }

    // HybridHestonHullWhiteProcess primitives, pinned directly so a port can
    // localise a failure to the process rather than to the engine.
    {
        auto rTS = flatCurve(0.04);
        auto qTS = flatCurve(0.03);
        auto heston = ext::make_shared<HestonProcess>(
            rTS, qTS, Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)),
            0.0625, 0.5, 0.0625, 1e-5, 0.3);
        auto hw = ext::make_shared<HullWhiteForwardProcess>(rTS, 0.01, 0.01);
        hw->setForwardMeasureTime(1.0);
        auto joint = ext::make_shared<HybridHestonHullWhiteProcess>(
            heston, hw, -0.4, HybridHestonHullWhiteProcess::BSMHullWhite);

        const Array x0 = joint->initialValues();
        const Array dr = joint->drift(0.25, x0);
        const Matrix dif = joint->diffusion(0.25, x0);
        const Array dw = {0.31, -0.72, 1.15};
        const Array ev = joint->evolve(0.25, x0, 0.5, dw);
        const Array evEuler =
            ext::make_shared<HybridHestonHullWhiteProcess>(
                heston, hw, -0.4, HybridHestonHullWhiteProcess::Euler)
                ->evolve(0.25, x0, 0.5, dw);

        Obj in;
        in.n("s0", 100.0);
        in.n("r", 0.04);
        in.n("q", 0.03);
        in.n("v0", 0.0625);
        in.n("kappa", 0.5);
        in.n("theta", 0.0625);
        in.n("heston_sigma", 1e-5);
        in.n("rho", 0.3);
        in.n("hw_a", 0.01);
        in.n("hw_sigma", 0.01);
        in.n("corr_equity_short_rate", -0.4);
        in.n("forward_measure_time", 1.0);
        in.n("t0", 0.25);
        in.n("dt", 0.5);
        in.n("dw0", 0.31);
        in.n("dw1", -0.72);
        in.n("dw2", 1.15);
        in.i("eval_year", kToday.year());
        in.i("eval_month", static_cast<int>(kToday.month()));
        in.i("eval_day", kToday.dayOfMonth());

        Obj ex;
        ex.i("size", static_cast<long long>(joint->size()));
        ex.i("factors", static_cast<long long>(joint->factors()));
        ex.n("x0_0", x0[0]);
        ex.n("x0_1", x0[1]);
        ex.n("x0_2", x0[2]);
        ex.n("drift_0", dr[0]);
        ex.n("drift_1", dr[1]);
        ex.n("drift_2", dr[2]);
        ex.n("diffusion_00", dif[0][0]);
        ex.n("diffusion_10", dif[1][0]);
        ex.n("diffusion_11", dif[1][1]);
        ex.n("diffusion_20", dif[2][0]);
        ex.n("diffusion_21", dif[2][1]);
        ex.n("diffusion_22", dif[2][2]);
        ex.n("evolve_bsmhw_0", ev[0]);
        ex.n("evolve_bsmhw_1", ev[1]);
        ex.n("evolve_bsmhw_2", ev[2]);
        ex.n("evolve_euler_0", evEuler[0]);
        ex.n("evolve_euler_1", evEuler[1]);
        ex.n("evolve_euler_2", evEuler[2]);
        ex.n("eta", joint->eta());
        ex.n("numeraire_at_1", joint->numeraire(1.0, ev));
        ex.n("numeraire_at_0p5", joint->numeraire(0.5, ev));
        addCase("hhw_process_primitives", in, ex);
    }

    // Builder / constructor validation.
    auto rTS = flatCurve(0.04);
    auto qTS = flatCurve(0.03);
    auto heston = ext::make_shared<HestonProcess>(
        rTS, qTS, Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)), 0.0625,
        0.5, 0.0625, 1e-5, 0.3);
    auto hw = ext::make_shared<HullWhiteForwardProcess>(rTS, 0.01, 0.01);
    hw->setForwardMeasureTime(1.0);
    auto joint =
        ext::make_shared<HybridHestonHullWhiteProcess>(heston, hw, -0.4);

    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("builder", "MakeMCHestonHullWhiteEngine");
        in.s("scenario", what);
        Obj ex;
        bool threw = false;
        try {
            f();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    probe("hhw_make_no_steps", "neither withSteps nor withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCHestonHullWhiteEngine<PseudoRandom>(joint).withSamples(1023);
    });
    probe("hhw_make_both_steps", "withSteps and withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCHestonHullWhiteEngine<PseudoRandom>(joint)
                .withSteps(4)
                .withStepsPerYear(12)
                .withSamples(1023);
    });
    probe("hhw_make_tolerance_lowdiscrepancy",
          "withAbsoluteTolerance on LowDiscrepancy", [&] {
              MakeMCHestonHullWhiteEngine<LowDiscrepancy>(joint)
                  .withAbsoluteTolerance(0.02);
          });
    // The process itself: correlation matrix must stay positive definite, and
    // the Hull-White vol must be strictly positive.
    probe("hhw_process_corr_not_positive_definite",
          "corrEquityShortRate^2 + rho^2 > 1", [&] {
              auto h = ext::make_shared<HestonProcess>(
                  rTS, qTS,
                  Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)), 0.0625,
                  0.5, 0.0625, 1e-5, 0.8);
              auto p = ext::make_shared<HybridHestonHullWhiteProcess>(h, hw,
                                                                     0.8);
          });
    probe("hhw_process_zero_hw_sigma", "Hull-White sigma == 0", [&] {
        auto z = ext::make_shared<HullWhiteForwardProcess>(rTS, 0.01, 0.0);
        z->setForwardMeasureTime(1.0);
        auto p = ext::make_shared<HybridHestonHullWhiteProcess>(heston, z,
                                                               -0.4);
    });
    // American exercise -> "only european exercise is supported".
    probe("hhw_engine_american_exercise", "AmericanExercise", [&] {
        VanillaOption opt(
            ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
            ext::make_shared<AmericanExercise>(kToday, kToday + 365));
        opt.setPricingEngine(MakeMCHestonHullWhiteEngine<PseudoRandom>(joint)
                                 .withSteps(4)
                                 .withSamples(1023)
                                 .withSeed(42));
        opt.NPV();
    });
}


// Diagnostics for the Longstaff-Schwartz machinery: the time grid, the
// per-period discount factors, and every regression coefficient. These pin the
// *inside* of MCAmericanEngine so a port can localise a failure to the grid,
// the calibration stream, or the least-squares solve rather than only seeing a
// wrong NPV.
namespace {
class ExposedLsmPricer : public LongstaffSchwartzPathPricer<Path> {
  public:
    ExposedLsmPricer(const TimeGrid& t,
                     const ext::shared_ptr<EarlyExercisePathPricer<Path>>& p,
                     const ext::shared_ptr<YieldTermStructure>& ts)
    : LongstaffSchwartzPathPricer<Path>(t, p, ts) {}
    const Array& coeff(Size i) const { return coeff_[i]; }
    DiscountFactor df(Size i) const { return dF_[i]; }
};
} // namespace

void americanLsmDiagnostics() {
    const Real s0 = 100.0, r = 0.05, q = 0.0, v = 0.20, strike = 100.0;
    const int days = 365;
    const Size steps = 10;
    const Size calibrationSamples = 2048;
    const BigNatural seed = 42;
    const BigNatural seedCalibration = seed + 1768237423UL;

    auto process = bsm(s0, r, q, v);
    std::vector<Time> mandatory = {process->time(kToday + days)};
    TimeGrid grid(mandatory.begin(), mandatory.end(), steps);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
    auto early = ext::make_shared<AmericanPathPricer>(payoff, 2,
                                                      LsmBasisSystem::Monomial);
    auto lsm = ext::make_shared<ExposedLsmPricer>(grid, early,
                                                  *(process->riskFreeRate()));

    PseudoRandom::rsg_type gen = PseudoRandom::make_sequence_generator(
        process->factors() * (grid.size() - 1), seedCalibration);
    auto pg = ext::make_shared<PathGenerator<PseudoRandom::rsg_type>>(
        process, grid, gen, false);

    MonteCarloModel<SingleVariate, PseudoRandom, Statistics> model(
        pg, lsm, Statistics(), false);
    model.addSamples(calibrationSamples);
    lsm->calibrate();

    Obj in;
    describeBsm(in, s0, r, q, v, days);
    in.s("option_type", "Put");
    in.n("strike", strike);
    in.i("steps", static_cast<long long>(steps));
    in.i("calibration_samples", static_cast<long long>(calibrationSamples));
    in.i("seed", static_cast<long long>(seed));
    in.i("seed_calibration", static_cast<long long>(seedCalibration));
    in.i("polynomial_order", 2);
    in.s("polynomial_type", "Monomial");

    Obj ex;
    ex.i("grid_size", static_cast<long long>(grid.size()));
    for (Size i = 0; i < grid.size(); ++i)
        ex.n("grid_t" + std::to_string(i), grid[i]);
    for (Size i = 0; i < grid.size() - 1; ++i)
        ex.n("df" + std::to_string(i), lsm->df(i));
    ex.i("basis_size", static_cast<long long>(early->basisSystem().size()));
    ex.n("scaled_state_at_1", early->state(pg->next().value, 1));
    for (Size i = 0; i + 2 < grid.size(); ++i) {
        const Array& c = lsm->coeff(i);
        for (Size l = 0; l < c.size(); ++l)
            ex.n("coeff" + std::to_string(i) + "_" + std::to_string(l), c[l]);
    }

    // Pricing pass on a *fresh* generator seeded with the pricing seed: the
    // first few per-path prices and the terminal asset values that produced
    // them, so a port can see whether a divergence is in the paths or in the
    // exercise rule.
    PseudoRandom::rsg_type pricingGen = PseudoRandom::make_sequence_generator(
        process->factors() * (grid.size() - 1), seed);
    PathGenerator<PseudoRandom::rsg_type> pricingPg(process, grid, pricingGen,
                                                    false);
    for (Size k = 0; k < 5; ++k) {
        const Path& path = pricingPg.next().value;
        for (Size j = 1; j < path.length(); ++j)
            ex.n("path" + std::to_string(k) + "_" + std::to_string(j), path[j]);
        ex.n("price" + std::to_string(k), (*lsm)(path));
    }
    ex.n("exercise_probability_first5", lsm->exerciseProbability());
    addCase("american_lsm_diagnostics", in, ex);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    europeanCases();
    europeanToleranceCases();
    europeanBuilderCases();
    americanCases();
    americanLsmDiagnostics();
    digitalCases();
    hestonCases();
    gjrGarchCases();
    hestonHullWhiteCases();

    emitDocument();
    return 0;
}
