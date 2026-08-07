// migration-harness/cpp/probes/v143_pe_barriermc/probe.cpp
//
// Reference values for the lattice + Monte-Carlo *single-barrier* pricing-engine
// cluster of C++ QuantLib v1.43:
//
//   * DiscretizedBarrierOption
//   * DiscretizedDermanKaniBarrierOption
//         ql/pricingengines/barrier/discretizedbarrieroption.{hpp,cpp}
//   * BinomialBarrierEngine<T, D>
//         ql/pricingengines/barrier/binomialbarrierengine.hpp
//   * BarrierPathPricer
//   * BiasedBarrierPathPricer
//   * MCBarrierEngine<RNG, S>
//   * MakeMCBarrierEngine<RNG, S>
//         ql/pricingengines/barrier/mcbarrierengine.{hpp,cpp}
//
// Why the values below are EXACT, not a confidence band
// -----------------------------------------------------
// Nothing here consults the clock.  The lattice engines are deterministic by
// construction.  The MC engines are deterministic once the seed is fixed:
//   PseudoRandom   = InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
//                                         InverseCumulativeNormal>
//   LowDiscrepancy = InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>
// and *every* MC case below pins an explicit nonzero seed (seed 0 routes
// through the clock-derived SeedGenerator, so no pinned value uses it).  A
// correct port therefore reproduces NPV and error estimate to ~1e-14 relative.
// A statistical band would pass with the wrong RNG, the wrong path
// construction, the wrong antithetic pairing, or -- the failure this file
// exists to catch -- a missing Brownian-bridge continuity correction, so
// nothing here is banded.
//
// What has to be pinned, and why
// ------------------------------
//  1. `DiscretizedBarrierOption::checkBarrier` (discretizedbarrieroption.cpp:58)
//     is the whole barrier rule, and it is NOT symmetric between the four
//     types:
//       - DownIn/UpIn on a knocked-in node take `max(vanilla[j], payoff(S))`
//         when `stoppingTime`, and the *bare* `vanilla[j]` otherwise; on a
//         not-yet-knocked node they take the rebate ONLY at `endTime`, and are
//         otherwise left untouched (so they carry the rolled-back continuation
//         value).
//       - DownOut/UpOut overwrite a knocked node with the rebate at EVERY
//         slice, and apply `max(values[j], payoff(S))` on the live side when
//         `stoppingTime`.
//     The comparison is `<=` for Down and `>=` for Up: a node sitting exactly
//     on the barrier is knocked.  `checkBarrier` is public, so each case below
//     calls it directly on an all-ones synthetic array over the lattice grid at
//     maturity -- which entries survive as 1.0 and which are overwritten
//     localises a barrier-rule defect without going through a rollback.
//  2. `DiscretizedDermanKaniBarrierOption` differs from the plain one by
//     exactly one step: `adjustBarrier` (discretizedbarrieroption.cpp:156)
//     linearly interpolates the node ADJACENT to the barrier between the
//     unenhanced barrier value and either the vanilla value (knock-in) or the
//     rebate (knock-out), weighted by the barrier's position between the two
//     grid nodes, and floors the result at 0.  It runs BEFORE
//     `unenhanced_.checkBarrier(values_, grid)`, so `checkBarrier` can and does
//     overwrite the interpolated node again on the knocked side; the visible
//     effect is only on the *live* side of the barrier.  Note the index
//     asymmetry, which a port gets wrong by symmetrising it: Down* adjust
//     `optvalues[j+1]` (the node above the barrier) and Up* adjust
//     `optvalues[j]` (the node below), and the Down guard is
//     `grid[j] <= barrier && grid[j+1] > barrier` while the Up guard is
//     `grid[j] < barrier && grid[j+1] >= barrier`.  Both variants are priced on
//     the same setup below so the difference is proven, not assumed.
//     Also note `DiscretizedDermanKaniBarrierOption::postAdjustValuesImpl`
//     rolls `unenhanced_` back unconditionally, whereas the plain
//     `DiscretizedBarrierOption::postAdjustValuesImpl` rolls its `vanilla_`
//     back ONLY for DownIn/UpIn.
//  3. `BinomialBarrierEngine<T,D>::calculate` (binomialbarrierengine.hpp:88).
//     There is no "run the tree twice" and no `additionalResults` in v1.43 --
//     the engine builds ONE tree and reads value/delta/gamma/theta off three
//     rollback stops (grid[2], grid[1], 0), Hull 6th ed. pp.397/398.  Two
//     things a port typically gets wrong:
//       (a) `theta = (p2m - p0) / grid[2]` -- a finite difference off the
//           MIDDLE node of the third-last slice, NOT the Black-Scholes-PDE
//           theta that `BinomialVanillaEngine` uses.  Pinned for every case.
//       (b) the Boyle-Lau step correction.  It is guarded by
//           `std::is_base_of_v<CoxRossRubinstein, T>`, so it fires for
//           CoxRossRubinstein and for nothing else -- NOT for Trigeorgis,
//           which is a sibling under EqualJumpsBinomialTree, not a subclass.
//           The rule is: divisor = log(max(s0,B)/min(s0,B))^2; scan i=1..steps-1
//           for the first `Size(i*i*v*v*maturity/divisor) > timeSteps` (an
//           integer TRUNCATION), take it, then clamp at maxTimeSteps.
//           maxTimeSteps defaults to max(1000, 5*timeSteps); passing
//           maxTimeSteps == timeSteps disables the correction entirely.
//           The `binom_boylelau_*` family drives all four outcomes with
//           s0=100, B=95, v=0.20, T=1: steps=40 promotes to 60, maxSteps=50
//           clamps to 50, maxSteps=40 disables, and Trigeorgis stays at 40.
//     `TimeGrid(maturity, optimum_steps)` and `BlackScholesLattice(tree, r,
//     maturity, optimum_steps)` both use the CORRECTED count, and the tree is
//     built on a re-flattened process (FlatForward at the zero rates read off
//     at the maturity date + BlackConstantVol at blackVol(maturity, s0)).
//  4. `BarrierPathPricer` (mcbarrierengine.cpp:45) carries the
//     Beaglehole-Dybvig-Zhou / El Babsiri-Noel Brownian-bridge continuity
//     correction:
//        x   = log(path[i+1]/path[i])
//        vol = diffProcess->diffusion(timeGrid[i], path[i])      <- time i, NOT i+1
//        y   = 0.5*(x -/+ sqrt(x*x - 2*vol*vol*dt*log(u)))
//        y   = path[i] * exp(y)
//     with the Down branches using `-` and `log(u[i])` and the Up branches
//     using `+` and `log(1-u[i])`.  The uniforms come from a SEPARATE,
//     hard-coded generator built in `MCBarrierEngine::pathPricer`:
//        PseudoRandom::ursg_type(grid.size()-1, PseudoRandom::urng_type(5))
//     -- seed 5, *uniforms* (not Gaussians), ONE fresh sequence per path,
//     independent of the engine seed and of the RNG policy (LowDiscrepancy
//     engines still use this MT19937-seeded-5 stream).  That generator is a
//     member of the path pricer and is therefore STATEFUL across paths: the
//     n-th path consumes the n-th uniform block.  The `pathpricer_*` cases call
//     one pricer instance on three explicit paths in order so the statefulness
//     is pinned too.
//  5. `BiasedBarrierPathPricer` (mcbarrierengine.cpp:174) only inspects the
//     discrete path nodes -- no bridge, no uniforms -- and starts its scan at
//     path[1], so the path's own starting value is never tested.  Unlike the
//     unbiased pricer it does NOT break out of the loop on the first hit, and
//     `asset_price` ends up being the loop's last assignment, i.e. path.back().
//     Both pricers are exercised on identical paths that genuinely knock, so
//     the two answers differ and the difference is the correction.
//  6. Knock-out rebates discount at `discounts_[knockNode]` -- the FIRST
//     crossing -- while knock-in rebates (never knocked in) discount at
//     `discounts_.back()`.  Cases with a nonzero rebate on all four types pin
//     both branches.
//  7. `MCBarrierEngine::calculate` requires `spot > 0` and `!triggered(spot)`
//     (strict `<` / `>` against the barrier), then fills `results_.value`
//     unconditionally and `results_.errorEstimate` only
//     `if constexpr (RNG::allowsErrorEstimate)`.  LowDiscrepancy has
//     allowsErrorEstimate == 0, so `option.errorEstimate()` THROWS; that
//     absence is pinned ("has_error_estimate": false).
//     `MCBarrierEngine` passes `controlVariate = false` to McSimulation and
//     overrides neither `controlPathPricer` nor `controlPricingEngine`: there
//     is NO control variate in v1.43, and no AnalyticBarrierEngine is involved.
//  8. `MakeMCBarrierEngine` validation.  `operator shared_ptr<PricingEngine>()`
//     requires steps XOR stepsPerYear ("number of steps not given" / "number of
//     steps overspecified"); `withSamples` after `withAbsoluteTolerance` throws
//     "tolerance already set"; `withAbsoluteTolerance` after `withSamples`
//     throws "number of samples already set"; `withAbsoluteTolerance` on a
//     LowDiscrepancy builder throws "chosen random generator policy does not
//     allow an error estimate".  `withBias()` defaults to true.  Defaults:
//     brownianBridge_ = antithetic_ = biased_ = false, seed_ = 0.
//
// Emits JSON on stdout; nothing else may be printed.

#include <cmath>
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
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/methods/lattices/bsmlattice.hpp>
#include <ql/methods/montecarlo/path.hpp>
#include <ql/pricingengines/barrier/binomialbarrierengine.hpp>
#include <ql/pricingengines/barrier/discretizedbarrieroption.hpp>
#include <ql/pricingengines/barrier/mcbarrierengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
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

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, "\"" + v + "\""); }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
    // An Array becomes "<k>_n" plus "<k>_0".."<k>_{n-1}" so the Python test can
    // rebuild it without a JSON array parser convention of its own.
    Obj& a(const std::string& k, const Array& v) {
        i(k + "_n", static_cast<long long>(v.size()));
        for (Size j = 0; j < v.size(); ++j)
            n(k + "_" + std::to_string(j), v[j]);
        return *this;
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

const char* barrierName(Barrier::Type t) {
    switch (t) {
        case Barrier::DownIn: return "DownIn";
        case Barrier::UpIn: return "UpIn";
        case Barrier::DownOut: return "DownOut";
        case Barrier::UpOut: return "UpOut";
    }
    return "?";
}

// Describe the market in `inputs` so the Python test reconstructs the case
// rather than restating constants.
void describeBsm(Obj& in, Real s0, Rate r, Rate q, Volatility v, int days) {
    in.n("s0", s0);
    in.n("r", r);
    in.n("q", q);
    in.n("vol", v);
    in.i("expiry_days", days);
    in.s("day_counter", "Actual365Fixed");
    in.s("calendar", "NullCalendar");
    in.i("eval_year", kToday.year());
    in.i("eval_month", static_cast<int>(kToday.month()));
    in.i("eval_day", kToday.dayOfMonth());
}

void describeBarrier(Obj& in, Barrier::Type bt, Real barrier, Real rebate,
                     Option::Type type, Real strike, const std::string& exercise) {
    in.s("barrier_type", barrierName(bt));
    in.n("barrier", barrier);
    in.n("rebate", rebate);
    in.s("payoff", "PlainVanilla");
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.s("exercise", exercise);
}

ext::shared_ptr<Exercise> makeExercise(const std::string& kind, int days) {
    if (kind == "American")
        return ext::make_shared<AmericanExercise>(kToday, kToday + days);
    return ext::make_shared<EuropeanExercise>(kToday + days);
}

BarrierOption::arguments makeArguments(Barrier::Type bt, Real barrier,
                                       Real rebate, Option::Type type,
                                       Real strike, const std::string& exercise,
                                       int days) {
    BarrierOption opt(bt, barrier, rebate,
                      ext::make_shared<PlainVanillaPayoff>(type, strike),
                      makeExercise(exercise, days));
    BarrierOption::arguments args;
    opt.setupArguments(&args);
    args.validate();
    return args;
}

// ===========================================================================
// 1. DiscretizedBarrierOption / DiscretizedDermanKaniBarrierOption internals
// ===========================================================================
//
// The lattice is built exactly the way BinomialBarrierEngine builds it, so the
// numbers below sit on the same grid the engine cases use.

struct DiscCfg {
    const char* name;
    Barrier::Type barrierType;
    Real barrier, rebate;
    Option::Type type;
    Real strike;
    Real s0, r, q, vol;
    int days;
    long long steps;
    const char* exercise; // "European" | "American"
};

struct Setup {
    ext::shared_ptr<GeneralizedBlackScholesProcess> process;
    ext::shared_ptr<BlackScholesLattice<CoxRossRubinstein>> lattice;
    TimeGrid grid;
    Time maturity;
};

// Reproduce binomialbarrierengine.hpp:99-164 (without the Boyle-Lau branch,
// which the dedicated `binom_boylelau_*` cases pin separately).
Setup makeSetup(const DiscCfg& c, const BarrierOption::arguments& args) {
    auto process = bsm(c.s0, c.r, c.q, c.vol);
    DayCounter rfdc = process->riskFreeRate()->dayCounter();
    DayCounter divdc = process->dividendYield()->dayCounter();
    DayCounter voldc = process->blackVolatility()->dayCounter();
    Calendar volcal = process->blackVolatility()->calendar();

    Real s0 = process->stateVariable()->value();
    Date maturityDate = args.exercise->lastDate();
    Volatility v = process->blackVolatility()->blackVol(maturityDate, s0);
    Rate r = process->riskFreeRate()->zeroRate(maturityDate, rfdc, Continuous,
                                               NoFrequency);
    Rate q = process->dividendYield()->zeroRate(maturityDate, divdc, Continuous,
                                                NoFrequency);
    Date referenceDate = process->riskFreeRate()->referenceDate();

    Handle<YieldTermStructure> flatRiskFree(
        ext::make_shared<FlatForward>(referenceDate, r, rfdc));
    Handle<YieldTermStructure> flatDividends(
        ext::make_shared<FlatForward>(referenceDate, q, divdc));
    Handle<BlackVolTermStructure> flatV(
        ext::make_shared<BlackConstantVol>(referenceDate, volcal, v, voldc));

    Time maturity = rfdc.yearFraction(referenceDate, maturityDate);
    auto bs = ext::make_shared<GeneralizedBlackScholesProcess>(
        process->stateVariable(), flatDividends, flatRiskFree, flatV);

    auto payoff = ext::dynamic_pointer_cast<StrikedTypePayoff>(args.payoff);
    Size steps = Size(c.steps);
    auto tree = ext::make_shared<CoxRossRubinstein>(bs, maturity, steps,
                                                    payoff->strike());
    auto lattice = ext::make_shared<BlackScholesLattice<CoxRossRubinstein>>(
        tree, r, maturity, steps);
    return Setup{process, lattice, TimeGrid(maturity, steps), maturity};
}

void discretizedCases() {
    const DiscCfg cfgs[] = {
        // Four barrier types, nonzero rebate, European. Barriers are placed so
        // the barrier does NOT coincide with a lattice node (95 / 108 vs a CRR
        // grid centred on 100), which is exactly what makes the Derman-Kani
        // interpolation observable.
        {"disc_downout_eur", Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0,
         100.0, 0.05, 0.02, 0.20, 365, 8, "European"},
        {"disc_downin_eur", Barrier::DownIn, 95.0, 3.0, Option::Call, 100.0,
         100.0, 0.05, 0.02, 0.20, 365, 8, "European"},
        {"disc_upout_eur", Barrier::UpOut, 108.0, 3.0, Option::Call, 100.0,
         100.0, 0.05, 0.02, 0.20, 365, 8, "European"},
        {"disc_upin_eur", Barrier::UpIn, 108.0, 3.0, Option::Call, 100.0, 100.0,
         0.05, 0.02, 0.20, 365, 8, "European"},
        // Zero rebate: the DownOut knocked side goes to 0.0, not to a rebate.
        {"disc_downout_eur_norebate", Barrier::DownOut, 95.0, 0.0, Option::Call,
         100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, "European"},
        // Put payoff on the Up side.
        {"disc_upout_put_eur", Barrier::UpOut, 108.0, 2.5, Option::Put, 100.0,
         100.0, 0.05, 0.02, 0.20, 365, 8, "European"},
        // American exercise drives the `now <= stoppingTimes_[1] && now >=
        // stoppingTimes_[0]` RANGE branch of checkBarrier, so every slice is a
        // stopping time (as opposed to only the maturity slice).
        {"disc_downout_amer", Barrier::DownOut, 95.0, 3.0, Option::Put, 100.0,
         100.0, 0.05, 0.02, 0.20, 365, 8, "American"},
        {"disc_upin_amer", Barrier::UpIn, 108.0, 3.0, Option::Call, 100.0,
         100.0, 0.05, 0.02, 0.20, 365, 8, "American"},
    };

    for (const auto& c : cfgs) {
        auto args = makeArguments(c.barrierType, c.barrier, c.rebate, c.type,
                                  c.strike, c.exercise, c.days);

        Obj in;
        describeBsm(in, c.s0, c.r, c.q, c.vol, c.days);
        describeBarrier(in, c.barrierType, c.barrier, c.rebate, c.type,
                        c.strike, c.exercise);
        in.i("steps", c.steps);
        in.s("tree", "CoxRossRubinstein");

        Obj ex;

        // --- plain DiscretizedBarrierOption ------------------------------
        {
            Setup su = makeSetup(c, args);
            DiscretizedBarrierOption option(args, *su.process, su.grid);
            option.initialize(su.lattice, su.maturity);

            Array gridArr = su.lattice->grid(su.maturity);
            ex.a("grid", gridArr);
            ex.a("plain_init", option.values());
            ex.a("plain_vanilla_init", option.vanilla());

            // checkBarrier is public: apply it to an all-ones synthetic array
            // so which entries the barrier rule overwrites (and with what) is
            // visible without a rollback in the way.
            Array synthetic(gridArr.size(), 1.0);
            option.checkBarrier(synthetic, gridArr);
            ex.a("plain_check_ones", synthetic);

            Size mid = Size(c.steps) / 2;
            option.rollback(su.grid[mid]);
            ex.n("plain_mid_t", su.grid[mid]);
            ex.a("plain_mid", option.values());
            ex.a("plain_mid_vanilla", option.vanilla());

            option.rollback(su.grid[2]);
            ex.a("plain_s2", option.values());
            option.rollback(su.grid[1]);
            ex.a("plain_s1", option.values());
            option.rollback(0.0);
            ex.n("plain_pv", option.presentValue());
        }

        // --- DiscretizedDermanKaniBarrierOption --------------------------
        {
            Setup su = makeSetup(c, args);
            DiscretizedDermanKaniBarrierOption dk(args, *su.process, su.grid);
            dk.initialize(su.lattice, su.maturity);
            ex.a("dk_init", dk.values());

            Size mid = Size(c.steps) / 2;
            dk.rollback(su.grid[mid]);
            ex.a("dk_mid", dk.values());
            dk.rollback(su.grid[2]);
            ex.a("dk_s2", dk.values());
            dk.rollback(su.grid[1]);
            ex.a("dk_s1", dk.values());
            dk.rollback(0.0);
            ex.n("dk_pv", dk.presentValue());
        }

        addCase(c.name, in, ex);
    }

    // DiscretizedBarrierOption requires at least one stopping date. C++ cannot
    // build an exercise with no dates, so this guard is reached by handing the
    // ctor an arguments object whose exercise carries an empty date list --
    // which BarrierOption cannot produce.  Pinned as "unreachable through the
    // public instrument API" rather than faked.
    {
        Obj in;
        in.s("scenario", "DiscretizedBarrierOption with empty exercise dates");
        Obj ex;
        ex.b("reachable_through_public_api", false);
        addCase("disc_empty_exercise_unreachable", in, ex);
    }
}

// ===========================================================================
// 2. BinomialBarrierEngine<T, D>
// ===========================================================================

struct BinCfg {
    const char* name;
    const char* tree;
    bool dermanKani;
    Barrier::Type barrierType;
    Real barrier, rebate;
    Option::Type type;
    Real strike;
    Real s0, r, q, vol;
    int days;
    long long steps, maxSteps;
    const char* exercise;
};

template <class T>
ext::shared_ptr<PricingEngine>
makeBinomialEngine(const ext::shared_ptr<GeneralizedBlackScholesProcess>& p,
                   Size steps, Size maxSteps, bool dk) {
    if (dk)
        return ext::make_shared<
            BinomialBarrierEngine<T, DiscretizedDermanKaniBarrierOption>>(
            p, steps, maxSteps);
    return ext::make_shared<BinomialBarrierEngine<T, DiscretizedBarrierOption>>(
        p, steps, maxSteps);
}

ext::shared_ptr<PricingEngine>
dispatchBinomialEngine(const std::string& tree,
                       const ext::shared_ptr<GeneralizedBlackScholesProcess>& p,
                       Size steps, Size maxSteps, bool dk) {
    if (tree == "JarrowRudd")
        return makeBinomialEngine<JarrowRudd>(p, steps, maxSteps, dk);
    if (tree == "CoxRossRubinstein")
        return makeBinomialEngine<CoxRossRubinstein>(p, steps, maxSteps, dk);
    if (tree == "AdditiveEQPBinomialTree")
        return makeBinomialEngine<AdditiveEQPBinomialTree>(p, steps, maxSteps, dk);
    if (tree == "Trigeorgis")
        return makeBinomialEngine<Trigeorgis>(p, steps, maxSteps, dk);
    if (tree == "Tian")
        return makeBinomialEngine<Tian>(p, steps, maxSteps, dk);
    if (tree == "LeisenReimer")
        return makeBinomialEngine<LeisenReimer>(p, steps, maxSteps, dk);
    if (tree == "Joshi4")
        return makeBinomialEngine<Joshi4>(p, steps, maxSteps, dk);
    QL_FAIL("unknown tree " << tree);
}

void runBinomial(const BinCfg& c) {
    auto process = bsm(c.s0, c.r, c.q, c.vol);
    BarrierOption opt(c.barrierType, c.barrier, c.rebate,
                      ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                      makeExercise(c.exercise, c.days));
    opt.setPricingEngine(dispatchBinomialEngine(c.tree, process, Size(c.steps),
                                                Size(c.maxSteps),
                                                c.dermanKani));

    Obj in;
    describeBsm(in, c.s0, c.r, c.q, c.vol, c.days);
    describeBarrier(in, c.barrierType, c.barrier, c.rebate, c.type, c.strike,
                    c.exercise);
    in.s("tree", c.tree);
    in.s("discretization",
         c.dermanKani ? "DiscretizedDermanKaniBarrierOption"
                      : "DiscretizedBarrierOption");
    in.i("steps", c.steps);
    in.i("max_steps", c.maxSteps);

    Obj ex;
    ex.n("npv", opt.NPV());
    ex.n("delta", opt.delta());
    ex.n("gamma", opt.gamma());
    ex.n("theta", opt.theta());
    addCase(c.name, in, ex);
}

void binomialCases() {
    const char* trees[] = {"JarrowRudd",  "CoxRossRubinstein",
                           "AdditiveEQPBinomialTree", "Trigeorgis",
                           "Tian",        "LeisenReimer",
                           "Joshi4"};

    // Sweep every tree builder x both discretizations on one DownOut call.
    // maxSteps == steps disables Boyle-Lau so this sweep isolates the tree.
    for (const char* t : trees) {
        for (int dk = 0; dk < 2; ++dk) {
            std::string name = std::string("binom_sweep_") + t +
                               (dk != 0 ? "_dk" : "_plain");
            BinCfg c{name.c_str(), t,   dk != 0, Barrier::DownOut,
                     95.0,         3.0, Option::Call, 100.0,
                     100.0,        0.05, 0.02,   0.20,
                     365,          41,   41,     "European"};
            runBinomial(c);
        }
    }

    // Four barrier types x both discretizations, CRR, Boyle-Lau disabled.
    const Barrier::Type types[] = {Barrier::DownIn, Barrier::UpIn,
                                   Barrier::DownOut, Barrier::UpOut};
    for (Barrier::Type bt : types) {
        Real barrier = (bt == Barrier::DownIn || bt == Barrier::DownOut) ? 95.0
                                                                        : 108.0;
        for (int dk = 0; dk < 2; ++dk) {
            std::string name = std::string("binom_type_") + barrierName(bt) +
                               (dk != 0 ? "_dk" : "_plain");
            BinCfg c{name.c_str(), "CoxRossRubinstein", dk != 0, bt,
                     barrier,      3.0,   Option::Call,  100.0,
                     100.0,        0.05,  0.02,          0.20,
                     365,          41,    41,            "European"};
            runBinomial(c);
        }
        // Zero rebate on the same geometry: proves the rebate is not folded in
        // somewhere it should not be.
        for (int dk = 0; dk < 2; ++dk) {
            std::string name = std::string("binom_type_") + barrierName(bt) +
                               "_norebate" + (dk != 0 ? "_dk" : "_plain");
            BinCfg c{name.c_str(), "CoxRossRubinstein", dk != 0, bt,
                     barrier,      0.0,   Option::Call,  100.0,
                     100.0,        0.05,  0.02,          0.20,
                     365,          41,    41,            "European"};
            runBinomial(c);
        }
    }

    // American exercise, both discretizations, put and call.
    {
        const BinCfg cfgs[] = {
            {"binom_amer_downout_put_plain", "CoxRossRubinstein", false,
             Barrier::DownOut, 95.0, 3.0, Option::Put, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 41, 41, "American"},
            {"binom_amer_downout_put_dk", "CoxRossRubinstein", true,
             Barrier::DownOut, 95.0, 3.0, Option::Put, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 41, 41, "American"},
            {"binom_amer_upin_call_plain", "CoxRossRubinstein", false,
             Barrier::UpIn, 108.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 41, 41, "American"},
            {"binom_amer_upin_call_dk", "CoxRossRubinstein", true,
             Barrier::UpIn, 108.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 41, 41, "American"},
        };
        for (const auto& c : cfgs)
            runBinomial(c);
    }

    // Boyle-Lau. s0=100, B=95, v=0.20, T=1 =>
    //   divisor = log(100/95)^2 = 0.0026310...
    //   Size(i*i*0.04/divisor): i=1 -> 15, i=2 -> 60, i=3 -> 136
    // steps=40 therefore promotes to 60 (first optimum > 40).
    {
        const BinCfg cfgs[] = {
            // maxSteps = 0 -> default max(1000, 5*40) = 1000 -> 60 stands.
            {"binom_boylelau_crr_default", "CoxRossRubinstein", false,
             Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 40, 0, "European"},
            // maxSteps = 50 clamps the promoted 60 down to 50.
            {"binom_boylelau_crr_clamped", "CoxRossRubinstein", false,
             Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 40, 50, "European"},
            // maxSteps == steps disables the correction: exactly 40 steps.
            {"binom_boylelau_crr_disabled", "CoxRossRubinstein", false,
             Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 40, 40, "European"},
            // Trigeorgis is a SIBLING of CoxRossRubinstein, not a subclass, so
            // is_base_of_v is false and Boyle-Lau never fires: 40 steps even
            // with the default maxSteps.  Its value must therefore equal the
            // Trigeorgis 40-step run below, and differ from a 60-step one.
            {"binom_boylelau_trigeorgis_default", "Trigeorgis", false,
             Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 40, 0, "European"},
            {"binom_boylelau_trigeorgis_disabled", "Trigeorgis", false,
             Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 40, 40, "European"},
            // The explicit twin of the promoted case: 60 steps, correction off.
            {"binom_boylelau_crr_explicit60", "CoxRossRubinstein", false,
             Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 60, 60, "European"},
            // Same geometry, Derman-Kani, promotion on.
            {"binom_boylelau_crr_default_dk", "CoxRossRubinstein", true,
             Barrier::DownOut, 95.0, 3.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
             0.20, 365, 40, 0, "European"},
            // s0 == barrier would make divisor == 0; close(divisor,0) then
            // leaves optimum_steps at timeSteps.  s0 sits just above so the
            // instrument is not triggered but the divisor is tiny, driving the
            // scan straight past maxTimeSteps -> clamped to the default 1000.
            // NOTE: 1000 steps, so this is the one slow case.
            {"binom_boylelau_tiny_divisor", "CoxRossRubinstein", false,
             Barrier::DownOut, 99.99, 0.0, Option::Call, 100.0, 100.0, 0.05,
             0.02, 0.20, 365, 40, 0, "European"},
        };
        for (const auto& c : cfgs)
            runBinomial(c);
    }

    // Engine + argument validation.
    auto process = bsm(100.0, 0.05, 0.02, 0.20);
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("engine", "BinomialBarrierEngine");
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

    probe("binom_engine_zero_steps", "timeSteps == 0", [&] {
        auto e = ext::make_shared<
            BinomialBarrierEngine<CoxRossRubinstein, DiscretizedBarrierOption>>(
            process, 0, 0);
    });
    probe("binom_engine_maxsteps_below_steps", "maxTimeSteps < timeSteps", [&] {
        auto e = ext::make_shared<
            BinomialBarrierEngine<CoxRossRubinstein, DiscretizedBarrierOption>>(
            process, 40, 39);
    });
    probe("binom_engine_maxsteps_equal_ok", "maxTimeSteps == timeSteps (must not throw)",
          [&] {
              auto e = ext::make_shared<BinomialBarrierEngine<
                  CoxRossRubinstein, DiscretizedBarrierOption>>(process, 40, 40);
          });
    probe("binom_engine_barrier_touched", "spot already past the barrier", [&] {
        BarrierOption opt(Barrier::DownOut, 105.0, 0.0,
                          ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                          ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(ext::make_shared<BinomialBarrierEngine<
                                 CoxRossRubinstein, DiscretizedBarrierOption>>(
            process, 41, 41));
        opt.NPV();
    });
    probe("binom_engine_zero_strike", "strike == 0", [&] {
        BarrierOption opt(Barrier::DownOut, 95.0, 0.0,
                          ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0),
                          ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(ext::make_shared<BinomialBarrierEngine<
                                 CoxRossRubinstein, DiscretizedBarrierOption>>(
            process, 41, 41));
        opt.NPV();
    });
}

// ===========================================================================
// 3. BarrierPathPricer / BiasedBarrierPathPricer
// ===========================================================================
//
// Explicit paths, so the pricers are pinned independently of any path
// generator.  The unbiased pricer's uniform generator is stateful, so the
// three paths are priced through ONE pricer instance in order.

struct PathPricerCfg {
    const char* name;
    Barrier::Type barrierType;
    Real barrier, rebate;
    Option::Type type;
    Real strike;
    // Four 4-step paths (5 values each), chosen so that:
    //   p0 never crosses,
    //   p1 crosses at node 2,
    //   p2 crosses at node 4 (last),
    //   p3 is a NEAR MISS -- every node stays on the live side of the barrier,
    //     but only just, so the Brownian bridge is the only thing that can
    //     trigger it.  That is the case the biased pricer structurally cannot
    //     see, and it ends in the money so the two pricers cannot agree by
    //     accident on a zero payoff.
    Real p[4][5];
};

void pathPricerCases() {
    const Real r = 0.05, q = 0.02, vol = 0.20;
    const Time maturity = 1.0;
    const Size steps = 4;
    auto process = bsm(100.0, r, q, vol);

    TimeGrid grid(maturity, steps);
    std::vector<DiscountFactor> discounts(grid.size());
    for (Size i = 0; i < grid.size(); ++i)
        discounts[i] = process->riskFreeRate()->discount(grid[i]);

    const PathPricerCfg cfgs[] = {
        {"pathpricer_downout", Barrier::DownOut, 90.0, 4.0, Option::Call, 100.0,
         {{100.0, 104.0, 109.0, 112.0, 118.0},
          {100.0, 97.0, 88.0, 93.0, 101.0},
          {100.0, 103.0, 99.0, 95.0, 89.0},
          {100.0, 92.0, 91.0, 96.0, 105.0}}},
        {"pathpricer_downin", Barrier::DownIn, 90.0, 4.0, Option::Call, 100.0,
         {{100.0, 104.0, 109.0, 112.0, 118.0},
          {100.0, 97.0, 88.0, 93.0, 101.0},
          {100.0, 103.0, 99.0, 95.0, 89.0},
          {100.0, 92.0, 91.0, 96.0, 105.0}}},
        {"pathpricer_upout", Barrier::UpOut, 110.0, 4.0, Option::Call, 100.0,
         {{100.0, 97.0, 94.0, 99.0, 103.0},
          {100.0, 104.0, 112.0, 108.0, 106.0},
          {100.0, 103.0, 106.0, 108.0, 113.0},
          {100.0, 108.0, 109.0, 106.0, 108.0}}},
        {"pathpricer_upin", Barrier::UpIn, 110.0, 4.0, Option::Call, 100.0,
         {{100.0, 97.0, 94.0, 99.0, 103.0},
          {100.0, 104.0, 112.0, 108.0, 106.0},
          {100.0, 103.0, 106.0, 108.0, 113.0},
          {100.0, 108.0, 109.0, 106.0, 108.0}}},
        // Zero rebate: separates "the rebate branch discounts at knockNode"
        // from "the payoff branch discounts at the back".
        {"pathpricer_downout_norebate", Barrier::DownOut, 90.0, 0.0,
         Option::Call, 100.0,
         {{100.0, 104.0, 109.0, 112.0, 118.0},
          {100.0, 97.0, 88.0, 93.0, 101.0},
          {100.0, 103.0, 99.0, 95.0, 89.0},
          {100.0, 92.0, 91.0, 96.0, 105.0}}},
        // Put payoff.
        {"pathpricer_upout_put", Barrier::UpOut, 110.0, 4.0, Option::Put, 100.0,
         {{100.0, 97.0, 94.0, 99.0, 103.0},
          {100.0, 104.0, 112.0, 108.0, 106.0},
          {100.0, 103.0, 106.0, 108.0, 113.0},
          {100.0, 108.0, 109.0, 106.0, 95.0}}},
    };

    const int kNPaths = 4;
    for (const auto& c : cfgs) {
        std::vector<Path> paths;
        paths.reserve(kNPaths);
        for (int k = 0; k < kNPaths; ++k)
            paths.emplace_back(grid, Array(c.p[k], c.p[k] + 5));

        Obj in;
        describeBsm(in, 100.0, r, q, vol, 365);
        describeBarrier(in, c.barrierType, c.barrier, c.rebate, c.type,
                        c.strike, "European");
        in.i("steps", static_cast<long long>(steps));
        in.i("n_paths", kNPaths);
        in.i("bridge_uniform_seed", 5);
        for (int k = 0; k < kNPaths; ++k)
            for (Size j = 0; j < 5; ++j)
                in.n("path" + std::to_string(k) + "_" + std::to_string(j),
                     c.p[k][j]);
        for (Size k = 0; k < grid.size(); ++k) {
            in.n("grid_t" + std::to_string(k), grid[k]);
            in.n("discount_" + std::to_string(k), discounts[k]);
        }

        Obj ex;

        // --- unbiased: ONE instance, four calls in order -----------------
        {
            PseudoRandom::ursg_type seqGen(grid.size() - 1,
                                           PseudoRandom::urng_type(5));
            BarrierPathPricer pricer(c.barrierType, c.barrier, c.rebate, c.type,
                                     c.strike, discounts, process, seqGen);
            for (int k = 0; k < kNPaths; ++k)
                ex.n("unbiased_p" + std::to_string(k), pricer(paths[k]));
        }
        // --- unbiased: a FRESH instance per path -------------------------
        // Different from the sequence above wherever the uniform block
        // matters, which is exactly the point: the generator is stateful.
        {
            for (int k = 0; k < kNPaths; ++k) {
                PseudoRandom::ursg_type seqGen(grid.size() - 1,
                                               PseudoRandom::urng_type(5));
                BarrierPathPricer pricer(c.barrierType, c.barrier, c.rebate,
                                         c.type, c.strike, discounts, process,
                                         seqGen);
                ex.n("unbiased_fresh_p" + std::to_string(k), pricer(paths[k]));
            }
        }
        // --- biased ------------------------------------------------------
        {
            BiasedBarrierPathPricer pricer(c.barrierType, c.barrier, c.rebate,
                                           c.type, c.strike, discounts);
            for (int k = 0; k < kNPaths; ++k)
                ex.n("biased_p" + std::to_string(k), pricer(paths[k]));
        }
        addCase(c.name, in, ex);
    }

    // Constructor guards.
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("class", "BarrierPathPricer / BiasedBarrierPathPricer");
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
    probe("pathpricer_negative_strike", "strike < 0", [&] {
        PseudoRandom::ursg_type g(steps, PseudoRandom::urng_type(5));
        BarrierPathPricer p(Barrier::DownOut, 90.0, 0.0, Option::Call, -1.0,
                            discounts, process, g);
    });
    probe("pathpricer_zero_barrier", "barrier <= 0", [&] {
        PseudoRandom::ursg_type g(steps, PseudoRandom::urng_type(5));
        BarrierPathPricer p(Barrier::DownOut, 0.0, 0.0, Option::Call, 100.0,
                            discounts, process, g);
    });
    probe("biased_pathpricer_negative_strike", "biased, strike < 0", [&] {
        BiasedBarrierPathPricer p(Barrier::DownOut, 90.0, 0.0, Option::Call,
                                  -1.0, discounts);
    });
    probe("biased_pathpricer_zero_barrier", "biased, barrier <= 0", [&] {
        BiasedBarrierPathPricer p(Barrier::DownOut, 0.0, 0.0, Option::Call,
                                  100.0, discounts);
    });
    probe("pathpricer_single_point_path", "path of length 1", [&] {
        TimeGrid one{0.0};
        Array vals(1, 100.0);
        Path p(one, vals);
        BiasedBarrierPathPricer pricer(Barrier::DownOut, 90.0, 0.0,
                                       Option::Call, 100.0, discounts);
        pricer(p);
    });
}

// ===========================================================================
// 4. MCBarrierEngine / MakeMCBarrierEngine
// ===========================================================================

struct McCfg {
    const char* name;
    const char* rng; // "pseudo" | "lowdiscrepancy"
    Barrier::Type barrierType;
    Real barrier, rebate;
    Option::Type type;
    Real strike;
    Real s0, r, q, vol;
    int days;
    long long steps, stepsPerYear;
    bool brownianBridge, antithetic, biased;
    long long samples;
    Real tolerance;   // -1 == Null<Real>()
    long long maxSamples; // -1 == Null<Size>()
    long long seed;
};

template <class RNG>
void runMc(const McCfg& c) {
    auto process = bsm(c.s0, c.r, c.q, c.vol);
    MakeMCBarrierEngine<RNG> make(process);
    if (c.steps > 0)
        make.withSteps(Size(c.steps));
    if (c.stepsPerYear > 0)
        make.withStepsPerYear(Size(c.stepsPerYear));
    make.withBrownianBridge(c.brownianBridge)
        .withAntitheticVariate(c.antithetic)
        .withBias(c.biased)
        .withSeed(BigNatural(c.seed));
    if (c.samples > 0)
        make.withSamples(Size(c.samples));
    else
        make.withAbsoluteTolerance(c.tolerance);
    if (c.maxSamples > 0)
        make.withMaxSamples(Size(c.maxSamples));

    BarrierOption opt(c.barrierType, c.barrier, c.rebate,
                      ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
                      ext::make_shared<EuropeanExercise>(kToday + c.days));
    opt.setPricingEngine(make);

    Obj in;
    describeBsm(in, c.s0, c.r, c.q, c.vol, c.days);
    describeBarrier(in, c.barrierType, c.barrier, c.rebate, c.type, c.strike,
                    "European");
    in.s("rng", c.rng);
    in.i("steps", c.steps);
    in.i("steps_per_year", c.stepsPerYear);
    in.b("brownian_bridge", c.brownianBridge);
    in.b("antithetic", c.antithetic);
    in.b("biased", c.biased);
    in.i("samples", c.samples);
    in.n("tolerance", c.tolerance);
    in.i("max_samples", c.maxSamples);
    in.i("seed", c.seed);
    in.i("bridge_uniform_seed", 5);

    Obj ex;
    ex.n("npv", opt.NPV());
    try {
        ex.n("error_estimate", opt.errorEstimate());
        ex.b("has_error_estimate", true);
    } catch (const std::exception&) {
        ex.b("has_error_estimate", false);
    }
    addCase(c.name, in, ex);
}

void mcCases() {
    // 8 steps/year on a 1y option: coarse enough that the Brownian-bridge
    // continuity correction dominates rather than merely refines, so a port
    // that drops it misses by a wide margin instead of a rounding.
    const McCfg cfgs[] = {
        // --- four barrier types, unbiased, nonzero rebate ----------------
        {"mc_ps_downout_unbiased", "pseudo", Barrier::DownOut, 90.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         false, 4095, -1.0, -1, 42},
        {"mc_ps_downin_unbiased", "pseudo", Barrier::DownIn, 90.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         false, 4095, -1.0, -1, 42},
        {"mc_ps_upout_unbiased", "pseudo", Barrier::UpOut, 110.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         false, 4095, -1.0, -1, 42},
        {"mc_ps_upin_unbiased", "pseudo", Barrier::UpIn, 110.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         false, 4095, -1.0, -1, 42},
        // --- the same four, biased --------------------------------------
        {"mc_ps_downout_biased", "pseudo", Barrier::DownOut, 90.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         true, 4095, -1.0, -1, 42},
        {"mc_ps_downin_biased", "pseudo", Barrier::DownIn, 90.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         true, 4095, -1.0, -1, 42},
        {"mc_ps_upout_biased", "pseudo", Barrier::UpOut, 110.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         true, 4095, -1.0, -1, 42},
        {"mc_ps_upin_biased", "pseudo", Barrier::UpIn, 110.0, 4.0, Option::Call,
         100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false, true, 4095,
         -1.0, -1, 42},
        // --- zero rebate -------------------------------------------------
        {"mc_ps_downout_unbiased_norebate", "pseudo", Barrier::DownOut, 90.0,
         0.0, Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false,
         false, false, 4095, -1.0, -1, 42},
        {"mc_ps_upin_biased_norebate", "pseudo", Barrier::UpIn, 110.0, 0.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         true, 4095, -1.0, -1, 42},
        // --- put payoff --------------------------------------------------
        {"mc_ps_downout_put_unbiased", "pseudo", Barrier::DownOut, 90.0, 4.0,
         Option::Put, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         false, 4095, -1.0, -1, 42},
        // --- a different seed on an otherwise identical case -------------
        {"mc_ps_downout_unbiased_seed7", "pseudo", Barrier::DownOut, 90.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         false, 4095, -1.0, -1, 7},
        // --- antithetic ---------------------------------------------------
        {"mc_ps_downout_unbiased_antithetic", "pseudo", Barrier::DownOut, 90.0,
         4.0, Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false,
         true, false, 4095, -1.0, -1, 42},
        {"mc_ps_upout_biased_antithetic", "pseudo", Barrier::UpOut, 110.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, true,
         true, 4095, -1.0, -1, 42},
        // --- brownian bridge ----------------------------------------------
        {"mc_ps_downout_unbiased_bb", "pseudo", Barrier::DownOut, 90.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, true, false,
         false, 4095, -1.0, -1, 42},
        {"mc_ps_downout_biased_bb_antithetic", "pseudo", Barrier::DownOut, 90.0,
         4.0, Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, true,
         true, true, 4095, -1.0, -1, 42},
        // --- stepsPerYear instead of steps --------------------------------
        {"mc_ps_downout_unbiased_stepsperyear", "pseudo", Barrier::DownOut,
         90.0, 4.0, Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, -1, 12,
         false, false, false, 2047, -1.0, -1, 42},
        // Size(4 * 0.0849315) == 0 -> max(0,1) == 1 step: the guard is driven
        // to exactly its clamp value.
        {"mc_ps_downout_stepsperyear_truncates_to_one", "pseudo",
         Barrier::DownOut, 90.0, 4.0, Option::Call, 100.0, 100.0, 0.05, 0.02,
         0.20, 31, -1, 4, false, false, false, 2047, -1.0, -1, 42},
        // --- finer grid: the bridge correction becomes a small refinement --
        {"mc_ps_downout_unbiased_52steps", "pseudo", Barrier::DownOut, 90.0,
         4.0, Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 52, -1, false,
         false, false, 2047, -1.0, -1, 42},
        // --- tolerance-driven termination ---------------------------------
        {"mc_ps_downout_tol_0p20", "pseudo", Barrier::DownOut, 90.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         false, -1, 0.20, -1, 42},
        {"mc_ps_downout_tol_0p20_maxsamples", "pseudo", Barrier::DownOut, 90.0,
         4.0, Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false,
         false, false, -1, 0.20, 1000000, 42},
    };
    for (const auto& c : cfgs)
        runMc<PseudoRandom>(c);

    // LowDiscrepancy: no error estimate at all.  The path pricer's own uniform
    // stream is still MT19937 seeded 5.
    const McCfg ldCfgs[] = {
        {"mc_ld_downout_unbiased", "lowdiscrepancy", Barrier::DownOut, 90.0,
         4.0, Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false,
         false, false, 4095, -1.0, -1, 42},
        {"mc_ld_upin_biased", "lowdiscrepancy", Barrier::UpIn, 110.0, 4.0,
         Option::Call, 100.0, 100.0, 0.05, 0.02, 0.20, 365, 8, -1, false, false,
         true, 4095, -1.0, -1, 42},
    };
    for (const auto& c : ldCfgs)
        runMc<LowDiscrepancy>(c);

    // Builder + engine validation.
    auto process = bsm(100.0, 0.05, 0.02, 0.20);
    auto probe = [&](const std::string& name, const std::string& what,
                     const std::function<void()>& f) {
        Obj in;
        in.s("builder", "MakeMCBarrierEngine");
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

    probe("mc_make_no_steps", "neither withSteps nor withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCBarrierEngine<PseudoRandom>(process).withSamples(1023);
    });
    probe("mc_make_both_steps", "withSteps and withStepsPerYear", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCBarrierEngine<PseudoRandom>(process)
                .withSteps(4)
                .withStepsPerYear(12)
                .withSamples(1023);
    });
    probe("mc_make_samples_after_tolerance", "withSamples after tolerance", [&] {
        MakeMCBarrierEngine<PseudoRandom>(process)
            .withAbsoluteTolerance(0.02)
            .withSamples(1023);
    });
    probe("mc_make_tolerance_after_samples", "tolerance after withSamples", [&] {
        MakeMCBarrierEngine<PseudoRandom>(process)
            .withSamples(1023)
            .withAbsoluteTolerance(0.02);
    });
    probe("mc_make_tolerance_lowdiscrepancy",
          "withAbsoluteTolerance on LowDiscrepancy", [&] {
              MakeMCBarrierEngine<LowDiscrepancy>(process)
                  .withAbsoluteTolerance(0.02);
          });
    probe("mc_make_steps_only_ok", "withSteps only (must not throw)", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCBarrierEngine<PseudoRandom>(process).withSteps(4).withSamples(
                1023);
    });
    probe("mc_engine_zero_steps", "timeSteps == 0", [&] {
        auto e = ext::make_shared<MCBarrierEngine<PseudoRandom>>(
            process, 0, Null<Size>(), false, false, 1023, Null<Real>(),
            Null<Size>(), false, 42);
    });
    probe("mc_engine_zero_steps_per_year", "timeStepsPerYear == 0", [&] {
        auto e = ext::make_shared<MCBarrierEngine<PseudoRandom>>(
            process, Null<Size>(), 0, false, false, 1023, Null<Real>(),
            Null<Size>(), false, 42);
    });
    probe("mc_engine_no_steps_at_all", "neither timeSteps nor timeStepsPerYear",
          [&] {
              auto e = ext::make_shared<MCBarrierEngine<PseudoRandom>>(
                  process, Null<Size>(), Null<Size>(), false, false, 1023,
                  Null<Real>(), Null<Size>(), false, 42);
          });
    probe("mc_engine_both_steps", "both timeSteps and timeStepsPerYear", [&] {
        auto e = ext::make_shared<MCBarrierEngine<PseudoRandom>>(
            process, 4, 12, false, false, 1023, Null<Real>(), Null<Size>(),
            false, 42);
    });
    probe("mc_engine_no_samples_no_tolerance",
          "requiredSamples and requiredTolerance both Null", [&] {
              BarrierOption opt(
                  Barrier::DownOut, 90.0, 0.0,
                  ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                  ext::make_shared<EuropeanExercise>(kToday + 365));
              opt.setPricingEngine(ext::make_shared<MCBarrierEngine<PseudoRandom>>(
                  process, 8, Null<Size>(), false, false, Null<Size>(),
                  Null<Real>(), Null<Size>(), false, 42));
              opt.NPV();
          });
    probe("mc_engine_barrier_touched", "spot already past the barrier", [&] {
        BarrierOption opt(Barrier::DownOut, 105.0, 0.0,
                          ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                          ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(MakeMCBarrierEngine<PseudoRandom>(process)
                                 .withSteps(8)
                                 .withSamples(1023)
                                 .withSeed(42));
        opt.NPV();
    });
    probe("mc_engine_non_plain_payoff", "CashOrNothingPayoff", [&] {
        BarrierOption opt(
            Barrier::DownOut, 90.0, 0.0,
            ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(MakeMCBarrierEngine<PseudoRandom>(process)
                                 .withSteps(8)
                                 .withSamples(1023)
                                 .withSeed(42));
        opt.NPV();
    });
    probe("mc_tol_maxsamples_exceeded", "maxSamples reached before tolerance",
          [&] {
              BarrierOption opt(
                  Barrier::DownOut, 90.0, 0.0,
                  ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                  ext::make_shared<EuropeanExercise>(kToday + 365));
              opt.setPricingEngine(MakeMCBarrierEngine<PseudoRandom>(process)
                                       .withSteps(8)
                                       .withAbsoluteTolerance(1e-4)
                                       .withMaxSamples(2000)
                                       .withSeed(42));
              opt.NPV();
          });

    // MakeMCBarrierEngine::withBias() defaults to true.  Priced twice on the
    // same setup -- once through `withBias()` with no argument, once through
    // the explicit `withBias(true)` -- so the default is proven by equality
    // rather than asserted.
    {
        auto priceWithBias = [&](bool explicitCall) {
            MakeMCBarrierEngine<PseudoRandom> make(process);
            make.withSteps(8).withSamples(4095).withSeed(42);
            if (explicitCall)
                make.withBias(true);
            else
                make.withBias();
            BarrierOption opt(
                Barrier::DownOut, 90.0, 4.0,
                ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                ext::make_shared<EuropeanExercise>(kToday + 365));
            opt.setPricingEngine(make);
            return opt.NPV();
        };
        Obj in;
        describeBsm(in, 100.0, 0.05, 0.02, 0.20, 365);
        describeBarrier(in, Barrier::DownOut, 90.0, 4.0, Option::Call, 100.0,
                        "European");
        in.s("scenario", "withBias() default argument");
        in.i("steps", 8);
        in.i("samples", 4095);
        in.i("seed", 42);
        Obj ex;
        ex.n("npv_default_arg", priceWithBias(false));
        ex.n("npv_explicit_true", priceWithBias(true));
        addCase("mc_make_bias_default_is_true", in, ex);
    }

    // MakeMCBarrierEngine defaults with nothing but steps/samples/seed set:
    // brownianBridge_ = antithetic_ = biased_ = false.  Must equal the
    // explicitly-parameterised twin `mc_ps_downout_unbiased`.
    {
        BarrierOption opt(
            Barrier::DownOut, 90.0, 4.0,
            ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
            ext::make_shared<EuropeanExercise>(kToday + 365));
        opt.setPricingEngine(MakeMCBarrierEngine<PseudoRandom>(process)
                                 .withSteps(8)
                                 .withSamples(4095)
                                 .withSeed(42));
        Obj in;
        describeBsm(in, 100.0, 0.05, 0.02, 0.20, 365);
        describeBarrier(in, Barrier::DownOut, 90.0, 4.0, Option::Call, 100.0,
                        "European");
        in.s("scenario", "all Make* knobs left at their defaults");
        in.i("steps", 8);
        in.i("samples", 4095);
        in.i("seed", 42);
        Obj ex;
        ex.n("npv", opt.NPV());
        ex.n("error_estimate", opt.errorEstimate());
        ex.b("has_error_estimate", true);
        addCase("mc_make_defaults", in, ex);
    }
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    discretizedCases();
    binomialCases();
    pathPricerCases();
    mcCases();

    emitDocument();
    return 0;
}
