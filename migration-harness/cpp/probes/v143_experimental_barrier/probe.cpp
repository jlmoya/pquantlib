// migration-harness/cpp/probes/v143_experimental_barrier/probe.cpp
//
// Reference values for ql/experimental/barrieroption @ v1.43, covering the
// eight classes PQuantLib was missing:
//
//   DiscretizedDoubleBarrierOption            discretizeddoublebarrieroption.hpp
//   DiscretizedDermanKaniDoubleBarrierOption  discretizeddoublebarrieroption.hpp
//   DoubleBarrierPathPricer                   mcdoublebarrierengine.hpp
//   MCDoubleBarrierEngine                     mcdoublebarrierengine.hpp
//   MakeMCDoubleBarrierEngine                 mcdoublebarrierengine.hpp
//   QuantoDoubleBarrierOption                 quantodoublebarrieroption.hpp
//   SuoWangDoubleBarrierEngine                suowangdoublebarrierengine.hpp
//   PerturbativeBarrierOptionEngine           perturbativebarrieroptionengine.hpp
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// A) The two discretized assets are lattice helpers, not engines. Pinning only
//    the converged NPV would hide a wrong barrier adjustment behind binomial
//    convergence, so block A drives the assets one lattice step at a time and
//    emits the FULL post-adjustValues() value vector at every step, plus the
//    inner vanilla() vector. A wrong `checkBarrier` or a wrong Derman-Kani
//    interpolation shows up in the very first slice.
//
// B) DoubleBarrierPathPricer is pinned twice: directly on hand-built Paths
//    (block B) so the knock logic is isolated from the RNG, and through the
//    engine (block C). Note DoubleBarrierPathPricer::operator() only handles
//    KnockIn/KnockOut; KIKO/KOKI reach `QL_FAIL("unknown barrier type")`. That
//    throw is pinned as behaviour.
//
// C) MCDoubleBarrierEngine is deterministic for a fixed seed, so NPV,
//    errorEstimate and the sample count are all pinned exactly (no statistical
//    band). MakeMCDoubleBarrierEngine is pinned by building the same engine
//    both ways and by pinning each QL_REQUIRE it raises.
//
// F) PerturbativeBarrierOptionEngine's numerics live entirely in an anonymous
//    namespace inside the .cpp, so they are not reachable by linking. This
//    probe therefore #includes the translation unit itself: that makes PHID,
//    ND2, tvtl, BVTL, STUDNT, ff, v, llold, dvv/dff/dll, ddvv/ddff/ddll,
//    derivn3 and BarrierUPD directly callable, and every one of them is pinned
//    at hand-chosen arguments. That matters because order-2 pricing is
//    "Too slow, skip" even in the C++ test suite (test-suite/barrieroption.cpp
//    testPerturbative), so the second-order helper functions can only be
//    cross-validated one at a time.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/barrier.json.

#include <ql/exercise.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/doublebarrieroption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/methods/lattices/bsmlattice.hpp>
#include <ql/methods/montecarlo/mctraits.hpp>
#include <ql/methods/montecarlo/path.hpp>
#include <ql/pricingengines/barrier/analyticdoublebarrierengine.hpp>
#include <ql/pricingengines/quanto/quantoengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/quantotermstructure.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>
#include <ql/version.hpp>

#include <ql/experimental/barrieroption/discretizeddoublebarrieroption.hpp>
#include <ql/experimental/barrieroption/mcdoublebarrierengine.hpp>
#include <ql/experimental/barrieroption/quantodoublebarrieroption.hpp>
#include <ql/experimental/barrieroption/suowangdoublebarrierengine.hpp>

// See note (F): including the TU exposes its anonymous-namespace numerics.
// The probe links against libQuantLib too, so PerturbativeBarrierOptionEngine's
// member definitions exist twice; the executable's copy wins at load time and
// both copies come from this exact source file.
#include <ql/experimental/barrieroption/perturbativebarrieroptionengine.cpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using QuantLib::Array;
using QuantLib::Real;
using QuantLib::Size;
using QuantLib::Time;

namespace {

// --------------------------------------------------------------------------
// JSON emission (flat top-level object; the pquantlib harness convention)
// --------------------------------------------------------------------------

bool g_first = true;

void sep() {
    if (!g_first) std::cout << ",\n";
    g_first = false;
}

void emit(const std::string& name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
}

void emit_int(const std::string& name, long long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_bool(const std::string& name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const std::string& name, const std::vector<long long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

std::vector<Real> to_vec(const Array& a) {
    return std::vector<Real>(a.begin(), a.end());
}

// --------------------------------------------------------------------------
// Shared market setup. Fixed evaluation date — never Date::todaysDate().
// --------------------------------------------------------------------------

const QuantLib::Date TODAY(15, QuantLib::May, 2023);

QuantLib::ext::shared_ptr<QuantLib::GeneralizedBlackScholesProcess>
make_bsm(Real spot, QuantLib::Rate q, QuantLib::Rate r, QuantLib::Volatility vol,
         const QuantLib::DayCounter& dc) {
    QuantLib::Handle<QuantLib::Quote> u(
        QuantLib::ext::make_shared<QuantLib::SimpleQuote>(spot));
    QuantLib::Handle<QuantLib::YieldTermStructure> qTS(
        QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, q, dc));
    QuantLib::Handle<QuantLib::YieldTermStructure> rTS(
        QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, r, dc));
    QuantLib::Handle<QuantLib::BlackVolTermStructure> volTS(
        QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
            TODAY, QuantLib::NullCalendar(), vol, dc));
    return QuantLib::ext::make_shared<QuantLib::BlackScholesMertonProcess>(
        u, qTS, rTS, volTS);
}

const char* dbt_name(QuantLib::DoubleBarrier::Type t) {
    switch (t) {
      case QuantLib::DoubleBarrier::KnockIn:  return "KnockIn";
      case QuantLib::DoubleBarrier::KnockOut: return "KnockOut";
      case QuantLib::DoubleBarrier::KIKO:     return "KIKO";
      case QuantLib::DoubleBarrier::KOKI:     return "KOKI";
    }
    return "?";
}

// ==========================================================================
// Block A — DiscretizedDoubleBarrierOption / DiscretizedDermanKaniDoubleBarrierOption
// ==========================================================================
//
// Drives the discretized asset through a CRR BlackScholesLattice one grid
// step at a time, emitting the post-adjustValues() vector at each slice. The
// lattice setup is exactly BinomialDoubleBarrierEngine::calculate's
// (binomialdoublebarrierengine.hpp:71-118): flat curves rebuilt from the
// maturity zero rates, TimeGrid(maturity, steps), CRR tree, BSM lattice.

struct DiscretizedCase {
    const char* tag;
    QuantLib::DoubleBarrier::Type barrierType;
    Real lo, hi, rebate;
    QuantLib::Option::Type type;
    Real spot, strike;
    QuantLib::Rate q, r;
    QuantLib::Volatility vol;
    QuantLib::Natural days;
    Size steps;
    bool dermanKani;
};

void run_discretized(const DiscretizedCase& c) {
    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    auto process = make_bsm(c.spot, c.q, c.r, c.vol, dc);

    QuantLib::Date exDate = TODAY + c.days;
    auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(c.type, c.strike);

    QuantLib::DoubleBarrierOption opt(c.barrierType, c.lo, c.hi, c.rebate, payoff, exercise);
    QuantLib::DoubleBarrierOption::arguments args;
    opt.setupArguments(&args);
    args.validate();

    // --- flat-coefficient rebuild, verbatim from the binomial engine ---
    QuantLib::Real s0 = process->stateVariable()->value();
    QuantLib::Volatility v = process->blackVolatility()->blackVol(exDate, s0);
    QuantLib::Rate r = process->riskFreeRate()->zeroRate(
        exDate, dc, QuantLib::Continuous, QuantLib::NoFrequency);
    QuantLib::Rate q = process->dividendYield()->zeroRate(
        exDate, dc, QuantLib::Continuous, QuantLib::NoFrequency);
    QuantLib::Date refDate = process->riskFreeRate()->referenceDate();

    QuantLib::Handle<QuantLib::YieldTermStructure> flatRiskFree(
        QuantLib::ext::make_shared<QuantLib::FlatForward>(refDate, r, dc));
    QuantLib::Handle<QuantLib::YieldTermStructure> flatDividends(
        QuantLib::ext::make_shared<QuantLib::FlatForward>(refDate, q, dc));
    QuantLib::Handle<QuantLib::BlackVolTermStructure> flatVol(
        QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
            refDate, QuantLib::NullCalendar(), v, dc));

    Time maturity = dc.yearFraction(refDate, exDate);
    QuantLib::ext::shared_ptr<QuantLib::StochasticProcess1D> bs(
        new QuantLib::GeneralizedBlackScholesProcess(
            process->stateVariable(), flatDividends, flatRiskFree, flatVol));

    QuantLib::TimeGrid grid(maturity, c.steps);
    auto tree = QuantLib::ext::make_shared<QuantLib::CoxRossRubinstein>(
        bs, maturity, c.steps, payoff->strike());
    auto lattice = QuantLib::ext::make_shared<
        QuantLib::BlackScholesLattice<QuantLib::CoxRossRubinstein> >(
            tree, r, maturity, c.steps);

    const std::string p = std::string("discretized_") + c.tag;

    emit_str(p + "_barrier_type", dbt_name(c.barrierType));
    emit(p + "_maturity", maturity);
    emit(p + "_flat_r", r);
    emit(p + "_flat_q", q);
    emit(p + "_flat_vol", v);
    emit_int(p + "_steps", static_cast<long long>(c.steps));
    emit_int(p + "_ex_date_serial", static_cast<long long>(exDate.serialNumber()));
    emit_int(p + "_ref_date_serial", static_cast<long long>(refDate.serialNumber()));

    std::vector<Real> gridtimes;
    for (Size i = 0; i < grid.size(); ++i) gridtimes.push_back(grid[i]);
    emit_arr(p + "_grid", gridtimes);

    // lattice underlyings per slice, flattened
    std::vector<Real> underlyings;
    for (Size i = 0; i <= c.steps; ++i)
        for (Size j = 0; j <= i; ++j)
            underlyings.push_back(lattice->underlying(i, j));
    emit_arr(p + "_underlyings_flat", underlyings);

    if (!c.dermanKani) {
        QuantLib::DiscretizedDoubleBarrierOption asset(args, *process, grid);
        asset.initialize(lattice, maturity);

        std::vector<Real> flat, vanilla_flat;
        for (Real x : to_vec(asset.values())) flat.push_back(x);
        for (Real x : to_vec(asset.vanilla())) vanilla_flat.push_back(x);
        for (Size i = c.steps; i > 0; --i) {
            asset.rollback(grid[i - 1]);
            for (Real x : to_vec(asset.values())) flat.push_back(x);
            for (Real x : to_vec(asset.vanilla())) vanilla_flat.push_back(x);
        }
        emit_arr(p + "_values_flat", flat);
        emit_arr(p + "_vanilla_flat", vanilla_flat);
        emit(p + "_present_value", asset.presentValue());

        std::vector<Real> mt = asset.mandatoryTimes();
        emit_arr(p + "_mandatory_times", mt);
    } else {
        QuantLib::DiscretizedDermanKaniDoubleBarrierOption asset(args, *process, grid);
        asset.initialize(lattice, maturity);

        std::vector<Real> flat;
        for (Real x : to_vec(asset.values())) flat.push_back(x);
        for (Size i = c.steps; i > 0; --i) {
            asset.rollback(grid[i - 1]);
            for (Real x : to_vec(asset.values())) flat.push_back(x);
        }
        emit_arr(p + "_values_flat", flat);
        emit(p + "_present_value", asset.presentValue());

        std::vector<Real> mt = asset.mandatoryTimes();
        emit_arr(p + "_mandatory_times", mt);
    }
}

// checkBarrier() called directly on a synthetic grid, so the branch table is
// pinned independently of any lattice.
void run_check_barrier() {
    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    auto process = make_bsm(100.0, 0.02, 0.05, 0.25, dc);
    QuantLib::Date exDate = TODAY + 180;
    auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
        QuantLib::Option::Call, 100.0);
    Time maturity = dc.yearFraction(TODAY, exDate);

    const Size steps = 8;
    QuantLib::TimeGrid grid(maturity, steps);

    Array synthetic(9);
    const Real levels[9] = {70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0};
    for (Size j = 0; j < 9; ++j) synthetic[j] = levels[j];
    emit_arr("check_barrier_grid_levels", to_vec(synthetic));

    // The one-step-back slice is probed with EIGHT levels, not nine.
    // checkBarrier's knocked-in branches read `vanilla()[j]` for every j in
    // [0, optvalues.size()), and after one rollback the contained
    // DiscretizedVanillaOption holds the step-7 slice, which has only 8 nodes.
    // Feeding a 9-long optvalues there made `vanilla()[8]` an out-of-bounds
    // read: the KnockIn and KOKI vectors came back with a different garbage
    // value on every run (observed 1.3e-114, 0, 9.8e+261 across three runs of
    // the previous probe build). Nine levels stay in use for the at-maturity
    // slice, where vanilla() legitimately has 9 nodes.
    Array synthetic8(8);
    for (Size j = 0; j < 8; ++j) synthetic8[j] = levels[j];
    emit_arr("check_barrier_grid_levels_one_step_back", to_vec(synthetic8));

    const QuantLib::DoubleBarrier::Type types[4] = {
        QuantLib::DoubleBarrier::KnockIn, QuantLib::DoubleBarrier::KnockOut,
        QuantLib::DoubleBarrier::KIKO, QuantLib::DoubleBarrier::KOKI};

    for (auto bt : types) {
        QuantLib::DoubleBarrierOption opt(bt, 90.0, 110.0, 3.0, payoff, exercise);
        QuantLib::DoubleBarrierOption::arguments args;
        opt.setupArguments(&args);
        args.validate();

        QuantLib::DiscretizedDoubleBarrierOption asset(args, *process, grid);
        // A real lattice is needed so vanilla()/time() are meaningful.
        auto tree = QuantLib::ext::make_shared<QuantLib::CoxRossRubinstein>(
            QuantLib::ext::shared_ptr<QuantLib::StochasticProcess1D>(process),
            maturity, steps, payoff->strike());
        auto lattice = QuantLib::ext::make_shared<
            QuantLib::BlackScholesLattice<QuantLib::CoxRossRubinstein> >(
                tree, 0.05, maturity, steps);
        asset.initialize(lattice, maturity);

        const std::string p = std::string("check_barrier_") + dbt_name(bt);
        // at maturity (endTime true, stoppingTime true for European)
        Array optvalues(9, 0.0);
        for (Size j = 0; j < 9; ++j) optvalues[j] = 1.5;
        asset.checkBarrier(optvalues, synthetic);
        emit_arr(p + "_at_maturity", to_vec(optvalues));
        emit_arr(p + "_vanilla_at_maturity", to_vec(asset.vanilla()));

        // one step back (endTime false, stoppingTime false)
        asset.rollback(grid[steps - 1]);
        Array optvalues2(8, 0.0);
        for (Size j = 0; j < 8; ++j) optvalues2[j] = 1.5;
        asset.checkBarrier(optvalues2, synthetic8);
        emit_arr(p + "_one_step_back", to_vec(optvalues2));
        emit_arr(p + "_vanilla_one_step_back", to_vec(asset.vanilla()));
        emit(p + "_time_one_step_back", asset.time());
    }
}

// ==========================================================================
// Block B — DoubleBarrierPathPricer on hand-built Paths
// ==========================================================================

struct PathCase {
    const char* tag;
    Real levels[6];
};

void run_path_pricer() {
    // 5 steps over one year, discounts from a flat 4% continuous curve.
    const Size n = 6;
    QuantLib::TimeGrid grid(1.0, n - 1);
    std::vector<QuantLib::DiscountFactor> discounts(n);
    for (Size i = 0; i < n; ++i) discounts[i] = std::exp(-0.04 * grid[i]);

    std::vector<Real> gt;
    for (Size i = 0; i < grid.size(); ++i) gt.push_back(grid[i]);
    emit_arr("pp_grid", gt);
    emit_arr("pp_discounts", std::vector<Real>(discounts.begin(), discounts.end()));

    const PathCase cases[6] = {
        {"inside",      {100.0, 101.0,  99.0, 103.0,  98.0, 102.0}},  // never touches
        {"cross_hi",    {100.0, 105.0, 112.0, 103.0,  98.0, 102.0}},  // hits hi at i=2
        {"cross_lo",    {100.0,  95.0,  92.0,  88.0,  98.0, 102.0}},  // hits lo at i=3
        {"touch_first", {100.0, 110.0, 100.0, 100.0, 100.0,  99.0}},  // hits hi at i=1 (>=)
        {"touch_last",  {100.0, 101.0, 102.0, 103.0, 104.0,  90.0}},  // hits lo at i=5 (<=)
        {"touch_zero",  { 90.0, 101.0, 102.0, 103.0, 104.0, 105.0}},  // index 0 is NOT scanned
    };

    const Real lo = 90.0, hi = 110.0, rebate = 2.5;
    const QuantLib::DoubleBarrier::Type kinds[2] = {
        QuantLib::DoubleBarrier::KnockOut, QuantLib::DoubleBarrier::KnockIn};
    const QuantLib::Option::Type otypes[2] = {QuantLib::Option::Call, QuantLib::Option::Put};

    for (const auto& pc : cases) {
        Array vals(n);
        for (Size i = 0; i < n; ++i) vals[i] = pc.levels[i];
        QuantLib::Path path(grid, vals);
        emit_arr(std::string("pp_path_") + pc.tag, to_vec(vals));

        for (auto bt : kinds) {
            for (auto ot : otypes) {
                QuantLib::DoubleBarrierPathPricer pricer(
                    bt, lo, hi, rebate, ot, 100.0, discounts);
                const std::string key = std::string("pp_") + pc.tag + "_" +
                                        dbt_name(bt) + "_" +
                                        (ot == QuantLib::Option::Call ? "Call" : "Put");
                emit(key, pricer(path));
            }
        }
    }

    // KIKO / KOKI reach QL_FAIL("unknown barrier type") in operator().
    for (auto bt : {QuantLib::DoubleBarrier::KIKO, QuantLib::DoubleBarrier::KOKI}) {
        Array vals(n);
        for (Size i = 0; i < n; ++i) vals[i] = cases[0].levels[i];
        QuantLib::Path path(grid, vals);
        QuantLib::DoubleBarrierPathPricer pricer(
            bt, lo, hi, rebate, QuantLib::Option::Call, 100.0, discounts);
        bool threw = false;
        try {
            (void)pricer(path);
        } catch (const std::exception&) {
            threw = true;
        }
        emit_bool(std::string("pp_") + dbt_name(bt) + "_throws", threw);
    }

    // ctor QL_REQUIREs
    {
        bool threw = false;
        try {
            QuantLib::DoubleBarrierPathPricer(QuantLib::DoubleBarrier::KnockOut, lo, hi,
                                              rebate, QuantLib::Option::Call, -1.0, discounts);
        } catch (const std::exception&) { threw = true; }
        emit_bool("pp_ctor_negative_strike_throws", threw);
    }
    {
        bool threw = false;
        try {
            QuantLib::DoubleBarrierPathPricer(QuantLib::DoubleBarrier::KnockOut, 0.0, hi,
                                              rebate, QuantLib::Option::Call, 100.0, discounts);
        } catch (const std::exception&) { threw = true; }
        emit_bool("pp_ctor_zero_low_barrier_throws", threw);
    }
    {
        bool threw = false;
        try {
            QuantLib::DoubleBarrierPathPricer(QuantLib::DoubleBarrier::KnockOut, lo, 0.0,
                                              rebate, QuantLib::Option::Call, 100.0, discounts);
        } catch (const std::exception&) { threw = true; }
        emit_bool("pp_ctor_zero_high_barrier_throws", threw);
    }
    {
        // single-point path -> QL_REQUIRE(n>1)
        QuantLib::TimeGrid g1(1.0, 1);
        Array v1(2);
        v1[0] = 100.0; v1[1] = 100.0;
        QuantLib::Path p1(g1, v1);
        std::vector<QuantLib::DiscountFactor> d1(2, 1.0);
        QuantLib::DoubleBarrierPathPricer pricer(QuantLib::DoubleBarrier::KnockOut, lo, hi,
                                                 rebate, QuantLib::Option::Call, 100.0, d1);
        emit("pp_two_point_path", pricer(p1));
    }
}

// ==========================================================================
// Block C — MCDoubleBarrierEngine / MakeMCDoubleBarrierEngine
// ==========================================================================

struct McCase {
    const char* tag;
    QuantLib::DoubleBarrier::Type barrierType;
    QuantLib::Option::Type type;
    Real lo, hi, rebate, strike;
    Size steps;
    Size samples;
    QuantLib::BigNatural seed;
    bool antithetic;
};

void run_mc() {
    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    auto process = make_bsm(100.0, 0.02, 0.05, 0.20, dc);
    QuantLib::Date exDate = TODAY + 180;
    auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);

    emit_int("mc_ex_date_serial", static_cast<long long>(exDate.serialNumber()));
    emit("mc_residual_time", process->time(exDate));

    const McCase cases[4] = {
        {"ko_call", QuantLib::DoubleBarrier::KnockOut, QuantLib::Option::Call,
         90.0, 110.0, 0.0, 100.0, 32, 4095, 42, false},
        {"ki_call", QuantLib::DoubleBarrier::KnockIn, QuantLib::Option::Call,
         90.0, 110.0, 0.0, 100.0, 32, 4095, 42, false},
        {"ko_put_rebate", QuantLib::DoubleBarrier::KnockOut, QuantLib::Option::Put,
         85.0, 115.0, 3.0, 100.0, 16, 2047, 7, false},
        {"ki_put_anti", QuantLib::DoubleBarrier::KnockIn, QuantLib::Option::Put,
         85.0, 115.0, 1.5, 100.0, 16, 2048, 7, true},
    };

    for (const auto& c : cases) {
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(c.type, c.strike);
        QuantLib::DoubleBarrierOption opt(c.barrierType, c.lo, c.hi, c.rebate, payoff, exercise);

        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
            new QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                process, c.steps, QuantLib::Null<Size>(), false, c.antithetic,
                c.samples, QuantLib::Null<Real>(), QuantLib::Null<Size>(), c.seed));
        opt.setPricingEngine(engine);

        const std::string p = std::string("mc_") + c.tag;
        emit(p + "_npv", opt.NPV());
        emit(p + "_error_estimate", opt.errorEstimate());
        emit_int(p + "_steps", static_cast<long long>(c.steps));
        emit_int(p + "_samples", static_cast<long long>(c.samples));
        emit_int(p + "_seed", static_cast<long long>(c.seed));
        emit_bool(p + "_antithetic", c.antithetic);
    }

    // steps-per-year variant: TimeGrid(residual, max(int(spy*residual),1))
    {
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 100.0);
        QuantLib::DoubleBarrierOption opt(QuantLib::DoubleBarrier::KnockOut,
                                          90.0, 110.0, 0.0, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
            new QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                process, QuantLib::Null<Size>(), 52, false, false,
                1023, QuantLib::Null<Real>(), QuantLib::Null<Size>(), 3));
        opt.setPricingEngine(engine);
        emit("mc_steps_per_year_npv", opt.NPV());
        emit("mc_steps_per_year_error_estimate", opt.errorEstimate());
    }

    // tolerance-driven run (exercises McSimulation::value's growth loop)
    {
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 100.0);
        QuantLib::DoubleBarrierOption opt(QuantLib::DoubleBarrier::KnockOut,
                                          90.0, 110.0, 0.0, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
            new QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                process, 16, QuantLib::Null<Size>(), false, false,
                QuantLib::Null<Size>(), 0.02, QuantLib::Null<Size>(), 11));
        opt.setPricingEngine(engine);
        emit("mc_tolerance_npv", opt.NPV());
        emit("mc_tolerance_error_estimate", opt.errorEstimate());
    }

    // brownian-bridge variant
    {
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 100.0);
        QuantLib::DoubleBarrierOption opt(QuantLib::DoubleBarrier::KnockOut,
                                          90.0, 110.0, 0.0, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
            new QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                process, 8, QuantLib::Null<Size>(), true, false,
                1023, QuantLib::Null<Real>(), QuantLib::Null<Size>(), 5));
        opt.setPricingEngine(engine);
        emit("mc_brownian_bridge_npv", opt.NPV());
        emit("mc_brownian_bridge_error_estimate", opt.errorEstimate());
    }

    // --- MakeMCDoubleBarrierEngine: same engine, fluent route --------------
    {
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 100.0);
        QuantLib::DoubleBarrierOption opt(QuantLib::DoubleBarrier::KnockOut,
                                          90.0, 110.0, 0.0, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> built =
            QuantLib::MakeMCDoubleBarrierEngine<QuantLib::PseudoRandom>(process)
                .withSteps(32)
                .withSamples(4095)
                .withSeed(42);
        opt.setPricingEngine(built);
        emit("make_mc_ko_call_npv", opt.NPV());
        emit("make_mc_ko_call_error_estimate", opt.errorEstimate());
    }
    {
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 100.0);
        QuantLib::DoubleBarrierOption opt(QuantLib::DoubleBarrier::KnockOut,
                                          90.0, 110.0, 0.0, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> built =
            QuantLib::MakeMCDoubleBarrierEngine<QuantLib::PseudoRandom>(process)
                .withStepsPerYear(52)
                .withBrownianBridge(true)
                .withAntitheticVariate(true)
                .withAbsoluteTolerance(0.05)
                .withMaxSamples(1 << 20)
                .withSeed(99);
        opt.setPricingEngine(built);
        emit("make_mc_fluent_all_npv", opt.NPV());
        emit("make_mc_fluent_all_error_estimate", opt.errorEstimate());
    }

    // requirement checks raised by the builder
    {
        bool threw = false;
        try {
            QuantLib::ext::shared_ptr<QuantLib::PricingEngine> e =
                QuantLib::MakeMCDoubleBarrierEngine<QuantLib::PseudoRandom>(process)
                    .withSamples(1023);
        } catch (const std::exception&) { threw = true; }
        emit_bool("make_mc_no_steps_throws", threw);
    }
    {
        bool threw = false;
        try {
            QuantLib::ext::shared_ptr<QuantLib::PricingEngine> e =
                QuantLib::MakeMCDoubleBarrierEngine<QuantLib::PseudoRandom>(process)
                    .withSteps(10).withStepsPerYear(12).withSamples(1023);
        } catch (const std::exception&) { threw = true; }
        emit_bool("make_mc_steps_overspecified_throws", threw);
    }
    {
        bool threw = false;
        try {
            QuantLib::MakeMCDoubleBarrierEngine<QuantLib::PseudoRandom>(process)
                .withAbsoluteTolerance(0.01).withSamples(1023);
        } catch (const std::exception&) { threw = true; }
        emit_bool("make_mc_samples_after_tolerance_throws", threw);
    }
    {
        bool threw = false;
        try {
            QuantLib::MakeMCDoubleBarrierEngine<QuantLib::PseudoRandom>(process)
                .withSamples(1023).withAbsoluteTolerance(0.01);
        } catch (const std::exception&) { threw = true; }
        emit_bool("make_mc_tolerance_after_samples_throws", threw);
    }
    // engine ctor requirements
    {
        bool threw = false;
        try {
            QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                process, QuantLib::Null<Size>(), QuantLib::Null<Size>(), false, false,
                1023, QuantLib::Null<Real>(), QuantLib::Null<Size>(), 1);
        } catch (const std::exception&) { threw = true; }
        emit_bool("mc_engine_no_time_steps_throws", threw);
    }
    {
        bool threw = false;
        try {
            QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                process, 10, 12, false, false,
                1023, QuantLib::Null<Real>(), QuantLib::Null<Size>(), 1);
        } catch (const std::exception&) { threw = true; }
        emit_bool("mc_engine_both_time_steps_throws", threw);
    }
    {
        bool threw = false;
        try {
            QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                process, 0, QuantLib::Null<Size>(), false, false,
                1023, QuantLib::Null<Real>(), QuantLib::Null<Size>(), 1);
        } catch (const std::exception&) { threw = true; }
        emit_bool("mc_engine_zero_time_steps_throws", threw);
    }
    // barrier already touched -> QL_REQUIRE(!triggered(spot))
    {
        auto p2 = make_bsm(100.0, 0.02, 0.05, 0.20, dc);
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 100.0);
        QuantLib::DoubleBarrierOption opt(QuantLib::DoubleBarrier::KnockOut,
                                          100.0, 110.0, 0.0, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
            new QuantLib::MCDoubleBarrierEngine<QuantLib::PseudoRandom>(
                p2, 8, QuantLib::Null<Size>(), false, false,
                1023, QuantLib::Null<Real>(), QuantLib::Null<Size>(), 1));
        opt.setPricingEngine(engine);
        bool threw = false;
        try { (void)opt.NPV(); } catch (const std::exception&) { threw = true; }
        emit_bool("mc_engine_barrier_touched_throws", threw);
    }
}

// ==========================================================================
// Block D — QuantoDoubleBarrierOption (through QuantoEngine)
// ==========================================================================

struct QuantoCase {
    QuantLib::DoubleBarrier::Type barrierType;
    Real lo, hi, rebate;
    QuantLib::Option::Type type;
    Real spot, strike;
    QuantLib::Rate q, r;
    Time t;
    QuantLib::Volatility v;
    QuantLib::Rate fxr;
    QuantLib::Volatility fxv;
    Real corr;
};

// A deterministic stand-in for the inner engine, so QuantoEngine's greek
// arithmetic (rho += dividendRho, vega += corr*fxvol*dividendRho, qvega,
// qrho, qlambda) and QuantoDoubleBarrierOption::fetchResults are exercised on
// non-Null inputs. `value` is tied to the quanto-adjusted dividend discount so
// QuantoTermStructure::zeroYieldImpl is pinned through it too.
class StubDoubleBarrierEngine : public QuantLib::DoubleBarrierOption::engine {
  public:
    explicit StubDoubleBarrierEngine(
        QuantLib::ext::shared_ptr<QuantLib::GeneralizedBlackScholesProcess> p)
    : process_(std::move(p)) {}

    void calculate() const override {
        Time t = process_->time(arguments_.exercise->lastDate());
        Real qdisc = process_->dividendYield()->discount(t);
        Real rdisc = process_->riskFreeRate()->discount(t);
        results_.value = 10.0 * qdisc;
        results_.delta = 0.5 * qdisc;
        results_.gamma = 0.05 * rdisc;
        results_.theta = -1.25 * qdisc;
        results_.rho = 7.0 * rdisc;
        results_.dividendRho = -3.0 * qdisc;
        results_.vega = 21.0 * qdisc;
    }

  private:
    QuantLib::ext::shared_ptr<QuantLib::GeneralizedBlackScholesProcess> process_;
};

void run_quanto_with_stub() {
    const QuantLib::DayCounter dc = QuantLib::Actual360();
    const Real spot_v = 100.0, strike_v = 102.0, corr_v = 0.3;
    const QuantLib::Rate q_v = 0.01, r_v = 0.1, fxr_v = 0.05;
    const QuantLib::Volatility vol_v = 0.15, fxvol_v = 0.2;

    auto spot = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(spot_v);
    QuantLib::Handle<QuantLib::YieldTermStructure> qTS(
        QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, q_v, dc));
    QuantLib::Handle<QuantLib::YieldTermStructure> rTS(
        QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, r_v, dc));
    QuantLib::Handle<QuantLib::BlackVolTermStructure> volTS(
        QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
            TODAY, QuantLib::NullCalendar(), vol_v, dc));
    QuantLib::Handle<QuantLib::YieldTermStructure> fxrTS(
        QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, fxr_v, dc));
    QuantLib::Handle<QuantLib::BlackVolTermStructure> fxVolTS(
        QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
            TODAY, QuantLib::NullCalendar(), fxvol_v, dc));
    QuantLib::Handle<QuantLib::Quote> corr(
        QuantLib::ext::make_shared<QuantLib::SimpleQuote>(corr_v));

    auto process = QuantLib::ext::make_shared<QuantLib::BlackScholesMertonProcess>(
        QuantLib::Handle<QuantLib::Quote>(spot), qTS, rTS, volTS);

    QuantLib::Date exDate = TODAY + 90;
    auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
        QuantLib::Option::Call, strike_v);

    QuantLib::QuantoDoubleBarrierOption option(
        QuantLib::DoubleBarrier::KnockOut, 90.0, 110.0, 1.5, payoff, exercise);
    QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
        new QuantLib::QuantoEngine<QuantLib::DoubleBarrierOption,
                                   StubDoubleBarrierEngine>(
            process, fxrTS, fxVolTS, corr));
    option.setPricingEngine(engine);

    emit_int("quanto_stub_ex_date_serial", static_cast<long long>(exDate.serialNumber()));
    emit("quanto_stub_spot", spot_v);
    emit("quanto_stub_strike", strike_v);
    emit("quanto_stub_q", q_v);
    emit("quanto_stub_r", r_v);
    emit("quanto_stub_vol", vol_v);
    emit("quanto_stub_fxr", fxr_v);
    emit("quanto_stub_fxvol", fxvol_v);
    emit("quanto_stub_corr", corr_v);
    emit("quanto_stub_npv", option.NPV());
    emit("quanto_stub_delta", option.delta());
    emit("quanto_stub_gamma", option.gamma());
    emit("quanto_stub_theta", option.theta());
    emit("quanto_stub_rho", option.rho());
    emit("quanto_stub_dividend_rho", option.dividendRho());
    emit("quanto_stub_vega", option.vega());
    emit("quanto_stub_qvega", option.qvega());
    emit("quanto_stub_qrho", option.qrho());
    emit("quanto_stub_qlambda", option.qlambda());

    // QuantoEngine's "wrong engine type" guard. calculate() downcasts the
    // inner engine's argument bundle to Instr::arguments; AnalyticEuropeanEngine
    // carries VanillaOption::arguments, which is a *base* of
    // DoubleBarrierOption::arguments, so the dynamic_cast returns null and the
    // QL_REQUIRE fires (quantoengine.hpp:119-121).
    {
        QuantLib::QuantoDoubleBarrierOption bad(
            QuantLib::DoubleBarrier::KnockOut, 90.0, 110.0, 1.5, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> badEngine(
            new QuantLib::QuantoEngine<QuantLib::DoubleBarrierOption,
                                       QuantLib::AnalyticEuropeanEngine>(
                process, fxrTS, fxVolTS, corr));
        bad.setPricingEngine(badEngine);
        bool threw = false;
        try { (void)bad.NPV(); } catch (const std::exception&) { threw = true; }
        emit_bool("quanto_wrong_engine_type_throws", threw);
    }

    // The quanto-adjusted dividend curve itself (QuantoTermStructure).
    {
        QuantLib::Handle<QuantLib::YieldTermStructure> quantoDiv(
            QuantLib::ext::make_shared<QuantLib::QuantoTermStructure>(
                process->dividendYield(), process->riskFreeRate(), fxrTS,
                process->blackVolatility(), strike_v, fxVolTS, 1.0, corr_v));
        std::vector<Real> ts, zs, dfs;
        for (Real t : {0.05, 0.25, 0.5, 1.0, 2.0}) {
            ts.push_back(t);
            zs.push_back(quantoDiv->zeroRate(t, QuantLib::Continuous,
                                             QuantLib::NoFrequency, true).rate());
            dfs.push_back(quantoDiv->discount(t, true));
        }
        emit_arr("quanto_ts_times", ts);
        emit_arr("quanto_ts_zero_rates", zs);
        emit_arr("quanto_ts_discounts", dfs);
        // The five forwarding accessors QuantoTermStructure overrides. Each
        // one delegates to the *underlying dividend* curve, which here is a
        // FlatForward built from an explicit reference date — so it carries
        // neither settlement days nor a calendar, and those two accessors
        // throw. That is pinned as behaviour rather than papered over.
        emit_int("quanto_ts_reference_date_serial",
                 static_cast<long long>(quantoDiv->referenceDate().serialNumber()));
        emit_int("quanto_ts_max_date_serial",
                 static_cast<long long>(quantoDiv->maxDate().serialNumber()));
        emit_str("quanto_ts_day_counter", quantoDiv->dayCounter().name());
        {
            bool threw = false;
            try { (void)quantoDiv->settlementDays(); }
            catch (const std::exception&) { threw = true; }
            emit_bool("quanto_ts_settlement_days_throws", threw);
        }
        {
            bool threw = false;
            try { (void)quantoDiv->calendar().name(); }
            catch (const std::exception&) { threw = true; }
            emit_bool("quanto_ts_calendar_name_throws", threw);
        }
    }
}

// ==========================================================================
// Block G — DiscretizedVanillaOption (standalone)
// ==========================================================================
//
// DiscretizedDoubleBarrierOption holds one by value, but only ever under a
// European exercise, so the American and Bermudan arms of
// DiscretizedVanillaOption::postAdjustValuesImpl are unreachable from block A.
// Drive the class directly on the same CRR/BSM lattice for all three exercise
// types and emit the full value vector at every slice.

void run_discretized_vanilla() {
    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    const Real spot = 100.0, strike = 100.0;
    const QuantLib::Rate q_ = 0.02, r_ = 0.05;
    const QuantLib::Volatility vol_ = 0.25;
    const Size steps = 8;

    auto process = make_bsm(spot, q_, r_, vol_, dc);
    QuantLib::Date exDate = TODAY + 180;
    Time maturity = dc.yearFraction(TODAY, exDate);
    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
        QuantLib::Option::Put, strike);

    QuantLib::TimeGrid grid(maturity, steps);
    std::vector<Real> gridtimes;
    for (Size i = 0; i < grid.size(); ++i) gridtimes.push_back(grid[i]);
    emit_arr("dvo_grid", gridtimes);
    emit("dvo_maturity", maturity);
    emit_int("dvo_steps", static_cast<long long>(steps));
    emit_int("dvo_ex_date_serial", static_cast<long long>(exDate.serialNumber()));

    std::vector<QuantLib::Date> bermudanDates;
    bermudanDates.push_back(TODAY + 45);
    bermudanDates.push_back(TODAY + 90);
    bermudanDates.push_back(TODAY + 135);
    bermudanDates.push_back(exDate);
    std::vector<long long> bermudanSerials;
    for (const auto& d : bermudanDates)
        bermudanSerials.push_back(static_cast<long long>(d.serialNumber()));
    emit_iarr("dvo_bermudan_date_serials", bermudanSerials);

    std::vector<QuantLib::ext::shared_ptr<QuantLib::Exercise> > exercises;
    std::vector<std::string> tags;
    exercises.push_back(QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate));
    tags.emplace_back("european");
    exercises.push_back(QuantLib::ext::make_shared<QuantLib::AmericanExercise>(TODAY, exDate));
    tags.emplace_back("american");
    exercises.push_back(QuantLib::ext::make_shared<QuantLib::BermudanExercise>(bermudanDates));
    tags.emplace_back("bermudan");

    bool underlyingsEmitted = false;
    for (Size k = 0; k < exercises.size(); ++k) {
        QuantLib::VanillaOption opt(payoff, exercises[k]);
        QuantLib::VanillaOption::arguments args;
        opt.setupArguments(&args);
        args.validate();

        auto tree = QuantLib::ext::make_shared<QuantLib::CoxRossRubinstein>(
            QuantLib::ext::shared_ptr<QuantLib::StochasticProcess1D>(process),
            maturity, steps, strike);
        auto lattice = QuantLib::ext::make_shared<
            QuantLib::BlackScholesLattice<QuantLib::CoxRossRubinstein> >(
                tree, r_, maturity, steps);

        if (!underlyingsEmitted) {
            std::vector<Real> underlyings;
            for (Size i = 0; i <= steps; ++i)
                for (Size j = 0; j <= i; ++j)
                    underlyings.push_back(lattice->underlying(i, j));
            emit_arr("dvo_underlyings_flat", underlyings);
            underlyingsEmitted = true;
        }

        QuantLib::DiscretizedVanillaOption asset(args, *process, grid);
        asset.initialize(lattice, maturity);

        std::vector<Real> flat;
        for (Real x : to_vec(asset.values())) flat.push_back(x);
        for (Size i = steps; i > 0; --i) {
            asset.rollback(grid[i - 1]);
            for (Real x : to_vec(asset.values())) flat.push_back(x);
        }
        const std::string p = std::string("dvo_") + tags[k];
        emit_arr(p + "_values_flat", flat);
        emit(p + "_present_value", asset.presentValue());
        emit_arr(p + "_mandatory_times", asset.mandatoryTimes());
    }
}

void run_quanto() {
    // Values from test-suite/quantooption.cpp testDoubleBarrierValues,
    // re-dated to the fixed TODAY.
    const QuantoCase cases[5] = {
        {QuantLib::DoubleBarrier::KnockOut,  50.0, 150.0, 0.0, QuantLib::Option::Call,
         100.0, 100.0, 0.00, 0.1, 0.25, 0.15, 0.05, 0.2, 0.3},
        {QuantLib::DoubleBarrier::KnockOut,  90.0, 110.0, 0.0, QuantLib::Option::Call,
         100.0, 100.0, 0.00, 0.1, 0.50, 0.15, 0.05, 0.2, 0.3},
        {QuantLib::DoubleBarrier::KnockOut,  90.0, 110.0, 0.0, QuantLib::Option::Put,
         100.0, 100.0, 0.00, 0.1, 0.25, 0.15, 0.05, 0.2, 0.3},
        {QuantLib::DoubleBarrier::KnockIn,   80.0, 120.0, 0.0, QuantLib::Option::Call,
         100.0, 102.0, 0.00, 0.1, 0.25, 0.25, 0.05, 0.2, 0.3},
        {QuantLib::DoubleBarrier::KnockIn,   80.0, 120.0, 0.0, QuantLib::Option::Call,
         100.0, 102.0, 0.00, 0.1, 0.50, 0.15, 0.05, 0.2, 0.3},
    };

    const QuantLib::DayCounter dc = QuantLib::Actual360();

    for (Size k = 0; k < 5; ++k) {
        const QuantoCase& c = cases[k];

        auto spot = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(c.spot);
        QuantLib::Handle<QuantLib::YieldTermStructure> qTS(
            QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, c.q, dc));
        QuantLib::Handle<QuantLib::YieldTermStructure> rTS(
            QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, c.r, dc));
        QuantLib::Handle<QuantLib::BlackVolTermStructure> volTS(
            QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
                TODAY, QuantLib::NullCalendar(), c.v, dc));
        QuantLib::Handle<QuantLib::YieldTermStructure> fxrTS(
            QuantLib::ext::make_shared<QuantLib::FlatForward>(TODAY, c.fxr, dc));
        QuantLib::Handle<QuantLib::BlackVolTermStructure> fxVolTS(
            QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
                TODAY, QuantLib::NullCalendar(), c.fxv, dc));
        QuantLib::Handle<QuantLib::Quote> corr(
            QuantLib::ext::make_shared<QuantLib::SimpleQuote>(c.corr));

        auto process = QuantLib::ext::make_shared<QuantLib::BlackScholesMertonProcess>(
            QuantLib::Handle<QuantLib::Quote>(spot), qTS, rTS, volTS);

        // timeToDays(t) with the test-suite default 360-day year
        QuantLib::Date exDate = TODAY + static_cast<QuantLib::Integer>(c.t * 360 + 0.5);
        auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(c.type, c.strike);

        QuantLib::QuantoDoubleBarrierOption option(
            c.barrierType, c.lo, c.hi, c.rebate, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
            new QuantLib::QuantoEngine<QuantLib::DoubleBarrierOption,
                                       QuantLib::AnalyticDoubleBarrierEngine>(
                process, fxrTS, fxVolTS, corr));
        option.setPricingEngine(engine);

        const std::string p = "quanto_" + std::to_string(k);
        emit_str(p + "_barrier_type", dbt_name(c.barrierType));
        emit_int(p + "_ex_date_serial", static_cast<long long>(exDate.serialNumber()));
        emit(p + "_npv", option.NPV());
        // AnalyticDoubleBarrierEngine publishes only `value`, so every greek
        // QuantoEngine copies over stays Null<Real>() and the accessors throw.
        // That is real v1.43 behaviour and is pinned as such.
        bool greeks_throw = false;
        try { (void)option.delta(); } catch (const std::exception&) { greeks_throw = true; }
        emit_bool(p + "_delta_throws", greeks_throw);
        bool qvega_throws = false;
        try { (void)option.qvega(); } catch (const std::exception&) { qvega_throws = true; }
        emit_bool(p + "_qvega_throws", qvega_throws);
        bool qrho_throws = false;
        try { (void)option.qrho(); } catch (const std::exception&) { qrho_throws = true; }
        emit_bool(p + "_qrho_throws", qrho_throws);
        bool qlambda_throws = false;
        try { (void)option.qlambda(); } catch (const std::exception&) { qlambda_throws = true; }
        emit_bool(p + "_qlambda_throws", qlambda_throws);
    }

    run_quanto_with_stub();
}

// ==========================================================================
// Block E — SuoWangDoubleBarrierEngine
// ==========================================================================

struct SuoWangCase {
    const char* tag;
    QuantLib::DoubleBarrier::Type barrierType;
    Real lo, hi, rebate;
    QuantLib::Option::Type type;
    Real spot, strike;
    QuantLib::Rate q, r;
    QuantLib::Volatility v;
    QuantLib::Natural days;
    int series;
};

void run_suowang() {
    const QuantLib::DayCounter dc = QuantLib::Actual360();

    const SuoWangCase cases[8] = {
        {"ko_call_atm",   QuantLib::DoubleBarrier::KnockOut,  90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.00, 0.10, 0.15,  90, 5},
        {"ko_put_atm",    QuantLib::DoubleBarrier::KnockOut,  90.0, 110.0, 0.0,
         QuantLib::Option::Put,  100.0, 100.0, 0.00, 0.10, 0.15,  90, 5},
        {"ki_call_atm",   QuantLib::DoubleBarrier::KnockIn,   90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.00, 0.10, 0.15,  90, 5},
        {"ki_put_wide",   QuantLib::DoubleBarrier::KnockIn,   50.0, 150.0, 0.0,
         QuantLib::Option::Put,  100.0, 100.0, 0.00, 0.10, 0.25, 180, 5},
        {"ko_call_otm",   QuantLib::DoubleBarrier::KnockOut,  80.0, 120.0, 0.0,
         QuantLib::Option::Call, 100.0, 110.0, 0.04, 0.10, 0.20, 180, 5},
        {"ko_put_itm",    QuantLib::DoubleBarrier::KnockOut,  80.0, 120.0, 0.0,
         QuantLib::Option::Put,  100.0, 110.0, 0.04, 0.10, 0.20, 180, 5},
        {"ko_rebate",     QuantLib::DoubleBarrier::KnockOut,  90.0, 110.0, 3.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.00, 0.10, 0.15,  90, 5},
        {"ko_series_12",  QuantLib::DoubleBarrier::KnockOut,  90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.00, 0.10, 0.15,  90, 12},
    };

    for (const auto& c : cases) {
        auto process = make_bsm(c.spot, c.q, c.r, c.v, dc);
        QuantLib::Date exDate = TODAY + c.days;
        auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(c.type, c.strike);

        QuantLib::DoubleBarrierOption opt(c.barrierType, c.lo, c.hi, c.rebate, payoff, exercise);
        QuantLib::ext::shared_ptr<QuantLib::PricingEngine> engine(
            new QuantLib::SuoWangDoubleBarrierEngine(process, c.series));
        opt.setPricingEngine(engine);

        const std::string p = std::string("suowang_") + c.tag;
        emit_int(p + "_ex_date_serial", static_cast<long long>(exDate.serialNumber()));
        emit_int(p + "_series", c.series);
        emit(p + "_npv", opt.NPV());
        const std::map<std::string, QuantLib::ext::any>& ar = opt.additionalResults();
        emit(p + "_vanilla", QuantLib::ext::any_cast<Real>(ar.at("vanilla")));
        emit(p + "_barrier_out", QuantLib::ext::any_cast<Real>(ar.at("barrierOut")));
        emit(p + "_barrier_in", QuantLib::ext::any_cast<Real>(ar.at("barrierIn")));
        emit(p + "_rebate_in", QuantLib::ext::any_cast<Real>(ar.at("rebateIn")));
    }

    // unsupported barrier types
    for (auto bt : {QuantLib::DoubleBarrier::KIKO, QuantLib::DoubleBarrier::KOKI}) {
        auto process = make_bsm(100.0, 0.0, 0.10, 0.15, dc);
        QuantLib::Date exDate = TODAY + 90;
        auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 100.0);
        QuantLib::DoubleBarrierOption opt(bt, 90.0, 110.0, 0.0, payoff, exercise);
        opt.setPricingEngine(QuantLib::ext::make_shared<QuantLib::SuoWangDoubleBarrierEngine>(
            process, 5));
        bool threw = false;
        try { (void)opt.NPV(); } catch (const std::exception&) { threw = true; }
        emit_bool(std::string("suowang_") + dbt_name(bt) + "_throws", threw);
    }
}

// ==========================================================================
// Block F — PerturbativeBarrierOptionEngine
// ==========================================================================

struct PerturbCase {
    const char* tag;
    Real spot, strike, barrier;
    QuantLib::Rate q, r;
    QuantLib::Volatility v0, v1;   // BlackVarianceCurve knots at +90d / +180d
    QuantLib::Natural days;
    QuantLib::Natural order;
    bool zeroGamma;
};

void run_perturbative() {
    const QuantLib::DayCounter dc = QuantLib::Actual360();

    const PerturbCase cases[12] = {
        // the C++ test-suite configuration (testPerturbative), both orders
        {"ref_o0", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 0, false},
        {"ref_o1", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 1, false},
        {"ref_o0_zg", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 0, true},
        {"ref_o1_zg", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 1, true},
        // strike below barrier (xstar < 0 branch)
        {"k_below_o0", 100.0, 95.0, 110.0, 0.02, 0.03, 0.105, 0.11, 180, 0, false},
        {"k_below_o1", 100.0, 95.0, 110.0, 0.02, 0.03, 0.105, 0.11, 180, 1, false},
        {"k_below_o1_zg", 100.0, 95.0, 110.0, 0.02, 0.03, 0.105, 0.11, 180, 1, true},
        // strike above barrier (xstar clamped to 0)
        {"k_above_o1", 100.0, 120.0, 105.0, 0.02, 0.03, 0.105, 0.11, 180, 1, false},
        // spot right under the barrier — worst-conditioned regime
        {"near_barrier_o0", 100.0, 105.0, 100.5, 0.02, 0.03, 0.105, 0.11, 180, 0, false},
        {"near_barrier_o1", 100.0, 105.0, 100.5, 0.02, 0.03, 0.105, 0.11, 180, 1, false},
        {"near_barrier_o1_zg", 100.0, 105.0, 100.5, 0.02, 0.03, 0.105, 0.11, 180, 1, true},
        // short maturity
        {"short_o1", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 90, 1, false},
    };

    for (const auto& c : cases) {
        auto underlying = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(c.spot);
        QuantLib::ext::shared_ptr<QuantLib::YieldTermStructure> qTS(
            new QuantLib::FlatForward(TODAY, c.q, dc));
        QuantLib::ext::shared_ptr<QuantLib::YieldTermStructure> rTS(
            new QuantLib::FlatForward(TODAY, c.r, dc));

        std::vector<QuantLib::Date> dates(2);
        std::vector<QuantLib::Volatility> vols(2);
        dates[0] = TODAY + 90;  vols[0] = c.v0;
        dates[1] = TODAY + 180; vols[1] = c.v1;
        QuantLib::ext::shared_ptr<QuantLib::BlackVolTermStructure> volTS(
            new QuantLib::BlackVarianceCurve(TODAY, dates, vols, dc));

        auto process = QuantLib::ext::make_shared<QuantLib::BlackScholesMertonProcess>(
            QuantLib::Handle<QuantLib::Quote>(underlying),
            QuantLib::Handle<QuantLib::YieldTermStructure>(qTS),
            QuantLib::Handle<QuantLib::YieldTermStructure>(rTS),
            QuantLib::Handle<QuantLib::BlackVolTermStructure>(volTS));

        QuantLib::Date exDate = TODAY + c.days;
        auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Put, c.strike);

        QuantLib::BarrierOption option(QuantLib::Barrier::UpOut, c.barrier, 0.0, payoff, exercise);
        option.setPricingEngine(
            QuantLib::ext::make_shared<QuantLib::PerturbativeBarrierOptionEngine>(
                process, c.order, c.zeroGamma));

        const std::string p = std::string("pert_") + c.tag;
        emit_int(p + "_ex_date_serial", static_cast<long long>(exDate.serialNumber()));
        emit_int(p + "_order", static_cast<long long>(c.order));
        emit_bool(p + "_zero_gamma", c.zeroGamma);
        emit(p + "_tau_max", process->time(exDate));
        emit(p + "_npv", option.NPV());
    }

    // engine requirement checks
    {
        auto process = make_bsm(100.0, 0.02, 0.03, 0.11, dc);
        QuantLib::Date exDate = TODAY + 180;
        auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
        auto put = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Put, 101.0);
        auto call = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, 101.0);

        struct { const char* key; QuantLib::Barrier::Type bt; Real rebate;
                 QuantLib::ext::shared_ptr<QuantLib::PlainVanillaPayoff> po;
                 QuantLib::Natural order; } bad[4] = {
            {"pert_down_out_throws",  QuantLib::Barrier::DownOut, 0.0, put,  1},
            {"pert_rebate_throws",    QuantLib::Barrier::UpOut,   1.0, put,  1},
            {"pert_call_throws",      QuantLib::Barrier::UpOut,   0.0, call, 1},
            {"pert_order_3_throws",   QuantLib::Barrier::UpOut,   0.0, put,  3},
        };
        for (const auto& b : bad) {
            QuantLib::BarrierOption option(b.bt, 101.0, b.rebate, b.po, exercise);
            option.setPricingEngine(
                QuantLib::ext::make_shared<QuantLib::PerturbativeBarrierOptionEngine>(
                    process, b.order, false));
            bool threw = false;
            try { (void)option.NPV(); } catch (const std::exception&) { threw = true; }
            emit_bool(b.key, threw);
        }
    }
}

// The anonymous-namespace numerics, reachable because this TU #includes the
// engine's .cpp. Every helper the second-order term composes is pinned here,
// because the order-2 price itself is "Too slow, skip" even in C++.
void run_perturbative_internals() {
    // PHID: the Hart/Miller normal CDF the engine uses instead of QuantLib's.
    const Real zs[13] = {-40.0, -8.0, -7.071067811865475, -7.0710678118654,
                         -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 7.5, 40.0};
    std::vector<Real> phid;
    for (Real z : zs) phid.push_back(QuantLib::PHID(z));
    emit_arr("int_phid_z", std::vector<Real>(zs, zs + 13));
    emit_arr("int_phid", phid);

    // ND2 (Drezner-Wesolowsky bivariate): the three |rho| regimes and the
    // |rho| >= 0.925 tail on both signs.
    const Real nd2_args[10][3] = {
        {-0.5, -0.5,  0.0},   {0.3, -1.2,  0.25},  {-1.0, 0.7,  0.5},
        {0.8,  0.8,   0.74},  {-1.5, 0.5,  0.9},   {0.2, -0.2,  0.93},
        {1.0,  1.0,  -0.93},  {-2.0, 1.3, -0.6},   {0.0,  0.0,  0.999},
        {0.0,  0.0,  -0.999},
    };
    std::vector<Real> nd2v, nd2a, nd2b, nd2r;
    for (const auto& a : nd2_args) {
        nd2a.push_back(a[0]); nd2b.push_back(a[1]); nd2r.push_back(a[2]);
        nd2v.push_back(QuantLib::ND2(a[0], a[1], a[2]));
    }
    emit_arr("int_nd2_a", nd2a);
    emit_arr("int_nd2_b", nd2b);
    emit_arr("int_nd2_rho", nd2r);
    emit_arr("int_nd2", nd2v);

    // STUDNT / BVTL (only reachable from tvtl's NU>0 branches, but pinned so
    // a Python port of those branches is checked too).
    std::vector<Real> studnt;
    const int nus[5] = {0, 1, 2, 5, 8};
    for (int nu : nus) { studnt.push_back(QuantLib::STUDNT(nu, 0.7)); }
    emit_iarr("int_studnt_nu", std::vector<long long>(nus, nus + 5));
    emit_arr("int_studnt", studnt);

    std::vector<Real> bvtl;
    for (int nu : nus) { bvtl.push_back(QuantLib::BVTL(nu, 0.4, -0.8, 0.35)); }
    emit_arr("int_bvtl", bvtl);
    emit("int_bvtl_r_plus_1", QuantLib::BVTL(0, 0.4, -0.8, -1.0));
    emit("int_bvtl_r_minus_1", QuantLib::BVTL(0, 0.4, -0.8, 1.0));

    // First-order helpers.
    emit("int_ff_a", QuantLib::ff(0.02, 0.05, 0.11, -0.4, 0.2));
    emit("int_ff_b", QuantLib::ff(0.005, 0.05, -0.11, 1.4, 0.2));
    emit("int_v_a", QuantLib::v(0.02, 0.05, 0.11, -0.06, 0.2));
    emit("int_v_b", QuantLib::v(0.005, 0.05, -0.11, 0.06, 0.2));
    emit("int_llold_a", QuantLib::llold(0.02, 0.05, 0.11, -0.8, 0.06, 0.2));
    emit("int_llold_b", QuantLib::llold(0.005, 0.05, -0.11, -1.2, -0.06, 0.2));

    // tvtl (trivariate normal, adaptive Kronrod) — the special-case ladder
    // plus the generic adaptive-integration branch.
    {
        Real limit[4], sr[4];
        // generic branch
        limit[1] = 0.3; limit[2] = -0.6; limit[3] = 0.9;
        sr[1] = 0.5; sr[2] = 0.35; sr[3] = 0.7;
        emit("int_tvtl_generic", QuantLib::tvtl(0, limit, sr, 1e-12));
        // negative correlations (the sorting branch)
        limit[1] = -0.4; limit[2] = 0.8; limit[3] = -0.2;
        sr[1] = -0.6; sr[2] = -0.3; sr[3] = 0.45;
        emit("int_tvtl_negative", QuantLib::tvtl(0, limit, sr, 1e-12));
        // all limits ~0 -> closed form  (1 + (asin r12+asin r13+asin r23)/PT)/8
        limit[1] = 0.0; limit[2] = 0.0; limit[3] = 0.0;
        sr[1] = 0.2; sr[2] = 0.4; sr[3] = 0.6;
        emit("int_tvtl_zero_limits", QuantLib::tvtl(0, limit, sr, 1e-12));
        // r23 -> 1
        limit[1] = 0.5; limit[2] = -0.5; limit[3] = 0.25;
        sr[1] = 0.1; sr[2] = 0.1; sr[3] = 1.0;
        emit("int_tvtl_r23_one", QuantLib::tvtl(0, limit, sr, 1e-12));
        // r23 -> -1
        limit[1] = 0.5; limit[2] = 0.6; limit[3] = 0.25;
        sr[1] = 0.1; sr[2] = 0.1; sr[3] = -1.0;
        emit("int_tvtl_r23_minus_one", QuantLib::tvtl(0, limit, sr, 1e-12));
        // r12 + r13 ~ 0 -> PHID(H1)*BVTL(0,H2,H3,R23)
        limit[1] = 0.5; limit[2] = -0.5; limit[3] = 0.25;
        sr[1] = 0.0; sr[2] = 0.0; sr[3] = 0.6;
        emit("int_tvtl_r12_r13_zero", QuantLib::tvtl(0, limit, sr, 1e-12));
    }

    // derivn3 for each idx.
    {
        Real limit[4], sr[4];
        limit[1] = 0.3; limit[2] = -0.6; limit[3] = 0.9;
        sr[1] = 0.5; sr[2] = 0.35; sr[3] = 0.7;
        emit("int_derivn3_1", QuantLib::derivn3(limit, sr, 1));
        emit("int_derivn3_2", QuantLib::derivn3(limit, sr, 2));
        emit("int_derivn3_3", QuantLib::derivn3(limit, sr, 3));
    }

    // Second-order helpers, at s < p < tt (the regime the double loop hits).
    {
        const Real s = 0.004, p = 0.012, tt = 0.05, gm = 0.2;
        const Real x = 0.11, xstar = -0.06;
        emit("int_dvv", QuantLib::dvv(s, p, tt, x, xstar, gm));
        emit("int_dvv_neg", QuantLib::dvv(s, p, tt, -x, xstar, gm));
        emit("int_dff", QuantLib::dff(s, p, tt, x, -1.0 + gm, gm));
        emit("int_dff_neg", QuantLib::dff(s, p, tt, -x, 1.0 + gm, gm));
        emit("int_dll", QuantLib::dll(s, p, tt, x, -1.0 + gm, -xstar, gm));
        emit("int_dll_neg", QuantLib::dll(s, p, tt, -x, -1.0 - gm, xstar, gm));
        emit("int_ddvv", QuantLib::ddvv(s, p, tt, x, xstar, gm));
        emit("int_ddvv_neg", QuantLib::ddvv(s, p, tt, -x, xstar, gm));
        emit("int_ddff", QuantLib::ddff(s, p, tt, x, -1.0 + gm, gm));
        emit("int_ddff_neg", QuantLib::ddff(s, p, tt, -x, 1.0 + gm, gm));
        emit("int_ddll", QuantLib::ddll(s, p, tt, x, -1.0 + gm, -xstar, gm));
        emit("int_ddll_neg", QuantLib::ddll(s, p, tt, -x, 1.0 + gm, -xstar, gm));
    }

    // BarrierUPD driven with analytic closures, so the quadrature itself is
    // pinned independently of any term structure. Flat r, q, sigma.
    {
        const Real rr = 0.03, qq = 0.02, sg = 0.11;
        auto integr = [rr](Real t1, Real t2) { return rr * (t2 - t1); };
        auto integalpha = [rr, qq](Real t1, Real t2) { return (rr - qq) * (t2 - t1); };
        auto integs = [sg](Real t1, Real t2) { return sg * sg * (t2 - t1); };
        auto alpha = [rr, qq](Real) { return rr - qq; };
        auto sigmaq = [sg](Real) { return sg * sg; };

        emit("int_barrierupd_o0", QuantLib::BarrierUPD(101.0, 100.0, 101.0, 0.0, 0.5,
                                                       0, 1, integr, integalpha, integs,
                                                       alpha, sigmaq));
        emit("int_barrierupd_o0_gm0", QuantLib::BarrierUPD(101.0, 100.0, 101.0, 0.0, 0.5,
                                                           0, 0, integr, integalpha, integs,
                                                           alpha, sigmaq));
        emit("int_barrierupd_o1", QuantLib::BarrierUPD(101.0, 100.0, 101.0, 0.0, 0.5,
                                                       1, 1, integr, integalpha, integs,
                                                       alpha, sigmaq));
        emit("int_barrierupd_o1_gm0", QuantLib::BarrierUPD(101.0, 100.0, 101.0, 0.0, 0.5,
                                                           1, 0, integr, integalpha, integs,
                                                           alpha, sigmaq));
        emit("int_barrierupd_o1_kbelow", QuantLib::BarrierUPD(95.0, 100.0, 110.0, 0.0, 0.5,
                                                              1, 1, integr, integalpha, integs,
                                                              alpha, sigmaq));
    }
}

}  // namespace

int main() {
    QuantLib::Settings::instance().evaluationDate() = TODAY;

    std::cout << "{\n";
    emit_int("today_serial", static_cast<long long>(TODAY.serialNumber()));
    emit_str("cpp_version", QL_VERSION);

    // The RNG policy MCDoubleBarrierEngine is templated on. The only member
    // with observable behaviour is `allowsErrorEstimate`: SingleVariate<RNG>
    // re-exports RNG's flag (mctraits.hpp:44/55) and
    // MCDoubleBarrierEngine::calculate publishes results_.errorEstimate only
    // when it is set (mcdoublebarrierengine.hpp:61-63).
    emit_int("rng_pseudo_allows_error_estimate",
             static_cast<long long>(QuantLib::PseudoRandom::allowsErrorEstimate));
    emit_int("rng_lowdiscrepancy_allows_error_estimate",
             static_cast<long long>(QuantLib::LowDiscrepancy::allowsErrorEstimate));
    emit_int("rng_singlevariate_pseudo_allows_error_estimate",
             static_cast<long long>(
                 QuantLib::SingleVariate<QuantLib::PseudoRandom>::allowsErrorEstimate));
    emit_int("rng_singlevariate_lowdiscrepancy_allows_error_estimate",
             static_cast<long long>(
                 QuantLib::SingleVariate<QuantLib::LowDiscrepancy>::allowsErrorEstimate));

    const DiscretizedCase dcases[7] = {
        {"ko_call", QuantLib::DoubleBarrier::KnockOut, 90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 12, false},
        {"ki_call", QuantLib::DoubleBarrier::KnockIn, 90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 12, false},
        {"kiko_put", QuantLib::DoubleBarrier::KIKO, 90.0, 110.0, 2.0,
         QuantLib::Option::Put, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 12, false},
        {"koki_put", QuantLib::DoubleBarrier::KOKI, 90.0, 110.0, 2.0,
         QuantLib::Option::Put, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 12, false},
        {"dk_ko_call", QuantLib::DoubleBarrier::KnockOut, 90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 12, true},
        {"dk_ki_call", QuantLib::DoubleBarrier::KnockIn, 90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 12, true},
        {"dk_ko_rebate", QuantLib::DoubleBarrier::KnockOut, 90.0, 110.0, 4.0,
         QuantLib::Option::Put, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 12, true},
    };
    for (const auto& c : dcases) run_discretized(c);

    // Converged prices at 300 steps — the configuration the C++ test suite uses.
    const DiscretizedCase conv[2] = {
        {"conv_std", QuantLib::DoubleBarrier::KnockOut, 90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 300, false},
        {"conv_dk", QuantLib::DoubleBarrier::KnockOut, 90.0, 110.0, 0.0,
         QuantLib::Option::Call, 100.0, 100.0, 0.02, 0.05, 0.25, 180, 300, true},
    };
    // Emit only the present value for the 300-step runs (the flattened value
    // vectors would be ~45k numbers each).
    for (const auto& c : conv) {
        const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
        auto process = make_bsm(c.spot, c.q, c.r, c.vol, dc);
        QuantLib::Date exDate = TODAY + c.days;
        auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(exDate);
        auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(c.type, c.strike);
        QuantLib::DoubleBarrierOption opt(c.barrierType, c.lo, c.hi, c.rebate, payoff, exercise);
        QuantLib::DoubleBarrierOption::arguments args;
        opt.setupArguments(&args);
        args.validate();

        QuantLib::Rate r = process->riskFreeRate()->zeroRate(
            exDate, dc, QuantLib::Continuous, QuantLib::NoFrequency);
        QuantLib::Rate q = process->dividendYield()->zeroRate(
            exDate, dc, QuantLib::Continuous, QuantLib::NoFrequency);
        QuantLib::Volatility v = process->blackVolatility()->blackVol(exDate, c.spot);
        QuantLib::Date refDate = process->riskFreeRate()->referenceDate();
        QuantLib::Handle<QuantLib::YieldTermStructure> flatRiskFree(
            QuantLib::ext::make_shared<QuantLib::FlatForward>(refDate, r, dc));
        QuantLib::Handle<QuantLib::YieldTermStructure> flatDividends(
            QuantLib::ext::make_shared<QuantLib::FlatForward>(refDate, q, dc));
        QuantLib::Handle<QuantLib::BlackVolTermStructure> flatVol(
            QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
                refDate, QuantLib::NullCalendar(), v, dc));
        Time maturity = dc.yearFraction(refDate, exDate);
        QuantLib::ext::shared_ptr<QuantLib::StochasticProcess1D> bs(
            new QuantLib::GeneralizedBlackScholesProcess(
                process->stateVariable(), flatDividends, flatRiskFree, flatVol));
        QuantLib::TimeGrid grid(maturity, c.steps);
        auto tree = QuantLib::ext::make_shared<QuantLib::CoxRossRubinstein>(
            bs, maturity, c.steps, payoff->strike());
        auto lattice = QuantLib::ext::make_shared<
            QuantLib::BlackScholesLattice<QuantLib::CoxRossRubinstein> >(
                tree, r, maturity, c.steps);
        const std::string p = std::string("discretized_") + c.tag;
        if (!c.dermanKani) {
            QuantLib::DiscretizedDoubleBarrierOption asset(args, *process, grid);
            asset.initialize(lattice, maturity);
            asset.rollback(0.0);
            emit(p + "_present_value", asset.presentValue());
        } else {
            QuantLib::DiscretizedDermanKaniDoubleBarrierOption asset(args, *process, grid);
            asset.initialize(lattice, maturity);
            asset.rollback(0.0);
            emit(p + "_present_value", asset.presentValue());
        }
    }

    run_check_barrier();
    run_discretized_vanilla();
    run_path_pricer();
    run_mc();
    run_quanto();
    run_suowang();
    run_perturbative();
    run_perturbative_internals();

    std::cout << "\n}\n";
    return 0;
}
