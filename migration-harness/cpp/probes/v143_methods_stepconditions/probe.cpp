// v1.43 methods / finite-differences step-conditions cluster probe.
//
// Emits references/v143/methods/stepconditions.json.
//
// Covers, one block per C++ class:
//
//   * ZeroCondition<Array>                 (ql/methods/finitedifferences/zerocondition.hpp)
//   * FdmSnapshotCondition                 (.../stepconditions/fdmsnapshotcondition.{hpp,cpp})
//   * FdmBermudanStepCondition             (.../stepconditions/fdmbermudanstepcondition.{hpp,cpp})
//   * FdmSimpleSwingCondition              (.../stepconditions/fdmsimpleswingcondition.{hpp,cpp})
//   * FdmSimpleStorageCondition            (.../stepconditions/fdmsimplestoragecondition.{hpp,cpp})
//   * FdmArithmeticAverageCondition        (.../stepconditions/fdmarithmeticaveragecondition.{hpp,cpp})
//   * FdmStepConditionComposite::joinConditions
//                                          (.../stepconditions/fdmstepconditioncomposite.{hpp,cpp})
//
// @ v1.43 (submodule commit 6b57206e0).
//
// Design notes
// ------------
// Every `applyTo` case emits BOTH the seed array and the resulting array, so
// the Python test never has to reproduce the seed: it loads the exact IEEE-754
// doubles the C++ run used.  That removes seed-construction as a source of
// spurious mismatch and isolates the condition arithmetic itself.
//
// The FdmInnerValueCalculator used throughout is FdmLogInnerValue, whose
// innerValue(iter, t) is simply payoff(exp(mesher->location(iter, direction))).
// The per-node innerValue arrays are emitted too (`*_inner`), so the Python
// test can first prove its own calculator stub reproduces C++ before comparing
// the condition output.
//
// FdmSimpleStorageCondition and FdmArithmeticAverageCondition both delegate to
// an interpolator (BilinearInterpolation / MonotonicCubicNaturalSpline).  The
// fixtures are deliberately chosen so those delegations are exercised off-grid
// AND outside the interpolation range:
//   - storage: changeRate 1.5 > the y grid step 1.0, so the intermediate-grid
//     scan loop runs and the bilinear interpolant is queried between nodes;
//     a second run with changeRate 0.4 pins the "loop body never runs" branch.
//   - arithmetic average: the equity grid spans [50, 150] while the average
//     grid spans [60, 140], so the convex combination
//     w0*a_[j] + w1*x_[i] lands BELOW a_.front() and ABOVE a_.back() for the
//     extreme equity nodes -- i.e. the `allowExtrapolation = true` cubic
//     continuation path is pinned, not just the in-range path.
//
// All floating-point output is std::setprecision(17) -- round-trip exact for
// IEEE-754 doubles.

#include <ql/instruments/payoffs.hpp>
#include <ql/math/array.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmarithmeticaveragecondition.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmbermudanstepcondition.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmsimplestoragecondition.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmsimpleswingcondition.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmsnapshotcondition.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/zerocondition.hpp>
#include <ql/time/date.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using namespace QuantLib;

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

void emit_int(const std::string& name, long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0U) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const std::string& name, const std::vector<long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0U) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

std::vector<Real> to_vector(const Array& a) {
    return {a.begin(), a.end()};
}

// Deterministic, jagged seed: reproducible in any language, but emitted
// anyway so the Python side reads the exact doubles.
Array makeSeed(Size n, Real scale, Real modulus, Real offset) {
    Array a(n);
    for (Size i = 0; i < n; ++i)
        a[i] = std::fmod(scale * Real(i), modulus) + offset;
    return a;
}

// Per-node innerValue of a calculator over the whole layout, in flat-index
// order.
std::vector<Real> innerValues(const ext::shared_ptr<FdmMesher>& mesher,
                              const ext::shared_ptr<FdmInnerValueCalculator>& calc,
                              Time t) {
    std::vector<Real> v(mesher->layout()->size());
    for (const auto& iter : *mesher->layout())
        v[iter.index()] = calc->innerValue(iter, t);
    return v;
}

// --------------------------------------------------------------------------
// Block 1 -- ZeroCondition<Array>
// --------------------------------------------------------------------------

void block_zero_condition() {
    // Signed zero is deliberately excluded: JSON round-trips -0 as 0 in
    // Python, so it cannot be pinned through this channel.  Everything else
    // about std::max(a[i], 0.0) is covered.
    Array seed(8);
    seed[0] = -2.5;
    seed[1] = -1e-17;
    seed[2] = 0.0;
    seed[3] = 0.5;
    seed[4] = 3.25;
    seed[5] = -100.0;
    seed[6] = 7.5;
    seed[7] = 1e-300;
    emit_arr("zero_seed", to_vector(seed));

    ZeroCondition<Array> zc;

    Array a1 = seed;
    zc.applyTo(a1, 0.42);
    emit_arr("zero_out_t042", to_vector(a1));

    // t is ignored by ZeroCondition: same output at a different time.
    Array a2 = seed;
    zc.applyTo(a2, 0.0);
    emit_arr("zero_out_t000", to_vector(a2));

    // Idempotence: a second pass changes nothing.
    zc.applyTo(a1, 0.42);
    emit_arr("zero_out_twice", to_vector(a1));
}

// --------------------------------------------------------------------------
// Block 2 -- FdmSnapshotCondition
// --------------------------------------------------------------------------

void block_snapshot_condition() {
    const Time snapT = 1.5;
    FdmSnapshotCondition snap(snapT);

    emit("snap_time", snap.getTime());
    emit_int("snap_values_size_initial", static_cast<long>(snap.getValues().size()));

    Array a(4);
    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 4.0;
    emit_arr("snap_seed", to_vector(a));

    // Wrong time: no snapshot taken, array untouched.
    snap.applyTo(a, 0.5);
    emit_int("snap_values_size_after_wrong_t", static_cast<long>(snap.getValues().size()));
    emit_arr("snap_a_after_wrong_t", to_vector(a));

    // Exact time: snapshot taken, array still untouched.
    snap.applyTo(a, snapT);
    emit_int("snap_values_size_at_t", static_cast<long>(snap.getValues().size()));
    emit_arr("snap_values_at_t", to_vector(snap.getValues()));
    emit_arr("snap_a_at_t", to_vector(a));

    // The snapshot is a COPY: mutating `a` afterwards must not disturb it,
    // and a later applyTo at a non-snapshot time must not overwrite it.
    a[0] = 99.0;
    a[3] = -7.0;
    snap.applyTo(a, 2.5);
    emit_arr("snap_values_after_mutation", to_vector(snap.getValues()));

    // Re-arming: applying again exactly at t replaces the stored values.
    snap.applyTo(a, snapT);
    emit_arr("snap_values_rearmed", to_vector(snap.getValues()));
}

// --------------------------------------------------------------------------
// Block 3 -- FdmBermudanStepCondition
// --------------------------------------------------------------------------

void block_bermudan_condition() {
    // 2-D mesh so the flat-index iteration order is pinned as well.
    auto equity = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 5);
    auto second = ext::make_shared<Uniform1dMesher>(0.0, 2.0, 3);
    auto mesher = ext::make_shared<FdmMesherComposite>(equity, second);

    const Size n = mesher->layout()->size();
    emit_int("berm_size", static_cast<long>(n));
    emit_iarr("berm_dim", {static_cast<long>(mesher->layout()->dim()[0]),
                           static_cast<long>(mesher->layout()->dim()[1])});
    emit_arr("berm_locations_0", to_vector(mesher->locations(0)));
    emit_arr("berm_locations_1", to_vector(mesher->locations(1)));

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0);
    auto calculator = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    const Date refDate(15, June, 2025);
    std::vector<Date> exerciseDates = {Date(15, December, 2025), Date(15, June, 2026),
                                       Date(15, December, 2026)};
    const DayCounter dc = Actual365Fixed();

    FdmBermudanStepCondition cond(exerciseDates, refDate, dc, mesher, calculator);

    const std::vector<Time>& ts = cond.exerciseTimes();
    emit_arr("berm_exercise_times", ts);
    emit_iarr("berm_exercise_day_counts",
              {static_cast<long>(dc.dayCount(refDate, exerciseDates[0])),
               static_cast<long>(dc.dayCount(refDate, exerciseDates[1])),
               static_cast<long>(dc.dayCount(refDate, exerciseDates[2]))});

    emit_arr("berm_inner", innerValues(mesher, calculator, ts[0]));

    const Array seed = makeSeed(n, 4.3, 27.0, -3.0);
    emit_arr("berm_seed", to_vector(seed));

    // Not an exercise time -> untouched.
    Array a0 = seed;
    cond.applyTo(a0, 0.3);
    emit_arr("berm_out_no_exercise", to_vector(a0));

    // Exactly at each exercise time -> max(a, innerValue) node by node.
    Array a1 = seed;
    cond.applyTo(a1, ts[0]);
    emit_arr("berm_out_ex0", to_vector(a1));

    Array a2 = seed;
    cond.applyTo(a2, ts[1]);
    emit_arr("berm_out_ex1", to_vector(a2));

    Array a3 = seed;
    cond.applyTo(a3, ts[2]);
    emit_arr("berm_out_ex2", to_vector(a3));

    // Idempotence at an exercise time.
    cond.applyTo(a1, ts[0]);
    emit_arr("berm_out_ex0_twice", to_vector(a1));

    // A time infinitesimally off an exercise time is NOT an exercise time:
    // std::find uses exact equality.
    Array a4 = seed;
    cond.applyTo(a4, std::nextafter(ts[0], 1.0));
    emit_arr("berm_out_nextafter_ex0", to_vector(a4));
}

// --------------------------------------------------------------------------
// Block 4 -- FdmSimpleSwingCondition
// --------------------------------------------------------------------------

void block_swing_condition() {
    // dim = (5, 4): direction 0 is log-spot, direction 1 counts exercises used.
    auto equity = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 5);
    auto swing = ext::make_shared<Uniform1dMesher>(0.0, 3.0, 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(equity, swing);

    const Size n = mesher->layout()->size();
    emit_int("swing_size", static_cast<long>(n));
    emit_iarr("swing_dim", {static_cast<long>(mesher->layout()->dim()[0]),
                            static_cast<long>(mesher->layout()->dim()[1])});
    emit_arr("swing_locations_0", to_vector(mesher->locations(0)));
    emit_arr("swing_locations_1", to_vector(mesher->locations(1)));

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    auto calculator = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);
    emit_arr("swing_inner", innerValues(mesher, calculator, 0.5));

    const std::vector<Time> exerciseTimes = {0.25, 0.5, 0.75};
    emit_arr("swing_exercise_times", exerciseTimes);

    const Array seed = makeSeed(n, 7.1, 23.0, 1.0);
    emit_arr("swing_seed", to_vector(seed));

    // minExercises = 0, swingDirection = 1.
    {
        FdmSimpleSwingCondition cond(exerciseTimes, mesher, calculator, 1, 0);

        Array a = seed;
        cond.applyTo(a, 0.1);  // not an exercise time
        emit_arr("swing_min0_no_exercise", to_vector(a));

        Array b = seed;
        cond.applyTo(b, exerciseTimes[0]);  // d = 3
        emit_arr("swing_min0_t025", to_vector(b));

        Array c = seed;
        cond.applyTo(c, exerciseTimes[1]);  // d = 2
        emit_arr("swing_min0_t050", to_vector(c));

        Array d = seed;
        cond.applyTo(d, exerciseTimes[2]);  // d = 1
        emit_arr("swing_min0_t075", to_vector(d));
    }

    // minExercises = 2: the `exercisesUsed + d <= minExercises_` branch fires.
    {
        FdmSimpleSwingCondition cond(exerciseTimes, mesher, calculator, 1, 2);

        Array a = seed;
        cond.applyTo(a, exerciseTimes[1]);  // d = 2 -> forces coor[1] == 0
        emit_arr("swing_min2_t050", to_vector(a));

        Array b = seed;
        cond.applyTo(b, exerciseTimes[2]);  // d = 1 -> forces coor[1] <= 1
        emit_arr("swing_min2_t075", to_vector(b));
    }

    // minExercises = 3 at the first exercise time: d = 3 -> forces coor[1] == 0.
    {
        FdmSimpleSwingCondition cond(exerciseTimes, mesher, calculator, 1, 3);
        Array a = seed;
        cond.applyTo(a, exerciseTimes[0]);
        emit_arr("swing_min3_t025", to_vector(a));
    }

    // swingDirection = 0: pins that the direction is a real parameter and the
    // neighbourhood()/maxExerciseValue arithmetic follows it.
    {
        FdmSimpleSwingCondition cond(exerciseTimes, mesher, calculator, 0, 0);
        Array a = seed;
        cond.applyTo(a, exerciseTimes[1]);
        emit_arr("swing_dir0_t050", to_vector(a));
    }

    // Default minExercises (omitted argument) == 0.
    {
        FdmSimpleSwingCondition cond(exerciseTimes, mesher, calculator, 1);
        Array a = seed;
        cond.applyTo(a, exerciseTimes[1]);
        emit_arr("swing_default_min_t050", to_vector(a));
    }
}

// --------------------------------------------------------------------------
// Block 5 -- FdmSimpleStorageCondition
// --------------------------------------------------------------------------

void block_storage_condition() {
    // direction 0: log-price, spot in [2, 6].  direction 1: storage volume
    // 0..4 in unit steps, so changeRate 1.5 spans more than one grid step.
    auto price = ext::make_shared<Uniform1dMesher>(std::log(2.0), std::log(6.0), 5);
    auto volume = ext::make_shared<Uniform1dMesher>(0.0, 4.0, 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(price, volume);

    const Size n = mesher->layout()->size();
    emit_int("storage_size", static_cast<long>(n));
    emit_iarr("storage_dim", {static_cast<long>(mesher->layout()->dim()[0]),
                              static_cast<long>(mesher->layout()->dim()[1])});

    // A zero-strike call payoff turns FdmLogInnerValue into the spot price
    // itself: innerValue(iter) = exp(location(iter, 0)).
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0);
    auto calculator = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);
    emit_arr("storage_inner", innerValues(mesher, calculator, 0.25));

    // x_ / y_ as the condition's constructor builds them.
    std::vector<Real> xs;
    std::vector<Real> ys;
    for (const auto& iter : *mesher->layout()) {
        if (iter.coordinates()[1] == 0U) xs.push_back(mesher->location(iter, 0));
        if (iter.coordinates()[0] == 0U) ys.push_back(mesher->location(iter, 1));
    }
    emit_arr("storage_x", xs);
    emit_arr("storage_y", ys);

    const std::vector<Time> exerciseTimes = {0.25, 0.5};
    emit_arr("storage_exercise_times", exerciseTimes);

    const Array seed = makeSeed(n, 3.3, 17.0, 0.5);
    emit_arr("storage_seed", to_vector(seed));

    // changeRate 1.5 > grid step 1.0: the intermediate-grid-point scan runs.
    {
        FdmSimpleStorageCondition cond(exerciseTimes, mesher, calculator, 1.5);

        Array a = seed;
        cond.applyTo(a, 0.1);  // not an exercise time
        emit_arr("storage_rate15_no_exercise", to_vector(a));

        Array b = seed;
        cond.applyTo(b, exerciseTimes[0]);
        emit_arr("storage_rate15_t025", to_vector(b));

        Array c = seed;
        cond.applyTo(c, exerciseTimes[1]);
        emit_arr("storage_rate15_t050", to_vector(c));
    }

    // changeRate 0.4 < grid step 1.0: the scan loop body never executes, and
    // every interpolation query is strictly between grid nodes.
    {
        FdmSimpleStorageCondition cond(exerciseTimes, mesher, calculator, 0.4);
        Array a = seed;
        cond.applyTo(a, exerciseTimes[0]);
        emit_arr("storage_rate04_t025", to_vector(a));
    }

    // changeRate 2.0 == exactly two grid steps: upper_bound / strict-less
    // boundary handling at grid nodes.
    {
        FdmSimpleStorageCondition cond(exerciseTimes, mesher, calculator, 2.0);
        Array a = seed;
        cond.applyTo(a, exerciseTimes[0]);
        emit_arr("storage_rate20_t025", to_vector(a));
    }

    // changeRate 0.0: maxWithDraw == maxInject == 0, so the value can only
    // stay put -- a degenerate but sharp regression case.
    {
        FdmSimpleStorageCondition cond(exerciseTimes, mesher, calculator, 0.0);
        Array a = seed;
        cond.applyTo(a, exerciseTimes[0]);
        emit_arr("storage_rate00_t025", to_vector(a));
    }
}

// --------------------------------------------------------------------------
// Block 6 -- FdmArithmeticAverageCondition
// --------------------------------------------------------------------------

void block_arithmetic_average_condition() {
    // Non-square dim (5, 6) so an i/j or spacing transposition cannot hide.
    // Equity grid spans [50, 150]; average grid spans [60, 140]: the convex
    // combination therefore leaves [60, 140] at both ends, exercising the
    // allowExtrapolation = true path of MonotonicCubicNaturalSpline.
    auto d0 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 5);
    auto d1 = ext::make_shared<Uniform1dMesher>(std::log(60.0), std::log(140.0), 6);
    auto mesher = ext::make_shared<FdmMesherComposite>(d0, d1);

    const Size n = mesher->layout()->size();
    emit_int("avg_size", static_cast<long>(n));
    emit_iarr("avg_dim", {static_cast<long>(mesher->layout()->dim()[0]),
                          static_cast<long>(mesher->layout()->dim()[1])});
    emit_arr("avg_locations_0", to_vector(mesher->locations(0)));
    emit_arr("avg_locations_1", to_vector(mesher->locations(1)));

    // The x_ / a_ arrays the constructor derives (equityDirection = 0).
    {
        std::vector<Real> xs(mesher->layout()->dim()[0]);
        const Size xSpacing = mesher->layout()->spacing()[0];
        Array tmp = mesher->locations(0);
        for (Size i = 0; i < xs.size(); ++i) xs[i] = std::exp(tmp[i * xSpacing]);
        emit_arr("avg_dir0_x", xs);

        std::vector<Real> as(mesher->layout()->dim()[1]);
        const Size aSpacing = mesher->layout()->spacing()[1];
        tmp = mesher->locations(1);
        for (Size i = 0; i < as.size(); ++i) as[i] = std::exp(tmp[i * aSpacing]);
        emit_arr("avg_dir0_a", as);
    }

    const Array seed = makeSeed(n, 2.9, 19.0, 0.25);
    emit_arr("avg_seed", to_vector(seed));

    const std::vector<Time> averageTimes = {0.25, 0.5, 0.75};
    emit_arr("avg_average_times", averageTimes);

    // equityDirection = 0, pastFixings = 0.
    {
        FdmArithmeticAverageCondition cond(averageTimes, 0.0, 0, mesher, 0);

        Array a = seed;
        cond.applyTo(a, 0.1);  // not an average time
        emit_arr("avg_dir0_pf0_no_fixing", to_vector(a));

        Array b = seed;
        cond.applyTo(b, averageTimes[0]);  // iT = 1, nTimes = 1 -> w0 = 0
        emit_arr("avg_dir0_pf0_t025", to_vector(b));

        Array c = seed;
        cond.applyTo(c, averageTimes[1]);  // iT = 2, nTimes = 1 -> w0 = 1/2
        emit_arr("avg_dir0_pf0_t050", to_vector(c));

        Array d = seed;
        cond.applyTo(d, averageTimes[2]);  // iT = 3, nTimes = 1 -> w0 = 2/3
        emit_arr("avg_dir0_pf0_t075", to_vector(d));
    }

    // equityDirection = 0, pastFixings = 2 -> iT shifts by 2.
    {
        FdmArithmeticAverageCondition cond(averageTimes, 0.0, 2, mesher, 0);
        Array a = seed;
        cond.applyTo(a, averageTimes[0]);  // iT = 3, nTimes = 1
        emit_arr("avg_dir0_pf2_t025", to_vector(a));

        Array b = seed;
        cond.applyTo(b, averageTimes[2]);  // iT = 5, nTimes = 1
        emit_arr("avg_dir0_pf2_t075", to_vector(b));
    }

    // equityDirection = 1: x_ comes from direction 1, a_ from direction 0.
    {
        std::vector<Real> xs(mesher->layout()->dim()[1]);
        const Size xSpacing = mesher->layout()->spacing()[1];
        Array tmp = mesher->locations(1);
        for (Size i = 0; i < xs.size(); ++i) xs[i] = std::exp(tmp[i * xSpacing]);
        emit_arr("avg_dir1_x", xs);

        std::vector<Real> as(mesher->layout()->dim()[0]);
        const Size aSpacing = mesher->layout()->spacing()[0];
        tmp = mesher->locations(0);
        for (Size i = 0; i < as.size(); ++i) as[i] = std::exp(tmp[i * aSpacing]);
        emit_arr("avg_dir1_a", as);

        FdmArithmeticAverageCondition cond(averageTimes, 0.0, 0, mesher, 1);
        Array a = seed;
        cond.applyTo(a, averageTimes[1]);
        emit_arr("avg_dir1_pf0_t050", to_vector(a));
    }

    // Duplicate average times: nTimes = 2 at t = 0.5, and `iter` still points
    // at the FIRST match, so iT = 2 and w0 = (2-2)/2 = 0 -- the interpolant is
    // queried at x_[i] alone, independent of j.
    {
        const std::vector<Time> dupTimes = {0.25, 0.5, 0.5, 0.75};
        emit_arr("avg_dup_times", dupTimes);
        FdmArithmeticAverageCondition cond(dupTimes, 0.0, 0, mesher, 0);
        Array a = seed;
        cond.applyTo(a, 0.5);
        emit_arr("avg_dir0_dup_t050", to_vector(a));
    }
}

// --------------------------------------------------------------------------
// Block 7 -- FdmStepConditionComposite::joinConditions
//
// joinConditions is the only consumer of FdmSnapshotCondition in the library,
// so it is pinned here rather than in a composite-specific probe. The second
// case (snapshot time == exercise time) is the sharp one: it proves the
// snapshot runs AFTER the wrapped composite and therefore records the
// post-condition values.
// --------------------------------------------------------------------------

void block_join_conditions() {
    auto equity = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 5);
    auto second = ext::make_shared<Uniform1dMesher>(0.0, 2.0, 3);
    auto mesher = ext::make_shared<FdmMesherComposite>(equity, second);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0);
    auto calculator = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    const Date refDate(15, June, 2025);
    std::vector<Date> exerciseDates = {Date(15, December, 2025), Date(15, June, 2026),
                                       Date(15, December, 2026)};
    auto bermudan = ext::make_shared<FdmBermudanStepCondition>(exerciseDates, refDate,
                                                               Actual365Fixed(), mesher, calculator);

    std::list<std::vector<Time> > stoppingTimes;
    stoppingTimes.push_back(bermudan->exerciseTimes());
    FdmStepConditionComposite::Conditions conditions;
    conditions.push_back(bermudan);
    auto c2 = ext::make_shared<FdmStepConditionComposite>(stoppingTimes, conditions);
    emit_arr("join_c2_stopping_times", c2->stoppingTimes());
    emit_int("join_c2_num_conditions", static_cast<long>(c2->conditions().size()));

    const Array seed = makeSeed(mesher->layout()->size(), 4.3, 27.0, -3.0);
    emit_arr("join_seed", to_vector(seed));

    // Case 1: snapshot at a time that is NOT an exercise time.
    auto c1 = ext::make_shared<FdmSnapshotCondition>(0.75);
    auto joined = FdmStepConditionComposite::joinConditions(c1, c2);
    emit_arr("join_stopping_times", joined->stoppingTimes());
    emit_int("join_num_conditions", static_cast<long>(joined->conditions().size()));

    Array a = seed;
    joined->applyTo(a, 0.75);
    emit_arr("join_out_at_snapshot", to_vector(a));
    emit_arr("join_snapshot_values", to_vector(c1->getValues()));

    // Case 2: snapshot exactly ON the first exercise time -- the recorded
    // values must be the exercised ones, not the seed.
    const Time ex0 = bermudan->exerciseTimes()[0];
    auto c1b = ext::make_shared<FdmSnapshotCondition>(ex0);
    auto joined2 = FdmStepConditionComposite::joinConditions(c1b, c2);
    emit_arr("join2_stopping_times", joined2->stoppingTimes());

    Array b = seed;
    joined2->applyTo(b, ex0);
    emit_arr("join2_out_at_exercise", to_vector(b));
    emit_arr("join2_snapshot_values", to_vector(c1b->getValues()));
}

}  // namespace

int main() {
    std::cout << "{\n";
    block_zero_condition();
    block_snapshot_condition();
    block_bermudan_condition();
    block_swing_condition();
    block_storage_condition();
    block_arithmetic_average_condition();
    block_join_conditions();
    std::cout << "\n}" << std::endl;
    return 0;
}
