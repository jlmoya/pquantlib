// migration-harness/cpp/probes/v143_ts_traits/probe.cpp
//
// Pins ql/termstructures/yield/bootstraptraits.hpp (v1.43) — the four yield
// bootstrap traits (Discount, ZeroYield, ForwardRate, SimpleZeroYield) plus
// detail::SpreadTraits<Discount> from spreadbootstraptraits.hpp.
//
// Why the trait members need a probe at all: every one of guess /
// minValueAfter / maxValueAfter takes the CURVE, not a bare data vector, so
// each of them reads c->times(), c->data(), c->dates(), c->dayCounter() and,
// for the rate traits, calls back into c->zeroRate() / c->forwardRate().  A
// port that passes only "data" has to invent substitutes for the time deltas
// and for the extrapolation, and every one of those substitutes is a silent
// divergence.  The values below are the arithmetic C++ actually performs.
//
// SAFETY: the traits index c->times()[i] and c->data()[i-1].  They are
// therefore only well-defined once the curve's arrays are sized.  Every trait
// call below is made against a FULLY CONSTRUCTED interpolated curve (never a
// mid-bootstrap PiecewiseYieldCurve), and every i satisfies 1 <= i <= n-1
// where n = number of nodes.  Nothing here reads past the end.
//
// Emits one JSON object on stdout; generate-references.sh redirects it to
// references/v143/ts/traits.json.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/interpolations/loginterpolation.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/bootstraptraits.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/forwardcurve.hpp>
#include <ql/termstructures/yield/interpolatedsimplezerocurve.hpp>
#include <ql/termstructures/yield/piecewisespreadyieldcurve.hpp>
#include <ql/termstructures/yield/piecewiseyieldcurve.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/termstructures/yield/spreadbootstraptraits.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// Evaluation date pinned so the relative-date deposit helpers below are
// reproducible.  The Python test pins the same date and restores the previous
// value in teardown (probe.cpp, kRef).
const Date kRef(15, January, 2024);

std::vector<Date> nodeDates() {
    return {kRef,
            kRef + 6 * Months,   // t ~ 0.5   -> SimpleZeroYield floor does NOT bind
            kRef + 1 * Years,    // t ~ 1.0
            kRef + 2 * Years,    // t ~ 2.0   -> SimpleZeroYield floor BINDS
            kRef + 5 * Years,    // t ~ 5.0   -> floor binds
            kRef + 10 * Years};  // t ~ 10.0  -> floor binds
}

// Deliberately not a single exponential: a flat curve is reproduced by every
// interpolator and by every guess, so it could not tell the branches apart.
std::vector<DiscountFactor> nodeDiscounts() {
    return {1.0, 0.985, 0.968, 0.930, 0.815, 0.640};
}

// Rate data with a NEGATIVE entry so the `r<0 ? r*2 : r/2` branch of
// min/maxValueAfter is exercised too (that sign test is the classic port bug).
std::vector<Rate> nodeRates() {
    return {0.030, 0.031, -0.005, 0.035, 0.0405, 0.0446};
}

// Pillars probed.  i == 1 hits the "first pillar" branch of guess; i > 1 hits
// the extrapolation branch.  i == 5 is the last valid index for a 6-node curve.
const Size kPillars[] = {1, 2, 3, 5};

// Fixed arguments for the transform round-trip.  transformDirect takes an
// unconstrained real, transformInverse takes a curve value.
const Real kTransformDirectX = -0.35;
const Real kTransformInverseX = 0.87;
// SimpleZeroYield's transforms shift by -1/t + 1e-8, so a rate-sized argument
// is the meaningful one for the inverse.
const Real kSimpleTransformInverseX = 0.05;

bool firstItem = true;

void key(const std::string& k) {
    if (!firstItem)
        std::cout << ",\n";
    firstItem = false;
    std::cout << "  \"" << k << "\": ";
}

void emit(const std::string& k, Real v) {
    key(k);
    std::cout << std::setprecision(17) << v;
}

void emit(const std::string& k, const Date& d) {
    key(k);
    std::cout << d.serialNumber();
}

void emitVec(const std::string& k, const std::vector<Real>& v) {
    key(k);
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emitDateVec(const std::string& k, const std::vector<Date>& v) {
    key(k);
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << v[i].serialNumber();
    }
    std::cout << "]";
}

std::string pillarKey(const std::string& prefix, Size i, const std::string& suffix) {
    return prefix + "_i" + std::to_string(i) + "_" + suffix;
}

// Emit guess / minValueAfter / maxValueAfter in BOTH validData branches, plus
// transformDirect / transformInverse, for one trait against one curve.
template <class Traits, class Curve>
void emitTrait(const std::string& prefix, const Curve& curve, Real inverseX) {
    const Curve* c = &curve;

    emit(prefix + "_initial_value", Traits::initialValue(c));
    emit(prefix + "_initial_date", Traits::initialDate(c));

    std::vector<Real> times(c->times().begin(), c->times().end());
    std::vector<Real> data(c->data().begin(), c->data().end());
    emitVec(prefix + "_times", times);
    emitVec(prefix + "_data", data);
    emitDateVec(prefix + "_dates", c->dates());

    for (Size i : kPillars) {
        // firstAliveHelper is 0 for all of these: the standard QL traits
        // ignore the parameter entirely (bootstraptraits.hpp passes it as an
        // unnamed Size), but it is part of the signature.
        emit(pillarKey(prefix, i, "guess_valid"), Traits::guess(i, c, true, 0));
        emit(pillarKey(prefix, i, "guess_invalid"), Traits::guess(i, c, false, 0));
        emit(pillarKey(prefix, i, "min_valid"), Traits::minValueAfter(i, c, true, 0));
        emit(pillarKey(prefix, i, "min_invalid"), Traits::minValueAfter(i, c, false, 0));
        emit(pillarKey(prefix, i, "max_valid"), Traits::maxValueAfter(i, c, true, 0));
        emit(pillarKey(prefix, i, "max_invalid"), Traits::maxValueAfter(i, c, false, 0));
        emit(pillarKey(prefix, i, "transform_direct"),
             Traits::transformDirect(kTransformDirectX, i, c));
        emit(pillarKey(prefix, i, "transform_inverse"),
             Traits::transformInverse(inverseX, i, c));
    }

    // updateGuess: Discount writes only data[i]; the rate traits also mirror
    // pillar 1 into pillar 0.  Probed on a scratch copy of the data vector.
    for (Size i : {Size(1), Size(2)}) {
        std::vector<Real> scratch = data;
        Traits::updateGuess(scratch, 0.123456789, i);
        emitVec(pillarKey(prefix, i, "update_guess"), scratch);
    }

    emit(prefix + "_max_iterations", Real(Traits::maxIterations()));
}

std::vector<ext::shared_ptr<RateHelper>> depositHelpers() {
    // (tenor months, rate).  Reaches 5Y so the bootstrapped SimpleZeroYield
    // curve has pillars on both sides of the t = 1 point where the
    // -1/t + 1e-8 floor starts to bind.
    const std::pair<Integer, Rate> quotes[] = {
        {3, 0.0325}, {6, 0.0340}, {12, 0.0355}, {24, 0.0370}, {60, 0.0390}};
    std::vector<ext::shared_ptr<RateHelper>> helpers;
    for (const auto& q : quotes) {
        helpers.push_back(ext::make_shared<DepositRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(q.second)),
            Period(q.first, Months), 2, TARGET(), ModifiedFollowing, true, Actual360()));
    }
    return helpers;
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = kRef;

    std::cout << "{\n";

    emit("ref_date", kRef);
    emit("avg_rate", detail::avgRate);
    emit("max_rate", detail::maxRate);

    // ---- static trait members against fully-constructed curves ------------

    InterpolatedDiscountCurve<LogLinear> discCurve(nodeDates(), nodeDiscounts(),
                                                  Actual365Fixed(), TARGET());
    discCurve.enableExtrapolation();
    emitTrait<Discount>("discount", discCurve, kTransformInverseX);

    InterpolatedZeroCurve<Linear> zeroCurve(nodeDates(), nodeRates(), Actual365Fixed(),
                                            TARGET());
    zeroCurve.enableExtrapolation();
    emitTrait<ZeroYield>("zeroyield", zeroCurve, kTransformInverseX);

    InterpolatedForwardCurve<Linear> fwdCurve(nodeDates(), nodeRates(), Actual365Fixed(),
                                              TARGET());
    fwdCurve.enableExtrapolation();
    emitTrait<ForwardRate>("forwardrate", fwdCurve, kTransformInverseX);

    InterpolatedSimpleZeroCurve<Linear> simpleCurve(nodeDates(), nodeRates(),
                                                    Actual365Fixed(), TARGET());
    simpleCurve.enableExtrapolation();
    emitTrait<SimpleZeroYield>("simplezero", simpleCurve, kSimpleTransformInverseX);

    // SpreadTraits<Discount> inherits every member from Discount and only
    // swaps the curve type, so its arithmetic must equal "discount_*" above.
    emitTrait<detail::SpreadTraits<Discount>>("spreadtraits", discCurve,
                                              kTransformInverseX);

    // ---- bootstrapped curves ----------------------------------------------

    {
        PiecewiseYieldCurve<SimpleZeroYield, Linear> pw(kRef, depositHelpers(),
                                                        Actual360());
        pw.enableExtrapolation();
        std::vector<Real> times(pw.times().begin(), pw.times().end());
        std::vector<Real> data(pw.data().begin(), pw.data().end());
        emitVec("pw_simplezero_times", times);
        emitVec("pw_simplezero_data", data);
        emitDateVec("pw_simplezero_dates", pw.dates());
        emit("pw_simplezero_max_date", pw.maxDate());
        std::vector<Real> dfs;
        for (Real t : {0.25, 0.5, 1.0, 2.0, 4.0, 6.0})
            dfs.push_back(pw.discount(t, true));
        emitVec("pw_simplezero_discounts", dfs);
    }

    {
        Handle<YieldTermStructure> base(ext::make_shared<FlatForward>(
            kRef, Handle<Quote>(ext::make_shared<SimpleQuote>(0.030)), Actual365Fixed(),
            Continuous, Annual));
        PiecewiseSpreadYieldCurve<Discount, LogLinear> pw(base, depositHelpers());
        pw.enableExtrapolation();
        std::vector<Real> times(pw.times().begin(), pw.times().end());
        std::vector<Real> data(pw.data().begin(), pw.data().end());
        emitVec("pw_spread_times", times);
        emitVec("pw_spread_data", data);
        emitDateVec("pw_spread_dates", pw.dates());
        emit("pw_spread_max_date", pw.maxDate());
        std::vector<Real> dfs;
        std::vector<Real> baseDfs;
        for (Real t : {0.25, 0.5, 1.0, 2.0, 4.0, 6.0}) {
            dfs.push_back(pw.discount(t, true));
            baseDfs.push_back(base->discount(t, true));
        }
        emitVec("pw_spread_discounts", dfs);
        emitVec("pw_spread_base_discounts", baseDfs);
    }

    std::cout << "\n}" << std::endl;
    return 0;
}
