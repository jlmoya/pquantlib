// migration-harness/cpp/probes/v143_ts_discountcurve/probe.cpp
//
// Pins InterpolatedDiscountCurve's extrapolation, which is where a port can
// quietly disagree.
//
// discountcurve.hpp:
//     if (t <= times_.back()) return interpolation_(t, true);
//     Time tMax = times_.back();
//     DiscountFactor dMax = data_.back();
//     Rate instFwdMax = -interpolation_.derivative(tMax) / dMax;
//     return dMax * exp(-instFwdMax * (t - tMax));
//
// i.e. past the last node the curve switches to an explicit FLAT-FORWARD
// extension rather than continuing to interpolate. Under LogLinear the two
// agree exactly (log-linear in the discount factor IS flat forward), which is
// why a port that simply keeps interpolating passes every LogLinear test and
// fails silently for Cubic / LogCubic / Linear-on-discount-factors. Every
// interpolator below is therefore probed at the same three regimes: on a node,
// between nodes, and past the last node.
//
// zeroRate and forwardRate are pinned alongside discount because they are the
// quantities a curve user actually reads, and they amplify the difference:
// the forward rate past tMax is constant iff the flat-forward branch is taken.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/discountcurve.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/interpolations/loginterpolation.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kRef(17, January, 2024);

std::vector<Date> nodeDates() {
    return {kRef,
            kRef + 6 * Months,
            kRef + 1 * Years,
            kRef + 2 * Years,
            kRef + 5 * Years,
            kRef + 10 * Years};
}

// Deliberately NOT on a single exponential: a curve whose discount factors are
// exp(-r t) for constant r is reproduced by every interpolator, so it could not
// tell them apart.
std::vector<DiscountFactor> nodeDiscounts() {
    return {1.0, 0.985, 0.968, 0.930, 0.815, 0.640};
}

// Times, in Actual/365F from the reference date, at which every curve is read:
// t=0, on nodes, between nodes, and three points past the last node.
const double kTimes[] = {0.0,   0.25,  0.5041095890410959, 0.75,
                         1.0027397260273974, 1.5, 2.0054794520547947,
                         3.5,   5.0082191780821919, 7.5,
                         10.010958904109589, 12.0, 15.0, 25.0};

bool firstEmitted = false;

template <class Interpolator>
void emitCurve(const std::string& key, const Interpolator& interpolator) {
    InterpolatedDiscountCurve<Interpolator> curve(nodeDates(), nodeDiscounts(),
                                                  Actual365Fixed(), TARGET(), {}, {},
                                                  interpolator);
    curve.enableExtrapolation();

    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"max_time\": " << curve.maxTime() << ",\n"
              << "    \"discount\": [";
    bool first = true;
    for (double t : kTimes) {
        std::cout << (first ? "" : ", ") << curve.discount(t, true);
        first = false;
    }
    std::cout << "],\n    \"zero\": [";
    first = true;
    for (double t : kTimes) {
        // t == 0 has no zero rate off a discount curve without a limit; skip by
        // reporting the t -> 0 value the structure itself returns.
        std::cout << (first ? "" : ", ")
                  << curve.zeroRate(t, Continuous, NoFrequency, true).rate();
        first = false;
    }
    std::cout << "],\n    \"inst_forward\": [";
    first = true;
    for (double t : kTimes) {
        std::cout << (first ? "" : ", ")
                  << curve.forwardRate(t, t, Continuous, NoFrequency, true).rate();
        first = false;
    }
    std::cout << "]\n  }";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"times\": [";
    bool first = true;
    for (double t : kTimes) {
        std::cout << (first ? "" : ", ") << t;
        first = false;
    }
    std::cout << "]";
    firstEmitted = true; // the "times" entry already printed, so keep commas flowing
    emitCurve("log_linear", LogLinear());
    emitCurve("linear", Linear());
    emitCurve("cubic_natural", Cubic(CubicInterpolation::Spline, false,
                                     CubicInterpolation::SecondDerivative, 0.0,
                                     CubicInterpolation::SecondDerivative, 0.0));
    emitCurve("log_cubic_natural", LogCubic(CubicInterpolation::Spline, false,
                                            CubicInterpolation::SecondDerivative, 0.0,
                                            CubicInterpolation::SecondDerivative, 0.0));
    std::cout << "\n}\n";
    return 0;
}
