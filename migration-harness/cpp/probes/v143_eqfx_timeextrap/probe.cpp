// migration-harness/cpp/probes/v143_eqfx_timeextrap/probe.cpp
//
// Pins BlackVolTimeExtrapolation (blackvoltimeextrapolation.{hpp,cpp}) — the
// three policies a Black vol structure uses past its last quoted maturity —
// and BlackVarianceCurve's use of them.
//
// The three policies agree nowhere except by accident, so every one is
// probed both INSIDE the node range (where the caller normally would not
// invoke them, but where the arithmetic is still well defined) and PAST the
// last node, which is the only regime that matters in production:
//
//   FlatVolatility   var(t) = max(var(t_last), 0) / t_last * t
//                    — a ray through the origin, i.e. the LAST NODE'S
//                      VOLATILITY held flat, not the last node's variance.
//   UseInterpolator  var(t) = max(curve(t), 0)
//                    — whatever the interpolation does on its own, floored
//                      at zero. With Linear on (t, variance) this is the
//                      straight-line continuation of the LAST SEGMENT, which
//                      differs from LinearVariance only when the last two
//                      interpolation nodes are not the last two entries of
//                      the `times` vector handed to the extrapolator.
//   LinearVariance   straight line through the last two nodes' VARIANCES.
//
// A curve whose variance is convex in t separates all three: flat-vol lies
// below the linear-variance ray, which lies below/above the interpolator
// depending on convexity. The vols below (0.13, 0.16, 0.20, 0.25) give a
// strictly convex variance, so the three answers are ~10-40% apart at
// t = 3.5 and no rounding accident can make a port pass with the wrong one.
//
// Both C++ overloads are probed: the strike-dependent surface form (whose
// variance functor ignores the strike here, so the surface and curve
// answers must agree — a port that mixed up the two argument orders fails)
// and the ATM curve form.
//
// The LinearVariance branch's preconditions are pinned too: it raises when
// asked for a t that is NOT past the last node (linearExtrapolation's
// "t must be greater than times[1]"), which is the trap in using it as a
// general interpolator. Failing cases appear as {"raises": true}.
//
// Finally BlackVarianceCurve is built with each policy and queried through
// blackVariance / blackVol, so the wiring (curve delegates past t_last,
// interpolates at or before it) is pinned end-to-end.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/eqfx/timeextrap.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <ql/errors.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltimeextrapolation.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kRef(15, June, 2026);
const Actual365Fixed kDc;

const std::vector<Date> kDates = {Date(15, September, 2026), Date(15, December, 2026),
                                  Date(15, June, 2027), Date(15, June, 2028)};
const std::vector<Volatility> kVols = {0.13, 0.16, 0.20, 0.25};

const std::vector<std::pair<std::string, BlackVolTimeExtrapolation::Type>> kPolicies = {
    {"flat_volatility", BlackVolTimeExtrapolation::FlatVolatility},
    {"use_interpolator", BlackVolTimeExtrapolation::UseInterpolator},
    {"linear_variance", BlackVolTimeExtrapolation::LinearVariance}};

// Node times as BlackVarianceCurve builds them: a prepended 0 plus the four
// quoted maturities. Kept explicit so the standalone-static-call block uses
// exactly the vector the curve hands to the extrapolator.
std::vector<Time> nodeTimes() {
    std::vector<Time> t(kDates.size() + 1, 0.0);
    for (Size j = 1; j <= kDates.size(); ++j)
        t[j] = kDc.yearFraction(kRef, kDates[j - 1]);
    return t;
}

std::vector<Real> nodeVariances(const std::vector<Time>& times) {
    std::vector<Real> v(times.size(), 0.0);
    for (Size j = 1; j < times.size(); ++j)
        v[j] = times[j] * kVols[j - 1] * kVols[j - 1];
    return v;
}

// Piecewise-linear variance through the nodes, extended linearly on the last
// (resp. first) segment outside the range — i.e. exactly what
// Linear::interpolate(...)(t, true) does inside QuantLib.
Real linearVariance(const std::vector<Time>& times, const std::vector<Real>& vars, Time t) {
    Size i = 1;
    while (i + 1 < times.size() && t > times[i])
        ++i;
    const Time t0 = times[i - 1], t1 = times[i];
    return vars[i - 1] + (t - t0) * (vars[i] - vars[i - 1]) / (t1 - t0);
}

const std::vector<std::pair<std::string, Time>> kQueryTimes = {
    {"t0p5_inside", 0.5},        // between nodes 1 and 2
    {"t1p8_inside", 1.8},        // between nodes 3 and 4
    {"t2p0_last_node", 2.0},     // ~ the last node itself (t_last = 731/365)
    {"t2p5_past", 2.5},          // just past the last node
    {"t3p5_past", 3.5},          // well past
    {"t10p0_far_past", 10.0}};   // far past, where the policies fan out hardest

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const std::vector<Time> times = nodeTimes();
    const std::vector<Real> vars = nodeVariances(times);

    std::cout << "  \"nodes\": {\n    \"times\": [";
    for (Size i = 0; i < times.size(); ++i)
        std::cout << (i ? ", " : "") << times[i];
    std::cout << "],\n    \"variances\": [";
    for (Size i = 0; i < vars.size(); ++i)
        std::cout << (i ? ", " : "") << vars[i];
    std::cout << "]\n  },\n";

    // --- the two static overloads, called directly -----------------------
    const auto curveFn = [&](Time t) { return linearVariance(times, vars, t); };
    const auto surfaceFn = [&](Time t, Real k) {
        (void)k;  // strike-independent on purpose: the surface and curve
                  // overloads must then agree value for value.
        return linearVariance(times, vars, t);
    };

    std::cout << "  \"static_curve\": {\n";
    bool firstPolicy = true;
    for (const auto& pol : kPolicies) {
        if (!firstPolicy) std::cout << ",\n";
        firstPolicy = false;
        std::cout << "    \"" << pol.first << "\": {\n";
        bool first = true;
        for (const auto& q : kQueryTimes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "      \"" << q.first << "\": ";
            try {
                std::cout << BlackVolTimeExtrapolation::extrapolatedVariance(
                    pol.second, q.second, times, curveFn);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
        std::cout << "\n    }";
    }
    std::cout << "\n  },\n";

    std::cout << "  \"static_surface\": {\n";
    firstPolicy = true;
    for (const auto& pol : kPolicies) {
        if (!firstPolicy) std::cout << ",\n";
        firstPolicy = false;
        std::cout << "    \"" << pol.first << "\": {\n";
        bool first = true;
        for (const auto& q : kQueryTimes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "      \"" << q.first << "\": ";
            try {
                std::cout << BlackVolTimeExtrapolation::extrapolatedVariance(
                    pol.second, q.second, 123.0, times, surfaceFn);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
        std::cout << "\n    }";
    }
    std::cout << "\n  },\n";

    // With a piecewise-LINEAR variance functor, UseInterpolator and
    // LinearVariance coincide past the last node — continuing the last
    // segment IS the line through the last two nodes. A quadratic variance
    // functor separates them, so this block is what actually proves the two
    // branches are not swapped. Same nodes, different functor:
    //     var(t) = 0.02 * t^2 + 0.03 * t
    const auto quadCurveFn = [](Time t) { return 0.02 * t * t + 0.03 * t; };
    const auto quadSurfaceFn = [](Time t, Real k) {
        (void)k;
        return 0.02 * t * t + 0.03 * t;
    };

    std::cout << "  \"static_curve_quadratic\": {\n";
    firstPolicy = true;
    for (const auto& pol : kPolicies) {
        if (!firstPolicy) std::cout << ",\n";
        firstPolicy = false;
        std::cout << "    \"" << pol.first << "\": {\n";
        bool first = true;
        for (const auto& q : kQueryTimes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "      \"" << q.first << "\": ";
            try {
                std::cout << BlackVolTimeExtrapolation::extrapolatedVariance(
                    pol.second, q.second, times, quadCurveFn);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
        std::cout << "\n    }";
    }
    std::cout << "\n  },\n";

    std::cout << "  \"static_surface_quadratic\": {\n";
    firstPolicy = true;
    for (const auto& pol : kPolicies) {
        if (!firstPolicy) std::cout << ",\n";
        firstPolicy = false;
        std::cout << "    \"" << pol.first << "\": {\n";
        bool first = true;
        for (const auto& q : kQueryTimes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "      \"" << q.first << "\": ";
            try {
                std::cout << BlackVolTimeExtrapolation::extrapolatedVariance(
                    pol.second, q.second, 123.0, times, quadSurfaceFn);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
        std::cout << "\n    }";
    }
    std::cout << "\n  },\n";

    // --- end-to-end through BlackVarianceCurve ---------------------------
    std::cout << "  \"curve\": {\n";
    firstPolicy = true;
    for (const auto& pol : kPolicies) {
        if (!firstPolicy) std::cout << ",\n";
        firstPolicy = false;
        BlackVarianceCurve c(kRef, kDates, kVols, kDc, true, pol.second);
        std::cout << "    \"" << pol.first << "\": {\n"
                  << "      \"max_date\": " << c.maxDate().serialNumber() << ",\n"
                  << "      \"variance\": {\n";
        bool first = true;
        for (const auto& q : kQueryTimes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "        \"" << q.first << "\": ";
            try {
                std::cout << c.blackVariance(q.second, 1.0, true);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
        std::cout << "\n      },\n      \"vol\": {\n";
        first = true;
        for (const auto& q : kQueryTimes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "        \"" << q.first << "\": ";
            try {
                std::cout << c.blackVol(q.second, 1.0, true);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
        std::cout << "\n      }\n    }";
    }
    std::cout << "\n  }\n";

    std::cout << "}\n";
    return 0;
}
