// migration-harness/cpp/probes/v143_ts_helpersort/probe.cpp
//
// Reference values from C++ QuantLib v1.43 for
//
//   detail::BootstrapHelperSorter  (ql/termstructures/bootstraphelper.hpp:248-258)
//
// a `detail::` functor whose whole body is
//
//   return (h1->pillarDate() < h2->pillarDate());
//
// and which `IterativeBootstrap::initialize` applies as
//
//   std::sort(ts_->instruments_.begin(), ts_->instruments_.end(),
//             detail::BootstrapHelperSorter());        (iterativebootstrap.hpp:163)
//
// Emits ONE JSON object on stdout and nothing else.
//
// -----------------------------------------------------------------------------
// HOW THE COMPARATOR IS MEASURED
// -----------------------------------------------------------------------------
// The sorter has no standalone observable output — it is a comparator handed to
// std::sort inside the bootstrap. What IS observable is its effect: a curve
// built from helpers supplied in a SCRAMBLED order must equal, node for node,
// the curve built from the same helpers in pillar order.
//
// So the probe builds the same PiecewiseYieldCurve twice — once with the
// helpers ascending by pillar date, once with them scrambled — and emits both
// curves' pillar dates, times, data and discounts. If the sort were absent or
// mis-ordered the two would differ, because the bootstrap solves pillar i
// against an interpolation built over pillars 0..i-1.
//
// It also emits the pillar dates of the scrambled INPUT, so a port can check
// that its own input really was out of order (a helper set that happens to be
// ordered would make this whole file vacuous).
//
// -----------------------------------------------------------------------------
// DETERMINISM
// -----------------------------------------------------------------------------
// Settings::instance().evaluationDate() IS set (EVAL_DATE below) because the
// deposit helpers are RelativeDateBootstrapHelpers. Nothing is read before it
// is written.
//
// NOTE on std::sort stability: std::sort is NOT stable, so two helpers with the
// SAME pillar date could come out in either order. This probe therefore uses
// six strictly distinct pillar dates — the port must reproduce an ordering, not
// a tie-break that C++ does not itself define.

#include <ql/version.hpp>

#include <ql/indexes/ibor/euribor.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/piecewiseyieldcurve.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

std::string num(Real x) {
    std::ostringstream os;
    os << std::setprecision(17) << x;
    return os.str();
}

std::string iso(const Date& d) {
    std::ostringstream os;
    os << std::setw(4) << std::setfill('0') << int(d.year()) << "-" << std::setw(2)
       << std::setfill('0') << int(d.month()) << "-" << std::setw(2) << std::setfill('0')
       << int(d.dayOfMonth());
    return os.str();
}

// tenor in months -> quoted deposit rate
struct Dep {
    Integer months;
    Rate rate;
};

std::vector<ext::shared_ptr<RateHelper>> makeHelpers(const std::vector<Dep>& deps) {
    std::vector<ext::shared_ptr<RateHelper>> out;
    out.reserve(deps.size());
    for (const auto& d : deps) {
        out.push_back(ext::make_shared<DepositRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(d.rate)),
            Period(d.months, Months), 2, TARGET(), ModifiedFollowing, true, Actual360()));
    }
    return out;
}

void emitCurve(std::ostringstream& out,
               const char* name,
               const std::vector<Dep>& deps,
               const Date& refDate) {
    auto helpers = makeHelpers(deps);

    std::ostringstream inputPillars;
    inputPillars << "[";
    for (Size i = 0; i < helpers.size(); ++i) {
        if (i)
            inputPillars << ", ";
        inputPillars << helpers[i]->pillarDate().serialNumber();
    }
    inputPillars << "]";

    PiecewiseYieldCurve<Discount, LogLinear> curve(refDate, helpers, Actual365Fixed());
    curve.discount(0.5); // force the bootstrap

    out << "    \"" << name << "\": {\n";
    out << "      \"input_pillar_serials\": " << inputPillars.str() << ",\n";

    out << "      \"dates\": [";
    const auto& dates = curve.dates();
    for (Size i = 0; i < dates.size(); ++i) {
        if (i)
            out << ", ";
        out << dates[i].serialNumber();
    }
    out << "],\n";

    out << "      \"date_strings\": [";
    for (Size i = 0; i < dates.size(); ++i) {
        if (i)
            out << ", ";
        out << "\"" << iso(dates[i]) << "\"";
    }
    out << "],\n";

    out << "      \"times\": [";
    const auto& times = curve.times();
    for (Size i = 0; i < times.size(); ++i) {
        if (i)
            out << ", ";
        out << num(times[i]);
    }
    out << "],\n";

    out << "      \"data\": [";
    const auto& data = curve.data();
    for (Size i = 0; i < data.size(); ++i) {
        if (i)
            out << ", ";
        out << num(data[i]);
    }
    out << "],\n";

    out << "      \"discounts\": [";
    const Real ts[] = {0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5};
    for (Size i = 0; i < sizeof(ts) / sizeof(ts[0]); ++i) {
        if (i)
            out << ", ";
        out << num(curve.discount(ts[i], true));
    }
    out << "],\n";
    out << "      \"discount_times\": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]\n";
    out << "    }";
}

} // namespace

int main() {
    // EVAL_DATE: DepositRateHelper is relative-date, so its pillar depends on it.
    const Date EVAL_DATE(15, January, 2024);
    Settings::instance().evaluationDate() = EVAL_DATE;

    // Six strictly distinct pillars. Rates are monotone so a mis-ordered
    // bootstrap produces a visibly different curve rather than a near-miss.
    const std::vector<Dep> ascending = {
        {1, 0.020}, {2, 0.0225}, {3, 0.025}, {6, 0.030}, {9, 0.0325}, {12, 0.035},
    };
    // A permutation with no fixed point in the first five slots.
    const std::vector<Dep> scrambled = {
        {12, 0.035}, {3, 0.025}, {9, 0.0325}, {1, 0.020}, {6, 0.030}, {2, 0.0225},
    };

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"quantlib_version\": \"" << QL_VERSION << "\",\n";
    out << "  \"eval_date\": \"" << iso(EVAL_DATE) << "\",\n";
    out << "  \"eval_date_serial\": " << EVAL_DATE.serialNumber() << ",\n";
    out << "  \"deposit_tenor_months_ascending\": [1, 2, 3, 6, 9, 12],\n";
    out << "  \"deposit_rates_ascending\": [0.02, 0.0225, 0.025, 0.03, 0.0325, 0.035],\n";
    out << "  \"deposit_tenor_months_scrambled\": [12, 3, 9, 1, 6, 2],\n";
    out << "  \"deposit_rates_scrambled\": [0.035, 0.025, 0.0325, 0.02, 0.03, 0.0225],\n";
    out << "  \"curves\": {\n";
    emitCurve(out, "ascending", ascending, EVAL_DATE);
    out << ",\n";
    emitCurve(out, "scrambled", scrambled, EVAL_DATE);
    out << "\n  }\n";
    out << "}\n";
    std::cout << out.str();
    return 0;
}
