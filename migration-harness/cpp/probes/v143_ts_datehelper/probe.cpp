// migration-harness/cpp/probes/v143_ts_datehelper/probe.cpp
//
// Reference values from C++ QuantLib v1.43 for
//
//   Gaussian1dSwaptionVolatility::DateHelper
//   (ql/termstructures/volatility/swaption/gaussian1dswaptionvolatility.hpp:70-86)
//
// and for the NewtonSafe inversion that consumes it
// (gaussian1dswaptionvolatility.cpp:46-59):
//
//     DateHelper hlp(*this, optionTime);
//     NewtonSafe newton;
//     Date d(static_cast<Date::serial_type>(newton.solve(
//         hlp, 0.1,
//         365.25 * optionTime + static_cast<Real>(referenceDate().serialNumber()),
//         1.0)));
//     d = indexBase_->fixingCalendar().adjust(d);
//
// Emits ONE JSON object on stdout and nothing else.
//
// -----------------------------------------------------------------------------
// WHY THE FUNCTOR IS TRANSCRIBED
// -----------------------------------------------------------------------------
// `DateHelper` is declared in the PRIVATE section of
// `class Gaussian1dSwaptionVolatility` (the class body opens `private:` at
// gaussian1dswaptionvolatility.hpp:64 and DateHelper follows at :70), so no
// translation unit outside that class can name it.  The eight-line body is
// copied verbatim below with its source cited.  What the probe MEASURES rather
// than assumes is everything the functor composes with: QuantLib's own
// `TermStructure::timeFromReference` under four different day counters,
// QuantLib's own `NewtonSafe` (bracket search, Newton/bisection switching,
// accuracy handling), and the `static_cast<Date::serial_type>` truncation and
// calendar adjustment applied to the result.
//
// -----------------------------------------------------------------------------
// WHAT THIS PINS, AND WHY IT MATTERS
// -----------------------------------------------------------------------------
// The functor is
//
//     f(date) = h * (T(floor(date)+1) - t) + (1-h) * (T(floor(date)) - t)
//     h       = date - floor(date)
//
// i.e. a piecewise-linear interpolation of `timeFromReference` through the
// integer serials, shifted so its root is the fractional date whose year
// fraction is exactly `optionTime`.  Under Actual/365Fixed the exact root is
// `refSerial + 365 * optionTime`; the NewtonSafe GUESS is
// `refSerial + 365.25 * optionTime`, which is a different day for every
// optionTime >= 4.  A port that keeps the guess and skips the solve therefore
// lands on the wrong date, and the error grows with maturity.  The
// `guess_serial` / `solved_serial` pair below makes that difference explicit
// per case.
//
// -----------------------------------------------------------------------------
// DETERMINISM
// -----------------------------------------------------------------------------
// Settings::instance().evaluationDate() IS set (see EVAL_DATE below) because
// TermStructure::referenceDate() reads it.  Nothing is read before it is
// written.

#include <ql/version.hpp>

#include <ql/math/solvers1d/newtonsafe.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Verbatim transcription of Gaussian1dSwaptionVolatility::DateHelper,
// gaussian1dswaptionvolatility.hpp:70-86.  See the header comment for why.
// ---------------------------------------------------------------------------
class DateHelper {
  public:
    DateHelper(const TermStructure& ts, const Time t) : ts_(ts), t_(t) {}
    Real operator()(Real date) const {
        Date d1(static_cast<Date::serial_type>(date));
        Date d2(static_cast<Date::serial_type>(date) + 1);
        Real t1 = ts_.timeFromReference(d1) - t_;
        Real t2 = ts_.timeFromReference(d2) - t_;
        Real h = date - static_cast<Date::serial_type>(date);
        return h * t2 + (1.0 - h) * t1;
    }
    Real derivative(Real date) const {
        // use fwd difference to avoid dates before reference date
        return (operator()(date + 1E-6) - operator()(date)) * 1E6;
    }
    const TermStructure& ts_;
    const Time t_;
};

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

} // namespace

int main() {
    // EVAL_DATE: TermStructure::referenceDate() reads the global evaluation
    // date; every case below is relative to it.
    const Date EVAL_DATE(15, January, 2024);
    Settings::instance().evaluationDate() = EVAL_DATE;

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"quantlib_version\": \"" << QL_VERSION << "\",\n";
    out << "  \"eval_date\": \"" << iso(EVAL_DATE) << "\",\n";
    out << "  \"eval_date_serial\": " << EVAL_DATE.serialNumber() << ",\n";

    struct DcCase {
        const char* name;
        DayCounter dc;
    };
    const std::vector<DcCase> dcs = {
        {"Actual365Fixed", Actual365Fixed()},
        {"Actual360", Actual360()},
        {"ActualActualISDA", ActualActual(ActualActual::ISDA)},
        {"Thirty360BondBasis", Thirty360(Thirty360::BondBasis)},
    };

    // Option times chosen so that the 365.25-per-year guess and the true
    // Actual/365Fixed root fall on different days for most of them.
    const std::vector<Time> optionTimes = {0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 7.5, 10.0, 20.0};

    TARGET cal;

    out << "  \"date_helper\": [\n";
    bool first = true;
    for (const auto& dcc : dcs) {
        FlatForward ts(EVAL_DATE, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dcc.dc);
        for (Time t : optionTimes) {
            if (!first)
                out << ",\n";
            first = false;

            DateHelper hlp(ts, t);
            const Real guess = 365.25 * t + static_cast<Real>(EVAL_DATE.serialNumber());

            NewtonSafe newton;
            const Real solved = newton.solve(hlp, 0.1, guess, 1.0);
            const Date d(static_cast<Date::serial_type>(solved));
            const Date adjusted = cal.adjust(d);

            out << "    {\"day_counter\": \"" << dcc.name << "\", \"option_time\": " << num(t)
                << ", \"guess\": " << num(guess)
                << ", \"guess_serial\": " << static_cast<Date::serial_type>(guess)
                << ", \"solved\": " << num(solved)
                << ", \"solved_serial\": " << static_cast<Date::serial_type>(solved)
                << ", \"solved_date\": \"" << iso(d) << "\""
                << ", \"adjusted_serial\": " << adjusted.serialNumber()
                << ", \"adjusted_date\": \"" << iso(adjusted) << "\""
                << ", \"time_at_solved_date\": " << num(ts.timeFromReference(d))
                << ", \"time_at_guess_date\": "
                << num(ts.timeFromReference(Date(static_cast<Date::serial_type>(guess))))
                << ", \"f_at_solved\": " << num(hlp(solved)) << "}";
        }
    }
    out << "\n  ],\n";

    // Raw functor values and derivatives at fractional serials, so the
    // interpolation and the 1e-6 forward difference are pinned independently
    // of the solver.
    {
        FlatForward ts(EVAL_DATE, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)),
                       Actual365Fixed());
        const Real base = static_cast<Real>(EVAL_DATE.serialNumber());
        const Real offsets[] = {0.0, 0.5, 1.0, 1.25, 100.0, 100.75, 365.0, 3652.5};
        const Time t = 1.0;
        DateHelper hlp(ts, t);
        out << "  \"date_helper_raw\": [\n";
        for (Size i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
            if (i)
                out << ",\n";
            const Real x = base + offsets[i];
            out << "    {\"t\": " << num(t) << ", \"offset\": " << num(offsets[i])
                << ", \"x\": " << num(x) << ", \"value\": " << num(hlp(x))
                << ", \"derivative\": " << num(hlp.derivative(x)) << "}";
        }
        out << "\n  ]\n";
    }

    out << "}\n";
    std::cout << out.str();
    return 0;
}
