// migration-harness/cpp/probes/v143_eqfx_impliedvol/probe.cpp
//
// Pins ImpliedVolTermStructure — the Black vol surface re-anchored at a
// future date (impliedvoltermstructure.hpp, header-only).
//
// The whole content of the class is the date shift:
//
//     timeShift = dayCounter().yearFraction(original->referenceDate(),
//                                           referenceDate())
//     variance(t, K) = original->blackForwardVariance(timeShift,
//                                                     timeShift + t, K, true)
//
// so the cases below are chosen to make a port that gets the shift wrong
// disagree loudly:
//
//   * shift == 0 (implied date == original reference date). The forward
//     variance then starts at t1 = 0, where BlackVarianceSurface's t==0
//     special case returns exactly 0.0, and the whole structure must
//     reproduce the original surface. A port that dropped the shift passes
//     only this case — which is why it is not the only one.
//   * shift landing exactly ON a quoted pillar of the original surface, and
//     shift landing strictly BETWEEN two pillars: the second one exercises
//     the original surface's interpolation at a time no query ever names
//     directly.
//   * shift PAST the original's last quoted date, where the original is in
//     its time-extrapolation regime for BOTH legs of the forward variance.
//     maxDate() is then EARLIER than referenceDate(), so maxTime() is
//     negative — pinned, because it decides whether checkRange lets any
//     non-extrapolating query through at all.
//
// For each implied structure the surface is sampled at strikes inside the
// quoted [80, 120] range and outside it on both sides (70, 130), and at
// times inside and past the original's horizon. blackVariance, blackVol and
// blackForwardVol are all pinned separately: a port that derives one from
// another with the wrong non-zero-time floor (BlackVarianceTermStructure
// uses 1e-5) shows up in blackVol but not in blackVariance.
//
// A time-only BlackVarianceCurve case is included as well, because that is
// the structure the class is documented to be used with (an asset-dependent
// implied vol term structure "doesn't make financial sense" per the header).
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/eqfx/impliedvol.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <ql/math/matrix.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancesurface.hpp>
#include <ql/termstructures/volatility/equityfx/impliedvoltermstructure.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kRef(15, June, 2026);
const Actual365Fixed kDc;
const NullCalendar kCal;

const std::vector<Date> kDates = {Date(15, September, 2026), Date(15, December, 2026),
                                  Date(15, June, 2027), Date(15, June, 2028)};
const std::vector<Real> kStrikes = {80.0, 100.0, 120.0};

// Rows = strikes, columns = dates. The middle row dips then rises so the
// surface has a genuine smile that is not symmetric in time.
Matrix volMatrix() {
    Matrix v(3, 4);
    v[0][0] = 0.20; v[0][1] = 0.21; v[0][2] = 0.22; v[0][3] = 0.23;
    v[1][0] = 0.10; v[1][1] = 0.15; v[1][2] = 0.20; v[1][3] = 0.25;
    v[2][0] = 0.20; v[2][1] = 0.21; v[2][2] = 0.22; v[2][3] = 0.23;
    return v;
}

// Query times. All are >= 0.3 so that, for the shift == 0 case, the second
// leg of the forward variance stays inside the region where the surface's
// backward linear extrapolation is still positive: below t ~ 0.25 the
// bilinear extension of the first two pillars goes negative and blackVol
// would be a NaN in C++ (sqrt of a negative) rather than an error.
// Labels are spelled out rather than streamed, so the JSON keys do not
// inherit setprecision(17) ("0.3" would print as 0.29999999999999999).
const std::vector<std::pair<std::string, Time>> kTimes = {
    {"t0p3", 0.3}, {"t0p75", 0.75}, {"t1p5", 1.5}, {"t3p0", 3.0}};
const std::vector<std::pair<std::string, Real>> kQueryStrikes = {
    {"k70", 70.0}, {"k100", 100.0}, {"k130", 130.0}};

bool firstEmitted = false;

void emitSurface(const std::string& key, const ImpliedVolTermStructure& ivts) {
    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;

    std::cout << "  \"" << key << "\": {\n"
              << "    \"reference_date\": " << ivts.referenceDate().serialNumber() << ",\n"
              << "    \"max_date\": " << ivts.maxDate().serialNumber() << ",\n"
              << "    \"max_time\": " << ivts.maxTime() << ",\n"
              << "    \"min_strike\": " << ivts.minStrike() << ",\n"
              << "    \"max_strike\": " << ivts.maxStrike() << ",\n";

    std::cout << "    \"variance\": {\n";
    bool first = true;
    for (const auto& tp : kTimes) {
        for (const auto& kp : kQueryStrikes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "      \"" << tp.first << "_" << kp.first << "\": "
                      << ivts.blackVariance(tp.second, kp.second, true);
        }
    }
    std::cout << "\n    },\n";

    std::cout << "    \"vol\": {\n";
    first = true;
    for (const auto& tp : kTimes) {
        for (const auto& kp : kQueryStrikes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "      \"" << tp.first << "_" << kp.first << "\": "
                      << ivts.blackVol(tp.second, kp.second, true);
        }
    }
    std::cout << "\n    },\n";

    // Forward vol / variance over consecutive pairs of the query times,
    // plus the degenerate t1 == t2 case that falls back to a finite
    // difference of variance.
    std::cout << "    \"forward_vol\": {\n";
    first = true;
    for (Size i = 0; i + 1 < kTimes.size(); ++i) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "      \"" << kTimes[i].first << "_to_" << kTimes[i + 1].first
                  << "_k100\": "
                  << ivts.blackForwardVol(kTimes[i].second, kTimes[i + 1].second, 100.0, true);
    }
    std::cout << ",\n      \"t0p75_to_t0p75_k100\": "
              << ivts.blackForwardVol(0.75, 0.75, 100.0, true);
    std::cout << "\n    },\n";

    std::cout << "    \"forward_variance\": {\n";
    first = true;
    for (Size i = 0; i + 1 < kTimes.size(); ++i) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "      \"" << kTimes[i].first << "_to_" << kTimes[i + 1].first
                  << "_k100\": "
                  << ivts.blackForwardVariance(kTimes[i].second, kTimes[i + 1].second, 100.0,
                                               true);
    }
    std::cout << "\n    },\n";

    // Date-anchored query: exercises checkRange + timeFromReference on the
    // implied (shifted) reference date rather than the original one.
    const Date d = ivts.referenceDate() + 200;
    std::cout << "    \"vol_by_date\": {\n"
              << "      \"date\": " << d.serialNumber() << ",\n"
              << "      \"k100\": " << ivts.blackVol(d, 100.0, true) << ",\n"
              << "      \"variance_k100\": " << ivts.blackVariance(d, 100.0, true) << "\n"
              << "    }\n"
              << "  }";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    auto surface = ext::make_shared<BlackVarianceSurface>(kRef, kCal, kDates, kStrikes,
                                                          volMatrix(), kDc);
    Handle<BlackVolTermStructure> surfaceHandle(surface);

    // shift == 0: must reproduce the original surface exactly.
    {
        ImpliedVolTermStructure ivts(surfaceHandle, kRef);
        emitSurface("surface_shift_zero", ivts);
    }
    // shift lands exactly on the second quoted pillar.
    {
        ImpliedVolTermStructure ivts(surfaceHandle, Date(15, December, 2026));
        emitSurface("surface_shift_on_pillar", ivts);
    }
    // shift lands strictly between the second and third pillars.
    {
        ImpliedVolTermStructure ivts(surfaceHandle, Date(3, March, 2027));
        emitSurface("surface_shift_between_pillars", ivts);
    }
    // shift lands past the original's last quoted date: both legs of the
    // forward variance are in the extrapolation regime and maxTime() < 0.
    {
        ImpliedVolTermStructure ivts(surfaceHandle, Date(15, December, 2028));
        emitSurface("surface_shift_past_max_date", ivts);
    }

    // Time-only structure — the documented use case.
    {
        std::vector<Volatility> vols = {0.13, 0.16, 0.20, 0.25};
        auto curve = ext::make_shared<BlackVarianceCurve>(kRef, kDates, vols, kDc, true);
        Handle<BlackVolTermStructure> curveHandle(curve);
        ImpliedVolTermStructure ivts(curveHandle, Date(3, March, 2027));
        emitSurface("curve_shift_between_pillars", ivts);
    }

    std::cout << "\n}\n";
    return 0;
}
