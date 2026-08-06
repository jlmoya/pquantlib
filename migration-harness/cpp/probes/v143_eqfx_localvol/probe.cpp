// migration-harness/cpp/probes/v143_eqfx_localvol/probe.cpp
//
// Pins LocalVolSurface (Dupire on a Black vol surface) with NON-FLAT yield
// curves, and NoExceptLocalVolSurface (the same surface with a fallback
// value instead of an exception).
//
// Why non-flat curves are the point: the yield curves enter localVolImpl in
// TWO places, and a port can get one right while dropping the other.
//
//   1. the forward, F(t) = S * dq(t) / dr(t), which sets the log-moneyness
//      y = log(K / F(t)) that the whole Dupire denominator is written in;
//   2. the time derivative, which is taken at constant forward moneyness —
//      the strike is carried along as
//          K(t±dt) = K * dr(t) * dq(t±dt) / (dr(t±dt) * dq(t))
//      i.e. K * exp(±(r - q) dt) for flat curves — NOT held fixed.
//
// With r == q == 0 both reduce to the trivial case (F = S, K(t±dt) = K), so
// a port that hard-codes zero rates reproduces a zero-rate reference to the
// last bit and is silently wrong everywhere else. Hence:
//
//   * "zero_rates"      r = q = 0            — the degenerate case; kept as
//                                              a regression guard so the
//                                              old (wrong) port still passes
//                                              exactly one block.
//   * "r5_q0"           r = 5%, q = 0        — forward above spot, and a
//                                              non-zero strike drift.
//   * "r5_q2"           r = 5%, q = 2%       — both curves non-trivial and
//                                              different from each other.
//   * "r2_q5"           r = 2%, q = 5%       — forward BELOW spot, so the
//                                              sign of both effects flips.
//
// Sampled on a (t, S) grid that includes t = 0 (the forward-difference
// branch, dt = 1e-4) and t > 0 (the central branch, dt = min(1e-4, t/2) —
// note t = 0.0001 makes dt = t/2, a different step from every other point),
// and at underlying levels inside and outside the quoted strike range.
//
// NoExceptLocalVolSurface is probed on a surface whose variance DECREASES in
// time at the low strike, which is exactly what QL_ENSURE(wpt >= w) exists
// to catch: LocalVolSurface throws there, NoExceptLocalVolSurface returns
// the overwrite. Both are pinned at the same points, together with points
// where the calculation succeeds and the two must agree bit for bit.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/eqfx/localvol.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <ql/errors.hpp>
#include <ql/math/matrix.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancesurface.hpp>
#include <ql/termstructures/volatility/equityfx/localvolsurface.hpp>
#include <ql/termstructures/volatility/equityfx/noexceptlocalvolsurface.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
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

Matrix goodVols() {
    Matrix v(3, 4);
    v[0][0] = 0.20; v[0][1] = 0.21; v[0][2] = 0.22; v[0][3] = 0.23;
    v[1][0] = 0.10; v[1][1] = 0.15; v[1][2] = 0.20; v[1][3] = 0.25;
    v[2][0] = 0.20; v[2][1] = 0.21; v[2][2] = 0.22; v[2][3] = 0.23;
    return v;
}

// Low strike's variance collapses between the first two pillars
// (0.2521 * 0.40^2 = 0.0403 -> 0.5013 * 0.15^2 = 0.0113), so the
// QL_ENSURE(wpt >= w) guard fires for underlying levels near 80.
Matrix nonMonotoneVols() {
    Matrix v(3, 4);
    v[0][0] = 0.40; v[0][1] = 0.15; v[0][2] = 0.14; v[0][3] = 0.13;
    v[1][0] = 0.20; v[1][1] = 0.21; v[1][2] = 0.22; v[1][3] = 0.23;
    v[2][0] = 0.20; v[2][1] = 0.21; v[2][2] = 0.22; v[2][3] = 0.23;
    return v;
}

// Labels spelled out so JSON keys do not inherit setprecision(17).
const std::vector<std::pair<std::string, Time>> kTimes = {
    {"t0", 0.0},          // forward-difference branch, dt = 1e-4
    {"t0p0001", 0.0001},  // central branch with dt = t/2, not 1e-4
    {"t0p3", 0.3},
    {"t0p75", 0.75},
    {"t1p5", 1.5},
    {"t2p5_past_max", 2.5}};

const std::vector<std::pair<std::string, Real>> kLevels = {
    {"s70", 70.0}, {"s80", 80.0}, {"s100", 100.0}, {"s120", 120.0}, {"s130", 130.0}};

Handle<YieldTermStructure> flat(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kRef, r, kDc, Continuous, Annual));
}

template <class Surface>
void emitGrid(const std::string& indent, Surface& lvs) {
    bool first = true;
    for (const auto& tp : kTimes) {
        for (const auto& sp : kLevels) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << indent << "\"" << tp.first << "_" << sp.first << "\": ";
            try {
                std::cout << lvs.localVol(tp.second, sp.second, true);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
    }
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    auto good = ext::make_shared<BlackVarianceSurface>(kRef, kCal, kDates, kStrikes,
                                                       goodVols(), kDc);
    Handle<BlackVolTermStructure> goodHandle(good);
    Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));

    const std::vector<std::pair<std::string, std::pair<Rate, Rate>>> kRateCases = {
        {"zero_rates", {0.00, 0.00}},
        {"r5_q0", {0.05, 0.00}},
        {"r5_q2", {0.05, 0.02}},
        {"r2_q5", {0.02, 0.05}}};

    std::cout << "  \"surface\": {\n";
    bool firstCase = true;
    for (const auto& rc : kRateCases) {
        if (!firstCase) std::cout << ",\n";
        firstCase = false;
        LocalVolSurface lvs(goodHandle, flat(rc.second.first), flat(rc.second.second), spot);
        std::cout << "    \"" << rc.first << "\": {\n"
                  << "      \"reference_date\": " << lvs.referenceDate().serialNumber() << ",\n"
                  << "      \"max_date\": " << lvs.maxDate().serialNumber() << ",\n"
                  << "      \"min_strike\": " << lvs.minStrike() << ",\n"
                  << "      \"max_strike\": " << lvs.maxStrike() << ",\n"
                  << "      \"local_vol\": {\n";
        emitGrid("        ", lvs);
        std::cout << "      }\n    }";
    }
    std::cout << "\n  },\n";

    // The Real-underlying constructor overload must agree with the
    // Handle<Quote> one at the same spot value.
    {
        LocalVolSurface lvs(goodHandle, flat(0.05), flat(0.02), 100.0);
        std::cout << "  \"surface_real_underlying_r5_q2\": {\n    \"local_vol\": {\n";
        emitGrid("      ", lvs);
        std::cout << "    }\n  },\n";
    }

    // --- NoExceptLocalVolSurface ----------------------------------------
    auto bad = ext::make_shared<BlackVarianceSurface>(kRef, kCal, kDates, kStrikes,
                                                      nonMonotoneVols(), kDc);
    Handle<BlackVolTermStructure> badHandle(bad);
    const Real kOverwrite = 0.123456789;

    {
        LocalVolSurface plain(badHandle, flat(0.05), flat(0.02), spot);
        std::cout << "  \"non_monotone_plain\": {\n    \"local_vol\": {\n";
        emitGrid("      ", plain);
        std::cout << "    }\n  },\n";
    }
    {
        NoExceptLocalVolSurface noexc(badHandle, flat(0.05), flat(0.02), spot, kOverwrite);
        std::cout << "  \"non_monotone_noexcept\": {\n"
                  << "    \"overwrite\": " << kOverwrite << ",\n"
                  << "    \"local_vol\": {\n";
        emitGrid("      ", noexc);
        std::cout << "    }\n  },\n";
    }
    // Same fallback class over the SMOOTH-VOL surface. The wrapper must be
    // inert wherever the plain surface returns — but note this block is NOT
    // all-numbers: at r = 5%, q = 2% the smooth surface still produces a
    // negative local variance at (t = 1.5, S = 100) and (t = 2.5, S = 100),
    // so the overwrite shows up there too. That is a property of the Dupire
    // denominator, not of the fallback, and it is exactly why the plain and
    // wrapped grids are both pinned.
    {
        NoExceptLocalVolSurface noexc(goodHandle, flat(0.05), flat(0.02), 100.0, kOverwrite);
        std::cout << "  \"good_noexcept_real_underlying\": {\n    \"local_vol\": {\n";
        emitGrid("      ", noexc);
        std::cout << "    }\n  }\n";
    }

    std::cout << "}\n";
    return 0;
}
