// migration-harness/cpp/probes/v143_eqfx_gridlocalvol/probe.cpp
//
// Pins FixedLocalVolSurface and GridModelLocalVolSurface.
//
// FixedLocalVolSurface has three behaviours a "reasonable" port gets wrong,
// and all three are probed here because an earlier version of the Python port
// had two of them wrong:
//
//   1. minStrike() / maxStrike() read the LAST time slice only
//      (strikes_.back()->front() / ->back()), not the union over slices. With
//      per-slice strike grids those differ, so the surface below deliberately
//      gives each slice a DIFFERENT strike range, widening then narrowing, so
//      that "last slice" and "union" and "first slice" are three different
//      answers.
//   2. The strike-extrapolation policy applies ONLY between time slices. ON a
//      time node localVolImpl calls localVolInterpol_[idx](strike, true) with
//      no clamping, so the strike is extrapolated LINEARLY there regardless of
//      ConstantExtrapolation (fixedlocalvolsurface.cpp:139-144). Every query
//      below is therefore run twice: at a node and just off it.
//   3. maxDate() for the time-anchored constructors is
//      yearFractionToDate(dayCounter, referenceDate, times.back()) — an
//      inverted day count, not the trivial referenceDate + 365*t. maxTime() is
//      times.back() exactly.
//
// Both Extrapolation policies are probed (Constant and
// InterpolatorDefault) at strikes below, inside and above each slice's grid.
//
// GridModelLocalVolSurface is a FixedLocalVolSurface whose every node is a
// CalibratedModel parameter. The discriminating question is the ARGUMENT
// LAYOUT: generateArguments() copies arguments_ into the (n_strikes x n_times)
// matrix through Matrix::begin(), which is ROW-major, so argument i*n_times+j
// is the node at strike i, time j. A transposed port still calibrates and
// still returns plausible numbers, so the probe sets a deliberately
// asymmetric parameter vector (1, 2, 3, ...)/10 and pins the surface it
// produces — under transposition every off-diagonal query changes.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/eqfx/gridlocalvol.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <ql/errors.hpp>
#include <ql/math/matrix.hpp>
#include <ql/termstructures/volatility/equityfx/fixedlocalvolsurface.hpp>
#include <ql/termstructures/volatility/equityfx/gridmodellocalvolsurface.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kRef(15, June, 2026);
const Actual365Fixed kDc;

// Deliberately unequal strike ranges per slice: [80,120], [60,140], [90,110].
// minStrike/maxStrike must come out 90/110 (the LAST slice), not 60/140 (the
// union) and not 80/120 (the first slice).
std::vector<ext::shared_ptr<std::vector<Real>>> perSliceStrikes() {
    return {ext::make_shared<std::vector<Real>>(std::vector<Real>{80.0, 100.0, 120.0}),
            ext::make_shared<std::vector<Real>>(std::vector<Real>{60.0, 100.0, 140.0}),
            ext::make_shared<std::vector<Real>>(std::vector<Real>{90.0, 100.0, 110.0})};
}

const std::vector<Time> kTimes = {0.25, 0.75, 1.5};

ext::shared_ptr<Matrix> volMatrix() {
    // rows = strike index within the slice, columns = time slice.
    auto m = ext::make_shared<Matrix>(3, 3);
    (*m)[0][0] = 0.25; (*m)[0][1] = 0.30; (*m)[0][2] = 0.34;
    (*m)[1][0] = 0.20; (*m)[1][1] = 0.22; (*m)[1][2] = 0.24;
    (*m)[2][0] = 0.22; (*m)[2][1] = 0.27; (*m)[2][2] = 0.29;
    return m;
}

// Times: on nodes, just off them, between slices and outside the grid on both
// sides (localVolImpl clamps t into [times.front(), times.back()]).
const std::vector<std::pair<std::string, Time>> kQueryTimes = {
    {"t0p1_before_grid", 0.1},
    {"t0p25_node0", 0.25},
    {"t0p2501_just_after_node0", 0.2501},
    {"t0p5_between", 0.5},
    {"t0p75_node1", 0.75},
    {"t1p0_between", 1.0},
    {"t1p5_node2", 1.5},
    {"t2p0_after_grid", 2.0}};

// Strikes below, inside and above every slice's range.
const std::vector<std::pair<std::string, Real>> kQueryStrikes = {
    {"k50", 50.0}, {"k70", 70.0}, {"k85", 85.0}, {"k100", 100.0},
    {"k105", 105.0}, {"k130", 130.0}, {"k150", 150.0}};

template <class Surface>
void emitGrid(const std::string& indent, const Surface& s) {
    bool first = true;
    for (const auto& tp : kQueryTimes) {
        for (const auto& kp : kQueryStrikes) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << indent << "\"" << tp.first << "_" << kp.first << "\": ";
            try {
                std::cout << s.localVol(tp.second, kp.second, true);
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

    const std::vector<std::pair<std::string, std::pair<FixedLocalVolSurface::Extrapolation,
                                                       FixedLocalVolSurface::Extrapolation>>>
        kPolicies = {
            {"constant_constant",
             {FixedLocalVolSurface::ConstantExtrapolation,
              FixedLocalVolSurface::ConstantExtrapolation}},
            {"interpolator_interpolator",
             {FixedLocalVolSurface::InterpolatorDefaultExtrapolation,
              FixedLocalVolSurface::InterpolatorDefaultExtrapolation}},
            {"constant_lower_only",
             {FixedLocalVolSurface::ConstantExtrapolation,
              FixedLocalVolSurface::InterpolatorDefaultExtrapolation}}};

    std::cout << "  \"fixed\": {\n";
    bool firstPolicy = true;
    for (const auto& pol : kPolicies) {
        if (!firstPolicy) std::cout << ",\n";
        firstPolicy = false;
        FixedLocalVolSurface s(kRef, kTimes, perSliceStrikes(), volMatrix(), kDc,
                               pol.second.first, pol.second.second);
        std::cout << "    \"" << pol.first << "\": {\n"
                  << "      \"max_date\": " << s.maxDate().serialNumber() << ",\n"
                  << "      \"max_time\": " << s.maxTime() << ",\n"
                  << "      \"min_strike\": " << s.minStrike() << ",\n"
                  << "      \"max_strike\": " << s.maxStrike() << ",\n"
                  << "      \"local_vol\": {\n";
        emitGrid("        ", s);
        std::cout << "      }\n    }";
    }
    std::cout << "\n  },\n";

    // The date-anchored constructor: maxDate() is the last DATE exactly,
    // whereas the time-anchored one round-trips through yearFractionToDate.
    {
        const std::vector<Date> dates = {Date(15, September, 2026), Date(15, March, 2027),
                                         Date(15, December, 2027)};
        const std::vector<Real> sharedStrikes = {80.0, 100.0, 120.0};
        FixedLocalVolSurface s(kRef, dates, sharedStrikes, volMatrix(), kDc);
        std::cout << "  \"fixed_from_dates\": {\n"
                  << "    \"max_date\": " << s.maxDate().serialNumber() << ",\n"
                  << "    \"max_time\": " << s.maxTime() << ",\n"
                  << "    \"min_strike\": " << s.minStrike() << ",\n"
                  << "    \"max_strike\": " << s.maxStrike() << ",\n"
                  << "    \"local_vol\": {\n";
        emitGrid("      ", s);
        std::cout << "    }\n  },\n";
    }

    // --- GridModelLocalVolSurface ---------------------------------------
    {
        const std::vector<Date> dates = {Date(15, September, 2026), Date(15, March, 2027),
                                         Date(15, December, 2027)};
        auto strikes = perSliceStrikes();
        GridModelLocalVolSurface g(kRef, dates, strikes, kDc);

        std::cout << "  \"grid_model_initial\": {\n"
                  << "    \"n_params\": " << g.params().size() << ",\n"
                  << "    \"max_date\": " << g.maxDate().serialNumber() << ",\n"
                  << "    \"max_time\": " << g.maxTime() << ",\n"
                  << "    \"min_strike\": " << g.minStrike() << ",\n"
                  << "    \"max_strike\": " << g.maxStrike() << ",\n"
                  << "    \"params\": [";
        for (Size i = 0; i < g.params().size(); ++i)
            std::cout << (i ? ", " : "") << g.params()[i];
        std::cout << "],\n    \"local_vol\": {\n";
        emitGrid("      ", g);
        std::cout << "    }\n  },\n";

        // Asymmetric parameter vector — the layout test.
        Array p(g.params().size());
        for (Size i = 0; i < p.size(); ++i)
            p[i] = (i + 1) / 10.0;
        g.setParams(p);

        std::cout << "  \"grid_model_after_set_params\": {\n"
                  << "    \"params\": [";
        for (Size i = 0; i < g.params().size(); ++i)
            std::cout << (i ? ", " : "") << g.params()[i];
        std::cout << "],\n    \"local_vol\": {\n";
        emitGrid("      ", g);
        std::cout << "    }\n  }\n";
    }

    std::cout << "}\n";
    return 0;
}
