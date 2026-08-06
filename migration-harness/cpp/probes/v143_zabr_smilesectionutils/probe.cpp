// migration-harness/cpp/probes/v143_zabr_smilesectionutils/probe.cpp
//
// Pins QuantLib::SmileSectionUtils
// (ql/termstructures/volatility/smilesectionutils.{hpp,cpp}).
//
// SmileSectionUtils does two things — build a sanitised strike grid, and find
// the maximal arbitrage-free window inside it — and both are full of branches
// that a "reasonable" reimplementation silently gets wrong. Each case below
// exists to force one of those branches:
//
//  * The DEFAULT moneyness grid is a hard-coded 21-entry table for shifted
//    lognormal sections and a DIFFERENT hard-coded 27-entry table for normal
//    ones. It is not a sigma-scaled or log-spaced grid. `sabr_default` and
//    `flat_normal` pin both tables verbatim via the emitted money grid.
//
//  * Moneyness maps to strike differently per vol type:
//        normal:            k = f + m
//        shifted lognormal: k = m*(f + shift) - shift
//    `sabr_shifted` uses a NEGATIVE forward with a positive shift so the two
//    formulas cannot be confused (they differ by more than rounding), and so
//    that the `tmp[0] <= QL_EPSILON` arm — which pushes (m = 0, k = -shift) —
//    is reached with a non-zero shift.
//
//  * When the caller's grid starts strictly above zero the ctor PREPENDS
//    (m = 0, k = -shift) before the loop. `sabr_custom_money` starts at 0.25
//    (prepend happens) and `sabr_money_from_zero` starts at 0.0 (it does not),
//    on otherwise identical inputs.
//
//  * `atm` defaults to Null<Real>() meaning "ask the section", but an explicit
//    atm REPLACES it and therefore rescales every strike. `sabr_atm_override`
//    passes an atm different from the section's own so a port that ignores the
//    argument produces a visibly different grid.
//
//  * Sections with a LIMITED strike range do not simply drop out-of-range
//    points: the first point below minStrike and the first above maxStrike are
//    replaced by the endpoint itself (with a moneyness recomputed as
//    (minStrike + shift)/f_, note the /f_ and not /(f_+shift)), and the
//    minStrikeAdded / maxStrikeAdded flags make that happen at most once each.
//    `interp_limited_range` uses an InterpolatedSmileSection whose pillars span
//    only [0.01, 0.05] while the default grid runs from 0 to 20x the forward,
//    so both endpoint insertions fire and most of the table is discarded.
//
//  * The call-price vector is NOT parallel to the strike vector in the obvious
//    way: for shifted lognormal, c_[0] is the analytic f_+shift and the pricing
//    loop starts at strike index 1; for normal it starts at index 0. Emitting
//    c_ alongside k_ catches an off-by-one here.
//
//  * The arbitrage-free window starts at a central index found by
//    upper_bound(m_, (normal ? 0.0 : 1.0) - QL_EPSILON), is pushed RIGHT while
//    the local af() test fails, then grows outward with a two-sided test —
//    and af() itself checks -1 <= dC/dK <= 0 on the left secant plus
//    convexity against the right secant. `sabr_arbitrageable` is a long-dated,
//    low-beta, high-vol-of-vol, strongly-negative-rho SABR whose Hagan
//    expansion is arbitrageable in the wings, so the window is a strict subset
//    of the grid. `sabr_arbitrageable_deleted` reruns it with
//    deleteArbitragePoints = true, which erases points OUTSIDE the window and
//    re-runs the scan until it is stable — so the two cases must differ in
//    grid length, not just in the reported indices.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/zabr/smilesectionutils.json.

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/termstructures/volatility/flatsmilesection.hpp>
#include <ql/termstructures/volatility/interpolatedsmilesection.hpp>
#include <ql/termstructures/volatility/sabrsmilesection.hpp>
#include <ql/termstructures/volatility/smilesectionutils.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

void emitArray(const char* key, const std::vector<Real>& v, bool trailingComma) {
    std::cout << "    \"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << v[i];
    std::cout << "]" << (trailingComma ? "," : "") << "\n";
}

void emitCase(const char* key,
              const SmileSection& section,
              const std::vector<Real>& moneyness,
              Real atm,
              bool deleteArbitragePoints,
              bool trailingComma) {
    std::cout << "  \"" << key << "\": {\n";
    try {
        const SmileSectionUtils u(section, moneyness, atm, deleteArbitragePoints);
        std::cout << "    \"atm_level\": " << u.atmLevel() << ",\n";
        emitArray("money_grid", u.moneyGrid(), true);
        emitArray("strike_grid", u.strikeGrid(), true);
        emitArray("call_prices", u.callPrices(), true);
        const std::pair<Real, Real> region = u.arbitragefreeRegion();
        const std::pair<Size, Size> indices = u.arbitragefreeIndices();
        std::cout << "    \"af_region_low\": " << region.first << ",\n";
        std::cout << "    \"af_region_high\": " << region.second << ",\n";
        std::cout << "    \"af_index_low\": " << indices.first << ",\n";
        std::cout << "    \"af_index_high\": " << indices.second << "\n";
    } catch (const std::exception& e) {
        std::cout << "    \"error\": \"" << e.what() << "\"\n";
    }
    std::cout << "  }" << (trailingComma ? "," : "") << "\n";
}

const std::vector<Real> kNoMoneyness;

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- shifted lognormal, unshifted, well behaved -----------------------
    // Same SABR parameters as the C++ ZabrTests consistency test, so the grid
    // and prices are comparable with the zabr probes.
    const SabrSmileSection sabr(5.0, 0.03, {0.08, 0.70, 0.20, -0.30});
    emitCase("sabr_default", sabr, kNoMoneyness, Null<Real>(), false, true);

    // First entry 0.25 > QL_EPSILON, so (m = 0, k = -shift) is PREPENDED.
    const std::vector<Real> customMoney = {0.25, 0.5, 0.75, 0.9, 1.0,
                                           1.1,  1.25, 1.5, 2.0, 3.0};
    emitCase("sabr_custom_money", sabr, customMoney, Null<Real>(), false, true);

    // Identical grid except for a leading 0.0 — no prepend this time.
    std::vector<Real> moneyFromZero;
    moneyFromZero.push_back(0.0);
    moneyFromZero.insert(moneyFromZero.end(), customMoney.begin(), customMoney.end());
    emitCase("sabr_money_from_zero", sabr, moneyFromZero, Null<Real>(), false, true);

    // Explicit atm != section.atmLevel() rescales every strike.
    emitCase("sabr_atm_override", sabr, kNoMoneyness, 0.035, false, true);

    // Negative forward + positive shift: k = m*(f + shift) - shift.
    const SabrSmileSection sabrShifted(5.0, -0.005, {0.08, 0.70, 0.20, -0.30}, 0.02);
    emitCase("sabr_shifted", sabrShifted, kNoMoneyness, Null<Real>(), false, true);

    // Normal vol type: the 27-entry default table, k = f + m, and the pricing
    // loop starts at index 0 rather than 1.
    const FlatSmileSection flatNormal(3.0, 0.0080, Actual365Fixed(), 0.02, Normal);
    emitCase("flat_normal", flatNormal, kNoMoneyness, Null<Real>(), false, true);

    // Limited strike range: both endpoint insertions fire.
    const Real t = 2.0;
    const std::vector<Rate> pillars = {0.01, 0.02, 0.03, 0.04, 0.05};
    const std::vector<Real> vols = {0.42, 0.36, 0.33, 0.34, 0.37};
    std::vector<Real> stdDevs;
    for (Real v : vols)
        stdDevs.push_back(v * std::sqrt(t));
    const InterpolatedSmileSection<Linear> interp(t, pillars, stdDevs, 0.03);
    emitCase("interp_limited_range", interp, kNoMoneyness, Null<Real>(), false, true);

    // Long-dated, low beta, high vol-of-vol, strongly negative rho: Hagan's
    // expansion is arbitrageable in the wings here, so the window is a strict
    // subset of the grid.
    const SabrSmileSection sabrArb(15.0, 0.02, {0.06, 0.10, 0.85, -0.65});
    emitCase("sabr_arbitrageable", sabrArb, kNoMoneyness, Null<Real>(), false, true);
    emitCase("sabr_arbitrageable_deleted", sabrArb, kNoMoneyness, Null<Real>(), true, false);

    std::cout << "}\n";
    return 0;
}
