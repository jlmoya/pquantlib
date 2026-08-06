// migration-harness/cpp/probes/v143_ts_yieldadapters/probe.cpp
//
// Reference values for five yield term structures layered on a base curve:
//
//   * UltimateForwardTermStructure                     (Dutch DNB UFR)
//   * CompositeZeroYieldStructure                      (two curves + a functor)
//   * QuantoTermStructure                              (dividend + quanto drift)
//   * InterpolatedPiecewiseZeroSpreadedTermStructure   (dated zero spreads)
//   * InterpolatedPiecewiseForwardSpreadedTermStructure(dated forward spreads)
//   * InterpolatedSimpleZeroCurve                      (simple, not compounded, zeros)
//
// Each is read at times chosen to sit in every regime the implementation
// branches on, because the branch boundaries are where a port goes wrong:
//
//   * UFR: before, exactly at, and past the first smoothing point (the
//     zeroYieldImpl branch is `deltaT > 0`), and with rounding on and off.
//     Rounding is probed in a non-continuous compounding, since that path
//     round-trips the rate through InterestRate::equivalentRate twice and is
//     the only place the compounding/frequency arguments matter at all.
//   * The two spreaded curves: before the first spread date, on a spread
//     date, between two, and past the last. calcSpread is FLAT outside the
//     quoted range rather than extrapolated by the interpolator, and the
//     forward-spreaded curve additionally continues its primitive linearly
//     past the last node — two separate off-by-a-branch opportunities.
//     The forward-spreaded curve also differs from the zero-spreaded one in
//     that it does NOT round-trip through a compounding: it adds the average
//     forward spread straight onto the continuous zero rate.
//   * InterpolatedSimpleZeroCurve: discount is 1/(1+Rt), NOT exp(-Rt), and
//     past the last node R is rebuilt from a flat instantaneous forward.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/yieldadapters.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/interpolations/backwardflatinterpolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/compositezeroyieldstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/interpolatedsimplezerocurve.hpp>
#include <ql/termstructures/yield/piecewiseforwardspreadedtermstructure.hpp>
#include <ql/termstructures/yield/piecewisezerospreadedtermstructure.hpp>
#include <ql/termstructures/yield/quantotermstructure.hpp>
#include <ql/termstructures/yield/ultimateforwardtermstructure.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kRef(17, January, 2024);

// Times at which every adapter is read: t=0, inside, on and past the
// structural boundaries below (the 5y first-smoothing-point, the 2y/5y/10y
// spread dates, the 10y last node).
const double kTimes[] = {0.0,  0.5,  1.0,  2.0,  3.0,  4.0,  5.0,
                         6.0,  7.5,  10.0, 12.0, 20.0, 40.0, 60.0};

std::vector<Date> spreadDates() {
    return {kRef + 2 * Years, kRef + 5 * Years, kRef + 10 * Years};
}

std::vector<Handle<Quote>> spreadQuotes() {
    return {Handle<Quote>(ext::make_shared<SimpleQuote>(0.0010)),
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.0035)),
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.0020))};
}

// Base curve: a non-flat zero curve, so the adapters cannot be reproduced by
// accident from a constant.
Handle<YieldTermStructure> baseCurve() {
    std::vector<Date> dates = {kRef, kRef + 1 * Years, kRef + 3 * Years,
                               kRef + 5 * Years, kRef + 10 * Years, kRef + 30 * Years};
    std::vector<Rate> zeros = {0.0250, 0.0280, 0.0315, 0.0330, 0.0345, 0.0360};
    auto c = ext::make_shared<InterpolatedZeroCurve<Linear>>(dates, zeros, Actual365Fixed(),
                                                             TARGET());
    c->enableExtrapolation();
    return Handle<YieldTermStructure>(c);
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    auto c = ext::make_shared<FlatForward>(kRef, r, Actual365Fixed(), Continuous, Annual);
    c->enableExtrapolation();
    return Handle<YieldTermStructure>(c);
}

bool firstEmitted = false;

void emitCurve(const std::string& key, const YieldTermStructure& curve) {
    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"reference_date\": " << curve.referenceDate().serialNumber() << ",\n"
              << "    \"max_date\": " << curve.maxDate().serialNumber() << ",\n"
              << "    \"discount\": [";
    bool first = true;
    for (double t : kTimes) {
        std::cout << (first ? "" : ", ") << curve.discount(t, true);
        first = false;
    }
    std::cout << "],\n    \"zero\": [";
    first = true;
    for (double t : kTimes) {
        std::cout << (first ? "" : ", ")
                  << curve.zeroRate(t, Continuous, NoFrequency, true).rate();
        first = false;
    }
    std::cout << "]\n  }";
}

Real addRates(Rate a, Rate b) { return a + b; }
Real subRates(Rate a, Rate b) { return a - b; }

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n  \"times\": [";
    bool first = true;
    for (double t : kTimes) {
        std::cout << (first ? "" : ", ") << t;
        first = false;
    }
    std::cout << "]";
    firstEmitted = true;

    Handle<YieldTermStructure> base = baseCurve();

    // --- UltimateForwardTermStructure ------------------------------------
    Handle<Quote> llfr(ext::make_shared<SimpleQuote>(0.0426));
    Handle<Quote> ufr(ext::make_shared<SimpleQuote>(0.0210));
    {
        UltimateForwardTermStructure ts(base, llfr, ufr, 5 * Years, 0.1);
        ts.enableExtrapolation();
        emitCurve("ufr_alpha_010", ts);
    }
    {
        UltimateForwardTermStructure ts(base, llfr, ufr, 5 * Years, 0.02);
        ts.enableExtrapolation();
        emitCurve("ufr_alpha_002", ts);
    }
    {
        // 20y smoothing point moves the branch boundary, so a port that
        // hard-codes or mis-converts the cut-off time shows up here.
        UltimateForwardTermStructure ts(base, llfr, ufr, 20 * Years, 0.1);
        ts.enableExtrapolation();
        emitCurve("ufr_fsp_20y", ts);
    }
    {
        // Rounding under a NON-continuous compounding: the only path where
        // the compounding / frequency arguments do anything.
        UltimateForwardTermStructure ts(base, llfr, ufr, 5 * Years, 0.1, 4, Compounded,
                                        Annual);
        ts.enableExtrapolation();
        emitCurve("ufr_rounded_4dp_annual", ts);
    }
    {
        UltimateForwardTermStructure ts(base, llfr, ufr, 5 * Years, 0.1, 5, Continuous,
                                        NoFrequency);
        ts.enableExtrapolation();
        emitCurve("ufr_rounded_5dp_continuous", ts);
    }

    // --- CompositeZeroYieldStructure --------------------------------------
    {
        CompositeZeroYieldStructure<Real (*)(Rate, Rate)> ts(base, flatCurve(0.005),
                                                             addRates);
        ts.enableExtrapolation();
        emitCurve("composite_sum_continuous", ts);
    }
    {
        // Compounded/Annual: both legs are read under that compounding and the
        // composite is converted back to continuous, so the round trip is not
        // the identity and a port that skips it diverges.
        CompositeZeroYieldStructure<Real (*)(Rate, Rate)> ts(base, flatCurve(0.005),
                                                             addRates, Compounded, Annual);
        ts.enableExtrapolation();
        emitCurve("composite_sum_annual", ts);
    }
    {
        CompositeZeroYieldStructure<Real (*)(Rate, Rate)> ts(base, flatCurve(0.005),
                                                             subRates, Compounded,
                                                             Semiannual);
        ts.enableExtrapolation();
        emitCurve("composite_diff_semiannual", ts);
    }

    // --- QuantoTermStructure ----------------------------------------------
    {
        Handle<BlackVolTermStructure> underlyingVol(ext::make_shared<BlackConstantVol>(
            kRef, TARGET(), 0.25, Actual365Fixed()));
        Handle<BlackVolTermStructure> fxVol(
            ext::make_shared<BlackConstantVol>(kRef, TARGET(), 0.12, Actual365Fixed()));
        QuantoTermStructure ts(flatCurve(0.015), flatCurve(0.030), flatCurve(0.022),
                               underlyingVol, 100.0, fxVol, 1.25, -0.4);
        ts.enableExtrapolation();
        emitCurve("quanto", ts);
    }

    // --- InterpolatedPiecewiseZeroSpreadedTermStructure --------------------
    {
        InterpolatedPiecewiseZeroSpreadedTermStructure<Linear> ts(base, spreadQuotes(),
                                                                  spreadDates());
        ts.enableExtrapolation();
        emitCurve("zero_spreaded_linear", ts);
    }
    {
        InterpolatedPiecewiseZeroSpreadedTermStructure<Linear> ts(
            base, spreadQuotes(), spreadDates(), Compounded, Annual);
        ts.enableExtrapolation();
        emitCurve("zero_spreaded_linear_annual", ts);
    }
    {
        InterpolatedPiecewiseZeroSpreadedTermStructure<BackwardFlat> ts(
            base, spreadQuotes(), spreadDates());
        ts.enableExtrapolation();
        emitCurve("zero_spreaded_backward_flat", ts);
    }

    // --- InterpolatedPiecewiseForwardSpreadedTermStructure ------------------
    {
        InterpolatedPiecewiseForwardSpreadedTermStructure<Linear> ts(base, spreadQuotes(),
                                                                     spreadDates());
        ts.enableExtrapolation();
        emitCurve("forward_spreaded_linear", ts);
    }
    {
        InterpolatedPiecewiseForwardSpreadedTermStructure<BackwardFlat> ts(
            base, spreadQuotes(), spreadDates());
        ts.enableExtrapolation();
        emitCurve("forward_spreaded_backward_flat", ts);
    }

    // --- InterpolatedSimpleZeroCurve ---------------------------------------
    {
        std::vector<Date> dates = {kRef, kRef + 1 * Years, kRef + 3 * Years,
                                   kRef + 5 * Years, kRef + 10 * Years};
        std::vector<Rate> simpleZeros = {0.0250, 0.0280, 0.0315, 0.0330, 0.0345};
        InterpolatedSimpleZeroCurve<Linear> ts(dates, simpleZeros, Actual365Fixed(),
                                               TARGET());
        ts.enableExtrapolation();
        emitCurve("simple_zero_linear", ts);
    }

    std::cout << "\n}\n";
    return 0;
}
