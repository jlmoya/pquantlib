// L7-B inflation curves + helpers probe.
//
// Emits reference values for:
//   * InterpolatedZeroInflationCurve sampled at intermediate dates
//     (Linear default; nodes at known [date, rate] pairs) — used by tests
//     to compare Python-side _zero_rate_impl against C++ at the same
//     times.
//   * InterpolatedYoYInflationCurve sampled likewise.
//   * PiecewiseZeroInflationCurve bootstrap from 3 ZeroCouponInflationSwapHelpers
//     (roundtrip — implied_quote should match input quote LOOSE).
//   * ZeroInflationTraits/YoYInflationTraits constants: avg_inflation,
//     max_inflation, max_iterations.
//
// References used by the Python L7-B tests at:
//   pquantlib/tests/termstructures/inflation/test_*.py
//
// Build with:
//   cmake --build migration-harness/cpp/build/probes --target cluster_l7b_probe

#include <ql/cashflows/inflationcouponpricer.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/inflation/inflationhelpers.hpp>
#include <ql/termstructures/inflation/interpolatedyoyinflationcurve.hpp>
#include <ql/termstructures/inflation/interpolatedzeroinflationcurve.hpp>
#include <ql/termstructures/inflation/piecewisezeroinflationcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    Date today(15, January, 2020);
    Settings::instance().evaluationDate() = today;

    DayCounter dc = Actual360();
    Calendar cal = TARGET();
    Frequency freq = Monthly;

    // ---- InterpolatedZeroInflationCurve: linear interpolation between known
    //      (date, zero rate) pairs.  Emits the input nodes and several
    //      intermediate evaluations.
    {
        std::cout << "  \"interpolated_zero\": {\n";
        std::vector<Date> dates = {
            Date(15, January, 2020),
            Date(15, January, 2021),
            Date(15, January, 2023),
            Date(15, January, 2025),
            Date(15, January, 2030)};
        std::vector<Rate> rates = {0.015, 0.02, 0.025, 0.022, 0.024};

        InterpolatedZeroInflationCurve<Linear> curve(
            today, dates, rates, freq, dc);

        std::cout << "    \"day_counter\": \"Actual/360\",\n";
        std::cout << "    \"frequency\": " << static_cast<int>(freq) << ",\n";
        std::cout << "    \"reference_serial\": " << today.serialNumber() << ",\n";
        std::cout << "    \"base_serial\": " << curve.baseDate().serialNumber() << ",\n";
        std::cout << "    \"max_serial\": " << curve.maxDate().serialNumber() << ",\n";

        std::cout << "    \"nodes\": [\n";
        for (size_t i = 0; i < dates.size(); ++i) {
            std::cout << "      {\"date_serial\": " << dates[i].serialNumber()
                      << ", \"rate\": " << rates[i] << "}"
                      << (i + 1 == dates.size() ? "" : ",") << "\n";
        }
        std::cout << "    ],\n";

        struct Sample { const char* label; Date d; };
        std::vector<Sample> samples = {
            {"midnode_2022_01", Date(15, January, 2022)},
            {"midnode_2024_07", Date(15, July, 2024)},
            {"midnode_2027_06", Date(15, June, 2027)},
            {"midnode_2021_jun_15", Date(15, June, 2021)},
            {"midnode_2026_dec_15", Date(15, December, 2026)},
        };
        std::cout << "    \"samples\": [\n";
        for (size_t i = 0; i < samples.size(); ++i) {
            // zeroRate() applies inflation-period bucketing first.
            Rate r = curve.zeroRate(samples[i].d, false);
            std::cout << "      {\n";
            std::cout << "        \"label\": \"" << samples[i].label << "\",\n";
            std::cout << "        \"date_serial\": " << samples[i].d.serialNumber() << ",\n";
            std::cout << "        \"zero_rate\": " << r << "\n";
            std::cout << "      }" << (i + 1 == samples.size() ? "" : ",") << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ---- InterpolatedYoYInflationCurve: same shape with YoY semantics.
    {
        std::cout << "  \"interpolated_yoy\": {\n";
        std::vector<Date> dates = {
            Date(15, January, 2020),
            Date(15, January, 2021),
            Date(15, January, 2023),
            Date(15, January, 2025),
            Date(15, January, 2030)};
        std::vector<Rate> rates = {0.020, 0.022, 0.025, 0.023, 0.024};

        InterpolatedYoYInflationCurve<Linear> curve(
            today, dates, rates, freq, dc);

        std::cout << "    \"base_serial\": " << curve.baseDate().serialNumber() << ",\n";
        std::cout << "    \"base_rate\": " << curve.baseRate() << ",\n";
        std::cout << "    \"max_serial\": " << curve.maxDate().serialNumber() << ",\n";

        std::cout << "    \"nodes\": [\n";
        for (size_t i = 0; i < dates.size(); ++i) {
            std::cout << "      {\"date_serial\": " << dates[i].serialNumber()
                      << ", \"rate\": " << rates[i] << "}"
                      << (i + 1 == dates.size() ? "" : ",") << "\n";
        }
        std::cout << "    ],\n";

        struct Sample { const char* label; Date d; };
        std::vector<Sample> samples = {
            {"midnode_2022_01", Date(15, January, 2022)},
            {"midnode_2024_07", Date(15, July, 2024)},
            {"midnode_2027_06", Date(15, June, 2027)},
        };
        std::cout << "    \"samples\": [\n";
        for (size_t i = 0; i < samples.size(); ++i) {
            Rate r = curve.yoyRate(samples[i].d, false);
            std::cout << "      {\n";
            std::cout << "        \"label\": \"" << samples[i].label << "\",\n";
            std::cout << "        \"date_serial\": " << samples[i].d.serialNumber() << ",\n";
            std::cout << "        \"yoy_rate\": " << r << "\n";
            std::cout << "      }" << (i + 1 == samples.size() ? "" : ",") << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ---- Bootstrap traits constants (no template magic — pure numeric).
    {
        std::cout << "  \"traits\": {\n";
        // detail::avgInflation = 0.02; detail::maxInflation = 0.5.
        // maxIterations = 40.  Surfaced as test-side constants.
        std::cout << "    \"avg_inflation\": 0.02,\n";
        std::cout << "    \"max_inflation\": 0.5,\n";
        std::cout << "    \"max_iterations\": 40\n";
        std::cout << "  },\n";
    }

    // ---- PiecewiseZeroInflationCurve: bootstrap roundtrip.
    //      Build a 3-instrument piecewise curve from ZeroCouponInflationSwapHelpers
    //      and verify the implied quotes match the inputs LOOSE.
    {
        std::cout << "  \"piecewise_zero_roundtrip\": {\n";

        // Nominal yield curve (flat 4% Continuous).
        Handle<YieldTermStructure> nominalH(
            ext::make_shared<FlatForward>(today, 0.04, Actual360(), Continuous));

        // ZeroInflationIndex (EUHICP variant — not-interpolated).
        auto zii = ext::make_shared<EUHICP>();
        Date baseFix(1, October, 2019);
        Date nextFix(1, November, 2019);
        zii->addFixing(baseFix, 100.0);
        zii->addFixing(nextFix, 100.5);

        std::vector<Rate> quotes = {0.022, 0.024, 0.025};
        std::vector<Period> maturities = {Period(2, Years), Period(5, Years), Period(10, Years)};
        Period swapObsLag(3, Months);
        DayCounter swapDc = Thirty360(Thirty360::BondBasis);

        std::vector<ext::shared_ptr<BootstrapHelper<ZeroInflationTermStructure>>> helpers;
        std::cout << "    \"quotes\": [";
        for (size_t i = 0; i < quotes.size(); ++i) {
            std::cout << quotes[i] << (i + 1 == quotes.size() ? "" : ", ");
            auto q = Handle<Quote>(ext::make_shared<SimpleQuote>(quotes[i]));
            Date maturity = cal.advance(today, maturities[i]);
            // Use the non-deprecated v1.42.1 constructor (no nominal curve;
            // C++ internally uses a flat-zero discount which cancels out
            // for the zero-coupon swap fair rate).
            auto helper = ext::make_shared<ZeroCouponInflationSwapHelper>(
                q, swapObsLag, maturity, cal, ModifiedFollowing, swapDc,
                zii, CPI::AsIndex);
            helpers.push_back(helper);
        }
        std::cout << "],\n";
        (void) nominalH;  // unused here but retained for future divergence notes

        std::cout << "    \"maturity_serials\": [";
        for (size_t i = 0; i < maturities.size(); ++i) {
            std::cout << cal.advance(today, maturities[i]).serialNumber()
                      << (i + 1 == maturities.size() ? "" : ", ");
        }
        std::cout << "],\n";

        std::cout << "    \"swap_obs_lag_months\": " << swapObsLag.length() << ",\n";

        // Build the piecewise curve.  Curve baseDate is the period start
        // of (today - swapObsLag).
        Date curveBase = inflationPeriod(today - swapObsLag, freq).first;
        auto curve = ext::make_shared<PiecewiseZeroInflationCurve<Linear>>(
            today, curveBase, freq, dc, helpers);

        std::cout << "    \"curve_base_serial\": " << curve->baseDate().serialNumber() << ",\n";

        // Trigger lazy bootstrap.
        std::vector<Real> implied;
        for (const auto& h : helpers) {
            h->setTermStructure(curve.get());
        }
        // Force calculation.
        (void) curve->zeroRate(cal.advance(today, maturities[0]), false);

        std::cout << "    \"implied_quotes\": [";
        for (size_t i = 0; i < helpers.size(); ++i) {
            Real iq = helpers[i]->impliedQuote();
            implied.push_back(iq);
            std::cout << iq << (i + 1 == helpers.size() ? "" : ", ");
        }
        std::cout << "]\n";

        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
