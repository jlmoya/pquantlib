// L8-A piecewise inflation + IterativeBootstrap probe.
//
// Emits reference values for:
//   * PiecewiseYoYInflationCurve bootstrap from 3 YearOnYearInflationSwapHelpers
//     (roundtrip — implied_quote should match input quote LOOSE under flat
//     zero discounting). The C++ helper builds a full YYIIS over a coupon
//     schedule; the PQuantLib helper short-circuits to ts.yoy_rate(...) due
//     to deferred YoY-coupon-leg builder. Both pin the same pillar value
//     for the linear-flat case used here.
//   * IterativeBootstrap on a simple synthetic yield-curve case where the
//     helper's implied_quote is a direct lookup of the curve at the pillar
//     (data[i] == quote[i]). Used to validate the generic loop.
//
// References used by the Python L8-A tests at:
//   pquantlib/tests/termstructures/bootstrap/test_iterative_bootstrap.py
//   pquantlib/tests/termstructures/inflation/test_piecewise_yoy_inflation_curve.py
//
// The L7-B probe already emits the PiecewiseZeroInflationCurve roundtrip
// reference at references/cluster/l7b.json — L8-A reuses it directly.
//
// Build with:
//   cmake --build migration-harness/cpp/build/probes --target cluster_l8a_probe

#include <ql/handle.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/inflation/inflationhelpers.hpp>
#include <ql/termstructures/inflation/piecewiseyoyinflationcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
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

    // ---- IterativeBootstrap mechanics check: 3-pillar linear curve where
    //      helper.impliedQuote() = curve.value(pillarTime) — bootstrap pins
    //      data[i] = quote[i] to within solver tolerance.
    //
    //      This is documented via the test_iterative_bootstrap.py
    //      synthetic case — no C++ value to emit since the PQuantLib test
    //      uses its own _FakeCurve / _FakeHelper.
    {
        std::cout << "  \"iterative_bootstrap_invariants\": {\n";
        std::cout << "    \"description\": \"synthetic — see "
                     "test_iterative_bootstrap.py for the algorithm contract\",\n";
        std::cout << "    \"max_iterations\": 40,\n";
        std::cout << "    \"accuracy_default\": 1e-12\n";
        std::cout << "  },\n";
    }

    // ---- PiecewiseYoYInflationCurve bootstrap roundtrip.
    //
    //      Build a 3-instrument curve over YYIIS helpers and emit the
    //      pillar dates + implied quotes (which should match inputs LOOSE
    //      under flat-zero discounting).
    //
    //      Note: the PQuantLib YoY helper simplifies the C++ YYIIS fairRate
    //      to ts.yoy_rate(maturity - lag); for the test's flat-zero nominal
    //      + LinearInterpolation YoY curve, both give the same pillar value.
    //      The C++ bootstrap result here is therefore a useful
    //      cross-validation despite the simplification.
    {
        std::cout << "  \"piecewise_yoy_roundtrip\": {\n";

        auto yyii = ext::make_shared<YYEUHICP>();
        std::vector<Rate> quotes = {0.020, 0.025, 0.030};
        std::vector<Period> maturities = {Period(2, Years), Period(5, Years), Period(10, Years)};
        Period swapObsLag(3, Months);
        DayCounter swapDc = Thirty360(Thirty360::BondBasis);

        std::vector<ext::shared_ptr<BootstrapHelper<YoYInflationTermStructure>>> helpers;
        std::cout << "    \"quotes\": [";
        for (size_t i = 0; i < quotes.size(); ++i) {
            std::cout << quotes[i] << (i + 1 == quotes.size() ? "" : ", ");
            auto q = Handle<Quote>(ext::make_shared<SimpleQuote>(quotes[i]));
            Date maturity = cal.advance(today, maturities[i]);
            auto helper = ext::make_shared<YearOnYearInflationSwapHelper>(
                q, swapObsLag, maturity, cal, ModifiedFollowing, swapDc, yyii);
            helpers.push_back(helper);
        }
        std::cout << "],\n";

        std::cout << "    \"maturity_serials\": [";
        for (size_t i = 0; i < maturities.size(); ++i) {
            std::cout << cal.advance(today, maturities[i]).serialNumber()
                      << (i + 1 == maturities.size() ? "" : ", ");
        }
        std::cout << "],\n";

        std::cout << "    \"swap_obs_lag_months\": " << swapObsLag.length() << ",\n";

        Date curveBase = inflationPeriod(today - swapObsLag, freq).first;
        // YoY needs a base rate; choose 0.018.
        Rate baseRate = 0.018;
        auto curve = ext::make_shared<PiecewiseYoYInflationCurve<Linear>>(
            today, curveBase, baseRate, freq, dc, helpers);

        std::cout << "    \"curve_base_serial\": " << curve->baseDate().serialNumber() << ",\n";
        std::cout << "    \"curve_base_rate\": " << baseRate << ",\n";

        for (const auto& h : helpers) {
            h->setTermStructure(curve.get());
        }
        (void) curve->yoyRate(cal.advance(today, maturities[0]), false);

        std::cout << "    \"implied_quotes\": [";
        for (size_t i = 0; i < helpers.size(); ++i) {
            Real iq = helpers[i]->impliedQuote();
            std::cout << iq << (i + 1 == helpers.size() ? "" : ", ");
        }
        std::cout << "]\n";

        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
