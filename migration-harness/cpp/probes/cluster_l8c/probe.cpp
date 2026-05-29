// L8-C cluster probe: capfloor + optionlet + swaption vol surfaces.
//
// Captures reference values for:
//
//   * ConstantCapFloorTermVolatility.volatility(d, K):
//     - constant 18% return regardless of date / strike.
//
//   * CapFloorTermVolCurve.volatility(t, K):
//     - quote at a known maturity (linear interp identity at node).
//     - intermediate maturity (linear interp on times).
//
//   * CapFloorTermVolSurface.volatility(t, K):
//     - exact value at a grid (maturity, strike) node.
//     - intermediate strike at a node maturity (bilinear: x = strike,
//       y = time).
//
//   * ConstantOptionletVolatility.volatility / blackVariance:
//     - constant vol, time-scaled variance.
//
//   * ConstantSwaptionVolatility.volatility(expiry, tenor, K):
//     - constant vol.
//     - blackVariance(t, T, K) = vol^2 * t.
//
//   * SwaptionVolatilityMatrix.volatility(expiry, tenor, K):
//     - grid pillar values at known (option, swap-tenor).
//     - intermediate point (linear in time, swap length).
//
//   * OptionletStripper1:
//     - on a flat 18% lognormal CapFloorTermVolSurface (single strike
//       column) + Euribor3M, the stripped caplet vols should each be
//       close to 18% — they reproduce the cap NPV under the same flat
//       vol. We probe the 1y / 2y / 3y caplet vols.
//
// C++ parity:
//   ql/termstructures/volatility/capfloor/{constantcapfloortermvol,
//     capfloortermvolcurve,capfloortermvolsurface}.{hpp,cpp},
//   ql/termstructures/volatility/optionlet/{constantoptionletvol,
//     optionletstripper1}.{hpp,cpp},
//   ql/termstructures/volatility/swaption/{swaptionconstantvol,
//     swaptionvolmatrix}.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/cashflows/iborcoupon.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/capfloor.hpp>
#include <ql/instruments/makecapfloor.hpp>
#include <ql/pricingengines/capfloor/blackcapfloorengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/capfloor/capfloortermvolcurve.hpp>
#include <ql/termstructures/volatility/capfloor/capfloortermvolsurface.hpp>
#include <ql/termstructures/volatility/capfloor/constantcapfloortermvol.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/volatility/optionlet/optionletstripper1.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/volatility/swaption/swaptionvolmatrix.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);

    Date evalDate(15, January, 2024);
    Settings::instance().evaluationDate() = evalDate;

    Calendar cal = TARGET();
    DayCounter dc = Actual365Fixed();
    BusinessDayConvention bdc = ModifiedFollowing;

    std::cout << "{\n";
    std::cout << "  \"setup\": {\n";
    std::cout << "    \"eval_date_serial\": " << evalDate.serialNumber() << ",\n";
    std::cout << "    \"calendar\": \"TARGET\",\n";
    std::cout << "    \"day_counter\": \"Actual365Fixed\"\n";
    std::cout << "  },\n";

    // ============================================================
    // 1) ConstantCapFloorTermVolatility — constant in (T, K).
    // ============================================================
    {
        Volatility vol = 0.18;
        ConstantCapFloorTermVolatility cv(evalDate, cal, bdc, vol, dc);
        Date d2y = cal.advance(evalDate, 2, Years);
        Real v_at_2y_5pct = cv.volatility(d2y, 0.05, true);
        Real v_at_2y_3pct = cv.volatility(d2y, 0.03, true);
        std::cout << "  \"constant_capfloor_term_vol\": {\n";
        std::cout << "    \"vol\": 0.18,\n";
        std::cout << "    \"v_at_2y_5pct\": " << v_at_2y_5pct << ",\n";
        std::cout << "    \"v_at_2y_3pct\": " << v_at_2y_3pct << ",\n";
        std::cout << "    \"max_date_serial\": " << cv.maxDate().serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) CapFloorTermVolCurve — 1-D linear interp over maturities.
    // ============================================================
    {
        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                       Period(3, Years), Period(5, Years)};
        std::vector<Volatility> vols = {0.20, 0.18, 0.16, 0.15};
        CapFloorTermVolCurve curve(evalDate, cal, bdc, tenors, vols, dc);

        // Force calculation
        Date d2y = cal.advance(evalDate, 2, Years);
        Real v_at_2y = curve.volatility(d2y, 0.05, true);  // expect ~0.18 (node)

        // Intermediate maturity — between 2y (vol=0.18) and 3y (vol=0.16).
        // The interp is on times, so we ask the curve at the actual
        // 2.5y point and reproduce the line t->vol.
        Date d2_5y = cal.advance(evalDate, Period(30, Months));
        Real v_at_2_5y = curve.volatility(d2_5y, 0.05, true);

        std::cout << "  \"capfloor_term_vol_curve\": {\n";
        std::cout << "    \"tenor_years\": [1, 2, 3, 5],\n";
        std::cout << "    \"vols\": [0.20, 0.18, 0.16, 0.15],\n";
        std::cout << "    \"v_at_2y\": " << v_at_2y << ",\n";
        std::cout << "    \"v_at_2_5y\": " << v_at_2_5y << ",\n";
        std::cout << "    \"d2y_serial\": " << d2y.serialNumber() << ",\n";
        std::cout << "    \"d2_5y_serial\": " << d2_5y.serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) CapFloorTermVolSurface — 2-D bilinear interp over (T, K).
    // ============================================================
    {
        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                       Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        // C++ Matrix: rows = tenors, columns = strikes.
        Matrix vols(4, 3);
        vols[0][0] = 0.22; vols[0][1] = 0.20; vols[0][2] = 0.18;
        vols[1][0] = 0.20; vols[1][1] = 0.18; vols[1][2] = 0.16;
        vols[2][0] = 0.18; vols[2][1] = 0.16; vols[2][2] = 0.14;
        vols[3][0] = 0.16; vols[3][1] = 0.15; vols[3][2] = 0.13;
        CapFloorTermVolSurface surface(evalDate, cal, bdc, tenors, strikes, vols, dc);

        Date d2y = cal.advance(evalDate, 2, Years);
        Date d3y = cal.advance(evalDate, 3, Years);
        Real v_2y_4pct = surface.volatility(d2y, 0.04, true);   // expect 0.18 (node)
        Real v_3y_2pct = surface.volatility(d3y, 0.02, true);   // expect 0.18 (node)
        Real v_2y_3pct = surface.volatility(d2y, 0.03, true);   // strike intermediate
        Real v_2_5y_4pct = surface.volatility(cal.advance(evalDate, Period(30, Months)),
                                              0.04, true);

        std::cout << "  \"capfloor_term_vol_surface\": {\n";
        std::cout << "    \"v_2y_4pct\": " << v_2y_4pct << ",\n";
        std::cout << "    \"v_3y_2pct\": " << v_3y_2pct << ",\n";
        std::cout << "    \"v_2y_3pct\": " << v_2y_3pct << ",\n";
        std::cout << "    \"v_2_5y_4pct\": " << v_2_5y_4pct << ",\n";
        std::cout << "    \"d2y_serial\": " << d2y.serialNumber() << ",\n";
        std::cout << "    \"d3y_serial\": " << d3y.serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 4) ConstantOptionletVolatility.
    // ============================================================
    {
        Volatility vol = 0.20;
        ConstantOptionletVolatility cv(evalDate, cal, bdc, vol, dc);
        Date d2y = cal.advance(evalDate, 2, Years);
        Real v = cv.volatility(d2y, 0.04, true);
        Real bv = cv.blackVariance(d2y, 0.04, true);
        Time t = dc.yearFraction(evalDate, d2y);
        std::cout << "  \"constant_optionlet_vol\": {\n";
        std::cout << "    \"vol\": 0.20,\n";
        std::cout << "    \"v\": " << v << ",\n";
        std::cout << "    \"black_variance\": " << bv << ",\n";
        std::cout << "    \"t_2y\": " << t << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 5) ConstantSwaptionVolatility.
    // ============================================================
    {
        Volatility vol = 0.20;
        ConstantSwaptionVolatility cv(evalDate, cal, bdc, vol, dc);
        Date d2y = cal.advance(evalDate, 2, Years);
        Real v = cv.volatility(d2y, Period(5, Years), 0.04, true);
        Real bv = cv.blackVariance(d2y, Period(5, Years), 0.04, true);
        Time t = dc.yearFraction(evalDate, d2y);
        std::cout << "  \"constant_swaption_vol\": {\n";
        std::cout << "    \"vol\": 0.20,\n";
        std::cout << "    \"v\": " << v << ",\n";
        std::cout << "    \"black_variance\": " << bv << ",\n";
        std::cout << "    \"t_2y\": " << t << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 6) SwaptionVolatilityMatrix.
    // ============================================================
    {
        std::vector<Period> optionTenors = {Period(1, Years), Period(2, Years),
                                             Period(5, Years), Period(10, Years)};
        std::vector<Period> swapTenors = {Period(1, Years), Period(5, Years),
                                           Period(10, Years)};
        // Matrix: rows = option tenor, columns = swap tenor.
        Matrix vols(4, 3);
        vols[0][0] = 0.30; vols[0][1] = 0.28; vols[0][2] = 0.25;
        vols[1][0] = 0.28; vols[1][1] = 0.25; vols[1][2] = 0.22;
        vols[2][0] = 0.25; vols[2][1] = 0.22; vols[2][2] = 0.20;
        vols[3][0] = 0.22; vols[3][1] = 0.20; vols[3][2] = 0.18;
        SwaptionVolatilityMatrix matrix(evalDate, cal, bdc, optionTenors,
                                         swapTenors, vols, dc);

        Date d2y = cal.advance(evalDate, 2, Years);
        Date d5y = cal.advance(evalDate, 5, Years);
        Real v_2y_5y = matrix.volatility(d2y, Period(5, Years), 0.04, true);  // 0.25 node
        Real v_5y_10y = matrix.volatility(d5y, Period(10, Years), 0.04, true); // 0.20 node
        // Intermediate.
        Date d3y = cal.advance(evalDate, 3, Years);
        Real v_3y_5y = matrix.volatility(d3y, Period(5, Years), 0.04, true);
        std::cout << "  \"swaption_vol_matrix\": {\n";
        std::cout << "    \"v_2y_5y\": " << v_2y_5y << ",\n";
        std::cout << "    \"v_5y_10y\": " << v_5y_10y << ",\n";
        std::cout << "    \"v_3y_5y\": " << v_3y_5y << ",\n";
        std::cout << "    \"d2y_serial\": " << d2y.serialNumber() << ",\n";
        std::cout << "    \"d3y_serial\": " << d3y.serialNumber() << ",\n";
        std::cout << "    \"d5y_serial\": " << d5y.serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 7) OptionletStripper1 — flat 18% lognormal cap term vol surface +
    //    Euribor3M => stripped caplet vols ~ 18%.
    //
    // Use 3 strikes so the strike-axis interpolation inside
    // OptionletStripperAdapter is valid even though we only test the
    // middle column. The adapter requires ≥ 2 strike pillars to build
    // its per-time linear strike-interp.
    // ============================================================
    {
        // Curve for fwd discounts.
        Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.03));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(evalDate, rateQuote, Actual365Fixed()));
        auto euribor = ext::make_shared<Euribor3M>(curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                       Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.18);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            evalDate, cal, bdc, tenors, strikes, vols, dc);

        OptionletStripper1 stripper(surface, euribor);
        stripper.optionletStrikes(0);  // force performCalculations
        const auto& times = stripper.optionletFixingTimes();
        // We expose just the middle-column (0.04) stripped vols.
        std::vector<Real> capletVols;
        for (Size i = 0; i < times.size(); ++i) {
            // optionletVolatilities(i) returns the vols across strikes
            // at the i-th option time. Index 1 is the 0.04 strike.
            capletVols.push_back(stripper.optionletVolatilities(i)[1]);
        }
        std::cout << "  \"optionlet_stripper_1\": {\n";
        std::cout << "    \"input_cap_vol\": 0.18,\n";
        std::cout << "    \"strikes\": [0.02, 0.04, 0.06],\n";
        std::cout << "    \"probe_strike\": 0.04,\n";
        std::cout << "    \"n_fixings\": " << times.size() << ",\n";
        std::cout << "    \"times\": [";
        for (Size i = 0; i < times.size(); ++i) {
            std::cout << times[i];
            if (i + 1 < times.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"caplet_vols_04\": [";
        for (Size i = 0; i < capletVols.size(); ++i) {
            std::cout << capletVols[i];
            if (i + 1 < capletVols.size()) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
