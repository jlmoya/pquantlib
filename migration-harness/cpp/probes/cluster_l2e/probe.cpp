// L2-E cluster probe: volatility term structures
//
// Captures reference values for the equity/FX-side volatility term
// structures needed by Phase-3 vanilla European-option pricing:
//
//   * SmileSection / FlatSmileSection
//   * BlackVolTermStructure (abstract) — exercised via concrete leaves
//   * BlackConstantVol
//   * BlackVarianceCurve (linear interp on variance)
//   * BlackVarianceSurface (bilinear interp on variance)
//   * LocalVolTermStructure (abstract) — exercised via concrete leaves
//   * LocalConstantVol
//   * LocalVolCurve (Dupire on a constant Black vol)
//   * LocalVolSurface (Dupire on a BlackVarianceSurface, with zero rates
//     / zero dividends — i.e. forward = spot)
//
// C++ parity: ql/termstructures/voltermstructure.{hpp,cpp},
//             ql/termstructures/volatility/smilesection.{hpp,cpp},
//             ql/termstructures/volatility/flatsmilesection.{hpp,cpp},
//             ql/termstructures/volatility/equityfx/*.{hpp,cpp}
//             @ v1.42.1 (099987f0).

#include <ql/termstructures/volatility/smilesection.hpp>
#include <ql/termstructures/volatility/flatsmilesection.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancesurface.hpp>
#include <ql/termstructures/volatility/equityfx/localconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/localvolcurve.hpp>
#include <ql/termstructures/volatility/equityfx/localvolsurface.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/handle.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date ref(15, June, 2026);

    // ---------------------------------------------------------------
    // FlatSmileSection (time-anchored constructor)
    // ---------------------------------------------------------------
    {
        Date exercise(15, June, 2027);  // 1 year out under Actual/365 Fixed
        FlatSmileSection smile(exercise, /*vol=*/0.20, dc, ref, /*atmLevel=*/100.0);

        std::cout << "  \"flat_smile_section\": {\n";
        std::cout << "    \"exercise_time\": " << smile.exerciseTime() << ",\n";
        std::cout << "    \"atm_level\": " << smile.atmLevel() << ",\n";
        std::cout << "    \"shift\": " << smile.shift() << ",\n";
        std::cout << "    \"vol_at_strike_100\": " << smile.volatility(100.0) << ",\n";
        std::cout << "    \"vol_at_strike_120\": " << smile.volatility(120.0) << ",\n";
        std::cout << "    \"variance_at_strike_100\": " << smile.variance(100.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BlackConstantVol
    // ---------------------------------------------------------------
    {
        BlackConstantVol bcv(ref, cal, 0.20, dc);
        std::cout << "  \"black_constant_vol\": {\n";
        std::cout << "    \"reference_date_serial\": " << ref.serialNumber() << ",\n";
        std::cout << "    \"vol_t1\": " << bcv.blackVol(1.0, 100.0) << ",\n";
        std::cout << "    \"vol_t2\": " << bcv.blackVol(2.0, 90.0) << ",\n";
        std::cout << "    \"variance_t1\": " << bcv.blackVariance(1.0, 100.0) << ",\n";
        std::cout << "    \"variance_t2\": " << bcv.blackVariance(2.0, 90.0) << ",\n";

        // black_forward_vol with non-zero variances (constant vol → exact .20)
        std::cout << "    \"forward_vol_t1_t2\": "
                  << bcv.blackForwardVol(1.0, 2.0, 100.0) << ",\n";
        std::cout << "    \"forward_variance_t1_t2\": "
                  << bcv.blackForwardVariance(1.0, 2.0, 100.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BlackConstantVol via Handle<Quote>
    // ---------------------------------------------------------------
    {
        auto q = ext::make_shared<SimpleQuote>(0.25);
        Handle<Quote> h(q);
        BlackConstantVol bcv(ref, cal, h, dc);
        std::cout << "  \"black_constant_vol_via_quote\": {\n";
        std::cout << "    \"initial_vol\": " << bcv.blackVol(1.0, 100.0) << ",\n";
        q->setValue(0.30);
        std::cout << "    \"after_update_vol\": " << bcv.blackVol(1.0, 100.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BlackVarianceCurve
    //   Dates: ref+3mo, ref+6mo, ref+1y, ref+2y
    //   Vols:  0.10,    0.15,    0.20,   0.25 (annualized Black vols)
    //   Variance points: t * vol^2
    // ---------------------------------------------------------------
    {
        std::vector<Date> dates = {
            Date(15, September, 2026),
            Date(15, December, 2026),
            Date(15, June, 2027),
            Date(15, June, 2028)
        };
        std::vector<Volatility> vols = {0.10, 0.15, 0.20, 0.25};
        BlackVarianceCurve curve(ref, dates, vols, dc, /*forceMonotoneVariance=*/true);

        std::cout << "  \"black_variance_curve\": {\n";

        // Emit times for each pillar so the Python test can replay them.
        std::cout << "    \"times\": [";
        for (size_t i = 0; i < dates.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << dc.yearFraction(ref, dates[i]);
        }
        std::cout << "],\n";

        // Emit variances at the pillars.
        std::cout << "    \"variances_at_pillars\": [";
        for (size_t i = 0; i < dates.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << curve.blackVariance(dates[i], 100.0);
        }
        std::cout << "],\n";

        // Interpolated point at t=9mo (between 6mo and 1y).
        Date t9mo(15, March, 2027);
        std::cout << "    \"variance_at_9mo\": " << curve.blackVariance(t9mo, 100.0) << ",\n";
        std::cout << "    \"vol_at_9mo\": " << curve.blackVol(t9mo, 100.0) << ",\n";

        // Strike-independent: same value for any strike.
        std::cout << "    \"variance_at_9mo_any_strike\": "
                  << curve.blackVariance(t9mo, 50.0) << "\n";

        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BlackVarianceSurface
    //   Strikes: 80, 100, 120
    //   Dates:   ref+3mo, ref+6mo, ref+1y, ref+2y
    //   Vol matrix (3 strikes x 4 dates):
    //     K=80:  [0.20, 0.21, 0.22, 0.23]
    //     K=100: [0.10, 0.15, 0.20, 0.25]
    //     K=120: [0.20, 0.21, 0.22, 0.23]
    // ---------------------------------------------------------------
    {
        std::vector<Date> dates = {
            Date(15, September, 2026),
            Date(15, December, 2026),
            Date(15, June, 2027),
            Date(15, June, 2028)
        };
        std::vector<Real> strikes = {80.0, 100.0, 120.0};

        Matrix vols(3, 4);
        vols[0][0] = 0.20; vols[0][1] = 0.21; vols[0][2] = 0.22; vols[0][3] = 0.23;
        vols[1][0] = 0.10; vols[1][1] = 0.15; vols[1][2] = 0.20; vols[1][3] = 0.25;
        vols[2][0] = 0.20; vols[2][1] = 0.21; vols[2][2] = 0.22; vols[2][3] = 0.23;

        BlackVarianceSurface surf(ref, cal, dates, strikes, vols, dc);

        std::cout << "  \"black_variance_surface\": {\n";

        // Emit times for each date pillar.
        std::cout << "    \"times\": [";
        for (size_t i = 0; i < dates.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << dc.yearFraction(ref, dates[i]);
        }
        std::cout << "],\n";

        // Variance at each pillar (strike=100, the middle row).
        std::cout << "    \"variances_atm\": [";
        for (size_t i = 0; i < dates.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << surf.blackVariance(dates[i], 100.0);
        }
        std::cout << "],\n";

        // Variance at intermediate (strike=110, t=9mo).
        Date t9mo(15, March, 2027);
        std::cout << "    \"variance_at_110_9mo\": " << surf.blackVariance(t9mo, 110.0) << ",\n";
        std::cout << "    \"vol_at_110_9mo\": " << surf.blackVol(t9mo, 110.0) << ",\n";

        // Variance at pillar nodes (no interpolation): exact values.
        std::cout << "    \"variance_at_80_1y\": " << surf.blackVariance(dates[2], 80.0) << ",\n";
        std::cout << "    \"variance_at_100_2y\": " << surf.blackVariance(dates[3], 100.0) << "\n";

        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // LocalConstantVol
    // ---------------------------------------------------------------
    {
        LocalConstantVol lcv(ref, 0.20, dc);
        std::cout << "  \"local_constant_vol\": {\n";
        std::cout << "    \"vol_t1_S100\": " << lcv.localVol(1.0, 100.0) << ",\n";
        std::cout << "    \"vol_t2_S90\": " << lcv.localVol(2.0, 90.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // LocalVolCurve from BlackVarianceCurve
    //
    // local_vol(t) = sqrt(d/dt[sigma_B(t)^2 * t]) — for the variance curve
    // with linear variance interpolation, dvar/dt = (var[i+1] - var[i]) /
    // (t[i+1] - t[i]). In our curve above, between t=3mo (var=0.0025/4) and
    // t=6mo (var=0.0225/2)... well, just emit reference values.
    // ---------------------------------------------------------------
    {
        std::vector<Date> dates = {
            Date(15, September, 2026),
            Date(15, December, 2026),
            Date(15, June, 2027),
            Date(15, June, 2028)
        };
        std::vector<Volatility> vols = {0.10, 0.15, 0.20, 0.25};
        auto curve_ptr = ext::make_shared<BlackVarianceCurve>(
            ref, dates, vols, dc, /*forceMonotoneVariance=*/true);
        Handle<BlackVarianceCurve> handle(curve_ptr);

        LocalVolCurve lvc(handle);

        std::cout << "  \"local_vol_curve\": {\n";
        // local_vol at t=0.5 (6 months) on the curve.
        std::cout << "    \"local_vol_t_0p5\": " << lvc.localVol(0.5, 100.0, true) << ",\n";
        std::cout << "    \"local_vol_t_0p75\": " << lvc.localVol(0.75, 100.0, true) << ",\n";
        std::cout << "    \"local_vol_t_1p5\": " << lvc.localVol(1.5, 100.0, true) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // LocalVolSurface from BlackVarianceSurface (with zero rates / zero dividends)
    //
    // Pquantlib L2-E is the equity/FX vol cluster but YieldTermStructure
    // isn't in this cluster — we'll use FlatForward(0%) on both sides.
    // ---------------------------------------------------------------
    {
        std::vector<Date> dates = {
            Date(15, September, 2026),
            Date(15, December, 2026),
            Date(15, June, 2027),
            Date(15, June, 2028)
        };
        std::vector<Real> strikes = {80.0, 100.0, 120.0};
        Matrix vols(3, 4);
        vols[0][0] = 0.20; vols[0][1] = 0.21; vols[0][2] = 0.22; vols[0][3] = 0.23;
        vols[1][0] = 0.10; vols[1][1] = 0.15; vols[1][2] = 0.20; vols[1][3] = 0.25;
        vols[2][0] = 0.20; vols[2][1] = 0.21; vols[2][2] = 0.22; vols[2][3] = 0.23;

        auto surf_ptr = ext::make_shared<BlackVarianceSurface>(
            ref, cal, dates, strikes, vols, dc);
        Handle<BlackVolTermStructure> blackTS(surf_ptr);

        auto yield_ptr = ext::make_shared<FlatForward>(ref, 0.0, dc);
        Handle<YieldTermStructure> riskFree(yield_ptr);
        Handle<YieldTermStructure> dividend(yield_ptr);

        auto under_ptr = ext::make_shared<SimpleQuote>(100.0);
        Handle<Quote> underlying(under_ptr);

        LocalVolSurface lvs(blackTS, riskFree, dividend, underlying);
        lvs.enableExtrapolation(true);

        std::cout << "  \"local_vol_surface\": {\n";
        // local vol at (t=0.5, S=100).
        std::cout << "    \"local_vol_t_0p5_S100\": " << lvs.localVol(0.5, 100.0, true) << ",\n";
        std::cout << "    \"local_vol_t_0p75_S100\": " << lvs.localVol(0.75, 100.0, true) << ",\n";
        std::cout << "    \"local_vol_t_1p0_S100\": " << lvs.localVol(1.0, 100.0, true) << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
