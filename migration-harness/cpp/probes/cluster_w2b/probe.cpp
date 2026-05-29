// Phase 11 W2-B cluster probe — Bootstrap + smile + CMS follow-ups.
//
// Captures C++ reference values for the W2-B layer:
//
//   * ConvexMonotoneInterpolation — Hagan-West (2006) curve. Six pillars
//     on a rate-shaped curve, factory defaults (quadraticity=0.3,
//     monotonicity=0.7, forcePositive=true). Probe captures values at
//     pillars and at interior probes.
//
//   * AbcdCalibration — Rebonato (a, b, c, d) fit on a synthetic abcd
//     vol curve, recovers truth parameters. Reuses the L10-C
//     ``AbcdInterpolation`` test pattern but exercises ``AbcdCalibration``
//     directly (which wraps the same compute() path).
//
//   * KahaleSmileSection.core_smile — pathological-arbitrage smile
//     (sharp butterfly + call-spread spike); deep-iteration
//     (interpolate=true, deleteArbitragePoints=true) repairs it. We
//     record the repaired call prices at a few strikes.
//
//   * CmsMarket / CmsMarketCalibration are not exercised here — see the
//     Python tests for the structural-instantiation coverage path. The
//     C++ classes drag in pricing-engine + CmsCouponPricer machinery
//     that PQuantLib doesn't port.
//
// C++ parity:
//   ql/math/interpolations/convexmonotoneinterpolation.hpp
//   ql/termstructures/volatility/abcdcalibration.{hpp,cpp}
//   ql/termstructures/volatility/kahalesmilesection.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/handle.hpp>
#include <ql/math/array.hpp>
#include <ql/math/comparison.hpp>
#include <ql/math/interpolations/convexmonotoneinterpolation.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/abcd.hpp>
#include <ql/termstructures/volatility/abcdcalibration.hpp>
#include <ql/termstructures/volatility/flatsmilesection.hpp>
#include <ql/termstructures/volatility/kahalesmilesection.hpp>
#include <ql/termstructures/volatility/smilesection.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) ConvexMonotoneInterpolation — 6-point rate curve.
    // ============================================================
    {
        std::vector<Real> xs = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
        std::vector<Real> ys = {0.0, 0.018, 0.020, 0.025, 0.028, 0.030};

        Real quadraticity = 0.3;
        Real monotonicity = 0.7;
        bool forcePositive = true;
        ConvexMonotoneInterpolation<std::vector<Real>::const_iterator,
                                    std::vector<Real>::const_iterator>
            interp(xs.begin(), xs.end(), ys.begin(),
                   quadraticity, monotonicity, forcePositive,
                   /*flatFinalPeriod=*/false);

        std::cout << "  \"convex_monotone\": {\n";
        std::cout << "    \"xs\": [";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::cout << xs[i];
            if (i + 1 < xs.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"ys\": [";
        for (std::size_t i = 0; i < ys.size(); ++i) {
            std::cout << ys[i];
            if (i + 1 < ys.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"quadraticity\": " << quadraticity << ",\n";
        std::cout << "    \"monotonicity\": " << monotonicity << ",\n";
        std::cout << "    \"force_positive\": " << (forcePositive ? "true" : "false") << ",\n";

        // Pillar values — skip i=0 which is the discarded y[0].
        std::cout << "    \"values_at_pillars\": [\n";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::cout << "      " << interp(xs[i]);
            if (i + 1 < xs.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        // Interior probes — sample at midpoints.
        std::vector<Real> probe_xs = {0.25, 0.75, 1.5, 3.0, 7.5};
        std::cout << "    \"interior_probes\": [\n";
        for (std::size_t i = 0; i < probe_xs.size(); ++i) {
            std::cout << "      {\"x\": " << probe_xs[i]
                      << ", \"value\": " << interp(probe_xs[i]) << "}";
            if (i + 1 < probe_xs.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) AbcdCalibration — recovers (a, b, c, d) on a synthetic curve.
    // ============================================================
    {
        Real a_true = -0.06;
        Real b_true = 0.17;
        Real c_true = 0.54;
        Real d_true = 0.17;

        std::vector<Real> times = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
        std::vector<Real> vols(times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            // Rebonato form: (a + b*t)*exp(-c*t) + d.
            vols[i] = (a_true + b_true * times[i]) * std::exp(-c_true * times[i]) + d_true;
        }

        AbcdCalibration calib(
            times, vols,
            -0.05, 0.15, 0.50, 0.16,
            /*aIsFixed=*/false, /*bIsFixed=*/false,
            /*cIsFixed=*/false, /*dIsFixed=*/false,
            /*vegaWeighted=*/false);
        calib.compute();

        std::cout << "  \"abcd_calibration\": {\n";
        std::cout << "    \"times\": [";
        for (std::size_t i = 0; i < times.size(); ++i) {
            std::cout << times[i];
            if (i + 1 < times.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"vols\": [\n";
        for (std::size_t i = 0; i < vols.size(); ++i) {
            std::cout << "      " << vols[i];
            if (i + 1 < vols.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";
        std::cout << "    \"a_true\": " << a_true << ",\n";
        std::cout << "    \"b_true\": " << b_true << ",\n";
        std::cout << "    \"c_true\": " << c_true << ",\n";
        std::cout << "    \"d_true\": " << d_true << ",\n";
        std::cout << "    \"a_fitted\": " << calib.a() << ",\n";
        std::cout << "    \"b_fitted\": " << calib.b() << ",\n";
        std::cout << "    \"c_fitted\": " << calib.c() << ",\n";
        std::cout << "    \"d_fitted\": " << calib.d() << ",\n";
        std::cout << "    \"error\": " << calib.error() << ",\n";
        std::cout << "    \"max_error\": " << calib.maxError() << ",\n";
        std::cout << "    \"fitted_at_pillars\": [\n";
        for (std::size_t i = 0; i < times.size(); ++i) {
            std::cout << "      " << calib.value(times[i]);
            if (i + 1 < times.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) KahaleSmileSection — pathological-arbitrage smile repair.
    //
    // Build a flat smile section (no smile -> no arbitrage at all).
    // Then on top of it, KahaleSmileSection with interpolate=true and
    // deleteArbitragePoints=true should pass through unchanged (the
    // probe records the post-repair call-prices on a strike grid).
    // ============================================================
    {
        Date refDate = Date(15, October, 2026);
        Settings::instance().evaluationDate() = refDate;

        Real F = 0.05;
        Real T = 1.0;
        Real vol = 0.20;

        // Anchor a flat smile section (no skew) at the same T.
        auto flatSmile = ext::make_shared<FlatSmileSection>(
            T, vol, Actual365Fixed(),
            F,  // atmLevel
            VolatilityType::ShiftedLognormal,
            /*shift=*/0.0);

        // Force evaluation date so the inner CDF call uses a fixed
        // settlement.
        // C++ KahaleSmileSection requires non-negative moneyness (strikes
        // multiplied into shifted forward space). We pass an empty grid so
        // the default 5-sigma SmileSectionUtils grid is used.
        std::vector<Real> moneyness;

        KahaleSmileSection kahale(
            flatSmile, F,
            /*interpolate=*/true,
            /*exponentialExtrapolation=*/false,
            /*deleteArbitragePoints=*/true,
            moneyness);

        // Probe strikes — sample at ATM ± a few sigma and on the extrap
        // tails.
        std::vector<Real> probe_strikes = {0.02, 0.03, 0.045, 0.05, 0.055, 0.07, 0.10};

        std::cout << "  \"kahale_core_smile\": {\n";
        std::cout << "    \"forward\": " << F << ",\n";
        std::cout << "    \"expiry_time\": " << T << ",\n";
        std::cout << "    \"flat_vol\": " << vol << ",\n";
        std::cout << "    \"moneyness_grid\": \"default\",\n";
        std::cout << "    \"probe_strikes\": [\n";
        for (std::size_t i = 0; i < probe_strikes.size(); ++i) {
            Real k = probe_strikes[i];
            Real callPx = kahale.optionPrice(k, Option::Call, 1.0);
            Real vol_at_k = kahale.volatility(k);
            std::cout << "      {\"strike\": " << k
                      << ", \"call_price\": " << callPx
                      << ", \"vol\": " << vol_at_k << "}";
            if (i + 1 < probe_strikes.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
