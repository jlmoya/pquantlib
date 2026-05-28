// L5-C cluster probe — Monte Carlo framework + simple MC engines.
//
// Captures reference values for the L5-C Monte Carlo layer:
//
//   * BrownianBridge(steps=4 on regular grid t in [0,1]) — exposes
//     stdDeviation / leftWeight / rightWeight / leftIndex / rightIndex
//     / bridgeIndex / size.
//   * Path / MultiPath structural inspectors (timeGrid.size, length,
//     etc.).
//   * MCEuropeanEngine (PseudoRandom = MersenneTwisterUniform +
//     InverseCumulativeNormal box-muller? actually MT19937 →
//     InverseCumulativeNormal) at fixed seed=42, samples=10000,
//     steps=1 → call/put NPV + errorEstimate. The MT-driven inverse-
//     cumulative-normal in QuantLib produces stable values across
//     architectures.
//   * Antithetic variance reduction: enable antithetic, samples=10000 —
//     errorEstimate should drop (~30% in textbook scenarios).
//   * AnalyticDiscreteGeometricAveragePriceAsianEngine call/put on
//     6 monthly fixings + 1y maturity.
//   * MCDiscreteArithmeticAPEngine (no CV, no antithetic) +
//     (CV=on, antithetic=on).
//
// All MC values are LOOSE-tier (Monte Carlo sampling). The seed
// generators are deterministic, but the values depend on the precise
// MT sequence + path-generator ordering, so we record them rather
// than computing tolerances against analytic.
//
// C++ parity:
//   ql/methods/montecarlo/brownianbridge.{hpp,cpp},
//   ql/methods/montecarlo/path.hpp,
//   ql/methods/montecarlo/pathgenerator.hpp,
//   ql/methods/montecarlo/multipath.hpp,
//   ql/methods/montecarlo/multipathgenerator.hpp,
//   ql/methods/montecarlo/pathpricer.hpp,
//   ql/methods/montecarlo/montecarlomodel.hpp,
//   ql/pricingengines/mcsimulation.hpp,
//   ql/pricingengines/vanilla/mcvanillaengine.hpp,
//   ql/pricingengines/vanilla/mceuropeanengine.hpp,
//   ql/pricingengines/asian/analytic_discr_geom_av_price.{hpp,cpp},
//   ql/pricingengines/asian/mc_discr_arith_av_price.{hpp,cpp},
//   ql/pricingengines/asian/mc_discr_geom_av_price.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/methods/montecarlo/brownianbridge.hpp>
#include <ql/pricingengines/asian/analytic_discr_geom_av_price.hpp>
#include <ql/pricingengines/asian/mc_discr_arith_av_price.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/mceuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

void emitArray(const std::vector<Real>& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

void emitSizes(const std::vector<Size>& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // -----------------------------------------------------------------
    // BrownianBridge on a regular grid with 4 steps, t in [0, 1]
    // -----------------------------------------------------------------
    {
        TimeGrid grid(1.0, 4);
        BrownianBridge bb(grid);
        std::cout << "  \"brownian_bridge_4steps\": {\n";
        std::cout << "    \"size\": " << bb.size() << ",\n";
        std::cout << "    \"times\": ";
        emitArray(bb.times());
        std::cout << ",\n";
        std::cout << "    \"std_deviation\": ";
        emitArray(bb.stdDeviation());
        std::cout << ",\n";
        std::cout << "    \"left_weight\": ";
        emitArray(bb.leftWeight());
        std::cout << ",\n";
        std::cout << "    \"right_weight\": ";
        emitArray(bb.rightWeight());
        std::cout << ",\n";
        std::cout << "    \"bridge_index\": ";
        emitSizes(bb.bridgeIndex());
        std::cout << ",\n";
        std::cout << "    \"left_index\": ";
        emitSizes(bb.leftIndex());
        std::cout << ",\n";
        std::cout << "    \"right_index\": ";
        emitSizes(bb.rightIndex());
        std::cout << "\n";
        std::cout << "  },\n";

        // Transform a fixed all-zeros sequence — should yield all-zero
        // bridge (no variation). Then transform a unit-impulse sequence
        // (first element = 1, rest = 0) — exposes the deterministic
        // "global step" path.
        std::vector<Real> input(bb.size(), 0.0);
        std::vector<Real> output(bb.size(), 0.0);
        bb.transform(input.begin(), input.end(), output.begin());
        std::cout << "  \"brownian_bridge_transform_zeros\": ";
        emitArray(output);
        std::cout << ",\n";

        input[0] = 1.0;
        bb.transform(input.begin(), input.end(), output.begin());
        std::cout << "  \"brownian_bridge_transform_impulse\": ";
        emitArray(output);
        std::cout << ",\n";
    }

    // -----------------------------------------------------------------
    // BSM process for MC engine probes (textbook scenario)
    // S=100, K=100, T=1y, r=5%, q=0%, sigma=20% — BSM call ~10.45.
    // -----------------------------------------------------------------
    Date today(15, May, 2026);
    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date expiry = today + 365;

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> rf(
        ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<YieldTermStructure> q(
        ext::make_shared<FlatForward>(today, 0.0, dc));
    Handle<BlackVolTermStructure> vol(
        ext::make_shared<BlackConstantVol>(today, cal, 0.20, dc));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, q, rf, vol);

    auto callPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    auto putPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0);
    auto exercise = ext::make_shared<EuropeanExercise>(expiry);

    // -----------------------------------------------------------------
    // AnalyticEuropeanEngine — reference NPV for cross-check
    // -----------------------------------------------------------------
    {
        VanillaOption callOpt(callPayoff, exercise);
        callOpt.setPricingEngine(
            ext::make_shared<AnalyticEuropeanEngine>(process));
        VanillaOption putOpt(putPayoff, exercise);
        putOpt.setPricingEngine(
            ext::make_shared<AnalyticEuropeanEngine>(process));
        std::cout << "  \"analytic_european\": {\n";
        std::cout << "    \"call_npv\": " << callOpt.NPV() << ",\n";
        std::cout << "    \"put_npv\": " << putOpt.NPV() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // MCEuropeanEngine — PseudoRandom (MT19937 + InverseCumulativeNormal),
    // samples=10000, steps=1, seed=42.
    // No antithetic.
    // -----------------------------------------------------------------
    {
        VanillaOption callOpt(callPayoff, exercise);
        callOpt.setPricingEngine(
            MakeMCEuropeanEngine<PseudoRandom>(process)
                .withSteps(1)
                .withSamples(10000)
                .withSeed(42));
        VanillaOption putOpt(putPayoff, exercise);
        putOpt.setPricingEngine(
            MakeMCEuropeanEngine<PseudoRandom>(process)
                .withSteps(1)
                .withSamples(10000)
                .withSeed(42));
        std::cout << "  \"mc_european_pseudo_seed42_n10000_steps1\": {\n";
        std::cout << "    \"call_npv\": " << callOpt.NPV() << ",\n";
        std::cout << "    \"call_error\": " << callOpt.errorEstimate() << ",\n";
        std::cout << "    \"put_npv\": " << putOpt.NPV() << ",\n";
        std::cout << "    \"put_error\": " << putOpt.errorEstimate() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // MCEuropeanEngine with antithetic — same seed, expect lower error.
    // -----------------------------------------------------------------
    {
        VanillaOption callOpt(callPayoff, exercise);
        callOpt.setPricingEngine(
            MakeMCEuropeanEngine<PseudoRandom>(process)
                .withSteps(1)
                .withSamples(10000)
                .withSeed(42)
                .withAntitheticVariate(true));
        std::cout << "  \"mc_european_antithetic_seed42_n10000_steps1\": {\n";
        std::cout << "    \"call_npv\": " << callOpt.NPV() << ",\n";
        std::cout << "    \"call_error\": " << callOpt.errorEstimate() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // MCEuropeanEngine — multi-step path (steps=12) at seed=42.
    // Tests that path-stepping yields the same terminal distribution.
    // -----------------------------------------------------------------
    {
        VanillaOption callOpt(callPayoff, exercise);
        callOpt.setPricingEngine(
            MakeMCEuropeanEngine<PseudoRandom>(process)
                .withSteps(12)
                .withSamples(10000)
                .withSeed(42));
        std::cout << "  \"mc_european_pseudo_seed42_n10000_steps12\": {\n";
        std::cout << "    \"call_npv\": " << callOpt.NPV() << ",\n";
        std::cout << "    \"call_error\": " << callOpt.errorEstimate() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // Discrete-average Asian — 12 monthly fixings, expiry 1y
    // -----------------------------------------------------------------
    {
        std::vector<Date> fixings;
        for (int m = 1; m <= 12; ++m) {
            fixings.push_back(today + m * 30);
        }
        Date asianExpiry = fixings.back();
        auto asianExercise =
            ext::make_shared<EuropeanExercise>(asianExpiry);

        // Analytic geometric (used as control variate for MC arith)
        DiscreteAveragingAsianOption analyticGeom(
            Average::Geometric, 1.0, 0, fixings,
            callPayoff, asianExercise);
        analyticGeom.setPricingEngine(
            ext::make_shared<
                AnalyticDiscreteGeometricAveragePriceAsianEngine>(process));

        std::cout << "  \"analytic_discr_geom_av_call\": {\n";
        std::cout << "    \"npv\": " << analyticGeom.NPV() << "\n";
        std::cout << "  },\n";

        // MC arithmetic (no CV)
        DiscreteAveragingAsianOption mcArith1(
            Average::Arithmetic, 0.0, 0, fixings,
            callPayoff, asianExercise);
        mcArith1.setPricingEngine(
            MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                .withSamples(10000)
                .withSeed(42)
                .withBrownianBridge(false));
        std::cout << "  \"mc_discr_arith_av_call_no_cv\": {\n";
        std::cout << "    \"npv\": " << mcArith1.NPV() << ",\n";
        std::cout << "    \"error\": " << mcArith1.errorEstimate() << "\n";
        std::cout << "  },\n";

        // MC arithmetic (with CV via analytic geometric)
        DiscreteAveragingAsianOption mcArith2(
            Average::Arithmetic, 0.0, 0, fixings,
            callPayoff, asianExercise);
        mcArith2.setPricingEngine(
            MakeMCDiscreteArithmeticAPEngine<PseudoRandom>(process)
                .withSamples(10000)
                .withSeed(42)
                .withControlVariate(true)
                .withBrownianBridge(false));
        std::cout << "  \"mc_discr_arith_av_call_with_cv\": {\n";
        std::cout << "    \"npv\": " << mcArith2.NPV() << ",\n";
        std::cout << "    \"error\": " << mcArith2.errorEstimate() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
