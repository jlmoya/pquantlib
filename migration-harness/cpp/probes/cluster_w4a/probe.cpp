// Phase 11 W4-A cluster probe: multi-asset exotic options.
//
// Captures reference values for:
//
//   * HimalayaOption + MCHimalayaEngine — max-of-N basket, pays max
//     between strike and average of best performers per fixing date.
//
//   * EverestOption + MCEverestEngine — pays (1 + min_yield +
//     guarantee) * notional * discount.
//
//   * PagodaOption + MCPagodaEngine — roofed Asian on portfolio of N
//     assets, pays fraction * min(roof, sum-perf+).
//
//   * TwoAssetBarrierOption + AnalyticTwoAssetBarrierEngine —
//     Heynen-Kat closed form for barrier-on-S2, payoff-on-S1.
//
//   * TwoAssetCorrelationOption + AnalyticTwoAssetCorrelationEngine —
//     Zhang closed form, payoff on S2 conditional on S1 in-the-money.
//
// MC engines use PseudoRandom (MT19937 + InverseCumulativeNormal +
// Polar via MT default) with fixed seed and explicit
// withSamples(N), so the same seed should yield reproducible NPVs
// against the Python port to LOOSE tolerance.
//
// C++ parity:
//   ql/experimental/exoticoptions/himalayaoption.hpp
//   ql/experimental/exoticoptions/everestoption.hpp
//   ql/experimental/exoticoptions/pagodaoption.hpp
//   ql/experimental/exoticoptions/mchimalayaengine.hpp
//   ql/experimental/exoticoptions/mceverestengine.hpp
//   ql/experimental/exoticoptions/mcpagodaengine.hpp
//   ql/instruments/twoassetbarrieroption.hpp
//   ql/instruments/twoassetcorrelationoption.hpp
//   ql/pricingengines/barrier/analytictwoassetbarrierengine.hpp
//   ql/pricingengines/exotic/analytictwoassetcorrelationengine.hpp
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/experimental/exoticoptions/everestoption.hpp>
#include <ql/experimental/exoticoptions/himalayaoption.hpp>
#include <ql/experimental/exoticoptions/mceverestengine.hpp>
#include <ql/experimental/exoticoptions/mchimalayaengine.hpp>
#include <ql/experimental/exoticoptions/mcpagodaengine.hpp>
#include <ql/experimental/exoticoptions/pagodaoption.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/twoassetbarrieroption.hpp>
#include <ql/instruments/twoassetcorrelationoption.hpp>
#include <ql/math/matrix.hpp>
#include <ql/pricingengines/barrier/analytictwoassetbarrierengine.hpp>
#include <ql/pricingengines/exotic/analytictwoassetcorrelationengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/stochasticprocessarray.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

// Helper: build a flat GBSM process.
ext::shared_ptr<GeneralizedBlackScholesProcess>
make_bsm_process(Date today, Real spot, Rate r, Rate q, Volatility sigma, DayCounter dc) {
    Handle<Quote> spot_h(ext::make_shared<SimpleQuote>(spot));
    Handle<YieldTermStructure> r_h(ext::make_shared<FlatForward>(today, r, dc));
    Handle<YieldTermStructure> q_h(ext::make_shared<FlatForward>(today, q, dc));
    Handle<BlackVolTermStructure> v_h(ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, dc));
    return ext::make_shared<GeneralizedBlackScholesProcess>(spot_h, q_h, r_h, v_h);
}

}  // anonymous namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    DayCounter dc = Actual360();
    Calendar cal = TARGET();
    Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;

    // --- common 3-asset basket setup --------------------------------------
    // 3 assets, spots 100/95/105, vols 0.20/0.25/0.30, q=0.02, r=0.05.
    // Correlation matrix: identity-ish with weak off-diagonal so that
    // results are not degenerate but reproducible.
    const Real rate = 0.05;
    const Real div = 0.02;
    Matrix corr3(3, 3, 0.30);
    corr3[0][0] = corr3[1][1] = corr3[2][2] = 1.0;

    std::vector<ext::shared_ptr<StochasticProcess1D>> procs3;
    procs3.push_back(make_bsm_process(today, 100.0, rate, div, 0.20, dc));
    procs3.push_back(make_bsm_process(today, 95.0,  rate, div, 0.25, dc));
    procs3.push_back(make_bsm_process(today, 105.0, rate, div, 0.30, dc));

    ext::shared_ptr<StochasticProcessArray> array3 =
        ext::make_shared<StochasticProcessArray>(procs3, corr3);

    // ============================================================
    // 1) HimalayaOption + MCHimalayaEngine
    // ============================================================
    {
        // 3 fixing dates over ~1y.
        std::vector<Date> fixings;
        fixings.push_back(cal.advance(today, 4, Months));
        fixings.push_back(cal.advance(today, 8, Months));
        fixings.push_back(cal.advance(today, 12, Months));
        Real strike = 100.0;

        HimalayaOption option(fixings, strike);

        // MC pricing — fixed seed, fixed samples (small for speed).
        ext::shared_ptr<PricingEngine> mc_engine =
            MakeMCHimalayaEngine<PseudoRandom>(array3)
                .withSamples(1023)
                .withSeed(42);
        option.setPricingEngine(mc_engine);

        Real npv = option.NPV();

        std::cout << "  \"himalaya\": {\n";
        std::cout << "    \"strike\": " << strike << ",\n";
        std::cout << "    \"n_fixings\": " << fixings.size() << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) EverestOption + MCEverestEngine
    // ============================================================
    {
        Real notional = 1.0e6;
        Rate guarantee = 0.03;
        Date exercise_date = cal.advance(today, 1, Years);
        ext::shared_ptr<Exercise> ex =
            ext::make_shared<EuropeanExercise>(exercise_date);

        EverestOption option(notional, guarantee, ex);

        ext::shared_ptr<PricingEngine> mc_engine =
            MakeMCEverestEngine<PseudoRandom>(array3)
                .withStepsPerYear(12)
                .withSamples(1023)
                .withSeed(42);
        option.setPricingEngine(mc_engine);

        Real npv = option.NPV();
        Real yield = option.yield();

        std::cout << "  \"everest\": {\n";
        std::cout << "    \"notional\": " << notional << ",\n";
        std::cout << "    \"guarantee\": " << guarantee << ",\n";
        std::cout << "    \"exercise_serial\": " << exercise_date.serialNumber() << ",\n";
        std::cout << "    \"npv\": " << npv << ",\n";
        std::cout << "    \"yield\": " << yield << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) PagodaOption + MCPagodaEngine
    // ============================================================
    {
        std::vector<Date> fixings;
        fixings.push_back(cal.advance(today, 3, Months));
        fixings.push_back(cal.advance(today, 6, Months));
        fixings.push_back(cal.advance(today, 9, Months));
        fixings.push_back(cal.advance(today, 12, Months));
        Real roof = 0.20;
        Real fraction = 0.50;

        PagodaOption option(fixings, roof, fraction);

        ext::shared_ptr<PricingEngine> mc_engine =
            MakeMCPagodaEngine<PseudoRandom>(array3)
                .withSamples(1023)
                .withSeed(42);
        option.setPricingEngine(mc_engine);

        Real npv = option.NPV();

        std::cout << "  \"pagoda\": {\n";
        std::cout << "    \"roof\": " << roof << ",\n";
        std::cout << "    \"fraction\": " << fraction << ",\n";
        std::cout << "    \"n_fixings\": " << fixings.size() << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 4) TwoAssetBarrierOption + AnalyticTwoAssetBarrierEngine
    // ============================================================
    {
        // Heynen-Kat: payoff on S1 (call/put), barrier on S2 trigger.
        // Use spot1=100, spot2=110, barrier=130 (UpOut), strike=100.
        // Vol1=0.20, vol2=0.25, correlation rho=0.50, T=1y.
        Real spot1 = 100.0, spot2 = 110.0;
        Volatility v1 = 0.20, v2 = 0.25;
        Real rho = 0.50;

        ext::shared_ptr<GeneralizedBlackScholesProcess> p1 =
            make_bsm_process(today, spot1, rate, div, v1, dc);
        ext::shared_ptr<GeneralizedBlackScholesProcess> p2 =
            make_bsm_process(today, spot2, rate, div, v2, dc);

        Date exercise_date = cal.advance(today, 1, Years);
        ext::shared_ptr<Exercise> ex =
            ext::make_shared<EuropeanExercise>(exercise_date);

        // UpOut call: barrier on S2 = 130 > S2(0)=110. S1 payoff K=100.
        Real strike = 100.0;
        Real barrier_lvl = 130.0;
        ext::shared_ptr<StrikedTypePayoff> po =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);

        TwoAssetBarrierOption opt_upout_call(
            Barrier::UpOut, barrier_lvl, po, ex);

        Handle<Quote> rho_h(ext::make_shared<SimpleQuote>(rho));
        ext::shared_ptr<PricingEngine> eng =
            ext::make_shared<AnalyticTwoAssetBarrierEngine>(p1, p2, rho_h);
        opt_upout_call.setPricingEngine(eng);

        Real npv_upout_call = opt_upout_call.NPV();

        // DownOut put: barrier S2 = 90 < S2(0). put payoff K=100.
        ext::shared_ptr<StrikedTypePayoff> po2 =
            ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
        TwoAssetBarrierOption opt_downout_put(
            Barrier::DownOut, 90.0, po2, ex);
        opt_downout_put.setPricingEngine(eng);
        Real npv_downout_put = opt_downout_put.NPV();

        std::cout << "  \"two_asset_barrier\": {\n";
        std::cout << "    \"strike\": " << strike << ",\n";
        std::cout << "    \"barrier_upout\": " << barrier_lvl << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"npv_upout_call\": " << npv_upout_call << ",\n";
        std::cout << "    \"npv_downout_put\": " << npv_downout_put << ",\n";
        std::cout << "    \"exercise_serial\": " << exercise_date.serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 5) TwoAssetCorrelationOption + AnalyticTwoAssetCorrelationEngine
    // ============================================================
    {
        // Zhang: payoff on S2 conditional on S1 > X1 (call) or < X1 (put).
        Real spot1 = 100.0, spot2 = 105.0;
        Volatility v1 = 0.20, v2 = 0.30;
        Real rho = 0.40;

        ext::shared_ptr<GeneralizedBlackScholesProcess> p1 =
            make_bsm_process(today, spot1, rate, div, v1, dc);
        ext::shared_ptr<GeneralizedBlackScholesProcess> p2 =
            make_bsm_process(today, spot2, rate, div, v2, dc);

        Date exercise_date = cal.advance(today, 1, Years);
        ext::shared_ptr<Exercise> ex =
            ext::make_shared<EuropeanExercise>(exercise_date);

        Real strike1 = 100.0;
        Real strike2 = 100.0;

        TwoAssetCorrelationOption opt_call(
            Option::Call, strike1, strike2, ex);
        Handle<Quote> rho_h(ext::make_shared<SimpleQuote>(rho));
        ext::shared_ptr<PricingEngine> eng =
            ext::make_shared<AnalyticTwoAssetCorrelationEngine>(p1, p2, rho_h);
        opt_call.setPricingEngine(eng);
        Real npv_call = opt_call.NPV();

        TwoAssetCorrelationOption opt_put(
            Option::Put, strike1, strike2, ex);
        opt_put.setPricingEngine(eng);
        Real npv_put = opt_put.NPV();

        std::cout << "  \"two_asset_correlation\": {\n";
        std::cout << "    \"strike1\": " << strike1 << ",\n";
        std::cout << "    \"strike2\": " << strike2 << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"npv_call\": " << npv_call << ",\n";
        std::cout << "    \"npv_put\": " << npv_put << ",\n";
        std::cout << "    \"exercise_serial\": " << exercise_date.serialNumber() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
