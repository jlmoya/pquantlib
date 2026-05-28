// L5-E cluster probe: exotic instruments + analytic closed-form engines.
//
// Captures reference values for:
//
//   * AnalyticContinuousGeometricAveragePriceAsianEngine:
//       Continuous-averaging geometric Asian call, (S=100, K=100, T=1,
//       r=5%, q=0%, sigma=30%) under GeneralizedBlackScholesProcess.
//       Call NPV + delta + gamma + vega + rho + dividendRho + theta.
//   * AnalyticDiscreteGeometricAveragePriceAsianEngine:
//       Discrete-averaging geometric Asian call w/ 12 monthly fixings,
//       (S=100, K=100, T=1, r=5%, q=0%, sigma=30%). Call NPV + delta + gamma.
//   * AnalyticBarrierEngine — all 4 barrier types under Reiner-Rubinstein:
//       (S=100, K=100, B=95 (down) or B=105 (up), rebate=3, T=1,
//        r=5%, q=2%, sigma=30%); Call+Put each times {DownIn, UpIn,
//        DownOut, UpOut} = 8 NPVs.
//   * AnalyticBinaryBarrierEngine — Reiner-Rubinstein one-touch family:
//       Cash-or-nothing + asset-or-nothing across 8 barrier-type/option-
//       type combinations with American payoff-at-expiry.
//   * StulzEngine — 2-asset min/max basket call+put across textbook
//       Hull-style parameters (S1=100, S2=100, K=100, q1=q2=0,
//       sigma1=20%, sigma2=30%, rho=0.5, T=1, r=5%).
//   * AnalyticContinuousFloatingLookbackEngine —
//       (S=100, prior minimum/maximum=100, T=1, r=5%, q=2%, sigma=30%).
//       Floating-strike lookback Call + Put NPVs.
//
// C++ parity:
//   ql/instruments/asianoption.hpp,
//   ql/instruments/barrieroption.hpp,
//   ql/instruments/basketoption.hpp,
//   ql/instruments/lookbackoption.hpp,
//   ql/pricingengines/asian/analytic_cont_geom_av_price.{hpp,cpp},
//   ql/pricingengines/asian/analytic_discr_geom_av_price.{hpp,cpp},
//   ql/pricingengines/barrier/analyticbarrierengine.{hpp,cpp},
//   ql/pricingengines/barrier/analyticbinarybarrierengine.{hpp,cpp},
//   ql/pricingengines/basket/stulzengine.{hpp,cpp},
//   ql/pricingengines/lookback/analyticcontinuousfloatinglookback.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/averagetype.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/barriertype.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/lookbackoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/pricingengines/asian/analytic_cont_geom_av_price.hpp>
#include <ql/pricingengines/asian/analytic_discr_geom_av_price.hpp>
#include <ql/pricingengines/barrier/analyticbarrierengine.hpp>
#include <ql/pricingengines/barrier/analyticbinarybarrierengine.hpp>
#include <ql/pricingengines/basket/stulzengine.hpp>
#include <ql/pricingengines/lookback/analyticcontinuousfloatinglookback.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

ext::shared_ptr<GeneralizedBlackScholesProcess>
makeProcess(Real spot, Rate r, Rate q, Volatility sigma, const Date& ref,
            const Calendar& cal, const DayCounter& dc) {
    Handle<Quote> spotH(ext::make_shared<SimpleQuote>(spot));
    Handle<YieldTermStructure> rfH(
        ext::make_shared<FlatForward>(ref, r, dc));
    Handle<YieldTermStructure> divH(
        ext::make_shared<FlatForward>(ref, q, dc));
    Handle<BlackVolTermStructure> volH(
        ext::make_shared<BlackConstantVol>(ref, cal, sigma, dc));
    return ext::make_shared<GeneralizedBlackScholesProcess>(
        spotH, divH, rfH, volH);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);

    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date ref(15, June, 2026);
    Settings::instance().evaluationDate() = ref;
    Date expiry = ref + 365; // 1-year tenor

    std::cout << "{\n";

    // ---------------------------------------------------------------
    // Analytic continuous geometric average price Asian
    // ---------------------------------------------------------------
    {
        Real spot = 100.0, strike = 100.0;
        Volatility sigma = 0.30;
        Rate r = 0.05, q = 0.00;
        auto process = makeProcess(spot, r, q, sigma, ref, cal, dc);

        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto exercise = ext::make_shared<EuropeanExercise>(expiry);
        ContinuousAveragingAsianOption asian(Average::Geometric, payoff, exercise);
        auto engine = ext::make_shared<AnalyticContinuousGeometricAveragePriceAsianEngine>(process);
        asian.setPricingEngine(engine);

        std::cout << "  \"analytic_continuous_geometric_asian\": {\n";
        std::cout << "    \"call_npv\": " << asian.NPV() << ",\n";
        std::cout << "    \"call_delta\": " << asian.delta() << ",\n";
        std::cout << "    \"call_gamma\": " << asian.gamma() << ",\n";
        std::cout << "    \"call_vega\": " << asian.vega() << ",\n";
        std::cout << "    \"call_rho\": " << asian.rho() << ",\n";
        std::cout << "    \"call_dividend_rho\": " << asian.dividendRho() << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // Analytic discrete geometric average price Asian
    // 12 monthly fixings over 1 year. Past fixings = 0 (fresh option),
    // runningAccumulator = 1.0.
    // ---------------------------------------------------------------
    {
        Real spot = 100.0, strike = 100.0;
        Volatility sigma = 0.30;
        Rate r = 0.05, q = 0.00;
        auto process = makeProcess(spot, r, q, sigma, ref, cal, dc);

        std::vector<Date> fixingDates;
        // 12 monthly fixings — every 365/12 ≈ 30.4 days, rounded.
        for (int i = 1; i <= 12; ++i) {
            fixingDates.push_back(ref + (i * 365 / 12));
        }
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto exercise = ext::make_shared<EuropeanExercise>(expiry);
        DiscreteAveragingAsianOption asian(
            Average::Geometric, 1.0, 0u, fixingDates, payoff, exercise);
        auto engine =
            ext::make_shared<AnalyticDiscreteGeometricAveragePriceAsianEngine>(process);
        asian.setPricingEngine(engine);

        std::cout << "  \"analytic_discrete_geometric_asian\": {\n";
        std::cout << "    \"call_npv\": " << asian.NPV() << ",\n";
        std::cout << "    \"call_delta\": " << asian.delta() << ",\n";
        std::cout << "    \"call_gamma\": " << asian.gamma() << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // Analytic barrier engine - 8 combos (Call+Put × 4 BarrierType)
    // (S=100, K=100, B(down)=95, B(up)=105, rebate=3, T=1, r=5%, q=2%, sigma=30%).
    // ---------------------------------------------------------------
    {
        Real spot = 100.0, strike = 100.0;
        Real barrierDown = 95.0, barrierUp = 105.0;
        Real rebate = 3.0;
        Volatility sigma = 0.30;
        Rate r = 0.05, q = 0.02;
        auto process = makeProcess(spot, r, q, sigma, ref, cal, dc);

        auto engine = ext::make_shared<AnalyticBarrierEngine>(process);

        struct Spec { const char* key; Option::Type type; Barrier::Type bt; Real B; };
        Spec specs[] = {
            {"down_in_call",   Option::Call, Barrier::DownIn,  barrierDown},
            {"up_in_call",     Option::Call, Barrier::UpIn,    barrierUp},
            {"down_out_call",  Option::Call, Barrier::DownOut, barrierDown},
            {"up_out_call",    Option::Call, Barrier::UpOut,   barrierUp},
            {"down_in_put",    Option::Put,  Barrier::DownIn,  barrierDown},
            {"up_in_put",      Option::Put,  Barrier::UpIn,    barrierUp},
            {"down_out_put",   Option::Put,  Barrier::DownOut, barrierDown},
            {"up_out_put",     Option::Put,  Barrier::UpOut,   barrierUp},
        };

        std::cout << "  \"analytic_barrier\": {\n";
        bool first = true;
        for (auto& s : specs) {
            auto payoff = ext::make_shared<PlainVanillaPayoff>(s.type, strike);
            auto exercise = ext::make_shared<EuropeanExercise>(expiry);
            BarrierOption opt(s.bt, s.B, rebate, payoff, exercise);
            opt.setPricingEngine(engine);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    \"" << s.key << "\": " << opt.NPV();
        }
        std::cout << "\n  },\n";
    }

    // ---------------------------------------------------------------
    // Analytic binary barrier engine — Cash-or-nothing + Asset-or-nothing.
    // (S=100, K=100, B=95(down)/105(up), cashPayoff=10, T=1, r=5%, q=2%, sigma=30%).
    // American exercise with payoff-at-expiry.
    // ---------------------------------------------------------------
    {
        Real spot = 100.0, strike = 100.0;
        Real barrierDown = 95.0, barrierUp = 105.0;
        Real cash = 10.0;
        Volatility sigma = 0.30;
        Rate r = 0.05, q = 0.02;
        auto process = makeProcess(spot, r, q, sigma, ref, cal, dc);
        auto engine = ext::make_shared<AnalyticBinaryBarrierEngine>(process);

        struct Spec { const char* key; Option::Type type; Barrier::Type bt; Real B; bool cashNotAsset; };
        Spec specs[] = {
            {"cash_down_in_call",   Option::Call, Barrier::DownIn,  barrierDown, true},
            {"cash_up_in_call",     Option::Call, Barrier::UpIn,    barrierUp,   true},
            {"cash_down_out_call",  Option::Call, Barrier::DownOut, barrierDown, true},
            {"cash_up_out_call",    Option::Call, Barrier::UpOut,   barrierUp,   true},
            {"cash_down_in_put",    Option::Put,  Barrier::DownIn,  barrierDown, true},
            {"cash_up_in_put",      Option::Put,  Barrier::UpIn,    barrierUp,   true},
            {"cash_down_out_put",   Option::Put,  Barrier::DownOut, barrierDown, true},
            {"cash_up_out_put",     Option::Put,  Barrier::UpOut,   barrierUp,   true},
            {"asset_down_in_call",  Option::Call, Barrier::DownIn,  barrierDown, false},
            {"asset_up_out_put",    Option::Put,  Barrier::UpOut,   barrierUp,   false},
        };

        std::cout << "  \"analytic_binary_barrier\": {\n";
        bool first = true;
        for (auto& s : specs) {
            ext::shared_ptr<StrikedTypePayoff> payoff;
            if (s.cashNotAsset) {
                payoff = ext::make_shared<CashOrNothingPayoff>(s.type, strike, cash);
            } else {
                payoff = ext::make_shared<AssetOrNothingPayoff>(s.type, strike);
            }
            // American exercise w/ payoff at expiry (one-touch / no-touch).
            auto exercise = ext::make_shared<AmericanExercise>(ref, expiry, true);
            BarrierOption opt(s.bt, s.B, 0.0, payoff, exercise);
            opt.setPricingEngine(engine);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    \"" << s.key << "\": " << opt.NPV();
        }
        std::cout << "\n  },\n";
    }

    // ---------------------------------------------------------------
    // Stulz engine — 2-asset min/max basket Call/Put.
    // S1=S2=100, K=100, q1=q2=0, sigma1=20%, sigma2=30%, rho=0.5, T=1, r=5%.
    // ---------------------------------------------------------------
    {
        Real spot1 = 100.0, spot2 = 100.0, strike = 100.0;
        Volatility sigma1 = 0.20, sigma2 = 0.30;
        Rate r = 0.05, q = 0.00;
        Real rho = 0.5;
        auto proc1 = makeProcess(spot1, r, q, sigma1, ref, cal, dc);
        auto proc2 = makeProcess(spot2, r, q, sigma2, ref, cal, dc);
        auto engine = ext::make_shared<StulzEngine>(proc1, proc2, rho);

        std::cout << "  \"stulz\": {\n";
        // Min basket call.
        {
            auto vp = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
            auto basketPayoff = ext::make_shared<MinBasketPayoff>(vp);
            auto exercise = ext::make_shared<EuropeanExercise>(expiry);
            BasketOption opt(basketPayoff, exercise);
            opt.setPricingEngine(engine);
            std::cout << "    \"min_call_npv\": " << opt.NPV() << ",\n";
        }
        // Min basket put.
        {
            auto vp = ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
            auto basketPayoff = ext::make_shared<MinBasketPayoff>(vp);
            auto exercise = ext::make_shared<EuropeanExercise>(expiry);
            BasketOption opt(basketPayoff, exercise);
            opt.setPricingEngine(engine);
            std::cout << "    \"min_put_npv\": " << opt.NPV() << ",\n";
        }
        // Max basket call.
        {
            auto vp = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
            auto basketPayoff = ext::make_shared<MaxBasketPayoff>(vp);
            auto exercise = ext::make_shared<EuropeanExercise>(expiry);
            BasketOption opt(basketPayoff, exercise);
            opt.setPricingEngine(engine);
            std::cout << "    \"max_call_npv\": " << opt.NPV() << ",\n";
        }
        // Max basket put.
        {
            auto vp = ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
            auto basketPayoff = ext::make_shared<MaxBasketPayoff>(vp);
            auto exercise = ext::make_shared<EuropeanExercise>(expiry);
            BasketOption opt(basketPayoff, exercise);
            opt.setPricingEngine(engine);
            std::cout << "    \"max_put_npv\": " << opt.NPV() << "\n";
        }
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // Analytic continuous floating lookback (Goldman-Sosin-Gatto).
    // S=100, current extremum = 100, T=1, r=5%, q=2%, sigma=30%.
    // ---------------------------------------------------------------
    {
        Real spot = 100.0, minmax = 100.0;
        Volatility sigma = 0.30;
        Rate r = 0.05, q = 0.02;
        auto process = makeProcess(spot, r, q, sigma, ref, cal, dc);
        auto engine = ext::make_shared<AnalyticContinuousFloatingLookbackEngine>(process);

        std::cout << "  \"analytic_continuous_floating_lookback\": {\n";
        // Call: floating-strike call uses the running MINIMUM
        // (payoff = S_T - min).
        {
            auto payoff = ext::make_shared<FloatingTypePayoff>(Option::Call);
            auto exercise = ext::make_shared<EuropeanExercise>(expiry);
            ContinuousFloatingLookbackOption opt(minmax, payoff, exercise);
            opt.setPricingEngine(engine);
            std::cout << "    \"call_npv\": " << opt.NPV() << ",\n";
        }
        // Put: floating-strike put uses the running MAXIMUM
        // (payoff = max - S_T).
        {
            auto payoff = ext::make_shared<FloatingTypePayoff>(Option::Put);
            auto exercise = ext::make_shared<EuropeanExercise>(expiry);
            ContinuousFloatingLookbackOption opt(minmax, payoff, exercise);
            opt.setPricingEngine(engine);
            std::cout << "    \"put_npv\": " << opt.NPV() << "\n";
        }
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
