// L6-C cluster probe: DoubleBarrierOption + AnalyticDoubleBarrierEngine.
//
// Captures Ikeda-Kunitomo (1992) closed-form reference values for the
// double-barrier knock-in / knock-out family under flat-vol
// Black-Scholes, plus the matching European-vanilla NPV so the
// Python in-out parity test can load both numbers from one JSON.
//
// Cases (T=1y, day-counter Actual365Fixed, calendar NullCalendar,
// reference 15 June 2026):
//
//   1. textbook_ko_call         — S=K=100, L=80, U=120, r=5%, q=0%, sigma=20%,
//                                 rebate=0, series=5, KnockOut Call.
//   2. textbook_ki_call         — same params, KnockIn Call.
//   3. textbook_vanilla_call    — European Call (for in-out-parity test).
//   4. asym_ko_call             — S=K=100, L=90, U=130, r=5%, q=2%, sigma=25%,
//                                 series=5, KnockOut Call.
//   5. textbook_ko_put          — S=K=100, L=80, U=120, r=5%, q=0%, sigma=20%,
//                                 rebate=0, series=5, KnockOut Put.
//   6. textbook_ki_put          — same params, KnockIn Put.
//   7. textbook_vanilla_put     — European Put (for in-out-parity test).
//   8. textbook_ko_call_s10     — KO Call case 1 with series=10
//                                 (convergence cross-check).
//
// C++ parity:
//   ql/instruments/doublebarrieroption.{hpp,cpp},
//   ql/pricingengines/barrier/analyticdoublebarrierengine.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/instruments/doublebarrieroption.hpp>
#include <ql/instruments/doublebarriertype.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/barrier/analyticdoublebarrierengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>

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

Real priceDouble(
    const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
    const Date& expiry,
    Option::Type optionType, DoubleBarrier::Type bt,
    Real strike, Real bLo, Real bHi, Real rebate, int series) {
    auto payoff = ext::make_shared<PlainVanillaPayoff>(optionType, strike);
    auto exercise = ext::make_shared<EuropeanExercise>(expiry);
    DoubleBarrierOption opt(bt, bLo, bHi, rebate, payoff, exercise);
    opt.setPricingEngine(
        ext::make_shared<AnalyticDoubleBarrierEngine>(process, series));
    return opt.NPV();
}

Real priceVanilla(
    const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
    const Date& expiry,
    Option::Type optionType, Real strike) {
    auto payoff = ext::make_shared<PlainVanillaPayoff>(optionType, strike);
    auto exercise = ext::make_shared<EuropeanExercise>(expiry);
    VanillaOption opt(payoff, exercise);
    opt.setPricingEngine(
        ext::make_shared<AnalyticEuropeanEngine>(process));
    return opt.NPV();
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);

    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date ref(15, June, 2026);
    Settings::instance().evaluationDate() = ref;
    Date expiry = ref + 365;  // 1-year tenor

    std::cout << "{\n";

    // -- textbook setup (Haug §4.13): S=K=100, L=80, U=120, r=5%, q=0%,
    //    sigma=20%, rebate=0.
    {
        auto process =
            makeProcess(100.0, 0.05, 0.00, 0.20, ref, cal, dc);

        Real ko_call = priceDouble(process, expiry, Option::Call,
                                   DoubleBarrier::KnockOut,
                                   100.0, 80.0, 120.0, 0.0, 5);
        Real ki_call = priceDouble(process, expiry, Option::Call,
                                   DoubleBarrier::KnockIn,
                                   100.0, 80.0, 120.0, 0.0, 5);
        Real vanilla_call = priceVanilla(process, expiry, Option::Call, 100.0);
        Real ko_put = priceDouble(process, expiry, Option::Put,
                                  DoubleBarrier::KnockOut,
                                  100.0, 80.0, 120.0, 0.0, 5);
        Real ki_put = priceDouble(process, expiry, Option::Put,
                                  DoubleBarrier::KnockIn,
                                  100.0, 80.0, 120.0, 0.0, 5);
        Real vanilla_put = priceVanilla(process, expiry, Option::Put, 100.0);
        Real ko_call_s10 = priceDouble(process, expiry, Option::Call,
                                       DoubleBarrier::KnockOut,
                                       100.0, 80.0, 120.0, 0.0, 10);

        std::cout << "  \"textbook\": {\n";
        std::cout << "    \"ko_call\": " << ko_call << ",\n";
        std::cout << "    \"ki_call\": " << ki_call << ",\n";
        std::cout << "    \"vanilla_call\": " << vanilla_call << ",\n";
        std::cout << "    \"ko_put\": " << ko_put << ",\n";
        std::cout << "    \"ki_put\": " << ki_put << ",\n";
        std::cout << "    \"vanilla_put\": " << vanilla_put << ",\n";
        std::cout << "    \"ko_call_s10\": " << ko_call_s10 << "\n";
        std::cout << "  },\n";
    }

    // -- asymmetric setup: S=K=100, L=90, U=130, r=5%, q=2%, sigma=25%.
    {
        auto process =
            makeProcess(100.0, 0.05, 0.02, 0.25, ref, cal, dc);

        Real ko_call = priceDouble(process, expiry, Option::Call,
                                   DoubleBarrier::KnockOut,
                                   100.0, 90.0, 130.0, 0.0, 5);

        std::cout << "  \"asymmetric\": {\n";
        std::cout << "    \"ko_call\": " << ko_call << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
