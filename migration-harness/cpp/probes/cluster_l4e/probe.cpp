// L4-E cluster probe: swaption + capfloor instruments, helpers, and
// analytic engines.
//
// Captures reference values for:
//
//   * Swaption (5y10y receiver @ 3% on Euribor3M):
//     - BlackSwaptionEngine NPV @ flat lognormal vol = 20%.
//     - BachelierSwaptionEngine NPV @ flat normal vol = 1%.
//     - JamshidianSwaptionEngine NPV under HullWhite(a=0.1, sigma=0.01).
//     - G2SwaptionEngine NPV under G2(a=0.1, sigma=0.01, b=0.1, eta=0.01, rho=-0.75).
//
//   * Cap (5y @ 4% on Euribor3M):
//     - BlackCapFloorEngine NPV @ flat lognormal vol = 20%.
//     - BachelierCapFloorEngine NPV @ flat normal vol = 1%.
//     - AnalyticCapFloorEngine NPV under HullWhite(a=0.1, sigma=0.01).
//
//   * Floor (5y @ 4% on Euribor3M):
//     - BlackCapFloorEngine NPV @ flat lognormal vol = 20%.
//
//   * SwaptionHelper round-trip:
//     market_value (Black) reproduces the input quote when the helper's
//     swaption uses the BlackSwaptionEngine on the same flat vol.
//
//   * CapHelper round-trip: same idea.
//
// C++ parity:
//   ql/instruments/swaption.{hpp,cpp},
//   ql/instruments/capfloor.{hpp,cpp},
//   ql/pricingengines/swaption/blackswaptionengine.{hpp,cpp},
//   ql/pricingengines/swaption/jamshidianswaptionengine.{hpp,cpp},
//   ql/pricingengines/swaption/g2swaptionengine.hpp,
//   ql/pricingengines/capfloor/{black,bachelier,analytic}capfloorengine.{hpp,cpp},
//   ql/models/shortrate/calibrationhelpers/{swaption,cap}helper.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/instruments/capfloor.hpp>
#include <ql/instruments/makecapfloor.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/cashflowvectors.hpp>
#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/models/shortrate/calibrationhelpers/swaptionhelper.hpp>
#include <ql/models/shortrate/calibrationhelpers/caphelper.hpp>
#include <ql/pricingengines/capfloor/analyticcapfloorengine.hpp>
#include <ql/pricingengines/capfloor/bacheliercapfloorengine.hpp>
#include <ql/pricingengines/capfloor/blackcapfloorengine.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/pricingengines/swaption/blackswaptionengine.hpp>
#include <ql/pricingengines/swaption/g2swaptionengine.hpp>
#include <ql/pricingengines/swaption/jamshidianswaptionengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);

    // Evaluation date — Wednesday, no TARGET holiday issues.
    Date evalDate(17, January, 2024);
    Settings::instance().evaluationDate() = evalDate;

    // Discount + forwarding curve: FlatForward(5%, Actual360, Continuous, Annual).
    Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.05));
    Handle<YieldTermStructure> curve(
        ext::make_shared<FlatForward>(evalDate, rateQuote, Actual360(),
                                      Continuous, Annual));
    Calendar cal = TARGET();
    DayCounter dc = Actual360();

    auto index = ext::make_shared<Euribor3M>(curve);

    auto swapEngine = ext::make_shared<DiscountingSwapEngine>(curve, false);

    std::cout << "{\n";

    // -----------------------------------------------------------------
    // Swaption 5y x 10y on Euribor3M @ 3%, RECEIVER.
    //
    // Build the swap (settle = eval + 2 days, fixed semi-annual,
    // float quarterly), wrap in a Swaption with European exercise at
    // settle.
    // -----------------------------------------------------------------
    Date settle = cal.advance(evalDate, 5 * Years);
    Date end = cal.advance(settle, 10 * Years);
    Schedule fixedSchedule(settle, end, Period(6, Months), cal,
                           ModifiedFollowing, ModifiedFollowing,
                           DateGeneration::Backward, false);
    Schedule floatSchedule(settle, end, Period(3, Months), cal,
                           ModifiedFollowing, ModifiedFollowing,
                           DateGeneration::Backward, false);

    Rate strike = 0.03;
    Real nominal = 1'000'000.0;
    auto swap = ext::make_shared<VanillaSwap>(
        Swap::Receiver, nominal,
        fixedSchedule, strike, Thirty360(Thirty360::BondBasis),
        floatSchedule, index, 0.0, index->dayCounter());
    swap->setPricingEngine(swapEngine);

    auto exercise = ext::make_shared<EuropeanExercise>(settle);
    auto swaption = ext::make_shared<Swaption>(swap, exercise);

    Real fairRate = swap->fairRate();

    std::cout << "  \"setup\": {\n";
    std::cout << "    \"eval_date_serial\": " << evalDate.serialNumber() << ",\n";
    std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
    std::cout << "    \"end_serial\": " << end.serialNumber() << ",\n";
    std::cout << "    \"strike\": " << strike << ",\n";
    std::cout << "    \"nominal\": " << nominal << ",\n";
    std::cout << "    \"flat_rate\": " << 0.05 << ",\n";
    std::cout << "    \"fixed_leg_size\": " << swap->fixedLeg().size() << ",\n";
    std::cout << "    \"floating_leg_size\": " << swap->floatingLeg().size() << ",\n";
    std::cout << "    \"swap_fair_rate\": " << fairRate << "\n";
    std::cout << "  },\n";

    // ---- BlackSwaptionEngine @ vol=20%
    {
        Volatility vol = 0.20;
        auto eng = ext::make_shared<BlackSwaptionEngine>(curve, vol);
        swaption->setPricingEngine(eng);
        Real npv = swaption->NPV();
        std::cout << "  \"black_swaption_5y10y\": {\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ---- BachelierSwaptionEngine @ normal vol = 1%
    {
        Volatility vol = 0.01;
        auto eng = ext::make_shared<BachelierSwaptionEngine>(curve, vol);
        swaption->setPricingEngine(eng);
        Real npv = swaption->NPV();
        std::cout << "  \"bachelier_swaption_5y10y\": {\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ---- JamshidianSwaptionEngine under HW(a=0.1, sigma=0.01)
    {
        auto hw = ext::make_shared<HullWhite>(curve, 0.1, 0.01);
        auto eng = ext::make_shared<JamshidianSwaptionEngine>(hw, curve);
        swaption->setPricingEngine(eng);
        Real npv = swaption->NPV();
        std::cout << "  \"jamshidian_swaption_5y10y\": {\n";
        std::cout << "    \"hw_a\": 0.1,\n";
        std::cout << "    \"hw_sigma\": 0.01,\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ---- G2SwaptionEngine under G2(a=0.1, sigma=0.01, b=0.1, eta=0.01, rho=-0.75)
    {
        auto g2 = ext::make_shared<G2>(curve, 0.1, 0.01, 0.1, 0.01, -0.75);
        auto eng = ext::make_shared<G2SwaptionEngine>(g2, 6.0, 32);
        swaption->setPricingEngine(eng);
        Real npv = swaption->NPV();
        std::cout << "  \"g2_swaption_5y10y\": {\n";
        std::cout << "    \"a\": 0.1, \"sigma\": 0.01, \"b\": 0.1, \"eta\": 0.01, \"rho\": -0.75,\n";
        std::cout << "    \"range\": 6.0,\n";
        std::cout << "    \"intervals\": 32,\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // Cap 5y on Euribor3M @ 4%.
    //
    // Build a quarterly floating leg, wrap with strike vector.
    // -----------------------------------------------------------------
    Date capStart = cal.advance(evalDate, 2, Days);
    Date capEnd = cal.advance(capStart, 5, Years);
    Schedule capSchedule(capStart, capEnd, Period(3, Months), cal,
                         ModifiedFollowing, ModifiedFollowing,
                         DateGeneration::Backward, false);

    std::vector<Real> capNominals = {nominal};
    Leg capLeg = IborLeg(capSchedule, index)
        .withNotionals(capNominals)
        .withPaymentAdjustment(ModifiedFollowing)
        .withFixingDays(2);
    Rate capStrike = 0.04;
    std::vector<Rate> capStrikes = {capStrike};
    auto cap = ext::make_shared<Cap>(capLeg, capStrikes);
    auto floor = ext::make_shared<Floor>(capLeg, capStrikes);

    std::cout << "  \"cap_setup\": {\n";
    std::cout << "    \"start_serial\": " << capStart.serialNumber() << ",\n";
    std::cout << "    \"end_serial\": " << capEnd.serialNumber() << ",\n";
    std::cout << "    \"strike\": " << capStrike << ",\n";
    std::cout << "    \"nominal\": " << nominal << ",\n";
    std::cout << "    \"leg_size\": " << capLeg.size() << "\n";
    std::cout << "  },\n";

    // ---- BlackCapFloorEngine on Cap @ vol=20%
    {
        Volatility vol = 0.20;
        auto eng = ext::make_shared<BlackCapFloorEngine>(curve, vol);
        cap->setPricingEngine(eng);
        Real npv = cap->NPV();
        std::cout << "  \"black_cap_5y_4pct\": {\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ---- BachelierCapFloorEngine on Cap @ normal vol=1%
    {
        Volatility vol = 0.01;
        auto eng = ext::make_shared<BachelierCapFloorEngine>(curve, vol);
        cap->setPricingEngine(eng);
        Real npv = cap->NPV();
        std::cout << "  \"bachelier_cap_5y_4pct\": {\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ---- AnalyticCapFloorEngine on Cap under HW(a=0.1, sigma=0.01)
    {
        auto hw = ext::make_shared<HullWhite>(curve, 0.1, 0.01);
        auto eng = ext::make_shared<AnalyticCapFloorEngine>(hw, curve);
        cap->setPricingEngine(eng);
        Real npv = cap->NPV();
        std::cout << "  \"analytic_cap_5y_4pct_hw\": {\n";
        std::cout << "    \"hw_a\": 0.1,\n";
        std::cout << "    \"hw_sigma\": 0.01,\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ---- BlackCapFloorEngine on Floor @ vol=20%
    {
        Volatility vol = 0.20;
        auto eng = ext::make_shared<BlackCapFloorEngine>(curve, vol);
        floor->setPricingEngine(eng);
        Real npv = floor->NPV();
        std::cout << "  \"black_floor_5y_4pct\": {\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // SwaptionHelper round-trip.
    //
    // Build a SwaptionHelper with maturity 5y, length 10y, vol = 20%
    // (ShiftedLognormal), strike = 0.03, nominal = 1.0; set its engine
    // to the BlackSwaptionEngine with the same vol; market_value()
    // should equal model_value() (round-trip identity).
    // -----------------------------------------------------------------
    {
        Volatility vol = 0.20;
        Handle<Quote> volQuote(ext::make_shared<SimpleQuote>(vol));
        ext::shared_ptr<SwaptionHelper> helper(new SwaptionHelper(
            5 * Years,             // maturity period
            10 * Years,            // swap length period
            volQuote,              // vol quote
            index,                 // Euribor3M
            6 * Months,            // fixed leg tenor
            Thirty360(Thirty360::BondBasis),  // fixed day count
            Actual360(),           // floating day count
            curve,                 // term structure
            BlackCalibrationHelper::PriceError,
            0.03,                  // strike
            1.0));                 // nominal
        auto eng = ext::make_shared<BlackSwaptionEngine>(curve, volQuote);
        helper->setPricingEngine(eng);
        Real mv = helper->marketValue();
        Real model = helper->modelValue();
        Real err = helper->calibrationError();
        std::cout << "  \"swaption_helper_roundtrip\": {\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"strike\": 0.03,\n";
        std::cout << "    \"market_value\": " << mv << ",\n";
        std::cout << "    \"model_value\": " << model << ",\n";
        std::cout << "    \"calibration_error\": " << err << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // CapHelper round-trip.
    //
    // 5y cap on Euribor3M, vol = 20%, helper finds the ATM strike (fair
    // forward rate) internally. With the BlackCapFloorEngine on the
    // same vol, market_value() == model_value().
    // -----------------------------------------------------------------
    {
        Volatility vol = 0.20;
        Handle<Quote> volQuote(ext::make_shared<SimpleQuote>(vol));
        ext::shared_ptr<CapHelper> helper(new CapHelper(
            5 * Years,             // length
            volQuote,              // vol quote
            index,                 // Euribor3M
            Quarterly,             // fixed leg frequency (for ATM-swap calc)
            Actual360(),           // fixed leg day count
            true,                  // includeFirstSwaplet
            curve,                 // term structure
            BlackCalibrationHelper::PriceError));
        auto eng = ext::make_shared<BlackCapFloorEngine>(curve, volQuote);
        helper->setPricingEngine(eng);
        Real mv = helper->marketValue();
        Real model = helper->modelValue();
        Real err = helper->calibrationError();
        std::cout << "  \"cap_helper_roundtrip\": {\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"length_years\": 5,\n";
        std::cout << "    \"market_value\": " << mv << ",\n";
        std::cout << "    \"model_value\": " << model << ",\n";
        std::cout << "    \"calibration_error\": " << err << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
