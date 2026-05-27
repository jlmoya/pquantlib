// L3-C cluster probe: swaps + DiscountingSwapEngine + L2-C carry-over
// roundtrips.
//
// Emits reference values for:
//   * VanillaSwap fixed 5% vs Euribor3M, 5-year, semi-annual fixed,
//     quarterly float, FlatForward(5%) curve. NPV, fair_rate,
//     fixedLegNPV, floatingLegNPV, fixedLegBPS, floatingLegBPS.
//   * OvernightIndexedSwap fixed 4% vs Sofr, 2-year, annual schedule,
//     FlatForward(4%) curve. NPV + fair_rate.
//   * ZeroCouponSwap (fixedRate constructor) 5-year, FlatForward(5%).
//     fixedLegNPV / floatingLegNPV / fairFixedPayment / fairFixedRate.
//   * SwapRateHelper roundtrip: input quote 5% → bootstrap a FlatForward
//     anchored at the quote → impliedQuote should ≈ 5% (LOOSE tier).
//   * OISRateHelper roundtrip: input quote 4% → FlatForward(4%) →
//     impliedQuote should ≈ 4% (LOOSE tier).
//
// C++ parity: v1.42.1 (099987f0).

#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/instruments/makevanillaswap.hpp>
#include <ql/instruments/makeois.hpp>
#include <ql/instruments/overnightindexedswap.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/instruments/zerocouponswap.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/oisratehelper.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <memory>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    Date evalDate(17, January, 2024); // Wed; clean of MLK + weekends.
    Settings::instance().evaluationDate() = evalDate;

    std::cout << "{\n";

    // ------------------------------------------------------------------
    // VanillaSwap: 5y fixed 5% vs Euribor3M, semi-annual fixed,
    // quarterly float, FlatForward(5%) curve.
    // ------------------------------------------------------------------
    {
        Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(
                evalDate, rateQuote, Actual360(), Continuous, Annual));

        auto idx = ext::make_shared<Euribor3M>(curve);

        Date settle = TARGET().advance(evalDate, 2, Days);
        Date end = settle + 5 * Years;
        Schedule fixedSched(settle, end, 6 * Months, TARGET(),
                            ModifiedFollowing, ModifiedFollowing,
                            DateGeneration::Backward, false);
        Schedule floatSched(settle, end, 3 * Months, TARGET(),
                            ModifiedFollowing, ModifiedFollowing,
                            DateGeneration::Backward, false);
        VanillaSwap swap(Swap::Payer, 1000000.0,
                         fixedSched, 0.05, Thirty360(Thirty360::BondBasis),
                         floatSched, idx, 0.0, idx->dayCounter());
        auto engine = ext::make_shared<DiscountingSwapEngine>(curve);
        swap.setPricingEngine(engine);

        std::cout << "  \"vanilla_swap_5y\": {\n";
        std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
        std::cout << "    \"end_serial\": " << end.serialNumber() << ",\n";
        std::cout << "    \"fixed_leg_size\": " << swap.fixedLeg().size() << ",\n";
        std::cout << "    \"floating_leg_size\": " << swap.floatingLeg().size() << ",\n";
        std::cout << "    \"npv\": " << swap.NPV() << ",\n";
        std::cout << "    \"fair_rate\": " << swap.fairRate() << ",\n";
        std::cout << "    \"fair_spread\": " << swap.fairSpread() << ",\n";
        std::cout << "    \"fixed_leg_npv\": " << swap.fixedLegNPV() << ",\n";
        std::cout << "    \"floating_leg_npv\": " << swap.floatingLegNPV() << ",\n";
        std::cout << "    \"fixed_leg_bps\": " << swap.fixedLegBPS() << ",\n";
        std::cout << "    \"floating_leg_bps\": " << swap.floatingLegBPS() << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // Receiver-flavour variant: same numbers, NPV sign flipped.
    // ------------------------------------------------------------------
    {
        Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(
                evalDate, rateQuote, Actual360(), Continuous, Annual));

        auto idx = ext::make_shared<Euribor3M>(curve);

        Date settle = TARGET().advance(evalDate, 2, Days);
        Date end = settle + 5 * Years;
        Schedule fixedSched(settle, end, 6 * Months, TARGET(),
                            ModifiedFollowing, ModifiedFollowing,
                            DateGeneration::Backward, false);
        Schedule floatSched(settle, end, 3 * Months, TARGET(),
                            ModifiedFollowing, ModifiedFollowing,
                            DateGeneration::Backward, false);
        VanillaSwap swap(Swap::Receiver, 1000000.0,
                         fixedSched, 0.06, Thirty360(Thirty360::BondBasis),
                         floatSched, idx, 0.0, idx->dayCounter());
        auto engine = ext::make_shared<DiscountingSwapEngine>(curve);
        swap.setPricingEngine(engine);

        std::cout << "  \"vanilla_swap_receiver_6pct\": {\n";
        std::cout << "    \"npv\": " << swap.NPV() << ",\n";
        std::cout << "    \"fair_rate\": " << swap.fairRate() << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // OvernightIndexedSwap: 2y fixed 4% vs Sofr, FlatForward(4%) curve.
    // ------------------------------------------------------------------
    {
        Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.04));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(
                evalDate, rateQuote, Actual360(), Continuous, Annual));

        auto idx = ext::make_shared<Sofr>(curve);

        Calendar cal = idx->fixingCalendar();
        Date settle = cal.advance(evalDate, 2, Days);
        Date end = settle + 2 * Years;
        Schedule sched(settle, end, 1 * Years, cal,
                       ModifiedFollowing, ModifiedFollowing,
                       DateGeneration::Backward, false);
        OvernightIndexedSwap swap(Swap::Payer, 1000000.0, sched,
                                  0.04, Actual360(), idx);
        auto engine = ext::make_shared<DiscountingSwapEngine>(curve);
        swap.setPricingEngine(engine);

        std::cout << "  \"ois_2y\": {\n";
        std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
        std::cout << "    \"end_serial\": " << end.serialNumber() << ",\n";
        std::cout << "    \"fixed_leg_size\": " << swap.fixedLeg().size() << ",\n";
        std::cout << "    \"overnight_leg_size\": " << swap.overnightLeg().size() << ",\n";
        std::cout << "    \"npv\": " << swap.NPV() << ",\n";
        std::cout << "    \"fair_rate\": " << swap.fairRate() << ",\n";
        std::cout << "    \"fixed_leg_npv\": " << swap.fixedLegNPV() << ",\n";
        std::cout << "    \"overnight_leg_npv\": " << swap.overnightLegNPV() << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // ZeroCouponSwap: explicit start/end + flat curve. Fixed rate 5%
    // compounded annually over Actual360.
    // ------------------------------------------------------------------
    {
        Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(
                evalDate, rateQuote, Actual360(), Continuous, Annual));

        auto idx = ext::make_shared<Euribor3M>(curve);

        Date startDate = TARGET().advance(evalDate, 2, Days);
        Date matDate = startDate + 5 * Years;
        ZeroCouponSwap swap(Swap::Payer, 1000000.0, startDate, matDate,
                            0.05, Actual360(), idx, TARGET(),
                            ModifiedFollowing, 0);
        auto engine = ext::make_shared<DiscountingSwapEngine>(curve);
        swap.setPricingEngine(engine);

        std::cout << "  \"zero_coupon_swap_5y\": {\n";
        std::cout << "    \"start_serial\": " << startDate.serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << matDate.serialNumber() << ",\n";
        std::cout << "    \"fixed_payment\": " << swap.fixedPayment() << ",\n";
        std::cout << "    \"npv\": " << swap.NPV() << ",\n";
        std::cout << "    \"fixed_leg_npv\": " << swap.fixedLegNPV() << ",\n";
        std::cout << "    \"floating_leg_npv\": " << swap.floatingLegNPV() << ",\n";
        std::cout << "    \"fair_fixed_payment\": " << swap.fairFixedPayment() << ",\n";
        std::cout << "    \"fair_fixed_rate\": " << swap.fairFixedRate(Actual360()) << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // SwapRateHelper roundtrip: quote=5%, flat 5% curve → implied ≈ 5%.
    // ------------------------------------------------------------------
    {
        Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(
                evalDate, rateQuote, Actual360(), Continuous, Annual));

        auto idx = ext::make_shared<Euribor3M>(curve);
        auto helper = ext::make_shared<SwapRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.05)),
            5 * Years, TARGET(), Annual, ModifiedFollowing,
            Thirty360(Thirty360::BondBasis), idx);
        helper->setTermStructure(curve.currentLink().get());
        Real implied = helper->impliedQuote();
        std::cout << "  \"swap_rate_helper\": {\n";
        std::cout << "    \"earliest_serial\": " << helper->earliestDate().serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << helper->maturityDate().serialNumber() << ",\n";
        std::cout << "    \"implied_quote\": " << implied << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // OISRateHelper roundtrip.
    // ------------------------------------------------------------------
    {
        Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.04));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(
                evalDate, rateQuote, Actual360(), Continuous, Annual));

        auto idx = ext::make_shared<Sofr>(curve);
        auto helper = ext::make_shared<OISRateHelper>(
            2, // settlement days
            2 * Years,
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.04)),
            idx);
        helper->setTermStructure(curve.currentLink().get());
        Real implied = helper->impliedQuote();
        std::cout << "  \"ois_rate_helper\": {\n";
        std::cout << "    \"earliest_serial\": " << helper->earliestDate().serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << helper->maturityDate().serialNumber() << ",\n";
        std::cout << "    \"implied_quote\": " << implied << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
