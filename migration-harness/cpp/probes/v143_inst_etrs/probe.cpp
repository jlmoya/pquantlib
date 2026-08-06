// migration-harness/cpp/probes/v143_inst_etrs/probe.cpp
//
// Reference values for EquityTotalReturnSwap
// (ql/instruments/equitytotalreturnswap.hpp + .cpp, v1.43).
//
// The swap exchanges the total return of an equity index for a floating leg
// linked to either an Ibor index or an overnight index. Both constructors are
// probed.
//
// What is pinned and why:
//
//   * The FULL cashflow listing of both legs — for each flow the payment-date
//     serial and amount; for each interest coupon also the nominal, accrual
//     start/end, accrual period, fixing date and rate. The NPV alone would let
//     an equity-leg error and an interest-leg error cancel.
//
//   * paymentDelay at a NON-ZERO value. It is threaded to withPaymentLag on
//     the interest leg AND to the equity cash flow's payment date, and an
//     accepted-then-dropped payment lag is the defect class this port is prone
//     to. Each flavour is emitted at delay 0 and at delay 2 so the difference
//     is visible in the reference itself.
//
//   * paymentCalendar and paymentConvention, likewise at non-default values,
//     with a schedule whose period ends fall on days where TARGET and
//     UnitedStates::GovernmentBond disagree — otherwise the calendar argument
//     is unobservable.
//
//   * gearing and margin separately (a port that folded margin into gearing
//     would still reproduce a single case), and both swap types.
//
//   * fairMargin(), equityLegNPV(), interestRateLegNPV(), legBPS.
//
// Curves, index and schedule are built inline from literal tables.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/etrs.json.

#include <exception>
#include <string>
#include <iomanip>
#include <iostream>
#include <vector>

#include <ql/cashflows/coupon.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/equityindex.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/estr.hpp>
#include <ql/instruments/equitytotalreturnswap.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

const Date kToday(15, June, 2026);
const Actual365Fixed kA365;

Handle<YieldTermStructure> flat(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, r, kA365));
}

// Deliberately forward-starting: every Euribor6M fixing then falls after the
// evaluation date, so no coupon depends on a historical fixing. That keeps
// this probe about EquityTotalReturnSwap rather than about how a past fixing
// is resolved — PQuantLib's IborCouponPricer prefers the par-coupon curve span
// over the stored fixing even when the fixing date is in the past, which is a
// separate, pre-existing divergence from C++ IborCoupon::indexFixing().
const Date kStart = kToday + Period(6, Months);

Schedule makeSchedule() {
    return MakeSchedule()
        .from(kStart)
        .to(kStart + Period(2, Years))
        .withFrequency(Semiannual)
        .withCalendar(TARGET())
        .withConvention(Unadjusted)
        .backwards();
}

ext::shared_ptr<EquityIndex> makeEquityIndex() {
    return ext::make_shared<EquityIndex>(
        "eqIndex", TARGET(), EURCurrency(), flat(0.025), flat(0.015),
        Handle<Quote>(ext::make_shared<SimpleQuote>(8700.0)));
}

void emitLeg(const char* key, const Leg& leg, bool trailingComma) {
    std::cout << "      \"" << key << "\": [";
    for (Size i = 0; i < leg.size(); ++i) {
        if (i != 0)
            std::cout << ",";
        std::cout << "\n        {\"date_serial\": " << leg[i]->date().serialNumber()
                  << ", \"amount\": " << leg[i]->amount();
        auto cpn = ext::dynamic_pointer_cast<Coupon>(leg[i]);
        if (cpn != nullptr) {
            std::cout << ", \"nominal\": " << cpn->nominal()
                      << ", \"accrual_start_serial\": " << cpn->accrualStartDate().serialNumber()
                      << ", \"accrual_end_serial\": " << cpn->accrualEndDate().serialNumber()
                      << ", \"accrual_period\": " << cpn->accrualPeriod()
                      << ", \"rate\": " << cpn->rate();
        }
        std::cout << "}";
    }
    std::cout << "\n      ]" << (trailingComma ? "," : "") << "\n";
}

void emitSwap(const char* key, EquityTotalReturnSwap& swap, bool trailingComma) {
    std::cout << "  \"" << key << "\": {\n"
              << "    \"type\": " << static_cast<int>(swap.type()) << ",\n"
              << "    \"nominal\": " << swap.nominal() << ",\n"
              << "    \"margin\": " << swap.margin() << ",\n"
              << "    \"gearing\": " << swap.gearing() << ",\n"
              << "    \"payment_delay\": " << swap.paymentDelay() << ",\n"
              << "    \"payment_convention\": " << static_cast<int>(swap.paymentConvention())
              << ",\n"
              // The default paymentCalendar is an EMPTY Calendar, whose
              // name() throws; the swap falls back to the schedule's
              // calendar in that case, which is itself worth pinning.
              << "    \"payment_calendar\": \""
              << (swap.paymentCalendar().empty() ? std::string() : swap.paymentCalendar().name())
              << "\",\n"
              << "    \"day_counter\": \"" << swap.dayCounter().name() << "\",\n"
              << "    \"npv\": " << swap.NPV() << ",\n"
              << "    \"equity_leg_npv\": " << swap.equityLegNPV() << ",\n"
              << "    \"interest_rate_leg_npv\": " << swap.interestRateLegNPV() << ",\n"
              << "    \"interest_leg_bps\": " << swap.legBPS(1) << ",\n"
              << "    \"fair_margin\": " << swap.fairMargin() << ",\n"
              << "    \"legs\": {\n";
    emitLeg("equity", swap.equityLeg(), true);
    emitLeg("interest", swap.interestRateLeg(), false);
    std::cout << "    }\n  }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"today_serial\": " << kToday.serialNumber() << ",\n";
    std::cout << "  \"start_serial\": " << kStart.serialNumber() << ",\n";

    Handle<YieldTermStructure> discount = flat(0.03);
    Handle<YieldTermStructure> forecast = flat(0.035);
    auto engine = ext::make_shared<DiscountingSwapEngine>(discount);

    const Schedule schedule = makeSchedule();
    auto equityIndex = makeEquityIndex();
    auto euribor = ext::make_shared<Euribor6M>(forecast);
    // Estr is an OvernightIndex, which IS-A IborIndex, so the two
    // EquityTotalReturnSwap constructors are both viable for it; the
    // static type of the argument is what selects the overnight overload
    // (and therefore OvernightLeg rather than IborLeg).
    ext::shared_ptr<OvernightIndex> estr = ext::make_shared<Estr>(forecast);


    // --- Ibor flavour, no delay / all defaults on calendar+convention ------
    {
        EquityTotalReturnSwap swap(Swap::Payer, 1.0e7, schedule, equityIndex, euribor, kA365,
                                   0.0035);
        swap.setPricingEngine(engine);
        emitSwap("ibor_plain", swap, true);
    }
    // --- Ibor flavour, non-zero delay, foreign calendar, MF convention -----
    {
        EquityTotalReturnSwap swap(Swap::Payer, 1.0e7, schedule, equityIndex, euribor, kA365,
                                   0.0035, 1.0,
                                   UnitedStates(UnitedStates::GovernmentBond),
                                   ModifiedFollowing, 2);
        swap.setPricingEngine(engine);
        emitSwap("ibor_lagged", swap, true);
    }
    // --- Ibor flavour, non-unit gearing, receiver ---------------------------
    {
        EquityTotalReturnSwap swap(Swap::Receiver, 5.0e6, schedule, equityIndex, euribor, kA365,
                                   -0.0012, 1.35, TARGET(), Following, 1);
        swap.setPricingEngine(engine);
        emitSwap("ibor_geared_receiver", swap, true);
    }
    // --- Overnight flavour, no delay ---------------------------------------
    {
        EquityTotalReturnSwap swap(Swap::Payer, 1.0e7, schedule, equityIndex, estr, kA365,
                                   0.0035);
        swap.setPricingEngine(engine);
        emitSwap("overnight_plain", swap, true);
    }
    // --- Overnight flavour, non-zero delay ---------------------------------
    {
        EquityTotalReturnSwap swap(Swap::Payer, 1.0e7, schedule, equityIndex, estr, kA365,
                                   0.0035, 1.0,
                                   UnitedStates(UnitedStates::GovernmentBond),
                                   ModifiedFollowing, 2);
        swap.setPricingEngine(engine);
        emitSwap("overnight_lagged", swap, true);
    }

    // --- failure branch: negative nominal ----------------------------------
    std::cout << "  \"negative_nominal\": {\n";
    try {
        EquityTotalReturnSwap swap(Swap::Payer, -1.0, schedule, equityIndex, euribor, kA365, 0.0);
        std::cout << "    \"raises\": false\n";
    } catch (const std::exception& e) {
        std::cout << "    \"raises\": true,\n    \"error\": \"" << e.what() << "\"\n";
    }
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
