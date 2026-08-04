// migration-harness/cpp/probes/v143_xccy_swaps/probe.cpp
//
// Reference values for the constant-notional cross-currency swap family
// introduced in C++ QuantLib v1.43:
//
//   * ConstNotionalCrossCurrencySwap                  (base, explicit legs)
//   * ConstNotionalCrossCurrencyBasisSwap             (float/float)
//   * ConstNotionalCrossCurrencyFixedVsFloatingSwap   (fixed/float)
//   * DiscountingConstNotionalCrossCurrencySwapEngine
//
// Design notes
// ------------
// The upstream test-suite files pin these instruments against hard-coded
// 27-point market curves and the USDLibor / GBPLibor / SOFR / SONIA index
// definitions. Reproducing that verbatim would make the reference depend on
// index *definitions* as much as on the swap logic, so a port failing here
// would not tell us which of the two broke.
//
// Instead this probe builds every input explicitly - two discount curves and
// two projection curves from literal (date, discount-factor) tables, and
// generic IborIndex / OvernightIndex instances constructed inline. Every one
// of those is trivially reproducible in Python, so a mismatch localises to the
// swap or the engine, which is the point.
//
// What is pinned, per case: NPV, and per leg leg_npv / leg_bps /
// in_ccy_leg_npv / in_ccy_leg_bps / npv_date_discounts / start_discounts /
// end_discounts, plus the full cashflow listing of every leg. The cashflow
// listing matters: without it an NPV match can hide two compensating errors in
// leg construction, and notional-exchange placement in particular is easy to
// get subtly wrong.
//
// Deliberately covered edge paths, each of which the engine special-cases:
//   * spotFXSettleDate != referenceDate  -> forward-FX adjustment of the rate
//   * npvDate          != referenceDate  -> NPV rebased off the reference date
//   * an overnight-index basis leg with a 2-day payment lag on both legs
//
// Deviation from the sister JQuantLib probe (instruments/v143_xccy_swaps):
// its overnight case additionally switches on compoundingSpreadDaily on the
// pay leg. PQuantLib's OvernightIndexedCoupon has no daily-spread-compounding
// (nor lookback / lockout / observation-shift / simple-averaging) support, so
// the port raises NotImplementedError on that combination rather than
// silently pricing it as if the flag were off. The overnight case here
// therefore leaves every one of those knobs at its C++ default and exercises
// the payment-lag path instead.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/xccy/swaps.json.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/currencies/america.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/instruments/constnotionalcrosscurrencybasisswap.hpp>
#include <ql/instruments/constnotionalcrosscurrencyfixedvsfloatingswap.hpp>
#include <ql/instruments/constnotionalcrosscurrencyswap.hpp>
#include <ql/pricingengines/swap/discountingconstnotionalcrosscurrencyswapengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

const Date kToday(11, September, 2018);
const Real kUsdNominal = 125000000.0;
const Real kSpotFx = 1.22; // USD per EUR

// --- curve construction -----------------------------------------------------
//
// Small, explicit discount-factor tables. DiscountCurve interpolates the log of
// the discount factors linearly, so these are exactly reproducible anywhere.

Handle<YieldTermStructure> discountCurve(const std::vector<Real>& dfs) {
    const std::vector<Date> dates = {
        Date(11, September, 2018), Date(11, December, 2018), Date(11, March, 2019),
        Date(11, September, 2019), Date(11, September, 2020), Date(13, September, 2021),
        Date(12, September, 2022), Date(11, September, 2023), Date(11, September, 2028),
    };
    QL_REQUIRE(dates.size() == dfs.size(), "curve table size mismatch");
    return Handle<YieldTermStructure>(
        ext::make_shared<DiscountCurve>(dates, dfs, Actual365Fixed()));
}

Handle<YieldTermStructure> usdDiscount() {
    return discountCurve({1.0, 0.9941, 0.9888, 0.9757, 0.9486, 0.9228, 0.8983, 0.8747, 0.7630});
}

Handle<YieldTermStructure> eurDiscount() {
    return discountCurve({1.0, 0.9998, 0.9995, 0.9986, 0.9955, 0.9910, 0.9850, 0.9775, 0.9210});
}

Handle<YieldTermStructure> usdProjection() {
    return discountCurve({1.0, 0.9935, 0.9871, 0.9727, 0.9433, 0.9148, 0.8876, 0.8615, 0.7386});
}

Handle<YieldTermStructure> eurProjection() {
    return discountCurve({1.0, 0.9996, 0.9991, 0.9978, 0.9938, 0.9881, 0.9808, 0.9720, 0.9040});
}

// --- index construction -----------------------------------------------------
//
// Generic indexes rather than named ones (USDLibor etc.), so the reference does
// not depend on a currency-specific index definition also being ported exactly.

ext::shared_ptr<IborIndex> usdIbor3M() {
    return ext::make_shared<IborIndex>("USD-XCCY-3M", Period(3, Months), 2, USDCurrency(), TARGET(),
                                       ModifiedFollowing, false, Actual360(), usdProjection());
}

ext::shared_ptr<IborIndex> eurIbor3M() {
    return ext::make_shared<IborIndex>("EUR-XCCY-3M", Period(3, Months), 2, EURCurrency(), TARGET(),
                                       ModifiedFollowing, false, Actual360(), eurProjection());
}

ext::shared_ptr<OvernightIndex> usdOn() {
    return ext::make_shared<OvernightIndex>("USD-XCCY-ON", 0, USDCurrency(), TARGET(), Actual360(),
                                            usdProjection());
}

ext::shared_ptr<OvernightIndex> eurOn() {
    return ext::make_shared<OvernightIndex>("EUR-XCCY-ON", 0, EURCurrency(), TARGET(), Actual360(),
                                            eurProjection());
}

Schedule quarterly(const Date& start, const Date& end) {
    return Schedule(start, end, Period(3, Months), TARGET(), ModifiedFollowing, ModifiedFollowing,
                    DateGeneration::Forward, false);
}

Schedule annual(const Date& start, const Date& end) {
    return Schedule(start, end, Period(1, Years), TARGET(), ModifiedFollowing, ModifiedFollowing,
                    DateGeneration::Forward, false);
}

Date startDate() { return TARGET().advance(kToday, Period(2, Days)); }

Date endDate() { return TARGET().advance(kToday, Period(5, Years)); }

// --- JSON emission ----------------------------------------------------------
//
// No JSON library in this harness: probes print JSON by hand. ``std::cout`` is
// left at ``setprecision(17)`` throughout so every double round-trips.

void emitReal(Real v) {
    if (v == Null<Real>())
        std::cout << "null";
    else
        std::cout << v;
}

void emitCashflows(const Leg& leg, const std::string& indent) {
    std::cout << "[\n";
    for (Size k = 0; k < leg.size(); ++k) {
        const auto& cf = leg[k];
        std::cout << indent << "  {\"date_serial\": " << cf->date().serialNumber()
                  << ", \"amount\": ";
        emitReal(cf->amount());
        if (auto c = ext::dynamic_pointer_cast<Coupon>(cf)) {
            std::cout << ", \"is_coupon\": true"
                      << ", \"nominal\": " << c->nominal()
                      << ", \"accrual_start_serial\": " << c->accrualStartDate().serialNumber()
                      << ", \"accrual_end_serial\": " << c->accrualEndDate().serialNumber()
                      << ", \"accrual_period\": " << c->accrualPeriod() << ", \"rate\": ";
            emitReal(c->rate());
        } else {
            std::cout << ", \"is_coupon\": false";
        }
        std::cout << "}" << (k + 1 < leg.size() ? "," : "") << "\n";
    }
    std::cout << indent << "]";
}

void emitLeg(const ConstNotionalCrossCurrencySwap& swap, Size i, bool trailingComma) {
    std::cout << "      {\n"
              << "        \"currency\": \"" << swap.legCurrency(i).code() << "\",\n"
              << "        \"leg_npv\": ";
    emitReal(swap.legNPV(i));
    std::cout << ",\n        \"leg_bps\": ";
    emitReal(swap.legBPS(i));
    std::cout << ",\n        \"in_ccy_leg_npv\": ";
    emitReal(swap.inCcyLegNPV(i));
    std::cout << ",\n        \"in_ccy_leg_bps\": ";
    emitReal(swap.inCcyLegBPS(i));
    std::cout << ",\n        \"npv_date_discounts\": ";
    emitReal(swap.npvDateDiscounts(i));
    std::cout << ",\n        \"start_discounts\": ";
    emitReal(swap.startDiscounts(i));
    std::cout << ",\n        \"end_discounts\": ";
    emitReal(swap.endDiscounts(i));
    std::cout << ",\n        \"cashflows\": ";
    emitCashflows(swap.leg(i), "        ");
    std::cout << "\n      }" << (trailingComma ? "," : "") << "\n";
}

// ``extra`` holds the product-specific fair-value results (fair spreads on the
// basis swap, fair rate/spread on the fixed-vs-floating one), already rendered
// as JSON object members without the enclosing braces.
void emitCase(const std::string& name,
              const ConstNotionalCrossCurrencySwap& swap,
              const std::string& extra,
              bool trailingComma) {
    std::cout << "  \"" << name << "\": {\n"
              << "    \"npv\": " << swap.NPV() << ",\n"
              << "    \"valuation_date_serial\": " << swap.valuationDate().serialNumber() << ",\n"
              << "    \"start_date_serial\": " << swap.startDate().serialNumber() << ",\n"
              << "    \"maturity_date_serial\": " << swap.maturityDate().serialNumber() << ",\n"
              << extra << "    \"legs\": [\n";
    emitLeg(swap, 0, true);
    emitLeg(swap, 1, false);
    std::cout << "    ]\n  }" << (trailingComma ? "," : "") << "\n";
}

std::string member(const std::string& key, Real value) {
    std::ostringstream os;
    os << std::setprecision(17);
    os << "    \"" << key << "\": ";
    if (value == Null<Real>())
        os << "null";
    else
        os << value;
    os << ",\n";
    return os.str();
}

// --- swap builders ----------------------------------------------------------

// Fixed/fixed, built through the base class's explicit-leg constructor, with
// the notional exchanges attached by hand exactly as the upstream test does.
ext::shared_ptr<ConstNotionalCrossCurrencySwap> makeFixFix() {
    const Calendar cal = TARGET();
    const Schedule sched = quarterly(startDate(), endDate());
    const DayCounter dc = Actual365Fixed();

    Leg usdLeg = FixedRateLeg(sched)
                     .withNotionals(kUsdNominal)
                     .withCouponRates(0.0575, dc)
                     .withPaymentAdjustment(ModifiedFollowing)
                     .withPaymentCalendar(cal);
    const Date first = cal.adjust(sched.dates().front(), ModifiedFollowing);
    usdLeg.insert(usdLeg.begin(), ext::make_shared<SimpleCashFlow>(-kUsdNominal, first));
    usdLeg.push_back(ext::make_shared<SimpleCashFlow>(kUsdNominal, usdLeg.back()->date()));

    const Real eurNominal = kUsdNominal / kSpotFx;
    Leg eurLeg = FixedRateLeg(sched)
                     .withNotionals(eurNominal)
                     .withCouponRates(0.0201, dc)
                     .withPaymentAdjustment(ModifiedFollowing)
                     .withPaymentCalendar(cal);
    eurLeg.insert(eurLeg.begin(), ext::make_shared<SimpleCashFlow>(-eurNominal, first));
    eurLeg.push_back(ext::make_shared<SimpleCashFlow>(eurNominal, eurLeg.back()->date()));

    return ext::make_shared<ConstNotionalCrossCurrencySwap>(usdLeg, USDCurrency(), eurLeg,
                                                            EURCurrency());
}

ext::shared_ptr<ConstNotionalCrossCurrencyBasisSwap> makeBasis(bool overnight) {
    const Schedule sched = quarterly(startDate(), endDate());
    const Real eurNominal = kUsdNominal / kSpotFx;

    if (overnight) {
        return ext::make_shared<ConstNotionalCrossCurrencyBasisSwap>(
            kUsdNominal, USDCurrency(), sched, usdOn(), 0.0010, 1.0, eurNominal, EURCurrency(),
            sched, eurOn(), 0.0025, 1.0,
            /*payPaymentLag*/ 2, /*recPaymentLag*/ 2,
            /*payCompoundSpread*/ false, /*payLookbackDays*/ Null<Natural>(),
            /*payObservationShift*/ false, /*payLockoutDays*/ 0, RateAveraging::Compound,
            /*recCompoundSpread*/ false, /*recLookbackDays*/ Null<Natural>(),
            /*recObservationShift*/ false, /*recLockoutDays*/ 0, RateAveraging::Compound,
            /*telescopicValueDates*/ false);
    }
    return ext::make_shared<ConstNotionalCrossCurrencyBasisSwap>(
        kUsdNominal, USDCurrency(), sched, usdIbor3M(), 0.0010, 1.0, eurNominal, EURCurrency(),
        sched, eurIbor3M(), 0.0025, 1.0);
}

ext::shared_ptr<ConstNotionalCrossCurrencyFixedVsFloatingSwap> makeFixFloat(Swap::Type type) {
    const Calendar cal = TARGET();
    return ext::make_shared<ConstNotionalCrossCurrencyFixedVsFloatingSwap>(
        type, kUsdNominal, USDCurrency(), annual(startDate(), endDate()), 0.0325, Actual365Fixed(),
        ModifiedFollowing, /*fixedPaymentLag*/ 0, cal, kUsdNominal / kSpotFx, EURCurrency(),
        quarterly(startDate(), endDate()), eurIbor3M(), 0.0015, ModifiedFollowing,
        /*floatPaymentLag*/ 0, cal);
}

ext::shared_ptr<PricingEngine> makeEngine(const Date& npvDate = Date(),
                                          const Date& spotFXSettleDate = Date()) {
    // kSpotFx is quoted as USD per EUR; the engine wants units of domestic per
    // foreign, and the domestic currency here is USD.
    const Handle<Quote> fx(ext::make_shared<SimpleQuote>(kSpotFx));
    return ext::make_shared<DiscountingConstNotionalCrossCurrencySwapEngine>(
        USDCurrency(), usdDiscount(), EURCurrency(), eurDiscount(), fx, ext::nullopt, Date(),
        npvDate, spotFXSettleDate);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- base class: fixed vs fixed ---------------------------------------
    {
        auto swap = makeFixFix();
        swap->setPricingEngine(makeEngine());
        emitCase("fix_fix", *swap, "", true);
    }

    // Same instrument, but the FX quote settles later than the curve reference
    // date, which switches on the engine's forward-FX adjustment.
    {
        auto swap = makeFixFix();
        swap->setPricingEngine(makeEngine(Date(), Date(11, September, 2019)));
        emitCase("fix_fix_fwd_fx_settle", *swap, "", true);
    }

    // Same instrument, discounted to a later NPV date.
    {
        auto swap = makeFixFix();
        swap->setPricingEngine(makeEngine(Date(11, March, 2019)));
        emitCase("fix_fix_forward_npv_date", *swap, "", true);
    }

    // --- basis swap: float vs float ---------------------------------------
    {
        auto swap = makeBasis(/*overnight*/ false);
        swap->setPricingEngine(makeEngine());
        const std::string extra = member("fair_pay_spread", swap->fairPaySpread()) +
                                  member("fair_rec_spread", swap->fairRecSpread());
        emitCase("basis_ibor", *swap, extra, true);
    }

    // Overnight legs, both with a two-business-day payment lag.
    {
        auto swap = makeBasis(/*overnight*/ true);
        swap->setPricingEngine(makeEngine());
        const std::string extra = member("fair_pay_spread", swap->fairPaySpread()) +
                                  member("fair_rec_spread", swap->fairRecSpread());
        emitCase("basis_overnight", *swap, extra, true);
    }

    // --- fixed vs floating -------------------------------------------------
    const std::vector<std::pair<std::string, Swap::Type>> types = {{"payer", Swap::Payer},
                                                                  {"receiver", Swap::Receiver}};
    for (Size i = 0; i < types.size(); ++i) {
        auto swap = makeFixFloat(types[i].second);
        swap->setPricingEngine(makeEngine());
        const std::string extra =
            member("fair_rate", swap->fairRate()) + member("fair_spread", swap->fairSpread());
        emitCase("fixed_vs_floating_" + types[i].first, *swap, extra, i + 1 < types.size());
    }

    std::cout << "}\n";
    return 0;
}
