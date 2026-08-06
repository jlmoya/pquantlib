// migration-harness/cpp/probes/v143_inst_makeoptions/probe.cpp
//
// Reference values for the four "Make*" option builders of C++ QuantLib v1.43:
//
//   MakeCapFloor              (ql/instruments/makecapfloor.{hpp,cpp})
//   MakeSwaption              (ql/instruments/makeswaption.{hpp,cpp})
//   MakeCms                   (ql/instruments/makecms.{hpp,cpp})
//   MakeYoYInflationCapFloor  (ql/instruments/makeyoyinflationcapfloor.{hpp,cpp})
//
// What is pinned and why
// ----------------------
// A builder is a pile of chained setters, and the failure mode this probe
// exists to catch is a setter that is *accepted and dropped*: the headline NPV
// stays right while a calendar, a date-generation rule, a day counter or a
// nominal silently does nothing. (Exactly what happened to `paymentLag` on
// FixedVsFloatingSwap / OvernightIndexedSwap: stored, never forwarded to a leg
// builder, invisible for the life of the port.)
//
// So *every* case here emits the complete leg listing — per flow the payment
// date serial and amount; per coupon additionally the nominal, accrual
// start/end serials, accrual period, fixing-date serial and rate — plus the
// strike(s), the exercise date, the instrument type and the NPV. Two errors
// can cancel inside an NPV; they cannot cancel inside the structure.
//
// Every case is a *one setter off its default* delta against the matching
// `*_default` case, so the diff between the two is the observable consequence
// of that one setter. Setters that configure pricing (withPricingEngine,
// withDiscountingTermStructure, withNominal, withCmsCouponPricer) are pinned
// by an NPV that moves.
//
// QL_REQUIRE failure branches are emitted as {"raises": true} so the port
// asserts the same *rejection*, not merely the same happy path.
//
// Determinism: evaluation date pinned to Monday 15 January 2024; a flat 3%
// Act/365F continuously-compounded curve forecasts, a flat 2.5% curve
// discounts where a distinct discount curve is called for; the YoY curve is
// flat at 2.5% so the inflation forward is exactly the curve rate at every
// fixing (which makes the ATM strike exactly 2.5% by construction, and
// therefore a meaningful check of atmRate rather than of the interpolator).
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/makeoptions.json.

#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/cmscoupon.hpp>
#include <ql/cashflows/conundrumpricer.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/yoyinflationcoupon.hpp>
#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/instruments/capfloor.hpp>
#include <ql/instruments/inflationcapfloor.hpp>
#include <ql/instruments/makecapfloor.hpp>
#include <ql/instruments/makecms.hpp>
#include <ql/instruments/makeswaption.hpp>
#include <ql/instruments/makeyoyinflationcapfloor.hpp>
#include <ql/instruments/swap.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/pricingengines/capfloor/bacheliercapfloorengine.hpp>
#include <ql/pricingengines/capfloor/blackcapfloorengine.hpp>
#include <ql/pricingengines/inflation/inflationcapfloorengines.hpp>
#include <ql/pricingengines/swaption/blackswaptionengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/inflation/interpolatedyoyinflationcurve.hpp>
#include <ql/termstructures/volatility/inflation/yoyinflationoptionletvolatilitystructure.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

const Date kToday(15, January, 2024);
const Rate kFwdRate = 0.03;
const Rate kDiscRate = 0.025;
const Volatility kCapVol = 0.20;
const Volatility kSwaptionVol = 0.16;
const Volatility kYoYVol = 0.15;
const Rate kYoYRate = 0.025;

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

void emitReal(const char* key, Real v, bool comma = true) {
    std::cout << "\"" << key << "\": ";
    if (v == Null<Real>())
        std::cout << "null";
    else
        std::cout << v;
    if (comma)
        std::cout << ", ";
}

void emitDate(const char* key, const Date& d, bool comma = true) {
    std::cout << "\"" << key << "\": ";
    if (d == Date())
        std::cout << "null";
    else
        std::cout << d.serialNumber();
    if (comma)
        std::cout << ", ";
}

void emitInt(const char* key, Integer v, bool comma = true) {
    std::cout << "\"" << key << "\": " << v;
    if (comma)
        std::cout << ", ";
}

void emitBool(const char* key, bool b, bool comma = true) {
    std::cout << "\"" << key << "\": " << (b ? "true" : "false");
    if (comma)
        std::cout << ", ";
}

Real safeAmount(const ext::shared_ptr<CashFlow>& cf) {
    try {
        return cf->amount();
    } catch (const std::exception&) {
        return Null<Real>();
    }
}

Real safeRate(const ext::shared_ptr<Coupon>& c) {
    try {
        return c->rate();
    } catch (const std::exception&) {
        return Null<Real>();
    }
}

Real safeNPV(const Instrument& inst) {
    try {
        return inst.NPV();
    } catch (const std::exception&) {
        return Null<Real>();
    }
}

Date safeFixingDate(const ext::shared_ptr<CashFlow>& cf) {
    auto frc = ext::dynamic_pointer_cast<FloatingRateCoupon>(cf);
    if (frc != nullptr) {
        try {
            return frc->fixingDate();
        } catch (const std::exception&) {
            return Date();
        }
    }
    auto inf = ext::dynamic_pointer_cast<InflationCoupon>(cf);
    if (inf != nullptr) {
        try {
            return inf->fixingDate();
        } catch (const std::exception&) {
            return Date();
        }
    }
    return Date();
}

std::string typeName(const ext::shared_ptr<CashFlow>& cf) {
    if (ext::dynamic_pointer_cast<CmsCoupon>(cf) != nullptr)
        return "CmsCoupon";
    if (ext::dynamic_pointer_cast<IborCoupon>(cf) != nullptr)
        return "IborCoupon";
    if (ext::dynamic_pointer_cast<YoYInflationCoupon>(cf) != nullptr)
        return "YoYInflationCoupon";
    if (ext::dynamic_pointer_cast<FixedRateCoupon>(cf) != nullptr)
        return "FixedRateCoupon";
    if (ext::dynamic_pointer_cast<Coupon>(cf) != nullptr)
        return "Coupon";
    return "CashFlow";
}

// The full per-flow listing. Everything a dropped setter could move.
void emitLegBody(const char* key, const Leg& leg, bool comma) {
    std::cout << "    \"" << key << "\": [";
    for (Size i = 0; i < leg.size(); ++i) {
        const ext::shared_ptr<CashFlow>& cf = leg[i];
        if (i > 0)
            std::cout << ",";
        std::cout << "\n      {";
        std::cout << "\"type\": \"" << typeName(cf) << "\", ";
        emitDate("payment_date", cf->date());
        emitReal("amount", safeAmount(cf), false);
        auto cpn = ext::dynamic_pointer_cast<Coupon>(cf);
        if (cpn != nullptr) {
            std::cout << ", ";
            emitReal("nominal", cpn->nominal());
            emitDate("accrual_start", cpn->accrualStartDate());
            emitDate("accrual_end", cpn->accrualEndDate());
            emitReal("accrual_period", cpn->accrualPeriod());
            emitDate("fixing_date", safeFixingDate(cf));
            emitReal("rate", safeRate(cpn), false);
        }
        std::cout << "}";
    }
    std::cout << "\n    ]";
    if (comma)
        std::cout << ",\n";
    else
        std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Market data
// ---------------------------------------------------------------------------

Handle<YieldTermStructure> forecastCurve() {
    static Handle<YieldTermStructure> h(ext::make_shared<FlatForward>(
        kToday, kFwdRate, Actual365Fixed(), Continuous, Annual));
    return h;
}

Handle<YieldTermStructure> discountCurve() {
    static Handle<YieldTermStructure> h(ext::make_shared<FlatForward>(
        kToday, kDiscRate, Actual365Fixed(), Continuous, Annual));
    return h;
}

ext::shared_ptr<IborIndex> euribor6m() {
    return ext::make_shared<Euribor6M>(forecastCurve());
}

ext::shared_ptr<IborIndex> euribor3m() {
    return ext::make_shared<Euribor3M>(forecastCurve());
}

ext::shared_ptr<SwapIndex> swapIndex10y() {
    return ext::make_shared<EuriborSwapIsdaFixA>(10 * Years, forecastCurve());
}

ext::shared_ptr<PricingEngine> blackCapEngine(Volatility v = kCapVol) {
    return ext::make_shared<BlackCapFloorEngine>(forecastCurve(), v, Actual365Fixed());
}

ext::shared_ptr<PricingEngine> bachelierCapEngine(Volatility v = 0.01) {
    return ext::make_shared<BachelierCapFloorEngine>(forecastCurve(), v, Actual365Fixed());
}

ext::shared_ptr<PricingEngine> blackSwaptionEngine() {
    return ext::make_shared<BlackSwaptionEngine>(forecastCurve(), kSwaptionVol,
                                                 Actual365Fixed());
}

// ---------------------------------------------------------------------------
// Emitters, one per instrument shape
// ---------------------------------------------------------------------------

void emitCapFloor(const char* key, const ext::shared_ptr<CapFloor>& cf, bool comma) {
    std::cout << "  \"" << key << "\": {\n    ";
    emitInt("type", static_cast<Integer>(cf->type()));
    std::cout << "\"cap_rates\": [";
    for (Size i = 0; i < cf->capRates().size(); ++i)
        std::cout << (i != 0U ? ", " : "") << cf->capRates()[i];
    std::cout << "], \"floor_rates\": [";
    for (Size i = 0; i < cf->floorRates().size(); ++i)
        std::cout << (i != 0U ? ", " : "") << cf->floorRates()[i];
    std::cout << "], ";
    emitDate("start_date", cf->startDate());
    emitDate("maturity_date", cf->maturityDate());
    emitReal("npv", safeNPV(*cf), false);
    std::cout << ",\n";
    emitLegBody("leg", cf->floatingLeg(), false);
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitSwaption(const char* key, const ext::shared_ptr<Swaption>& sw, bool comma) {
    const ext::shared_ptr<FixedVsFloatingSwap>& u = sw->underlying();
    std::cout << "  \"" << key << "\": {\n    ";
    emitInt("settlement_type", static_cast<Integer>(sw->settlementType()));
    emitInt("settlement_method", static_cast<Integer>(sw->settlementMethod()));
    emitInt("swap_type", static_cast<Integer>(sw->type()));
    emitDate("exercise_date", sw->exercise()->dates().front());
    emitReal("strike", u->fixedRate());
    emitReal("nominal", u->nominal());
    emitReal("npv", safeNPV(*sw), false);
    std::cout << ",\n";
    emitLegBody("fixed_leg", u->fixedLeg(), true);
    emitLegBody("floating_leg", u->floatingLeg(), false);
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitCmsSwap(const char* key, const ext::shared_ptr<Swap>& swap, bool comma) {
    std::cout << "  \"" << key << "\": {\n    ";
    emitReal("npv", safeNPV(*swap), false);
    std::cout << ",\n";
    emitLegBody("leg0", swap->leg(0), true);
    emitLegBody("leg1", swap->leg(1), false);
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitYoYCapFloor(const char* key,
                     const ext::shared_ptr<YoYInflationCapFloor>& cf,
                     bool comma) {
    std::cout << "  \"" << key << "\": {\n    ";
    emitInt("type", static_cast<Integer>(cf->type()));
    std::cout << "\"cap_rates\": [";
    for (Size i = 0; i < cf->capRates().size(); ++i)
        std::cout << (i != 0U ? ", " : "") << cf->capRates()[i];
    std::cout << "], \"floor_rates\": [";
    for (Size i = 0; i < cf->floorRates().size(); ++i)
        std::cout << (i != 0U ? ", " : "") << cf->floorRates()[i];
    std::cout << "], ";
    emitDate("start_date", cf->startDate());
    emitDate("maturity_date", cf->maturityDate());
    emitReal("npv", safeNPV(*cf), false);
    std::cout << ",\n";
    emitLegBody("leg", cf->yoyLeg(), false);
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

// A QL_REQUIRE branch: what matters is that C++ rejects it at all.
void emitRaises(const char* key, const std::function<void()>& body, bool comma) {
    bool raised = false;
    try {
        body();
    } catch (const std::exception&) {
        raised = true;
    }
    std::cout << "  \"" << key << "\": {";
    emitBool("raises", raised, false);
    std::cout << "}" << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------------------------
// MakeCapFloor
// ---------------------------------------------------------------------------

void emitMakeCapFloor() {
    const Rate k = 0.03;

    // Default: 5Y cap on Euribor6M, zero forward start. Note that a zero
    // forward start makes firstCapletExcluded_ TRUE in the constructor, so the
    // leg is 9 caplets and not 10 — that alone is worth pinning.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_default", mk, true);
    }
    // Non-zero forward start => firstCapletExcluded_ false => 10 caplets.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k, 1 * Years);
        mk.withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_forward_start_1y", mk, true);
    }
    // Floor instead of cap.
    {
        MakeCapFloor mk(CapFloor::Floor, 5 * Years, euribor6m(), k);
        mk.withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_floor", mk, true);
    }
    // withNominal — every amount scales, NPV scales.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withNominal(2.0e6).withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_nominal", mk, true);
    }
    // withEffectiveDate(d, firstCapletExcluded=false) — 10 caplets from d.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(20, March, 2024), false)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_effective_date_incl", mk, true);
    }
    // ... and the same date with firstCapletExcluded=true — 9 caplets.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(20, March, 2024), true)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_effective_date_excl", mk, true);
    }
    // withTenor — drives the *floating* leg tenor, so 3M => 20 periods.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withTenor(3 * Months).withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_tenor_3m", mk, true);
    }
    // withCalendar — needs a date the two calendars disagree about, or the
    // setter is untestable: 26-May-2024 is a Sunday, both roll it to Monday
    // 27-May, and *that* is the UK spring bank holiday but a TARGET business
    // day. Emitted with its own TARGET baseline so the delta isolates the
    // calendar and nothing else.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(26, February, 2024), false)
            .withTenor(3 * Months)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_calendar_target", mk, true);
    }
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(26, February, 2024), false)
            .withTenor(3 * Months)
            .withCalendar(UnitedKingdom())
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_calendar_uk", mk, true);
    }
    // withConvention — Following vs the index's ModifiedFollowing only differ
    // when the roll crosses a month end: 30-Nov-2028/2026/2024 are Saturdays.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(31, May, 2024), false)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_convention_modified_following", mk, true);
    }
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(31, May, 2024), false)
            .withConvention(Following)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_convention_following", mk, true);
    }
    // withTerminationDateConvention — only the last accrual end moves, so the
    // termination has to fall on a non-business day: a Monday effective date
    // after 29-Feb-2024 lands five years later on a Sunday.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(3, June, 2024), false)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_termination_convention_modified_following", mk, true);
    }
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(3, June, 2024), false)
            .withTerminationDateConvention(Unadjusted)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_termination_convention_unadjusted", mk, true);
    }
    // withRule — Backward and Forward only differ when the schedule has a
    // stub, so the coupon tenor has to divide the 5Y term unevenly (2Y does,
    // 6M does not); Backward puts the stub first, Forward puts it last.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(31, January, 2024), false)
            .withTenor(2 * Years)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_rule_backward", mk, true);
    }
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(31, January, 2024), false)
            .withTenor(2 * Years)
            .withRule(DateGeneration::Forward)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_rule_forward", mk, true);
    }
    // withEndOfMonth — start on a month end so the flag actually bites, with
    // its own flag-off baseline.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(29, February, 2024), false)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_no_end_of_month", mk, true);
    }
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(29, February, 2024), false)
            .withEndOfMonth(true)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_end_of_month", mk, true);
    }
    // Stub baseline: the same effective date with no stub dates, so
    // capfloor_first_date / capfloor_next_to_last_date isolate the stub.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(17, January, 2024), false)
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_stub_baseline", mk, true);
    }
    // withFirstDate — a short first (stub) period.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(17, January, 2024), false)
            .withFirstDate(Date(17, April, 2024))
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_first_date", mk, true);
    }
    // withNextToLastDate — a short last (stub) period.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withEffectiveDate(Date(17, January, 2024), false)
            .withNextToLastDate(Date(17, October, 2028))
            .withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_next_to_last_date", mk, true);
    }
    // withDayCount — accrual periods (and hence amounts) change.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withDayCount(Actual365Fixed()).withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_day_count", mk, true);
    }
    // asOptionlet — only the last caplet survives.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.asOptionlet(true).withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_optionlet", mk, true);
    }
    // withPricingEngine — a different vol must move the NPV and nothing else.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m(), k);
        mk.withPricingEngine(blackCapEngine(0.40));
        emitCapFloor("capfloor_engine_vol_40", mk, true);
    }
    // Null strike + a Black engine => ATM from CashFlows::atmRate on the
    // engine's own discount curve.
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m());
        mk.withPricingEngine(blackCapEngine());
        emitCapFloor("capfloor_atm_black", mk, true);
    }
    // Same, through a Bachelier engine (the other accepted engine type).
    {
        MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m());
        mk.withPricingEngine(bachelierCapEngine());
        emitCapFloor("capfloor_atm_bachelier", mk, true);
    }
    // QL_FAIL branch: null strike with no Black/Bachelier engine attached.
    emitRaises(
        "capfloor_atm_without_engine_raises",
        [] {
            MakeCapFloor mk(CapFloor::Cap, 5 * Years, euribor6m());
            ext::shared_ptr<CapFloor> cf = mk;
        },
        true);
}

// ---------------------------------------------------------------------------
// MakeSwaption
// ---------------------------------------------------------------------------

void emitMakeSwaption() {
    const Rate k = 0.03;

    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_default", mk, true);
    }
    // Null strike => ATM = fairRate of the index's underlying swap.
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years);
        mk.withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_atm", mk, true);
    }
    // The fixing-date constructor bypasses optionTenor/optionConvention; the
    // date is deliberately NOT the one the 5Y tenor would produce.
    {
        MakeSwaption mk(swapIndex10y(), Date(20, June, 2029), k);
        mk.withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_fixing_date_ctor", mk, true);
    }
    // withNominal — legs scale, NPV scales.
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withNominal(1.0e6).withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_nominal", mk, true);
    }
    // withSettlementType / withSettlementMethod — cash settlement changes the
    // annuity the engine uses, so the NPV moves.
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withSettlementType(Settlement::Cash)
            .withSettlementMethod(Settlement::ParYieldCurve)
            .withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_cash_par_yield", mk, true);
    }
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withSettlementType(Settlement::Cash)
            .withSettlementMethod(Settlement::CollateralizedCashPrice)
            .withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_cash_collateralized", mk, true);
    }
    // withOptionConvention — moves the fixing date off ModifiedFollowing.
    // 15-Jan-2024 + 6Y3M lands on 15-Apr-2030 (a Monday); Preceding vs
    // ModifiedFollowing differ on the TARGET Easter Monday of 2030.
    {
        MakeSwaption mk(swapIndex10y(), Period(6, Years) + Period(3, Months), k);
        mk.withOptionConvention(Preceding).withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_option_convention_preceding", mk, true);
    }
    // withExerciseDate — exercise strictly earlier than the fixing date.
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withExerciseDate(Date(10, January, 2029))
            .withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_exercise_date", mk, true);
    }
    // withExerciseCalendar — needs an option tenor whose raw fixing date is a
    // holiday in exactly one calendar. 15-Jan-2024 + 19 weeks = 27-May-2024,
    // the UK spring bank holiday and a TARGET business day. Emitted with its
    // own TARGET baseline so the delta isolates the calendar.
    {
        MakeSwaption mk(swapIndex10y(), Period(19, Weeks), k);
        mk.withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_exercise_calendar_target", mk, true);
    }
    {
        MakeSwaption mk(swapIndex10y(), Period(19, Weeks), k);
        mk.withExerciseCalendar(UnitedKingdom())
            .withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_exercise_calendar_uk", mk, true);
    }
    // withUnderlyingType — receiver instead of payer.
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withUnderlyingType(Swap::Receiver).withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_receiver", mk, true);
    }
    // withIndexedCoupons / withAtParCoupons — the underlying's floating
    // coupons switch between indexed and par conventions.
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withIndexedCoupons(true).withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_indexed_coupons", mk, true);
    }
    {
        MakeSwaption mk(swapIndex10y(), 5 * Years, k);
        mk.withAtParCoupons(true).withPricingEngine(blackSwaptionEngine());
        emitSwaption("swaption_at_par_coupons", mk, true);
    }
    // QL_REQUIRE: exercise date after the fixing date.
    emitRaises(
        "swaption_exercise_after_fixing_raises",
        [&] {
            MakeSwaption mk(swapIndex10y(), 5 * Years, k);
            mk.withExerciseDate(Date(15, January, 2035));
            ext::shared_ptr<Swaption> s = mk;
        },
        true);
    // QL_REQUIRE: ATM with no forwarding term structure on the index.
    emitRaises(
        "swaption_atm_without_curve_raises",
        [] {
            auto bare = ext::make_shared<EuriborSwapIsdaFixA>(10 * Years);
            MakeSwaption mk(bare, 5 * Years);
            ext::shared_ptr<Swaption> s = mk;
        },
        true);
}

// ---------------------------------------------------------------------------
// MakeCms
// ---------------------------------------------------------------------------

ext::shared_ptr<CmsCouponPricer> haganPricer() {
    Handle<SwaptionVolatilityStructure> vol(
        ext::make_shared<ConstantSwaptionVolatility>(
            kToday, TARGET(), Following, kSwaptionVol, Actual365Fixed()));
    Handle<Quote> meanReversion(ext::make_shared<SimpleQuote>(0.01));
    return ext::make_shared<AnalyticHaganPricer>(
        vol, GFunctionFactory::Standard, meanReversion);
}

void emitMakeCms() {
    const Spread s = 0.0010;

    // Default: pay CMS (leg0 = CMS, leg1 = float), 3M CMS leg, index-tenor
    // float leg. No coupon pricer, so the CMS rate/amount is deliberately
    // absent — the port must show the same absence, not a silent zero.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        emitCmsSwap("cms_default_no_pricer", mk, true);
    }
    // withCmsCouponPricer — same structure, now priceable.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_default", mk, true);
    }
    // The 3-arg constructor takes the ibor index from the swap index.
    {
        MakeCms mk(5 * Years, swapIndex10y(), s);
        mk.withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_ctor_without_ibor_index", mk, true);
    }
    // receiveCms — legs swap order.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.receiveCms(true).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_receive", mk, true);
    }
    // withNominal.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withNominal(1.0e6).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_nominal", mk, true);
    }
    // withEffectiveDate. The three extra effective-date-only cases below are
    // the baselines the rule / end-of-month / stub-date cases are deltas
    // against, so each of those isolates its own setter.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(20, March, 2024)).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_effective_date", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(31, January, 2024)).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_effective_31jan", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(29, February, 2024))
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_effective_29feb", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(17, January, 2024)).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_effective_17jan", mk, true);
    }
    // 26-Feb: the quarterly roll lands on Sunday 26-May-2024, which both
    // calendars push to Monday 27-May — the UK spring bank holiday.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(26, February, 2024))
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_effective_26feb", mk, true);
    }
    // 31-May: the quarterly roll crosses a month end (Sat 31-Aug-2024), where
    // Following and ModifiedFollowing disagree.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(31, May, 2024)).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_effective_31may", mk, true);
    }
    // 3-Jun: five years later is a Sunday, so the termination-date convention
    // has something to adjust.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(3, June, 2024)).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_effective_3jun", mk, true);
    }
    // --- CMS-leg setters ---
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withCmsLegTenor(6 * Months).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_tenor_6m", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(26, February, 2024))
            .withCmsLegCalendar(UnitedKingdom())
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_calendar_uk", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(31, May, 2024))
            .withCmsLegConvention(Following)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_convention_following", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(3, June, 2024))
            .withCmsLegTerminationDateConvention(Unadjusted)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_termination_convention_unadjusted", mk, true);
    }
    // 5Y / 2Y leaves a stub; Backward puts it first, Forward puts it last.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withCmsLegTenor(2 * Years).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_rule_backward_2y", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withCmsLegTenor(2 * Years)
            .withCmsLegRule(DateGeneration::Forward)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_rule_forward", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(29, February, 2024))
            .withCmsLegEndOfMonth(true)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_end_of_month", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(17, January, 2024))
            .withCmsLegFirstDate(Date(17, March, 2024))
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_first_date", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(17, January, 2024))
            .withCmsLegNextToLastDate(Date(17, November, 2028))
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_next_to_last_date", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withCmsLegDayCount(Thirty360(Thirty360::BondBasis))
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_leg_day_count", mk, true);
    }
    // --- floating-leg setters ---
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withFloatingLegTenor(6 * Months).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_tenor_6m", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(26, February, 2024))
            .withFloatingLegCalendar(UnitedKingdom())
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_calendar_uk", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(31, May, 2024))
            .withFloatingLegConvention(Following)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_convention_following", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(3, June, 2024))
            .withFloatingLegTerminationDateConvention(Unadjusted)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_termination_convention_unadjusted", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withFloatingLegTenor(2 * Years).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_rule_backward_2y", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withFloatingLegTenor(2 * Years)
            .withFloatingLegRule(DateGeneration::Forward)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_rule_forward", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(29, February, 2024))
            .withFloatingLegEndOfMonth(true)
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_end_of_month", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(17, January, 2024))
            .withFloatingLegFirstDate(Date(17, March, 2024))
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_first_date", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withEffectiveDate(Date(17, January, 2024))
            .withFloatingLegNextToLastDate(Date(17, November, 2028))
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_next_to_last_date", mk, true);
    }
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withFloatingLegDayCount(Actual365Fixed()).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_float_leg_day_count", mk, true);
    }
    // withDiscountingTermStructure — replaces the whole engine, NPV moves.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withDiscountingTermStructure(discountCurve())
            .withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_discount_curve", mk, true);
    }
    // withAtmSpread — solves the float spread that zeroes the swap NPV.
    {
        MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
        mk.withAtmSpread(true).withCmsCouponPricer(haganPricer());
        emitCmsSwap("cms_atm_spread", mk, true);
    }
    // QL_REQUIRE: atm spread without a CMS coupon pricer.
    emitRaises(
        "cms_atm_spread_without_pricer_raises",
        [&] {
            MakeCms mk(5 * Years, swapIndex10y(), euribor3m(), s);
            mk.withAtmSpread(true);
            ext::shared_ptr<Swap> sw = mk;
        },
        true);
}

// ---------------------------------------------------------------------------
// MakeYoYInflationCapFloor
// ---------------------------------------------------------------------------

Handle<YoYInflationTermStructure> yoyCurve() {
    static Handle<YoYInflationTermStructure> h;
    if (h.empty()) {
        std::vector<Date> dates = {kToday - Period(3, Months),
                                   kToday + Period(30, Years)};
        std::vector<Rate> rates = {kYoYRate, kYoYRate};
        h = Handle<YoYInflationTermStructure>(
            ext::make_shared<InterpolatedYoYInflationCurve<Linear>>(
                kToday, dates, rates, Monthly, Actual365Fixed()));
    }
    return h;
}

ext::shared_ptr<YoYInflationIndex> yoyIndex() {
    return ext::make_shared<YYEUHICP>(yoyCurve());
}

ext::shared_ptr<PricingEngine> yoyBlackEngine() {
    Handle<YoYOptionletVolatilitySurface> vol(
        ext::make_shared<ConstantYoYOptionletVolatility>(
            kYoYVol, 0, TARGET(), ModifiedFollowing, Actual360(),
            Period(3, Months), Monthly, false));
    return ext::make_shared<YoYInflationBlackCapFloorEngine>(yoyIndex(), vol,
                                                             forecastCurve());
}

void emitMakeYoY() {
    const Calendar cal = TARGET();
    const Period lag(3, Months);
    const CPI::InterpolationType interp = CPI::AsIndex;

    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02).withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_default", mk, true);
    }
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Floor, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02).withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_floor", mk, true);
    }
    // withNominal — the default is 1,000,000; move it.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02).withNominal(2.5e6).withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_nominal", mk, true);
    }
    // withEffectiveDate.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02)
            .withEffectiveDate(Date(20, March, 2024))
            .withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_effective_date", mk, true);
    }
    // withFixingDays — note this only shifts the *spot* date; C++ never
    // forwards it to yoyInflationLeg, so the coupons keep fixingDays 0.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02).withFixingDays(2).withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_fixing_days", mk, true);
    }
    // withForwardStart.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02)
            .withForwardStart(Period(1, Years))
            .withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_forward_start", mk, true);
    }
    // withPaymentDayCounter — accrual periods move.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02)
            .withPaymentDayCounter(Actual360())
            .withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_payment_day_counter", mk, true);
    }
    // withPaymentAdjustment — payment dates move (schedule is Unadjusted, so
    // the roll convention is the only thing adjusting them).
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02)
            .withPaymentAdjustment(Preceding)
            .withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_payment_adjustment_preceding", mk, true);
    }
    // asOptionlet — last coupon only.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02).asOptionlet(true).withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_optionlet", mk, true);
    }
    // withPricingEngine — no engine at all means no NPV.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withStrike(0.02);
        emitYoYCapFloor("yoy_no_engine", mk, true);
    }
    // withAtmStrike — the strike becomes CashFlows::atmRate of the leg. With a
    // flat 2.5% YoY curve and gearing 1 / spread 0 that is exactly 2.5%, so
    // this pins atmRate itself rather than the interpolator.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5, cal,
                                    lag, interp);
        mk.withAtmStrike(forecastCurve()).withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_atm_strike", mk, true);
    }
    // A different length changes the number of coupons.
    {
        MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 3, cal,
                                    lag, interp);
        mk.withStrike(0.02).withPricingEngine(yoyBlackEngine());
        emitYoYCapFloor("yoy_length_3", mk, true);
    }
    // QL_REQUIRE: explicit strike after an ATM strike.
    emitRaises(
        "yoy_strike_after_atm_raises",
        [&] {
            MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5,
                                        cal, lag, interp);
            mk.withAtmStrike(forecastCurve()).withStrike(0.02);
        },
        true);
    // QL_REQUIRE: ATM strike after an explicit strike.
    emitRaises(
        "yoy_atm_after_strike_raises",
        [&] {
            MakeYoYInflationCapFloor mk(YoYInflationCapFloor::Cap, yoyIndex(), 5,
                                        cal, lag, interp);
            mk.withStrike(0.02).withAtmStrike(forecastCurve());
        },
        false);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    emitMakeCapFloor();
    emitMakeSwaption();
    emitMakeCms();
    emitMakeYoY();
    std::cout << "}\n";
    return 0;
}
