// migration-harness/cpp/probes/v143_inst_makeswaps/probe.cpp
//
// Reference values for the C++ v1.43 "Make*" swap builders and the instrument
// one of them produces:
//
//   * MakeVanillaSwap        (ql/instruments/makevanillaswap.{hpp,cpp})
//   * MakeOIS                (ql/instruments/makeois.{hpp,cpp})
//   * MakeCreditDefaultSwap  (ql/instruments/makecds.{hpp,cpp})
//   * MakeMultipleResetsSwap (ql/instruments/makemultipleresetsswap.{hpp,cpp})
//   * MultipleResetsSwap     (ql/instruments/multipleresetsswap.{hpp,cpp})
//
// WHAT IS PINNED AND WHY
// ----------------------
// A builder is a bag of a dozen-plus optional arguments, each of which only
// shows up somewhere deep in the constructed instrument. The historical
// failure mode in this port is exactly that: FixedVsFloatingSwap and
// OvernightIndexedSwap each accepted a paymentLag, stored it, and never handed
// it to a leg builder, so every lagged swap paid on its accrual end dates and
// no test noticed for the life of the port.
//
// So this probe does NOT pin headline numbers. For every case it emits the
// COMPLETE cashflow listing of both legs — per flow the payment-date serial
// and the amount, and per coupon additionally the nominal, the accrual
// start/end serials, the accrual period and the rate — plus the fair
// rate/spread, both legs' BPS and NPV, and the swap NPV. An NPV alone can
// match while two errors cancel; the structure is what pins the setters.
//
// Each case is the same baseline with exactly ONE chained setter moved off its
// default, so a Python setter that is accepted and dropped cannot pass.
// Where the C++ QL_REQUIRE / QL_FAIL branches are reachable from the builder
// surface they are probed as {"raises": true}.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/makeswaps.json.

#include <ql/cashflows/coupon.hpp>
#include <ql/cashflows/rateaveraging.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/currencies/america.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/indexes/ibor/sonia.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/instruments/creditdefaultswap.hpp>
#include <ql/instruments/makecds.hpp>
#include <ql/instruments/makemultipleresetsswap.hpp>
#include <ql/instruments/makeois.hpp>
#include <ql/instruments/makevanillaswap.hpp>
#include <ql/instruments/multipleresetsswap.hpp>
#include <ql/instruments/overnightindexedswap.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/pricingengines/credit/midpointcdsengine.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/calendars/weekendsonly.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>
#include <ql/utilities/null.hpp>

#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

using namespace QuantLib;

namespace {

std::ostream& out = std::cout;

// --- fixture ---------------------------------------------------------------

// A Monday, so the spot date (today + 2 business days) is a plain Wednesday
// and the default MakeVanillaSwap/MakeOIS start dates are not weekend-driven.
const Date kToday(15, June, 2026);

// An end-of-month TARGET business day: Feb 2027 ends on Sunday the 28th, so
// the calendar end-of-month is Friday the 26th. Used for the end-of-month
// cases, which only bite when the start date IS the calendar end of month.
const Date kEomStart(26, February, 2027);

const Actual365Fixed dcA365;
const Actual360 dc360;

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, r, dcA365));
}

// Forecast curve (3%) and a DIFFERENT discount curve (2.5%), so
// withDiscountingTermStructure / withPricingEngine change the numbers.
Handle<YieldTermStructure> fwdCurve() {
    static Handle<YieldTermStructure> h = flatCurve(0.03);
    return h;
}
Handle<YieldTermStructure> discCurve() {
    static Handle<YieldTermStructure> h = flatCurve(0.025);
    return h;
}

ext::shared_ptr<IborIndex> euribor6m() {
    return ext::make_shared<Euribor6M>(fwdCurve());
}
ext::shared_ptr<IborIndex> euribor3m() {
    return ext::make_shared<Euribor3M>(fwdCurve());
}
ext::shared_ptr<OvernightIndex> sofr() {
    return ext::make_shared<Sofr>(fwdCurve());
}

// A non-default Claim for withClaim(). FaceValueAccrualClaim would drag a whole
// reference Bond into the fixture; a trivial subclass pins the same thing — that
// the builder's claim really reaches the instrument's default leg.
class HalfNotionalClaim : public Claim {
  public:
    Real amount(const Date& d, Real notional, Real recoveryRate) const override {
        return 0.5 * notional * (1.0 - recoveryRate);
    }
};

// A generic index in a given currency, to pin MakeVanillaSwap's
// currency-driven fixed-leg tenor / day-count defaults without depending on a
// specific index class.
ext::shared_ptr<IborIndex> genericIndex(const Currency& ccy, const Period& tenor) {
    return ext::make_shared<IborIndex>("Generic", tenor, 2, ccy, TARGET(), ModifiedFollowing,
                                       false, Actual360(), fwdCurve());
}

// --- JSON emitters ---------------------------------------------------------
//
// Every entry emits a trailing comma; a "_sentinel" member closes the object.
// That keeps the emitters composable with no comma bookkeeping.

void emitFlow(const ext::shared_ptr<CashFlow>& cf) {
    out << "{\"pay\": " << cf->date().serialNumber() << ", \"amount\": " << cf->amount();
    auto c = ext::dynamic_pointer_cast<Coupon>(cf);
    if (c != nullptr) {
        out << ", \"nominal\": " << c->nominal()
            << ", \"start\": " << c->accrualStartDate().serialNumber()
            << ", \"end\": " << c->accrualEndDate().serialNumber()
            << ", \"accrual\": " << c->accrualPeriod() << ", \"rate\": " << c->rate();
    }
    out << "}";
}

// Some result accessors QL_REQUIRE that the engine actually produced the
// quantity (e.g. fairUpfront is unavailable once the upfront payment date has
// passed). Emit JSON null rather than aborting the whole probe.
template <class F>
void emitMaybe(const char* name, F f) {
    out << "    \"" << name << "\": ";
    try {
        out << f();
    } catch (const std::exception&) {
        out << "null";
    }
    out << ",\n";
}

void emitLeg(const std::string& name, const Leg& leg, bool comma) {
    out << "    \"" << name << "\": [";
    for (Size i = 0; i < leg.size(); ++i) {
        if (i != 0)
            out << ", ";
        emitFlow(leg[i]);
    }
    out << "]" << (comma ? "," : "") << "\n";
}

void emitSwap(const std::string& key, const ext::shared_ptr<FixedVsFloatingSwap>& s) {
    out << "  \"" << key << "\": {\n"
        << "    \"type\": " << static_cast<int>(s->type()) << ",\n"
        << "    \"start\": " << s->startDate().serialNumber() << ",\n"
        << "    \"maturity\": " << s->maturityDate().serialNumber() << ",\n"
        << "    \"fixed_rate\": " << s->fixedRate() << ",\n"
        << "    \"spread\": " << s->spread() << ",\n"
        << "    \"payment_convention\": " << static_cast<int>(s->paymentConvention()) << ",\n";
    emitMaybe("fair_rate", [&] { return s->fairRate(); });
    emitMaybe("fair_spread", [&] { return s->fairSpread(); });
    out << "    \"fixed_leg_bps\": " << s->fixedLegBPS() << ",\n"
        << "    \"fixed_leg_npv\": " << s->fixedLegNPV() << ",\n"
        << "    \"floating_leg_bps\": " << s->floatingLegBPS() << ",\n"
        << "    \"floating_leg_npv\": " << s->floatingLegNPV() << ",\n"
        << "    \"npv\": " << s->NPV() << ",\n";
    emitLeg("fixed_leg", s->fixedLeg(), true);
    emitLeg("floating_leg", s->floatingLeg(), false);
    out << "  },\n";
}

// The builders expose their result only through a conversion operator, and a
// conversion straight to shared_ptr<FixedVsFloatingSwap> would need two
// user-defined conversions. One thin overload per builder does it in one step.
void emitSwap(const std::string& key, const MakeVanillaSwap& b) {
    ext::shared_ptr<VanillaSwap> s = b;
    emitSwap(key, s);
}
void emitSwap(const std::string& key, const MakeOIS& b) {
    ext::shared_ptr<OvernightIndexedSwap> s = b;
    emitSwap(key, s);
}
void emitSwap(const std::string& key, const MakeMultipleResetsSwap& b) {
    ext::shared_ptr<MultipleResetsSwap> s = b;
    emitSwap(key, s);
}

void emitCds(const std::string& key, const ext::shared_ptr<CreditDefaultSwap>& cds) {
    out << "  \"" << key << "\": {\n"
        << "    \"side\": " << static_cast<int>(cds->side()) << ",\n"
        << "    \"notional\": " << cds->notional() << ",\n"
        << "    \"running_spread\": " << cds->runningSpread() << ",\n"
        << "    \"upfront\": " << (cds->upfront() ? *cds->upfront() : 0.0) << ",\n"
        << "    \"has_upfront\": " << (cds->upfront() ? "true" : "false") << ",\n"
        << "    \"settles_accrual\": " << (cds->settlesAccrual() ? "true" : "false") << ",\n"
        << "    \"pays_at_default_time\": " << (cds->paysAtDefaultTime() ? "true" : "false")
        << ",\n"
        << "    \"rebates_accrual\": " << (cds->rebatesAccrual() ? "true" : "false") << ",\n"
        << "    \"protection_start\": " << cds->protectionStartDate().serialNumber() << ",\n"
        << "    \"protection_end\": " << cds->protectionEndDate().serialNumber() << ",\n"
        << "    \"trade_date\": " << cds->tradeDate().serialNumber() << ",\n"
        << "    \"cash_settlement_days\": " << cds->cashSettlementDays() << ",\n"
        << "    \"upfront_pay\": " << cds->upfrontPayment()->date().serialNumber() << ",\n"
        << "    \"upfront_amount\": " << cds->upfrontPayment()->amount() << ",\n";
    if (cds->accrualRebate() != nullptr) {
        out << "    \"rebate_pay\": " << cds->accrualRebate()->date().serialNumber() << ",\n"
            << "    \"rebate_amount\": " << cds->accrualRebate()->amount() << ",\n";
    }
    out << "    \"npv\": " << cds->NPV() << ",\n";
    emitMaybe("fair_spread", [&] { return cds->fairSpread(); });
    emitMaybe("fair_upfront", [&] { return cds->fairUpfront(); });
    out << "    \"coupon_leg_bps\": " << cds->couponLegBPS() << ",\n"
        << "    \"coupon_leg_npv\": " << cds->couponLegNPV() << ",\n"
        << "    \"default_leg_npv\": " << cds->defaultLegNPV() << ",\n"
        << "    \"upfront_npv\": " << cds->upfrontNPV() << ",\n"
        << "    \"accrual_rebate_npv\": " << cds->accrualRebateNPV() << ",\n";
    emitLeg("coupons", cds->coupons(), false);
    out << "  },\n";
}

void emitCds(const std::string& key, const MakeCreditDefaultSwap& b) {
    ext::shared_ptr<CreditDefaultSwap> cds = b;
    emitCds(key, cds);
}

void emitRaises(const std::string& key, const std::function<void()>& body) {
    bool raised = false;
    try {
        body();
    } catch (const std::exception&) {
        raised = true;
    }
    out << "  \"" << key << "\": {\"raises\": " << (raised ? "true" : "false") << "},\n";
}

void emitDate(const std::string& key, const Date& d) {
    out << "  \"" << key << "\": " << d.serialNumber() << ",\n";
}

// ===========================================================================
// MakeVanillaSwap
// ===========================================================================

// Baseline: 5Y EUR vs Euribor6M, fixed rate solved for (fixedRate == Null).
MakeVanillaSwap vsBase() { return MakeVanillaSwap(Period(5, Years), euribor6m()); }

void blockVanillaSwap() {
    // --- baseline + the two constructor arguments beyond the required pair
    emitSwap("vs_base", vsBase());
    emitSwap("vs_ctor_fixed_rate", MakeVanillaSwap(Period(5, Years), euribor6m(), 0.02));
    emitSwap("vs_ctor_forward_start",
             MakeVanillaSwap(Period(5, Years), euribor6m(), Null<Rate>(), Period(3, Months)));

    // --- type / nominal
    emitSwap("vs_receive_fixed", vsBase().receiveFixed());
    emitSwap("vs_with_type_receiver", vsBase().withType(Swap::Receiver));
    emitSwap("vs_nominal", vsBase().withNominal(1.0e6));

    // --- start / end date resolution
    emitSwap("vs_settlement_days", vsBase().withSettlementDays(5));
    emitSwap("vs_effective_date", vsBase().withEffectiveDate(Date(1, July, 2026)));
    emitSwap("vs_termination_date", vsBase().withTerminationDate(Date(17, June, 2031)));
    // withTerminationDate clears swapTenor_, so the currency-driven fixed-leg
    // default tenor is inferred from the actual swap length (makevanillaswap.cpp
    // "approximate months = days * 12/365"). A GBP index makes that inference
    // observable: <= 1Y gives an annual fixed leg, > 1Y a semiannual one.
    emitSwap("vs_termination_date_gbp_short",
             MakeVanillaSwap(Period(5, Years), genericIndex(GBPCurrency(), Period(6, Months)))
                 .withEffectiveDate(Date(17, June, 2026))
                 .withTerminationDate(Date(17, June, 2027)));
    emitSwap("vs_termination_date_gbp_long",
             MakeVanillaSwap(Period(5, Years), genericIndex(GBPCurrency(), Period(6, Months)))
                 .withEffectiveDate(Date(17, June, 2026))
                 .withTerminationDate(Date(17, June, 2029)));

    // --- rules / conventions
    emitSwap("vs_rule", vsBase().withRule(DateGeneration::Forward));
    emitSwap("vs_payment_convention", vsBase().withPaymentConvention(Preceding));

    // --- fixed leg
    emitSwap("vs_fixed_leg_tenor", vsBase().withFixedLegTenor(Period(6, Months)));
    emitSwap("vs_fixed_leg_calendar", vsBase().withFixedLegCalendar(UnitedKingdom()));
    emitSwap("vs_fixed_leg_convention", vsBase().withFixedLegConvention(Preceding));
    emitSwap("vs_fixed_leg_termination_convention",
             vsBase().withFixedLegTerminationDateConvention(Preceding));
    emitSwap("vs_fixed_leg_rule", vsBase().withFixedLegRule(DateGeneration::Forward));
    emitSwap("vs_fixed_leg_day_count", vsBase().withFixedLegDayCount(Actual360()));
    // Stub-period anchors: an explicit first / next-to-last date on the annual
    // fixed leg of a swap running 17-Jun-2026 → 17-Jun-2031.
    emitSwap("vs_fixed_leg_first_date", vsBase()
                                            .withEffectiveDate(Date(17, June, 2026))
                                            .withFixedLegFirstDate(Date(17, December, 2026)));
    emitSwap("vs_fixed_leg_next_to_last_date",
             vsBase()
                 .withEffectiveDate(Date(17, June, 2026))
                 .withFixedLegNextToLastDate(Date(17, December, 2030)));

    // --- floating leg
    emitSwap("vs_float_leg_tenor", vsBase().withFloatingLegTenor(Period(3, Months)));
    emitSwap("vs_float_leg_calendar", vsBase().withFloatingLegCalendar(UnitedKingdom()));
    emitSwap("vs_float_leg_convention", vsBase().withFloatingLegConvention(Preceding));
    emitSwap("vs_float_leg_termination_convention",
             vsBase().withFloatingLegTerminationDateConvention(Preceding));
    emitSwap("vs_float_leg_rule", vsBase().withFloatingLegRule(DateGeneration::Forward));
    emitSwap("vs_float_leg_day_count", vsBase().withFloatingLegDayCount(Actual365Fixed()));
    emitSwap("vs_float_leg_spread", vsBase().withFloatingLegSpread(0.0025));
    emitSwap("vs_float_leg_first_date", vsBase()
                                            .withEffectiveDate(Date(17, June, 2026))
                                            .withFloatingLegFirstDate(Date(17, September, 2026)));
    emitSwap("vs_float_leg_next_to_last_date",
             vsBase()
                 .withEffectiveDate(Date(17, June, 2026))
                 .withFloatingLegNextToLastDate(Date(17, March, 2031)));

    // --- end-of-month family (only observable from an end-of-month start)
    emitSwap("vs_eom_none", vsBase().withEffectiveDate(kEomStart));
    emitSwap("vs_eom_fixed",
             vsBase().withEffectiveDate(kEomStart).withFixedLegEndOfMonth(true));
    emitSwap("vs_eom_float",
             vsBase().withEffectiveDate(kEomStart).withFloatingLegEndOfMonth(true));
    emitSwap("vs_eom_maturity",
             vsBase().withEffectiveDate(kEomStart).withMaturityEndOfMonth(true));

    // --- discounting / engine
    emitSwap("vs_discounting_ts", vsBase().withDiscountingTermStructure(discCurve()));
    emitSwap("vs_pricing_engine",
             vsBase().withPricingEngine(
                 ext::make_shared<DiscountingSwapEngine>(discCurve(), false)));

    // --- indexed vs at-par coupons. Only observable when the floating accrual
    // period differs from the index tenor, so run a 3M float leg off a 6M index.
    emitSwap("vs_coupons_default", vsBase().withFloatingLegTenor(Period(3, Months)));
    emitSwap("vs_indexed_coupons",
             vsBase().withFloatingLegTenor(Period(3, Months)).withIndexedCoupons(true));
    emitSwap("vs_at_par_coupons",
             vsBase().withFloatingLegTenor(Period(3, Months)).withAtParCoupons(true));

    // --- currency-driven fixed-leg defaults
    emitSwap("vs_ccy_usd",
             MakeVanillaSwap(Period(5, Years), genericIndex(USDCurrency(), Period(3, Months))));
    emitSwap("vs_ccy_gbp_1y",
             MakeVanillaSwap(Period(1, Years), genericIndex(GBPCurrency(), Period(6, Months))));
    emitSwap("vs_ccy_gbp_5y",
             MakeVanillaSwap(Period(5, Years), genericIndex(GBPCurrency(), Period(6, Months))));
    emitSwap("vs_ccy_chf",
             MakeVanillaSwap(Period(5, Years), genericIndex(CHFCurrency(), Period(6, Months))));

    // --- QL_REQUIRE / QL_FAIL branches
    emitRaises("vs_raises_effective_and_settlement", [] {
        ext::shared_ptr<VanillaSwap> s =
            vsBase().withEffectiveDate(Date(1, July, 2026)).withSettlementDays(2);
    });
    emitRaises("vs_raises_unknown_currency", [] {
        ext::shared_ptr<VanillaSwap> s =
            MakeVanillaSwap(Period(5, Years), genericIndex(CADCurrency(), Period(3, Months)));
    });
    emitRaises("vs_raises_null_term_structure", [] {
        auto idx = ext::make_shared<Euribor6M>();  // no forwarding curve
        ext::shared_ptr<VanillaSwap> s = MakeVanillaSwap(Period(5, Years), idx);
    });
}

// ===========================================================================
// MakeOIS
// ===========================================================================

// 2Y so the compounded overnight legs stay cheap to reproduce in Python.
MakeOIS oisBase() { return MakeOIS(Period(2, Years), sofr()); }

void blockOis() {
    emitSwap("ois_base", oisBase());
    emitSwap("ois_ctor_fixed_rate", MakeOIS(Period(2, Years), sofr(), 0.02));
    emitSwap("ois_ctor_forward_start",
             MakeOIS(Period(2, Years), sofr(), Null<Rate>(), Period(3, Months)));

    emitSwap("ois_receive_fixed", oisBase().receiveFixed());
    emitSwap("ois_with_type_receiver", oisBase().withType(Swap::Receiver));
    emitSwap("ois_nominal", oisBase().withNominal(1.0e6));

    emitSwap("ois_settlement_days", oisBase().withSettlementDays(5));
    emitSwap("ois_effective_date", oisBase().withEffectiveDate(Date(1, July, 2026)));
    emitSwap("ois_termination_date", oisBase().withTerminationDate(Date(17, June, 2028)));

    emitSwap("ois_rule", oisBase().withRule(DateGeneration::Forward));
    emitSwap("ois_fixed_leg_rule", oisBase().withFixedLegRule(DateGeneration::Forward));
    emitSwap("ois_overnight_leg_rule", oisBase().withOvernightLegRule(DateGeneration::Forward));
    // A Zero rule collapses the leg to a single period AND forces Once, even
    // though the payment frequency was left at its Annual default.
    emitSwap("ois_fixed_leg_rule_zero", oisBase().withFixedLegRule(DateGeneration::Zero));
    emitSwap("ois_overnight_leg_rule_zero",
             oisBase().withOvernightLegRule(DateGeneration::Zero));

    emitSwap("ois_payment_frequency", oisBase().withPaymentFrequency(Semiannual));
    emitSwap("ois_fixed_leg_payment_frequency",
             oisBase().withFixedLegPaymentFrequency(Semiannual));
    emitSwap("ois_overnight_leg_payment_frequency",
             oisBase().withOvernightLegPaymentFrequency(Semiannual));
    emitSwap("ois_payment_frequency_once", oisBase().withPaymentFrequency(Once));

    emitSwap("ois_payment_adjustment", oisBase().withPaymentAdjustment(Preceding));
    // The defect this cluster exists to prevent: a NON-ZERO payment lag.
    emitSwap("ois_payment_lag", oisBase().withPaymentLag(2));
    emitSwap("ois_payment_calendar", oisBase().withPaymentCalendar(UnitedKingdom()));

    emitSwap("ois_calendar", oisBase().withCalendar(UnitedKingdom()));
    emitSwap("ois_fixed_leg_calendar", oisBase().withFixedLegCalendar(UnitedKingdom()));
    emitSwap("ois_overnight_leg_calendar", oisBase().withOvernightLegCalendar(UnitedKingdom()));

    emitSwap("ois_convention", oisBase().withConvention(Preceding));
    emitSwap("ois_fixed_leg_convention", oisBase().withFixedLegConvention(Preceding));
    emitSwap("ois_overnight_leg_convention", oisBase().withOvernightLegConvention(Preceding));
    emitSwap("ois_termination_convention", oisBase().withTerminationDateConvention(Preceding));
    emitSwap("ois_fixed_leg_termination_convention",
             oisBase().withFixedLegTerminationDateConvention(Preceding));
    emitSwap("ois_overnight_leg_termination_convention",
             oisBase().withOvernightLegTerminationDateConvention(Preceding));

    // End-of-month family. MakeOIS differs from MakeVanillaSwap here: with NO
    // end-of-month setter called, all three flags default to "is the start date
    // the calendar end of month?"; calling any one of them switches all three
    // to their explicit values.
    emitSwap("ois_eom_default", oisBase().withEffectiveDate(kEomStart));
    emitSwap("ois_eom_true", oisBase().withEffectiveDate(kEomStart).withEndOfMonth(true));
    emitSwap("ois_eom_false", oisBase().withEffectiveDate(kEomStart).withEndOfMonth(false));
    emitSwap("ois_eom_fixed_leg",
             oisBase().withEffectiveDate(kEomStart).withFixedLegEndOfMonth(true));
    emitSwap("ois_eom_overnight_leg",
             oisBase().withEffectiveDate(kEomStart).withOvernightLegEndOfMonth(true));
    emitSwap("ois_eom_maturity",
             oisBase().withEffectiveDate(kEomStart).withMaturityEndOfMonth(true));

    emitSwap("ois_fixed_leg_day_count", oisBase().withFixedLegDayCount(Actual365Fixed()));
    emitSwap("ois_overnight_leg_spread", oisBase().withOvernightLegSpread(0.0025));

    emitSwap("ois_discounting_ts", oisBase().withDiscountingTermStructure(discCurve()));
    emitSwap("ois_pricing_engine",
             oisBase().withPricingEngine(
                 ext::make_shared<DiscountingSwapEngine>(discCurve(), false)));

    emitSwap("ois_telescopic_value_dates", oisBase().withTelescopicValueDates(true));
    emitSwap("ois_averaging_simple", oisBase().withAveragingMethod(RateAveraging::Simple));
    emitSwap("ois_lookback_days", oisBase().withLookbackDays(2));
    emitSwap("ois_lockout_days", oisBase().withLockoutDays(2));
    emitSwap("ois_observation_shift",
             oisBase().withLookbackDays(2).withObservationShift(true));

    // Per-index default settlement days: Sonia 0, everything else (Sofr here) 2.
    emitSwap("ois_sonia_default_spot",
             MakeOIS(Period(2, Years), ext::make_shared<Sonia>(fwdCurve())));

    emitRaises("ois_raises_effective_and_settlement", [] {
        ext::shared_ptr<OvernightIndexedSwap> s =
            oisBase().withEffectiveDate(Date(1, July, 2026)).withSettlementDays(2);
    });
    emitRaises("ois_raises_null_term_structure", [] {
        ext::shared_ptr<OvernightIndexedSwap> s =
            MakeOIS(Period(2, Years), ext::make_shared<Sofr>());
    });
}

// ===========================================================================
// MakeCreditDefaultSwap
// ===========================================================================

ext::shared_ptr<PricingEngine> cdsEngine() {
    Handle<DefaultProbabilityTermStructure> prob(
        ext::make_shared<FlatHazardRate>(kToday, 0.02, dcA365));
    return ext::make_shared<MidPointCdsEngine>(prob, 0.4, discCurve());
}

MakeCreditDefaultSwap cdsBase() {
    return MakeCreditDefaultSwap(Period(5, Years), 0.01).withPricingEngine(cdsEngine());
}

void blockCds() {
    emitCds("cds_base", cdsBase());

    // The other two constructor overloads.
    emitCds("cds_ctor_term_date",
            MakeCreditDefaultSwap(Date(20, June, 2031), 0.01).withPricingEngine(cdsEngine()));
    Schedule cdsSchedule(Date(20, June, 2026), Date(20, June, 2029), Period(3, Months),
                         WeekendsOnly(), Following, Unadjusted, DateGeneration::CDS, false);
    emitCds("cds_ctor_schedule",
            MakeCreditDefaultSwap(cdsSchedule, 0.01).withPricingEngine(cdsEngine()));

    emitCds("cds_side_seller", cdsBase().withSide(Protection::Seller));
    emitCds("cds_nominal", cdsBase().withNominal(1.0e7));
    emitCds("cds_upfront_rate", cdsBase().withUpfrontRate(0.05));
    emitCds("cds_coupon_tenor", cdsBase().withCouponTenor(Period(6, Months)));
    emitCds("cds_rule_cds2015", cdsBase().withDateGenerationRule(DateGeneration::CDS2015));
    emitCds("cds_rule_old_cds", cdsBase().withDateGenerationRule(DateGeneration::OldCDS));
    emitCds("cds_rule_backward", cdsBase().withDateGenerationRule(DateGeneration::Backward));
    emitCds("cds_convention", cdsBase().withConvention(ModifiedFollowing));
    emitCds("cds_day_counter", cdsBase().withDayCounter(Actual365Fixed()));
    emitCds("cds_settle_accrual_false", cdsBase().settleAccrual(false));
    emitCds("cds_pay_at_default_false", cdsBase().payAtDefaultTime(false));
    emitCds("cds_protection_start", cdsBase().withProtectionStart(Date(10, June, 2026)));
    emitCds("cds_upfront_date",
            cdsBase().withUpfrontRate(0.05).withUpfrontDate(Date(25, June, 2026)));
    // An upfront DATE with a ZERO upfront rate still moves the accrual-rebate
    // payment date — the case a "only wire the upfront date when the rate is
    // non-zero" shortcut silently drops.
    emitCds("cds_upfront_date_zero_rate", cdsBase().withUpfrontDate(Date(25, June, 2026)));
    emitCds("cds_claim", cdsBase().withClaim(ext::make_shared<HalfNotionalClaim>()));
    emitCds("cds_last_period_day_counter", cdsBase().withLastPeriodDayCounter(Actual360(false)));
    emitCds("cds_rebate_accrual_false", cdsBase().rebateAccrual(false));
    emitCds("cds_trade_date", cdsBase().withTradeDate(Date(10, June, 2026)));
    emitCds("cds_cash_settlement_days", cdsBase().withCashSettlementDays(5));

    // cdsMaturity, the free function MakeCreditDefaultSwap uses to turn a tenor
    // into a standard CDS maturity, pinned directly for the three rules that
    // accept it plus its two QL_REQUIRE branches.
    emitDate("cds_maturity_5y_cds", cdsMaturity(kToday, Period(5, Years), DateGeneration::CDS));
    emitDate("cds_maturity_5y_cds2015",
             cdsMaturity(kToday, Period(5, Years), DateGeneration::CDS2015));
    emitDate("cds_maturity_5y_old_cds",
             cdsMaturity(kToday, Period(5, Years), DateGeneration::OldCDS));
    emitDate("cds_maturity_3m_cds", cdsMaturity(kToday, Period(3, Months), DateGeneration::CDS));
    // 20-Dec anchor + CDS2015 rolls the anchor back one quarter.
    emitDate("cds_maturity_dec_anchor_cds2015",
             cdsMaturity(Date(15, January, 2027), Period(5, Years), DateGeneration::CDS2015));
    emitDate("cds_maturity_dec_anchor_cds",
             cdsMaturity(Date(15, January, 2027), Period(5, Years), DateGeneration::CDS));
    emitRaises("cds_maturity_raises_bad_rule",
               [] { cdsMaturity(kToday, Period(5, Years), DateGeneration::Backward); });
    emitRaises("cds_maturity_raises_bad_tenor",
               [] { cdsMaturity(kToday, Period(4, Months), DateGeneration::CDS); });
    emitRaises("cds_maturity_raises_zero_old_cds",
               [] { cdsMaturity(kToday, Period(0, Months), DateGeneration::OldCDS); });
}

// ===========================================================================
// MakeMultipleResetsSwap / MultipleResetsSwap
// ===========================================================================

// 1Y rather than 2Y: MakeMultipleResetsSwap business-day-adjusts the end date
// BEFORE generating the reset schedule, so a tenor whose unadjusted end lands on
// a weekend seeds the backward generation one roll away from the start and
// produces a short stub — i.e. an ODD number of reset periods, which then trips
// MultipleResetsSwap's "not a multiple of resetsPerCoupon" requirement. 2Y from
// the 17-Jun-2026 spot does exactly that (17-Jun-2028 is a Saturday); 1Y does
// not (17-Jun-2027 is a Thursday). The 2Y trap is pinned below as
// mrs_raises_adjusted_end_stub.
MakeMultipleResetsSwap mrsBase() {
    return MakeMultipleResetsSwap(Period(1, Years), euribor3m(), 2);
}

void blockMultipleResets() {
    emitSwap("mrs_base", mrsBase());

    emitSwap("mrs_receive_fixed", mrsBase().receiveFixed());
    emitSwap("mrs_with_type_receiver", mrsBase().withType(Swap::Receiver));
    emitSwap("mrs_nominal", mrsBase().withNominal(1.0e6));
    emitSwap("mrs_fixed_rate", mrsBase().withFixedRate(0.02));
    emitSwap("mrs_settlement_days", mrsBase().withSettlementDays(5));
    emitSwap("mrs_effective_date", mrsBase().withEffectiveDate(Date(1, July, 2026)));
    emitSwap("mrs_termination_date", mrsBase().withTerminationDate(Date(17, June, 2027)));
    emitSwap("mrs_forward_start", mrsBase().withForwardStart(Period(3, Months)));
    emitSwap("mrs_fixed_leg_frequency", mrsBase().withFixedLegFrequency(Annual));
    emitSwap("mrs_fixed_leg_day_count", mrsBase().withFixedLegDayCount(Actual365Fixed()));
    emitSwap("mrs_fixed_leg_convention", mrsBase().withFixedLegConvention(Preceding));
    emitSwap("mrs_float_leg_spread", mrsBase().withFloatingLegSpread(0.0025));
    emitSwap("mrs_averaging_simple", mrsBase().withAveragingMethod(RateAveraging::Simple));
    emitSwap("mrs_discounting_ts", mrsBase().withDiscountingTermStructure(discCurve()));
    emitSwap("mrs_pricing_engine",
             mrsBase().withPricingEngine(
                 ext::make_shared<DiscountingSwapEngine>(discCurve(), false)));
    emitSwap("mrs_resets_per_coupon_4",
             MakeMultipleResetsSwap(Period(1, Years), euribor3m(), 4));

    emitRaises("mrs_raises_adjusted_end_stub", [] {
        ext::shared_ptr<MultipleResetsSwap> s =
            MakeMultipleResetsSwap(Period(2, Years), euribor3m(), 2);
        (void)s;
    });
    emitRaises("mrs_raises_effective_and_settlement", [] {
        ext::shared_ptr<MultipleResetsSwap> s =
            mrsBase().withEffectiveDate(Date(1, July, 2026)).withSettlementDays(2);
    });
    emitRaises("mrs_raises_null_term_structure", [] {
        ext::shared_ptr<MultipleResetsSwap> s =
            MakeMultipleResetsSwap(Period(1, Years), ext::make_shared<Euribor3M>(), 2);
        (void)s;
    });

    // --- MultipleResetsSwap built directly, to reach the three constructor
    // arguments MakeMultipleResetsSwap does not expose: paymentConvention,
    // paymentLag and paymentCalendar.
    // A directly-built Schedule seeds backward generation from the UNADJUSTED
    // termination date, so 2Y here really is 8 quarterly reset periods.
    const Date start(17, June, 2026);
    const Date end(17, June, 2028);
    TARGET cal;
    Schedule fixedSchedule(start, end, Period(6, Months), cal, ModifiedFollowing,
                           ModifiedFollowing, DateGeneration::Backward, false);
    Schedule resetSchedule(start, end, Period(3, Months), cal, ModifiedFollowing,
                           ModifiedFollowing, DateGeneration::Backward, false);
    auto engine = ext::make_shared<DiscountingSwapEngine>(fwdCurve(), false);

    auto direct = [&](ext::optional<BusinessDayConvention> conv, Integer lag,
                      const Calendar& payCal, RateAveraging::Type avg) {
        auto s = ext::make_shared<MultipleResetsSwap>(Swap::Payer, 1.0e6, fixedSchedule, 0.02,
                                                      dc360, resetSchedule, euribor3m(), 2, 0.0,
                                                      avg, conv, lag, payCal);
        s->setPricingEngine(engine);
        return s;
    };
    emitSwap("mrs_direct_base",
             direct(ext::nullopt, 0, Calendar(), RateAveraging::Compound));
    emitSwap("mrs_direct_payment_convention",
             direct(ext::optional<BusinessDayConvention>(Preceding), 0, Calendar(),
                    RateAveraging::Compound));
    // The defect this cluster exists to prevent, on the instrument itself.
    emitSwap("mrs_direct_payment_lag",
             direct(ext::nullopt, 3, Calendar(), RateAveraging::Compound));
    emitSwap("mrs_direct_payment_calendar",
             direct(ext::nullopt, 3, UnitedKingdom(), RateAveraging::Compound));
    emitSwap("mrs_direct_averaging_simple",
             direct(ext::nullopt, 0, Calendar(), RateAveraging::Simple));

    // (fullResetSchedule.size() - 1) must be a multiple of resetsPerCoupon.
    emitRaises("mrs_direct_raises_not_a_multiple", [&] {
        MultipleResetsSwap s(Swap::Payer, 1.0e6, fixedSchedule, 0.02, dc360, resetSchedule,
                             euribor3m(), 3);
    });
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"today\": " << kToday.serialNumber() << ",\n";
    out << "  \"eom_start\": " << kEomStart.serialNumber() << ",\n";
    out << "  \"forecast_rate\": 0.03,\n";
    out << "  \"discount_rate\": 0.025,\n";
    blockVanillaSwap();
    blockOis();
    blockCds();
    blockMultipleResets();
    out << "  \"_sentinel\": 1\n";
    out << "}\n";
    return 0;
}
