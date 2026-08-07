// migration-harness/cpp/probes/v143_experimental_coupons/probe.cpp
//
// Reference values for the ql/experimental/coupons CMS-spread coupon family
// @ v1.43:
//
//   CappedFlooredCmsSpreadCoupon   ql/experimental/coupons/cmsspreadcoupon.hpp
//   CmsSpreadLeg                   ql/experimental/coupons/cmsspreadcoupon.hpp
//   DigitalCmsSpreadCoupon         ql/experimental/coupons/digitalcmsspreadcoupon.hpp
//   DigitalCmsSpreadLeg            ql/experimental/coupons/digitalcmsspreadcoupon.hpp
//   StrippedCappedFlooredCouponLeg ql/experimental/coupons/strippedcapflooredcoupon.hpp
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// None of the five gap classes has arithmetic of its own worth speaking of:
// they are wiring. CappedFlooredCmsSpreadCoupon is a one-line delegation to
// CappedFlooredCoupon over a CmsSpreadCoupon; the three *Leg types are
// chained builders whose only real content is what `operator Leg()` feeds to
// the FloatingLeg / FloatingDigitalLeg templates in
// ql/cashflows/cashflowvectors.hpp. So "does it construct" pins nothing.
// What has to be pinned is the WIRING:
//
//   * which coupon TYPE each leg period gets (plain / capped-floored /
//     fixed-because-gearing-is-zero), period by period;
//   * the payment date each period gets (paymentAdjustment, and the
//     withZeroPayments collapse to the terminal date);
//   * the reference-period dates for irregular first/last stubs, which only
//     move if schedule.isRegular() says the stub is irregular;
//   * the sign-aware cap/floor swap CappedFlooredCoupon performs when
//     gearing < 0 (cap()/floor() and effectiveCap()/effectiveFloor() then
//     report *different* numbers from the constructor arguments);
//   * the two DIFFERENT fixed-coupon fallbacks: FloatingLeg uses
//     detail::effectiveFixedRate(spreads, caps, floors, i) with a 0.0 default
//     for the spread, FloatingDigitalLeg uses a bare
//     detail::get(spreads, i, 1.0) — default ONE, i.e. a missing spread
//     vector produces a 100% fixed coupon. That asymmetry is in v1.43 and is
//     pinned below (block G4) rather than tidied away.
//
// Everything numeric downstream comes from a LognormalCmsSpreadPricer over an
// AnalyticHaganPricer, on a flat 5% curve with a constant 16% lognormal
// swaption vol and correlation 0.5 — the same configuration the existing
// cluster_w12a probe uses, so the two references stay comparable.
//
// Dates are emitted as INTEGER SERIAL NUMBERS (emit_int) and are checked at
// tolerance.exact on the Python side; nothing about a date is a float here.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/coupons.json.

#include <ql/cashflows/capflooredcoupon.hpp>
#include <ql/cashflows/cashflowvectors.hpp>
#include <ql/cashflows/conundrumpricer.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/digitalcoupon.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/replication.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/experimental/coupons/cmsspreadcoupon.hpp>
#include <ql/experimental/coupons/digitalcmsspreadcoupon.hpp>
#include <ql/experimental/coupons/lognormalcmsspreadpricer.hpp>
#include <ql/experimental/coupons/strippedcapflooredcoupon.hpp>
#include <ql/experimental/coupons/swapspreadindex.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>
#include <ql/utilities/null.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --------------------------------------------------------------------------
// JSON emission (flat top-level object; the pquantlib harness convention)
// --------------------------------------------------------------------------

bool g_first = true;

void sep() {
    if (!g_first) std::cout << ",\n";
    g_first = false;
}

// Render a Real so it always parses back as a Python float, never an int:
// `1000000` and `-0` are legal JSON but json.load() hands them back as ints,
// which is a needless trap on the assertion side.
std::string fmtReal(Real v) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    std::string s = os.str();
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos && s.find("inf") == std::string::npos &&
        s.find("nan") == std::string::npos) {
        s += ".0";
    }
    return s;
}

void emit(const std::string& name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << fmtReal(v);
}

void emit_int(const std::string& name, long long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_bool(const std::string& name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << fmtReal(v[i]);
    }
    std::cout << "]";
}

void emit_iarr(const std::string& name, const std::vector<long long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

// A Rate that may be Null<Rate>() is emitted as a (is_null, value) pair, so
// the Python side can assert `is None` rather than comparing against
// QL_NULL_REAL (3.4e38), which is a C++ sentinel and not a number.
void emit_opt_rate(const std::string& name, Rate r) {
    bool isNull = (r == Null<Rate>());
    emit_bool(name + "_is_null", isNull);
    emit(name, isNull ? 0.0 : r);
}

// --------------------------------------------------------------------------
// Market setup — identical in shape to cluster_w12a so the two references
// can be diffed against each other.
// --------------------------------------------------------------------------

const Date today(15, January, 2024);
const Actual365Fixed dcA365;
const Actual360 dc360;
const TARGET cal;

Handle<YieldTermStructure> flatCurve(Real r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(today, r, dcA365));
}

ext::shared_ptr<SwapIndex> makeSwapIndex(const Period& tenor,
                                         const ext::shared_ptr<IborIndex>& ibor) {
    return ext::make_shared<SwapIndex>("EuriborSwapIsdaFixA", tenor, ibor->fixingDays(),
                                       ibor->currency(), ibor->fixingCalendar(),
                                       Period(1, Years), Unadjusted, ibor->dayCounter(), ibor);
}

Handle<SwaptionVolatilityStructure> constLogVol(Real v) {
    return Handle<SwaptionVolatilityStructure>(
        ext::make_shared<ConstantSwaptionVolatility>(today, cal, Following, v, dcA365,
                                                     ShiftedLognormal));
}

struct Market {
    Handle<YieldTermStructure> curve;
    ext::shared_ptr<IborIndex> ibor;
    ext::shared_ptr<SwapIndex> s10;
    ext::shared_ptr<SwapIndex> s2;
    ext::shared_ptr<SwapSpreadIndex> ssi;
    ext::shared_ptr<CmsSpreadCouponPricer> pricer;
};

Market makeMarket(Real correlation = 0.5) {
    Market m;
    m.curve = flatCurve(0.05);
    m.ibor = ext::make_shared<Euribor6M>(m.curve);
    m.s10 = makeSwapIndex(Period(10, Years), m.ibor);
    m.s2 = makeSwapIndex(Period(2, Years), m.ibor);
    m.ssi = ext::make_shared<SwapSpreadIndex>("CMS10Y-2Y", m.s10, m.s2, 1.0, -1.0);

    auto vol = constLogVol(0.16);
    Handle<Quote> zeroMeanRev(ext::make_shared<SimpleQuote>(0.0));
    Handle<Quote> corr(ext::make_shared<SimpleQuote>(correlation));
    auto cmsPricer =
        ext::make_shared<AnalyticHaganPricer>(vol, GFunctionFactory::Standard, zeroMeanRev);
    m.pricer = ext::make_shared<LognormalCmsSpreadPricer>(cmsPricer, corr, m.curve, 16);
    return m;
}

// setCouponPricer() dispatches through an AcyclicVisitor that has no
// CmsSpreadCoupon case, so it silently does nothing for these legs. Walk the
// leg by hand instead — which is also exactly what the Python port does.
void setPricerOnLeg(const Leg& leg, const ext::shared_ptr<FloatingRateCouponPricer>& p) {
    for (const auto& cf : leg) {
        if (auto c = ext::dynamic_pointer_cast<FloatingRateCoupon>(cf))
            c->setPricer(p);
    }
}

// Most-derived-first type tag for a cash flow, so the leg builders' dispatch
// (plain vs capped/floored vs fixed vs digital vs stripped) is pinned.
std::string typeTag(const ext::shared_ptr<CashFlow>& cf) {
    if (ext::dynamic_pointer_cast<StrippedCappedFlooredCoupon>(cf))
        return "StrippedCappedFlooredCoupon";
    if (ext::dynamic_pointer_cast<DigitalCmsSpreadCoupon>(cf)) return "DigitalCmsSpreadCoupon";
    if (ext::dynamic_pointer_cast<DigitalCoupon>(cf)) return "DigitalCoupon";
    if (ext::dynamic_pointer_cast<CappedFlooredCmsSpreadCoupon>(cf))
        return "CappedFlooredCmsSpreadCoupon";
    if (ext::dynamic_pointer_cast<CappedFlooredCoupon>(cf)) return "CappedFlooredCoupon";
    if (ext::dynamic_pointer_cast<CmsSpreadCoupon>(cf)) return "CmsSpreadCoupon";
    if (ext::dynamic_pointer_cast<FixedRateCoupon>(cf)) return "FixedRateCoupon";
    return "CashFlow";
}

std::string joinTypes(const Leg& leg) {
    std::ostringstream os;
    for (Size i = 0; i < leg.size(); ++i) {
        if (i) os << ",";
        os << typeTag(leg[i]);
    }
    return os.str();
}

// Everything a leg publishes: size, per-flow payment date (serial), amount,
// accrual start/end (serial), accrual period, and the type dispatch.
void emitLeg(const std::string& tag, const Leg& leg) {
    emit_int(tag + "_size", static_cast<long long>(leg.size()));
    emit_str(tag + "_types", joinTypes(leg));

    std::vector<long long> dates, accStart, accEnd, refStart, refEnd;
    std::vector<Real> amounts, accrualPeriods, nominals, rates;
    for (const auto& cf : leg) {
        dates.push_back(static_cast<long long>(cf->date().serialNumber()));
        amounts.push_back(cf->amount());
        if (auto c = ext::dynamic_pointer_cast<Coupon>(cf)) {
            accStart.push_back(static_cast<long long>(c->accrualStartDate().serialNumber()));
            accEnd.push_back(static_cast<long long>(c->accrualEndDate().serialNumber()));
            refStart.push_back(static_cast<long long>(c->referencePeriodStart().serialNumber()));
            refEnd.push_back(static_cast<long long>(c->referencePeriodEnd().serialNumber()));
            accrualPeriods.push_back(c->accrualPeriod());
            nominals.push_back(c->nominal());
            rates.push_back(c->rate());
        }
    }
    emit_iarr(tag + "_dates", dates);
    emit_arr(tag + "_amounts", amounts);
    emit_iarr(tag + "_accrual_start", accStart);
    emit_iarr(tag + "_accrual_end", accEnd);
    emit_iarr(tag + "_ref_start", refStart);
    emit_iarr(tag + "_ref_end", refEnd);
    emit_arr(tag + "_accrual_periods", accrualPeriods);
    emit_arr(tag + "_nominals", nominals);
    emit_arr(tag + "_rates", rates);
}

// --------------------------------------------------------------------------
// Block A — the market itself, so a mismatch downstream can be localised.
// --------------------------------------------------------------------------

void blockA() {
    Market m = makeMarket();
    Date start = m.curve->referenceDate() + 20 * Years;
    Date end = start + 1 * Years;

    CmsSpreadCoupon probe(end, 1.0, start, end, m.ssi->fixingDays(), m.ssi, 1.0, 0.0, start, end,
                          m.ibor->dayCounter());

    emit_int("A_today", static_cast<long long>(today.serialNumber()));
    emit_int("A_reference_date", static_cast<long long>(m.curve->referenceDate().serialNumber()));
    emit_int("A_start", static_cast<long long>(start.serialNumber()));
    emit_int("A_end", static_cast<long long>(end.serialNumber()));
    emit_int("A_fixing_date", static_cast<long long>(probe.fixingDate().serialNumber()));
    emit_int("A_ssi_fixing_days", static_cast<long long>(m.ssi->fixingDays()));
    emit("A_fix1_10y", m.s10->fixing(probe.fixingDate()));
    emit("A_fix2_2y", m.s2->fixing(probe.fixingDate()));
    emit("A_ssi_fixing", m.ssi->fixing(probe.fixingDate()));
    emit("A_gearing1", m.ssi->gearing1());
    emit("A_gearing2", m.ssi->gearing2());
    emit("A_accrual_period", probe.accrualPeriod());
}

// --------------------------------------------------------------------------
// Block B — plain CmsSpreadCoupon (the base the four other classes wrap).
// --------------------------------------------------------------------------

void blockB() {
    Market m = makeMarket();
    Date start = m.curve->referenceDate() + 20 * Years;
    Date end = start + 1 * Years;

    CmsSpreadCoupon c(end, 1.0e6, start, end, m.ssi->fixingDays(), m.ssi, 1.0, 0.001, start, end,
                      m.ibor->dayCounter());
    c.setPricer(m.pricer);

    emit("B_rate", c.rate());
    emit("B_amount", c.amount());
    emit("B_adjusted_fixing", c.adjustedFixing());
    emit("B_convexity_adjustment", c.convexityAdjustment());
    emit("B_index_fixing", c.indexFixing());
    emit("B_accrual_period", c.accrualPeriod());
    emit_int("B_payment_date", static_cast<long long>(c.date().serialNumber()));
    emit_int("B_accrual_start", static_cast<long long>(c.accrualStartDate().serialNumber()));
    emit_int("B_accrual_end", static_cast<long long>(c.accrualEndDate().serialNumber()));
    emit_int("B_fixing_date", static_cast<long long>(c.fixingDate().serialNumber()));

    // gearing / spread pass-through
    CmsSpreadCoupon g(end, 1.0e6, start, end, m.ssi->fixingDays(), m.ssi, 2.0, -0.002, start, end,
                      m.ibor->dayCounter());
    g.setPricer(m.pricer);
    emit("B_geared_rate", g.rate());
    emit("B_geared_amount", g.amount());
    emit("B_geared_adjusted_fixing", g.adjustedFixing());
    emit("B_geared_convexity_adjustment", g.convexityAdjustment());
}

// --------------------------------------------------------------------------
// Block C — CappedFlooredCmsSpreadCoupon.
//
// Four configurations. C4 has gearing < 0, which is the interesting one:
// CappedFlooredCoupon's constructor SWAPS the roles (the constructor's `cap`
// argument becomes the stored floor_ and vice versa), so cap()/floor() and
// effectiveCap()/effectiveFloor() report different numbers from the ones
// passed in. Reproduced verbatim.
// --------------------------------------------------------------------------

void emitCapFloored(const std::string& tag, CappedFlooredCmsSpreadCoupon& c) {
    emit("C_" + tag + "_rate", c.rate());
    emit("C_" + tag + "_amount", c.amount());
    emit("C_" + tag + "_convexity_adjustment", c.convexityAdjustment());
    emit_bool("C_" + tag + "_is_capped", c.isCapped());
    emit_bool("C_" + tag + "_is_floored", c.isFloored());
    emit_opt_rate("C_" + tag + "_cap", c.cap());
    emit_opt_rate("C_" + tag + "_floor", c.floor());
    emit_opt_rate("C_" + tag + "_effective_cap", c.effectiveCap());
    emit_opt_rate("C_" + tag + "_effective_floor", c.effectiveFloor());
    emit_int("C_" + tag + "_payment_date", static_cast<long long>(c.date().serialNumber()));
    emit_int("C_" + tag + "_accrual_start",
             static_cast<long long>(c.accrualStartDate().serialNumber()));
    emit_int("C_" + tag + "_accrual_end",
             static_cast<long long>(c.accrualEndDate().serialNumber()));
    emit("C_" + tag + "_accrual_period", c.accrualPeriod());
}

void blockC() {
    Market m = makeMarket();
    Date start = m.curve->referenceDate() + 20 * Years;
    Date end = start + 1 * Years;
    const Natural fd = m.ssi->fixingDays();
    const DayCounter& dc = m.ibor->dayCounter();

    // The uncapped spread rate sits near 0.0059, so cap 0.008 / floor 0.004
    // straddle it and both optionalities carry value.
    {
        CappedFlooredCmsSpreadCoupon c(end, 1.0e6, start, end, fd, m.ssi, 1.0, 0.0, 0.008,
                                       Null<Rate>(), start, end, dc);
        c.setPricer(m.pricer);
        emitCapFloored("cap_only", c);
    }
    {
        CappedFlooredCmsSpreadCoupon c(end, 1.0e6, start, end, fd, m.ssi, 1.0, 0.0, Null<Rate>(),
                                       0.004, start, end, dc);
        c.setPricer(m.pricer);
        emitCapFloored("floor_only", c);
    }
    {
        CappedFlooredCmsSpreadCoupon c(end, 1.0e6, start, end, fd, m.ssi, 1.0, 0.0, 0.008, 0.004,
                                       start, end, dc);
        c.setPricer(m.pricer);
        emitCapFloored("collar", c);
    }
    {
        // gearing < 0 -> cap/floor roles swap inside CappedFlooredCoupon.
        CappedFlooredCmsSpreadCoupon c(end, 1.0e6, start, end, fd, m.ssi, -1.0, 0.02, 0.020, 0.010,
                                       start, end, dc);
        c.setPricer(m.pricer);
        emitCapFloored("neg_gearing_collar", c);
    }
    {
        // in-arrears + spread, to pin the fixing-date shift through the wrapper
        CappedFlooredCmsSpreadCoupon c(end, 1.0e6, start, end, fd, m.ssi, 1.0, 0.0015, 0.010,
                                       Null<Rate>(), start, end, dc, true);
        c.setPricer(m.pricer);
        emitCapFloored("in_arrears_cap", c);
        emit_int("C_in_arrears_cap_fixing_date",
                 static_cast<long long>(c.fixingDate().serialNumber()));
        emit_bool("C_in_arrears_cap_is_in_arrears", c.isInArrears());
    }
}

// --------------------------------------------------------------------------
// Block D — StrippedCappedFlooredCoupon over a CappedFlooredCmsSpreadCoupon.
//
// The class itself is already ported; what is new here is that it composes
// with the CMS-spread family, and that its rate is the *embedded option
// only* (collar => floorlet - caplet, single side => the long option).
// --------------------------------------------------------------------------

void emitStripped(const std::string& tag, StrippedCappedFlooredCoupon& s) {
    emit("D_" + tag + "_rate", s.rate());
    emit("D_" + tag + "_amount", s.amount());
    emit("D_" + tag + "_convexity_adjustment", s.convexityAdjustment());
    emit_bool("D_" + tag + "_is_cap", s.isCap());
    emit_bool("D_" + tag + "_is_floor", s.isFloor());
    emit_bool("D_" + tag + "_is_collar", s.isCollar());
    emit_opt_rate("D_" + tag + "_cap", s.cap());
    emit_opt_rate("D_" + tag + "_floor", s.floor());
    emit_opt_rate("D_" + tag + "_effective_cap", s.effectiveCap());
    emit_opt_rate("D_" + tag + "_effective_floor", s.effectiveFloor());
    emit_int("D_" + tag + "_payment_date", static_cast<long long>(s.date().serialNumber()));
    emit("D_" + tag + "_accrual_period", s.accrualPeriod());
}

void blockD() {
    Market m = makeMarket();
    Date start = m.curve->referenceDate() + 20 * Years;
    Date end = start + 1 * Years;
    const Natural fd = m.ssi->fixingDays();
    const DayCounter& dc = m.ibor->dayCounter();

    struct Case {
        const char* tag;
        Rate cap, floor;
    };
    const Case cases[] = {
        {"cap_only", 0.008, Null<Rate>()},
        {"floor_only", Null<Rate>(), 0.004},
        {"collar", 0.008, 0.004},
    };
    for (const auto& cse : cases) {
        auto u = ext::make_shared<CappedFlooredCmsSpreadCoupon>(
            end, 1.0e6, start, end, fd, m.ssi, 1.0, 0.0, cse.cap, cse.floor, start, end, dc);
        StrippedCappedFlooredCoupon s(u);
        s.setPricer(m.pricer);
        emitStripped(cse.tag, s);
    }
}

// --------------------------------------------------------------------------
// Block E — DigitalCmsSpreadCoupon.
//
// Cash-or-nothing and asset-or-nothing, long and short, Central / Sub / Super
// replication, and the nakedOption variant. callOptionRate()/putOptionRate()
// are the replication itself and are pinned separately from rate(), so a port
// that gets the composition right but the replication wrong is caught.
// --------------------------------------------------------------------------

void emitDigital(const std::string& tag, DigitalCmsSpreadCoupon& d) {
    emit("E_" + tag + "_rate", d.rate());
    emit("E_" + tag + "_amount", d.amount());
    emit("E_" + tag + "_convexity_adjustment", d.convexityAdjustment());
    emit_bool("E_" + tag + "_has_call", d.hasCall());
    emit_bool("E_" + tag + "_has_put", d.hasPut());
    emit_bool("E_" + tag + "_has_collar", d.hasCollar());
    emit_bool("E_" + tag + "_is_long_call", d.isLongCall());
    emit_bool("E_" + tag + "_is_long_put", d.isLongPut());
    emit_opt_rate("E_" + tag + "_call_strike", d.callStrike());
    emit_opt_rate("E_" + tag + "_put_strike", d.putStrike());
    emit_opt_rate("E_" + tag + "_call_digital_payoff", d.callDigitalPayoff());
    emit_opt_rate("E_" + tag + "_put_digital_payoff", d.putDigitalPayoff());
    emit("E_" + tag + "_call_option_rate", d.callOptionRate());
    emit("E_" + tag + "_put_option_rate", d.putOptionRate());
    emit_int("E_" + tag + "_payment_date", static_cast<long long>(d.date().serialNumber()));
    emit("E_" + tag + "_accrual_period", d.accrualPeriod());
}

void blockE() {
    Market m = makeMarket();
    Date start = m.curve->referenceDate() + 20 * Years;
    Date end = start + 1 * Years;
    const Natural fd = m.ssi->fixingDays();
    const DayCounter& dc = m.ibor->dayCounter();

    auto underlying = [&]() {
        return ext::make_shared<CmsSpreadCoupon>(end, 1.0e6, start, end, fd, m.ssi, 1.0, 0.0, start,
                                                 end, dc);
    };

    auto central = ext::make_shared<DigitalReplication>(Replication::Central, 1e-4);
    auto sub = ext::make_shared<DigitalReplication>(Replication::Sub, 1e-4);
    auto super_ = ext::make_shared<DigitalReplication>(Replication::Super, 1e-4);

    // E1 cash-or-nothing long call
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, 0.005, Position::Long, false, 0.03, Null<Rate>(),
                                 Position::Long, false, Null<Rate>(), central);
        d.setPricer(m.pricer);
        emitDigital("con_long_call", d);
    }
    // E2 cash-or-nothing long put
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, Null<Rate>(), Position::Long, false, Null<Rate>(), 0.007,
                                 Position::Long, false, 0.02, central);
        d.setPricer(m.pricer);
        emitDigital("con_long_put", d);
    }
    // E3 collar: short call + long put, cash-or-nothing
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, 0.005, Position::Short, false, 0.03, 0.007, Position::Long,
                                 false, 0.02, central);
        d.setPricer(m.pricer);
        emitDigital("con_collar", d);
    }
    // E4 asset-or-nothing long call (no digital payoff)
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, 0.005, Position::Long, false, Null<Rate>(), Null<Rate>(),
                                 Position::Long, false, Null<Rate>(), central);
        d.setPricer(m.pricer);
        emitDigital("aon_long_call", d);
    }
    // E5 asset-or-nothing long put
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, Null<Rate>(), Position::Long, false, Null<Rate>(), 0.007,
                                 Position::Long, false, Null<Rate>(), central);
        d.setPricer(m.pricer);
        emitDigital("aon_long_put", d);
    }
    // E6 / E7 Sub and Super replication of the same long call — the gap
    // placement (left/right eps) is the whole difference.
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, 0.005, Position::Long, false, 0.03, Null<Rate>(),
                                 Position::Long, false, Null<Rate>(), sub);
        d.setPricer(m.pricer);
        emitDigital("con_long_call_sub", d);
    }
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, 0.005, Position::Long, false, 0.03, Null<Rate>(),
                                 Position::Long, false, Null<Rate>(), super_);
        d.setPricer(m.pricer);
        emitDigital("con_long_call_super", d);
    }
    // E8 naked option: the underlying swaplet rate is dropped from rate().
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, 0.005, Position::Long, false, 0.03, Null<Rate>(),
                                 Position::Long, false, Null<Rate>(), central, true);
        d.setPricer(m.pricer);
        emitDigital("con_long_call_naked", d);
    }
    // E9 default replication argument (`{}` -> DigitalCoupon substitutes
    // DigitalReplication(), i.e. Central with gap 1e-4). Pinned so a port
    // cannot quietly pick a different default.
    {
        auto u = underlying();
        DigitalCmsSpreadCoupon d(u, 0.005, Position::Long, false, 0.03);
        d.setPricer(m.pricer);
        emitDigital("con_long_call_default_replication", d);
    }
}

// --------------------------------------------------------------------------
// Block F — CmsSpreadLeg.
// --------------------------------------------------------------------------

Schedule regularSchedule(const Market& m) {
    Date start = m.curve->referenceDate() + 2 * Years;
    Date end = start + 5 * Years;
    return Schedule(start, end, Period(1, Years), cal, ModifiedFollowing, ModifiedFollowing,
                    DateGeneration::Forward, false);
}

// A schedule whose FIRST period is a short stub, so schedule.isRegular(1) is
// false and FloatingLeg rolls refStart back by one tenor.
Schedule stubSchedule(const Market& m) {
    Date start = m.curve->referenceDate() + 2 * Years;
    Date firstRegular = start + 5 * Months;  // short first period
    Date end = start + 3 * Years + 5 * Months;
    return Schedule(start, end, Period(1, Years), cal, ModifiedFollowing, ModifiedFollowing,
                    DateGeneration::Forward, false, firstRegular);
}

void blockF() {
    Market m = makeMarket();
    Schedule sched = regularSchedule(m);

    emit_int("F_schedule_size", static_cast<long long>(sched.size()));
    {
        std::vector<long long> d;
        for (Size i = 0; i < sched.size(); ++i)
            d.push_back(static_cast<long long>(sched.date(i).serialNumber()));
        emit_iarr("F_schedule_dates", d);
    }

    // F1 plain
    {
        Leg leg = CmsSpreadLeg(sched, m.ssi).withNotionals(1.0e6).withPaymentDayCounter(dc360);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F1_plain", leg);
    }
    // F2 capped
    {
        Leg leg = CmsSpreadLeg(sched, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .withCaps(0.008);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F2_capped", leg);
    }
    // F3 collared, with per-period notionals / gearings / spreads and a mixed
    // caps vector: entries equal to Null<Rate>() switch that period back to a
    // plain CmsSpreadCoupon, so the type dispatch alternates.
    {
        std::vector<Real> notionals = {1.0e6, 2.0e6, 3.0e6};  // shorter than n -> back() repeats
        std::vector<Real> gearings = {1.0, 1.5, 0.5, 2.0, 1.0};
        std::vector<Spread> spreads = {0.0, 0.001, -0.001, 0.0005, 0.0};
        std::vector<Rate> caps = {0.008, Null<Rate>(), 0.010, Null<Rate>(), 0.012};
        std::vector<Rate> floors = {0.002, Null<Rate>(), Null<Rate>(), 0.001, 0.003};
        Leg leg = CmsSpreadLeg(sched, m.ssi)
                      .withNotionals(notionals)
                      .withPaymentDayCounter(dc360)
                      .withFixingDays(2)
                      .withGearings(gearings)
                      .withSpreads(spreads)
                      .withCaps(caps)
                      .withFloors(floors);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F3_mixed", leg);
    }
    // F4 gearing 0 on some periods -> FixedRateCoupon via
    // detail::effectiveFixedRate(spreads, caps, floors, i), which clamps the
    // spread by floor then cap.
    {
        std::vector<Real> gearings = {0.0, 1.0, 0.0, 1.0, 0.0};
        std::vector<Spread> spreads = {0.03, 0.001, 0.03, 0.001, 0.03};
        std::vector<Rate> caps = {0.02, Null<Rate>(), Null<Rate>(), Null<Rate>(), Null<Rate>()};
        std::vector<Rate> floors = {Null<Rate>(), Null<Rate>(), 0.05, Null<Rate>(), Null<Rate>()};
        Leg leg = CmsSpreadLeg(sched, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .withGearings(gearings)
                      .withSpreads(spreads)
                      .withCaps(caps)
                      .withFloors(floors);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F4_fixed_mix", leg);
    }
    // F5 zero payments: every coupon pays on the schedule's terminal date.
    {
        Leg leg = CmsSpreadLeg(sched, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .withZeroPayments(true);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F5_zero", leg);
    }
    // F6 paymentAdjustment = Following instead of the schedule's convention,
    // plus in-arrears fixing.
    {
        Leg leg = CmsSpreadLeg(sched, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .withPaymentAdjustment(Preceding)
                      .inArrears(true);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F6_preceding_arrears", leg);
    }
    // F7 no payment day counter at all -> FloatingRateCoupon falls back to the
    // index's day counter (Act/360 here, via SwapSpreadIndex -> swapIndex1).
    {
        Leg leg = CmsSpreadLeg(sched, m.ssi).withNotionals(1.0e6);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F7_default_daycounter", leg);
        emit_str("F7_daycounter_name",
                 ext::dynamic_pointer_cast<Coupon>(leg[0])->dayCounter().name());
    }
    // F8 irregular first stub -> refStart rolls back one tenor.
    {
        Schedule stub = stubSchedule(m);
        std::vector<long long> d;
        for (Size i = 0; i < stub.size(); ++i)
            d.push_back(static_cast<long long>(stub.date(i).serialNumber()));
        emit_iarr("F8_schedule_dates", d);
        emit_bool("F8_schedule_has_is_regular", stub.hasIsRegular());
        {
            std::vector<long long> reg;
            for (Size i = 1; i <= stub.size() - 1; ++i)
                reg.push_back(stub.isRegular(i) ? 1 : 0);
            emit_iarr("F8_schedule_is_regular", reg);
        }
        Leg leg = CmsSpreadLeg(stub, m.ssi).withNotionals(1.0e6).withPaymentDayCounter(dc360);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("F8_stub", leg);
    }
}

// --------------------------------------------------------------------------
// Block G — DigitalCmsSpreadLeg.
//
// CAUTION, and the reason the option-rate emitters below are handed a FRESH
// leg: FloatingRateCoupon::rate() is LazyObject-cached in C++
// (floatingratecoupon.cpp:88-96 — rate() calls calculate(), and
// performCalculations() is what calls pricer_->initialize(*this)). Every
// coupon in a leg shares ONE LognormalCmsSpreadPricer, and the pricer's
// per-coupon state lives in that shared object. So once coupon i has been
// calculated, a later `leg[i]->callOptionRate()` finds underlying_->rate()
// already cached, skips the re-initialize, and prices the caplet spread
// against whichever coupon happened to initialize the pricer LAST. Asking a
// whole leg for its option rates after asking it for its amounts therefore
// returns the same number five times over, which is an artefact of the cache,
// not of the class under test.
//
// PQuantLib deliberately does not port the LazyObject cache (documented in
// floating_rate_coupon.py: "rate() recomputes on every call"), so it cannot
// reproduce that artefact and should not be asked to. Querying option rates
// on a leg nothing has priced yet makes both implementations take the same
// path — each coupon re-initializes the shared pricer against its own
// underlying — and the reference value is then the meaningful one.
// --------------------------------------------------------------------------

void emitLegOptionRates(const std::string& tag, const Leg& freshLeg) {
    std::vector<Real> co, po;
    for (const auto& cf : freshLeg) {
        auto d = ext::dynamic_pointer_cast<DigitalCoupon>(cf);
        if (d == nullptr) continue;
        co.push_back(d->callOptionRate());
        po.push_back(d->putOptionRate());
    }
    emit_arr(tag + "_call_option_rates", co);
    emit_arr(tag + "_put_option_rates", po);
}

void blockG() {
    Market m = makeMarket();
    Schedule sched = regularSchedule(m);
    auto central = ext::make_shared<DigitalReplication>(Replication::Central, 1e-4);

    // G1 cash-or-nothing long call on every period
    auto buildG1 = [&]() {
        Leg leg = DigitalCmsSpreadLeg(sched, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .withCallStrikes(0.005)
                      .withLongCallOption(Position::Long)
                      .withCallPayoffs(0.03)
                      .withReplication(central)
                      .withNakedOption(false);
        setPricerOnLeg(leg, m.pricer);
        return leg;
    };
    emitLegOptionRates("G1_call", buildG1());  // fresh leg, nothing priced yet
    emitLeg("G1_call", buildG1());
    // G2 collar with per-period strikes/payoffs and a short call
    auto buildG2 = [&]() {
        std::vector<Rate> callStrikes = {0.004, 0.005, 0.006};
        std::vector<Rate> callPayoffs = {0.03, 0.02};
        std::vector<Rate> putStrikes = {0.007, 0.008, 0.009, 0.010, 0.011};
        std::vector<Rate> putPayoffs = {0.01};
        Leg leg = DigitalCmsSpreadLeg(sched, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .withFixingDays(2)
                      .withSpreads(0.0005)
                      .withCallStrikes(callStrikes)
                      .withLongCallOption(Position::Short)
                      .withCallATM(true)
                      .withCallPayoffs(callPayoffs)
                      .withPutStrikes(putStrikes)
                      .withLongPutOption(Position::Long)
                      .withPutATM(false)
                      .withPutPayoffs(putPayoffs)
                      .withReplication(central)
                      .withNakedOption(false);
        setPricerOnLeg(leg, m.pricer);
        return leg;
    };
    emitLegOptionRates("G2_collar", buildG2());  // fresh leg, nothing priced yet
    {
        Leg leg = buildG2();
        emitLeg("G2_collar", leg);
        std::vector<long long> longCall, longPut, callStrikeNull, putStrikeNull;
        std::vector<Real> callStrikeVals, putStrikeVals, callPayoffVals, putPayoffVals;
        for (const auto& cf : leg) {
            auto d = ext::dynamic_pointer_cast<DigitalCmsSpreadCoupon>(cf);
            longCall.push_back(d->isLongCall() ? 1 : 0);
            longPut.push_back(d->isLongPut() ? 1 : 0);
            callStrikeNull.push_back(d->callStrike() == Null<Rate>() ? 1 : 0);
            putStrikeNull.push_back(d->putStrike() == Null<Rate>() ? 1 : 0);
            callStrikeVals.push_back(d->callStrike());
            putStrikeVals.push_back(d->putStrike());
            callPayoffVals.push_back(d->callDigitalPayoff());
            putPayoffVals.push_back(d->putDigitalPayoff());
        }
        emit_iarr("G2_is_long_call", longCall);
        emit_iarr("G2_is_long_put", longPut);
        emit_iarr("G2_call_strike_is_null", callStrikeNull);
        emit_iarr("G2_put_strike_is_null", putStrikeNull);
        emit_arr("G2_call_strikes", callStrikeVals);
        emit_arr("G2_put_strikes", putStrikeVals);
        emit_arr("G2_call_payoffs", callPayoffVals);
        emit_arr("G2_put_payoffs", putPayoffVals);
    }
    // G3 naked option + in-arrears
    {
        Leg leg = DigitalCmsSpreadLeg(sched, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .inArrears(true)
                      .withCallStrikes(0.005)
                      .withCallPayoffs(0.03)
                      .withReplication(central)
                      .withNakedOption(true);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("G3_naked_arrears", leg);
    }
    // G4 the FloatingDigitalLeg fixed-coupon fallback. Note the C++ default:
    //     FixedRateCoupon(..., detail::get(spreads, i, 1.0), ...)
    // i.e. a gearing-0 period with NO spreads vector pays 100%. That is what
    // v1.43 does; it is pinned, not fixed.
    {
        std::vector<Real> gearings = {0.0, 1.0, 0.0, 1.0, 1.0};
        Leg legNoSpreads = DigitalCmsSpreadLeg(sched, m.ssi)
                               .withNotionals(1.0e6)
                               .withPaymentDayCounter(dc360)
                               .withGearings(gearings)
                               .withCallStrikes(0.005)
                               .withCallPayoffs(0.03)
                               .withReplication(central)
                               .withNakedOption(false);
        setPricerOnLeg(legNoSpreads, m.pricer);
        emitLeg("G4_fixed_default_one", legNoSpreads);

        std::vector<Spread> spreads = {0.02, 0.001, 0.03, 0.001, 0.0};
        Leg legSpreads = DigitalCmsSpreadLeg(sched, m.ssi)
                             .withNotionals(1.0e6)
                             .withPaymentDayCounter(dc360)
                             .withGearings(gearings)
                             .withSpreads(spreads)
                             .withCallStrikes(0.005)
                             .withCallPayoffs(0.03)
                             .withReplication(central)
                             .withNakedOption(false);
        setPricerOnLeg(legSpreads, m.pricer);
        emitLeg("G4_fixed_from_spreads", legSpreads);
    }
    // G5 stub schedule, so the digital leg's refStart/refEnd roll is pinned too.
    {
        Schedule stub = stubSchedule(m);
        Leg leg = DigitalCmsSpreadLeg(stub, m.ssi)
                      .withNotionals(1.0e6)
                      .withPaymentDayCounter(dc360)
                      .withCallStrikes(0.005)
                      .withCallPayoffs(0.03)
                      .withReplication(central)
                      .withNakedOption(false);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("G5_stub", leg);
    }
}

// --------------------------------------------------------------------------
// Block H — StrippedCappedFlooredCouponLeg.
//
// Wraps an existing Leg, replacing every CappedFlooredCoupon with a
// StrippedCappedFlooredCoupon and passing everything else through
// UNCHANGED — including identity: the pass-through entries are the very same
// objects, not copies. H3 pins that.
// --------------------------------------------------------------------------

void blockH() {
    Market m = makeMarket();
    Schedule sched = regularSchedule(m);

    // H1 fully capped leg -> every coupon is stripped
    {
        Leg base = CmsSpreadLeg(sched, m.ssi)
                       .withNotionals(1.0e6)
                       .withPaymentDayCounter(dc360)
                       .withCaps(0.008)
                       .withFloors(0.004);
        Leg leg = StrippedCappedFlooredCouponLeg(base);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("H1_all_stripped", leg);
    }
    // H2 mixed leg -> only the capped/floored periods are stripped
    {
        std::vector<Rate> caps = {0.008, Null<Rate>(), 0.010, Null<Rate>(), 0.012};
        Leg base = CmsSpreadLeg(sched, m.ssi)
                       .withNotionals(1.0e6)
                       .withPaymentDayCounter(dc360)
                       .withCaps(caps);
        Leg leg = StrippedCappedFlooredCouponLeg(base);
        setPricerOnLeg(leg, m.pricer);
        emitLeg("H2_mixed", leg);
        emit_str("H2_base_types", joinTypes(base));
    }
    // H3 pass-through identity: entries that are not CappedFlooredCoupon are
    // the SAME shared_ptr, not a copy.
    {
        Leg base =
            CmsSpreadLeg(sched, m.ssi).withNotionals(1.0e6).withPaymentDayCounter(dc360);
        Leg leg = StrippedCappedFlooredCouponLeg(base);
        std::vector<long long> same;
        for (Size i = 0; i < base.size(); ++i)
            same.push_back(leg[i].get() == base[i].get() ? 1 : 0);
        emit_iarr("H3_passthrough_identity", same);
        emit_str("H3_types", joinTypes(leg));
        emit_int("H3_size", static_cast<long long>(leg.size()));
    }
    // H4 an empty leg stays empty.
    {
        Leg empty;
        Leg leg = StrippedCappedFlooredCouponLeg(empty);
        emit_int("H4_empty_size", static_cast<long long>(leg.size()));
    }
}

// --------------------------------------------------------------------------
// Block I — v1.43 defects observed while porting, recorded as evidence.
// --------------------------------------------------------------------------

void blockI() {
    // DigitalCmsSpreadLeg::nakedOption_ has no default member initialiser and
    // the constructor does not assign it (digitalcmsspreadcoupon.hpp:104,
    // .cpp:52-54), so a DigitalCmsSpreadLeg on which withNakedOption() is
    // never called reads an indeterminate bool. Same defect in DigitalIborLeg
    // (digitaliborcoupon.hpp:106) and DigitalCmsLeg (digitalcmscoupon.hpp:106).
    // Every other member of the class HAS an initialiser, so the omission is
    // clearly accidental and `false` is the intended value; the port uses
    // false, and every leg case above sets it explicitly so no reference
    // value here depends on the UB.
    //
    // This is not theoretical. The G4/G5 cases above originally omitted
    // withNakedOption() and the flag came back TRUE on this build: every
    // digital coupon in those legs dropped its underlying swaplet rate, so
    // e.g. G5_stub_rates was the pure option rate and differed from the
    // equivalent explicitly-false leg by exactly F8_stub_rates. The
    // measurement below reproduces it deliberately.
    {
        Market m = makeMarket();
        Schedule sched = regularSchedule(m);
        auto central = ext::make_shared<DigitalReplication>(Replication::Central, 1e-4);
        auto build = [&](bool setFlag) {
            DigitalCmsSpreadLeg b = DigitalCmsSpreadLeg(sched, m.ssi)
                                        .withNotionals(1.0e6)
                                        .withPaymentDayCounter(dc360)
                                        .withCallStrikes(0.005)
                                        .withCallPayoffs(0.03)
                                        .withReplication(central);
            if (setFlag) b.withNakedOption(false);
            Leg leg = b;
            setPricerOnLeg(leg, m.pricer);
            return leg;
        };
        Leg unset = build(false);
        Leg explicitFalse = build(true);
        Real rUnset = ext::dynamic_pointer_cast<Coupon>(unset[0])->rate();
        Real rFalse = ext::dynamic_pointer_cast<Coupon>(explicitFalse[0])->rate();
        // NOT a reference value: this is an observation of undefined
        // behaviour, and it is not even stable within this file. An earlier
        // revision of this probe, in which G4/G5 omitted withNakedOption(),
        // read the flag as TRUE (their rates came out as the pure option
        // rate, short by exactly the swaplet rate). Adding this very block
        // changed the stack layout and the same construct now reads FALSE.
        // That instability IS the finding. The Python port must not be held
        // to either value.
        emit_bool("I_naked_option_uninitialised_differs_from_false", rUnset != rFalse);
        emit_str("I_naked_option_uninitialised_note",
                 "Indeterminate: observed true in one build of this probe and false in another. "
                 "Not a reference value. The port is held to the explicitly-false leg.");
    }
    emit_str("I_naked_option_defect",
             "v1.43 DigitalCmsSpreadLeg::nakedOption_ is never initialised (hpp:104, cpp:52-54); "
             "reading it without withNakedOption() is UB. Port uses false; probe always sets it.");
    emit_bool("I_naked_option_port_default_is_false", true);

    // CmsSpreadLeg's constructor QL_REQUIREs a non-null index;
    // DigitalCmsSpreadLeg's does not (cmsspreadcoupon.cpp:48-51 vs
    // digitalcmsspreadcoupon.cpp:52-54). Asymmetry is intentional to record.
    emit_bool("I_cms_spread_leg_requires_index", true);
    emit_bool("I_digital_cms_spread_leg_requires_index", false);

    // setCouponPricer()'s PricerSetter visitor has no CmsSpreadCoupon case, so
    // it is a silent no-op on these legs (ql/cashflows/couponpricer.cpp).
    emit_str("I_set_coupon_pricer_note",
             "ql setCouponPricer's PricerSetter has no CmsSpreadCoupon visit; it silently does "
             "nothing for CMS-spread legs. Probe and port both walk the leg by hand.");
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = today;

    std::cout << "{\n";
    blockA();
    blockB();
    blockC();
    blockD();
    blockE();
    blockF();
    blockG();
    blockH();
    blockI();
    std::cout << "\n}\n";
    return 0;
}
