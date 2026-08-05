// migration-harness/cpp/probes/v143_cf_multipleresets/probe.cpp
//
// Reference values for the multiple-resets coupon family added in C++
// QuantLib v1.43 (ql/cashflows/multipleresetscoupon.{hpp,cpp}):
//
//   * MultipleResetsCoupon              — FloatingRateCoupon over a reset schedule
//   * MultipleResetsPricer              — base pricer (every price/rate but
//                                         swapletRate is a QL_FAIL)
//   * AveragingMultipleResetsPricer     — RateAveraging::Simple
//   * CompoundingMultipleResetsPricer   — RateAveraging::Compound
//   * MultipleResetsLeg                 — chained builder
//
// WHAT IS PINNED AND WHY
//
// Everything is built inline from literal tables — an explicit zero curve from
// hard-coded knots, an explicit IborIndex, and explicit date-list Schedules —
// so a Python mismatch localises to the coupon/leg code rather than to a shared
// fixture (schedule generation, index presets, curve bootstrapping).
//
// For each of three legs we pin, per coupon: payment date, nominal, accrual
// start/end, accrual period, the fixing/value date lists, the sub-period
// accrual times dt(), the sub-period index fixings that feed the pricer, the
// coupon rate, the amount, gearing/spread/rateSpread, fixingDate() and the
// ex-coupon date — plus the NPV of the whole leg against the inline curve. An
// NPV can match while two errors cancel, hence the full listing.
//
// Every MultipleResetsLeg setter is exercised with a NON-DEFAULT value in at
// least one leg, because the failure mode this subsystem is prone to is an
// argument that is accepted, stored and then never passed on (see
// FixedVsFloatingSwap / OvernightIndexedSwap and their dropped payment lags):
//
//   leg_compound : withNotionals(vector), withPaymentDayCounter,
//                  withPaymentAdjustment, withPaymentCalendar,
//                  withPaymentLag(3 — NON-ZERO), withFixingDays(vector),
//                  withGearings(vector), withCouponSpreads(vector),
//                  withRateSpreads(vector), withExCouponPeriod
//   leg_simple   : withNotionals(scalar), withFixingDays(scalar),
//                  withGearings(scalar), withCouponSpreads(scalar),
//                  withRateSpreads(scalar), withPaymentAdjustment(Preceding)
//                  with a ZERO lag (Calendar::advance ignores the convention
//                  when n != 0 and the unit is Days, so the adjustment is only
//                  observable at lag 0), withAveragingMethod(Simple),
//                  withExCouponPeriod with an EMPTY calendar (the branch that
//                  falls back to the schedule calendar)
//   leg_eom      : withPaymentAdjustment(Unadjusted) + ex-coupon endOfMonth
//
// Leg-B coupon end dates (19-Jun-2026, 11-Nov-2026) are deliberately
// US-GovernmentBond holidays but TARGET business days, so the payment calendar
// AND the payment adjustment both move the payment date measurably.
//
// ex_coupon_matrix re-builds leg A under seven ex-coupon settings whose results
// differ pairwise, so a port that drops the ex-coupon calendar, convention,
// period or end-of-month flag cannot pass:
//   a_6d_us_preceding      vs a_6d_empty_preceding   -> calendar
//   a_1w_us_preceding      vs a_1w_us_following      -> convention
//   a_1w_target_preceding  vs a_1w_us_preceding      -> calendar (again)
//   eom_1m_true            vs eom_1m_false           -> endOfMonth
//   a_none                                           -> null ex-coupon date
//
// Failing branches are pinned as {"raises": true} rather than by message text
// (QL_FAIL messages carry file/line and are not part of the port contract).
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/multipleresets.json.

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/multipleresetscoupon.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

// --- inline fixture (literal tables only) --------------------------------

const Date kToday(5, January, 2026);

// Zero curve knots: continuously-compounded zero rates, Act/365F, linear.
// Deliberately NOT flat: with a flat forward curve every forward rate over the
// same number of days is identical, so a one-day error in a fixing date that
// preserves the period length would be invisible.
const std::vector<Date> kCurveDates = {Date(5, January, 2026), Date(5, July, 2026),
                                       Date(5, January, 2027), Date(5, January, 2028),
                                       Date(5, January, 2030)};
const std::vector<Rate> kCurveZeros = {0.0200, 0.0245, 0.0280, 0.0310, 0.0335};

// Leg A / ex-coupon-matrix reset schedule: 13 monthly TARGET business days.
const std::vector<Date> kSchedA = {
    Date(20, January, 2026),  Date(20, February, 2026), Date(20, March, 2026),
    Date(20, April, 2026),    Date(20, May, 2026),      Date(22, June, 2026),
    Date(20, July, 2026),     Date(20, August, 2026),   Date(21, September, 2026),
    Date(20, October, 2026),  Date(20, November, 2026), Date(21, December, 2026),
    Date(20, January, 2027)};

// Leg B reset schedule: 5 TARGET business days; coupon ends (index 2 and 4) are
// US-GovernmentBond holidays.
const std::vector<Date> kSchedB = {Date(19, February, 2026), Date(20, April, 2026),
                                   Date(19, June, 2026), Date(20, August, 2026),
                                   Date(11, November, 2026)};

// End-of-month schedule: the single coupon ends on a month end, which is the
// precondition Calendar::advance needs before endOfMonth does anything.
const std::vector<Date> kSchedEom = {Date(30, January, 2026), Date(27, February, 2026),
                                     Date(31, March, 2026)};

ext::shared_ptr<YieldTermStructure> makeCurve() {
    return ext::make_shared<ZeroCurve>(kCurveDates, kCurveZeros, Actual365Fixed());
}

ext::shared_ptr<IborIndex> makeIndex(const Handle<YieldTermStructure>& h) {
    return ext::make_shared<IborIndex>("MRTestIbor", Period(1, Months), 2, EURCurrency(), TARGET(),
                                       ModifiedFollowing, false, Actual360(), h);
}

Schedule makeSchedule(const std::vector<Date>& dates) {
    // Self-check: fixing days of 0 make the value dates double as fixing dates,
    // and IborIndex::fixing rejects a non-business fixing date — so a typo in
    // the literal table above must fail loudly here, not silently downstream.
    for (const Date& d : dates)
        QL_REQUIRE(TARGET().isBusinessDay(d), "literal schedule date " << d
                                                                       << " is not a TARGET business day");
    return Schedule(dates, TARGET(), ModifiedFollowing);
}

// --- leg builders --------------------------------------------------------

// Leg A: every vector-form setter, a non-zero payment lag, a payment calendar
// and day counter that differ from the schedule's / index's, Compound averaging.
Leg buildLegA(const Schedule& sched,
              const ext::shared_ptr<IborIndex>& index,
              bool withEx,
              const Period& exPeriod = Period(),
              const Calendar& exCal = Calendar(),
              BusinessDayConvention exConv = Unadjusted,
              bool exEom = false) {
    MultipleResetsLeg builder(sched, index, 3);
    builder.withNotionals(std::vector<Real>{1000000.0, 2000000.0, 3000000.0, 4000000.0})
        .withPaymentDayCounter(Thirty360(Thirty360::BondBasis))
        .withPaymentAdjustment(Preceding)
        .withPaymentCalendar(UnitedStates(UnitedStates::GovernmentBond))
        .withPaymentLag(3)
        .withFixingDays(std::vector<Natural>{0, 1, 2, 5})
        .withGearings(std::vector<Real>{1.0, 1.5, 0.8, 2.0})
        .withCouponSpreads(std::vector<Spread>{0.0010, 0.0020, -0.0005, 0.0})
        .withRateSpreads(std::vector<Spread>{0.0001, 0.0002, 0.0003, 0.0004})
        .withAveragingMethod(RateAveraging::Compound);
    if (withEx)
        builder.withExCouponPeriod(exPeriod, exCal, exConv, exEom);
    return builder;
}

// Leg B: every scalar-form setter, Simple averaging, zero payment lag so that
// withPaymentAdjustment is observable, empty ex-coupon calendar.
Leg buildLegB(const Schedule& sched, const ext::shared_ptr<IborIndex>& index) {
    MultipleResetsLeg builder(sched, index, 2);
    builder.withNotionals(5000000.0)
        .withPaymentAdjustment(Preceding)
        .withPaymentCalendar(UnitedStates(UnitedStates::GovernmentBond))
        .withFixingDays(1)
        .withGearings(0.75)
        .withCouponSpreads(0.0025)
        .withRateSpreads(0.0015)
        .withExCouponPeriod(Period(1, Weeks), Calendar(), Preceding, false)
        .withAveragingMethod(RateAveraging::Simple);
    return builder;
}

Leg buildLegEom(const Schedule& sched, const ext::shared_ptr<IborIndex>& index, bool exEom) {
    MultipleResetsLeg builder(sched, index, 2);
    builder.withNotionals(1000000.0)
        .withPaymentAdjustment(Unadjusted)
        .withExCouponPeriod(Period(1, Months), TARGET(), Following, exEom);
    return builder;
}

// --- JSON emitters -------------------------------------------------------

void emitDateArray(const char* key, const std::vector<Date>& v, const char* indent, bool comma) {
    std::cout << indent << "\"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i != 0 ? ", " : "") << v[i].serialNumber();
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitRealArray(const char* key, const std::vector<Real>& v, const char* indent, bool comma) {
    std::cout << indent << "\"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i != 0 ? ", " : "") << v[i];
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitCoupon(const MultipleResetsCoupon& c,
                const ext::shared_ptr<IborIndex>& index,
                bool comma) {
    std::vector<Real> fixings;
    for (const Date& d : c.fixingDates())
        fixings.push_back(index->fixing(d));

    std::cout << "      {\n";
    std::cout << "        \"payment_date_serial\": " << c.date().serialNumber() << ",\n";
    std::cout << "        \"nominal\": " << c.nominal() << ",\n";
    std::cout << "        \"accrual_start_serial\": " << c.accrualStartDate().serialNumber() << ",\n";
    std::cout << "        \"accrual_end_serial\": " << c.accrualEndDate().serialNumber() << ",\n";
    std::cout << "        \"ref_period_start_serial\": " << c.referencePeriodStart().serialNumber()
              << ",\n";
    std::cout << "        \"ref_period_end_serial\": " << c.referencePeriodEnd().serialNumber() << ",\n";
    std::cout << "        \"ex_coupon_date_serial\": " << c.exCouponDate().serialNumber() << ",\n";
    std::cout << "        \"day_counter\": \"" << c.dayCounter().name() << "\",\n";
    std::cout << "        \"accrual_period\": " << c.accrualPeriod() << ",\n";
    std::cout << "        \"accrual_days\": " << c.accrualDays() << ",\n";
    std::cout << "        \"fixing_days\": " << c.fixingDays() << ",\n";
    std::cout << "        \"gearing\": " << c.gearing() << ",\n";
    std::cout << "        \"spread\": " << c.spread() << ",\n";
    std::cout << "        \"rate_spread\": " << c.rateSpread() << ",\n";
    std::cout << "        \"fixing_date_serial\": " << c.fixingDate().serialNumber() << ",\n";
    emitDateArray("value_date_serials", c.valueDates(), "        ", true);
    emitDateArray("fixing_date_serials", c.fixingDates(), "        ", true);
    emitRealArray("dt", c.dt(), "        ", true);
    emitRealArray("sub_period_fixings", fixings, "        ", true);
    std::cout << "        \"rate\": " << c.rate() << ",\n";
    std::cout << "        \"amount\": " << c.amount() << "\n";
    std::cout << "      }" << (comma ? "," : "") << "\n";
}

void emitLeg(const char* key,
             const Leg& leg,
             const ext::shared_ptr<IborIndex>& index,
             const ext::shared_ptr<YieldTermStructure>& curve,
             bool comma) {
    std::cout << "  \"" << key << "\": {\n";
    std::cout << "    \"size\": " << leg.size() << ",\n";
    std::cout << "    \"npv\": " << CashFlows::npv(leg, *curve, false, kToday, kToday) << ",\n";
    std::cout << "    \"coupons\": [\n";
    for (Size i = 0; i < leg.size(); ++i) {
        auto c = ext::dynamic_pointer_cast<MultipleResetsCoupon>(leg[i]);
        QL_REQUIRE(c, "leg entry is not a MultipleResetsCoupon");
        emitCoupon(*c, index, i + 1 != leg.size());
    }
    std::cout << "    ]\n";
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitExCouponRow(const char* key, const Leg& leg, bool comma) {
    std::vector<Date> dates;
    for (const auto& cf : leg)
        dates.push_back(ext::dynamic_pointer_cast<Coupon>(cf)->exCouponDate());
    emitDateArray(key, dates, "    ", comma);
}

bool throws(const std::function<void()>& f) {
    try {
        f();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

void emitRaises(const char* key, const std::function<void()>& f, bool comma) {
    std::cout << "    \"" << key << "\": { \"raises\": " << (throws(f) ? "true" : "false") << " }"
              << (comma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = kToday;

    auto curve = makeCurve();
    Handle<YieldTermStructure> curveHandle(curve);
    auto index = makeIndex(curveHandle);

    const Schedule schedA = makeSchedule(kSchedA);
    const Schedule schedB = makeSchedule(kSchedB);
    const Schedule schedEom = makeSchedule(kSchedEom);

    std::cout << "{\n";

    // ---- fixture echo (lets the Python test assert it rebuilt the same setup)
    std::cout << "  \"setup\": {\n";
    std::cout << "    \"evaluation_date_serial\": " << kToday.serialNumber() << ",\n";
    emitDateArray("curve_date_serials", kCurveDates, "    ", true);
    emitRealArray("curve_zeros", kCurveZeros, "    ", true);
    std::cout << "    \"index_name\": \"" << index->name() << "\",\n";
    std::cout << "    \"index_fixing_days\": " << index->fixingDays() << ",\n";
    std::cout << "    \"index_day_counter\": \"" << index->dayCounter().name() << "\",\n";
    emitDateArray("schedule_a_serials", schedA.dates(), "    ", true);
    emitDateArray("schedule_b_serials", schedB.dates(), "    ", true);
    emitDateArray("schedule_eom_serials", schedEom.dates(), "    ", false);
    std::cout << "  },\n";

    // ---- the three probed legs
    const Leg legA = buildLegA(schedA, index, true, Period(6, Days),
                               UnitedStates(UnitedStates::GovernmentBond), Preceding, false);
    const Leg legB = buildLegB(schedB, index);
    const Leg legEom = buildLegEom(schedEom, index, true);

    emitLeg("leg_compound", legA, index, curve, true);
    emitLeg("leg_simple", legB, index, curve, true);
    emitLeg("leg_eom", legEom, index, curve, true);

    // ---- ex-coupon matrix: each row differs from at least one sibling
    std::cout << "  \"ex_coupon_matrix\": {\n";
    emitExCouponRow("a_none", buildLegA(schedA, index, false), true);
    emitExCouponRow("a_6d_us_preceding",
                    buildLegA(schedA, index, true, Period(6, Days),
                              UnitedStates(UnitedStates::GovernmentBond), Preceding, false),
                    true);
    emitExCouponRow("a_6d_empty_preceding",
                    buildLegA(schedA, index, true, Period(6, Days), Calendar(), Preceding, false), true);
    emitExCouponRow("a_1w_target_preceding",
                    buildLegA(schedA, index, true, Period(1, Weeks), TARGET(), Preceding, false), true);
    emitExCouponRow("a_1w_us_preceding",
                    buildLegA(schedA, index, true, Period(1, Weeks),
                              UnitedStates(UnitedStates::GovernmentBond), Preceding, false),
                    true);
    emitExCouponRow("a_1w_us_following",
                    buildLegA(schedA, index, true, Period(1, Weeks),
                              UnitedStates(UnitedStates::GovernmentBond), Following, false),
                    true);
    emitExCouponRow("eom_1m_true", buildLegEom(schedEom, index, true), true);
    emitExCouponRow("eom_1m_false", buildLegEom(schedEom, index, false), false);
    std::cout << "  },\n";

    // ---- MultipleResetsPricer: everything but swapletRate is a QL_FAIL
    const auto firstCoupon = ext::dynamic_pointer_cast<MultipleResetsCoupon>(legA.front());
    std::cout << "  \"pricer_failures\": {\n";
    emitRaises("swaplet_price",
               [&] {
                   CompoundingMultipleResetsPricer p;
                   p.initialize(*firstCoupon);
                   p.swapletPrice();
               },
               true);
    emitRaises("caplet_price",
               [&] {
                   CompoundingMultipleResetsPricer p;
                   p.initialize(*firstCoupon);
                   p.capletPrice(0.03);
               },
               true);
    emitRaises("caplet_rate",
               [&] {
                   CompoundingMultipleResetsPricer p;
                   p.initialize(*firstCoupon);
                   p.capletRate(0.03);
               },
               true);
    emitRaises("floorlet_price",
               [&] {
                   CompoundingMultipleResetsPricer p;
                   p.initialize(*firstCoupon);
                   p.floorletPrice(0.01);
               },
               true);
    emitRaises("floorlet_rate",
               [&] {
                   CompoundingMultipleResetsPricer p;
                   p.initialize(*firstCoupon);
                   p.floorletRate(0.01);
               },
               true);
    emitRaises("initialize_wrong_coupon_type",
               [&] {
                   IborCoupon other(Date(20, April, 2026), 1000000.0, Date(20, January, 2026),
                                    Date(20, April, 2026), 2, index);
                   AveragingMultipleResetsPricer p;
                   p.initialize(other);
               },
               false);
    std::cout << "  },\n";

    // ---- constructor + operator Leg() guards
    std::cout << "  \"raises\": {\n";
    emitRaises("ctor_no_index",
               [&] {
                   MultipleResetsLeg b(schedA, ext::shared_ptr<IborIndex>(), 3);
                   (void)b;
               },
               true);
    emitRaises("ctor_empty_schedule",
               [&] {
                   MultipleResetsLeg b(Schedule(), index, 3);
                   (void)b;
               },
               true);
    emitRaises("ctor_resets_do_not_divide",
               [&] {
                   MultipleResetsLeg b(schedA, index, 5);
                   (void)b;
               },
               true);
    emitRaises("build_no_notional",
               [&] {
                   MultipleResetsLeg b(schedA, index, 3);
                   Leg l = b;
                   (void)l;
               },
               true);
    emitRaises("build_too_many_notionals",
               [&] {
                   MultipleResetsLeg b(schedA, index, 3);
                   b.withNotionals(std::vector<Real>{1.0, 2.0, 3.0, 4.0, 5.0});
                   Leg l = b;
                   (void)l;
               },
               true);
    emitRaises("build_too_many_gearings",
               [&] {
                   MultipleResetsLeg b(schedA, index, 3);
                   b.withNotionals(1000000.0).withGearings(std::vector<Real>{1.0, 1.0, 1.0, 1.0, 1.0});
                   Leg l = b;
                   (void)l;
               },
               true);
    emitRaises("build_too_many_coupon_spreads",
               [&] {
                   MultipleResetsLeg b(schedA, index, 3);
                   b.withNotionals(1000000.0)
                       .withCouponSpreads(std::vector<Spread>{0.0, 0.0, 0.0, 0.0, 0.0});
                   Leg l = b;
                   (void)l;
               },
               true);
    emitRaises("build_too_many_rate_spreads",
               [&] {
                   MultipleResetsLeg b(schedA, index, 3);
                   b.withNotionals(1000000.0)
                       .withRateSpreads(std::vector<Spread>{0.0, 0.0, 0.0, 0.0, 0.0});
                   Leg l = b;
                   (void)l;
               },
               true);
    emitRaises("build_too_many_fixing_days",
               [&] {
                   MultipleResetsLeg b(schedA, index, 3);
                   b.withNotionals(1000000.0).withFixingDays(std::vector<Natural>{2, 2, 2, 2, 2});
                   Leg l = b;
                   (void)l;
               },
               false);
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
