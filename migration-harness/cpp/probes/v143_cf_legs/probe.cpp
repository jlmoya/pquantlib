// migration-harness/cpp/probes/v143_cf_legs/probe.cpp
//
// Reference values for the eight chained leg-builder classes in C++ QuantLib
// v1.43:
//
//   FixedRateLeg, IborLeg, OvernightLeg, CmsLeg, CPILeg,
//   DigitalIborLeg, DigitalCmsLeg, AverageBMALeg
//
// What is pinned and why
// ----------------------
// A leg builder is a pile of optional setters, and the failure mode this probe
// exists to catch is a setter that is *accepted and dropped*: a headline NPV
// stays right while a payment lag, an ex-coupon period or a gearing silently
// does nothing. So every leg is emitted as the **full per-coupon listing** —
// runtime type, payment date, nominal, accrual start/end, accrual period,
// rate, amount, and (where the coupon has them) fixing date, gearing, spread,
// ex-coupon date, cap/floor and digital strikes/payoffs. Two legs are emitted
// per builder at minimum: one built with defaults only, and one with every
// honourable setter moved off its default, so the diff between them is what
// proves each setter reached the coupons.
//
// Non-zero payment lags are used wherever the builder has one (FixedRateLeg,
// IborLeg, OvernightLeg); CmsLeg, the digital legs and AverageBMALeg have none
// (C++ hard-codes lag 0 for CmsLeg and the digital legs adjust the accrual end
// directly), which is itself worth pinning.
//
// Rates that C++ itself cannot produce — a capped/floored coupon on a leg that
// C++ deliberately leaves without a pricer, or a CMS coupon with no Hagan
// pricer attached — are emitted as JSON null rather than skipped, so the Python
// side asserts the *same* absence rather than quietly asserting nothing.
//
// Determinism: evaluation date pinned to 15 January 2024; a flat 3% Act/365F
// forward curve drives every forecast; the CPI leg lives entirely in the past
// against a seeded EUHICP ramp, so no inflation term structure is needed.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/legs.json.

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/cashflows/averagebmacoupon.hpp>
#include <ql/cashflows/capflooredcoupon.hpp>
#include <ql/cashflows/cmscoupon.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/cpicoupon.hpp>
#include <ql/cashflows/digitalcmscoupon.hpp>
#include <ql/cashflows/digitaliborcoupon.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/overnightindexedcoupon.hpp>
#include <ql/cashflows/overnightindexedcouponpricer.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/bmaindex.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/eonia.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

const Date kToday(15, January, 2024);
const Rate kFlatRate = 0.03;
const Real kNominal = 1000000.0;
const Volatility kCapletVol = 0.15;

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

void emitBool(const char* key, bool b, bool comma = true) {
    std::cout << "\"" << key << "\": " << (b ? "true" : "false");
    if (comma)
        std::cout << ", ";
}

// Runtime type name — a leg builder choosing the wrong coupon class for a
// period (capped vs plain, fixed vs floating) is exactly the kind of error a
// matching amount can hide.
std::string typeName(const ext::shared_ptr<CashFlow>& cf) {
    if (ext::dynamic_pointer_cast<DigitalIborCoupon>(cf)) return "DigitalIborCoupon";
    if (ext::dynamic_pointer_cast<DigitalCmsCoupon>(cf)) return "DigitalCmsCoupon";
    if (ext::dynamic_pointer_cast<CappedFlooredOvernightIndexedCoupon>(cf))
        return "CappedFlooredOvernightIndexedCoupon";
    if (ext::dynamic_pointer_cast<CappedFlooredCmsCoupon>(cf)) return "CappedFlooredCmsCoupon";
    if (ext::dynamic_pointer_cast<CappedFlooredIborCoupon>(cf)) return "CappedFlooredIborCoupon";
    if (ext::dynamic_pointer_cast<OvernightIndexedCoupon>(cf)) return "OvernightIndexedCoupon";
    if (ext::dynamic_pointer_cast<AverageBMACoupon>(cf)) return "AverageBMACoupon";
    if (ext::dynamic_pointer_cast<CmsCoupon>(cf)) return "CmsCoupon";
    if (ext::dynamic_pointer_cast<IborCoupon>(cf)) return "IborCoupon";
    if (ext::dynamic_pointer_cast<CPICoupon>(cf)) return "CPICoupon";
    if (ext::dynamic_pointer_cast<CPICashFlow>(cf)) return "CPICashFlow";
    if (ext::dynamic_pointer_cast<FixedRateCoupon>(cf)) return "FixedRateCoupon";
    return "CashFlow";
}

// Anything that needs a pricer we deliberately did not attach throws; emit the
// absence rather than silently omitting the field.
Real safeRate(const ext::shared_ptr<Coupon>& c) {
    try {
        return c->rate();
    } catch (const std::exception&) {
        return Null<Real>();
    }
}

Real safeAmount(const ext::shared_ptr<CashFlow>& c) {
    try {
        return c->amount();
    } catch (const std::exception&) {
        return Null<Real>();
    }
}

Date safeFixingDate(const ext::shared_ptr<FloatingRateCoupon>& c) {
    try {
        return c->fixingDate();
    } catch (const std::exception&) {
        return Date();
    }
}

void emitLeg(const std::string& key, const Leg& leg, bool comma) {
    std::cout << "  \"" << key << "\": [\n";
    for (Size i = 0; i < leg.size(); ++i) {
        const ext::shared_ptr<CashFlow>& cf = leg[i];
        std::cout << "    {";
        std::cout << "\"type\": \"" << typeName(cf) << "\", ";
        emitDate("payment_date", cf->date());
        emitReal("amount", safeAmount(cf), false);

        auto cpn = ext::dynamic_pointer_cast<Coupon>(cf);
        if (cpn != nullptr) {
            std::cout << ", ";
            emitReal("nominal", cpn->nominal());
            emitDate("accrual_start", cpn->accrualStartDate());
            emitDate("accrual_end", cpn->accrualEndDate());
            emitDate("ref_period_start", cpn->referencePeriodStart());
            emitDate("ref_period_end", cpn->referencePeriodEnd());
            emitReal("accrual_period", cpn->accrualPeriod());
            emitDate("ex_coupon_date", cpn->exCouponDate());
            emitReal("rate", safeRate(cpn), false);
        }

        auto frc = ext::dynamic_pointer_cast<FloatingRateCoupon>(cf);
        if (frc != nullptr) {
            std::cout << ", ";
            emitReal("gearing", frc->gearing());
            emitReal("spread", frc->spread());
            emitBool("is_in_arrears", frc->isInArrears());
            emitReal("fixing_days", static_cast<Real>(frc->fixingDays()));
            emitDate("fixing_date", safeFixingDate(frc), false);
        }

        auto cfc = ext::dynamic_pointer_cast<CappedFlooredCoupon>(cf);
        if (cfc != nullptr) {
            std::cout << ", ";
            emitBool("is_capped", cfc->isCapped());
            emitBool("is_floored", cfc->isFloored());
            emitReal("cap", cfc->isCapped() ? cfc->cap() : Null<Real>());
            emitReal("floor", cfc->isFloored() ? cfc->floor() : Null<Real>(), false);
        }

        auto cfon = ext::dynamic_pointer_cast<CappedFlooredOvernightIndexedCoupon>(cf);
        if (cfon != nullptr) {
            std::cout << ", ";
            emitBool("is_capped", cfon->isCapped());
            emitBool("is_floored", cfon->isFloored());
            emitReal("cap", cfon->isCapped() ? cfon->cap() : Null<Real>());
            emitReal("floor", cfon->isFloored() ? cfon->floor() : Null<Real>());
            emitBool("naked_option", cfon->nakedOption(), false);
        }

        auto oic = ext::dynamic_pointer_cast<OvernightIndexedCoupon>(cf);
        if (oic != nullptr) {
            std::cout << ", ";
            emitDate("rate_computation_start", oic->rateComputationStartDate());
            emitDate("rate_computation_end", oic->rateComputationEndDate());
            emitReal("n_fixings", static_cast<Real>(oic->fixingDates().size()));
            std::cout << "\"averaging_method\": " << static_cast<int>(oic->averagingMethod());
        }

        auto dig = ext::dynamic_pointer_cast<DigitalCoupon>(cf);
        if (dig != nullptr) {
            std::cout << ", ";
            emitBool("has_call", dig->hasCall());
            emitBool("has_put", dig->hasPut());
            emitBool("is_long_call", dig->isLongCall());
            emitBool("is_long_put", dig->isLongPut());
            emitReal("call_strike", dig->hasCall() ? dig->callStrike() : Null<Real>());
            emitReal("put_strike", dig->hasPut() ? dig->putStrike() : Null<Real>());
            emitReal("call_digital_payoff", dig->callDigitalPayoff());
            emitReal("put_digital_payoff", dig->putDigitalPayoff(), false);
        }

        auto cpi = ext::dynamic_pointer_cast<CPICoupon>(cf);
        if (cpi != nullptr) {
            std::cout << ", ";
            emitReal("fixed_rate", cpi->fixedRate());
            emitReal("base_cpi", cpi->baseCPI());
            emitDate("base_date", cpi->baseDate());
            emitReal("index_fixing", cpi->indexFixing());
            std::cout << "\"observation_interpolation\": "
                      << static_cast<int>(cpi->observationInterpolation());
        }

        auto cpicf = ext::dynamic_pointer_cast<CPICashFlow>(cf);
        if (cpicf != nullptr) {
            std::cout << ", ";
            emitReal("notional", cpicf->notional());
            emitDate("observation_date", cpicf->observationDate());
            emitBool("growth_only", cpicf->growthOnly());
            emitReal("index_fixing", cpicf->indexFixing(), false);
        }

        std::cout << "}" << (i + 1 < leg.size() ? "," : "") << "\n";
    }
    std::cout << "  ]" << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------------------------
// Shared market data
// ---------------------------------------------------------------------------

Handle<YieldTermStructure> flatCurve() {
    static Handle<YieldTermStructure> h(ext::make_shared<FlatForward>(
        kToday, kFlatRate, Actual365Fixed()));
    return h;
}

// Semiannual 2Y schedule, entirely forward of the evaluation date so every
// fixing is a forecast off the flat curve.
Schedule mainSchedule() {
    return Schedule(Date(15, February, 2024), Date(15, February, 2026),
                    Period(6, Months), TARGET(), ModifiedFollowing,
                    ModifiedFollowing, DateGeneration::Forward, false);
}

// Short-first-stub schedule: exercises the irregular-stub reference-period
// widening inside FloatingLeg / FixedRateLeg::operator Leg().
Schedule stubSchedule() {
    return Schedule(Date(20, March, 2024), Date(15, February, 2026),
                    Period(6, Months), TARGET(), ModifiedFollowing,
                    ModifiedFollowing, DateGeneration::Backward, false);
}

// An *unadjusted* monthly schedule, so several period ends land on a weekend.
// The comparison convention is Unadjusted rather than Preceding because an
// OvernightIndexedCoupon requires paymentDate >= accrualEndDate, which a
// backward roll would violate.
// withPaymentAdjustment is invisible on an adjusted schedule (the payment date
// is already a business day, and Calendar::advance over Days ignores the
// convention entirely once the lag is non-zero), so this is the only shape in
// which a dropped payment-adjustment setter would show up.
Schedule unadjustedSchedule() {
    return Schedule(Date(15, May, 2024), Date(15, November, 2024), Period(1, Months),
                    TARGET(), Unadjusted, Unadjusted, DateGeneration::Forward, false);
}

Schedule unadjustedPastSchedule() {
    return Schedule(Date(15, May, 2021), Date(15, November, 2021), Period(1, Months),
                    TARGET(), Unadjusted, Unadjusted, DateGeneration::Forward, false);
}

ext::shared_ptr<IborIndex> euribor6m() {
    return ext::make_shared<Euribor6M>(flatCurve());
}

ext::shared_ptr<OvernightIndex> eonia() {
    return ext::make_shared<Eonia>(flatCurve());
}

ext::shared_ptr<SwapIndex> cms10y() {
    auto ibor = euribor6m();
    return ext::make_shared<SwapIndex>("EuriborSwapIsdaFixA", Period(10, Years),
                                       ibor->fixingDays(), EURCurrency(),
                                       ibor->fixingCalendar(), Period(1, Years),
                                       Unadjusted, ibor->dayCounter(), ibor);
}

ext::shared_ptr<IborCouponPricer> blackPricer() {
    Handle<OptionletVolatilityStructure> vol(
        ext::make_shared<ConstantOptionletVolatility>(
            kToday, TARGET(), Following, kCapletVol, Actual365Fixed()));
    return ext::make_shared<BlackIborCouponPricer>(vol);
}

// ---------------------------------------------------------------------------
// FixedRateLeg
// ---------------------------------------------------------------------------

void emitFixedRateLegs() {
    Schedule s = mainSchedule();

    emitLeg("fixed_default",
            FixedRateLeg(s).withNotionals(kNominal).withCouponRates(0.035, Actual360()),
            true);

    emitLeg("fixed_full",
            FixedRateLeg(s)
                .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                .withCouponRates(std::vector<Rate>{0.03, 0.04}, Actual360(),
                                 Compounded, Semiannual)
                .withPaymentAdjustment(ModifiedFollowing)
                .withFirstPeriodDayCounter(Thirty360(Thirty360::BondBasis))
                .withLastPeriodDayCounter(Actual365Fixed())
                .withPaymentCalendar(TARGET())
                .withPaymentLag(3)
                .withExCouponPeriod(Period(2, Days), TARGET(), Preceding, false),
            true);

    // withCouponRates(const InterestRate&) — the third C++ overload.
    emitLeg("fixed_interest_rate",
            FixedRateLeg(s).withNotionals(kNominal).withCouponRates(
                InterestRate(0.037, Actual365Fixed(), Compounded, Annual)),
            true);

    // withCouponRates(const std::vector<InterestRate>&) — the fourth overload.
    emitLeg("fixed_interest_rate_vector",
            FixedRateLeg(s).withNotionals(kNominal).withCouponRates(
                std::vector<InterestRate>{
                    InterestRate(0.031, Actual360(), Simple, Annual),
                    InterestRate(0.041, Thirty360(Thirty360::BondBasis), Compounded,
                                 Semiannual)}),
            true);

    // Short first stub: the first coupon's reference period is widened back by
    // one tenor, which the first-period day counter then measures.
    Schedule u = unadjustedSchedule();
    emitLeg("fixed_adj_following",
            FixedRateLeg(u)
                .withNotionals(kNominal)
                .withCouponRates(0.035, Actual360())
                .withPaymentAdjustment(Following),
            true);
    emitLeg("fixed_adj_unadjusted",
            FixedRateLeg(u)
                .withNotionals(kNominal)
                .withCouponRates(0.035, Actual360())
                .withPaymentAdjustment(Unadjusted),
            true);

    emitLeg("fixed_stub",
            FixedRateLeg(stubSchedule())
                .withNotionals(kNominal)
                .withCouponRates(0.035, Actual360())
                .withFirstPeriodDayCounter(Thirty360(Thirty360::BondBasis))
                .withLastPeriodDayCounter(Actual365Fixed()),
            true);
}

// ---------------------------------------------------------------------------
// IborLeg
// ---------------------------------------------------------------------------

void emitIborLegs() {
    Schedule s = mainSchedule();
    auto idx = euribor6m();

    emitLeg("ibor_default", IborLeg(s, idx).withNotionals(kNominal), true);

    emitLeg("ibor_full",
            IborLeg(s, idx)
                .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                .withPaymentDayCounter(Actual360())
                .withPaymentAdjustment(ModifiedFollowing)
                .withPaymentLag(2)
                .withPaymentCalendar(TARGET())
                .withFixingDays(1)
                .withGearings(1.5)
                .withSpreads(0.002)
                .withFixingConvention(Following)
                .withExCouponPeriod(Period(3, Days), TARGET(), Preceding, false),
            true);

    // withZeroPayments: every coupon pays on the leg's final payment date.
    emitLeg("ibor_zero_payments",
            IborLeg(s, idx).withNotionals(kNominal).withPaymentLag(2).withZeroPayments(true),
            true);

    // Caps/floors: C++ attaches no default pricer here, so rate/amount are null
    // and only the structure is pinned.
    emitLeg("ibor_capped",
            IborLeg(s, idx).withNotionals(kNominal).withCaps(0.035).withFloors(0.01),
            true);

    // inArrears: no default pricer either; the fixing date moves to the end.
    emitLeg("ibor_in_arrears", IborLeg(s, idx).withNotionals(kNominal).inArrears(true),
            true);

    // Zero gearing degenerates to a fixed coupon paying the clamped spread.
    emitLeg("ibor_zero_gearing",
            IborLeg(s, idx)
                .withNotionals(kNominal)
                .withPaymentDayCounter(Actual360())
                .withGearings(0.0)
                .withSpreads(0.025)
                .withFloors(0.03),
            true);

    // withIndexedCoupons / withAtParCoupons change the default pricer's
    // forecast span. On a schedule whose periods match the index tenor the two
    // forecasts coincide, so the flag is only observable on a *quarterly*
    // schedule against the 6M index — which is exactly the configuration that
    // would let a dropped flag hide on a semiannual leg.
    Schedule q(Date(15, February, 2024), Date(15, February, 2025), Period(3, Months),
               TARGET(), ModifiedFollowing, ModifiedFollowing, DateGeneration::Forward,
               false);
    emitLeg("ibor_quarterly_default", IborLeg(q, idx).withNotionals(kNominal), true);
    emitLeg("ibor_indexed_coupons",
            IborLeg(q, idx).withNotionals(kNominal).withIndexedCoupons(true), true);
    emitLeg("ibor_at_par_coupons",
            IborLeg(q, idx).withNotionals(kNominal).withAtParCoupons(true), true);

    Schedule u = unadjustedSchedule();
    emitLeg("ibor_adj_following",
            IborLeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Following), true);
    emitLeg("ibor_adj_unadjusted",
            IborLeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Unadjusted), true);

    emitLeg("ibor_stub", IborLeg(stubSchedule(), idx).withNotionals(kNominal), true);
}

// ---------------------------------------------------------------------------
// OvernightLeg
// ---------------------------------------------------------------------------

void emitOvernightLegs() {
    Schedule s = mainSchedule();
    auto idx = eonia();

    emitLeg("on_default", OvernightLeg(s, idx).withNotionals(kNominal), true);

    emitLeg("on_full",
            OvernightLeg(s, idx)
                .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                .withPaymentDayCounter(Actual360())
                .withPaymentAdjustment(ModifiedFollowing)
                .withPaymentCalendar(TARGET())
                .withPaymentLag(2)
                .withGearings(1.5)
                .withSpreads(0.001),
            true);

    // Simple (arithmetic) averaging picks a different default coupon pricer.
    emitLeg("on_simple_averaging",
            OvernightLeg(s, idx)
                .withNotionals(kNominal)
                .withAveragingMethod(RateAveraging::Simple),
            true);

    // In-advance fixing: the observation window moves to the previous period.
    emitLeg("on_in_advance", OvernightLeg(s, idx).withNotionals(kNominal).inArrears(false),
            true);

    // lastRecentPeriod shortens the observation window from the period end.
    emitLeg("on_last_recent",
            OvernightLeg(s, idx)
                .withNotionals(kNominal)
                .withLastRecentPeriod(Period(1, Months))
                .withLastRecentPeriodCalendar(TARGET()),
            true);

    // Explicit payment dates override the computed roll entirely.
    std::vector<Date> payDates;
    for (Size i = 1; i < s.size(); ++i)
        payDates.push_back(TARGET().advance(s.date(i), 10, Days, Following));
    emitLeg("on_payment_dates",
            OvernightLeg(s, idx).withNotionals(kNominal).withPaymentDates(payDates), true);

    // Caps/floors: the compounding pricer has no capletRate, so rate is null;
    // the wrapper type + cap/floor/naked flags are what get pinned.
    emitLeg("on_capped",
            OvernightLeg(s, idx)
                .withNotionals(kNominal)
                .withCaps(0.032)
                .withFloors(0.01)
                .withNakedOption(true),
            true);

    // An explicitly supplied pricer must match the averaging method. With the
    // pricer's default (zero-volatility) parameters this reproduces the
    // averaging-method default exactly, which is the point: the setter is
    // proved by the pricer instance the coupons end up holding.
    emitLeg("on_explicit_pricer",
            OvernightLeg(s, idx)
                .withNotionals(kNominal)
                .withAveragingMethod(RateAveraging::Simple)
                .withCouponPricer(
                    ext::make_shared<ArithmeticAveragedOvernightIndexedCouponPricer>()),
            true);

    // Same setter with a convexity-adjusting pricer (mean reversion 0.05,
    // volatility 0.20, Takada approximation). PQuantLib's arithmetic pricer
    // documents the Hull-White convexity correction as a deferred carve-out,
    // so the Python test for this leg is skipped — but the C++ number is
    // pinned here so the gap is measurable the moment it is closed.
    emitLeg("on_explicit_pricer_convexity",
            OvernightLeg(s, idx)
                .withNotionals(kNominal)
                .withAveragingMethod(RateAveraging::Simple)
                .withCouponPricer(
                    ext::make_shared<ArithmeticAveragedOvernightIndexedCouponPricer>(
                        0.05, 0.20, true)),
            true);

    emitLeg("on_zero_gearing",
            OvernightLeg(s, idx)
                .withNotionals(kNominal)
                .withPaymentDayCounter(Actual360())
                .withGearings(0.0)
                .withSpreads(0.022)
                .withCaps(0.02),
            true);

    Schedule u = unadjustedSchedule();
    emitLeg("on_adj_following",
            OvernightLeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Following),
            true);
    emitLeg("on_adj_unadjusted",
            OvernightLeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Unadjusted),
            true);

    emitLeg("on_stub", OvernightLeg(stubSchedule(), idx).withNotionals(kNominal), true);
}

// ---------------------------------------------------------------------------
// CmsLeg
// ---------------------------------------------------------------------------

void emitCmsLegs() {
    Schedule s = mainSchedule();
    auto idx = cms10y();

    emitLeg("cms_default", CmsLeg(s, idx).withNotionals(kNominal), true);

    emitLeg("cms_full",
            CmsLeg(s, idx)
                .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                .withPaymentDayCounter(Thirty360(Thirty360::BondBasis))
                .withPaymentAdjustment(ModifiedFollowing)
                .withFixingDays(1)
                .withGearings(1.5)
                .withSpreads(0.002)
                .withFixingConvention(Following)
                .withExCouponPeriod(Period(4, Days), TARGET(), Preceding, false),
            true);

    emitLeg("cms_in_arrears", CmsLeg(s, idx).withNotionals(kNominal).inArrears(true), true);

    emitLeg("cms_zero_payments",
            CmsLeg(s, idx).withNotionals(kNominal).withZeroPayments(true), true);

    emitLeg("cms_capped",
            CmsLeg(s, idx).withNotionals(kNominal).withCaps(0.05).withFloors(0.01), true);

    Schedule u = unadjustedSchedule();
    emitLeg("cms_adj_following",
            CmsLeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Following), true);
    emitLeg("cms_adj_unadjusted",
            CmsLeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Unadjusted), true);

    emitLeg("cms_zero_gearing",
            CmsLeg(s, idx)
                .withNotionals(kNominal)
                .withPaymentDayCounter(Actual360())
                .withGearings(0.0)
                .withSpreads(0.028)
                .withCaps(0.02),
            true);
}

// ---------------------------------------------------------------------------
// CPILeg
// ---------------------------------------------------------------------------

Real rampFixing(Integer year, Integer month) {
    return 100.0 + 0.5 * (12 * (year - 2018) + (month - 1));
}

void seedEuHicp(const ext::shared_ptr<ZeroInflationIndex>& idx) {
    for (Integer y = 2018; y <= 2023; ++y)
        for (Integer m = 1; m <= 12; ++m)
            idx->addFixing(Date(1, Month(m), y), rampFixing(y, m), true);
}

void emitCpiLegs() {
    auto idx = ext::make_shared<EUHICP>();
    idx->clearFixings();
    seedEuHicp(idx);

    // Entirely in the past relative to the 15-Jan-2024 evaluation date, so
    // every observation resolves out of the seeded history.
    Schedule s(Date(15, January, 2021), Date(15, January, 2023), Period(1, Years),
               TARGET(), ModifiedFollowing, ModifiedFollowing, DateGeneration::Forward,
               false);
    Period lag(3, Months);
    const Real baseCpi = rampFixing(2020, 10);

    emitLeg("cpi_default",
            CPILeg(s, idx, baseCpi, lag).withNotionals(kNominal).withFixedRates(0.02),
            true);

    emitLeg("cpi_full",
            CPILeg(s, idx, baseCpi, lag)
                .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                .withFixedRates(std::vector<Real>{0.02, 0.025})
                .withPaymentDayCounter(Actual360())
                .withPaymentAdjustment(Following)
                .withPaymentCalendar(TARGET())
                .withObservationInterpolation(CPI::Linear)
                .withSubtractInflationNominal(false)
                .withExCouponPeriod(Period(5, Days), TARGET(), Preceding, false)
                .withBaseDate(Date(15, October, 2020)),
            true);

    // A zero fixed rate degenerates the period to a FixedRateCoupon paying the
    // floor/cap-clamped zero — the only place CPILeg's caps and floors bite.
    emitLeg("cpi_zero_rate_floored",
            CPILeg(s, idx, baseCpi, lag)
                .withNotionals(kNominal)
                .withFixedRates(0.0)
                .withPaymentDayCounter(Actual360())
                .withFloors(0.015),
            true);

    emitLeg("cpi_zero_rate_capped",
            CPILeg(s, idx, baseCpi, lag)
                .withNotionals(kNominal)
                .withFixedRates(0.0)
                .withPaymentDayCounter(Actual360())
                .withFloors(0.015)
                .withCaps(0.005),
            true);

    // No base CPI, base date supplied instead.
    emitLeg("cpi_base_date_only",
            CPILeg(s, idx, Null<Real>(), lag)
                .withNotionals(kNominal)
                .withFixedRates(0.02)
                .withBaseDate(Date(15, October, 2020)),
            true);

    Schedule u = unadjustedPastSchedule();
    emitLeg("cpi_adj_following",
            CPILeg(u, idx, baseCpi, lag)
                .withNotionals(kNominal)
                .withFixedRates(0.02)
                .withPaymentAdjustment(Following),
            true);
    emitLeg("cpi_adj_unadjusted",
            CPILeg(u, idx, baseCpi, lag)
                .withNotionals(kNominal)
                .withFixedRates(0.02)
                .withPaymentAdjustment(Unadjusted),
            true);

    idx->clearFixings();
}

// ---------------------------------------------------------------------------
// DigitalIborLeg / DigitalCmsLeg
// ---------------------------------------------------------------------------

Leg pricedDigitalIborLeg(DigitalIborLeg builder) {
    Leg leg = builder;
    // The digital rate is a call/put spread on the underlying coupon, so the
    // underlying needs a pricer; C++'s DigitalIborLeg attaches none.
    setCouponPricer(leg, blackPricer());
    return leg;
}

void emitDigitalIborLegs() {
    Schedule s = mainSchedule();
    auto idx = euribor6m();

    emitLeg("digital_ibor_default",
            pricedDigitalIborLeg(
                DigitalIborLeg(s, idx).withNotionals(kNominal).withCallStrikes(0.03)),
            true);

    emitLeg("digital_ibor_full",
            pricedDigitalIborLeg(
                DigitalIborLeg(s, idx)
                    .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                    .withPaymentDayCounter(Actual360())
                    .withPaymentAdjustment(ModifiedFollowing)
                    .withFixingDays(1)
                    .withGearings(1.5)
                    .withSpreads(0.002)
                    .withCallStrikes(0.03)
                    .withLongCallOption(Position::Short)
                    .withCallATM(true)
                    .withCallPayoffs(0.04)
                    .withPutStrikes(0.02)
                    .withLongPutOption(Position::Short)
                    .withPutATM(true)
                    .withPutPayoffs(0.015)
                    .withReplication(ext::make_shared<DigitalReplication>(
                        Replication::Central, 1e-3))
                    .withNakedOption(true)),
            true);

    emitLeg("digital_ibor_in_arrears",
            pricedDigitalIborLeg(DigitalIborLeg(s, idx)
                                     .withNotionals(kNominal)
                                     .withCallStrikes(0.03)
                                     .inArrears(true)),
            true);

    emitLeg("digital_ibor_zero_gearing",
            pricedDigitalIborLeg(DigitalIborLeg(s, idx)
                                     .withNotionals(kNominal)
                                     .withPaymentDayCounter(Actual360())
                                     .withGearings(0.0)
                                     .withSpreads(0.027)
                                     .withCallStrikes(0.03)),
            true);
}

void emitDigitalCmsLegs() {
    Schedule s = mainSchedule();
    auto idx = cms10y();

    emitLeg("digital_cms_default",
            DigitalCmsLeg(s, idx).withNotionals(kNominal).withCallStrikes(0.03), true);

    emitLeg("digital_cms_full",
            DigitalCmsLeg(s, idx)
                .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                .withPaymentDayCounter(Actual360())
                .withPaymentAdjustment(ModifiedFollowing)
                .withFixingDays(1)
                .withGearings(1.5)
                .withSpreads(0.002)
                .withCallStrikes(0.03)
                .withLongCallOption(Position::Short)
                .withCallATM(true)
                .withCallPayoffs(0.04)
                .withPutStrikes(0.02)
                .withLongPutOption(Position::Short)
                .withPutATM(true)
                .withPutPayoffs(0.015)
                .withReplication(
                    ext::make_shared<DigitalReplication>(Replication::Central, 1e-3))
                .withNakedOption(true),
            true);

    emitLeg("digital_cms_in_arrears",
            DigitalCmsLeg(s, idx)
                .withNotionals(kNominal)
                .withCallStrikes(0.03)
                .inArrears(true),
            true);

    emitLeg("digital_cms_zero_gearing",
            DigitalCmsLeg(s, idx)
                .withNotionals(kNominal)
                .withPaymentDayCounter(Actual360())
                .withGearings(0.0)
                .withSpreads(0.027)
                .withCallStrikes(0.03),
            true);
}

// ---------------------------------------------------------------------------
// AverageBMALeg
// ---------------------------------------------------------------------------

void emitAverageBmaLegs() {
    auto idx = ext::make_shared<BMAIndex>(flatCurve());
    // Quarterly, forward of the evaluation date: every weekly BMA fixing is a
    // forecast off the flat curve, so no history is needed.
    Schedule s(Date(15, February, 2024), Date(15, February, 2025), Period(3, Months),
               TARGET(), ModifiedFollowing, ModifiedFollowing, DateGeneration::Forward,
               false);

    emitLeg("bma_default", AverageBMALeg(s, idx).withNotionals(kNominal), true);

    emitLeg("bma_full",
            AverageBMALeg(s, idx)
                .withNotionals(std::vector<Real>{kNominal, 2.0 * kNominal})
                .withPaymentDayCounter(Actual360())
                .withPaymentAdjustment(Preceding)
                .withGearings(1.5)
                .withSpreads(0.002),
            true);

    Schedule u = unadjustedSchedule();
    emitLeg("bma_adj_following",
            AverageBMALeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Following),
            true);
    emitLeg("bma_adj_unadjusted",
            AverageBMALeg(u, idx).withNotionals(kNominal).withPaymentAdjustment(Unadjusted),
            true);

    emitLeg("bma_stub",
            AverageBMALeg(Schedule(Date(20, March, 2024), Date(15, February, 2025),
                                   Period(3, Months), TARGET(), ModifiedFollowing,
                                   ModifiedFollowing, DateGeneration::Backward, false),
                          idx)
                .withNotionals(kNominal),
            false);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    emitFixedRateLegs();
    emitIborLegs();
    emitOvernightLegs();
    emitCmsLegs();
    emitCpiLegs();
    emitDigitalIborLegs();
    emitDigitalCmsLegs();
    emitAverageBmaLegs();
    std::cout << "}\n";
    return 0;
}
