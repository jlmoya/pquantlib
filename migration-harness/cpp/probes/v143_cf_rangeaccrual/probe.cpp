// migration-harness/cpp/probes/v143_cf_rangeaccrual/probe.cpp
//
// Reference values for the range-accrual family of ql/cashflows/rangeaccrual.hpp
// + .cpp @ v1.43:
//
//   * RangeAccrualFloatersCoupon — FloatingRateCoupon carrying an observation
//     schedule plus a [lowerTrigger, upperTrigger] band.
//   * RangeAccrualPricer        — abstract base; supplies swapletRate() and the
//     per-coupon state (initialValues_, discount_, spreadLegValue_, ...).
//   * RangeAccrualPricerByBgm   — the Brace-Gatarek-Musiela smile pricer.
//   * RangeAccrualLeg           — the chained builder.
//
// The two [[deprecated]] overloads (the ctor taking ext::shared_ptr<Schedule>
// and observationsSchedule()) are deliberately NOT probed — they are pure
// forwarders scheduled for removal, and PQuantLib does not port them.
//
// WHAT IS PINNED AND WHY
// ----------------------
// 1. Structure, per coupon: payment date, nominal, accrual start/end, accrual
//    period, fixingDays/fixingDate, gearing, spread, startTime, endTime,
//    lowerTrigger, upperTrigger, observationsNo, the observationDates serial
//    list and observationTimes. These are pure calendar/day-count arithmetic —
//    a wrong day counter or a dropped builder argument shows up here first and
//    nowhere else, because the NPV can still match while two errors cancel.
//
// 2. priceWithoutOptionality(discountCurve) — accrualPeriod * (g*L + s) *
//    nominal * df. Pricer-independent, so it isolates the coupon from the
//    pricer.
//
// 3. rate() / amount() / price(discountCurve) per coupon plus the leg NPV, with
//    a RangeAccrualPricerByBgm attached in ALL FOUR (withSmile, byCallSpread)
//    combinations. Those four select genuinely different code paths:
//        withSmile=false            -> digitalPriceWithoutSmile   (byCallSpread
//                                      is unused; both combos must agree, which
//                                      is itself pinned)
//        withSmile=true , cs=true   -> digitalPriceWithSmile -> callSpreadPrice
//        withSmile=true , cs=false  -> digitalPriceWithoutSmile + smileCorrection
//
// 4. RangeAccrualLeg builder arguments. Every setter is exercised at a
//    NON-DEFAULT value with an observable consequence, several as A/B pairs of
//    otherwise-identical legs (payment adjustment, observation convention,
//    observation tenor) so that an argument which is accepted and silently
//    dropped cannot pass.
//
// 5. The C++ v1.43 defect in RangeAccrualLeg::operator Leg(): it does
//    `Leg leg(n);` (n default-constructed, i.e. NULL, shared_ptrs) and then
//    push_back()s n coupons, so the returned Leg has 2n entries of which the
//    first n are null. We pin raw_leg_size / null_prefix explicitly so the
//    divergence is documented rather than accidental; NPV is taken over the
//    compacted leg (dereferencing a null entry would crash).
//
// SMILE FIXTURES
// --------------
// RangeAccrualPricerByBgm takes two SmileSections. A *flat* smile makes
// smileCorrection identically zero (its finite-difference dSigma/dK vanishes),
// which would leave the withSmile=true/byCallSpread=false path untested. So the
// probe uses a local affine section, vol(K) = base + slope*(K - anchor), which
// is trivially reproducible in Python and gives a non-zero, exactly-known
// dSigma/dK. FlatSmileSection is probed too, precisely to pin that the
// correction collapses to zero there.
//
// Everything (curve, index, schedules) is built inline from literal tables; the
// evaluation date is pinned so index fixings are always forecast, never looked
// up in a fixing history.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/rangeaccrual.json.

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/rangeaccrual.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/flatsmilesection.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --------------------------------------------------------------------------
// affine smile fixture: vol(K) = base + slope * (K - anchor)
// --------------------------------------------------------------------------
class AffineSmileSection : public SmileSection {
  public:
    AffineSmileSection(Time exerciseTime, Real base, Real slope, Real anchor, Real atm)
    : SmileSection(exerciseTime, Actual365Fixed()), base_(base), slope_(slope), anchor_(anchor),
      atm_(atm) {}

    Real minStrike() const override { return -1.0; }
    Real maxStrike() const override { return 10.0; }
    Real atmLevel() const override { return atm_; }

  protected:
    Volatility volatilityImpl(Rate strike) const override {
        // Split across statements so the compiler cannot contract this into an
        // FMA: RangeAccrualPricerByBgm differentiates the smile by a 1e-8
        // finite difference, which amplifies any 1-ulp wobble by 1e8.
        const Real dk = strike - anchor_;
        const Real scaled = slope_ * dk;
        return base_ + scaled;
    }

  private:
    Real base_, slope_, anchor_, atm_;
};

// --------------------------------------------------------------------------
// pinned market
// --------------------------------------------------------------------------
const Date kEvalDate(15, February, 2024);
const Real kCorrelation = 0.7;

const Real kAffineExpiryBase = 0.25;
const Real kAffineExpirySlope = -2.0;
const Real kAffinePaymentBase = 0.22;
const Real kAffinePaymentSlope = -1.5;
const Real kSmileAnchor = 0.04;
const Real kSmileAtm = 0.04;
const Real kFlatExpiryVol = 0.25;
const Real kFlatPaymentVol = 0.22;

struct Market {
    Calendar calendar;
    std::vector<Date> curveDates;
    std::vector<Rate> curveZeros;
    RelinkableHandle<YieldTermStructure> curve;
    ext::shared_ptr<IborIndex> index;
};

Market makeMarket() {
    Market m;
    m.calendar = TARGET();
    m.curveDates = {Date(15, February, 2024), Date(15, August, 2024),   Date(15, February, 2025),
                    Date(15, February, 2026), Date(15, February, 2027), Date(15, February, 2029),
                    Date(15, February, 2034)};
    m.curveZeros = {0.0300, 0.0320, 0.0345, 0.0375, 0.0395, 0.0420, 0.0450};
    m.curve.linkTo(ext::shared_ptr<YieldTermStructure>(
        new ZeroCurve(m.curveDates, m.curveZeros, Actual365Fixed())));
    m.index = ext::shared_ptr<IborIndex>(new Euribor6M(m.curve));
    return m;
}

// --------------------------------------------------------------------------
// JSON helpers
// --------------------------------------------------------------------------
void emitSerials(const std::vector<Date>& dates) {
    std::cout << "[";
    for (Size i = 0; i < dates.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << dates[i].serialNumber();
    std::cout << "]";
}

void emitReals(const std::vector<Real>& xs) {
    std::cout << "[";
    for (Size i = 0; i < xs.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << xs[i];
    std::cout << "]";
}

void emitSizes(const std::vector<Size>& xs) {
    std::cout << "[";
    for (Size i = 0; i < xs.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << xs[i];
    std::cout << "]";
}

// --------------------------------------------------------------------------
// The v1.43 `Leg leg(n)` + push_back defect: the returned Leg carries n leading
// null shared_ptrs. Split it so the rest of the probe can work with real flows.
// --------------------------------------------------------------------------
Leg compact(const Leg& raw, Size& nullPrefix) {
    Leg out;
    nullPrefix = 0;
    for (const auto& cf : raw) {
        if (cf == nullptr) {
            if (out.empty())
                ++nullPrefix;
            continue;
        }
        out.push_back(cf);
    }
    return out;
}

// --------------------------------------------------------------------------
// leg builders — every one names its non-default setters explicitly
// --------------------------------------------------------------------------
Schedule scheduleMain(const Market& m) {
    return {Date(17, March, 2025),
            Date(17, March, 2027),
            Period(6, Months),
            m.calendar,
            ModifiedFollowing,
            ModifiedFollowing,
            DateGeneration::Forward,
            false};
}

Schedule scheduleVectors(const Market& m) {
    return {Date(16, June, 2025),
            Date(16, December, 2026),
            Period(6, Months),
            m.calendar,
            ModifiedFollowing,
            ModifiedFollowing,
            DateGeneration::Forward,
            false};
}

Schedule scheduleFixed(const Market& m) {
    return {Date(15, September, 2025),
            Date(15, September, 2026),
            Period(6, Months),
            m.calendar,
            ModifiedFollowing,
            ModifiedFollowing,
            DateGeneration::Forward,
            false};
}

// Raw (Unadjusted) schedule whose 2026-05-17 termination falls on a Sunday, so
// the payment adjustment actually moves the payment date.
Schedule scheduleUnadjusted(const Market& m) {
    return {Date(17, May, 2025),
            Date(17, May, 2026),
            Period(6, Months),
            m.calendar,
            Unadjusted,
            Unadjusted,
            DateGeneration::Forward,
            false};
}

Leg legMain(const Market& m) {
    return RangeAccrualLeg(scheduleMain(m), m.index)
        .withNotionals(std::vector<Real>{1000000.0, 900000.0, 800000.0})  // vector overload
        .withPaymentDayCounter(Actual360())
        .withPaymentAdjustment(ModifiedFollowing)
        .withFixingDays(std::vector<Natural>{3, 1})  // vector overload, default is 2
        .withGearings(0.75)                          // scalar, default 1.0
        .withSpreads(0.0025)                         // scalar, default 0.0
        .withLowerTriggers(0.030)                    // scalar
        .withUpperTriggers(0.055)                    // scalar
        .withObservationTenor(Period(1, Months))
        .withObservationConvention(ModifiedFollowing);
}

Leg legVectors(const Market& m) {
    return RangeAccrualLeg(scheduleVectors(m), m.index)
        .withNotionals(2000000.0)  // scalar overload
        .withPaymentDayCounter(Actual365Fixed())
        .withPaymentAdjustment(Following)
        .withFixingDays(0U)                                       // scalar, default is 2
        .withGearings(std::vector<Real>{1.25, 0.5})               // vector overload
        .withSpreads(std::vector<Spread>{-0.001, 0.002})          // vector overload
        .withLowerTriggers(std::vector<Rate>{0.025, 0.028, 0.031})  // vector overload
        .withUpperTriggers(std::vector<Rate>{0.060, 0.058, 0.056})  // vector overload
        .withObservationTenor(Period(2, Months))
        .withObservationConvention(Unadjusted);
}

// gearing == 0 selects the FixedRateCoupon branch of operator Leg().
Leg legFixed(const Market& m) {
    return RangeAccrualLeg(scheduleFixed(m), m.index)
        .withNotionals(500000.0)
        .withPaymentDayCounter(Actual360())
        .withGearings(0.0)
        .withSpreads(0.035)
        .withLowerTriggers(0.030)
        .withUpperTriggers(0.055)
        .withObservationTenor(Period(1, Months));
}

Leg legUnadjusted(const Market& m, BusinessDayConvention paymentAdjustment) {
    return RangeAccrualLeg(scheduleUnadjusted(m), m.index)
        .withNotionals(1000000.0)
        .withPaymentDayCounter(Actual360())
        .withPaymentAdjustment(paymentAdjustment)
        .withGearings(1.0)
        .withLowerTriggers(0.030)
        .withUpperTriggers(0.055)
        .withObservationTenor(Period(3, Months))
        // Unadjusted is mandatory here: the coupon ctor requires the observation
        // schedule to start/end exactly on the (weekend) accrual dates.
        .withObservationConvention(Unadjusted);
}

// legVectors with a single setter changed, for A/B comparison.
Leg legVectorsVariant(const Market& m, const Period& observationTenor,
                      BusinessDayConvention observationConvention) {
    return RangeAccrualLeg(scheduleVectors(m), m.index)
        .withNotionals(2000000.0)
        .withPaymentDayCounter(Actual365Fixed())
        .withPaymentAdjustment(Following)
        .withFixingDays(0U)
        .withGearings(std::vector<Real>{1.25, 0.5})
        .withSpreads(std::vector<Spread>{-0.001, 0.002})
        .withLowerTriggers(std::vector<Rate>{0.025, 0.028, 0.031})
        .withUpperTriggers(std::vector<Rate>{0.060, 0.058, 0.056})
        .withObservationTenor(observationTenor)
        .withObservationConvention(observationConvention);
}

// --------------------------------------------------------------------------
// emission
// --------------------------------------------------------------------------
void emitCoupon(const Market& m, const ext::shared_ptr<CashFlow>& cf, const std::string& indent,
                bool trailingComma) {
    auto c = ext::dynamic_pointer_cast<RangeAccrualFloatersCoupon>(cf);
    QL_REQUIRE(c != nullptr, "expected a RangeAccrualFloatersCoupon");

    std::vector<Date> obsDates(c->observationDates());
    std::vector<Real> obsTimes(c->observationTimes());

    std::cout << indent << "{\n"
              << indent << "  \"payment_date\": " << c->date().serialNumber() << ",\n"
              << indent << "  \"nominal\": " << c->nominal() << ",\n"
              << indent << "  \"accrual_start_date\": " << c->accrualStartDate().serialNumber()
              << ",\n"
              << indent << "  \"accrual_end_date\": " << c->accrualEndDate().serialNumber() << ",\n"
              << indent << "  \"accrual_period\": " << c->accrualPeriod() << ",\n"
              << indent << "  \"day_counter\": \"" << c->dayCounter().name() << "\",\n"
              << indent << "  \"fixing_days\": " << c->fixingDays() << ",\n"
              << indent << "  \"fixing_date\": " << c->fixingDate().serialNumber() << ",\n"
              << indent << "  \"index_fixing\": " << c->indexFixing() << ",\n"
              << indent << "  \"gearing\": " << c->gearing() << ",\n"
              << indent << "  \"spread\": " << c->spread() << ",\n"
              << indent << "  \"start_time\": " << c->startTime() << ",\n"
              << indent << "  \"end_time\": " << c->endTime() << ",\n"
              << indent << "  \"lower_trigger\": " << c->lowerTrigger() << ",\n"
              << indent << "  \"upper_trigger\": " << c->upperTrigger() << ",\n"
              << indent << "  \"observations_no\": " << c->observationsNo() << ",\n"
              << indent << "  \"observation_schedule_dates\": ";
    emitSerials(std::vector<Date>(c->observationSchedule().dates()));
    std::cout << ",\n" << indent << "  \"observation_dates\": ";
    emitSerials(obsDates);
    std::cout << ",\n" << indent << "  \"observation_times\": ";
    emitReals(obsTimes);
    std::cout << ",\n"
              << indent << "  \"price_without_optionality\": "
              << c->priceWithoutOptionality(m.curve) << "\n"
              << indent << "}" << (trailingComma ? "," : "") << "\n";
}

void emitPricedLeg(const Market& m, const Leg& leg, const std::string& key,
                   const ext::shared_ptr<SmileSection>& onExpiry,
                   const ext::shared_ptr<SmileSection>& onPayment, bool withSmile,
                   bool byCallSpread, const std::string& indent, bool trailingComma) {
    auto pricer = ext::make_shared<RangeAccrualPricerByBgm>(kCorrelation, onExpiry, onPayment,
                                                            withSmile, byCallSpread);
    setCouponPricer(leg, pricer);

    std::vector<Real> rates;
    std::vector<Real> amounts;
    std::vector<Real> prices;
    for (const auto& cf : leg) {
        auto c = ext::dynamic_pointer_cast<RangeAccrualFloatersCoupon>(cf);
        QL_REQUIRE(c != nullptr, "expected a RangeAccrualFloatersCoupon");
        rates.push_back(c->rate());
        amounts.push_back(c->amount());
        prices.push_back(c->price(m.curve));
    }

    std::cout << indent << "\"" << key << "\": {\n" << indent << "  \"rates\": ";
    emitReals(rates);
    std::cout << ",\n" << indent << "  \"amounts\": ";
    emitReals(amounts);
    std::cout << ",\n" << indent << "  \"prices\": ";
    emitReals(prices);
    std::cout << ",\n"
              << indent << "  \"npv\": "
              << CashFlows::npv(leg, **m.curve, false, kEvalDate) << "\n"
              << indent << "}" << (trailingComma ? "," : "") << "\n";
}

void emitLegStructure(const Market& m, const Schedule& schedule, const Leg& raw,
                      const std::string& key, bool trailingComma) {
    Size nullPrefix = 0;
    const Leg leg = compact(raw, nullPrefix);

    std::cout << "  \"" << key << "\": {\n    \"schedule_dates\": ";
    emitSerials(std::vector<Date>(schedule.dates()));
    std::cout << ",\n"
              << "    \"raw_leg_size\": " << raw.size() << ",\n"
              << "    \"null_prefix\": " << nullPrefix << ",\n"
              << "    \"coupon_count\": " << leg.size() << ",\n"
              << "    \"coupons\": [\n";
    for (Size i = 0; i < leg.size(); ++i)
        emitCoupon(m, leg[i], "      ", i + 1 != leg.size());
    std::cout << "    ]\n  }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kEvalDate;
    std::cout << std::setprecision(17);

    const Market m = makeMarket();

    const ext::shared_ptr<SmileSection> affineExpiry = ext::make_shared<AffineSmileSection>(
        1.5, kAffineExpiryBase, kAffineExpirySlope, kSmileAnchor, kSmileAtm);
    const ext::shared_ptr<SmileSection> affinePayment = ext::make_shared<AffineSmileSection>(
        2.0, kAffinePaymentBase, kAffinePaymentSlope, kSmileAnchor, kSmileAtm);
    const ext::shared_ptr<SmileSection> flatExpiry =
        ext::make_shared<FlatSmileSection>(1.5, kFlatExpiryVol, Actual365Fixed(), kSmileAtm);
    const ext::shared_ptr<SmileSection> flatPayment =
        ext::make_shared<FlatSmileSection>(2.0, kFlatPaymentVol, Actual365Fixed(), kSmileAtm);

    std::cout << "{\n";

    // ---- setup ------------------------------------------------------------
    std::cout << "  \"setup\": {\n"
              << "    \"evaluation_date\": " << kEvalDate.serialNumber() << ",\n"
              << "    \"curve_dates\": ";
    emitSerials(m.curveDates);
    std::cout << ",\n    \"curve_zeros\": ";
    emitReals(m.curveZeros);
    std::cout << ",\n"
              << "    \"index_name\": \"" << m.index->name() << "\",\n"
              << "    \"correlation\": " << kCorrelation << ",\n"
              << "    \"affine_expiry\": {\"base\": " << kAffineExpiryBase
              << ", \"slope\": " << kAffineExpirySlope << ", \"anchor\": " << kSmileAnchor
              << ", \"atm\": " << kSmileAtm << ", \"exercise_time\": 1.5},\n"
              << "    \"affine_payment\": {\"base\": " << kAffinePaymentBase
              << ", \"slope\": " << kAffinePaymentSlope << ", \"anchor\": " << kSmileAnchor
              << ", \"atm\": " << kSmileAtm << ", \"exercise_time\": 2.0},\n"
              << "    \"flat_expiry_vol\": " << kFlatExpiryVol << ",\n"
              << "    \"flat_payment_vol\": " << kFlatPaymentVol << "\n"
              << "  },\n";

    // ---- structural legs --------------------------------------------------
    emitLegStructure(m, scheduleMain(m), legMain(m), "leg_main", true);
    emitLegStructure(m, scheduleVectors(m), legVectors(m), "leg_vectors", true);

    // ---- leg_main priced in every (withSmile, byCallSpread) combination ----
    {
        Size nullPrefix = 0;
        const Leg leg = compact(legMain(m), nullPrefix);
        std::cout << "  \"leg_main_pricers\": {\n";
        emitPricedLeg(m, leg, "affine_nosmile_nocs", affineExpiry, affinePayment, false, false,
                      "    ", true);
        emitPricedLeg(m, leg, "affine_nosmile_cs", affineExpiry, affinePayment, false, true,
                      "    ", true);
        emitPricedLeg(m, leg, "affine_smile_nocs", affineExpiry, affinePayment, true, false,
                      "    ", true);
        emitPricedLeg(m, leg, "affine_smile_cs", affineExpiry, affinePayment, true, true, "    ",
                      true);
        emitPricedLeg(m, leg, "flat_nosmile_nocs", flatExpiry, flatPayment, false, false, "    ",
                      true);
        emitPricedLeg(m, leg, "flat_smile_nocs", flatExpiry, flatPayment, true, false, "    ",
                      true);
        emitPricedLeg(m, leg, "flat_smile_cs", flatExpiry, flatPayment, true, true, "    ", false);
        std::cout << "  },\n";
    }

    // ---- leg_vectors priced, to prove the vector setters reach the pricer --
    {
        Size nullPrefix = 0;
        const Leg leg = compact(legVectors(m), nullPrefix);
        std::cout << "  \"leg_vectors_pricers\": {\n";
        emitPricedLeg(m, leg, "affine_smile_cs", affineExpiry, affinePayment, true, true, "    ",
                      false);
        std::cout << "  },\n";
    }

    // ---- gearing == 0 -> FixedRateCoupon branch ---------------------------
    {
        Size nullPrefix = 0;
        const Leg leg = compact(legFixed(m), nullPrefix);
        std::cout << "  \"leg_zero_gearing\": {\n"
                  << "    \"raw_leg_size\": " << (leg.size() * 2) << ",\n"
                  << "    \"null_prefix\": " << nullPrefix << ",\n"
                  << "    \"coupon_count\": " << leg.size() << ",\n"
                  << "    \"coupons\": [\n";
        for (Size i = 0; i < leg.size(); ++i) {
            auto c = ext::dynamic_pointer_cast<FixedRateCoupon>(leg[i]);
            QL_REQUIRE(c != nullptr, "expected a FixedRateCoupon for gearing == 0");
            std::cout << "      {\"payment_date\": " << c->date().serialNumber()
                      << ", \"nominal\": " << c->nominal()
                      << ", \"accrual_start_date\": " << c->accrualStartDate().serialNumber()
                      << ", \"accrual_end_date\": " << c->accrualEndDate().serialNumber()
                      << ", \"accrual_period\": " << c->accrualPeriod()
                      << ", \"rate\": " << c->rate() << ", \"amount\": " << c->amount() << "}"
                      << (i + 1 != leg.size() ? "," : "") << "\n";
        }
        std::cout << "    ],\n"
                  << "    \"npv\": " << CashFlows::npv(leg, **m.curve, false, kEvalDate) << "\n"
                  << "  },\n";
    }

    // ---- A/B: withPaymentAdjustment ---------------------------------------
    {
        std::cout << "  \"variant_payment_adjustment\": {\n    \"schedule_dates\": ";
        emitSerials(std::vector<Date>(scheduleUnadjusted(m).dates()));
        std::cout << ",\n";
        const char* names[] = {"preceding", "following", "modified_following"};
        const BusinessDayConvention conventions[] = {Preceding, Following, ModifiedFollowing};
        for (Size k = 0; k < 3; ++k) {
            Size nullPrefix = 0;
            const Leg leg = compact(legUnadjusted(m, conventions[k]), nullPrefix);
            std::vector<Date> payments;
            for (const auto& cf : leg)
                payments.push_back(cf->date());
            std::cout << "    \"" << names[k] << "\": ";
            emitSerials(payments);
            std::cout << (k + 1 != 3 ? "," : "") << "\n";
        }
        std::cout << "  },\n";
    }

    // ---- A/B: withObservationConvention and withObservationTenor -----------
    {
        struct Variant {
            const char* name;
            Period tenor;
            BusinessDayConvention convention;
        };
        const std::vector<Variant> variants = {
            {"tenor_2m_unadjusted", Period(2, Months), Unadjusted},
            {"tenor_2m_modified_following", Period(2, Months), ModifiedFollowing},
            {"tenor_3m_unadjusted", Period(3, Months), Unadjusted},
        };
        std::cout << "  \"variant_observation\": {\n";
        for (Size k = 0; k < variants.size(); ++k) {
            Size nullPrefix = 0;
            const Leg leg =
                compact(legVectorsVariant(m, variants[k].tenor, variants[k].convention),
                        nullPrefix);
            std::vector<Size> counts;
            std::cout << "    \"" << variants[k].name << "\": {\"observation_dates\": [";
            for (Size i = 0; i < leg.size(); ++i) {
                auto c = ext::dynamic_pointer_cast<RangeAccrualFloatersCoupon>(leg[i]);
                QL_REQUIRE(c != nullptr, "expected a RangeAccrualFloatersCoupon");
                counts.push_back(c->observationsNo());
                if (i != 0U)
                    std::cout << ", ";
                emitSerials(std::vector<Date>(c->observationDates()));
            }
            std::cout << "], \"observations_no\": ";
            emitSizes(counts);
            std::cout << "}" << (k + 1 != variants.size() ? "," : "") << "\n";
        }
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
