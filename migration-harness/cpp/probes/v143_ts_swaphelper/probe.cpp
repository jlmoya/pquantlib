// migration-harness/cpp/probes/v143_ts_swaphelper/probe.cpp
//
// Pins SwapRateHelper's four dates and its implied quote.
//
// C++ SwapRateHelper::initializeDates builds a real VanillaSwap through
// MakeVanillaSwap and reads earliestDate_ = swap->startDate(),
// maturityDate_ = swap->maturityDate() and
// latestRelevantDate_ = max(maturityDate_, lastCoupon->fixingEndDate())
// (ratehelpers.cpp). A port that instead approximates the schedule with
// calendar advances agrees on the easy cases and silently disagrees on the
// hard ones, and since Pillar::LastRelevantDate is the DEFAULT pillar choice
// a one-business-day error moves every bootstrapped curve node.
//
// The cases below are chosen so that the approximation and the real schedule
// can come apart:
//   * odd tenors (18M, 4M) where the backward schedule carries a stub;
//   * end-of-month rolls;
//   * a fixed/float calendar that is NOT the index's fixing calendar, so the
//     schedule is adjusted on one calendar and fixingEndDate is rolled on
//     another;
//   * indexed coupons, where fixingEndDate is the index's natural maturity
//     rather than the accrual end;
//   * explicit settlementDays vs the Null<Natural> default, which in
//     MakeVanillaSwap selects two different spot-date derivations
//     (index->valueDate on the fixing calendar vs advance on the float
//     calendar);
//   * forward-start swaps;
//   * every Pillar::Choice.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/swaphelper.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/cashflows/iborcoupon.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

using namespace QuantLib;

namespace {

const Date kEval(17, January, 2024);
// 26 February 2024 + 3 settlement days lands the spot date on 29 February,
// i.e. an end-of-month date, so the endOfMonth cases actually roll. Choosing
// the settlement offset (rather than the evaluation date) to hit month end
// keeps every fixing date strictly in the future, so no historical fixing is
// needed for this block.
const Date kEvalEom(26, February, 2024);

ext::shared_ptr<YieldTermStructure> flatCurve(const Date& eval) {
    return ext::make_shared<FlatForward>(eval, 0.03, Actual365Fixed(), Continuous, Annual);
}

bool firstEmitted = false;

void emit(const std::string& key, const ext::shared_ptr<SwapRateHelper>& h, const Date& eval) {
    auto curve = flatCurve(eval);
    h->setTermStructure(curve.get());
    const auto& swap = h->swap();
    const auto& floatLeg = swap->floatingLeg();
    auto lastCoupon = ext::dynamic_pointer_cast<IborCoupon>(floatLeg.back());

    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"earliest_date\": " << h->earliestDate().serialNumber() << ",\n"
              << "    \"maturity_date\": " << h->maturityDate().serialNumber() << ",\n"
              << "    \"latest_relevant_date\": " << h->latestRelevantDate().serialNumber() << ",\n"
              << "    \"pillar_date\": " << h->pillarDate().serialNumber() << ",\n"
              << "    \"latest_date\": " << h->latestDate().serialNumber() << ",\n"
              << "    \"swap_start_date\": " << swap->startDate().serialNumber() << ",\n"
              << "    \"swap_maturity_date\": " << swap->maturityDate().serialNumber() << ",\n"
              << "    \"n_float_coupons\": " << floatLeg.size() << ",\n"
              << "    \"last_coupon_accrual_start\": " << lastCoupon->accrualStartDate().serialNumber() << ",\n"
              << "    \"last_coupon_accrual_end\": " << lastCoupon->accrualEndDate().serialNumber() << ",\n"
              << "    \"last_coupon_fixing_date\": " << lastCoupon->fixingDate().serialNumber() << ",\n"
              << "    \"last_coupon_fixing_value_date\": " << lastCoupon->fixingValueDate().serialNumber() << ",\n"
              << "    \"last_coupon_fixing_end_date\": " << lastCoupon->fixingEndDate().serialNumber() << ",\n"
              << "    \"last_coupon_fixing_maturity_date\": " << lastCoupon->fixingMaturityDate().serialNumber() << ",\n"
              << "    \"implied_quote\": " << h->impliedQuote() << "\n"
              << "  }";
}

// Full-signature construction; every knob explicit so the probe reads as the
// specification of what the port has to reproduce.
ext::shared_ptr<SwapRateHelper> makeHelper(
    const Period& tenor,
    const Calendar& calendar,
    Frequency fixedFrequency,
    BusinessDayConvention fixedConvention,
    const DayCounter& fixedDayCount,
    const ext::shared_ptr<IborIndex>& index,
    const Period& fwdStart = 0 * Days,
    Natural settlementDays = Null<Natural>(),
    Pillar::Choice pillar = Pillar::LastRelevantDate,
    Date customPillarDate = Date(),
    bool endOfMonth = false,
    const ext::optional<bool>& useIndexedCoupons = ext::nullopt,
    Real spread = Null<Real>()) {
    Handle<Quote> spreadHandle;
    if (spread != Null<Real>())
        spreadHandle = Handle<Quote>(ext::make_shared<SimpleQuote>(spread));
    return ext::make_shared<SwapRateHelper>(
        0.04, tenor, calendar, fixedFrequency, fixedConvention, fixedDayCount, index,
        spreadHandle, fwdStart, Handle<YieldTermStructure>(), settlementDays, pillar,
        customPillarDate, endOfMonth, useIndexedCoupons);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const DayCounter thirty360 = Thirty360(Thirty360::BondBasis);
    const Calendar target = TARGET();
    // UnitedStates(GovernmentBond) is open on TARGET-only holidays (1 May,
    // 26 Dec, Easter Monday), so a schedule adjusted on it can land on a day
    // the index's own fixing calendar (TARGET) treats as a holiday.
    const Calendar us = UnitedStates(UnitedStates::GovernmentBond);

    {
        Settings::instance().evaluationDate() = kEval;
        auto e6m = ext::make_shared<Euribor6M>();
        auto e3m = ext::make_shared<Euribor3M>();
        // Two cases below start on or before the evaluation date and so need
        // the corresponding past fixing: the settlementDays == 0 case fixes on
        // 15 January 2024, the fwdStart == -1M case on 15 December 2023. Both
        // are strictly before 17 January 2024, so no "fixes today" tie-break
        // is involved.
        e6m->addFixing(Date(15, December, 2023), 0.038);
        e6m->addFixing(Date(15, January, 2024), 0.039);

        emit("eur_5y_euribor6m",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m), kEval);
        emit("eur_5y_euribor3m",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e3m), kEval);
        // Odd tenor: the backward-generated float schedule carries a stub.
        emit("eur_18m_euribor6m",
             makeHelper(18 * Months, target, Annual, ModifiedFollowing, thirty360, e6m), kEval);
        emit("eur_4m_euribor3m",
             makeHelper(4 * Months, target, Quarterly, ModifiedFollowing, thirty360, e3m), kEval);
        emit("eur_1y_euribor12m_once",
             makeHelper(1 * Years, target, Once, ModifiedFollowing, thirty360,
                        ext::make_shared<Euribor1Y>()), kEval);
        // Forward-start.
        emit("eur_5y_fwd1y",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        1 * Years), kEval);
        // Negative forward start: the branch where MakeVanillaSwap adjusts the
        // start date Preceding rather than Following.
        emit("eur_5y_fwd_minus1m",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        -1 * Months), kEval);
        // settlementDays explicitly given (selects MakeVanillaSwap's other
        // spot-date branch) vs the Null<Natural> default used above.
        emit("eur_5y_settlement0",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 0), kEval);
        emit("eur_5y_settlement5",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 5), kEval);
        // Schedule calendar != index fixing calendar.
        emit("us_cal_5y_euribor6m",
             makeHelper(5 * Years, us, Annual, ModifiedFollowing, thirty360, e6m), kEval);
        emit("us_cal_7y_euribor6m",
             makeHelper(7 * Years, us, Annual, ModifiedFollowing, thirty360, e6m), kEval);
        emit("us_cal_10y_euribor3m",
             makeHelper(10 * Years, us, Annual, ModifiedFollowing, thirty360, e3m), kEval);
        // Indexed coupons: fixingEndDate becomes the index's natural maturity.
        emit("eur_18m_indexed",
             makeHelper(18 * Months, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, Null<Natural>(), Pillar::LastRelevantDate, Date(), false,
                        true), kEval);
        emit("us_cal_5y_indexed",
             makeHelper(5 * Years, us, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, Null<Natural>(), Pillar::LastRelevantDate, Date(), false,
                        true), kEval);
        emit("eur_5y_at_par_explicit",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, Null<Natural>(), Pillar::LastRelevantDate, Date(), false,
                        false), kEval);
        // Pillar choices.
        emit("eur_5y_pillar_maturity",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, Null<Natural>(), Pillar::MaturityDate), kEval);
        emit("eur_5y_pillar_custom",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, Null<Natural>(), Pillar::CustomDate,
                        Date(19, January, 2027)), kEval);
        // Non-zero spread moves the implied quote (dates unchanged).
        emit("eur_5y_spread_25bp",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, Null<Natural>(), Pillar::LastRelevantDate, Date(), false,
                        ext::nullopt, 0.0025), kEval);
        // Semi-annual fixed leg + Actual/360.
        emit("eur_7y_semi_act360",
             makeHelper(7 * Years, target, Semiannual, Following, Actual360(), e6m), kEval);
    }

    {
        Settings::instance().evaluationDate() = kEvalEom;
        auto e6m = ext::make_shared<Euribor6M>();
        emit("eom_5y_euribor6m",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 3, Pillar::LastRelevantDate, Date(), true),
             kEvalEom);
        emit("eom_off_5y_euribor6m",
             makeHelper(5 * Years, target, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 3, Pillar::LastRelevantDate, Date(), false),
             kEvalEom);
        emit("eom_us_cal_5y_euribor6m",
             makeHelper(5 * Years, us, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 3, Pillar::LastRelevantDate, Date(), true),
             kEvalEom);
    }

    {
        // The case that actually separates latestRelevantDate from
        // maturityDate. Evaluation 26 April 2023 + 3 settlement days rolled on
        // the US calendar puts the spot date on Monday 1 May 2023, and five
        // years later the maturity lands on Monday 1 May 2028 — a US business
        // day but a TARGET holiday (Labour Day). The last coupon's accrual
        // ends there, so the par-coupon fixingEndDate round trip
        // (advance -2 business days, then +2, on the index's TARGET fixing
        // calendar) cannot come back to 1 May and overshoots to 2 May. Hence
        // latestRelevantDate == maturityDate + 1 day, and since
        // Pillar::LastRelevantDate is the default the curve node moves with it.
        const Date evalMay(26, April, 2023);
        Settings::instance().evaluationDate() = evalMay;
        auto e6m = ext::make_shared<Euribor6M>();
        emit("us_cal_5y_target_holiday_maturity",
             makeHelper(5 * Years, us, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 3),
             evalMay);
        emit("us_cal_5y_target_holiday_maturity_indexed",
             makeHelper(5 * Years, us, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 3, Pillar::LastRelevantDate, Date(), false, true),
             evalMay);
        emit("us_cal_5y_target_holiday_maturity_pillar_maturity",
             makeHelper(5 * Years, us, Annual, ModifiedFollowing, thirty360, e6m,
                        0 * Days, 3, Pillar::MaturityDate),
             evalMay);
    }

    std::cout << "\n}\n";
    return 0;
}
