// migration-harness/cpp/probes/v143_ts_ratehelpers/probe.cpp
//
// Pins three rate helpers that PQuantLib was missing:
//
//   * BMASwapRateHelper            (ql/termstructures/yield/ratehelpers.{hpp,cpp})
//   * CPIBondHelper                (ql/termstructures/yield/bondhelpers.{hpp,cpp})
//   * MultipleResetsSwapRateHelper (ql/termstructures/yield/multipleresetsswaphelper.{hpp,cpp})
//
// What is hard about each, i.e. what these cases are here to catch:
//
// BMASwapRateHelper::initializeDates (ratehelpers.cpp:675-723) is the only
// helper in the file whose latestDate_ is NOT any date on the underlying
// instrument.  It is
//
//     d              = calendar_.adjust(swap_->maturityDate(), Following)
//     nextWednesday  = (d.weekday() >= 4) ? d + (11 - w) : d + (4 - w)
//     latestDate_    = clonedIndex->valueDate(
//                          clonedIndex->fixingCalendar().adjust(nextWednesday))
//
// i.e. up to a fortnight past the swap's own maturity, and it depends on the
// WEEKDAY of the adjusted maturity — so a port that returns the swap maturity
// (or the schedule end) agrees on nothing.  earliestDate_ is likewise not a
// plain "today + settlementDays": the evaluation date is first rolled on the
// JOINT calendar of the helper calendar and the ibor index's fixing calendar,
// and only then advanced on the helper calendar alone.  The two steps use
// different calendars on purpose, so every evaluation date below that is a
// holiday on exactly one of them separates the two.
//
// Evaluation dates cover: a plain business day, a Saturday, a Sunday, a US
// holiday that is a UK business day (4 July), a UK holiday that is a US
// business day (26 August, Summer bank holiday), and a day both are closed
// (1 January).  Tenors cover 1Y/2Y/5Y/10Y and the odd 18M/9M, where the
// backward-generated schedules carry a stub.
//
// CPIBondHelper (bondhelpers.cpp:110-165) is a BondHelper over a CPIBond, so
// its dates are the BOND's, not the schedule's:
//     latestDate_   = bond->cashflows().back()->date()
//     earliestDate_ = bond->nextCashFlowDate()
// The second moves with the evaluation date, so it is emitted at three of
// them.  impliedQuote is the clean or dirty price off the curve being
// bootstrapped, and the coupons need the UKRPI fixing history plus a
// zero-inflation curve — both pinned literally below so the Python test can
// reproduce the market node for node, and the history is truncated per
// evaluation date for the reason makeCpiMarket sets out.  The bond
// deliberately matures on a Saturday-adjacent schedule date so that Following
// pushes the redemption past the bond's own maturity date and latestDate_
// separates from it.
//
// MultipleResetsSwapRateHelper (multipleresetsswaphelper.cpp:56-72) sets
//     earliestDate_ = swap->startDate()
//     latestRelevantDate_ = latestDate_
//         = max(fixedLeg.back()->date(), floatingLeg.back()->date())
// — the PAYMENT dates, not the accrual ends, so a fixed leg rolled on a
// different convention from the floating one can win the max.  It has no
// Pillar::Choice at all: pillarDate() therefore falls through to latestDate().
// Cases vary resetsPerCoupon, the averaging method, the spread, all three
// fixed-leg overrides and the presence of an exogenous discount curve.
//
// Every date is emitted BOTH as a serial number and as an ISO string, so a
// date bug cannot hide behind a plausible-looking serial.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/ratehelpers.json.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/coupon.hpp>
#include <ql/indexes/bmaindex.hpp>
#include <ql/indexes/indexmanager.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/usdlibor.hpp>
#include <ql/indexes/inflation/ukrpi.hpp>
#include <ql/instruments/bmaswap.hpp>
#include <ql/instruments/multipleresetsswap.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/inflation/interpolatedzeroinflationcurve.hpp>
#include <ql/termstructures/inflationtermstructure.hpp>
#include <ql/termstructures/yield/bondhelpers.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/multipleresetsswaphelper.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/time/calendars/jointcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// JSON emission helpers
// ---------------------------------------------------------------------------

bool firstEmitted = false;

void openObject(const std::string& key) {
    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n";
}

std::string isoOf(const Date& d) {
    std::ostringstream o;
    o << d.year() << '-' << std::setw(2) << std::setfill('0') << int(d.month()) << '-'
      << std::setw(2) << std::setfill('0') << d.dayOfMonth();
    return o.str();
}

// Serial AND ISO for every date: a wrong date cannot masquerade as a
// plausible-looking serial number.
void emitDate(const std::string& name, const Date& d) {
    std::cout << "    \"" << name << "_serial\": " << d.serialNumber() << ",\n"
              << "    \"" << name << "_iso\": \"" << isoOf(d) << "\",\n";
}

void emitHelperDates(const ext::shared_ptr<RateHelper>& h) {
    emitDate("earliest", h->earliestDate());
    emitDate("latest", h->latestDate());
    emitDate("maturity", h->maturityDate());
    emitDate("latest_relevant", h->latestRelevantDate());
    emitDate("pillar", h->pillarDate());
}

// ---------------------------------------------------------------------------
// Deterministic historical fixings
// ---------------------------------------------------------------------------
//
// The BMA leg of a BMA swap looks back up to a fortnight before its accrual
// start (averagebmacoupon.cpp:115-124 walks fixingStart back until the
// associated value date is <= the accrual start), and the Libor leg fixes
// `fixingDays` business days before its own start, which for settlementDays
// <= 2 lands on or before the evaluation date.  So both indexes need history.
//
// The rate is a pure function of the fixing date's serial number, so the
// Python test can seed exactly the same table without transcribing it:
//     rate(d) = 0.02 + 1e-5 * (d.serialNumber() % 100)
// Only dates STRICTLY BEFORE the evaluation date are seeded, which keeps the
// "fixes today" tie-break (InterestRateIndex::fixing falls back to a forecast
// when today has no stored fixing) out of the picture entirely.
// IndexManager's histories are GLOBAL and keyed by index NAME, so a window
// seeded for one evaluation date survives into the next case and can turn a
// "fixes today, so forecast it" date into a stored historical fixing. Every
// case therefore starts from an empty history; the Python test does the same.
Real seededRate(const Date& d) { return 0.02 + 1.0e-5 * Real(d.serialNumber() % 100); }

void seedFixings(const ext::shared_ptr<InterestRateIndex>& index, const Date& eval,
                 Integer daysBack) {
    for (Date d = eval - daysBack; d < eval; ++d) {
        if (index->isValidFixingDate(d))
            index->addFixing(d, seededRate(d), true);
    }
}

// ---------------------------------------------------------------------------
// Block 1 — BMASwapRateHelper
// ---------------------------------------------------------------------------

const Real kBmaQuote = 0.70;      // quoted Libor fraction
const Rate kBmaCurveRate = 0.020; // curve being bootstrapped (BMA leg forecast)
const Rate kLiborCurveRate = 0.035; // ibor index's own curve (Libor + discounting)

// BMASwapRateHelper keeps swap_ protected and exposes no inspector; deriving
// is the only way to pin the schedules the date arithmetic was read off.
class ProbeBmaHelper : public BMASwapRateHelper {
  public:
    using BMASwapRateHelper::BMASwapRateHelper;
    const ext::shared_ptr<BMASwap>& swap() const { return swap_; }
};

struct BmaCase {
    std::string key;
    Date eval;
    Period tenor;
    Natural settlementDays;
    Calendar calendar;
    Period bmaPeriod;
    BusinessDayConvention bmaConvention;
    DayCounter bmaDayCount;
    Period liborTenor;
};

void emitLeg(const std::string& name, const Leg& leg) {
    std::cout << "    \"" << name << "_size\": " << leg.size() << ",\n";
    auto first = ext::dynamic_pointer_cast<Coupon>(leg.front());
    auto last = ext::dynamic_pointer_cast<Coupon>(leg.back());
    emitDate(name + "_first_accrual_start", first->accrualStartDate());
    emitDate(name + "_first_payment", first->date());
    emitDate(name + "_last_accrual_end", last->accrualEndDate());
    emitDate(name + "_last_payment", last->date());
}

void emitBma(const BmaCase& c) {
    Settings::instance().evaluationDate() = c.eval;
    IndexManager::instance().clearHistories();

    auto liborCurve = ext::make_shared<FlatForward>(c.eval, kLiborCurveRate, Actual365Fixed(),
                                                    Continuous, Annual);
    auto bmaIndex = ext::make_shared<BMAIndex>();
    auto liborIndex =
        ext::make_shared<USDLibor>(c.liborTenor, Handle<YieldTermStructure>(liborCurve));

    seedFixings(bmaIndex, c.eval, 60);
    seedFixings(liborIndex, c.eval, 60);

    auto helper = ext::make_shared<ProbeBmaHelper>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(kBmaQuote)), c.tenor, c.settlementDays,
        c.calendar, c.bmaPeriod, c.bmaConvention, c.bmaDayCount, bmaIndex, liborIndex);

    // The curve being bootstrapped: the BMA leg forecasts off THIS one, while
    // the Libor leg and the discounting both use the ibor index's own curve
    // (ratehelpers.cpp:713-714).  The two rates differ so a mix-up shows.
    auto bootstrapped = ext::make_shared<FlatForward>(c.eval, kBmaCurveRate, Actual365Fixed(),
                                                      Continuous, Annual);
    helper->setTermStructure(bootstrapped.get());

    openObject(c.key);
    emitDate("evaluation", c.eval);
    std::cout << "    \"evaluation_weekday\": " << int(c.eval.weekday()) << ",\n";
    emitHelperDates(helper);
    const auto& swap = helper->swap();
    emitDate("swap_start", swap->startDate());
    emitDate("swap_maturity", swap->maturityDate());
    // The weekday of the adjusted swap maturity selects the branch of the
    // next-Wednesday roll (ratehelpers.cpp:716-720).
    Date adjustedMaturity = c.calendar.adjust(swap->maturityDate(), Following);
    emitDate("adjusted_swap_maturity", adjustedMaturity);
    std::cout << "    \"adjusted_swap_maturity_weekday\": " << int(adjustedMaturity.weekday())
              << ",\n";
    emitLeg("libor_leg", swap->liborLeg());
    emitLeg("bma_leg", swap->bmaLeg());
    std::cout << "    \"swap_npv\": " << swap->NPV() << ",\n"
              << "    \"swap_libor_leg_npv\": " << swap->liborLegNPV() << ",\n"
              << "    \"swap_bma_leg_npv\": " << swap->bmaLegNPV() << ",\n"
              << "    \"implied_quote\": " << helper->impliedQuote() << ",\n"
              << "    \"quote_error\": " << helper->quoteError() << "\n"
              << "  }";
}

void blockBma() {
    const Calendar us = UnitedStates(UnitedStates::GovernmentBond);
    const Calendar uk = UnitedKingdom(UnitedKingdom::Exchange);
    // The test-suite's own choice (piecewiseyieldcurve.cpp:571-573): the joint
    // calendar of the BMA fixing calendar and the Libor fixing calendar.
    const Calendar joint = JointCalendar(BMAIndex().fixingCalendar(),
                                         USDLibor(3 * Months).fixingCalendar(), JoinHolidays);
    const DayCounter aaIsda = ActualActual(ActualActual::ISDA);
    const DayCounter a360 = Actual360();
    const Period q = 3 * Months;

    // Wednesday, a business day everywhere.
    const Date bizDay(17, January, 2024);
    // Saturday / Sunday: jc.adjust rolls the reference date forward.
    const Date saturday(20, January, 2024);
    const Date sunday(30, June, 2024);
    // US Independence Day 2024 (Thursday): US closed, UK open — so the joint
    // calendar and the helper calendar disagree about the reference date.
    const Date usHoliday(4, July, 2024);
    // UK Summer bank holiday 2024 (Monday): UK closed, US open — the mirror
    // case, which a port that rolls on the helper calendar alone gets wrong.
    const Date ukHoliday(26, August, 2024);
    // New Year's Day 2025 (Wednesday): both closed.
    const Date bothClosed(1, January, 2025);
    // Monday 29 April 2024: a business day everywhere, chosen so that
    // 2 settlement days land the swap start on 1 May and the 5Y maturity on
    // TARGET's Labour Day — see the case comment below.
    const Date targetHolidayStart(29, April, 2024);

    const std::vector<BmaCase> cases = {
        // --- tenor sweep on a plain business day -------------------------
        {"bma_1y", bizDay, 1 * Years, 2, joint, q, Following, aaIsda, q},
        {"bma_2y", bizDay, 2 * Years, 2, joint, q, Following, aaIsda, q},
        {"bma_5y", bizDay, 5 * Years, 2, joint, q, Following, aaIsda, q},
        {"bma_10y", bizDay, 10 * Years, 2, joint, q, Following, aaIsda, q},
        // Odd tenors: the backward-generated schedules carry a stub.
        {"bma_18m", bizDay, 18 * Months, 2, joint, q, Following, aaIsda, q},
        {"bma_9m", bizDay, 9 * Months, 2, joint, q, Following, aaIsda, q},
        // --- evaluation dates that are not business days -----------------
        {"bma_5y_saturday", saturday, 5 * Years, 2, joint, q, Following, aaIsda, q},
        {"bma_5y_sunday", sunday, 5 * Years, 2, joint, q, Following, aaIsda, q},
        {"bma_5y_us_holiday", usHoliday, 5 * Years, 2, joint, q, Following, aaIsda, q},
        {"bma_5y_uk_holiday", ukHoliday, 5 * Years, 2, joint, q, Following, aaIsda, q},
        {"bma_5y_both_closed", bothClosed, 5 * Years, 2, joint, q, Following, aaIsda, q},
        // Same holidays with the helper calendar = US only, so the joint-calendar
        // roll and the helper-calendar advance really do use different calendars.
        {"bma_us_cal_5y_us_holiday", usHoliday, 5 * Years, 2, us, q, Following, aaIsda, q},
        {"bma_us_cal_5y_uk_holiday", ukHoliday, 5 * Years, 2, us, q, Following, aaIsda, q},
        {"bma_uk_cal_5y_us_holiday", usHoliday, 5 * Years, 2, uk, q, Following, aaIsda, q},
        {"bma_uk_cal_5y_uk_holiday", ukHoliday, 5 * Years, 2, uk, q, Following, aaIsda, q},
        // A helper calendar that is neither index's: TARGET.  Both schedules are
        // rolled on the INDEXES' calendars, so on these two dates the TARGET
        // helper calendar reproduces the joint-calendar result exactly — which
        // is the invariant, not an accident.
        {"bma_target_cal_5y", bizDay, 5 * Years, 2, TARGET(), q, Following, aaIsda, q},
        {"bma_target_cal_5y_saturday", saturday, 5 * Years, 2, TARGET(), q, Following, aaIsda,
         q},
        // ...and the date where it does NOT.  29 April 2024 + 2 settlement days
        // starts the swap on 1 May 2024, so the 5Y maturity is 1 May 2029 — a
        // business day on both index calendars (so both schedules end there)
        // but TARGET's Labour Day.  calendar_.adjust therefore moves it to
        // Wednesday 2 May, whose weekday (4) takes the `d + (11 - w)` branch and
        // jumps a further full week; the joint-calendar control leaves it on
        // Tuesday 1 May and takes the `d + (4 - w)` branch.  latestDate() comes
        // out a week apart (10 May vs 3 May 2029).
        {"bma_target_cal_5y_target_holiday_maturity", targetHolidayStart, 5 * Years, 2,
         TARGET(), q, Following, aaIsda, q},
        {"bma_joint_cal_5y_target_holiday_maturity", targetHolidayStart, 5 * Years, 2, joint, q,
         Following, aaIsda, q},
        // --- settlement days ---------------------------------------------
        {"bma_5y_settle0", bizDay, 5 * Years, 0, joint, q, Following, aaIsda, q},
        {"bma_5y_settle1", bizDay, 5 * Years, 1, joint, q, Following, aaIsda, q},
        {"bma_5y_settle3", bizDay, 5 * Years, 3, joint, q, Following, aaIsda, q},
        {"bma_5y_settle5", bizDay, 5 * Years, 5, joint, q, Following, aaIsda, q},
        // --- BMA leg knobs ------------------------------------------------
        {"bma_5y_semiannual", bizDay, 5 * Years, 2, joint, 6 * Months, Following, aaIsda, q},
        {"bma_5y_annual", bizDay, 5 * Years, 2, joint, 1 * Years, Following, aaIsda, q},
        {"bma_5y_modfollowing", bizDay, 5 * Years, 2, joint, q, ModifiedFollowing, aaIsda, q},
        {"bma_5y_preceding", bizDay, 5 * Years, 2, joint, q, Preceding, aaIsda, q},
        {"bma_5y_a360_bma_dc", bizDay, 5 * Years, 2, joint, q, Following, a360, q},
        // --- Libor leg tenor ----------------------------------------------
        {"bma_5y_libor1m", bizDay, 5 * Years, 2, joint, q, Following, aaIsda, 1 * Months},
        {"bma_5y_libor6m", bizDay, 5 * Years, 2, joint, q, Following, aaIsda, 6 * Months},
    };

    for (const auto& c : cases)
        emitBma(c);
}

// ---------------------------------------------------------------------------
// Block 2 — CPIBondHelper
// ---------------------------------------------------------------------------

// Canonical UKRPI monthly fixings Jul-2007 .. Sep-2009
// (test-suite/inflationcpibond.cpp CommonVars, first 27 values), continued
// through Sep-2010 so that the later evaluation dates below still have every
// fixing they need.  InflationIndex::needsForecast keys off
// `today - availabilityLag`, NOT off what is stored, so the extra entries do
// not change any case that runs at kCpiToday: those coupons stay forecast.
const Real kRpiFixings[] = {206.1, 207.3, 208.0, 208.9, 209.7, 210.9, 209.8, 211.4,
                            212.1, 214.0, 215.1, 216.8, 216.5, 217.2, 218.4, 217.7,
                            216.0, 212.9, 210.1, 211.4, 211.3, 211.5, 212.8, 213.4,
                            213.4, 213.4, 214.4,
                            // Oct-2009 .. Sep-2010
                            215.3, 216.0, 218.0, 217.9, 219.2, 220.7, 222.8, 223.6,
                            224.1, 223.6, 224.5, 225.3};

const Date kCpiToday(25, November, 2009);

struct CpiMarket {
    ext::shared_ptr<UKRPI> index;
    Handle<YieldTermStructure> nominal;
};

// Built PER CASE, at that case's evaluation date, and seeded only with the
// fixings that date makes unambiguously historical.
//
// ZeroInflationIndex::needsForecast (inflationindex.cpp:197-220) is a
// three-way test against `inflationPeriod(today - availabilityLag, frequency)`:
// a fixing whose period starts BEFORE that window must be provided, one that
// starts AFTER it is forecast *even when a fixing is stored for it*, and only
// inside the window does the stored table decide.  Seeding past the window
// would therefore make each case turn on that asymmetry rather than on
// CPIBondHelper, so the table is cut at the month before the window starts:
// every fixing kept is required, every fixing dropped is forecast on both
// sides of the port.
CpiMarket makeCpiMarket(const Date& eval) {
    CpiMarket m;
    Settings::instance().evaluationDate() = eval;
    IndexManager::instance().clearHistories();
    ActualActual dcISDA(ActualActual::ISDA);

    RelinkableHandle<ZeroInflationTermStructure> hcpi;
    auto ii = ext::make_shared<UKRPI>(hcpi);
    Schedule rpiSchedule =
        MakeSchedule().from(Date(1, July, 2007)).to(Date(1, September, 2010)).withFrequency(
            Monthly);
    // Last day of the month before the availability window.
    Date lastStored =
        inflationPeriod(eval - ii->availabilityLag(), ii->frequency()).first - 1;
    for (Size i = 0; i < std::size(kRpiFixings); ++i) {
        if (rpiSchedule[i] <= lastStored)
            ii->addFixing(rpiSchedule[i], kRpiFixings[i]);
    }

    // Both curves keep kCpiToday as their reference date whatever the case's
    // evaluation date is: the nodes below are literal, so pinning the reference
    // keeps the curve identical across cases and leaves the evaluation date
    // free to move only the BOND's dates — which is what these cases are for.
    m.nominal =
        Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kCpiToday, 0.05, dcISDA));

    // Literal zero-inflation nodes — no bootstrap, so the curve is reproducible
    // node for node.  dates[0] is the curve's base date.
    std::vector<Date> zDates = {Date(1, September, 2009), Date(1, September, 2012),
                                Date(1, September, 2016), Date(1, September, 2021),
                                Date(1, September, 2035), Date(1, September, 2060)};
    std::vector<Rate> zRates = {0.0305, 0.0293, 0.0315, 0.0348, 0.0377, 0.0371};
    hcpi.linkTo(ext::make_shared<InterpolatedZeroInflationCurve<Linear>>(kCpiToday, zDates,
                                                                        zRates, Monthly,
                                                                        dcISDA));
    m.index = ii;
    return m;
}

Schedule cpiSchedule() {
    // 2 October 2027 is a Saturday, so the ModifiedFollowing payment convention
    // used below pushes the final coupon and the redemption to Monday 4 October
    // — which makes latestDate() (the last CASHFLOW date) land strictly after
    // the bond's own maturity date.
    return MakeSchedule()
        .from(Date(2, October, 2007))
        .to(Date(2, October, 2027))
        .withTenor(Period(6, Months))
        .withCalendar(UnitedKingdom())
        .withConvention(Unadjusted)
        .backwards();
}

struct CpiCase {
    std::string key;
    Date eval = kCpiToday;
    Real price = 101.5;
    Natural settlementDays = 3;
    Real faceAmount = 1000000.0;
    Real baseCPI = 206.1;
    Period observationLag = Period(3, Months);
    CPI::InterpolationType interpolation = CPI::Flat;
    std::vector<Rate> coupons = {0.02};
    DayCounter accrualDayCounter = ActualActual(ActualActual::ISDA);
    BusinessDayConvention paymentConvention = ModifiedFollowing;
    Date issueDate = Date(2, October, 2007);
    Bond::Price::Type priceType = Bond::Price::Clean;
};

void emitCpi(const CpiCase& c) {
    // Per-case market: the fixing table depends on the evaluation date (see
    // makeCpiMarket), and IndexManager's histories are global.
    CpiMarket market = makeCpiMarket(c.eval);

    auto helper = ext::make_shared<CPIBondHelper>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(c.price)), c.settlementDays, c.faceAmount,
        c.baseCPI, c.observationLag, market.index, c.interpolation, cpiSchedule(), c.coupons,
        c.accrualDayCounter, c.paymentConvention, c.issueDate, UnitedKingdom(), Period(),
        Calendar(), Unadjusted, false, c.priceType);

    helper->setTermStructure(market.nominal.currentLink().get());

    openObject(c.key);
    emitDate("evaluation", c.eval);
    emitHelperDates(helper);
    const auto& bond = helper->bond();
    emitDate("bond_maturity", bond->maturityDate());
    emitDate("bond_settlement", bond->settlementDate());
    emitDate("bond_next_cashflow", bond->nextCashFlowDate());
    emitDate("bond_last_cashflow", bond->cashflows().back()->date());
    std::cout << "    \"n_cashflows\": " << bond->cashflows().size() << ",\n"
              << "    \"price_type\": " << int(helper->priceType()) << ",\n"
              << "    \"clean_price\": " << bond->cleanPrice() << ",\n"
              << "    \"dirty_price\": " << bond->dirtyPrice() << ",\n"
              << "    \"accrued_amount\": " << bond->accruedAmount() << ",\n"
              << "    \"implied_quote\": " << helper->impliedQuote() << ",\n"
              << "    \"quote_error\": " << helper->quoteError() << "\n"
              << "  }";
}

void blockCpi() {
    emitCpi({"cpi_clean"});

    CpiCase dirty{"cpi_dirty"};
    dirty.priceType = Bond::Price::Dirty;
    emitCpi(dirty);

    // earliestDate_ is bond->nextCashFlowDate(), so it steps to the next
    // coupon as the evaluation date crosses an accrual boundary.  2 April 2010
    // is a UK holiday (Good Friday) and the settlement roll is what decides
    // which side of the boundary settlement lands on.
    CpiCase later{"cpi_eval_2010_03_31"};
    later.eval = Date(31, March, 2010);
    emitCpi(later);

    CpiCase later2{"cpi_eval_2010_04_06"};
    later2.eval = Date(6, April, 2010);
    emitCpi(later2);

    CpiCase settle0{"cpi_settle0"};
    settle0.settlementDays = 0;
    emitCpi(settle0);

    CpiCase settle7{"cpi_settle7"};
    settle7.settlementDays = 7;
    emitCpi(settle7);

    CpiCase interp{"cpi_interp_linear"};
    interp.interpolation = CPI::Linear;
    emitCpi(interp);

    CpiCase lag{"cpi_lag_8m"};
    lag.observationLag = Period(8, Months);
    emitCpi(lag);

    CpiCase base{"cpi_base_cpi_210"};
    base.baseCPI = 210.0;
    emitCpi(base);

    CpiCase face{"cpi_face_2_5m"};
    face.faceAmount = 2500000.0;
    emitCpi(face);

    CpiCase coupons{"cpi_coupon_vector"};
    coupons.coupons = {0.03, 0.025, 0.02};
    emitCpi(coupons);

    CpiCase dc{"cpi_daycount_a365f"};
    dc.accrualDayCounter = Actual365Fixed();
    emitCpi(dc);

    CpiCase conv{"cpi_payment_preceding"};
    conv.paymentConvention = Preceding;
    emitCpi(conv);

    CpiCase quote{"cpi_quote_98"};
    quote.price = 98.0;
    emitCpi(quote);
}

// ---------------------------------------------------------------------------
// Block 3 — MultipleResetsSwapRateHelper
// ---------------------------------------------------------------------------

const Date kMrsEval(15, June, 2026);

// swap_ is protected here too.
class ProbeMrsHelper : public MultipleResetsSwapRateHelper {
  public:
    using MultipleResetsSwapRateHelper::MultipleResetsSwapRateHelper;
    const ext::shared_ptr<MultipleResetsSwap>& swap() const { return swap_; }
};

struct MrsCase {
    std::string key;
    Period tenor = 1 * Years;
    Natural settlementDays = 2;
    Period indexTenor = 3 * Months;
    Size resetsPerCoupon = 2;
    bool withDiscountCurve = false;
    RateAveraging::Type averaging = RateAveraging::Compound;
    Spread spread = 0.0;
    Frequency fixedFrequency = NoFrequency;
    DayCounter fixedDayCount = DayCounter();
    BusinessDayConvention fixedConvention = ModifiedFollowing;
    Rate quote = 0.025;
};

void emitMrs(const MrsCase& c) {
    Settings::instance().evaluationDate() = kMrsEval;
    IndexManager::instance().clearHistories();

    auto forecast = ext::make_shared<FlatForward>(kMrsEval, 0.030, Actual365Fixed(), Continuous,
                                                  Annual);
    auto discount = ext::make_shared<FlatForward>(kMrsEval, 0.025, Actual365Fixed(), Continuous,
                                                  Annual);
    // The helper clones the index onto its own (initially empty) handle, so the
    // curve attached here is only a placeholder; what matters is that a curve
    // exists for MakeMultipleResetsSwap's own bookkeeping.
    auto index = ext::make_shared<Euribor>(c.indexTenor, Handle<YieldTermStructure>(forecast));
    // settlementDays == 0 starts the swap on the evaluation date, so its first
    // reset fixes two business days earlier and must come from history.
    seedFixings(index, kMrsEval, 30);

    auto helper = ext::make_shared<ProbeMrsHelper>(
        c.settlementDays, c.tenor, Handle<Quote>(ext::make_shared<SimpleQuote>(c.quote)), index,
        c.resetsPerCoupon,
        c.withDiscountCurve ? Handle<YieldTermStructure>(discount)
                            : Handle<YieldTermStructure>(),
        c.averaging, c.spread, c.fixedFrequency, c.fixedDayCount, c.fixedConvention);

    // The curve being bootstrapped: the cloned index forecasts off it, and (when
    // no exogenous discount curve was supplied) it discounts as well
    // (multipleresetsswaphelper.cpp:74-83).
    auto bootstrapped = ext::make_shared<FlatForward>(kMrsEval, 0.028, Actual365Fixed(),
                                                      Continuous, Annual);
    helper->setTermStructure(bootstrapped.get());

    openObject(c.key);
    emitDate("evaluation", kMrsEval);
    emitHelperDates(helper);
    const auto& swap = helper->swap();
    emitDate("swap_start", swap->startDate());
    emitDate("swap_maturity", swap->maturityDate());
    emitDate("fixed_leg_last_payment", swap->fixedLeg().back()->date());
    emitDate("floating_leg_last_payment", swap->floatingLeg().back()->date());
    std::cout << "    \"n_fixed_coupons\": " << swap->fixedLeg().size() << ",\n"
              << "    \"n_floating_coupons\": " << swap->floatingLeg().size() << ",\n"
              << "    \"resets_per_coupon\": " << swap->resetsPerCoupon() << ",\n"
              << "    \"averaging_method\": " << int(swap->averagingMethod()) << ",\n"
              << "    \"fixed_rate\": " << swap->fixedRate() << ",\n"
              << "    \"spread\": " << swap->spread() << ",\n"
              << "    \"swap_npv\": " << swap->NPV() << ",\n"
              << "    \"fixed_leg_bps\": " << swap->fixedLegBPS() << ",\n"
              << "    \"floating_leg_npv\": " << swap->floatingLegNPV() << ",\n"
              << "    \"implied_quote\": " << helper->impliedQuote() << ",\n"
              << "    \"quote_error\": " << helper->quoteError() << "\n"
              << "  }";
}

void blockMrs() {
    const std::vector<MrsCase> cases = {
        {"mrs_base"},
        // Tenor sweep.  2Y and 3Y off the 17-Jun-2026 spot end on a Saturday
        // and a Sunday respectively; MakeMultipleResetsSwap adjusts the end
        // date BEFORE generating the reset schedule, so the backward roll is
        // seeded one day away from the start and leaves 9 / 13 reset periods —
        // not a multiple of resetsPerCoupon, which MultipleResetsSwap rejects.
        // 4Y / 5Y / 10Y end on a Monday, a Tuesday and a Wednesday.
        {"mrs_4y", 4 * Years},
        {"mrs_5y", 5 * Years},
        {"mrs_10y", 10 * Years},
        // resetsPerCoupon.
        {"mrs_resets4", 1 * Years, 2, 3 * Months, 4},
        {"mrs_5y_resets4", 5 * Years, 2, 3 * Months, 4},
        {"mrs_1m_index_resets3", 1 * Years, 2, 1 * Months, 3},
        {"mrs_6m_index_resets2", 5 * Years, 2, 6 * Months, 2},
        // Settlement days.
        {"mrs_settle0", 1 * Years, 0},
        {"mrs_settle5", 1 * Years, 5},
        // Exogenous discount curve (vs discounting off the bootstrapped curve).
        {"mrs_discount_curve", 1 * Years, 2, 3 * Months, 2, true},
        {"mrs_5y_discount_curve", 5 * Years, 2, 3 * Months, 2, true},
        // Averaging method.
        {"mrs_simple", 1 * Years, 2, 3 * Months, 2, false, RateAveraging::Simple},
        {"mrs_5y_simple", 5 * Years, 2, 3 * Months, 2, false, RateAveraging::Simple},
        // Spread.
        {"mrs_spread", 1 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound, 0.0025},
        {"mrs_negative_spread", 1 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound,
         -0.0015},
        // Fixed-leg overrides.
        {"mrs_fixed_annual", 5 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound, 0.0,
         Annual},
        {"mrs_fixed_daycount", 1 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound, 0.0,
         NoFrequency, Actual365Fixed()},
        {"mrs_fixed_thirty360", 5 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound,
         0.0, NoFrequency, Thirty360(Thirty360::BondBasis)},
        {"mrs_fixed_following", 5 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound,
         0.0, NoFrequency, DayCounter(), Following},
        {"mrs_fixed_preceding", 5 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound,
         0.0, NoFrequency, DayCounter(), Preceding},
        // Quote value only moves quoteError, never the dates or implied quote.
        {"mrs_quote_04", 1 * Years, 2, 3 * Months, 2, false, RateAveraging::Compound, 0.0,
         NoFrequency, DayCounter(), ModifiedFollowing, 0.04},
    };

    for (const auto& c : cases)
        emitMrs(c);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    blockBma();
    blockCpi();
    blockMrs();

    std::cout << "\n}\n";
    return 0;
}
