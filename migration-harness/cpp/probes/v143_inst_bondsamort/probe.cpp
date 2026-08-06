// migration-harness/cpp/probes/v143_inst_bondsamort/probe.cpp
//
// Reference values for the four coupon-driven bond concretes that PQuantLib
// was still missing:
//
//   * CmsRateBond                 (ql/instruments/bonds/cmsratebond.{hpp,cpp})
//   * AmortizingCmsRateBond       (ql/instruments/bonds/amortizingcmsratebond.{hpp,cpp})
//   * AmortizingFloatingRateBond  (ql/instruments/bonds/amortizingfloatingratebond.{hpp,cpp})
//   * CPIBond                     (ql/instruments/bonds/cpibond.{hpp,cpp})
//
// WHAT IS PINNED AND WHY
// ----------------------
// Each of these classes is a *pass-through*: the constructor forwards a dozen
// optional arguments into a leg builder (CmsLeg / IborLeg / CPILeg) and then
// calls addRedemptionsToCashflows (or calculateNotionalsFromCashflows). The
// characteristic defect is an argument that is accepted, stored and never
// forwarded — the bond still prices, it just silently prices a *different*
// bond (most damagingly: a dropped amortisation vector prices as a bullet).
//
// An NPV alone cannot catch that: two errors cancel, or the difference hides
// in the fourth decimal. So for every bond this probe emits the COMPLETE
// cashflow listing — for every flow the payment-date serial and the amount,
// and for every coupon additionally the nominal, the accrual start/end
// serials, the accrual period, the rate, the fixing-date serial (floating
// coupons) and the ex-coupon-date serial. Plus NPV / cleanPrice / dirtyPrice
// / accruedAmount / settlementDate / startDate / maturityDate / yield and
// notional(d) sampled at five dates spanning the amortisation schedule.
//
// Every bond family is emitted as a BASE case plus one variant per optional
// constructor argument, each varying exactly ONE argument away from its
// default. That is what makes a dropped argument fail: the variant's
// cashflow listing differs from the base's in a specific, pinned way.
//
// The amortisation sweep is deliberately three-way: a sinking-fund schedule
// (sinkingNotionals, French amortisation), a hand-written per-period notional
// vector, and a single constant notional (bullet). The per-coupon nominal is
// pinned in the listing, so a dropped notional vector cannot hide.
//
// DETERMINISM NOTES
// -----------------
// * Historic index fixings are seeded over every TARGET business day in
//   [2020-01-01, today] from a serial-number formula, so a coupon that reads
//   the WRONG fixing date reads a different rate (a constant history would
//   make a wrong fixing date invisible).
// * The capped/floored IBOR variants use a 20% flat optionlet vol; C++'s
//   BlackIborCouponPricer::adjustedFixing short-circuits for non-in-arrears
//   coupons (Black76 timing adjustment), so no convexity adjustment is
//   involved and the port reproduces it exactly.
// * The in-arrears IBOR variant uses a ZERO optionlet vol on purpose: C++
//   applies an in-arrears convexity adjustment there and PQuantLib's
//   BlackIborCouponPricer documents that adjustment as a carve-out. At zero
//   variance the adjustment is identically zero, so the variant isolates the
//   thing under test (the fixing date moving to the accrual end) instead of
//   re-testing a known, documented pricer carve-out.
// * The CMS bonds are priced with AnalyticHaganPricer (the pricer the C++
//   test suite attaches to CmsRateBond in assetswap.cpp) over a constant
//   lognormal swaption vol.
// * The CPI block runs on its own evaluation date (25-Nov-2009) with the
//   canonical UKRPI fixing table, and uses a ZeroInflationCurve built from
//   literal (date, zero-rate) nodes rather than a bootstrap, so the curve is
//   reproducible node-for-node on both sides.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/bondsamort.json.

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/conundrumpricer.hpp>
#include <ql/cashflows/coupon.hpp>
#include <ql/cashflows/cpicoupon.hpp>
#include <ql/cashflows/floatingratecoupon.hpp>
#include <ql/cashflows/inflationcoupon.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/inflation/ukrpi.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/instruments/bonds/amortizingcmsratebond.hpp>
#include <ql/instruments/bonds/amortizingfixedratebond.hpp>
#include <ql/instruments/bonds/amortizingfloatingratebond.hpp>
#include <ql/instruments/bonds/cmsratebond.hpp>
#include <ql/instruments/bonds/cpibond.hpp>
#include <ql/pricingengines/bond/discountingbondengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/inflation/interpolatedzeroinflationcurve.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <ql/qldefines.hpp>

#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------
// JSON emission helpers (hand-rolled; the harness has no JSON dependency)
// ---------------------------------------------------------------------

int gIndent = 1;

void pad() {
    for (int i = 0; i < gIndent; ++i)
        std::cout << "  ";
}

void key(const std::string& k) {
    pad();
    std::cout << "\"" << k << "\": ";
}

void emitNum(const std::string& k, Real v, bool comma = true) {
    key(k);
    std::cout << std::setprecision(17) << v << (comma ? "," : "") << "\n";
}

void emitInt(const std::string& k, long v, bool comma = true) {
    key(k);
    std::cout << v << (comma ? "," : "") << "\n";
}

void emitBool(const std::string& k, bool v, bool comma = true) {
    key(k);
    std::cout << (v ? "true" : "false") << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------------------
// Shared market data — nominal / CMS block
// ---------------------------------------------------------------------

const Date kToday(15, January, 2024);
const Actual365Fixed dcA365;
const Actual360 dc360;
const Thirty360 dc30360(Thirty360::BondBasis);
const TARGET calTARGET;
const NullCalendar calNull;

// Bond schedules. 22-Aug rolls onto a weekend in 2020/2021/2026/2027, which
// is what makes Following vs Preceding an observable difference.
const Date kCmsStart(22, August, 2020);
const Date kCmsEnd(22, August, 2030);
const Date kFrnStart(22, August, 2020);
const Date kFrnEnd(22, August, 2025);

// Deterministic historic-fixing formula: serial-dependent so a coupon that
// reads the wrong fixing date reads a different rate.
Real historicIbor(const Date& d) { return 0.0250 + 0.00001 * Real(d.serialNumber() % 50); }
Real historicSwap(const Date& d) { return 0.0300 + 0.00001 * Real(d.serialNumber() % 50); }

struct CmsMarket {
    ext::shared_ptr<IborIndex> ibor;
    ext::shared_ptr<SwapIndex> swapIndex;
    Handle<YieldTermStructure> discountCurve;
    ext::shared_ptr<PricingEngine> bondEngine;
    ext::shared_ptr<FloatingRateCouponPricer> cmsPricer;
    ext::shared_ptr<FloatingRateCouponPricer> iborPricerVol;   // 20% flat
    ext::shared_ptr<FloatingRateCouponPricer> iborPricerZero;  // 0% flat
};

CmsMarket makeMarket() {
    CmsMarket m;
    Handle<YieldTermStructure> forecast(ext::make_shared<FlatForward>(kToday, 0.03, dcA365));
    m.discountCurve =
        Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, 0.04, dcA365));
    m.ibor = ext::make_shared<Euribor6M>(forecast);
    m.swapIndex = ext::make_shared<SwapIndex>("EuriborSwapIsdaFixA", Period(10, Years),
                                              m.ibor->fixingDays(), EURCurrency(),
                                              m.ibor->fixingCalendar(), Period(1, Years),
                                              Unadjusted, m.ibor->dayCounter(), m.ibor);

    // Seed the fixing histories over every TARGET business day up to today.
    for (Date d(1, January, 2020); d <= kToday; ++d) {
        if (!calTARGET.isBusinessDay(d))
            continue;
        m.ibor->addFixing(d, historicIbor(d));
        m.swapIndex->addFixing(d, historicSwap(d));
    }

    m.bondEngine = ext::make_shared<DiscountingBondEngine>(m.discountCurve);

    Handle<SwaptionVolatilityStructure> swaptionVol(ext::make_shared<ConstantSwaptionVolatility>(
        kToday, calTARGET, Following, 0.16, dcA365, ShiftedLognormal));
    m.cmsPricer = ext::make_shared<AnalyticHaganPricer>(
        swaptionVol, GFunctionFactory::Standard,
        Handle<Quote>(ext::make_shared<SimpleQuote>(0.0)));

    Handle<OptionletVolatilityStructure> ovs20(ext::make_shared<ConstantOptionletVolatility>(
        kToday, calTARGET, Following, 0.20, dcA365, ShiftedLognormal));
    Handle<OptionletVolatilityStructure> ovs0(ext::make_shared<ConstantOptionletVolatility>(
        kToday, calTARGET, Following, 0.0, dcA365, ShiftedLognormal));
    m.iborPricerVol = ext::make_shared<BlackIborCouponPricer>(ovs20);
    m.iborPricerZero = ext::make_shared<BlackIborCouponPricer>(ovs0);
    return m;
}

Schedule cmsSchedule() {
    return {kCmsStart, kCmsEnd, Period(Annual), calTARGET, Unadjusted, Unadjusted,
            DateGeneration::Backward, false};
}

Schedule frnSchedule() {
    return {kFrnStart, kFrnEnd, Period(Semiannual), calTARGET, Unadjusted, Unadjusted,
            DateGeneration::Backward, false};
}

// ---------------------------------------------------------------------
// The generic bond dump
// ---------------------------------------------------------------------

// notional(d) sample dates — chosen to straddle the amortisation steps.
// Reset per block (the CPI bond lives on a different calendar decade).
std::vector<Date> gNotionalSamples = {Date(1, January, 2021), Date(22, August, 2022),
                                      Date(22, August, 2025), Date(22, August, 2028),
                                      Date(1, January, 2031)};

// A date before every issue date used here, so settlementDate(d) exercises
// the issue-date clamp in Bond::settlementDate.
const Date kEarlySettleProbe(3, January, 2006);

void emitCashflows(const Bond& bond) {
    key("cashflows");
    std::cout << "[\n";
    ++gIndent;
    const Leg& leg = bond.cashflows();
    for (Size i = 0; i < leg.size(); ++i) {
        const ext::shared_ptr<CashFlow>& cf = leg[i];
        pad();
        std::cout << "{\n";
        ++gIndent;
        emitInt("date_serial", static_cast<long>(cf->date().serialNumber()));
        emitNum("amount", cf->amount());
        Date ex = cf->exCouponDate();
        emitInt("ex_coupon_serial", ex == Date() ? 0 : static_cast<long>(ex.serialNumber()));
        auto cpn = ext::dynamic_pointer_cast<Coupon>(cf);
        if (cpn == nullptr) {
            emitBool("is_coupon", false, false);
        } else {
            emitBool("is_coupon", true);
            emitNum("nominal", cpn->nominal());
            emitInt("accrual_start_serial",
                    static_cast<long>(cpn->accrualStartDate().serialNumber()));
            emitInt("accrual_end_serial",
                    static_cast<long>(cpn->accrualEndDate().serialNumber()));
            emitNum("accrual_period", cpn->accrualPeriod());
            // Both floating and inflation coupons carry a fixing date; it is
            // the structural fingerprint of fixingDays / inArrears (floating)
            // and of the observation lag (inflation).
            auto flt = ext::dynamic_pointer_cast<FloatingRateCoupon>(cf);
            auto inf = ext::dynamic_pointer_cast<InflationCoupon>(cf);
            long fixingSerial = 0;
            if (flt != nullptr)
                fixingSerial = static_cast<long>(flt->fixingDate().serialNumber());
            else if (inf != nullptr)
                fixingSerial = static_cast<long>(inf->fixingDate().serialNumber());
            emitInt("fixing_serial", fixingSerial);
            emitNum("rate", cpn->rate(), false);
        }
        --gIndent;
        pad();
        std::cout << "}" << (i + 1 < leg.size() ? "," : "") << "\n";
    }
    --gIndent;
    pad();
    std::cout << "],\n";
}

using ExtraEmitter = std::function<void()>;

void emitBond(const std::string& name, const Bond& bond, bool comma = true,
              const ExtraEmitter& extra = ExtraEmitter()) {
    key(name);
    std::cout << "{\n";
    ++gIndent;

    if (extra)
        extra();
    emitInt("settlement_serial", static_cast<long>(bond.settlementDate().serialNumber()));
    emitInt("settlement_before_issue_serial",
            static_cast<long>(bond.settlementDate(kEarlySettleProbe).serialNumber()));
    emitInt("start_serial", static_cast<long>(bond.startDate().serialNumber()));
    emitInt("maturity_serial", static_cast<long>(bond.maturityDate().serialNumber()));
    emitInt("issue_serial",
            bond.issueDate() == Date() ? 0 : static_cast<long>(bond.issueDate().serialNumber()));
    emitInt("n_cashflows", static_cast<long>(bond.cashflows().size()));
    emitInt("n_redemptions", static_cast<long>(bond.redemptions().size()));

    emitNum("npv", bond.NPV());
    emitNum("clean_price", bond.cleanPrice());
    emitNum("dirty_price", bond.dirtyPrice());
    emitNum("accrued", bond.accruedAmount());
    emitNum("settlement_value", bond.settlementValue());

    try {
        Real y = bond.yield(dcA365, Compounded, Annual);
        emitNum("yield", y);
    } catch (std::exception&) {
        key("yield");
        std::cout << "null,\n";
    }

    // Bond::notionals() — the amortisation step vector itself.
    key("notionals");
    std::cout << "[";
    const std::vector<Real>& ns = bond.notionals();
    for (Size i = 0; i < ns.size(); ++i)
        std::cout << std::setprecision(17) << ns[i] << (i + 1 < ns.size() ? ", " : "");
    std::cout << "],\n";

    key("notional_at");
    std::cout << "[";
    for (Size i = 0; i < gNotionalSamples.size(); ++i) {
        std::cout << "{\"date_serial\": " << gNotionalSamples[i].serialNumber()
                  << ", \"value\": " << std::setprecision(17)
                  << bond.notional(gNotionalSamples[i]) << "}"
                  << (i + 1 < gNotionalSamples.size() ? ", " : "");
    }
    std::cout << "],\n";

    emitCashflows(bond);
    emitBool("is_tradable", bond.isTradable(), false);

    --gIndent;
    pad();
    std::cout << "}" << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------------------
// CmsRateBond variants
// ---------------------------------------------------------------------

struct CmsSpec {
    Natural settlementDays = 2;
    Real faceAmount = 100.0;
    BusinessDayConvention paymentConvention = Following;
    Natural fixingDays = Null<Natural>();
    std::vector<Real> gearings = {1.0};
    std::vector<Spread> spreads = {0.0};
    std::vector<Rate> caps = {};
    std::vector<Rate> floors = {};
    bool inArrears = false;
    Real redemption = 100.0;
    Date issueDate = Date();
};

void emitCmsRateBond(const std::string& name, const CmsSpec& s, const CmsMarket& m) {
    CmsRateBond bond(s.settlementDays, s.faceAmount, cmsSchedule(), m.swapIndex, dc30360,
                     s.paymentConvention, s.fixingDays, s.gearings, s.spreads, s.caps,
                     s.floors, s.inArrears, s.redemption, s.issueDate);
    bond.setPricingEngine(m.bondEngine);
    setCouponPricer(bond.cashflows(), m.cmsPricer);
    emitBond(name, bond);
}

void blockCmsRateBond(const CmsMarket& m) {
    emitCmsRateBond("cms_base", CmsSpec(), m);

    { CmsSpec s; s.paymentConvention = Preceding;      emitCmsRateBond("cms_payment_convention", s, m); }
    { CmsSpec s; s.fixingDays = 10;                    emitCmsRateBond("cms_fixing_days", s, m); }
    { CmsSpec s; s.gearings = {0.8};                   emitCmsRateBond("cms_gearings", s, m); }
    { CmsSpec s; s.gearings = {0.8, 0.9, 1.1};         emitCmsRateBond("cms_gearings_vector", s, m); }
    { CmsSpec s; s.spreads = {0.002};                  emitCmsRateBond("cms_spreads", s, m); }
    { CmsSpec s; s.spreads = {0.001, 0.002, 0.003};    emitCmsRateBond("cms_spreads_vector", s, m); }
    { CmsSpec s; s.caps = {0.035};                     emitCmsRateBond("cms_caps", s, m); }
    { CmsSpec s; s.floors = {0.045};                   emitCmsRateBond("cms_floors", s, m); }
    { CmsSpec s; s.caps = {0.055}; s.floors = {0.025}; emitCmsRateBond("cms_collar", s, m); }
    { CmsSpec s; s.inArrears = true;                   emitCmsRateBond("cms_in_arrears", s, m); }
    { CmsSpec s; s.redemption = 102.0;                 emitCmsRateBond("cms_redemption", s, m); }
    { CmsSpec s; s.issueDate = kCmsStart;              emitCmsRateBond("cms_issue_date", s, m); }
    { CmsSpec s; s.settlementDays = 5;                 emitCmsRateBond("cms_settlement_days", s, m); }
    { CmsSpec s; s.faceAmount = 250.0;                 emitCmsRateBond("cms_face_amount", s, m); }
    // A zero gearing degenerates every coupon to a FixedRateCoupon paying the
    // cap/floor-clamped spread (cashflowvectors.hpp:133-141) — the one place
    // where gearings/spreads/caps/floors interact.
    { CmsSpec s; s.gearings = {0.0}; s.spreads = {0.04}; s.caps = {0.035};
      emitCmsRateBond("cms_zero_gearing", s, m); }
}

// ---------------------------------------------------------------------
// AmortizingCmsRateBond variants
// ---------------------------------------------------------------------

// The three amortisation shapes swept for every amortising bond.
std::vector<Real> sinkingVector() {
    // sinkingNotionals returns nPeriods+1 entries whose last is 0.0; the leg
    // consumes indices [0, n) only, and FloatingLeg rejects more than n
    // nominals, so the trailing 0.0 is dropped (it is never read).
    std::vector<Real> v = sinkingNotionals(Period(10, Years), Annual, 0.05, 100.0);
    v.pop_back();
    return v;
}

const std::vector<Real> kCustomNotionals10 = {100.0, 95.0, 88.0, 80.0, 71.0,
                                              61.0,  50.0, 38.0, 25.0, 11.0};
const std::vector<Real> kCustomNotionals10Frn = {100.0, 96.0, 91.0, 85.0, 78.0,
                                                 70.0,  61.0, 51.0, 40.0, 28.0};

struct AmCmsSpec {
    Natural settlementDays = 2;
    std::vector<Real> notionals = {100.0};
    BusinessDayConvention paymentConvention = Following;
    Natural fixingDays = Null<Natural>();
    std::vector<Real> gearings = {1.0};
    std::vector<Spread> spreads = {0.0};
    std::vector<Rate> caps = {};
    std::vector<Rate> floors = {};
    bool inArrears = false;
    Date issueDate = Date();
    std::vector<Real> redemptions = {100.0};
};

void emitAmCmsBond(const std::string& name, const AmCmsSpec& s, const CmsMarket& m) {
    AmortizingCmsRateBond bond(s.settlementDays, s.notionals, cmsSchedule(), m.swapIndex,
                               dc30360, s.paymentConvention, s.fixingDays, s.gearings,
                               s.spreads, s.caps, s.floors, s.inArrears, s.issueDate,
                               s.redemptions);
    bond.setPricingEngine(m.bondEngine);
    setCouponPricer(bond.cashflows(), m.cmsPricer);
    emitBond(name, bond);
}

void blockAmortizingCmsRateBond(const CmsMarket& m) {
    emitAmCmsBond("amcms_constant", AmCmsSpec(), m);

    { AmCmsSpec s; s.notionals = sinkingVector();          emitAmCmsBond("amcms_sinking", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10;       emitAmCmsBond("amcms_custom", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10;
      s.redemptions = {100.0, 101.0, 102.0};               emitAmCmsBond("amcms_redemptions", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10;
      s.paymentConvention = Preceding;                     emitAmCmsBond("amcms_payment_convention", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.fixingDays = 10;
                                                           emitAmCmsBond("amcms_fixing_days", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.gearings = {0.8};
                                                           emitAmCmsBond("amcms_gearings", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.spreads = {0.002};
                                                           emitAmCmsBond("amcms_spreads", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.caps = {0.035};
                                                           emitAmCmsBond("amcms_caps", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.floors = {0.045};
                                                           emitAmCmsBond("amcms_floors", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.inArrears = true;
                                                           emitAmCmsBond("amcms_in_arrears", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.issueDate = kCmsStart;
                                                           emitAmCmsBond("amcms_issue_date", s, m); }
    { AmCmsSpec s; s.notionals = kCustomNotionals10; s.settlementDays = 5;
                                                           emitAmCmsBond("amcms_settlement_days", s, m); }
}

// ---------------------------------------------------------------------
// AmortizingFloatingRateBond variants
// ---------------------------------------------------------------------

struct AmFrnSpec {
    Natural settlementDays = 2;
    std::vector<Real> notionals = {100.0};
    BusinessDayConvention paymentConvention = Following;
    Natural fixingDays = Null<Natural>();
    std::vector<Real> gearings = {1.0};
    std::vector<Spread> spreads = {0.0};
    std::vector<Rate> caps = {};
    std::vector<Rate> floors = {};
    bool inArrears = false;
    Date issueDate = Date();
    Period exCouponPeriod = Period();
    Calendar exCouponCalendar = Calendar();
    BusinessDayConvention exCouponConvention = Unadjusted;
    bool exCouponEndOfMonth = false;
    std::vector<Real> redemptions = {100.0};
    Integer paymentLag = 0;
    // 0 = leave the default pricer IborLeg attaches; 1 = 20% vol; 2 = 0% vol.
    int pricer = 0;
};

void emitAmFrnBond(const std::string& name, const AmFrnSpec& s, const CmsMarket& m) {
    AmortizingFloatingRateBond bond(s.settlementDays, s.notionals, frnSchedule(), m.ibor,
                                    dc360, s.paymentConvention, s.fixingDays, s.gearings,
                                    s.spreads, s.caps, s.floors, s.inArrears, s.issueDate,
                                    s.exCouponPeriod, s.exCouponCalendar,
                                    s.exCouponConvention, s.exCouponEndOfMonth,
                                    s.redemptions, s.paymentLag);
    bond.setPricingEngine(m.bondEngine);
    if (s.pricer == 1)
        setCouponPricer(bond.cashflows(), m.iborPricerVol);
    else if (s.pricer == 2)
        setCouponPricer(bond.cashflows(), m.iborPricerZero);
    emitBond(name, bond);
}

void blockAmortizingFloatingRateBond(const CmsMarket& m) {
    emitAmFrnBond("amfrn_constant", AmFrnSpec(), m);

    { AmFrnSpec s; s.notionals = sinkingVector();       emitAmFrnBond("amfrn_sinking", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; emitAmFrnBond("amfrn_custom", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn;
      s.redemptions = {100.0, 101.0, 102.0};            emitAmFrnBond("amfrn_redemptions", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn;
      s.paymentConvention = Preceding;                  emitAmFrnBond("amfrn_payment_convention", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.fixingDays = 10;
                                                        emitAmFrnBond("amfrn_fixing_days", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.gearings = {0.8};
                                                        emitAmFrnBond("amfrn_gearings", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.spreads = {0.002};
                                                        emitAmFrnBond("amfrn_spreads", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.caps = {0.026}; s.pricer = 1;
                                                        emitAmFrnBond("amfrn_caps", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.floors = {0.032}; s.pricer = 1;
                                                        emitAmFrnBond("amfrn_floors", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.inArrears = true; s.pricer = 2;
                                                        emitAmFrnBond("amfrn_in_arrears", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.issueDate = kFrnStart;
                                                        emitAmFrnBond("amfrn_issue_date", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.settlementDays = 5;
                                                        emitAmFrnBond("amfrn_settlement_days", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.paymentLag = 3;
                                                        emitAmFrnBond("amfrn_payment_lag", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn;
      s.exCouponPeriod = Period(5, Days); s.exCouponCalendar = calTARGET;
      s.exCouponConvention = Preceding;                 emitAmFrnBond("amfrn_ex_coupon", s, m); }
    // end-of-month + a different ex-coupon calendar/convention: the ex-date
    // rolls off a 1M offset with eom=true, which is only visible if all four
    // ex-coupon arguments are forwarded.
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn;
      s.exCouponPeriod = Period(1, Months); s.exCouponCalendar = calNull;
      s.exCouponConvention = Following; s.exCouponEndOfMonth = true;
                                                        emitAmFrnBond("amfrn_ex_coupon_eom", s, m); }
    { AmFrnSpec s; s.notionals = kCustomNotionals10Frn; s.gearings = {0.0}; s.spreads = {0.04};
      s.floors = {0.045};                               emitAmFrnBond("amfrn_zero_gearing", s, m); }
}

// ---------------------------------------------------------------------
// CPIBond — its own evaluation date + the canonical UKRPI fixing table
// ---------------------------------------------------------------------

const Date kCpiToday(25, November, 2009);

// Canonical UKRPI monthly fixings, Jul-2007 .. Sep-2009
// (test-suite/inflationcpibond.cpp CommonVars).
const Real kRpiFixings[] = {206.1, 207.3, 208.0, 208.9, 209.7, 210.9, 209.8, 211.4, 212.1,
                            214.0, 215.1, 216.8, 216.5, 217.2, 218.4, 217.7, 216.0, 212.9,
                            210.1, 211.4, 211.3, 211.5, 212.8, 213.4, 213.4, 213.4, 214.4};

struct CpiMarket {
    ext::shared_ptr<ZeroInflationIndex> index;
    Handle<YieldTermStructure> nominal;
    ext::shared_ptr<PricingEngine> bondEngine;
};

CpiMarket makeCpiMarket() {
    CpiMarket m;
    Settings::instance().evaluationDate() = kCpiToday;
    UnitedKingdom cal;
    ActualActual dcISDA(ActualActual::ISDA);

    RelinkableHandle<ZeroInflationTermStructure> hcpi;
    auto ii = ext::make_shared<UKRPI>(hcpi);
    Schedule rpiSchedule = MakeSchedule()
                               .from(Date(1, July, 2007))
                               .to(Date(1, September, 2009))
                               .withFrequency(Monthly);
    for (Size i = 0; i < std::size(kRpiFixings); ++i)
        ii->addFixing(rpiSchedule[i], kRpiFixings[i]);

    m.nominal = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kCpiToday, 0.05, dcISDA));

    // Literal zero-inflation nodes — no bootstrap, so the curve is
    // reproducible node-for-node. dates[0] is the curve's base date.
    std::vector<Date> zDates = {Date(1, September, 2009), Date(1, September, 2012),
                                Date(1, September, 2016), Date(1, September, 2021),
                                Date(1, September, 2035), Date(1, September, 2060)};
    std::vector<Rate> zRates = {0.0305, 0.0293, 0.0315, 0.0348, 0.0377, 0.0371};
    auto cpiTS = ext::make_shared<InterpolatedZeroInflationCurve<Linear>>(
        kCpiToday, zDates, zRates, Monthly, dcISDA);
    hcpi.linkTo(cpiTS);

    m.index = ii;
    m.bondEngine = ext::make_shared<DiscountingBondEngine>(m.nominal);
    return m;
}

Schedule cpiSchedule() {
    return MakeSchedule()
        .from(Date(2, October, 2007))
        .to(Date(2, October, 2017))
        .withTenor(Period(6, Months))
        .withCalendar(UnitedKingdom())
        .withConvention(Unadjusted)
        .backwards();
}

struct CpiSpec {
    Natural settlementDays = 3;
    Real faceAmount = 1000000.0;
    bool growthOnly = false;
    Real baseCPI = 206.1;
    Period observationLag = Period(3, Months);
    CPI::InterpolationType observationInterpolation = CPI::Flat;
    std::vector<Rate> coupons = {0.1};
    BusinessDayConvention paymentConvention = ModifiedFollowing;
    Date issueDate = Date();
    Calendar paymentCalendar = Calendar();
    Period exCouponPeriod = Period();
    Calendar exCouponCalendar = Calendar();
    BusinessDayConvention exCouponConvention = Unadjusted;
    bool exCouponEndOfMonth = false;
};

QL_DEPRECATED_DISABLE_WARNING

void emitCpiBond(const std::string& name, const CpiSpec& s, const CpiMarket& m,
                 bool comma = true) {
    // The growthOnly overload is the deprecated one; using it uniformly keeps
    // one construction path, and growthOnly=false reproduces the current
    // (non-deprecated) constructor exactly (cpibond.cpp:38-60).
    CPIBond bond(s.settlementDays, s.faceAmount, s.growthOnly, s.baseCPI, s.observationLag,
                 m.index, s.observationInterpolation, cpiSchedule(), s.coupons,
                 Actual365Fixed(), s.paymentConvention, s.issueDate, s.paymentCalendar,
                 s.exCouponPeriod, s.exCouponCalendar, s.exCouponConvention,
                 s.exCouponEndOfMonth);
    bond.setPricingEngine(m.bondEngine);
    emitBond(name, bond, comma, [&bond]() {
        emitInt("frequency", static_cast<long>(bond.frequency()));
        emitBool("growth_only", bond.growthOnly());
        emitNum("base_cpi", bond.baseCPI());
        emitInt("observation_lag_length", static_cast<long>(bond.observationLag().length()));
        emitInt("observation_lag_units", static_cast<long>(bond.observationLag().units()));
        emitInt("observation_interpolation",
                static_cast<long>(bond.observationInterpolation()));
        emitBool("calendar_is_uk", UnitedKingdom() == bond.calendar());
        emitBool("calendar_is_null", NullCalendar() == bond.calendar());
    });
}

QL_DEPRECATED_ENABLE_WARNING

void blockCpiBond(const CpiMarket& m) {
    gNotionalSamples = {Date(1, January, 2008), Date(2, April, 2010), Date(2, October, 2013),
                        Date(2, October, 2017), Date(1, January, 2020)};
    emitCpiBond("cpi_base", CpiSpec(), m);

    { CpiSpec s; s.observationInterpolation = CPI::Linear;
                                             emitCpiBond("cpi_interp_linear", s, m); }
    { CpiSpec s; s.paymentConvention = Preceding;
                                             emitCpiBond("cpi_payment_convention", s, m); }
    { CpiSpec s; s.paymentCalendar = calNull; emitCpiBond("cpi_payment_calendar", s, m); }
    { CpiSpec s; s.issueDate = Date(2, October, 2007);
                                             emitCpiBond("cpi_issue_date", s, m); }
    { CpiSpec s; s.growthOnly = true;        emitCpiBond("cpi_growth_only", s, m); }
    { CpiSpec s; s.baseCPI = 210.0;          emitCpiBond("cpi_base_cpi", s, m); }
    { CpiSpec s; s.observationLag = Period(8, Months);
                                             emitCpiBond("cpi_observation_lag", s, m); }
    { CpiSpec s; s.coupons = {0.10, 0.08, 0.06};
                                             emitCpiBond("cpi_coupon_vector", s, m); }
    { CpiSpec s; s.settlementDays = 7;       emitCpiBond("cpi_settlement_days", s, m); }
    { CpiSpec s; s.faceAmount = 2500000.0;   emitCpiBond("cpi_face_amount", s, m); }
    { CpiSpec s; s.exCouponPeriod = Period(10, Days); s.exCouponCalendar = UnitedKingdom();
      s.exCouponConvention = Preceding;      emitCpiBond("cpi_ex_coupon", s, m); }
    { CpiSpec s; s.exCouponPeriod = Period(1, Months); s.exCouponCalendar = calNull;
      s.exCouponConvention = Following; s.exCouponEndOfMonth = true;
                                             emitCpiBond("cpi_ex_coupon_eom", s, m, false); }
}

// ---------------------------------------------------------------------
// sinkingNotionals / sinkingSchedule reference block
// ---------------------------------------------------------------------

void blockSinking() {
    key("sinking");
    std::cout << "{\n";
    ++gIndent;
    struct Case { const char* name; Frequency f; Rate r; };
    const Case cases[] = {{"annual_5pc", Annual, 0.05},
                          {"annual_0pc", Annual, 0.0},
                          {"semi_3pc", Semiannual, 0.03},
                          {"quarterly_7pc", Quarterly, 0.07}};
    for (Size c = 0; c < std::size(cases); ++c) {
        std::vector<Real> v = sinkingNotionals(Period(10, Years), cases[c].f, cases[c].r, 100.0);
        key(cases[c].name);
        std::cout << "[";
        for (Size i = 0; i < v.size(); ++i)
            std::cout << std::setprecision(17) << v[i] << (i + 1 < v.size() ? ", " : "");
        std::cout << "],\n";
    }
    Schedule sch = sinkingSchedule(Date(15, January, 2024), Period(5, Years), Annual, calTARGET);
    key("schedule_5y_annual_serials");
    std::cout << "[";
    for (Size i = 0; i < sch.size(); ++i)
        std::cout << sch[i].serialNumber() << (i + 1 < sch.size() ? ", " : "");
    std::cout << "]\n";
    --gIndent;
    pad();
    std::cout << "},\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    Settings::instance().evaluationDate() = kToday;
    CmsMarket market = makeMarket();

    emitInt("today_serial", static_cast<long>(kToday.serialNumber()));
    emitInt("cpi_today_serial", static_cast<long>(kCpiToday.serialNumber()));

    blockSinking();
    blockCmsRateBond(market);
    blockAmortizingCmsRateBond(market);
    blockAmortizingFloatingRateBond(market);

    CpiMarket cpiMarket = makeCpiMarket();
    blockCpiBond(cpiMarket);

    std::cout << "}\n";
    return 0;
}
