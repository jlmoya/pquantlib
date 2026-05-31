// Phase 11 W12-C cluster probe: core ql/cashflows gap-fill —
//   dividends + average-BMA coupon + equity (quanto) cashflow +
//   overnight-indexed coupon pricers (compounding + arithmetic average).
//
//   * Dividend family (FixedDividend / FractionalDividend): amount() and
//     amount(underlying). EXACT (fixed) / TIGHT (rate * underlying).
//   * AverageBMACoupon rate via the built-in AverageBMACouponPricer over a
//     fully-known BMA fixing history (weekly Wednesday fixings). The forward
//     part never runs (all fixings are past), so the result is the
//     calendar-day-weighted average of the BMA fixings. LOOSE.
//   * EquityQuantoCashFlowPricer quanto adjustment at known flat vols + flat
//     curves + correlation. price() = I1/I0 - 1 (growth-only), where the
//     quanto re-cloned index forecasts the fixing with the quanto-adjusted
//     dividend curve. LOOSE.
//   * CompoundingOvernightIndexedCouponPricer and
//     ArithmeticAveragedOvernightIndexedCouponPricer coupon rate over a fully
//     known overnight fixing history (deterministic compounding / averaging,
//     no forward curve required). TIGHT.
//
// C++ parity:
//   ql/cashflows/dividend.{hpp,cpp}
//   ql/cashflows/averagebmacoupon.{hpp,cpp} + ql/indexes/bmaindex.{hpp,cpp}
//   ql/cashflows/equitycashflow.{hpp,cpp} + ql/indexes/equityindex.{hpp,cpp}
//   ql/cashflows/overnightindexedcouponpricer.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/cashflows/dividend.hpp>
#include <ql/cashflows/averagebmacoupon.hpp>
#include <ql/cashflows/equitycashflow.hpp>
#include <ql/cashflows/overnightindexedcoupon.hpp>
#include <ql/cashflows/overnightindexedcouponpricer.hpp>
#include <ql/indexes/bmaindex.hpp>
#include <ql/indexes/equityindex.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/currencies/america.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/settings.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

bool g_first = true;

void emit(const char* name, Real v) {
    if (!g_first) std::cout << ",\n";
    g_first = false;
    std::cout << "  \"" << name << "\": " << v;
}

// ---------------------------------------------------------------------------
// (a) Dividends
// ---------------------------------------------------------------------------
void block_dividends() {
    Date d(15, June, 2024);

    FixedDividend fixed(2.5, d);
    emit("fixed_amount", fixed.amount());
    emit("fixed_amount_with_underlying", fixed.amount(100.0));

    FractionalDividend frac(0.03, 200.0, d);
    emit("frac_rate", frac.rate());
    emit("frac_nominal", frac.nominal());
    emit("frac_amount", frac.amount());                 // rate * nominal
    emit("frac_amount_underlying", frac.amount(150.0)); // rate * underlying

    // DividendVector helper: builds FixedDividends.
    std::vector<Date> dates = {Date(15, March, 2024), Date(15, June, 2024)};
    std::vector<Real> amts = {1.0, 2.0};
    auto vec = DividendVector(dates, amts);
    emit("divvec_size", (Real)vec.size());
    emit("divvec_0_amount", vec[0]->amount());
    emit("divvec_1_amount", vec[1]->amount());
}

// ---------------------------------------------------------------------------
// (b) Average BMA coupon
// ---------------------------------------------------------------------------
void block_bma() {
    // Evaluation date well after the coupon window so that the entire fixing
    // schedule (which extends to the next Wednesday past the window end) is
    // historical and neither the pricer's forward part nor indexFixings()
    // triggers a forecast.
    Date today(1, March, 2024);
    Settings::instance().evaluationDate() = today;

    auto index = ext::make_shared<BMAIndex>();
    Calendar cal = index->fixingCalendar();

    // Coupon over a one-month-ish window with a fully-known fixing history.
    Date start(2, January, 2024);
    Date end(1, February, 2024);
    Date payment = cal.adjust(end, Following);

    // Seed BMA fixings for every valid (weekly-Wednesday) fixing date in the
    // window plus the lead-in fixings the pricer needs. We seed a generous
    // range of Wednesdays around the window with a deterministic ramp.
    // BMA is fixed weekly on Wednesdays; isValidFixingDate handles holidays.
    for (Date d(1, December, 2023); d <= Date(15, February, 2024); ++d) {
        if (index->isValidFixingDate(d)) {
            // deterministic ramp: 0.01 + 0.0001 * (serial mod 50)
            Real f = 0.01 + 0.0001 * (Real)((d.serialNumber()) % 50);
            index->addFixing(d, f);
        }
    }

    AverageBMACoupon coupon(payment, 1000000.0, start, end, index);

    emit("bma_rate", coupon.rate());
    emit("bma_amount", coupon.amount());
    emit("bma_accrual_period", coupon.accrualPeriod());
    auto fixings = coupon.indexFixings();
    emit("bma_n_fixings", (Real)fixings.size());
    emit("bma_first_fixing", fixings.front());
    emit("bma_last_fixing", fixings.back());

    Settings::instance().evaluationDate() = Date();
}

// ---------------------------------------------------------------------------
// (c) Equity quanto cashflow
// ---------------------------------------------------------------------------
void block_equity_quanto() {
    Date today(2, January, 2024);
    Settings::instance().evaluationDate() = today;
    DayCounter dc = Actual365Fixed();
    Calendar cal = TARGET();

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> interest(
        ext::make_shared<FlatForward>(today, 0.03, dc));
    Handle<YieldTermStructure> dividend(
        ext::make_shared<FlatForward>(today, 0.01, dc));

    auto equityIndex = ext::make_shared<EquityIndex>(
        "eqIndex", cal, USDCurrency(), interest, dividend, spot);

    Date baseDate = today;
    Date fixingDate(2, January, 2025);
    Date paymentDate(6, January, 2025);

    // plain (no pricer) equity cashflow growth amount
    EquityCashFlow plain(1.0, equityIndex, baseDate, fixingDate, paymentDate);
    emit("equity_plain_amount", plain.amount());          // I1/I0 - 1
    emit("equity_base_fixing", equityIndex->fixing(baseDate));
    emit("equity_fwd_fixing", equityIndex->fixing(fixingDate));

    // quanto pricer
    Handle<YieldTermStructure> quantoCcyTs(
        ext::make_shared<FlatForward>(today, 0.02, dc));
    Handle<BlackVolTermStructure> equityVol(
        ext::make_shared<BlackConstantVol>(today, cal, 0.20, dc));
    Handle<BlackVolTermStructure> fxVol(
        ext::make_shared<BlackConstantVol>(today, cal, 0.10, dc));
    Handle<Quote> correlation(ext::make_shared<SimpleQuote>(0.4));

    auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(
        quantoCcyTs, equityVol, fxVol, correlation);

    EquityCashFlow cf(1.0, equityIndex, baseDate, fixingDate, paymentDate);
    cf.setPricer(pricer);
    emit("equity_quanto_amount", cf.amount());

    // growthOnly = false variant
    EquityCashFlow cf2(1.0, equityIndex, baseDate, fixingDate, paymentDate, false);
    cf2.setPricer(pricer);
    emit("equity_quanto_amount_total", cf2.amount());

    Settings::instance().evaluationDate() = Date();
}

// ---------------------------------------------------------------------------
// (d) Overnight-indexed coupon pricers (compounding + arithmetic average)
// ---------------------------------------------------------------------------
void block_overnight_pricers() {
    Date today(1, March, 2024);
    Settings::instance().evaluationDate() = today;

    auto index = ext::make_shared<Sofr>();
    Calendar cal = index->fixingCalendar();

    Date start(1, February, 2024);
    Date end(1, March, 2024);
    Date payment = cal.adjust(end, Following);

    // --- Compounding pricer ------------------------------------------------
    OvernightIndexedCoupon couponC(payment, 1000000.0, start, end, index);

    // Seed every fixing date in the coupon's window as a past fixing
    // (all < today == evaluationDate), so the deterministic compute() path runs.
    {
        auto fd = couponC.fixingDates();
        for (Size i = 0; i < fd.size(); ++i) {
            Real f = 0.05 + 0.0001 * (Real)i; // deterministic ramp
            index->addFixing(fd[i], f, true);
        }
    }

    couponC.setPricer(ext::make_shared<CompoundingOvernightIndexedCouponPricer>());
    emit("on_compound_rate", couponC.rate());
    emit("on_compound_amount", couponC.amount());
    emit("on_compound_accrual_period", couponC.accrualPeriod());
    emit("on_n_fixings", (Real)couponC.fixingDates().size());

    // --- Arithmetic average pricer (same coupon window + fixings) ----------
    OvernightIndexedCoupon couponA(payment, 1000000.0, start, end, index,
                                   1.0, 0.0, Date(), Date(), DayCounter(),
                                   RateAveraging::Simple);
    couponA.setPricer(
        ext::make_shared<ArithmeticAveragedOvernightIndexedCouponPricer>());
    emit("on_arith_rate", couponA.rate());
    emit("on_arith_amount", couponA.amount());

    // --- With gearing + spread (compounding) -------------------------------
    OvernightIndexedCoupon couponGS(payment, 1000000.0, start, end, index,
                                    2.0, 0.001);
    couponGS.setPricer(ext::make_shared<CompoundingOvernightIndexedCouponPricer>());
    emit("on_compound_gs_rate", couponGS.rate()); // 2*avg + 0.001

    Settings::instance().evaluationDate() = Date();
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_dividends();
    block_bma();
    block_equity_quanto();
    block_overnight_pricers();

    std::cout << "\n}\n";
    return 0;
}
