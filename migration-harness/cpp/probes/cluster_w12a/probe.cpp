// Phase 11 W12-A cluster probe: CMS coupons + GFunction/Hagan (Conundrum)
// replication pricers + LognormalCmsSpreadPricer.
//
// FINAL-wave core-cashflows gap-fill — the CMS-coupon-pricer family that
// W8-A deferred (CmsCoupon / CmsCouponPricer / conundrum replication pricers)
// is genuinely missing from PQuantLib. This probe captures the canonical
// `cms.cpp` conundrum reference values.
//
//   * GFunctionStandard value/firstDerivative/secondDerivative at known x
//     (TIGHT — closed form: x / (1+x/q)^delta / (1 - 1/(1+x/q)^n)).
//
//   * ConundrumIntegrand value at a strike (TIGHT — option * F''(x), with the
//     vanilla-option pricer driven off a constant lognormal swaption vol).
//
//   * AnalyticHaganPricer convexity-adjusted CMS coupon rate at a flat 5%
//     curve + constant lognormal swaption vol, for the Standard / ExactYield
//     / ParallelShifts / NonParallelShifts yield-curve models (LOOSE —
//     static replication). This is the canonical cms/conundrum test setup.
//
//   * NumericHaganPricer vs AnalyticHaganPricer coupon-rate agreement
//     (LOOSE — the cms.cpp `testFairRate` 2e-4 equivalence).
//
//   * LognormalCmsSpreadPricer CMS-spread coupon rate at a known correlation
//     (LOOSE — Brigo-Mercurio 13.6.2 bivariate-lognormal replication).
//
// C++ parity:
//   ql/cashflows/cmscoupon.hpp + .cpp
//   ql/cashflows/conundrumpricer.hpp + .cpp
//   ql/cashflows/couponpricer.hpp (CmsCouponPricer base)
//   ql/experimental/coupons/lognormalcmsspreadpricer.hpp + .cpp
//   ql/experimental/coupons/cmsspreadcoupon.hpp (CmsSpreadCoupon)
//   @ v1.42.1 (099987f0).

#include <ql/cashflows/cashflowvectors.hpp>
#include <ql/cashflows/cmscoupon.hpp>
#include <ql/cashflows/conundrumpricer.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/experimental/coupons/cmsspreadcoupon.hpp>
#include <ql/experimental/coupons/lognormalcmsspreadpricer.hpp>
#include <ql/experimental/coupons/swapspreadindex.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

const Date today(15, January, 2024);
const Actual365Fixed dcA365;
const Actual360 dc360;
const TARGET cal;

Handle<YieldTermStructure> flatCurve(Real r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(today, r, dcA365));
}

// A 10Y EuriborSwapIsdaFixA-style swap index on Euribor6M (as in cms.cpp).
ext::shared_ptr<SwapIndex> makeSwapIndex(const Period& tenor,
                                         const ext::shared_ptr<IborIndex>& ibor) {
    return ext::make_shared<SwapIndex>(
        "EuriborSwapIsdaFixA", tenor, ibor->fixingDays(), ibor->currency(),
        ibor->fixingCalendar(), Period(1, Years), Unadjusted,
        ibor->dayCounter(), ibor);
}

// Constant lognormal swaption vol (16%, the cms.cpp ATM-ish level).
Handle<SwaptionVolatilityStructure> constLogVol(Real v) {
    return Handle<SwaptionVolatilityStructure>(
        ext::make_shared<ConstantSwaptionVolatility>(
            today, cal, Following, v, dcA365, ShiftedLognormal));
}

// ---------------------------------------------------------------------
// GFunctionStandard closed form
// ---------------------------------------------------------------------
void block_gfunction_standard() {
    // q = 1 (annual fixed leg), delta = 0.5 (pay halfway), swapLength = 10.
    auto g = GFunctionFactory::newGFunctionStandard(1, 0.5, 10);
    Real x = 0.05;
    emit("gstd_value", (*g)(x));
    emit("gstd_first", g->firstDerivative(x));
    emit("gstd_second", g->secondDerivative(x));

    // a second sample point
    auto g2 = GFunctionFactory::newGFunctionStandard(2, 0.25, 5);
    Real x2 = 0.04;
    emit("gstd2_value", (*g2)(x2));
    emit("gstd2_first", g2->firstDerivative(x2));
    emit("gstd2_second", g2->secondDerivative(x2));
}

// ---------------------------------------------------------------------
// ConundrumIntegrand value at a strike
// ---------------------------------------------------------------------
void block_conundrum_integrand() {
    Handle<YieldTermStructure> curve = flatCurve(0.05);
    auto ibor = ext::make_shared<Euribor6M>(curve);
    auto swapIndex = makeSwapIndex(Period(10, Years), ibor);

    Date startDate = curve->referenceDate() + 20 * Years;
    Date paymentDate = startDate + 1 * Years;
    Date endDate = paymentDate;

    CmsCoupon coupon(paymentDate, 1.0, startDate, endDate,
                     swapIndex->fixingDays(), swapIndex, 1.0, 0.0,
                     startDate, endDate, ibor->dayCounter());

    Real forwardValue = swapIndex->fixing(coupon.fixingDate());
    auto g = GFunctionFactory::newGFunctionStandard(1, 0.5, 10);

    auto vanillaPricer = ext::shared_ptr<VanillaOptionPricer>(
        new MarketQuotedOptionPricer(forwardValue, coupon.fixingDate(),
                                     swapIndex->tenor(), *constLogVol(0.16)));

    Real strike = 0.04;
    Real annuity = 4.5;  // arbitrary deterministic annuity for the integrand probe
    NumericHaganPricer::ConundrumIntegrand integrand(
        vanillaPricer, nullptr, g, coupon.fixingDate(), paymentDate, annuity,
        forwardValue, strike, Option::Call);

    Real xEval = 0.06;
    emit("ci_forward", forwardValue);
    emit("ci_value", integrand(xEval));
}

// ---------------------------------------------------------------------
// AnalyticHaganPricer + NumericHaganPricer coupon rate (canonical cms test)
// ---------------------------------------------------------------------
void block_hagan_coupon_rate() {
    Handle<YieldTermStructure> curve = flatCurve(0.05);
    auto ibor = ext::make_shared<Euribor6M>(curve);
    auto swapIndex = makeSwapIndex(Period(10, Years), ibor);
    auto vol = constLogVol(0.16);
    Handle<Quote> zeroMeanRev(ext::make_shared<SimpleQuote>(0.0));

    Date startDate = curve->referenceDate() + 20 * Years;
    Date paymentDate = startDate + 1 * Years;
    Date endDate = paymentDate;

    CmsCoupon coupon(paymentDate, 1.0, startDate, endDate,
                     swapIndex->fixingDays(), swapIndex, 1.0, 0.0,
                     startDate, endDate, ibor->dayCounter());

    emit("hagan_forward", swapIndex->fixing(coupon.fixingDate()));

    GFunctionFactory::YieldCurveModel models[4] = {
        GFunctionFactory::Standard, GFunctionFactory::ExactYield,
        GFunctionFactory::ParallelShifts, GFunctionFactory::NonParallelShifts};
    const char* anNames[4] = {"hagan_an_standard", "hagan_an_exactyield",
                              "hagan_an_parallel", "hagan_an_nonparallel"};
    const char* numNames[4] = {"hagan_num_standard", "hagan_num_exactyield",
                               "hagan_num_parallel", "hagan_num_nonparallel"};

    for (int j = 0; j < 4; ++j) {
        auto an = ext::make_shared<AnalyticHaganPricer>(vol, models[j],
                                                        zeroMeanRev);
        coupon.setPricer(an);
        emit(anNames[j], coupon.rate());

        auto num = ext::make_shared<NumericHaganPricer>(vol, models[j],
                                                        zeroMeanRev);
        coupon.setPricer(num);
        emit(numNames[j], coupon.rate());
    }
}

// ---------------------------------------------------------------------
// LognormalCmsSpreadPricer CMS-spread coupon rate
// ---------------------------------------------------------------------
void block_cms_spread() {
    Handle<YieldTermStructure> curve = flatCurve(0.05);
    auto ibor = ext::make_shared<Euribor6M>(curve);
    auto s10 = makeSwapIndex(Period(10, Years), ibor);
    auto s2 = makeSwapIndex(Period(2, Years), ibor);
    auto vol = constLogVol(0.16);
    Handle<Quote> zeroMeanRev(ext::make_shared<SimpleQuote>(0.0));
    Handle<Quote> corr(ext::make_shared<SimpleQuote>(0.5));

    auto ssi = ext::make_shared<SwapSpreadIndex>("CMS10Y-2Y", s10, s2, 1.0, -1.0);

    Date startDate = curve->referenceDate() + 20 * Years;
    Date paymentDate = startDate + 1 * Years;
    Date endDate = paymentDate;

    CmsSpreadCoupon coupon(paymentDate, 1.0, startDate, endDate,
                           ssi->fixingDays(), ssi, 1.0, 0.0,
                           startDate, endDate, ibor->dayCounter());

    auto cmsPricer = ext::make_shared<AnalyticHaganPricer>(
        vol, GFunctionFactory::Standard, zeroMeanRev);
    auto pricer = ext::make_shared<LognormalCmsSpreadPricer>(
        cmsPricer, corr, curve, 16);

    coupon.setPricer(pricer);

    emit("ssp_fix1_10y", s10->fixing(coupon.fixingDate()));
    emit("ssp_fix2_2y", s2->fixing(coupon.fixingDate()));
    emit("ssp_rate", coupon.rate());

    // also at zero correlation
    Handle<Quote> corr0(ext::make_shared<SimpleQuote>(0.0));
    auto pricer0 = ext::make_shared<LognormalCmsSpreadPricer>(
        cmsPricer, corr0, curve, 16);
    coupon.setPricer(pricer0);
    emit("ssp_rate_corr0", coupon.rate(), false);
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = today;

    std::cout << "{\n";
    block_gfunction_standard();
    block_conundrum_integrand();
    block_hagan_coupon_rate();
    block_cms_spread();
    std::cout << "}\n";
    return 0;
}
