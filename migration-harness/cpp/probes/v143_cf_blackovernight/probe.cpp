// migration-harness/cpp/probes/v143_cf_blackovernight/probe.cpp
//
// Reference values for the two Black pricers for capped / floored overnight
// coupons (ql/cashflows/blackovernightindexedcouponpricer.{hpp,cpp}):
//
//   * BlackCompoundingOvernightIndexedCouponPricer  (compounded ON coupon)
//   * BlackAveragingOvernightIndexedCouponPricer    (simple-average ON coupon)
//
// What is pinned and why
// ----------------------
// Both pricers have two completely different optionlet formulas selected by
// the `dailyCapFloor` flag, and inside the "global" one a further branch on
// the optionlet volatility TYPE:
//
//   optionletRateGlobal(): a single option on the whole-period rate.
//     - fixing already known (fixingDate <= evaluationDate) -> intrinsic value
//       gearing * max(fwd - K, 0);
//     - otherwise Black-76 when the vol is ShiftedLognormal, Bachelier when it
//       is Normal, on a standard deviation that is NOT vol*sqrt(t): unless
//       effectiveVolatilityInput is set, the vol is damped by the
//       Lyashenko/Mercurio backward-looking-rate correction
//       T = Ts + (Te-T)^3 / (Te-Ts)^2 / 3          (cpp:169-176).
//   optionletRateLocal(): daily cap/floor. Prices ONE cap/floor in the middle
//     of the remaining period, folds it into an average daily rate, recompounds
//     (compounding pricer) or re-accumulates (averaging pricer) and returns the
//     DIFFERENCE against the uncapped rate (cpp:198-343 / 441-586).
//
// A matching caplet price can therefore hide two cancelling errors in the
// standard-deviation chain, so the inputs to that chain are pinned separately:
// the fixing-date serials, the fixing start/end times off the vol surface, the
// damped time T, sigma, the resulting stdDev, and the forward the option is
// written on (effectiveIndexFixing for the compounding pricer, forwardRate for
// the averaging one). Alongside them the full coupon listing (nominal, accrual
// serials, accrual period, n fixings) is emitted, because the swaplet rate that
// the optionlet is added to comes from the same coupon.
//
// Every optional argument is exercised with a non-default value:
//   - the vol handle:            supplied (default is an empty handle, whose
//                                only observable behaviour is the missing-vol
//                                QL_REQUIRE, pinned as vol_missing_raises);
//   - effectiveVolatilityInput:  true (changes stdDev to plain vol*sqrt(t) and
//                                makes optionletRateLocal throw);
//   - dailyCapFloor:             true (selects optionletRateLocal);
//   - coupon gearing / spread:   2.0 / 0.001, so a dropped `gearing_ *` or a
//                                dropped spread in the effective strike shows;
//   - vol displacement:          0.01, so a dropped shift shows;
//   - nakedOption:               true.
//
// Scenarios:
//   fwd_*   evaluation date 15-Jan-2024, coupon 1-Apr-2024..1-Jul-2024 on a
//           flat 3% curve -> every fixing is projected, Black/Bachelier branch.
//   past_*  evaluation date 1-Mar-2024, coupon 1-Feb-2024..1-Mar-2024 with a
//           complete SOFR fixing history -> intrinsic branch.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/blackovernight.json.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/cashflows/blackovernightindexedcouponpricer.hpp>
#include <ql/cashflows/overnightindexedcoupon.hpp>
#include <ql/errors.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/math/comparison.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------- emitters

int gIndent = 2;

void pad() { std::cout << std::string(gIndent, ' '); }

// Null<Real>() is emitted as JSON null rather than as its 3.4e38 sentinel, so
// the Python side compares against None instead of a magic number.
void emitReal(const std::string& key, Real v, bool comma) {
    pad();
    std::cout << "\"" << key << "\": ";
    if (v == Null<Real>())
        std::cout << "null";
    else
        std::cout << v;
    std::cout << (comma ? "," : "") << "\n";
}

void emitInt(const std::string& key, long v, bool comma) {
    pad();
    std::cout << "\"" << key << "\": " << v << (comma ? "," : "") << "\n";
}

void emitBool(const std::string& key, bool v, bool comma) {
    pad();
    std::cout << "\"" << key << "\": " << (v ? "true" : "false") << (comma ? "," : "") << "\n";
}

void open(const std::string& key) {
    pad();
    std::cout << "\"" << key << "\": {\n";
    gIndent += 2;
}

void close(bool comma) {
    gIndent -= 2;
    pad();
    std::cout << "}" << (comma ? "," : "") << "\n";
}

// Run `f`, require that it throws, and emit {"raises": true}. Aborts the probe
// if it does NOT throw, so the reference can never claim a failure that the
// library does not actually produce.
template <class F>
void emitRaises(const std::string& key, F f, bool comma) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "probe error: expected '" << key << "' to throw, it did not\n";
        std::exit(1);
    }
    pad();
    std::cout << "\"" << key << "\": {\"raises\": true}" << (comma ? "," : "") << "\n";
}

// ------------------------------------------------------------------ market

const Date kFwdToday(15, January, 2024);
const Date kFwdStart(1, April, 2024);   // Monday, SOFR business day
const Date kFwdEnd(1, July, 2024);      // Monday, SOFR business day
const Real kNominal = 1000000.0;
const Real kGearing = 2.0;    // non-default
const Real kSpread = 0.001;   // non-default
const Real kCap = 0.075;
const Real kFloor = 0.045;
const Real kFlatRate = 0.03;
const Real kLnVol = 0.20;
const Real kDisplacement = 0.01; // non-default
const Real kNormalVol = 0.0080;

const Date kPastToday(1, March, 2024);
const Date kPastStart(1, February, 2024);
const Date kPastEnd(1, March, 2024);

Handle<YieldTermStructure> flatCurve() {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(
        kFwdToday, Handle<Quote>(ext::make_shared<SimpleQuote>(kFlatRate)), Actual365Fixed()));
}

// Fixed reference date + NullCalendar so that the Python side can reproduce the
// surface without depending on a holiday list; only the day counter feeds
// timeFromReference, which is all these pricers use.
Handle<OptionletVolatilityStructure> constVol(Volatility v, VolatilityType type, Real shift) {
    return Handle<OptionletVolatilityStructure>(ext::make_shared<ConstantOptionletVolatility>(
        kFwdToday, NullCalendar(), Following, v, Actual365Fixed(), type, shift));
}

ext::shared_ptr<OvernightIndexedCoupon> makeCoupon(const ext::shared_ptr<OvernightIndex>& index,
                                                   const Date& start,
                                                   const Date& end,
                                                   Real gearing,
                                                   Spread spread,
                                                   RateAveraging::Type averaging) {
    return ext::make_shared<OvernightIndexedCoupon>(end, kNominal, start, end, index, gearing,
                                                    spread, Date(), Date(), DayCounter(), false,
                                                    averaging);
}

// The coupon geometry the optionlet is written on. Emitted for every scenario
// because both the swaplet rate and the effective strike are functions of it.
void emitCouponShape(const std::string& key,
                     const ext::shared_ptr<OvernightIndexedCoupon>& c,
                     bool comma) {
    open(key);
    emitReal("nominal", c->nominal(), true);
    emitInt("accrual_start_serial", c->accrualStartDate().serialNumber(), true);
    emitInt("accrual_end_serial", c->accrualEndDate().serialNumber(), true);
    emitInt("payment_serial", c->date().serialNumber(), true);
    emitReal("accrual_period", c->accrualPeriod(), true);
    emitReal("gearing", c->gearing(), true);
    emitReal("spread", c->spread(), true);
    emitInt("n_fixings", static_cast<long>(c->fixingDates().size()), true);
    emitInt("first_fixing_serial", c->fixingDates().front().serialNumber(), true);
    emitInt("last_fixing_serial", c->fixingDates().back().serialNumber(), true);
    emitInt("fixing_date_serial", c->fixingDate().serialNumber(), true);
    emitInt("first_value_date_serial", c->valueDates().front().serialNumber(), true);
    emitInt("last_value_date_serial", c->valueDates().back().serialNumber(), true);
    emitReal("first_dt", c->dt().front(), true);
    emitReal("last_dt", c->dt().back(), false);
    close(comma);
}

// The full standard-deviation chain of optionletRateGlobal (cpp:155-177),
// recomputed here from the public API so that a Python port reproducing only
// the final price cannot hide a compensating pair of errors.
void emitStdDevChain(const ext::shared_ptr<OvernightIndexedCoupon>& c,
                     const Handle<OptionletVolatilityStructure>& vol,
                     Real effStrike,
                     bool effectiveVolatilityInput,
                     bool comma) {
    const std::vector<Date>& fd = c->fixingDates();
    const Real fixingStartTime = vol->timeFromReference(fd.front());
    const Real fixingEndTime = vol->timeFromReference(fd.back());
    const Real effectiveTime = fixingEndTime;
    Real sigma, T, stdDev;
    if (effectiveVolatilityInput) {
        sigma = vol->volatility(fd.back(), effStrike);
        T = effectiveTime;
        stdDev = sigma * std::sqrt(effectiveTime);
    } else {
        sigma = vol->volatility(std::max(fd.front(), vol->referenceDate() + 1), effStrike);
        T = std::max(fixingStartTime, 0.0);
        if (!close_enough(fixingEndTime, T))
            T += std::pow(fixingEndTime - T, 3.0) / std::pow(fixingEndTime - fixingStartTime, 2.0) /
                 3.0;
        stdDev = sigma * std::sqrt(T);
    }
    open("std_dev_chain");
    emitReal("eff_strike", effStrike, true);
    emitReal("fixing_start_time", fixingStartTime, true);
    emitReal("fixing_end_time", fixingEndTime, true);
    emitReal("effective_time", effectiveTime, true);
    emitReal("sigma", sigma, true);
    emitReal("damped_time", T, true);
    emitReal("std_dev", stdDev, false);
    close(comma);
}

// ------------------------------------------------------- forward scenarios

// One (pricer x vol type) block: swaplet rate, the forward the option sees,
// the global and local caplet/floorlet rates, and the effective vols the
// pricer records as a side effect of each call.
// NOTE the template parameter: OvernightIndexedCouponPricer PRIVATISES the
// one-argument capletRate/floorletRate it inherits from FloatingRateCouponPricer
// (overnightindexedcouponpricer.hpp:52-53), so they are only reachable through
// the concrete Black pricer type, which re-declares them public.
template <class P>
void emitPricerBlock(const std::string& key,
                     const ext::shared_ptr<P>& pricer,
                     const ext::shared_ptr<OvernightIndexedCoupon>& coupon,
                     const Handle<OptionletVolatilityStructure>& vol,
                     Real effCap,
                     Real effFloor,
                     bool effectiveVolatilityInput,
                     bool emitLocal,
                     bool comma) {
    coupon->setPricer(pricer);
    pricer->initialize(*coupon);

    open(key);
    emitReal("swaplet_rate", pricer->swapletRate(), true);
    emitStdDevChain(coupon, vol, effCap, effectiveVolatilityInput, true);

    const Rate capletGlobal = pricer->capletRate(effCap, false);
    emitReal("caplet_rate_global", capletGlobal, true);
    emitReal("effective_caplet_volatility", pricer->effectiveCapletVolatility(), true);
    const Rate floorletGlobal = pricer->floorletRate(effFloor, false);
    emitReal("floorlet_rate_global", floorletGlobal, true);
    emitReal("effective_floorlet_volatility", pricer->effectiveFloorletVolatility(), true);

    // capletRate(k) must be exactly capletRate(k, false).
    emitReal("caplet_rate_one_arg", pricer->capletRate(effCap), true);
    emitReal("floorlet_rate_one_arg", pricer->floorletRate(effFloor), true);

    if (emitLocal) {
        const Rate capletLocal = pricer->capletRate(effCap, true);
        emitReal("caplet_rate_local", capletLocal, true);
        emitReal("effective_caplet_volatility_local", pricer->effectiveCapletVolatility(), true);
        const Rate floorletLocal = pricer->floorletRate(effFloor, true);
        emitReal("floorlet_rate_local", floorletLocal, true);
        emitReal("effective_floorlet_volatility_local", pricer->effectiveFloorletVolatility(),
                 true);
    } else {
        emitRaises(
            "caplet_rate_local_raises", [&] { pricer->capletRate(effCap, true); }, true);
        emitRaises(
            "floorlet_rate_local_raises", [&] { pricer->floorletRate(effFloor, true); }, true);
    }

    emitRaises(
        "swaplet_price_raises", [&] { pricer->swapletPrice(); }, true);
    emitRaises(
        "caplet_price_raises", [&] { pricer->capletPrice(effCap); }, true);
    emitRaises(
        "floorlet_price_raises", [&] { pricer->floorletPrice(effFloor); }, false);
    close(comma);
}

// A capped / floored / collared coupon end to end.
void emitCappedFlooredBlock(const std::string& key,
                            const ext::shared_ptr<OvernightIndexedCoupon>& underlying,
                            const ext::shared_ptr<OvernightIndexedCouponPricer>& pricer,
                            Real cap,
                            Real floor,
                            bool nakedOption,
                            bool comma) {
    auto cf = ext::make_shared<CappedFlooredOvernightIndexedCoupon>(underlying, cap, floor,
                                                                    nakedOption, false);
    cf->setPricer(pricer);
    underlying->setPricer(pricer);

    open(key);
    emitBool("is_capped", cf->isCapped(), true);
    emitBool("is_floored", cf->isFloored(), true);
    emitBool("naked_option", cf->nakedOption(), true);
    if (cf->isCapped()) {
        emitReal("cap", cf->cap(), true);
        emitReal("effective_cap", cf->effectiveCap(), true);
    }
    if (cf->isFloored()) {
        emitReal("floor", cf->floor(), true);
        emitReal("effective_floor", cf->effectiveFloor(), true);
    }
    emitReal("underlying_rate", underlying->rate(), true);
    emitReal("rate", cf->rate(), true);
    emitReal("amount", cf->amount(), true);
    emitReal("effective_caplet_volatility", cf->effectiveCapletVolatility(), true);
    emitReal("effective_floorlet_volatility", cf->effectiveFloorletVolatility(), false);
    close(comma);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ================================================== forward-looking case
    Settings::instance().evaluationDate() = kFwdToday;
    const Handle<YieldTermStructure> curve = flatCurve();
    const auto sofr = ext::make_shared<Sofr>(curve);

    const Handle<OptionletVolatilityStructure> lnVol =
        constVol(kLnVol, ShiftedLognormal, kDisplacement);
    const Handle<OptionletVolatilityStructure> nVol = constVol(kNormalVol, Normal, 0.0);

    const auto compCoupon =
        makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread, RateAveraging::Compound);
    emitCouponShape("coupon", compCoupon, true);

    // Effective cap/floor of a non-daily capped/floored coupon
    // (overnightindexedcoupon.cpp:341-370, compoundSpreadDaily == false):
    //   effectiveCap   = (cap   - effectiveSpread) / gearing
    //   effectiveFloor = (floor - effectiveSpread) / gearing
    const Real effCap = (kCap - kSpread) / kGearing;
    const Real effFloor = (kFloor - kSpread) / kGearing;
    emitReal("eff_cap", effCap, true);
    emitReal("eff_floor", effFloor, true);

    emitPricerBlock("fwd_compounding_lognormal",
                    ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>(lnVol, false),
                    compCoupon, lnVol, effCap, effFloor, false, true, true);
    emitPricerBlock("fwd_compounding_normal",
                    ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>(nVol, false),
                    compCoupon, nVol, effCap, effFloor, false, true, true);
    // effectiveVolatilityInput = true: plain Black on vol*sqrt(t), and the
    // daily (local) formula is explicitly refused.
    emitPricerBlock("fwd_compounding_effective_vol",
                    ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>(lnVol, true),
                    compCoupon, lnVol, effCap, effFloor, true, false, true);

    // Averaging pricer: needs a simple-averaged coupon.
    const auto avgCoupon =
        makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread, RateAveraging::Simple);
    emitPricerBlock("fwd_averaging_lognormal",
                    ext::make_shared<BlackAveragingOvernightIndexedCouponPricer>(lnVol, false),
                    avgCoupon, lnVol, effCap, effFloor, false, true, true);
    emitPricerBlock("fwd_averaging_normal",
                    ext::make_shared<BlackAveragingOvernightIndexedCouponPricer>(nVol, false),
                    avgCoupon, nVol, effCap, effFloor, false, true, true);
    emitPricerBlock("fwd_averaging_effective_vol",
                    ext::make_shared<BlackAveragingOvernightIndexedCouponPricer>(lnVol, true),
                    avgCoupon, lnVol, effCap, effFloor, true, false, true);

    // The averaging pricer refuses a compounded coupon (cpp:383-384).
    emitRaises(
        "averaging_pricer_rejects_compounded",
        [&] {
            BlackAveragingOvernightIndexedCouponPricer p(lnVol, false);
            p.initialize(*compCoupon);
        },
        true);
    // Missing optionlet volatility on a not-yet-fixed coupon (cpp:154).
    emitRaises(
        "vol_missing_raises",
        [&] {
            auto p = ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>();
            compCoupon->setPricer(p);
            p->initialize(*compCoupon);
            p->capletRate(effCap);
        },
        true);

    // Capped / floored / collared coupons, both volatility types.
    {
        const auto ln = ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>(lnVol, false);
        emitCappedFlooredBlock("fwd_capped_lognormal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               ln, kCap, Null<Real>(), false, true);
        emitCappedFlooredBlock("fwd_floored_lognormal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               ln, Null<Real>(), kFloor, false, true);
        emitCappedFlooredBlock("fwd_collared_lognormal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               ln, kCap, kFloor, false, true);
        emitCappedFlooredBlock("fwd_capped_naked_lognormal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               ln, kCap, Null<Real>(), true, true);
        const auto nm = ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>(nVol, false);
        emitCappedFlooredBlock("fwd_capped_normal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               nm, kCap, Null<Real>(), false, true);
        emitCappedFlooredBlock("fwd_floored_normal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               nm, Null<Real>(), kFloor, false, true);
        emitCappedFlooredBlock("fwd_collared_normal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               nm, kCap, kFloor, false, true);
        const auto avgLn =
            ext::make_shared<BlackAveragingOvernightIndexedCouponPricer>(lnVol, false);
        emitCappedFlooredBlock("fwd_collared_averaging_lognormal",
                               makeCoupon(sofr, kFwdStart, kFwdEnd, kGearing, kSpread,
                                          RateAveraging::Simple),
                               avgLn, kCap, kFloor, false, true);
    }

    // ============================================== already-fixed (intrinsic)
    Settings::instance().evaluationDate() = kPastToday;
    const auto pastIndex = ext::make_shared<Sofr>();
    {
        // Deterministic ramp on the coupon's own fixing dates, matching the
        // Python fixture: 0.05 + 0.0001 * i.
        const auto tmpl = makeCoupon(pastIndex, kPastStart, kPastEnd, 1.0, 0.0,
                                     RateAveraging::Compound);
        const std::vector<Date>& fd = tmpl->fixingDates();
        for (Size i = 0; i < fd.size(); ++i)
            pastIndex->addFixing(fd[i], 0.05 + 0.0001 * static_cast<Real>(i), true);
    }

    const auto pastComp =
        makeCoupon(pastIndex, kPastStart, kPastEnd, kGearing, kSpread, RateAveraging::Compound);
    emitCouponShape("past_coupon", pastComp, true);
    // No vol handle at all: the intrinsic branch must not need one.
    {
        const auto p = ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>();
        pastComp->setPricer(p);
        p->initialize(*pastComp);
        open("past_compounding");
        emitReal("swaplet_rate", p->swapletRate(), true);
        emitReal("effective_index_fixing", pastComp->effectiveIndexFixing(), true);
        emitReal("effective_spread", pastComp->effectiveSpread(), true);
        // In the money for the caplet, out of the money for the floorlet.
        emitReal("caplet_rate_itm", p->capletRate(0.0400), true);
        emitReal("caplet_rate_otm", p->capletRate(0.0600), true);
        emitReal("floorlet_rate_itm", p->floorletRate(0.0600), true);
        emitReal("floorlet_rate_otm", p->floorletRate(0.0400), true);
        // Daily cap/floor with every fixing in the past: optionletRateLocal
        // never reaches its forward branch, so it needs no vol surface at all.
        // The fixing ramp runs 0.0500..0.0519, so a 0.0510 strike bites on
        // roughly half the days.
        emitReal("caplet_rate_local", p->capletRate(0.0510, true), true);
        emitReal("floorlet_rate_local", p->floorletRate(0.0510, true), true);
        // Nothing stochastic happened, so no effective vol was recorded.
        emitBool("effective_caplet_volatility_is_null",
                 p->effectiveCapletVolatility() == Null<Real>(), false);
        close(true);
    }
    {
        const auto pastAvg =
            makeCoupon(pastIndex, kPastStart, kPastEnd, kGearing, kSpread, RateAveraging::Simple);
        const auto p = ext::make_shared<BlackAveragingOvernightIndexedCouponPricer>();
        pastAvg->setPricer(p);
        p->initialize(*pastAvg);
        open("past_averaging");
        emitReal("swaplet_rate", p->swapletRate(), true);
        emitReal("forward_rate", (p->swapletRate() - pastAvg->spread()) / pastAvg->gearing(), true);
        emitReal("caplet_rate_itm", p->capletRate(0.0400), true);
        emitReal("caplet_rate_otm", p->capletRate(0.0600), true);
        emitReal("floorlet_rate_itm", p->floorletRate(0.0600), true);
        emitReal("floorlet_rate_otm", p->floorletRate(0.0400), true);
        emitReal("caplet_rate_local", p->capletRate(0.0510, true), true);
        emitReal("floorlet_rate_local", p->floorletRate(0.0510, true), false);
        close(true);
    }
    {
        // Swaplet rate is 0.10304 (gearing 2 x 0.051018 + spread 0.001), so a
        // cap of 0.102 bites and a floor of 0.105 bites; the collar can only
        // have one of the two in the money, and the resulting rate must land
        // exactly on the binding level.
        const auto pricer = ext::make_shared<BlackCompoundingOvernightIndexedCouponPricer>();
        emitCappedFlooredBlock("past_capped",
                               makeCoupon(pastIndex, kPastStart, kPastEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               pricer, 0.102, Null<Real>(), false, true);
        emitCappedFlooredBlock("past_floored",
                               makeCoupon(pastIndex, kPastStart, kPastEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               pricer, Null<Real>(), 0.105, false, true);
        emitCappedFlooredBlock("past_collared",
                               makeCoupon(pastIndex, kPastStart, kPastEnd, kGearing, kSpread,
                                          RateAveraging::Compound),
                               pricer, 0.102, 0.098, false, false);
    }

    std::cout << "}\n";
    return 0;
}
