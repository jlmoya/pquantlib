// migration-harness/cpp/probes/v143_cf_lineartsr/probe.cpp
//
// Reference values for ql/cashflows/lineartsrpricer.{hpp,cpp} @ v1.43 —
// LinearTsrPricer, its nested Settings bundle (RateBound / VegaRatio /
// PriceThreshold / BSStdDevs, one- and three-argument overloads) and the two
// nested root-solver objectives (PriceHelper, VegaRatioHelper) that the
// VegaRatio / PriceThreshold strategies drive through Brent.
//
// WHAT IS PINNED AND WHY
// ----------------------
// LinearTsrPricer is a bundle of optional configuration wrapped around a
// numerical static replication. Every one of those options is exactly the sort
// of argument that gets accepted by a port and then silently dropped, so the
// probe does not pin one headline number — it pins, for FOUR CMS coupons
// (in-advance, in-arrears with gearing+spread, capped, floored):
//
//   * the structural inputs the coupon contributes — fixing / payment /
//     accrual-start / accrual-end serials, accrual period, nominal, gearing,
//     spread, in-arrears flag;
//   * the market inputs that feed the model — the underlying swap's fair rate
//     (the unadjusted forward), the swap annuity 1e4*|fixedLegBPS|, the ATM
//     swaption vol at that forward, the discount factor to the payment date and
//     the coupon-discount ratio. Without these an NPV can match while two
//     errors cancel;
//   * the pricer's own convexity output — swapletRate / swapletPrice, the
//     implied convexity adjustment (swapletRate-spread)/gearing - swapRate,
//     capletRate / capletPrice, floorletRate / floorletPrice, plus the
//     coupon-level rate() and amount();
//   * pricer->meanReversion(), which proves the mean-reversion quote actually
//     reached the object.
//
// ...and it repeats that whole set ONCE PER CONFIGURATION, eighteen in all:
// every Settings strategy with a NON-DEFAULT parameter, in both its one- and
// three-argument overload where C++ has both, a high and a sub-1e-4
// mean reversion (the latter takes GsrG's `|a| < 1e-4 -> yf` branch), an empty
// coupon-discount curve, an explicitly-supplied integrator, normal-vol input
// with default vs explicit bounds (the `min(lower, -upper)` adjustment fires
// only for default bounds) and a shifted-lognormal smile (bounds shift by the
// section's shift). A configuration that is accepted and dropped cannot
// survive this.
//
// It also pins GaussKronrodNonAdaptive itself (values, evaluation counts and
// error estimates on five integrands), because that class is the pricer's
// default integrator and had to be ported alongside it.
//
// TWO KNOWN UPSTREAM DEFECTS, DELIBERATELY PINNED
// -----------------------------------------------
// (1) lineartsrpricer.cpp:283 (v1.43) reads
//
//         case Settings::PriceThreshold: {
//             Real bound = strikeFromPrice(settings_.vegaRatio_, optionType, strike);
//
//     i.e. the PriceThreshold strategy solves against `vegaRatio_`, NOT against
//     `priceThreshold_`. `priceThreshold_` is stored and never read.
//
// (2) strikeFromPrice (lineartsrpricer.cpp:237-262) then hands Brent
//     `swapRateValue_` as its GUESS while `swapRateValue_` is simultaneously
//     one end of the bracket it passes (`a` for a call, `b` for a put).
//     Solver1D::solve requires xMin < guess < xMax, so it always throws; the
//     blanket `catch (...)` swallows it and `k` keeps the plain rate bound.
//
// Together these make the whole PriceThreshold strategy inert: it collapses
// onto RateBound with whatever bounds were configured. C++ v1.43 is the source
// of truth, so the port must reproduce that collapse, and the case set pins it
// as two IDENTITIES that a "helpfully fixed" port would break:
//
//     price_threshold_1arg  ==  default            (default bounds)
//     price_threshold_3arg  ==  rate_bound_narrow  (explicit bounds 0.005/0.045)
//
// plus price_threshold_quirk, which changes vegaRatio_ under the same
// priceThreshold_ and must STILL be identical.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/lineartsr.json.

#include <ql/cashflows/capflooredcoupon.hpp>
#include <ql/cashflows/cmscoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/lineartsrpricer.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/math/integrals/kronrodintegral.hpp>
#include <ql/math/integrals/segmentintegral.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --------------------------------------------------------------------------
// Minimal JSON writer (objects only; values are numbers / bools / nested).
// --------------------------------------------------------------------------

std::string fmt(Real v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

struct J {
    std::string body;
    bool first = true;

    void raw(const std::string& key, const std::string& value) {
        if (!first)
            body += ",\n";
        first = false;
        body += "\"" + key + "\": " + value;
    }
    void num(const std::string& key, Real v) { raw(key, fmt(v)); }
    void integer(const std::string& key, BigInteger v) { raw(key, std::to_string(v)); }
    void boolean(const std::string& key, bool v) { raw(key, v ? "true" : "false"); }
    void text(const std::string& key, const std::string& v) { raw(key, "\"" + v + "\""); }
    void obj(const std::string& key, const J& o) { raw(key, o.str()); }
    std::string str() const { return "{\n" + body + "\n}"; }
};

// --------------------------------------------------------------------------
// Market — three DISTINCT flat curves so that every curve slot is separately
// observable: forwarding (3.0%), exogenous discounting on the swap index
// (2.5%) and the coupon-discount curve (2.8%). Equal curves would let a port
// that wires the wrong handle still match.
// --------------------------------------------------------------------------

const Date kToday(15, January, 2024);
const Real kFwdRate = 0.030;
const Real kDiscRate = 0.025;
const Real kCouponDiscRate = 0.028;

Handle<YieldTermStructure> flatCurve(Real r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kToday, r, Actual365Fixed()));
}

struct Market {
    Handle<YieldTermStructure> forwarding;
    Handle<YieldTermStructure> discounting;
    Handle<YieldTermStructure> couponDiscount;
    ext::shared_ptr<IborIndex> ibor;
    ext::shared_ptr<SwapIndex> swapIndex;
};

Market makeMarket() {
    Market m;
    m.forwarding = flatCurve(kFwdRate);
    m.discounting = flatCurve(kDiscRate);
    m.couponDiscount = flatCurve(kCouponDiscRate);
    m.ibor = ext::make_shared<Euribor6M>(m.forwarding);
    // 10Y EUR fixed-vs-Euribor6M swap index, annual Unadjusted 30/360 fixed
    // leg, with an EXOGENOUS discount curve so that
    // SwapIndex::exogenousDiscount() is true and LinearTsrPricer takes the
    // discountingTermStructure() branch rather than reusing the forward curve.
    m.swapIndex = ext::make_shared<SwapIndex>(
        "EuriborSwapIsdaFixA", Period(10, Years), m.ibor->fixingDays(),
        EURCurrency(), TARGET(), Period(1, Years), Unadjusted,
        Thirty360(Thirty360::BondBasis), m.ibor, m.discounting);
    return m;
}

Handle<SwaptionVolatilityStructure> constVol(Volatility v,
                                             VolatilityType type = ShiftedLognormal,
                                             Real shift = 0.0) {
    return Handle<SwaptionVolatilityStructure>(
        ext::make_shared<ConstantSwaptionVolatility>(
            kToday, TARGET(), Following, v, Actual365Fixed(), type, shift));
}

// --------------------------------------------------------------------------
// Coupons
// --------------------------------------------------------------------------

struct CouponSpec {
    std::string key;
    Date start, end, payment;
    Real nominal, gearing, spread;
    bool inArrears;
    Rate cap, floor;         // Null<Rate>() when absent (plain CmsCoupon)
    Rate testCap, testFloor; // strikes fed to capletRate() / floorletRate()
};

std::vector<CouponSpec> makeCouponSpecs() {
    const TARGET cal;
    const Date s1 = cal.advance(kToday, Period(5, Years));
    const Date e1 = cal.advance(s1, Period(1, Years));
    const Date s2 = cal.advance(kToday, Period(7, Years));
    const Date e2 = cal.advance(s2, Period(1, Years));
    const Date s3 = cal.advance(kToday, Period(3, Years));
    const Date e3 = cal.advance(s3, Period(1, Years));
    const Date s4 = cal.advance(kToday, Period(12, Years));
    const Date e4 = cal.advance(s4, Period(1, Years));

    std::vector<CouponSpec> v;
    // in-advance, unit nominal, no gearing / spread — swapletRate here IS the
    // convexity-adjusted forward.
    v.push_back({"advance", s1, e1, e1, 1.0, 1.0, 0.0, false,
                 Null<Rate>(), Null<Rate>(), 0.045, 0.025});
    // in-arrears with a non-unit nominal, gearing and spread — the fixing date
    // moves to the END of the accrual period, and gearing / spread must both
    // reach the price.
    v.push_back({"arrears", s2, e2, e2, 1.0e6, 1.5, 0.0025, true,
                 Null<Rate>(), Null<Rate>(), 0.05, 0.02});
    // capped: rate() = swaplet + floorlet - caplet must lose the caplet leg.
    v.push_back({"capped", s3, e3, e3, 1.0e6, 1.0, 0.0, false,
                 0.045, Null<Rate>(), 0.045, 0.025});
    // floored, long expiry so the vega-ratio / price-threshold solves bite on a
    // materially wider smile.
    v.push_back({"floored", s4, e4, e4, 1.0e6, 1.0, 0.0, false,
                 Null<Rate>(), 0.030, 0.05, 0.030});
    return v;
}

ext::shared_ptr<FloatingRateCoupon> buildCoupon(const CouponSpec& s,
                                                const ext::shared_ptr<SwapIndex>& idx) {
    const DayCounter dc = Thirty360(Thirty360::BondBasis);
    if (s.cap == Null<Rate>() && s.floor == Null<Rate>()) {
        return ext::make_shared<CmsCoupon>(
            s.payment, s.nominal, s.start, s.end, idx->fixingDays(), idx,
            s.gearing, s.spread, s.start, s.end, dc, s.inArrears);
    }
    return ext::make_shared<CappedFlooredCmsCoupon>(
        s.payment, s.nominal, s.start, s.end, idx->fixingDays(), idx,
        s.gearing, s.spread, s.cap, s.floor, s.start, s.end, dc, s.inArrears);
}

// The coupon the pricer must be initialize()d on: LinearTsrPricer::initialize
// dynamic_casts to CmsCoupon, which a CappedFlooredCmsCoupon is NOT — it wraps
// one. CappedFlooredCoupon::rate() itself goes through underlying()->rate().
ext::shared_ptr<CmsCoupon> pricingCoupon(const ext::shared_ptr<FloatingRateCoupon>& c) {
    if (auto cf = ext::dynamic_pointer_cast<CappedFlooredCoupon>(c))
        return ext::dynamic_pointer_cast<CmsCoupon>(cf->underlying());
    return ext::dynamic_pointer_cast<CmsCoupon>(c);
}

// LinearTsrPricer::initialize is a private override; it is only reachable
// through the FloatingRateCouponPricer interface (which is how
// FloatingRateCoupon::rate() reaches it too).
void initPricer(const ext::shared_ptr<LinearTsrPricer>& p, const FloatingRateCoupon& c) {
    static_cast<FloatingRateCouponPricer&>(*p).initialize(c);
}

J emitCoupon(const CouponSpec& spec,
             const Market& mkt,
             const Handle<SwaptionVolatilityStructure>& vol,
             const Handle<YieldTermStructure>& couponDiscount,
             const ext::shared_ptr<LinearTsrPricer>& pricer) {
    J j;
    auto cpn = buildCoupon(spec, mkt.swapIndex);
    auto base = pricingCoupon(cpn);
    cpn->setPricer(pricer);
    initPricer(pricer, *base);

    const Date fixing = base->fixingDate();
    const Date pay = base->date();

    // --- structural (TIGHT: pure date arithmetic / stored fields) ---
    j.integer("fixing_serial", fixing.serialNumber());
    j.integer("payment_serial", pay.serialNumber());
    j.integer("accrual_start_serial", base->accrualStartDate().serialNumber());
    j.integer("accrual_end_serial", base->accrualEndDate().serialNumber());
    j.num("accrual_period", base->accrualPeriod());
    j.num("nominal", base->nominal());
    j.num("gearing", base->gearing());
    j.num("spread", base->spread());
    j.boolean("is_in_arrears", base->isInArrears());

    // --- market inputs that feed the model ---
    auto swap = mkt.swapIndex->underlyingSwap(fixing);
    const Real swapRate = swap->fairRate();
    const Real annuity = 1.0e4 * std::fabs(swap->fixedLegBPS());
    const Real discPay = mkt.discounting->discount(pay);
    const Real couponPay = couponDiscount.empty() ? 1.0 : couponDiscount->discount(pay);
    j.num("swap_rate", swapRate);
    j.num("annuity", annuity);
    j.num("atm_vol", vol->volatility(fixing, mkt.swapIndex->tenor(), swapRate));
    j.num("discount_payment", discPay);
    j.num("coupon_discount_ratio", couponPay / discPay);
    j.num("mean_reversion", pricer->meanReversion());

    // --- pricer output ---
    const Real swapletRate = pricer->swapletRate();
    j.num("swaplet_rate", swapletRate);
    j.num("swaplet_price", pricer->swapletPrice());
    // (swapletRate - spread)/gearing - swapRate isolates the convexity
    // adjustment from the affine gearing/spread wrapper, so a sign or scale
    // error in the linear TSR parameters a_ / b_ cannot hide behind them.
    j.num("convexity_adjustment", (swapletRate - base->spread()) / base->gearing() - swapRate);
    j.num("caplet_rate", pricer->capletRate(spec.testCap));
    j.num("caplet_price", pricer->capletPrice(spec.testCap));
    j.num("floorlet_rate", pricer->floorletRate(spec.testFloor));
    j.num("floorlet_price", pricer->floorletPrice(spec.testFloor));
    j.num("test_cap", spec.testCap);
    j.num("test_floor", spec.testFloor);

    // --- coupon level ---
    j.num("rate", cpn->rate());
    j.num("amount", cpn->amount());
    return j;
}

// --------------------------------------------------------------------------
// Configuration sweep
// --------------------------------------------------------------------------

struct Case {
    std::string key;
    LinearTsrPricer::Settings settings;
    Real meanReversion;
    Handle<SwaptionVolatilityStructure> vol;
    bool withCouponDiscount;
    ext::shared_ptr<Integrator> integrator; // null => C++ default GaussKronrodNonAdaptive
};

std::vector<Case> makeCases() {
    using S = LinearTsrPricer::Settings;
    const Real mr = 0.01;
    const auto lnVol = constVol(0.20);
    std::vector<Case> cases;

    // 1. default-constructed Settings: RateBound with (1e-4, 2.0).
    cases.push_back({"default", S(), mr, lnVol, true, nullptr});
    // 2. RateBound, both bounds non-default and narrow enough to truncate the
    //    replication integral materially.
    cases.push_back({"rate_bound", S().withRateBound(0.01, 0.08), mr, lnVol, true, nullptr});
    // 3. RateBound clamped INSIDE the probed cap/floor strikes, so that
    //    optionletPrice's two early-outs fire: Call with strike >= upper and
    //    Put with strike <= lower both return exactly 0.
    cases.push_back({"rate_bound_clamped", S().withRateBound(0.028, 0.045), mr, lnVol, true, nullptr});
    // 4. The SAME bounds the PriceThreshold 3-arg overload uses, under the
    //    RateBound strategy — the degeneracy reference for case 8.
    cases.push_back({"rate_bound_narrow", S().withRateBound(0.005, 0.045), mr, lnVol, true, nullptr});
    // 5/6. VegaRatio, non-default ratio (default is 0.01); 1-arg keeps default
    //      bounds (defaultBounds_ = true), 3-arg clamps the vega-implied upper
    //      bound at 0.045 while leaving the vega-implied lower bound alone.
    cases.push_back({"vega_ratio_1arg", S().withVegaRatio(0.05), mr, lnVol, true, nullptr});
    cases.push_back({"vega_ratio_3arg", S().withVegaRatio(0.05, 0.005, 0.045), mr, lnVol, true, nullptr});
    // 7/8. PriceThreshold. In v1.43 this strategy is INERT — see the header:
    //      strikeFromPrice reads vegaRatio_ instead of priceThreshold_, and it
    //      then hands Brent a guess that equals one of its own brackets, so
    //      Solver1D always throws "guess must be in (xMin, xMax)" and the
    //      catch-all restores the plain rate bound. The strategy therefore
    //      collapses onto RateBound with whatever bounds were configured.
    //      Both cases are pinned so the port must reproduce that collapse:
    //      case 7 must equal case 1, case 8 must equal case 4.
    cases.push_back({"price_threshold_1arg", S().withPriceThreshold(1.0e-6), mr, lnVol, true, nullptr});
    cases.push_back({"price_threshold_3arg", S().withPriceThreshold(1.0e-6, 0.005, 0.045), mr, lnVol, true, nullptr});
    // 9. Same priceThreshold_ as case 7 but a different vegaRatio_ — still
    //    identical, which pins that NEITHER field survives into the result.
    cases.push_back({"price_threshold_quirk", S().withVegaRatio(1.0e-3).withPriceThreshold(1.0e-6), mr, lnVol, true, nullptr});
    // 10/11. BSStdDevs, non-default stdDevs (default is 3.0); the 3-arg
    //        overload's bounds clamp the std-dev-implied window.
    cases.push_back({"bs_std_devs_1arg", S().withBSStdDevs(4.0), mr, lnVol, true, nullptr});
    cases.push_back({"bs_std_devs_3arg", S().withBSStdDevs(4.0, 0.02, 0.05), mr, lnVol, true, nullptr});
    // 12. Mean reversion well above the 1e-4 GsrG cutoff.
    cases.push_back({"mean_reversion_high", S(), 0.05, lnVol, true, nullptr});
    // 13. Mean reversion BELOW the 1e-4 cutoff: GsrG returns the year fraction
    //     itself rather than (1-exp(-a*t))/a.
    cases.push_back({"mean_reversion_tiny", S(), 5.0e-5, lnVol, true, nullptr});
    // 14. No coupon-discount curve: couponDiscountRatio_ collapses to 1 and the
    //     *_price members change while the *_rate members do not.
    cases.push_back({"no_coupon_discount_curve", S(), mr, lnVol, false, nullptr});
    // 15. Explicit integrator (a coarse 50-segment trapezoid) instead of the
    //     default GaussKronrodNonAdaptive(1e-10, 5000, 1e-10) — a dropped
    //     integrator argument shows up immediately.
    cases.push_back({"segment_integrator", S(), mr, lnVol, true,
                     ext::make_shared<SegmentIntegral>(50)});
    // 16/17. Normal volatility. With DEFAULT bounds the lower bound is pulled
    //        to min(lower, -upper); with explicit bounds it is not.
    cases.push_back({"normal_vol_default_bounds", S(), mr, constVol(0.0060, Normal), true, nullptr});
    cases.push_back({"normal_vol_explicit_bounds", S().withRateBound(-0.01, 0.10), mr,
                     constVol(0.0060, Normal), true, nullptr});
    // 18. Shifted lognormal: both bounds are reduced by the section's shift.
    cases.push_back({"shifted_lognormal", S(), mr, constVol(0.18, ShiftedLognormal, 0.02), true, nullptr});
    return cases;
}

// A coupon whose fixing is already in the past: initialize() skips the model
// entirely and swaplet/caplet/floorlet fall back to the recorded fixing.
J emitPastFixing(const Market& mkt,
                 const Handle<SwaptionVolatilityStructure>& vol,
                 const Handle<YieldTermStructure>& couponDiscount) {
    const TARGET cal;
    const Date start = cal.advance(kToday, Period(-3, Months));
    const Date end = cal.advance(start, Period(1, Years));
    // testCap below and testFloor above the recorded 3.45% fixing, so both
    // intrinsic-value branches produce a NON-zero number.
    const CouponSpec spec{"past", start, end, end, 1.0e6, 1.2, 0.001, false,
                          Null<Rate>(), Null<Rate>(), 0.030, 0.040};

    auto cpn = buildCoupon(spec, mkt.swapIndex);
    auto base = pricingCoupon(cpn);
    const Date fixing = base->fixingDate();
    const Real recorded = 0.0345;
    mkt.swapIndex->addFixing(fixing, recorded, true);

    auto pricer = ext::make_shared<LinearTsrPricer>(
        vol, Handle<Quote>(ext::make_shared<SimpleQuote>(0.01)), couponDiscount);
    cpn->setPricer(pricer);
    initPricer(pricer, *base);

    J j;
    j.integer("fixing_serial", fixing.serialNumber());
    j.integer("payment_serial", base->date().serialNumber());
    j.integer("accrual_start_serial", base->accrualStartDate().serialNumber());
    j.integer("accrual_end_serial", base->accrualEndDate().serialNumber());
    j.num("accrual_period", base->accrualPeriod());
    j.num("nominal", base->nominal());
    j.num("gearing", base->gearing());
    j.num("spread", base->spread());
    j.num("recorded_fixing", recorded);
    j.num("discount_payment", mkt.discounting->discount(base->date()));
    j.num("coupon_discount_ratio",
          couponDiscount->discount(base->date()) / mkt.discounting->discount(base->date()));
    j.num("swaplet_rate", pricer->swapletRate());
    j.num("swaplet_price", pricer->swapletPrice());
    j.num("caplet_rate", pricer->capletRate(spec.testCap));
    j.num("caplet_price", pricer->capletPrice(spec.testCap));
    j.num("floorlet_rate", pricer->floorletRate(spec.testFloor));
    j.num("floorlet_price", pricer->floorletPrice(spec.testFloor));
    j.num("test_cap", spec.testCap);
    j.num("test_floor", spec.testFloor);
    j.num("rate", cpn->rate());
    j.num("amount", cpn->amount());
    return j;
}

// setMeanReversion() must re-link the quote AND change the answer.
J emitSetMeanReversion(const Market& mkt,
                       const Handle<SwaptionVolatilityStructure>& vol,
                       const Handle<YieldTermStructure>& couponDiscount) {
    const auto specs = makeCouponSpecs();
    const CouponSpec& spec = specs.front();
    auto cpn = buildCoupon(spec, mkt.swapIndex);
    auto base = pricingCoupon(cpn);

    auto pricer = ext::make_shared<LinearTsrPricer>(
        vol, Handle<Quote>(ext::make_shared<SimpleQuote>(0.01)), couponDiscount);
    cpn->setPricer(pricer);

    J j;
    initPricer(pricer, *base);
    j.num("mean_reversion_before", pricer->meanReversion());
    j.num("swaplet_rate_before", pricer->swapletRate());

    pricer->setMeanReversion(Handle<Quote>(ext::make_shared<SimpleQuote>(0.08)));
    initPricer(pricer, *base);
    j.num("mean_reversion_after", pricer->meanReversion());
    j.num("swaplet_rate_after", pricer->swapletRate());
    return j;
}

// An empty mean-reversion Handle is dereferenced by GsrG during initialize().
J emitEmptyMeanReversion(const Market& mkt,
                         const Handle<SwaptionVolatilityStructure>& vol,
                         const Handle<YieldTermStructure>& couponDiscount) {
    const auto specs = makeCouponSpecs();
    auto cpn = buildCoupon(specs.front(), mkt.swapIndex);
    auto base = pricingCoupon(cpn);
    auto pricer = ext::make_shared<LinearTsrPricer>(vol, Handle<Quote>(), couponDiscount);

    J j;
    bool threw = false;
    try {
        initPricer(pricer, *base);
        (void)pricer->swapletRate();
    } catch (const std::exception&) {
        threw = true;
    }
    j.boolean("initialize_throws", threw);

    threw = false;
    try {
        (void)pricer->meanReversion();
    } catch (const std::exception&) {
        threw = true;
    }
    j.boolean("mean_reversion_throws", threw);
    return j;
}

// GaussKronrodNonAdaptive is the integrator LinearTsrPricer installs by
// default, and PQuantLib did not have it (only the adaptive variant), so it is
// ported alongside the pricer and pinned directly here rather than only
// through the coupon numbers. Beyond the values themselves, the EVALUATION
// COUNT is pinned: it identifies which rule in the 10/21/43/87 cascade the
// convergence test stopped at, which is what the rescaleError logic actually
// controls. A port that got rescaleError wrong but the weights right would
// still match the integrals and fail here.
J emitNonAdaptiveIntegrator() {
    GaussKronrodNonAdaptive gk(1.0e-10, 5000, 1.0e-10);

    struct Fn {
        const char* key;
        std::function<Real(Real)> f;
        Real a, b;
    };
    const std::vector<Fn> fns = {
        // polynomial — integrated exactly by the 10/21 pair
        {"poly", [](Real x) { return x * x * x - 2.0 * x + 1.0; }, 0.0, 1.0},
        // entire function, still easy
        {"gauss", [](Real x) { return std::exp(-x * x); }, -3.0, 3.0},
        // Runge function — needs more points
        {"runge", [](Real x) { return 1.0 / (1.0 + 25.0 * x * x); }, -1.0, 1.0},
        // square-root singularity in the derivative at the left end
        {"sqrt", [](Real x) { return std::sqrt(x); }, 0.0, 1.0},
        // kink inside the interval — worst case, drives the 87-point rule
        {"kink", [](Real x) { return std::fabs(x - 0.3); }, 0.0, 1.0},
    };

    J j;
    for (const auto& fn : fns) {
        const Real value = gk(fn.f, fn.a, fn.b);
        J e;
        e.num("value", value);
        e.integer("evaluations", static_cast<BigInteger>(gk.numberOfEvaluations()));
        e.num("absolute_error", gk.absoluteError());
        // reversed limits must negate (base-class Integrator::operator())
        e.num("reversed", gk(fn.f, fn.b, fn.a));
        j.obj(fn.key, e);
    }
    j.num("degenerate_interval", gk(fns[1].f, 1.0, 1.0));
    j.num("relative_accuracy", gk.relativeAccuracy());
    gk.setRelativeAccuracy(1.0e-3);
    j.num("relative_accuracy_after_set", gk.relativeAccuracy());
    // A slack relative accuracy stops the cascade earlier: same integrand,
    // fewer evaluations and a different answer.
    j.num("kink_loose_rel", gk(fns[4].f, 0.0, 1.0));
    j.integer("kink_loose_rel_evaluations", static_cast<BigInteger>(gk.numberOfEvaluations()));
    return j;
}

// A non-CMS coupon must be rejected ("CMS coupon needed").
J emitNonCmsRejected(const Market& mkt,
                     const Handle<SwaptionVolatilityStructure>& vol) {
    const TARGET cal;
    const Date start = cal.advance(kToday, Period(5, Years));
    const Date end = cal.advance(start, Period(6, Months));
    IborCoupon ibor(end, 1.0, start, end, mkt.ibor->fixingDays(), mkt.ibor);
    auto pricer = ext::make_shared<LinearTsrPricer>(
        vol, Handle<Quote>(ext::make_shared<SimpleQuote>(0.01)));
    J j;
    bool threw = false;
    try {
        initPricer(pricer, ibor);
    } catch (const std::exception&) {
        threw = true;
    }
    j.boolean("initialize_throws", threw);
    return j;
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    const Market mkt = makeMarket();
    const auto specs = makeCouponSpecs();
    const auto cases = makeCases();

    J setup;
    setup.integer("evaluation_date_serial", kToday.serialNumber());
    setup.num("forward_rate", kFwdRate);
    setup.num("discount_rate", kDiscRate);
    setup.num("coupon_discount_rate", kCouponDiscRate);
    setup.num("default_lower_bound", 0.0001);
    setup.num("default_upper_bound", 2.0000);
    setup.text("swap_index_family", mkt.swapIndex->familyName());
    setup.integer("swap_index_fixing_days", mkt.swapIndex->fixingDays());
    setup.boolean("swap_index_exogenous_discount", mkt.swapIndex->exogenousDiscount());

    J allCases;
    for (const auto& c : cases) {
        J caseJson;
        for (const auto& s : specs) {
            auto pricer = ext::make_shared<LinearTsrPricer>(
                c.vol,
                Handle<Quote>(ext::make_shared<SimpleQuote>(c.meanReversion)),
                c.withCouponDiscount ? mkt.couponDiscount : Handle<YieldTermStructure>(),
                c.settings,
                c.integrator);
            caseJson.obj(s.key,
                         emitCoupon(s, mkt, c.vol,
                                    c.withCouponDiscount ? mkt.couponDiscount
                                                         : Handle<YieldTermStructure>(),
                                    pricer));
        }
        allCases.obj(c.key, caseJson);
    }

    const auto lnVol = constVol(0.20);

    J root;
    root.obj("setup", setup);
    root.obj("cases", allCases);
    root.obj("past_fixing", emitPastFixing(mkt, lnVol, mkt.couponDiscount));
    root.obj("set_mean_reversion", emitSetMeanReversion(mkt, lnVol, mkt.couponDiscount));
    root.obj("empty_mean_reversion", emitEmptyMeanReversion(mkt, lnVol, mkt.couponDiscount));
    root.obj("non_cms_coupon", emitNonCmsRejected(mkt, lnVol));
    root.obj("gauss_kronrod_non_adaptive", emitNonAdaptiveIntegrator());

    std::cout << root.str() << "\n";
    return 0;
}
