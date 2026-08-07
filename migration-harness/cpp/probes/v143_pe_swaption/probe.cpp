// migration-harness/cpp/probes/v143_pe_swaption/probe.cpp
//
// Reference values for the swaption / cap-floor pricing-engine cluster of
// C++ QuantLib v1.43:
//
//   * detail::Black76Spec, detail::BachelierSpec
//       (ql/pricingengines/swaption/blackswaptionengine.hpp:81-125)
//       The two policy structs that parameterise
//       detail::BlackStyleSwaptionEngine<Spec>. Each carries a static
//       VolatilityType `type` and three functions value()/vega()/delta().
//       They are NOT syntax-only tags: BlackSwaptionEngine and
//       BachelierSwaptionEngine differ ONLY by which Spec they instantiate.
//
//   * detail::BlackStyleSwaptionEngine<Spec> + BlackSwaptionEngine +
//     BachelierSwaptionEngine  (blackswaptionengine.hpp:53-78, 136-175)
//
//   * Gaussian1dSwaptionEngine   (gaussian1dswaptionengine.{hpp,cpp})
//   * Gaussian1dJamshidianSwaptionEngine
//                                (gaussian1djamshidianswaptionengine.{hpp,cpp})
//
//   * detail::HullWhiteCapFloorPricer, MCHullWhiteCapFloorEngine,
//     MakeMCHullWhiteCapFloorEngine   (capfloor/mchullwhiteengine.{hpp,cpp})
//
//   * BasketGeneratingEngine::calibrationBasket + its private nested
//     MatchHelper cost function     (swaption/basketgeneratingengine.{hpp,cpp})
//
// What has to be pinned, and why
// ------------------------------
//
// 1. Black76Spec / BachelierSpec are pinned STANDALONE, not only through the
//    engines. Both structs are stateless, so `Spec().value(...)` can be called
//    directly. Pinning them separately means a failing engine case can be
//    bisected into "wrong market inputs" vs "wrong formula". Note the
//    signatures differ in an easy-to-miss way: BachelierSpec::value/vega/delta
//    take a trailing UNNAMED Real (the displacement) and IGNORE it. A port
//    that lets displacement leak into the Bachelier branch produces wrong
//    numbers only when displacement != 0, which no default-argument test
//    would catch. Cases `spec_bachelier_*_disp*` pin exactly that.
//
// 2. blackFormulaForwardDerivative / bachelierBlackFormulaForwardDerivative
//    (the `delta` results) have a stdDev == 0 branch:
//        sign * max(1.0 * sign((forward - strike) * sign), 0.0) * discount
//    i.e. at zero vol delta is annuity for ITM, 0 for OTM, and — because
//    boost::math::sign(0) == 0 — exactly 0 AT THE MONEY. Cases
//    `spec_*_stddev0_atm` drive that to zero deliberately.
//    blackFormulaForwardDerivative also has a `strike + displacement == 0`
//    early return: discount for a Call, 0 for a Put, regardless of stdDev.
//
// 3. BlackStyleSwaptionEngine::calculate() (blackswaptionengine.hpp:220-327)
//    is one long body with several things a port typically gets wrong:
//      a. It re-prices the underlying swap on the ENGINE's discountCurve_
//         (`DiscountingSwapEngine(discountCurve_, false)`), not on the index's
//         forwarding curve. The probe deliberately uses discount != forward
//         curve so atmForward/annuity change if a port skips this.
//      b. `swapLength` comes from `vol_->swapLength(firstFloatDate,
//         lastFloatDate)`, which is (end-start)/365.25*12 ROUNDED TO WHOLE
//         MONTHS and then divided by 12, floored at 1/12. It is NOT a
//         day-count year fraction. Pinned in additionalResults.
//      c. `exerciseTime` is `vol_->timeFromReference(exerciseDate)` — the
//         VOL structure's reference date, which for the flat-vol ctors is
//         ConstantSwaptionVolatility(0 settlement days, NullCalendar) i.e.
//         the evaluation date. Cases `black_volts_settle2_*` build the
//         engine from an explicit ConstantSwaptionVolatility with TWO
//         settlement days on TARGET, moving the vol reference date and hence
//         stdDev, vega, timeToExpiry and impliedVolatility. A port that uses
//         the DISCOUNT curve's reference date instead passes the first block
//         and fails this one.
//      d. Cash/ParYieldCurve settlement uses CashFlows::bps of the fixed leg
//         under InterestRate(atmForward, firstCouponDayCounter, Compounded,
//         fixedScheduleFrequency) discounted to `discountDate`, where
//         discountDate is the first fixed coupon's accrual start under
//         CashAnnuityModel::DiscountCurve but the SWAP's valuation date under
//         CashAnnuityModel::SwapRate. Both are pinned; they differ.
//      e. A non-zero floating spread produces
//         `correction = spread * |floatingLegBPS / fixedLegBPS|`
//         subtracted from BOTH strike and atmForward.
//      f. Option type comes from the SWAP type: Payer -> Call, Receiver -> Put.
//      g. `results_.valuationDate` is filled from the re-priced swap.
//    Every additionalResults key is pinned for every engine case, so a port
//    that computes the right NPV by two compensating errors still fails.
//
// 4. Gaussian1dSwaptionEngine (gaussian1dswaptionengine.cpp:26-345)
//    is a backward-induction integrator on the model's y-grid.
//      - The interpolant is CubicInterpolation(Spline, monotonic=TRUE,
//        Lagrange BC at both ends) — NOT a natural cubic spline. The
//        difference is visible at the 1e-5 level on these markets, which is
//        why the probe sweeps integrationPoints and stddevs: a port using
//        scipy's natural spline cannot match all of {16,32,64} x {4,7}.
//      - `extrapolatePayoff` / `flatPayoffExtrapolation` change the tail
//        treatment: flat extension of p[] vs continuation of the boundary
//        cubic, and in the non-flat case the tail that is extended depends
//        on the option TYPE (Call -> upper tail only, Put -> lower tail
//        only). Both flags are swept for both types.
//      - `probabilities` (None/Naive/Digital) adds an additionalResults
//        vector "probabilities" of size (idx - minIdxAlive + 2). Pinned for
//        a 3-date Bermudan.
//      - Guard: settlementMethod == ParYieldCurve throws.
//      - Guard: last exercise date <= settlement returns value 0.0 WITHOUT
//        touching the swap (pinned as `g1d_expired`).
//      - Coupon selection uses `upper_bound(schedule.dates(), expiry0 - 1)`,
//        i.e. coupons whose period start is >= expiry0 (the "-1" makes the
//        comparison inclusive of expiry0 itself). Off-by-one here silently
//        drops or adds a coupon.
//
// 5. Gaussian1dJamshidianSwaptionEngine (…jamshidian….cpp:60-125)
//      - Brent root-search on y in [-8, 8] with accuracy 1e-8, guess 0.00,
//        for the y* that makes the remaining fixed flows worth `nominal`
//        (discounted from the swap's value date to expiry under the model).
//      - Option type is INVERTED relative to the standard engine:
//        Payer -> Put on the bond, Receiver -> Call.
//      - Guards: non-European exercise throws; non-zero swap spread throws;
//        ParYieldCurve throws.
//      - `g1d_vs_jamshidian_*` pin the SAME European swaption under both
//        engines. They must agree to the integration tolerance; that
//        agreement is the strongest single check on either port.
//
// 6. MCHullWhiteCapFloorEngine.
//      - There is NO control variate in v1.43: the engine passes
//        `McSimulation<SingleVariate, RNG, S>(antitheticVariate, false)` and
//        MakeMCHullWhiteCapFloorEngine has no withControlVariate(). The
//        `mc_make_no_control_variate_builder` case pins that the builder's
//        named-parameter set is exactly {BrownianBridge, Samples,
//        AbsoluteTolerance, MaxSamples, Seed, AntitheticVariate}.
//      - Everything is pinned EXACTLY (NPV and errorEstimate), because
//        MersenneTwisterUniformRng + InverseCumulativeNormal + PathGenerator
//        are fully deterministic given a seed. To make a mismatch
//        bisectable the probe ALSO pins, separately:
//          * the raw uniform-to-normal stream from
//            PseudoRandom::make_sequence_generator(dim, seed)  (`mc_rng_*`)
//          * the engine's TimeGrid                             (`mc_time_grid`)
//          * the first two generated paths                     (`mc_path_*`)
//          * detail::HullWhiteCapFloorPricer applied to a HAND-BUILT path
//            (`mc_pathpricer_*`), which isolates the payoff from the RNG.
//      - HullWhiteCapFloorPricer indexes the path as path[i-pastFixings+1]
//        and path[i-pastFixings+2]; with an all-future-fixing cap the first
//        caplet reads path[1] and path[2], NOT path[0]. It also multiplies by
//        1/discountBond(end, Tb, r_i2) and finally by endDiscount_ =
//        curve->discount(forwardMeasureTime) — i.e. it prices under the
//        Tb-forward measure. A port that discounts on the curve instead is
//        wrong by a stochastic factor and cannot be caught by a 3-sigma band.
//      - MakeMCHullWhiteCapFloorEngine::withSamples/withAbsoluteTolerance are
//        mutually exclusive and throw on the second call (pinned both ways).
//
// 7. BasketGeneratingEngine::calibrationBasket + MatchHelper.
//      - Reached through Gaussian1dNonstandardSwaptionEngine (the only public
//        route: calibrationBasket() is a non-virtual member of the protected
//        base and MatchHelper is a private nested class).
//      - Naive: one SwaptionHelper per alive exercise date, expiry = that
//        date, maturity = underlyingLastDate(), vol = the smile's ATM vol,
//        strike = Null (i.e. ATM), nominal 1.0.
//      - MaturityStrikeByDeltaGamma: a LevenbergMarquardt fit of
//        (nominal, maturity, rate) to match (npv, delta, gamma) of the exotic
//        at y=0 with h = 1e-4, EndCriteria(1000, 200, 1e-8, 1e-8, 1e-8),
//        NoConstraint. Post-processing floors the strike at 0.00001 - shift
//        and the nominal at 0.000001, and rounds the maturity to whole
//        months with `floor(months + 0.5)` (round-half-up), NOT floor().
//      - Pinned: per-helper expiry, maturity, strike, nominal and vol.
//
// Emits JSON on stdout; nothing else may be printed.

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/instruments/capfloor.hpp>
#include <ql/instruments/makecapfloor.hpp>
#include <ql/instruments/makevanillaswap.hpp>
#include <ql/instruments/nonstandardswap.hpp>
#include <ql/instruments/nonstandardswaption.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/methods/montecarlo/pathgenerator.hpp>
#include <ql/models/shortrate/calibrationhelpers/swaptionhelper.hpp>
#include <ql/models/shortrate/onefactormodels/gsr.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/pricingengines/capfloor/analyticcapfloorengine.hpp>
#include <ql/pricingengines/capfloor/mchullwhiteengine.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/pricingengines/swaption/blackswaptionengine.hpp>
#include <ql/pricingengines/swaption/gaussian1djamshidianswaptionengine.hpp>
#include <ql/pricingengines/swaption/gaussian1dnonstandardswaptionengine.hpp>
#include <ql/pricingengines/swaption/gaussian1dswaptionengine.hpp>
#include <ql/processes/hullwhiteprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// ---------------------------------------------------------------------------
std::string num(Real v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

std::string isoDate(const Date& d) {
    std::ostringstream o;
    o << std::setfill('0') << d.year() << "-" << std::setw(2)
      << static_cast<int>(d.month()) << "-" << std::setw(2) << d.dayOfMonth();
    return o.str();
}

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, "\"" + v + "\""); }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
    Obj& d(const std::string& k, const Date& v) { return s(k, isoDate(v)); }
    Obj& a(const std::string& k, const std::vector<Real>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j ? ", " : "") + num(v[j]);
        return put(k, body + "]");
    }
    Obj& da(const std::string& k, const std::vector<Date>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j ? ", " : "") + std::string("\"") + isoDate(v[j]) + "\"";
        return put(k, body + "]");
    }
    Obj& sa(const std::string& k, const std::vector<std::string>& v) {
        std::string body = "[";
        for (std::size_t j = 0; j < v.size(); ++j)
            body += (j ? ", " : "") + std::string("\"") + v[j] + "\"";
        return put(k, body + "]");
    }
    std::string str() const { return "{" + body_ + "}"; }

  private:
    Obj& put(const std::string& k, const std::string& v) {
        if (!body_.empty())
            body_ += ", ";
        body_ += "\"" + k + "\": " + v;
        return *this;
    }
    std::string body_;
};

std::vector<std::pair<std::string, std::string>> gCases;

void addCase(const std::string& name, const Obj& inputs, const Obj& expected) {
    gCases.emplace_back(name, "{\"inputs\": " + inputs.str() +
                                  ", \"expected\": " + expected.str() + "}");
}

void emitDocument() {
    std::cout << "{\n";
    for (std::size_t i = 0; i < gCases.size(); ++i)
        std::cout << "  \"" << gCases[i].first << "\": " << gCases[i].second
                  << (i + 1 < gCases.size() ? "," : "") << "\n";
    std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// Market. Evaluation date 3 March 2025 is a Monday, so TARGET().adjust() is
// the identity on it and no port can drift by a business-day roll.
// The FORWARDING curve (3%) and the DISCOUNT curve (2.5%) are deliberately
// different: BlackStyleSwaptionEngine re-prices the swap on the discount
// curve, so a port that forgets to do so gets a different atmForward.
// Both curves take 0 settlement days, so their reference date IS the
// evaluation date and there is no settlement-lag ambiguity.
// ---------------------------------------------------------------------------
const Date kToday(3, March, 2025);
const Rate kFwdRate = 0.03;
const Rate kDiscRate = 0.025;

const DayCounter& dcA365() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(
        kToday, r, dcA365(), Continuous, Annual));
}

const Handle<YieldTermStructure>& fwdCurve() {
    static const Handle<YieldTermStructure> h = flatCurve(kFwdRate);
    return h;
}

const Handle<YieldTermStructure>& discCurve() {
    static const Handle<YieldTermStructure> h = flatCurve(kDiscRate);
    return h;
}

const ext::shared_ptr<IborIndex>& euribor6m() {
    static const ext::shared_ptr<IborIndex> idx =
        ext::make_shared<Euribor>(6 * Months, fwdCurve());
    return idx;
}

// The swaption structure used everywhere below: 5y option into a 5y swap,
// annual 30/360 fixed vs semi-annual Act/360 Euribor6M float, TARGET,
// ModifiedFollowing, backward generation, notional 1.0.
Date swaptionExpiry() { return TARGET().advance(kToday, 5, Years); }
Date swapStart() { return TARGET().advance(swaptionExpiry(), 2, Days); }
Date swapEnd() { return TARGET().advance(swapStart(), 5, Years); }

Schedule fixedSchedule() {
    return Schedule(swapStart(), swapEnd(), 1 * Years, TARGET(), ModifiedFollowing,
                    ModifiedFollowing, DateGeneration::Backward, false);
}

Schedule floatSchedule() {
    return Schedule(swapStart(), swapEnd(), 6 * Months, TARGET(), ModifiedFollowing,
                    ModifiedFollowing, DateGeneration::Backward, false);
}

ext::shared_ptr<VanillaSwap> makeSwap(Swap::Type type, Rate fixedRate,
                                      Spread spread = 0.0) {
    return ext::make_shared<VanillaSwap>(type, 1.0, fixedSchedule(), fixedRate,
                                         Thirty360(Thirty360::BondBasis),
                                         floatSchedule(), euribor6m(), spread,
                                         Actual360());
}

ext::shared_ptr<Swaption> makeSwaption(
    Swap::Type type, Rate fixedRate, Spread spread = 0.0,
    Settlement::Type st = Settlement::Physical,
    Settlement::Method sm = Settlement::PhysicalOTC,
    const ext::shared_ptr<Exercise>& ex = nullptr) {
    return ext::make_shared<Swaption>(
        makeSwap(type, fixedRate, spread),
        ex ? ex : ext::shared_ptr<Exercise>(
                      ext::make_shared<EuropeanExercise>(swaptionExpiry())),
        st, sm);
}

Real ar(const Instrument& inst, const std::string& key) {
    return ext::any_cast<Real>(inst.additionalResults().at(key));
}

const char* typeName(Option::Type t) { return t == Option::Call ? "Call" : "Put"; }
const char* swapTypeName(Swap::Type t) { return t == Swap::Payer ? "Payer" : "Receiver"; }

Obj marketInputs() {
    Obj o;
    o.d("today", kToday)
        .s("calendar", "TARGET")
        .s("curve_day_counter", "Actual365Fixed")
        .s("curve_compounding", "Continuous")
        .s("curve_frequency", "Annual")
        .n("forwarding_rate", kFwdRate)
        .n("discount_rate", kDiscRate)
        .s("ibor_index", "Euribor6M")
        .d("exercise_date", swaptionExpiry())
        .d("swap_start", swapStart())
        .d("swap_end", swapEnd())
        .s("fixed_tenor", "1Y")
        .s("fixed_day_counter", "Thirty360(BondBasis)")
        .s("float_tenor", "6M")
        .s("float_day_counter", "Actual360")
        .s("convention", "ModifiedFollowing")
        .s("date_generation", "Backward")
        .b("end_of_month", false)
        .n("nominal", 1.0);
    return o;
}

// ===========================================================================
// Section A -- Black76Spec / BachelierSpec, standalone.
// ===========================================================================

struct SpecRow {
    const char* name;
    Option::Type type;
    Real strike;
    Real atmForward;
    Real stdDev;
    Real annuity;
    Real displacement;
    Real exerciseTime;
};

void emitSpecCases(const std::vector<SpecRow>& rows) {
    for (const SpecRow& r : rows) {
        for (int which = 0; which < 2; ++which) {
            const bool black = (which == 0);
            Obj in;
            in.s("spec", black ? "Black76Spec" : "BachelierSpec")
                .s("option_type", typeName(r.type))
                .n("strike", r.strike)
                .n("atm_forward", r.atmForward)
                .n("std_dev", r.stdDev)
                .n("annuity", r.annuity)
                .n("displacement", r.displacement)
                .n("exercise_time", r.exerciseTime);

            Obj ex;
            bool threw = false;
            Real value = 0.0, vega = 0.0, delta = 0.0;
            try {
                if (black) {
                    detail::Black76Spec spec;
                    value = spec.value(r.type, r.strike, r.atmForward, r.stdDev,
                                       r.annuity, r.displacement);
                    vega = spec.vega(r.strike, r.atmForward, r.stdDev,
                                     r.exerciseTime, r.annuity, r.displacement);
                    delta = spec.delta(r.type, r.strike, r.atmForward, r.stdDev,
                                       r.annuity, r.displacement);
                } else {
                    detail::BachelierSpec spec;
                    value = spec.value(r.type, r.strike, r.atmForward, r.stdDev,
                                       r.annuity, r.displacement);
                    vega = spec.vega(r.strike, r.atmForward, r.stdDev,
                                     r.exerciseTime, r.annuity, r.displacement);
                    delta = spec.delta(r.type, r.strike, r.atmForward, r.stdDev,
                                       r.annuity, r.displacement);
                }
            } catch (const std::exception&) {
                threw = true;
            }
            ex.b("throws", threw);
            if (!threw) {
                ex.n("value", value).n("vega", vega).n("delta", delta);
            }
            ex.s("volatility_type", black ? "ShiftedLognormal" : "Normal");
            addCase(std::string(black ? "spec_black76_" : "spec_bachelier_") + r.name,
                    in, ex);
        }
    }
}

// ===========================================================================
// Section B -- Black / Bachelier swaption engines.
// ===========================================================================

void emitSwaptionEngineCase(const std::string& name, Swap::Type swapType,
                            Rate fixedRate, Spread spread, bool black,
                            Volatility vol, Real displacement,
                            Settlement::Type st, Settlement::Method sm,
                            detail::BlackStyleSwaptionEngine<detail::Black76Spec>::
                                CashAnnuityModel model) {
    Obj in = marketInputs();
    in.s("engine", black ? "BlackSwaptionEngine" : "BachelierSwaptionEngine")
        .s("swap_type", swapTypeName(swapType))
        .n("fixed_rate", fixedRate)
        .n("float_spread", spread)
        .n("volatility", vol)
        .n("displacement", displacement)
        .s("vol_day_counter", "Actual365Fixed")
        .s("settlement_type", st == Settlement::Physical ? "Physical" : "Cash")
        .s("settlement_method",
           sm == Settlement::PhysicalOTC          ? "PhysicalOTC"
           : sm == Settlement::PhysicalCleared    ? "PhysicalCleared"
           : sm == Settlement::CollateralizedCashPrice
               ? "CollateralizedCashPrice"
               : "ParYieldCurve")
        .s("cash_annuity_model",
           model == BlackSwaptionEngine::DiscountCurve ? "DiscountCurve" : "SwapRate");

    Obj ex;
    try {
        auto swaption = makeSwaption(swapType, fixedRate, spread, st, sm);
        ext::shared_ptr<PricingEngine> engine;
        if (black)
            engine = ext::make_shared<BlackSwaptionEngine>(
                discCurve(), vol, dcA365(), displacement,
                static_cast<BlackSwaptionEngine::CashAnnuityModel>(model));
        else
            engine = ext::make_shared<BachelierSwaptionEngine>(
                discCurve(), vol, dcA365(),
                static_cast<BachelierSwaptionEngine::CashAnnuityModel>(model));
        swaption->setPricingEngine(engine);
        const Real npv = swaption->NPV();
        ex.b("throws", false)
            .n("npv", npv)
            .n("spread_correction", ar(*swaption, "spreadCorrection"))
            .n("strike", ar(*swaption, "strike"))
            .n("atm_forward", ar(*swaption, "atmForward"))
            .n("annuity", ar(*swaption, "annuity"))
            .n("swap_length", ar(*swaption, "swapLength"))
            .n("std_dev", ar(*swaption, "stdDev"))
            .n("vega", ar(*swaption, "vega"))
            .n("delta", ar(*swaption, "delta"))
            .n("time_to_expiry", ar(*swaption, "timeToExpiry"))
            .n("implied_volatility", ar(*swaption, "impliedVolatility"))
            .n("forward_price", ar(*swaption, "forwardPrice"))
            .d("valuation_date", swaption->valuationDate())
            .i("additional_results_count",
               static_cast<long long>(swaption->additionalResults().size()));
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// Engine built from an explicit ConstantSwaptionVolatility. `settlementDays`
// moves the vol structure's reference date away from the evaluation date,
// which is the only way to see that exerciseTime/stdDev come from the VOL
// structure and not from the discount curve.
void emitVolStructureCase(const std::string& name, Swap::Type swapType,
                          Rate fixedRate, bool black, Volatility vol,
                          Real shift, Natural settlementDays,
                          VolatilityType volType) {
    Obj in = marketInputs();
    in.s("engine", black ? "BlackSwaptionEngine" : "BachelierSwaptionEngine")
        .s("ctor", "SwaptionVolatilityStructure")
        .s("swap_type", swapTypeName(swapType))
        .n("fixed_rate", fixedRate)
        .n("volatility", vol)
        .n("shift", shift)
        .i("vol_settlement_days", static_cast<long long>(settlementDays))
        .s("vol_calendar", "TARGET")
        .s("vol_convention", "Following")
        .s("vol_day_counter", "Actual365Fixed")
        .s("vol_type", volType == ShiftedLognormal ? "ShiftedLognormal" : "Normal");

    Obj ex;
    try {
        auto volTs = ext::make_shared<ConstantSwaptionVolatility>(
            settlementDays, TARGET(), Following, vol, dcA365(), volType, shift);
        Handle<SwaptionVolatilityStructure> h(volTs);
        auto swaption = makeSwaption(swapType, fixedRate);
        ext::shared_ptr<PricingEngine> engine;
        if (black)
            engine = ext::make_shared<BlackSwaptionEngine>(discCurve(), h);
        else
            engine = ext::make_shared<BachelierSwaptionEngine>(discCurve(), h);
        swaption->setPricingEngine(engine);
        const Real npv = swaption->NPV();
        ex.b("throws", false)
            .n("npv", npv)
            .n("vol_reference_time_to_expiry", volTs->timeFromReference(swaptionExpiry()))
            .d("vol_reference_date", volTs->referenceDate())
            .n("strike", ar(*swaption, "strike"))
            .n("atm_forward", ar(*swaption, "atmForward"))
            .n("annuity", ar(*swaption, "annuity"))
            .n("swap_length", ar(*swaption, "swapLength"))
            .n("std_dev", ar(*swaption, "stdDev"))
            .n("vega", ar(*swaption, "vega"))
            .n("delta", ar(*swaption, "delta"))
            .n("time_to_expiry", ar(*swaption, "timeToExpiry"))
            .n("implied_volatility", ar(*swaption, "impliedVolatility"))
            .n("forward_price", ar(*swaption, "forwardPrice"));
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// ===========================================================================
// Section C/D -- Gaussian1d engines.
// ===========================================================================

const Real kGsrSigma = 0.01;
const Real kGsrReversion = 0.01;
const Real kGsrT = 60.0;

ext::shared_ptr<Gsr> makeGsr() {
    std::vector<Date> stepDates;
    std::vector<Real> vols(1, kGsrSigma);
    std::vector<Real> reversions(1, kGsrReversion);
    return ext::make_shared<Gsr>(fwdCurve(), stepDates, vols, reversions, kGsrT);
}

Obj gsrInputs() {
    Obj o = marketInputs();
    o.n("gsr_sigma", kGsrSigma)
        .n("gsr_reversion", kGsrReversion)
        .n("gsr_T", kGsrT)
        .s("gsr_curve", "forwarding");
    return o;
}

std::vector<Date> bermudanDates() {
    return {TARGET().advance(kToday, 5, Years), TARGET().advance(kToday, 6, Years),
            TARGET().advance(kToday, 7, Years)};
}

void emitG1dCase(const std::string& name, Swap::Type swapType, Rate fixedRate,
                 int integrationPoints, Real stddevs, bool extrapolatePayoff,
                 bool flatPayoffExtrapolation,
                 Gaussian1dSwaptionEngine::Probabilities probabilities,
                 bool bermudan, bool useDiscountCurveOverride,
                 Settlement::Type st = Settlement::Physical,
                 Settlement::Method sm = Settlement::PhysicalOTC) {
    Obj in = gsrInputs();
    in.s("engine", "Gaussian1dSwaptionEngine")
        .s("swap_type", swapTypeName(swapType))
        .n("fixed_rate", fixedRate)
        .i("integration_points", integrationPoints)
        .n("stddevs", stddevs)
        .b("extrapolate_payoff", extrapolatePayoff)
        .b("flat_payoff_extrapolation", flatPayoffExtrapolation)
        .s("probabilities", probabilities == Gaussian1dSwaptionEngine::None ? "None"
                            : probabilities == Gaussian1dSwaptionEngine::Naive
                                ? "Naive"
                                : "Digital")
        .b("bermudan", bermudan)
        .b("discount_curve_override", useDiscountCurveOverride)
        .s("settlement_type", st == Settlement::Physical ? "Physical" : "Cash")
        .s("settlement_method",
           sm == Settlement::PhysicalOTC          ? "PhysicalOTC"
           : sm == Settlement::PhysicalCleared    ? "PhysicalCleared"
           : sm == Settlement::CollateralizedCashPrice
               ? "CollateralizedCashPrice"
               : "ParYieldCurve");
    if (bermudan)
        in.da("exercise_dates", bermudanDates());

    Obj ex;
    try {
        ext::shared_ptr<Exercise> exercise;
        if (bermudan)
            exercise = ext::make_shared<BermudanExercise>(bermudanDates());
        else
            exercise = ext::make_shared<EuropeanExercise>(swaptionExpiry());
        auto swaption = makeSwaption(swapType, fixedRate, 0.0, st, sm, exercise);
        auto engine = ext::make_shared<Gaussian1dSwaptionEngine>(
            makeGsr(), integrationPoints, stddevs, extrapolatePayoff,
            flatPayoffExtrapolation,
            useDiscountCurveOverride ? discCurve() : Handle<YieldTermStructure>(),
            probabilities);
        swaption->setPricingEngine(engine);
        const Real npv = swaption->NPV();
        ex.b("throws", false).n("npv", npv);
        if (probabilities != Gaussian1dSwaptionEngine::None) {
            ex.a("probabilities",
                 ext::any_cast<std::vector<Real>>(
                     swaption->additionalResults().at("probabilities")));
        }
        ex.i("additional_results_count",
             static_cast<long long>(swaption->additionalResults().size()));
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

void emitJamshidianCase(const std::string& name, Swap::Type swapType,
                        Rate fixedRate, Spread spread, bool bermudan,
                        Settlement::Type st = Settlement::Physical,
                        Settlement::Method sm = Settlement::PhysicalOTC) {
    Obj in = gsrInputs();
    in.s("engine", "Gaussian1dJamshidianSwaptionEngine")
        .s("swap_type", swapTypeName(swapType))
        .n("fixed_rate", fixedRate)
        .n("float_spread", spread)
        .b("bermudan", bermudan)
        .s("settlement_type", st == Settlement::Physical ? "Physical" : "Cash")
        .s("settlement_method",
           sm == Settlement::PhysicalOTC          ? "PhysicalOTC"
           : sm == Settlement::PhysicalCleared    ? "PhysicalCleared"
           : sm == Settlement::CollateralizedCashPrice
               ? "CollateralizedCashPrice"
               : "ParYieldCurve");
    if (bermudan)
        in.da("exercise_dates", bermudanDates());

    Obj ex;
    try {
        ext::shared_ptr<Exercise> exercise;
        if (bermudan)
            exercise = ext::make_shared<BermudanExercise>(bermudanDates());
        else
            exercise = ext::make_shared<EuropeanExercise>(swaptionExpiry());
        auto swaption = makeSwaption(swapType, fixedRate, spread, st, sm, exercise);
        swaption->setPricingEngine(
            ext::make_shared<Gaussian1dJamshidianSwaptionEngine>(makeGsr()));
        ex.b("throws", false).n("npv", swaption->NPV());
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// ===========================================================================
// Section E -- MC Hull-White cap/floor.
// ===========================================================================

const Real kHwA = 0.05;
const Real kHwSigma = 0.01;
const Rate kCapStrike = 0.03;

ext::shared_ptr<HullWhite> makeHullWhite() {
    return ext::make_shared<HullWhite>(fwdCurve(), kHwA, kHwSigma);
}

// 5y cap/floor on Euribor6M starting at the index's spot date, notional 1.
Leg capFloorLeg() {
    const Date start = euribor6m()->fixingCalendar().advance(
        kToday, euribor6m()->fixingDays(), Days);
    const Date end = TARGET().advance(start, 5, Years);
    const Schedule sched(start, end, 6 * Months, TARGET(), ModifiedFollowing,
                         ModifiedFollowing, DateGeneration::Backward, false);
    return IborLeg(sched, euribor6m())
        .withNotionals(1.0)
        .withPaymentDayCounter(Actual360())
        .withPaymentAdjustment(ModifiedFollowing)
        .withFixingDays(euribor6m()->fixingDays());
}

ext::shared_ptr<CapFloor> makeCapFloor(bool isCap) {
    if (isCap)
        return ext::make_shared<Cap>(capFloorLeg(), std::vector<Rate>(1, kCapStrike));
    return ext::make_shared<Floor>(capFloorLeg(), std::vector<Rate>(1, kCapStrike));
}

Obj mcInputs(bool isCap) {
    Obj o;
    const Date start = euribor6m()->fixingCalendar().advance(
        kToday, euribor6m()->fixingDays(), Days);
    o.d("today", kToday)
        .s("calendar", "TARGET")
        .s("curve_day_counter", "Actual365Fixed")
        .s("curve_compounding", "Continuous")
        .s("curve_frequency", "Annual")
        .n("forwarding_rate", kFwdRate)
        .s("ibor_index", "Euribor6M")
        .n("hw_a", kHwA)
        .n("hw_sigma", kHwSigma)
        .s("instrument", isCap ? "Cap" : "Floor")
        .n("strike", kCapStrike)
        .d("leg_start", start)
        .d("leg_end", TARGET().advance(start, 5, Years))
        .s("leg_tenor", "6M")
        .s("leg_day_counter", "Actual360")
        .s("convention", "ModifiedFollowing")
        .s("date_generation", "Backward")
        .n("nominal", 1.0);
    return o;
}

// The engine's TimeGrid: future fixing times plus the last end date, as
// computed by MCHullWhiteCapFloorEngine::timeGrid().
TimeGrid mcTimeGrid(const CapFloor::arguments& args) {
    const Date referenceDate = fwdCurve()->referenceDate();
    const DayCounter dayCounter = fwdCurve()->dayCounter();
    std::vector<Time> times;
    for (auto fixingDate : args.fixingDates)
        if (fixingDate > referenceDate)
            times.push_back(dayCounter.yearFraction(referenceDate, fixingDate));
    times.push_back(dayCounter.yearFraction(referenceDate, args.endDates.back()));
    return TimeGrid(times.begin(), times.end());
}

void emitMcCase(const std::string& name, bool isCap, bool brownianBridge,
                bool antithetic, Size samples, BigNatural seed) {
    Obj in = mcInputs(isCap);
    in.s("engine", "MCHullWhiteCapFloorEngine")
        .b("brownian_bridge", brownianBridge)
        .b("antithetic_variate", antithetic)
        .i("required_samples", static_cast<long long>(samples))
        .i("seed", static_cast<long long>(seed));

    Obj ex;
    try {
        auto instrument = makeCapFloor(isCap);
        instrument->setPricingEngine(
            MakeMCHullWhiteCapFloorEngine<PseudoRandom>(makeHullWhite())
                .withBrownianBridge(brownianBridge)
                .withAntitheticVariate(antithetic)
                .withSamples(samples)
                .withSeed(seed));
        ex.b("throws", false)
            .n("npv", instrument->NPV())
            .n("error_estimate", instrument->errorEstimate());
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    // =======================================================================
    // Section A -- Black76Spec / BachelierSpec, standalone.
    // Forward 3.5%, annuity 4.3, exerciseTime 5, stdDev 0.20*sqrt(5) unless
    // stated otherwise. `disp01` rows use displacement 1%; on Bachelier those
    // rows MUST return the same number as the disp0 rows, because
    // BachelierSpec ignores its last argument.
    // =======================================================================
    const Real sd = 0.20 * std::sqrt(5.0);
    emitSpecCases({
        // ITM / ATM / OTM, positive rates, no displacement.
        {"call_itm", Option::Call, 0.030, 0.035, sd, 4.3, 0.0, 5.0},
        {"put_itm", Option::Put, 0.040, 0.035, sd, 4.3, 0.0, 5.0},
        {"call_atm", Option::Call, 0.035, 0.035, sd, 4.3, 0.0, 5.0},
        {"put_atm", Option::Put, 0.035, 0.035, sd, 4.3, 0.0, 5.0},
        {"call_otm", Option::Call, 0.060, 0.035, sd, 4.3, 0.0, 5.0},
        {"put_otm", Option::Put, 0.010, 0.035, sd, 4.3, 0.0, 5.0},
        // stdDev == 0: value collapses to intrinsic, vega to 0, and delta to
        // the sign branch. The ATM rows hit boost::math::sign(0) == 0.
        {"call_stddev0_itm", Option::Call, 0.030, 0.035, 0.0, 4.3, 0.0, 5.0},
        {"call_stddev0_otm", Option::Call, 0.060, 0.035, 0.0, 4.3, 0.0, 5.0},
        {"call_stddev0_atm", Option::Call, 0.035, 0.035, 0.0, 4.3, 0.0, 5.0},
        {"put_stddev0_atm", Option::Put, 0.035, 0.035, 0.0, 4.3, 0.0, 5.0},
        // strike exactly 0 with no displacement: blackFormulaForwardDerivative
        // early-returns discount (Call) / 0 (Put).
        {"call_strike0", Option::Call, 0.0, 0.035, sd, 4.3, 0.0, 5.0},
        {"put_strike0", Option::Put, 0.0, 0.035, sd, 4.3, 0.0, 5.0},
        // Displacement 1%: legal for Black even with a NEGATIVE forward/strike,
        // and silently ignored by Bachelier.
        {"call_disp01", Option::Call, 0.030, 0.035, sd, 4.3, 0.01, 5.0},
        {"put_disp01", Option::Put, 0.040, 0.035, sd, 4.3, 0.01, 5.0},
        {"call_neg_disp01", Option::Call, -0.002, -0.005, sd, 4.3, 0.01, 5.0},
        {"put_neg_disp01", Option::Put, -0.002, -0.005, sd, 4.3, 0.01, 5.0},
        // Negative rates with NO displacement: legal for Bachelier, throws for
        // Black (checkParameters requires strike + displacement >= 0).
        {"call_neg_disp0", Option::Call, -0.002, -0.005, sd, 4.3, 0.0, 5.0},
        {"put_neg_disp0", Option::Put, -0.002, -0.005, sd, 4.3, 0.0, 5.0},
        // A different exerciseTime scales vega by sqrt(t) only.
        {"call_itm_t1", Option::Call, 0.030, 0.035, sd, 4.3, 0.0, 1.0},
        {"call_itm_t25", Option::Call, 0.030, 0.035, sd, 4.3, 0.0, 25.0},
        // Larger stdDev, small annuity.
        {"call_hivol", Option::Call, 0.030, 0.035, 1.5, 1.0, 0.0, 5.0},
        {"put_hivol", Option::Put, 0.030, 0.035, 1.5, 1.0, 0.0, 5.0},
    });

    // =======================================================================
    // Section B -- Black / Bachelier swaption engines.
    // =======================================================================
    using CAM = BlackSwaptionEngine::CashAnnuityModel;

    // Diagnostic: what the engine derives before it reaches the Spec. Pinning
    // this makes a wrong NPV bisectable into market-setup vs formula.
    {
        auto swap = makeSwap(Swap::Payer, 0.03);
        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(discCurve(), false));
        Obj ex;
        ex.n("fair_rate", swap->fairRate())
            .n("fixed_leg_bps", swap->fixedLegBPS())
            .n("floating_leg_bps", swap->floatingLegBPS())
            .n("fixed_leg_npv", swap->fixedLegNPV())
            .n("floating_leg_npv", swap->floatingLegNPV())
            .n("npv", swap->NPV())
            .d("valuation_date", swap->valuationDate())
            .da("fixed_dates", fixedSchedule().dates())
            .da("float_dates", floatSchedule().dates())
            .n("discount_at_exercise", discCurve()->discount(swaptionExpiry()));
        addCase("swap_derived_inputs", marketInputs(), ex);
    }

    // Payer/receiver x ITM/ATM/OTM, Black flat 20% lognormal, physical.
    for (int p = 0; p < 2; ++p) {
        const Swap::Type st = p == 0 ? Swap::Payer : Swap::Receiver;
        const char* tag = p == 0 ? "payer" : "receiver";
        emitSwaptionEngineCase(std::string("black_") + tag + "_k020", st, 0.020, 0.0,
                               true, 0.20, 0.0, Settlement::Physical,
                               Settlement::PhysicalOTC, CAM::DiscountCurve);
        emitSwaptionEngineCase(std::string("black_") + tag + "_k030", st, 0.030, 0.0,
                               true, 0.20, 0.0, Settlement::Physical,
                               Settlement::PhysicalOTC, CAM::DiscountCurve);
        emitSwaptionEngineCase(std::string("black_") + tag + "_k045", st, 0.045, 0.0,
                               true, 0.20, 0.0, Settlement::Physical,
                               Settlement::PhysicalOTC, CAM::DiscountCurve);
        emitSwaptionEngineCase(std::string("bachelier_") + tag + "_k020", st, 0.020,
                               0.0, false, 0.0060, 0.0, Settlement::Physical,
                               Settlement::PhysicalOTC, CAM::DiscountCurve);
        emitSwaptionEngineCase(std::string("bachelier_") + tag + "_k030", st, 0.030,
                               0.0, false, 0.0060, 0.0, Settlement::Physical,
                               Settlement::PhysicalOTC, CAM::DiscountCurve);
        emitSwaptionEngineCase(std::string("bachelier_") + tag + "_k045", st, 0.045,
                               0.0, false, 0.0060, 0.0, Settlement::Physical,
                               Settlement::PhysicalOTC, CAM::DiscountCurve);
    }

    // Shifted lognormal (displacement 1%), including a NEGATIVE strike which
    // is only priceable because of the shift.
    emitSwaptionEngineCase("black_payer_k030_disp01", Swap::Payer, 0.030, 0.0, true,
                           0.20, 0.01, Settlement::Physical,
                           Settlement::PhysicalOTC, CAM::DiscountCurve);
    emitSwaptionEngineCase("black_payer_km0005_disp01", Swap::Payer, -0.0005, 0.0,
                           true, 0.20, 0.01, Settlement::Physical,
                           Settlement::PhysicalOTC, CAM::DiscountCurve);
    emitSwaptionEngineCase("bachelier_payer_km0005", Swap::Payer, -0.0005, 0.0, false,
                           0.0060, 0.0, Settlement::Physical,
                           Settlement::PhysicalOTC, CAM::DiscountCurve);
    // Black with a negative strike and NO shift must throw.
    emitSwaptionEngineCase("black_payer_km0005_disp0_throws", Swap::Payer, -0.0005,
                           0.0, true, 0.20, 0.0, Settlement::Physical,
                           Settlement::PhysicalOTC, CAM::DiscountCurve);

    // Non-zero floating spread -> spreadCorrection path.
    emitSwaptionEngineCase("black_payer_k030_spread25bp", Swap::Payer, 0.030, 0.0025,
                           true, 0.20, 0.0, Settlement::Physical,
                           Settlement::PhysicalOTC, CAM::DiscountCurve);
    emitSwaptionEngineCase("black_receiver_k030_spread_m25bp", Swap::Receiver, 0.030,
                           -0.0025, true, 0.20, 0.0, Settlement::Physical,
                           Settlement::PhysicalOTC, CAM::DiscountCurve);

    // Settlement variants. CollateralizedCashPrice reuses the physical annuity;
    // ParYieldCurve takes the CashFlows::bps branch and DIFFERS between the two
    // CashAnnuityModel values.
    emitSwaptionEngineCase("black_payer_k030_cash_collateralized", Swap::Payer, 0.030,
                           0.0, true, 0.20, 0.0, Settlement::Cash,
                           Settlement::CollateralizedCashPrice, CAM::DiscountCurve);
    emitSwaptionEngineCase("black_payer_k030_cash_paryield_disccurve", Swap::Payer,
                           0.030, 0.0, true, 0.20, 0.0, Settlement::Cash,
                           Settlement::ParYieldCurve, CAM::DiscountCurve);
    emitSwaptionEngineCase("black_payer_k030_cash_paryield_swaprate", Swap::Payer,
                           0.030, 0.0, true, 0.20, 0.0, Settlement::Cash,
                           Settlement::ParYieldCurve, CAM::SwapRate);
    emitSwaptionEngineCase("bachelier_payer_k030_cash_paryield_disccurve", Swap::Payer,
                           0.030, 0.0, false, 0.0060, 0.0, Settlement::Cash,
                           Settlement::ParYieldCurve, CAM::DiscountCurve);
    emitSwaptionEngineCase("black_payer_k030_physical_cleared", Swap::Payer, 0.030,
                           0.0, true, 0.20, 0.0, Settlement::Physical,
                           Settlement::PhysicalCleared, CAM::DiscountCurve);

    // Zero volatility -> intrinsic. delta collapses to the sign branch.
    emitSwaptionEngineCase("black_payer_k030_vol0", Swap::Payer, 0.030, 0.0, true,
                           0.0, 0.0, Settlement::Physical, Settlement::PhysicalOTC,
                           CAM::DiscountCurve);
    emitSwaptionEngineCase("bachelier_payer_k030_vol0", Swap::Payer, 0.030, 0.0,
                           false, 0.0, 0.0, Settlement::Physical,
                           Settlement::PhysicalOTC, CAM::DiscountCurve);

    // SwaptionVolatilityStructure-driven ctors. settlementDays 0 vs 2 moves the
    // vol reference date; the second block is where a port that reads the
    // discount curve's reference date breaks.
    emitVolStructureCase("black_volts_settle0_payer", Swap::Payer, 0.030, true, 0.20,
                         0.0, 0, ShiftedLognormal);
    emitVolStructureCase("black_volts_settle2_payer", Swap::Payer, 0.030, true, 0.20,
                         0.0, 2, ShiftedLognormal);
    emitVolStructureCase("black_volts_settle2_shift01_payer", Swap::Payer, 0.030,
                         true, 0.20, 0.01, 2, ShiftedLognormal);
    emitVolStructureCase("bachelier_volts_settle2_payer", Swap::Payer, 0.030, false,
                         0.0060, 0.0, 2, Normal);
    // Wrong volatility type for the concrete engine -> ctor throws.
    emitVolStructureCase("black_volts_normal_throws", Swap::Payer, 0.030, true, 0.0060,
                         0.0, 0, Normal);
    emitVolStructureCase("bachelier_volts_lognormal_throws", Swap::Payer, 0.030, false,
                         0.20, 0.0, 0, ShiftedLognormal);

    // Guards: Bermudan exercise, and a swap that starts BEFORE the exercise.
    {
        Obj in = marketInputs();
        in.s("engine", "BlackSwaptionEngine").s("exercise", "Bermudan")
            .da("exercise_dates", bermudanDates());
        Obj ex;
        try {
            auto swaption = makeSwaption(
                Swap::Payer, 0.03, 0.0, Settlement::Physical,
                Settlement::PhysicalOTC,
                ext::make_shared<BermudanExercise>(bermudanDates()));
            swaption->setPricingEngine(ext::make_shared<BlackSwaptionEngine>(
                discCurve(), 0.20, dcA365(), 0.0));
            static_cast<void>(swaption->NPV());
            ex.b("throws", false);
        } catch (const std::exception&) {
            ex.b("throws", true);
        }
        addCase("black_bermudan_throws", in, ex);
    }
    {
        // Exercise one year AFTER the swap starts -> QL_REQUIRE fires.
        const Date lateExercise = TARGET().advance(swapStart(), 1, Years);
        Obj in = marketInputs();
        in.s("engine", "BlackSwaptionEngine").d("exercise_date_override", lateExercise);
        Obj ex;
        try {
            auto swaption = makeSwaption(
                Swap::Payer, 0.03, 0.0, Settlement::Physical,
                Settlement::PhysicalOTC,
                ext::make_shared<EuropeanExercise>(lateExercise));
            swaption->setPricingEngine(ext::make_shared<BlackSwaptionEngine>(
                discCurve(), 0.20, dcA365(), 0.0));
            static_cast<void>(swaption->NPV());
            ex.b("throws", false);
        } catch (const std::exception&) {
            ex.b("throws", true);
        }
        addCase("black_swap_starts_before_exercise_throws", in, ex);
    }

    // =======================================================================
    // Section C -- Gaussian1dSwaptionEngine.
    // =======================================================================
    using Prob = Gaussian1dSwaptionEngine::Probabilities;

    for (int p = 0; p < 2; ++p) {
        const Swap::Type st = p == 0 ? Swap::Payer : Swap::Receiver;
        const char* tag = p == 0 ? "payer" : "receiver";
        emitG1dCase(std::string("g1d_") + tag + "_k030_n64_s7", st, 0.030, 64, 7.0,
                    true, false, Prob::None, false, false);
        emitG1dCase(std::string("g1d_") + tag + "_k020_n64_s7", st, 0.020, 64, 7.0,
                    true, false, Prob::None, false, false);
        emitG1dCase(std::string("g1d_") + tag + "_k045_n64_s7", st, 0.045, 64, 7.0,
                    true, false, Prob::None, false, false);
    }
    // Knob sweeps. A port that hardcodes 64/7.0 fails these.
    emitG1dCase("g1d_payer_k030_n16_s7", Swap::Payer, 0.030, 16, 7.0, true, false,
                Prob::None, false, false);
    emitG1dCase("g1d_payer_k030_n32_s7", Swap::Payer, 0.030, 32, 7.0, true, false,
                Prob::None, false, false);
    emitG1dCase("g1d_payer_k030_n64_s4", Swap::Payer, 0.030, 64, 4.0, true, false,
                Prob::None, false, false);
    emitG1dCase("g1d_payer_k030_n64_s2", Swap::Payer, 0.030, 64, 2.0, true, false,
                Prob::None, false, false);
    // Tail-extrapolation knobs, for both option types (the non-flat branch is
    // type-dependent: Call extends the upper tail, Put the lower).
    emitG1dCase("g1d_payer_k030_noextrap", Swap::Payer, 0.030, 64, 7.0, false, false,
                Prob::None, false, false);
    emitG1dCase("g1d_payer_k030_flatextrap", Swap::Payer, 0.030, 64, 7.0, true, true,
                Prob::None, false, false);
    emitG1dCase("g1d_receiver_k030_noextrap", Swap::Receiver, 0.030, 64, 7.0, false,
                false, Prob::None, false, false);
    emitG1dCase("g1d_receiver_k030_flatextrap", Swap::Receiver, 0.030, 64, 7.0, true,
                true, Prob::None, false, false);
    // With a small stddevs the tail treatment matters much more.
    emitG1dCase("g1d_payer_k030_n64_s2_noextrap", Swap::Payer, 0.030, 64, 2.0, false,
                false, Prob::None, false, false);
    emitG1dCase("g1d_payer_k030_n64_s2_flatextrap", Swap::Payer, 0.030, 64, 2.0, true,
                true, Prob::None, false, false);
    // Discount-curve override (2.5% instead of the model's own 3%).
    emitG1dCase("g1d_payer_k030_disc_override", Swap::Payer, 0.030, 64, 7.0, true,
                false, Prob::None, false, true);
    // Bermudan, with and without probabilities.
    emitG1dCase("g1d_bermudan_payer_k030", Swap::Payer, 0.030, 64, 7.0, true, false,
                Prob::None, true, false);
    emitG1dCase("g1d_bermudan_receiver_k030", Swap::Receiver, 0.030, 64, 7.0, true,
                false, Prob::None, true, false);
    emitG1dCase("g1d_bermudan_payer_k030_prob_naive", Swap::Payer, 0.030, 64, 7.0,
                true, false, Prob::Naive, true, false);
    emitG1dCase("g1d_bermudan_payer_k030_prob_digital", Swap::Payer, 0.030, 64, 7.0,
                true, false, Prob::Digital, true, false);
    emitG1dCase("g1d_european_payer_k030_prob_naive", Swap::Payer, 0.030, 64, 7.0,
                true, false, Prob::Naive, false, false);
    // Cash settlement: CollateralizedCashPrice is allowed, ParYieldCurve throws.
    emitG1dCase("g1d_payer_k030_cash_collateralized", Swap::Payer, 0.030, 64, 7.0,
                true, false, Prob::None, false, false, Settlement::Cash,
                Settlement::CollateralizedCashPrice);
    emitG1dCase("g1d_payer_k030_cash_paryield_throws", Swap::Payer, 0.030, 64, 7.0,
                true, false, Prob::None, false, false, Settlement::Cash,
                Settlement::ParYieldCurve);

    // Expired swaption: last exercise date <= model reference date -> 0.0.
    {
        Obj in = gsrInputs();
        in.s("engine", "Gaussian1dSwaptionEngine")
            .s("note", "exercise date == today, so exercise->dates().back() <= settlement")
            .d("exercise_date_override", kToday);
        Obj ex;
        try {
            auto swap = ext::make_shared<VanillaSwap>(
                Swap::Payer, 1.0, fixedSchedule(), 0.03,
                Thirty360(Thirty360::BondBasis), floatSchedule(), euribor6m(), 0.0,
                Actual360());
            auto swaption = ext::make_shared<Swaption>(
                swap, ext::shared_ptr<Exercise>(
                          ext::make_shared<EuropeanExercise>(kToday)));
            swaption->setPricingEngine(
                ext::make_shared<Gaussian1dSwaptionEngine>(makeGsr(), 64, 7.0));
            ex.b("throws", false).n("npv", swaption->NPV());
        } catch (const std::exception&) {
            ex.b("throws", true);
        }
        addCase("g1d_expired", in, ex);
    }

    // =======================================================================
    // Section D -- Gaussian1dJamshidianSwaptionEngine, and the cross-check.
    // =======================================================================
    emitJamshidianCase("jam_payer_k030", Swap::Payer, 0.030, 0.0, false);
    emitJamshidianCase("jam_receiver_k030", Swap::Receiver, 0.030, 0.0, false);
    emitJamshidianCase("jam_payer_k020", Swap::Payer, 0.020, 0.0, false);
    emitJamshidianCase("jam_payer_k045", Swap::Payer, 0.045, 0.0, false);
    emitJamshidianCase("jam_receiver_k045", Swap::Receiver, 0.045, 0.0, false);
    emitJamshidianCase("jam_bermudan_throws", Swap::Payer, 0.030, 0.0, true);
    emitJamshidianCase("jam_spread_throws", Swap::Payer, 0.030, 0.0025, false);
    emitJamshidianCase("jam_cash_paryield_throws", Swap::Payer, 0.030, 0.0, false,
                       Settlement::Cash, Settlement::ParYieldCurve);

    // Head-to-head on the same European swaption. These MUST agree to the
    // integration tolerance; the pinned difference documents how close.
    for (int k = 0; k < 3; ++k) {
        const Rate rate = k == 0 ? 0.020 : k == 1 ? 0.030 : 0.045;
        for (int p = 0; p < 2; ++p) {
            const Swap::Type st = p == 0 ? Swap::Payer : Swap::Receiver;
            auto exercise = ext::shared_ptr<Exercise>(
                ext::make_shared<EuropeanExercise>(swaptionExpiry()));
            auto s1 = makeSwaption(st, rate, 0.0, Settlement::Physical,
                                   Settlement::PhysicalOTC, exercise);
            s1->setPricingEngine(ext::make_shared<Gaussian1dSwaptionEngine>(
                makeGsr(), 64, 7.0, true, false));
            const Real integ = s1->NPV();
            auto s2 = makeSwaption(st, rate, 0.0, Settlement::Physical,
                                   Settlement::PhysicalOTC, exercise);
            s2->setPricingEngine(
                ext::make_shared<Gaussian1dJamshidianSwaptionEngine>(makeGsr()));
            const Real jam = s2->NPV();

            Obj in = gsrInputs();
            in.s("swap_type", swapTypeName(st)).n("fixed_rate", rate)
                .i("integration_points", 64).n("stddevs", 7.0);
            Obj ex;
            ex.n("integration_npv", integ).n("jamshidian_npv", jam)
                .n("abs_difference", std::fabs(integ - jam));
            std::ostringstream nm;
            nm << "g1d_vs_jamshidian_" << (p == 0 ? "payer" : "receiver") << "_k"
               << static_cast<int>(rate * 1000.0 + 0.5);
            addCase(nm.str(), in, ex);
        }
    }

    // =======================================================================
    // Section E -- MC Hull-White cap/floor.
    // =======================================================================

    // E.0 -- the raw PseudoRandom stream. If these fail nothing downstream can
    // match, so they come first.
    for (BigNatural seed : {BigNatural(42), BigNatural(12345)}) {
        const Size dim = 4;
        PseudoRandom::rsg_type rsg = PseudoRandom::make_sequence_generator(dim, seed);
        std::vector<Real> flat;
        for (int draw = 0; draw < 3; ++draw) {
            const auto& sample = rsg.nextSequence();
            for (Real v : sample.value)
                flat.push_back(v);
        }
        Obj in;
        in.s("generator", "PseudoRandom::make_sequence_generator")
            .i("dimension", static_cast<long long>(dim))
            .i("seed", static_cast<long long>(seed))
            .i("draws", 3);
        Obj ex;
        ex.a("values", flat);
        addCase(std::string("mc_rng_seed") + std::to_string(seed), in, ex);
    }

    // E.1 -- the engine's TimeGrid and the cap's argument vectors.
    {
        auto cap = makeCapFloor(true);
        CapFloor::arguments args;
        cap->setupArguments(&args);
        const TimeGrid grid = mcTimeGrid(args);
        std::vector<Real> times(grid.begin(), grid.end());

        const Date referenceDate = fwdCurve()->referenceDate();
        const DayCounter dc = fwdCurve()->dayCounter();
        std::vector<Real> startTimes, endTimes, fixingTimes;
        for (const auto& d : args.startDates)
            startTimes.push_back(dc.yearFraction(referenceDate, d));
        for (const auto& d : args.endDates)
            endTimes.push_back(dc.yearFraction(referenceDate, d));
        for (const auto& d : args.fixingDates)
            fixingTimes.push_back(dc.yearFraction(referenceDate, d));

        Obj ex;
        ex.a("time_grid", times)
            .i("time_grid_size", static_cast<long long>(times.size()))
            .da("start_dates", args.startDates)
            .da("end_dates", args.endDates)
            .da("fixing_dates", args.fixingDates)
            .a("start_times", startTimes)
            .a("end_times", endTimes)
            .a("fixing_times", fixingTimes)
            .a("accrual_times", args.accrualTimes)
            .a("forwards", args.forwards)
            .a("cap_rates", args.capRates)
            .a("nominals", args.nominals)
            .a("gearings", args.gearings)
            .n("forward_measure_time",
               dc.yearFraction(referenceDate, args.endDates.back()))
            .n("end_discount",
               fwdCurve()->discount(dc.yearFraction(referenceDate,
                                                    args.endDates.back())));
        addCase("mc_time_grid", mcInputs(true), ex);
    }

    // E.2 -- the first two generated paths, exactly as the engine builds them.
    for (BigNatural seed : {BigNatural(42), BigNatural(12345)}) {
        auto cap = makeCapFloor(true);
        CapFloor::arguments args;
        cap->setupArguments(&args);
        const TimeGrid grid = mcTimeGrid(args);
        const Date referenceDate = fwdCurve()->referenceDate();
        const DayCounter dc = fwdCurve()->dayCounter();
        const Time forwardMeasureTime =
            dc.yearFraction(referenceDate, args.endDates.back());
        auto process = ext::make_shared<HullWhiteForwardProcess>(fwdCurve(), kHwA,
                                                                 kHwSigma);
        process->setForwardMeasureTime(forwardMeasureTime);
        PseudoRandom::rsg_type generator =
            PseudoRandom::make_sequence_generator(grid.size() - 1, seed);
        PathGenerator<PseudoRandom::rsg_type> pathGen(process, grid, generator, false);

        std::vector<Real> path0, path1, anti0;
        {
            const auto& s = pathGen.next();
            for (Size i = 0; i < s.value.length(); ++i)
                path0.push_back(s.value[i]);
        }
        {
            const auto& s = pathGen.antithetic();
            for (Size i = 0; i < s.value.length(); ++i)
                anti0.push_back(s.value[i]);
        }
        {
            const auto& s = pathGen.next();
            for (Size i = 0; i < s.value.length(); ++i)
                path1.push_back(s.value[i]);
        }
        Obj in = mcInputs(true);
        in.i("seed", static_cast<long long>(seed)).b("brownian_bridge", false);
        Obj ex;
        ex.a("path0", path0).a("antithetic0", anti0).a("path1", path1);
        addCase(std::string("mc_path_seed") + std::to_string(seed), in, ex);
    }

    // E.3 -- HullWhiteCapFloorPricer against hand-built paths. This isolates
    // the payoff from the RNG entirely.
    for (int isCapI = 0; isCapI < 2; ++isCapI) {
        const bool isCap = isCapI == 0;
        auto instrument = makeCapFloor(isCap);
        CapFloor::arguments args;
        instrument->setupArguments(&args);
        const TimeGrid grid = mcTimeGrid(args);
        const Date referenceDate = fwdCurve()->referenceDate();
        const DayCounter dc = fwdCurve()->dayCounter();
        const Time forwardMeasureTime =
            dc.yearFraction(referenceDate, args.endDates.back());
        detail::HullWhiteCapFloorPricer pricer(args, makeHullWhite(),
                                               forwardMeasureTime);

        // Deterministic short-rate paths centred on the curve's instantaneous
        // forward (3%), so BOTH the cap and the floor produce non-zero payoffs.
        // Note the FIRST caplet of this leg fixes exactly on the reference date
        // (fixing time == 0.0), so the pricer takes its "current caplet"
        // branch: pastFixings becomes 1 immediately, currentLibor is read from
        // args_.forwards[0] and the path is indexed at [i] / [i+1] rather than
        // [i+1] / [i+2] thereafter. That off-by-one is the single easiest thing
        // to get wrong in a port, which is why the ramps are asymmetric --
        // a shifted index gives a visibly different number.
        const Size n = grid.size();
        const char* variantNames[] = {"flat_3pc", "ramp_up_40bp", "ramp_down_40bp",
                                      "zigzag"};
        for (int variant = 0; variant < 4; ++variant) {
            Array values(n);
            for (Size i = 0; i < n; ++i) {
                const Real k = static_cast<Real>(i);
                if (variant == 0)
                    values[i] = 0.03;
                else if (variant == 1)
                    values[i] = 0.03 + 0.004 * k;
                else if (variant == 2)
                    values[i] = 0.03 - 0.004 * k;
                else
                    values[i] = 0.03 + ((i % 2 == 0) ? 0.015 : -0.015);
            }
            Path path(grid, values);
            std::vector<Real> pathVals;
            for (Size i = 0; i < n; ++i)
                pathVals.push_back(values[i]);

            Obj in = mcInputs(isCap);
            in.s("path_variant", variantNames[variant])
                .a("path_values", pathVals)
                .n("forward_measure_time", forwardMeasureTime);
            Obj ex;
            ex.n("payoff", pricer(path));
            std::ostringstream nm;
            nm << "mc_pathpricer_" << (isCap ? "cap" : "floor") << "_v" << variant;
            addCase(nm.str(), in, ex);
        }
    }

    // E.4 -- full engine runs. Everything pinned exactly.
    emitMcCase("mc_cap_n1023_seed42", true, false, false, 1023, 42);
    emitMcCase("mc_cap_n4095_seed42", true, false, false, 4095, 42);
    emitMcCase("mc_cap_n1023_seed12345", true, false, false, 1023, 12345);
    emitMcCase("mc_cap_n1023_seed42_antithetic", true, false, true, 1023, 42);
    emitMcCase("mc_cap_n1023_seed42_bridge", true, true, false, 1023, 42);
    emitMcCase("mc_cap_n1023_seed42_bridge_antithetic", true, true, true, 1023, 42);
    emitMcCase("mc_floor_n1023_seed42", false, false, false, 1023, 42);
    emitMcCase("mc_floor_n4095_seed42", false, false, false, 4095, 42);
    emitMcCase("mc_floor_n1023_seed42_antithetic", false, false, true, 1023, 42);

    // E.5 -- analytic benchmark for the same cap/floor, so the MC value can be
    // sanity-checked independently of the stream.
    for (int isCapI = 0; isCapI < 2; ++isCapI) {
        const bool isCap = isCapI == 0;
        auto instrument = makeCapFloor(isCap);
        instrument->setPricingEngine(
            ext::make_shared<AnalyticCapFloorEngine>(makeHullWhite(), fwdCurve()));
        Obj ex;
        ex.n("npv", instrument->NPV());
        addCase(std::string("mc_analytic_benchmark_") + (isCap ? "cap" : "floor"),
                mcInputs(isCap), ex);
    }

    // E.6 -- tolerance-driven termination and the builder guards.
    {
        Obj in = mcInputs(true);
        in.n("required_tolerance", 1e-4).i("seed", 42).i("max_samples", 1000000);
        Obj ex;
        try {
            auto cap = makeCapFloor(true);
            cap->setPricingEngine(
                MakeMCHullWhiteCapFloorEngine<PseudoRandom>(makeHullWhite())
                    .withAbsoluteTolerance(1e-4)
                    .withMaxSamples(1000000)
                    .withSeed(42));
            ex.b("throws", false)
                .n("npv", cap->NPV())
                .n("error_estimate", cap->errorEstimate());
        } catch (const std::exception&) {
            ex.b("throws", true);
        }
        addCase("mc_cap_tolerance_1em4_seed42", in, ex);
    }
    {
        auto builderThrows = [](bool samplesFirst) {
            try {
                auto maker =
                    MakeMCHullWhiteCapFloorEngine<PseudoRandom>(makeHullWhite());
                if (samplesFirst) {
                    maker.withSamples(1023);
                    maker.withAbsoluteTolerance(1e-4);
                } else {
                    maker.withAbsoluteTolerance(1e-4);
                    maker.withSamples(1023);
                }
                return false;
            } catch (const std::exception&) {
                return true;
            }
        };
        {
            Obj in;
            in.s("order", "withSamples then withAbsoluteTolerance");
            Obj ex;
            ex.b("throws", builderThrows(true));
            addCase("mc_make_samples_then_tolerance_throws", in, ex);
        }
        {
            Obj in;
            in.s("order", "withAbsoluteTolerance then withSamples");
            Obj ex;
            ex.b("throws", builderThrows(false));
            addCase("mc_make_tolerance_then_samples_throws", in, ex);
        }
        {
            // Neither samples nor tolerance -> McSimulation::calculate rejects.
            Obj in;
            in.s("order", "no termination criterion");
            Obj ex;
            try {
                auto cap = makeCapFloor(true);
                cap->setPricingEngine(
                    MakeMCHullWhiteCapFloorEngine<PseudoRandom>(makeHullWhite())
                        .withSeed(42));
                static_cast<void>(cap->NPV());
                ex.b("throws", false);
            } catch (const std::exception&) {
                ex.b("throws", true);
            }
            addCase("mc_make_no_criterion_throws", in, ex);
        }
        {
            // v1.43 fact: the builder has NO withControlVariate, and the engine
            // hardcodes controlVariate = false. Pinned as a structural claim.
            Obj in;
            in.s("class", "MakeMCHullWhiteCapFloorEngine");
            Obj ex;
            ex.sa("named_parameters",
                  {"withBrownianBridge", "withSamples", "withAbsoluteTolerance",
                   "withMaxSamples", "withSeed", "withAntitheticVariate"})
                .b("has_with_control_variate", false)
                .b("engine_control_variate_flag", false);
            addCase("mc_make_no_control_variate_builder", in, ex);
        }
    }

    // =======================================================================
    // Section F -- BasketGeneratingEngine::calibrationBasket + MatchHelper.
    // Reached through Gaussian1dNonstandardSwaptionEngine on a *standard*
    // 5y-into-5y payer swaption (so the exotic and the standard swap coincide
    // and the delta/gamma fit has a well-posed target).
    // =======================================================================
    {
        auto swapIndexFwd = ext::make_shared<SwapIndex>(
            "EurSwap5Y", 5 * Years, 2, EURCurrency(), TARGET(), 1 * Years,
            ModifiedFollowing, Thirty360(Thirty360::BondBasis), euribor6m());
        auto volTs = ext::make_shared<ConstantSwaptionVolatility>(
            0, TARGET(), Following, 0.20, dcA365(), ShiftedLognormal, 0.0);

        for (int bt = 0; bt < 2; ++bt) {
            const auto basketType =
                bt == 0 ? BasketGeneratingEngine::Naive
                        : BasketGeneratingEngine::MaturityStrikeByDeltaGamma;
            const char* btName = bt == 0 ? "naive" : "delta_gamma";

            Obj in = gsrInputs();
            in.s("engine", "Gaussian1dNonstandardSwaptionEngine")
                .s("basket_type",
                   bt == 0 ? "Naive" : "MaturityStrikeByDeltaGamma")
                .s("standard_swap_base", "SwapIndex(EurSwap5Y, 5Y, 2, TARGET, 1Y, "
                                         "ModifiedFollowing, Thirty360(BondBasis), "
                                         "Euribor6M)")
                .s("swaption_volatility",
                   "ConstantSwaptionVolatility(0, TARGET, Following, 0.20, "
                   "Actual365Fixed, ShiftedLognormal, 0.0)")
                .n("fixed_rate", 0.030)
                .s("swap_type", "Payer");

            Obj ex;
            try {
                auto vanilla = makeSwap(Swap::Payer, 0.030);
                auto stdSwaption = ext::make_shared<Swaption>(
                    vanilla, ext::shared_ptr<Exercise>(
                                 ext::make_shared<EuropeanExercise>(swaptionExpiry())));
                auto nonstd = ext::make_shared<NonstandardSwaption>(*stdSwaption);
                auto engine =
                    ext::make_shared<Gaussian1dNonstandardSwaptionEngine>(makeGsr());
                nonstd->setPricingEngine(engine);
                const Real exoticNpv = nonstd->NPV();

                const auto basket = engine->calibrationBasket(
                    nonstd->exercise(), swapIndexFwd, volTs, basketType);

                std::vector<Date> expiries, maturities;
                std::vector<Real> strikes, nominals, vols;
                for (const auto& h : basket) {
                    auto sh = ext::dynamic_pointer_cast<SwaptionHelper>(h);
                    // swaption() forces the helper's lazy calculate(), which is
                    // what builds underlying(); the order matters.
                    expiries.push_back(sh->swaption()->exercise()->date(0));
                    maturities.push_back(sh->underlying()->maturityDate());
                    strikes.push_back(sh->underlying()->fixedRate());
                    nominals.push_back(sh->underlying()->nominal());
                    vols.push_back(sh->volatility()->value());
                }
                ex.b("throws", false)
                    .n("exotic_npv", exoticNpv)
                    .i("basket_size", static_cast<long long>(basket.size()))
                    .da("expiries", expiries)
                    .da("maturities", maturities)
                    .a("strikes", strikes)
                    .a("nominals", nominals)
                    .a("volatilities", vols);
            } catch (const std::exception&) {
                ex.b("throws", true);
            }
            addCase(std::string("basket_") + btName, in, ex);
        }
    }

    emitDocument();
    return 0;
}
