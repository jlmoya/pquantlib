// Phase 11 W12-B cluster probe: capped/floored + digital coupon families.
//
// FINAL-wave core-cashflows gap-fill — the cap/floored + digital coupon
// classes that L2-D deferred (CappedFlooredCoupon / CappedFlooredIborCoupon /
// CappedFlooredCmsCoupon / DigitalCoupon / DigitalReplication / DigitalIbor /
// DigitalCms legs / StrippedCappedFlooredCoupon). These are genuine gaps —
// L2-D only ported plain (uncapped) FloatingRateCoupon / IborCoupon.
//
// Setup mirrors the canonical QuantLib `digitalcoupon.cpp` test:
//   * Euribor6M on a flat 5% curve, fixingDays = 2.
//   * ConstantOptionletVolatility caplet vol + BlackIborCouponPricer.
//   * 10Y-forward-starting coupon (k = 9 in the C++ loop).
//
// Probe coverage:
//   * CappedFlooredIborCoupon.rate() with a cap / a floor / a collar — the
//     Black-vol-adjusted swaplet rate (LOOSE — Black caplet/floorlet).
//   * effectiveCap / effectiveFloor (TIGHT — (strike - spread) / gearing).
//   * DigitalCoupon callOptionRate / putOptionRate via call/put-spread
//     replication at a small gap (LOOSE — replication ≈ Cox-Rubinstein
//     asset-or-nothing). Also Central/Sub/Super replication-type rates.
//   * StrippedCappedFlooredCoupon.rate() = floorletRate (long floor) /
//     -capletRate (... actually +capletRate? see code) — the stripped
//     optionality value (TIGHT vs the underlying's caplet/floorlet rates).
//   * CappedFlooredCmsCoupon with an AnalyticHaganPricer (LOOSE — Hagan
//     convexity-adjusted CMS swaplet, no cap/floor pricer → only the
//     uncapped CMS swaplet rate is exercised).
//
// C++ parity:
//   ql/cashflows/capflooredcoupon.hpp + .cpp
//   ql/cashflows/digitalcoupon.hpp + .cpp
//   ql/cashflows/replication.hpp + .cpp
//   ql/cashflows/digitaliborcoupon.hpp + .cpp
//   ql/cashflows/digitalcmscoupon.hpp + .cpp
//   ql/experimental/coupons/strippedcapflooredcoupon.hpp + .cpp
//   @ v1.42.1 (099987f0).

#include <ql/cashflows/capflooredcoupon.hpp>
#include <ql/cashflows/cashflowvectors.hpp>
#include <ql/cashflows/conundrumpricer.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/digitalcoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/replication.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/experimental/coupons/strippedcapflooredcoupon.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

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

const Actual365Fixed dcA365;
const Actual360 dc360;
const TARGET cal;
const Natural fixingDays = 2;
const Real nominal = 1000000.0;

// digitalcoupon.cpp CommonVars: today = adjust(eval date), settlement =
// today + 2 business days, curve = flat 5% from settlement.
Date g_today;
Date g_settlement;
RelinkableHandle<YieldTermStructure> g_ts;

ext::shared_ptr<IborIndex> makeEuribor6M() {
    return ext::make_shared<Euribor6M>(g_ts);
}

Handle<OptionletVolatilityStructure> constCapletVol(Volatility v) {
    return Handle<OptionletVolatilityStructure>(
        ext::make_shared<ConstantOptionletVolatility>(
            g_today, cal, Following, v, dc360));
}

ext::shared_ptr<SwapIndex> makeSwapIndex(const Period& tenor,
                                         const ext::shared_ptr<IborIndex>& ibor) {
    return ext::make_shared<SwapIndex>(
        "EuriborSwapIsdaFixA", tenor, ibor->fixingDays(), ibor->currency(),
        ibor->fixingCalendar(), Period(1, Years), Unadjusted,
        ibor->dayCounter(), ibor);
}

Handle<SwaptionVolatilityStructure> constSwaptionVol(Real v) {
    return Handle<SwaptionVolatilityStructure>(
        ext::make_shared<ConstantSwaptionVolatility>(
            g_today, cal, Following, v, dcA365, ShiftedLognormal));
}

// k = 9 forward-starting coupon (matches the digitalcoupon.cpp loop body).
struct CouponDates {
    Date start, end, payment;
};
CouponDates couponDates() {
    Date startDate = cal.advance(g_settlement, 10 * Years);
    Date endDate = cal.advance(g_settlement, 11 * Years);
    return {startDate, endDate, endDate};
}

// ---------------------------------------------------------------------
// CappedFlooredIborCoupon — cap / floor / collar rate + effective strikes
// ---------------------------------------------------------------------
void block_capped_floored_ibor() {
    auto index = makeEuribor6M();
    auto vol = constCapletVol(0.15);
    auto pricer = ext::make_shared<BlackIborCouponPricer>(vol);
    auto d = couponDates();

    Real gearing = 1.0, spread = 0.0;
    Rate cap = 0.04, floor = 0.03;
    Rate nullstrike = Null<Rate>();

    // plain underlying rate (the forecast swaplet) — anchor for the tests
    {
        IborCoupon underlying(d.payment, nominal, d.start, d.end, fixingDays,
                              index, gearing, spread);
        underlying.setPricer(pricer);
        emit("cfi_underlying_rate", underlying.rate());
    }

    // capped only
    {
        CappedFlooredIborCoupon c(d.payment, nominal, d.start, d.end,
                                  fixingDays, index, gearing, spread, cap,
                                  nullstrike);
        c.setPricer(pricer);
        emit("cfi_capped_rate", c.rate());
        emit("cfi_capped_effcap", c.effectiveCap());
    }

    // floored only
    {
        CappedFlooredIborCoupon c(d.payment, nominal, d.start, d.end,
                                  fixingDays, index, gearing, spread,
                                  nullstrike, floor);
        c.setPricer(pricer);
        emit("cfi_floored_rate", c.rate());
        emit("cfi_floored_efffloor", c.effectiveFloor());
    }

    // collar (cap + floor)
    {
        CappedFlooredIborCoupon c(d.payment, nominal, d.start, d.end,
                                  fixingDays, index, gearing, spread, cap,
                                  floor);
        c.setPricer(pricer);
        emit("cfi_collar_rate", c.rate());
    }

    // gearing / spread effective strikes (TIGHT — pure arithmetic)
    {
        Real g2 = 2.0, s2 = 0.005;
        CappedFlooredIborCoupon c(d.payment, nominal, d.start, d.end,
                                  fixingDays, index, g2, s2, cap, floor);
        c.setPricer(pricer);
        emit("cfi_geared_effcap", c.effectiveCap());
        emit("cfi_geared_efffloor", c.effectiveFloor());
        emit("cfi_geared_rate", c.rate());
    }
}

// ---------------------------------------------------------------------
// DigitalCoupon — call/put option rate via replication + replication types
// ---------------------------------------------------------------------
void block_digital_coupon() {
    auto index = makeEuribor6M();
    auto vol = constCapletVol(0.15);
    auto pricer = ext::make_shared<BlackIborCouponPricer>(vol);
    auto d = couponDates();

    Real gearing = 1.0, spread = 0.0;
    Rate strike = 0.04;
    Rate nullstrike = Null<Rate>();
    Real gap = 1e-4;

    auto underlying = ext::make_shared<IborCoupon>(
        d.payment, nominal, d.start, d.end, fixingDays, index, gearing, spread);
    underlying->setPricer(pricer);
    emit("dc_underlying_rate", underlying->rate());

    // forward / effFwd / effStrike for the Cox-Rubinstein cross-check
    Rate forward = underlying->rate();
    Rate effFwd = (forward - spread) / gearing;
    Rate effStrike = (strike - spread) / gearing;
    emit("dc_eff_fwd", effFwd);
    emit("dc_eff_strike", effStrike);

    // Asset-or-nothing digital call (short), Central replication, small gap.
    auto rep = ext::make_shared<DigitalReplication>(Replication::Central, gap);
    {
        DigitalCoupon dig(underlying, strike, Position::Short, false,
                          nullstrike, nullstrike, Position::Short, false,
                          nullstrike, rep);
        dig.setPricer(pricer);
        emit("dc_call_option_rate", dig.callOptionRate());
        emit("dc_call_rate", dig.rate());
    }

    // Asset-or-nothing digital put (long), Central replication.
    {
        DigitalCoupon dig(underlying, nullstrike, Position::Long, false,
                          nullstrike, strike, Position::Long, false,
                          nullstrike, rep);
        dig.setPricer(pricer);
        emit("dc_put_option_rate", dig.putOptionRate());
        emit("dc_put_rate", dig.rate());
    }

    // Cash-or-nothing digital call (long), fixed payoff.
    {
        Rate payoff = 0.10;
        DigitalCoupon dig(underlying, strike, Position::Long, false, payoff,
                          nullstrike, Position::Long, false, nullstrike, rep);
        dig.setPricer(pricer);
        emit("dc_cash_call_option_rate", dig.callOptionRate());
    }

    // Replication-type comparison (Central / Sub / Super) on the call rate.
    {
        auto repSub = ext::make_shared<DigitalReplication>(Replication::Sub, gap);
        auto repSuper =
            ext::make_shared<DigitalReplication>(Replication::Super, gap);

        DigitalCoupon central(underlying, strike, Position::Long, false,
                              nullstrike, nullstrike, Position::Long, false,
                              nullstrike, rep);
        central.setPricer(pricer);
        DigitalCoupon sub(underlying, strike, Position::Long, false, nullstrike,
                          nullstrike, Position::Long, false, nullstrike, repSub);
        sub.setPricer(pricer);
        DigitalCoupon super(underlying, strike, Position::Long, false,
                            nullstrike, nullstrike, Position::Long, false,
                            nullstrike, repSuper);
        super.setPricer(pricer);

        emit("dc_central_call_rate", central.callOptionRate());
        emit("dc_sub_call_rate", sub.callOptionRate());
        emit("dc_super_call_rate", super.callOptionRate());
    }
}

// ---------------------------------------------------------------------
// StrippedCappedFlooredCoupon — extracted optionality value
// ---------------------------------------------------------------------
void block_stripped() {
    auto index = makeEuribor6M();
    auto vol = constCapletVol(0.15);
    auto pricer = ext::make_shared<BlackIborCouponPricer>(vol);
    auto d = couponDates();

    Real gearing = 1.0, spread = 0.0;
    Rate cap = 0.04, floor = 0.03;
    Rate nullstrike = Null<Rate>();

    // long cap: stripped value = capletRate, and underlying.rate() =
    // swaplet - capletRate, so stripped = swaplet - underlying.rate().
    {
        auto cfc = ext::make_shared<CappedFlooredIborCoupon>(
            d.payment, nominal, d.start, d.end, fixingDays, index, gearing,
            spread, cap, nullstrike);
        cfc->setPricer(pricer);
        StrippedCappedFlooredCoupon stripped(cfc);
        stripped.setPricer(pricer);
        emit("scf_capped_rate", stripped.rate());
        emit("scf_capped_cfc_rate", cfc->rate());
    }

    // long floor.
    {
        auto cfc = ext::make_shared<CappedFlooredIborCoupon>(
            d.payment, nominal, d.start, d.end, fixingDays, index, gearing,
            spread, nullstrike, floor);
        cfc->setPricer(pricer);
        StrippedCappedFlooredCoupon stripped(cfc);
        stripped.setPricer(pricer);
        emit("scf_floored_rate", stripped.rate());
        emit("scf_floored_cfc_rate", cfc->rate());
    }

    // collar: stripped = floorletRate - capletRate.
    {
        auto cfc = ext::make_shared<CappedFlooredIborCoupon>(
            d.payment, nominal, d.start, d.end, fixingDays, index, gearing,
            spread, cap, floor);
        cfc->setPricer(pricer);
        StrippedCappedFlooredCoupon stripped(cfc);
        stripped.setPricer(pricer);
        emit("scf_collar_rate", stripped.rate());
        emit("scf_collar_cap", stripped.cap());
        emit("scf_collar_floor", stripped.floor());
    }
}

// ---------------------------------------------------------------------
// CappedFlooredCmsCoupon — uncapped CMS swaplet via Hagan (no CMS cap pricer)
// ---------------------------------------------------------------------
void block_capped_floored_cms() {
    auto index = makeEuribor6M();
    auto swapIndex = makeSwapIndex(Period(10, Years), index);
    auto vol = constSwaptionVol(0.16);
    Handle<Quote> zeroMeanRev(ext::make_shared<SimpleQuote>(0.0));
    auto hagan = ext::make_shared<AnalyticHaganPricer>(
        vol, GFunctionFactory::Standard, zeroMeanRev);

    // 20Y-forward CMS coupon (matches the cms.cpp conundrum setup).
    Date startDate = g_ts->referenceDate() + 20 * Years;
    Date paymentDate = startDate + 1 * Years;
    Date endDate = paymentDate;

    Real gearing = 1.0, spread = 0.0;
    Rate nullstrike = Null<Rate>();

    // Uncapped (cap = floor = null) → CappedFlooredCmsCoupon degenerates to the
    // plain Hagan CMS swaplet rate. This is the only branch exercisable without
    // a CMS cap/floor pricer (the CmsCouponPricer base doesn't price cap/floor).
    {
        CappedFlooredCmsCoupon c(paymentDate, 1.0, startDate, endDate,
                                 swapIndex->fixingDays(), swapIndex, gearing,
                                 spread, nullstrike, nullstrike, startDate,
                                 endDate, index->dayCounter());
        c.setPricer(hagan);
        emit("cfcms_uncapped_rate", c.rate());
    }

    // Plain CmsCoupon rate for the equivalence check.
    {
        CmsCoupon plain(paymentDate, 1.0, startDate, endDate,
                        swapIndex->fixingDays(), swapIndex, gearing, spread,
                        startDate, endDate, index->dayCounter());
        plain.setPricer(hagan);
        emit("cfcms_plain_rate", plain.rate(), false);
    }
}

}  // namespace

int main() {
    // digitalcoupon.cpp CommonVars setup, but with a *pinned* evaluation date
    // (15-Jan-2024) so the probe is reproducible and matches the Python test.
    // The C++ test uses the machine default eval date; we fix it instead.
    Settings::instance().evaluationDate() = Date(15, January, 2024);
    g_today = cal.adjust(Settings::instance().evaluationDate());
    Settings::instance().evaluationDate() = g_today;
    g_settlement = cal.advance(g_today, fixingDays, Days);
    g_ts.linkTo(ext::shared_ptr<YieldTermStructure>(
        new FlatForward(g_settlement, 0.05, dcA365)));

    std::cout << "{\n";
    block_capped_floored_ibor();
    block_digital_coupon();
    block_stripped();
    block_capped_floored_cms();
    std::cout << "}\n";
    return 0;
}
