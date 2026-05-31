// Phase 11 W8-A cluster probe: CMS-spread index + irregular swaptions +
// basis-model (tenor-rescaled) volatility term structures.
//
// Captures reference values for the *buildable* subset of the W8-A scope.
// The CMS-coupon-pricer family (CmsCoupon / CmsCouponPricer) and the
// capped/floored + digital coupon hierarchies are deliberately deferred in
// PQuantLib (documented carve-out), so LognormalCmsSpreadPricer,
// CappedFlooredCmsSpreadCoupon, DigitalCmsSpreadCoupon and
// StrippedCappedFlooredCoupon are NOT exercised here.
//
//   * SwapSpreadIndex.fixing = g1 * swapIndex1.fixing + g2 * swapIndex2.fixing
//     (TIGHT — exact algebraic combination of two swap-rate fixings).
//
//   * IrregularSwap NPV with a step-down notional schedule on the fixed leg
//     (LOOSE — discounted-cashflow swap).
//
//   * HaganIrregularSwaptionEngine NPV at a known normal vol (LOOSE —
//     Hagan linear-TSR super-replication basket).
//
//   * TenorSwaptionVTS.volatility rescaling at known params (LOOSE).
//
//   * TenorOptionletVTS.volatility at known params (LOOSE).
//
//   * SwaptionCashFlows annuity/float/fixed weight sums (TIGHT — deterministic
//     cash-flow decomposition).
//
// C++ parity:
//   ql/experimental/coupons/swapspreadindex.hpp
//   ql/experimental/swaptions/irregularswap.hpp
//   ql/experimental/swaptions/irregularswaption.hpp
//   ql/experimental/swaptions/haganirregularswaptionengine.hpp
//   ql/experimental/basismodels/swaptioncfs.hpp
//   ql/experimental/basismodels/tenoroptionletvts.hpp
//   ql/experimental/basismodels/tenorswaptionvts.hpp
//   @ v1.42.1 (099987f0).

#include <ql/cashflows/cashflowvectors.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/exercise.hpp>
#include <ql/experimental/basismodels/swaptioncfs.hpp>
#include <ql/experimental/basismodels/tenoroptionletvts.hpp>
#include <ql/experimental/basismodels/tenorswaptionvts.hpp>
#include <ql/experimental/coupons/swapspreadindex.hpp>
#include <ql/experimental/swaptions/haganirregularswaptionengine.hpp>
#include <ql/experimental/swaptions/irregularswap.hpp>
#include <ql/experimental/swaptions/irregularswaption.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/instruments/makevanillaswap.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
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

// A SwapIndex on Euribor6M with a given tenor, bound to the discount curve as
// forwarding curve.
ext::shared_ptr<SwapIndex> makeSwapIndex(const Period& tenor,
                                         const Handle<YieldTermStructure>& curve) {
    auto euribor6m = ext::make_shared<Euribor6M>(curve);
    return ext::make_shared<SwapIndex>(
        "EuriborSwap", tenor, 2, EURCurrency(), cal,
        Period(1, Years), ModifiedFollowing, Thirty360(Thirty360::BondBasis),
        euribor6m, curve);
}

// ---------------------------------------------------------------------
// SwapSpreadIndex
// ---------------------------------------------------------------------
void block_swap_spread_index() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);

    auto s10 = makeSwapIndex(Period(10, Years), curve);
    auto s2 = makeSwapIndex(Period(2, Years), curve);

    SwapSpreadIndex idx("CMS10Y-2Y", s10, s2, 1.0, -1.0);

    // a future fixing date (well after today, so the underlying swaps are
    // fully forward-starting and need no historic float fixing)
    Date fix(15, January, 2025);
    Real f1 = s10->fixing(fix, false);
    Real f2 = s2->fixing(fix, false);

    emit("ssi_fix1_10y", f1);
    emit("ssi_fix2_2y", f2);
    emit("ssi_spread_fixing", idx.fixing(fix, false));
    emit("ssi_gearing1", idx.gearing1());
    emit("ssi_gearing2", idx.gearing2());
}

// ---------------------------------------------------------------------
// IrregularSwap with step-down notional
// ---------------------------------------------------------------------
void block_irregular_swap() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto euribor6m = ext::make_shared<Euribor6M>(curve);

    // forward-starting so no historic float fixing is needed
    Date start(15, January, 2025);
    Date maturity(15, January, 2028);

    Schedule fixedSched(start, maturity, Period(1, Years), cal,
                        ModifiedFollowing, ModifiedFollowing,
                        DateGeneration::Forward, false);
    Schedule floatSched(start, maturity, Period(6, Months), cal,
                        ModifiedFollowing, ModifiedFollowing,
                        DateGeneration::Forward, false);

    // step-down notionals on the fixed leg: 1.0e6, 0.7e6, 0.4e6
    std::vector<Real> fixedNominals = {1.0e6, 0.7e6, 0.4e6};
    Real fixedRate = 0.035;

    Leg fixedLeg = FixedRateLeg(fixedSched)
                       .withNotionals(fixedNominals)
                       .withCouponRates(fixedRate, Thirty360(Thirty360::BondBasis))
                       .withPaymentAdjustment(ModifiedFollowing);

    // floating leg: matching step-down on the 6 semi-annual periods
    std::vector<Real> floatNominals = {1.0e6, 1.0e6, 0.7e6, 0.7e6, 0.4e6, 0.4e6};
    Leg floatLeg = IborLeg(floatSched, euribor6m)
                       .withNotionals(floatNominals)
                       .withPaymentDayCounter(dc360)
                       .withPaymentAdjustment(ModifiedFollowing);

    auto swap = ext::make_shared<IrregularSwap>(
        IrregularSwap::Receiver, fixedLeg, floatLeg);

    auto engine = ext::make_shared<DiscountingSwapEngine>(curve);
    swap->setPricingEngine(engine);

    emit("irr_swap_npv", swap->NPV());
    emit("irr_swap_fixed_npv", swap->fixedLegNPV());
    emit("irr_swap_float_npv", swap->floatingLegNPV());
}

// ---------------------------------------------------------------------
// HaganIrregularSwaptionEngine
// ---------------------------------------------------------------------
void block_hagan_engine() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto euribor6m = ext::make_shared<Euribor6M>(curve);

    // exercise at swap start (Hagan engine assumes exercise == swap start)
    Date start(15, January, 2025);
    Date maturity(15, January, 2028);

    Schedule fixedSched(start, maturity, Period(1, Years), cal,
                        ModifiedFollowing, ModifiedFollowing,
                        DateGeneration::Forward, false);
    Schedule floatSched(start, maturity, Period(6, Months), cal,
                        ModifiedFollowing, ModifiedFollowing,
                        DateGeneration::Forward, false);

    std::vector<Real> fixedNominals = {1.0e6, 0.7e6, 0.4e6};
    Real fixedRate = 0.035;

    Leg fixedLeg = FixedRateLeg(fixedSched)
                       .withNotionals(fixedNominals)
                       .withCouponRates(fixedRate, Thirty360(Thirty360::BondBasis))
                       .withPaymentAdjustment(ModifiedFollowing);

    std::vector<Real> floatNominals = {1.0e6, 1.0e6, 0.7e6, 0.7e6, 0.4e6, 0.4e6};
    Leg floatLeg = IborLeg(floatSched, euribor6m)
                       .withNotionals(floatNominals)
                       .withPaymentDayCounter(dc360)
                       .withPaymentAdjustment(ModifiedFollowing);

    auto swap = ext::make_shared<IrregularSwap>(
        IrregularSwap::Receiver, fixedLeg, floatLeg);

    // normal swaption vol, constant 80 bp
    Handle<SwaptionVolatilityStructure> vol(
        ext::make_shared<ConstantSwaptionVolatility>(
            today, cal, ModifiedFollowing, 0.0080, dcA365, Normal));

    auto exercise = ext::make_shared<EuropeanExercise>(start);
    IrregularSwaption swaption(swap, exercise);

    auto engine = ext::make_shared<HaganIrregularSwaptionEngine>(vol, curve);
    swaption.setPricingEngine(engine);

    emit("hagan_swaption_npv", swaption.NPV());
}

// ---------------------------------------------------------------------
// TenorSwaptionVTS
// ---------------------------------------------------------------------
void block_tenor_swaption_vts() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);

    // base 6M index, target 3M index
    auto base6m = ext::make_shared<Euribor6M>(curve);
    auto targ3m = ext::make_shared<Euribor3M>(curve);

    Handle<SwaptionVolatilityStructure> baseVol(
        ext::make_shared<ConstantSwaptionVolatility>(
            today, cal, ModifiedFollowing, 0.0090, dcA365, Normal));

    TenorSwaptionVTS vts(baseVol, curve, base6m, targ3m,
                         Period(1, Years), Period(1, Years),
                         Thirty360(Thirty360::BondBasis),
                         Thirty360(Thirty360::BondBasis));

    // volatility at (optionTime=5, swapLength=10, strike=ATM-ish 0.03)
    emit("tsvts_vol_5x10_atm", vts.volatility(5.0, 10.0, 0.03));
    emit("tsvts_vol_2x5_atm", vts.volatility(2.0, 5.0, 0.03));
    emit("tsvts_vol_5x10_otm", vts.volatility(5.0, 10.0, 0.04));
}

// ---------------------------------------------------------------------
// TenorOptionletVTS
//
// NOTE: not cross-validated here. The transformation formula queries the
// *base* optionlet smile section via volatility(strike, Normal, 0.0); a flat
// ConstantOptionletVolatility section reports a ShiftedLognormal native type
// (and no atmLevel), so the type-conversion path in SmileSection::volatility
// aborts ("must provide atm level"). A faithful probe would need a non-flat,
// atm-aware base optionlet surface (out of this cluster's scope). The Python
// TenorOptionletVTS is exercised structurally (smoke test) instead.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// SwaptionCashFlows
// ---------------------------------------------------------------------
void block_swaption_cfs() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto euribor6m = ext::make_shared<Euribor6M>(curve);

    Date exercise(15, January, 2026);

    ext::shared_ptr<VanillaSwap> swap =
        MakeVanillaSwap(Period(5, Years), euribor6m, 0.03)
            .withEffectiveDate(cal.advance(exercise, 2, Days))
            .withFixedLegTenor(Period(1, Years))
            .withFixedLegDayCount(Thirty360(Thirty360::BondBasis))
            .withDiscountingTermStructure(curve);

    auto swaption = ext::make_shared<Swaption>(
        swap, ext::make_shared<EuropeanExercise>(exercise));

    SwaptionCashFlows cfs(swaption, curve);

    Real sumAnnuity = 0.0;
    for (Real w : cfs.annuityWeights())
        sumAnnuity += w;
    Real sumFloat = 0.0;
    for (Real w : cfs.floatWeights())
        sumFloat += w;
    Real sumFixed = 0.0;
    for (Real w : cfs.fixedWeights())
        sumFixed += w;

    emit("scfs_sum_annuity", sumAnnuity);
    emit("scfs_sum_float", sumFloat);
    emit("scfs_sum_fixed", sumFixed);
    emit("scfs_num_exercise", (Real)cfs.exerciseTimes().size());
    emit("scfs_num_float", (Real)cfs.floatTimes().size());
    emit("scfs_first_float_time", cfs.floatTimes().front(), false);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = today;
    std::cout << "{\n";

    block_swap_spread_index();
    block_irregular_swap();
    block_hagan_engine();
    block_tenor_swaption_vts();
    block_swaption_cfs();

    std::cout << "}\n";
    return 0;
}
