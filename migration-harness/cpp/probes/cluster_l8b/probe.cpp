// L8-B cluster probe: credit cluster.
//
// Captures reference values for:
//
//   * FlatHazardRate(lambda=0.02) — survival probabilities + default densities
//     + hazard rates at sample times t = {0.5, 1, 2, 5}.
//
//   * InterpolatedSurvivalProbabilityCurve (LogLinear) — survival probs at
//     known node dates plus a mid-point interpolated value.
//
//   * InterpolatedHazardRateCurve (BackwardFlat) — hazard rate at nodes +
//     survival probability via piecewise-constant integral.
//
//   * InterpolatedDefaultDensityCurve (Linear) — default density at nodes
//     + survival probability via 1 - integral_0^t p(tau) d tau.
//
//   * Claim::FaceValueClaim::amount(t, N=1e7, recovery=0.4)  → 6e6.
//
//   * MidPointCdsEngine NPV for a 5y CDS, running spread 2%, FlatHazardRate
//     lambda=0.02, FlatForward discount=3% (continuous, Actual365Fixed),
//     recovery=0.4, Schedule(Quarterly), Following BDC, Actual360 day counter,
//     notional 10M, protection_start = trade_date = ref_date.
//
//   * IntegralCdsEngine NPV (step = 1M) with the same parameters.
//
//   * CreditDefaultSwap.fairSpread / couponLegNPV / defaultLegNPV for the
//     midpoint engine. Used to validate engine accessors round-trip.
//
// C++ parity: ql/termstructures/credit/{flathazardrate,survivalprobabilitystructure,
//                                       hazardratestructure,defaultdensitystructure,
//                                       interpolated*}.{hpp,cpp},
//             ql/instruments/{creditdefaultswap,claim}.{hpp,cpp},
//             ql/pricingengines/credit/{midpointcdsengine,integralcdsengine}.{hpp,cpp}
//             @ v1.42.1 (099987f0).

#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/instruments/creditdefaultswap.hpp>
#include <ql/math/interpolations/backwardflatinterpolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/interpolations/loginterpolation.hpp>
#include <ql/pricingengines/credit/integralcdsengine.hpp>
#include <ql/pricingengines/credit/midpointcdsengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/credit/interpolateddefaultdensitycurve.hpp>
#include <ql/termstructures/credit/interpolatedhazardratecurve.hpp>
#include <ql/termstructures/credit/interpolatedsurvivalprobabilitycurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/weekendsonly.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);

    Date evalDate(15, June, 2026);
    Settings::instance().evaluationDate() = evalDate;

    DayCounter dc365 = Actual365Fixed();
    DayCounter dc360 = Actual360();
    Calendar cal = WeekendsOnly();
    BusinessDayConvention bdc = Following;

    std::cout << "{\n";
    std::cout << "  \"setup\": {\n";
    std::cout << "    \"eval_date_serial\": " << evalDate.serialNumber() << "\n";
    std::cout << "  },\n";

    // ============================================================================
    // FlatHazardRate(lambda=0.02): S(t) = exp(-lambda * t), p(t) = lambda * S(t)
    // ============================================================================
    {
        Real lambda = 0.02;
        FlatHazardRate fhr(evalDate, lambda, dc365);
        std::cout << "  \"flat_hazard_rate\": {\n";
        std::cout << "    \"lambda\": " << lambda << ",\n";
        std::cout << "    \"survival_t05\": " << fhr.survivalProbability(0.5) << ",\n";
        std::cout << "    \"survival_t1\":  " << fhr.survivalProbability(1.0) << ",\n";
        std::cout << "    \"survival_t2\":  " << fhr.survivalProbability(2.0) << ",\n";
        std::cout << "    \"survival_t5\":  " << fhr.survivalProbability(5.0) << ",\n";
        std::cout << "    \"default_t05\":  " << fhr.defaultProbability(0.5) << ",\n";
        std::cout << "    \"default_t5\":   " << fhr.defaultProbability(5.0) << ",\n";
        std::cout << "    \"hazard_t5\":    " << fhr.hazardRate(5.0) << ",\n";
        std::cout << "    \"default_density_t5\": " << fhr.defaultDensity(5.0) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // InterpolatedSurvivalProbabilityCurve — LogLinear
    // ============================================================================
    {
        std::vector<Date> dates = {
            evalDate,
            evalDate + Period(1, Years),
            evalDate + Period(2, Years),
            evalDate + Period(5, Years),
        };
        std::vector<Probability> probs = {1.0, 0.98, 0.95, 0.85};
        InterpolatedSurvivalProbabilityCurve<LogLinear> spc(dates, probs, dc365);
        std::cout << "  \"interpolated_survival_probability\": {\n";
        std::cout << "    \"sp_at_node1y\": " << spc.survivalProbability(dates[1]) << ",\n";
        std::cout << "    \"sp_at_node2y\": " << spc.survivalProbability(dates[2]) << ",\n";
        std::cout << "    \"sp_at_node5y\": " << spc.survivalProbability(dates[3]) << ",\n";
        // Mid-point between node 0 and node 1: dc365.year_fraction(eval, eval+1y)/2.
        Time tmid = dc365.yearFraction(evalDate, dates[1]) / 2.0;
        std::cout << "    \"sp_mid_0_1\":   " << spc.survivalProbability(tmid) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // InterpolatedHazardRateCurve — BackwardFlat
    // ============================================================================
    {
        std::vector<Date> dates = {
            evalDate,
            evalDate + Period(1, Years),
            evalDate + Period(2, Years),
            evalDate + Period(5, Years),
        };
        std::vector<Rate> hazards = {0.01, 0.02, 0.025, 0.03};
        InterpolatedHazardRateCurve<BackwardFlat> hrc(dates, hazards, dc365);
        std::cout << "  \"interpolated_hazard_rate\": {\n";
        // BackwardFlat: hazard between (t_{i-1}, t_i] is hazards[i].
        Time t05 = 0.5;
        std::cout << "    \"hazard_t05\": " << hrc.hazardRate(t05) << ",\n";
        std::cout << "    \"hazard_t15\": " << hrc.hazardRate(1.5) << ",\n";
        std::cout << "    \"hazard_t3\":  " << hrc.hazardRate(3.0) << ",\n";
        std::cout << "    \"sp_t1\":      " << hrc.survivalProbability(1.0) << ",\n";
        std::cout << "    \"sp_t2\":      " << hrc.survivalProbability(2.0) << ",\n";
        std::cout << "    \"sp_t5\":      " << hrc.survivalProbability(5.0) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // InterpolatedDefaultDensityCurve — Linear
    // ============================================================================
    {
        std::vector<Date> dates = {
            evalDate,
            evalDate + Period(1, Years),
            evalDate + Period(2, Years),
            evalDate + Period(5, Years),
        };
        std::vector<Real> densities = {0.01, 0.012, 0.015, 0.02};
        InterpolatedDefaultDensityCurve<Linear> ddc(dates, densities, dc365);
        std::cout << "  \"interpolated_default_density\": {\n";
        std::cout << "    \"density_at_1y\": " << ddc.defaultDensity(1.0) << ",\n";
        std::cout << "    \"density_at_05\": " << ddc.defaultDensity(0.5) << ",\n";
        std::cout << "    \"sp_t1\":         " << ddc.survivalProbability(1.0) << ",\n";
        std::cout << "    \"sp_t2\":         " << ddc.survivalProbability(2.0) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // FaceValueClaim
    // ============================================================================
    {
        FaceValueClaim claim;
        Real amt = claim.amount(evalDate, 10000000.0, 0.4);
        std::cout << "  \"face_value_claim\": {\n";
        std::cout << "    \"amount_N10M_R40\": " << amt << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // MidPointCdsEngine + IntegralCdsEngine + CDS instrument round-trip.
    //
    // 5y CDS, running spread 200bps, FlatHazardRate lambda=0.02, FlatForward
    // discount=3% (continuous, Actual365Fixed), recovery=0.4, Quarterly schedule,
    // Following BDC, Actual360 day counter, notional 10M.
    // ============================================================================
    {
        Real lambda = 0.02;
        Real recoveryRate = 0.4;
        Real notional = 10000000.0;
        Rate spread = 0.02;

        Handle<DefaultProbabilityTermStructure> probability(
            ext::make_shared<FlatHazardRate>(evalDate, lambda, dc365));

        Handle<YieldTermStructure> discountCurve(
            ext::make_shared<FlatForward>(evalDate, 0.03, dc365, Continuous, Annual));

        Date maturity = evalDate + Period(5, Years);
        Schedule schedule(evalDate, maturity, Period(Quarterly),
                          cal, bdc, bdc, DateGeneration::TwentiethIMM, false);

        // Bog-standard CDS: protection_start = evalDate; trade_date defaults
        // to protection_start; the running-spread-only constructor.
        CreditDefaultSwap cds(Protection::Buyer, notional, spread, schedule,
                              bdc, dc360,
                              true,        // settlesAccrual
                              true,        // paysAtDefaultTime
                              evalDate,    // protectionStart
                              ext::shared_ptr<Claim>(),
                              DayCounter(),
                              true,        // rebatesAccrual
                              evalDate);   // tradeDate

        // MidPoint engine.
        cds.setPricingEngine(ext::make_shared<MidPointCdsEngine>(
            probability, recoveryRate, discountCurve));

        Real npv_mid = cds.NPV();
        Real fair_spread_mid = cds.fairSpread();
        Real coupon_leg_npv_mid = cds.couponLegNPV();
        Real default_leg_npv_mid = cds.defaultLegNPV();
        Real coupon_leg_bps_mid = cds.couponLegBPS();

        // Integral engine (1M step).
        cds.setPricingEngine(ext::make_shared<IntegralCdsEngine>(
            Period(1, Months), probability, recoveryRate, discountCurve));

        Real npv_int = cds.NPV();
        Real fair_spread_int = cds.fairSpread();
        Real coupon_leg_npv_int = cds.couponLegNPV();
        Real default_leg_npv_int = cds.defaultLegNPV();

        std::cout << "  \"cds_engine\": {\n";
        std::cout << "    \"lambda\":          " << lambda << ",\n";
        std::cout << "    \"spread\":          " << spread << ",\n";
        std::cout << "    \"recovery\":        " << recoveryRate << ",\n";
        std::cout << "    \"notional\":        " << notional << ",\n";
        std::cout << "    \"discount_rate\":   0.03,\n";
        std::cout << "    \"schedule_size\":   " << schedule.size() << ",\n";
        std::cout << "    \"first_coupon_date_serial\": " << schedule.date(0).serialNumber() << ",\n";
        std::cout << "    \"midpoint\": {\n";
        std::cout << "      \"npv\":             " << npv_mid << ",\n";
        std::cout << "      \"fair_spread\":     " << fair_spread_mid << ",\n";
        std::cout << "      \"coupon_leg_npv\":  " << coupon_leg_npv_mid << ",\n";
        std::cout << "      \"default_leg_npv\": " << default_leg_npv_mid << ",\n";
        std::cout << "      \"coupon_leg_bps\":  " << coupon_leg_bps_mid << "\n";
        std::cout << "    },\n";
        std::cout << "    \"integral\": {\n";
        std::cout << "      \"npv\":             " << npv_int << ",\n";
        std::cout << "      \"fair_spread\":     " << fair_spread_int << ",\n";
        std::cout << "      \"coupon_leg_npv\":  " << coupon_leg_npv_int << ",\n";
        std::cout << "      \"default_leg_npv\": " << default_leg_npv_int << "\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
