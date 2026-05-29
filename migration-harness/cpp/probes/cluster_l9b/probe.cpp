// L9-B cluster probe: IsdaCdsEngine + implied_hazard_rate + conventional_spread.
//
// Emits reference values for:
//
//   * IsdaCdsEngine NPV / fairSpread / couponLegNPV / defaultLegNPV
//     for a 5y CDS with running spread 2%, FlatHazardRate lambda=0.02,
//     FlatForward discount=3% (continuous, Act/365F), recovery=40%,
//     Quarterly schedule, notional 10M.
//
//   * implied_hazard_rate roundtrip: starting from the above NPV, the
//     Brent solver should recover hazard ~ 0.02 to 1e-8 accuracy.
//
// C++ parity: ql/pricingengines/credit/isdacdsengine.{hpp,cpp},
//             ql/instruments/creditdefaultswap.cpp (impliedHazardRate +
//             conventionalSpread) @ v1.42.1.

#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/instruments/creditdefaultswap.hpp>
#include <ql/pricingengines/credit/isdacdsengine.hpp>
#include <ql/pricingengines/credit/midpointcdsengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
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
    std::cout << "{\n";

    // -- common setup --
    Date eval = Date(15, June, 2026);
    Settings::instance().evaluationDate() = eval;
    Actual365Fixed dc365;
    Actual360 dc360;
    Calendar cal = WeekendsOnly();
    BusinessDayConvention bdc = Following;

    Real lambda = 0.02;
    Real recovery = 0.4;
    Real notional = 10000000.0;
    Rate spread = 0.02;

    Handle<DefaultProbabilityTermStructure> probability(
        ext::make_shared<FlatHazardRate>(eval, lambda, dc365));
    Handle<YieldTermStructure> discountCurve(
        ext::make_shared<FlatForward>(eval, 0.03, dc365, Continuous, Annual));

    Date maturity = eval + Period(5, Years);
    Schedule schedule(eval, maturity, Period(Quarterly), cal, bdc, bdc,
                      DateGeneration::TwentiethIMM, false);

    CreditDefaultSwap cds(Protection::Buyer, notional, spread, schedule,
                          bdc, dc360,
                          /*settlesAccrual*/ true,
                          /*paysAtDefaultTime*/ true,
                          /*protectionStart*/ eval,
                          ext::shared_ptr<Claim>(),  // FaceValueClaim default
                          DayCounter(),
                          /*rebatesAccrual*/ true,
                          /*tradeDate*/ eval);

    // == IsdaCdsEngine ==
    cds.setPricingEngine(ext::make_shared<IsdaCdsEngine>(
        probability, recovery, discountCurve,
        ext::nullopt,
        IsdaCdsEngine::Taylor,
        IsdaCdsEngine::HalfDayBias,
        IsdaCdsEngine::Piecewise));

    Real npv_isda = cds.NPV();
    Real fair_spread_isda = cds.fairSpread();
    Real coupon_leg_npv_isda = cds.couponLegNPV();
    Real default_leg_npv_isda = cds.defaultLegNPV();

    std::cout << "  \"isda_engine\": {\n";
    std::cout << "    \"lambda\":          " << lambda << ",\n";
    std::cout << "    \"spread\":          " << spread << ",\n";
    std::cout << "    \"recovery\":        " << recovery << ",\n";
    std::cout << "    \"notional\":        " << notional << ",\n";
    std::cout << "    \"discount_rate\":   0.03,\n";
    std::cout << "    \"npv\":             " << npv_isda << ",\n";
    std::cout << "    \"fair_spread\":     " << fair_spread_isda << ",\n";
    std::cout << "    \"coupon_leg_npv\":  " << coupon_leg_npv_isda << ",\n";
    std::cout << "    \"default_leg_npv\": " << default_leg_npv_isda << "\n";
    std::cout << "  },\n";

    // == implied_hazard_rate roundtrip ==
    // Restart from a MidPoint engine to mirror the Python default.
    cds.setPricingEngine(ext::make_shared<MidPointCdsEngine>(
        probability, recovery, discountCurve));
    Real npv_mid = cds.NPV();
    Real implied_h = cds.impliedHazardRate(
        npv_mid, discountCurve, dc365, recovery, 1e-8,
        CreditDefaultSwap::Midpoint);
    std::cout << "  \"implied_hazard_rate\": {\n";
    std::cout << "    \"target_npv\": " << npv_mid << ",\n";
    std::cout << "    \"hazard\":     " << implied_h << "\n";
    std::cout << "  },\n";

    // == conventional_spread ==
    Real cs = cds.conventionalSpread(recovery, discountCurve, dc365,
                                     CreditDefaultSwap::Midpoint);
    std::cout << "  \"conventional_spread_midpoint\": " << cs << "\n";

    std::cout << "}\n";
    return 0;
}
