// L7-D cluster probe: inflation instruments + vol surfaces + YoY cap/floor engines.
//
// Captures reference values for:
//
//   * ConstantCPIVolatility.volatility(d, k) at sample (date, strike) — verifies
//     the constant surface returns the configured vol regardless of input.
//   * ConstantYoYOptionletVolatility.volatility(d, k) with the same property,
//     plus totalVariance(d, k) which encodes the time-from-base scaling.
//
//   * ZeroCouponInflationSwap fairRate() for a swap whose inflation leg
//     pays nominal * (I(T)/I(0) - 1) at maturity:
//       - I(0) = 100 baseline (stored as past UKRPI fixing on the period start)
//       - I(T) = 130 stored as past UKRPI fixing on the maturity period start
//       - growth = 130/100 = 1.30, so fairRate = 1.30^(1/T) - 1.
//     Plus the inflation-leg NPV at a flat-forward 3% nominal curve.
//
//   * YearOnYearInflationSwap fixedLeg NPV (deterministic, curve-only) + the
//     yoyLeg NPV path under a flat YoY-rate curve of 2.5%.
//
//   * YoYInflationBachelierCapFloorEngine: NPV of a 5y YoY cap @ 2.5% strike
//     on a 10M EUR nominal, with vol=0.5% (normal), index = YYEUHICP wired
//     to a flat 2.5% YoY curve, discounted on a flat 3% nominal curve.
//
//   * YoYInflationBlackCapFloorEngine: same setup at vol=20% lognormal.
//   * YoYInflationUnitDisplacedBlackCapFloorEngine: shifted Black variant.
//
// C++ parity:
//   ql/instruments/zerocouponinflationswap.{hpp,cpp},
//   ql/instruments/yearonyearinflationswap.{hpp,cpp},
//   ql/instruments/inflationcapfloor.{hpp,cpp},
//   ql/instruments/cpicapfloor.{hpp,cpp},
//   ql/termstructures/volatility/inflation/constantcpivolatility.{hpp,cpp},
//   ql/termstructures/volatility/inflation/yoyinflationoptionletvolatilitystructure.{hpp,cpp},
//   ql/pricingengines/inflation/inflationcapfloorengines.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/cashflows/yoyinflationcoupon.hpp>
#include <ql/cashflows/cashflowvectors.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/cashflows/zeroinflationcashflow.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/indexes/inflation/ukrpi.hpp>
#include <ql/indexes/inflationindex.hpp>
#include <ql/instruments/inflationcapfloor.hpp>
#include <ql/instruments/yearonyearinflationswap.hpp>
#include <ql/instruments/zerocouponinflationswap.hpp>
#include <ql/pricingengines/inflation/inflationcapfloorengines.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/inflation/interpolatedyoyinflationcurve.hpp>
#include <ql/termstructures/inflation/interpolatedzeroinflationcurve.hpp>
#include <ql/termstructures/inflationtermstructure.hpp>
#include <ql/termstructures/volatility/inflation/constantcpivolatility.hpp>
#include <ql/termstructures/volatility/inflation/yoyinflationoptionletvolatilitystructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);

    Date evalDate(17, January, 2024);
    Settings::instance().evaluationDate() = evalDate;

    DayCounter dc = ActualActual(ActualActual::ISDA);
    Calendar cal = TARGET();
    BusinessDayConvention bdc = ModifiedFollowing;

    // Nominal discount curve: flat 3% Act/365F, Continuous, Annual.
    Handle<Quote> nominalRateQuote(ext::make_shared<SimpleQuote>(0.03));
    Handle<YieldTermStructure> nominalCurve(
        ext::make_shared<FlatForward>(evalDate, nominalRateQuote, Actual365Fixed(),
                                      Continuous, Annual));

    std::cout << "{\n";
    std::cout << "  \"setup\": {\n";
    std::cout << "    \"eval_date_serial\": " << evalDate.serialNumber() << ",\n";
    std::cout << "    \"nominal_rate\": 0.03\n";
    std::cout << "  },\n";

    // ============================================================================
    // ConstantCPIVolatility — constant surface (no T or K dependence).
    // ============================================================================
    {
        Volatility cpiVol = 0.18;
        Period obsLag(2, Months);
        ConstantCPIVolatility cv(cpiVol, 0, cal, bdc, dc, obsLag, Monthly, false);
        // The vol overload uses (date, strike, obsLag, extrapolate).
        Date d1 = evalDate + Period(1, Years);
        Date d2 = evalDate + Period(5, Years);

        std::cout << "  \"constant_cpi_volatility\": {\n";
        std::cout << "    \"vol\": " << cpiVol << ",\n";
        std::cout << "    \"observation_lag_months\": 2,\n";
        std::cout << "    \"vol_at_1y_strike_3pct\": " << cv.volatility(d1, 0.03) << ",\n";
        std::cout << "    \"vol_at_5y_strike_0pct\": " << cv.volatility(d2, 0.00) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // ConstantYoYOptionletVolatility — constant surface for YoY.
    // ============================================================================
    {
        Volatility yoyVol = 0.005;
        Period obsLag(3, Months);
        ConstantYoYOptionletVolatility cv(yoyVol, 0, cal, bdc, dc, obsLag, Monthly, false);
        Date d1 = evalDate + Period(1, Years);
        Date d2 = evalDate + Period(5, Years);
        Real var1 = cv.totalVariance(d1, 0.025, Period(0, Days));
        Real var5 = cv.totalVariance(d2, 0.025, Period(0, Days));

        std::cout << "  \"constant_yoy_optionlet_volatility\": {\n";
        std::cout << "    \"vol\": " << yoyVol << ",\n";
        std::cout << "    \"observation_lag_months\": 3,\n";
        std::cout << "    \"vol_at_1y_strike_2_5pct\": " << cv.volatility(d1, 0.025) << ",\n";
        std::cout << "    \"vol_at_5y_strike_0pct\": " << cv.volatility(d2, 0.00) << ",\n";
        std::cout << "    \"total_variance_1y_2_5pct\": " << var1 << ",\n";
        std::cout << "    \"total_variance_5y_2_5pct\": " << var5 << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // ZeroCouponInflationSwap NPV at known fixings + flat nominal curve.
    //
    // I(base) = 100 stored on UKRPI for period containing baseDate.
    // I(maturity-lag) = 130 stored for period containing obs date.
    // ratio = 130/100 = 1.30, growth = ratio - 1 = 0.30.
    // Expected inflation-leg amount = nominal * (ratio - 1) = 1M * 0.30 = 300_000.
    // ============================================================================
    {
        Real nominal = 1'000'000.0;
        Period obsLag(3, Months);
        Date startDate(15, January, 2020);
        Date maturity(15, January, 2030);
        Real T = ActualActual(ActualActual::ISDA).yearFraction(startDate, maturity);
        Rate fixedRate = 0.025;
        Real expectedFixedAmount = nominal * (std::pow(1.0 + fixedRate, T) - 1.0);

        auto ukrpi = ext::make_shared<UKRPI>(false);  // not interpolated by default
        // Add the base and observation fixings.
        // ZCIIS uses observationInterpolation = AsIndex (here equiv. Flat),
        // so we store fixings at the start-of-period for the obs date.
        Date baseDate = startDate - obsLag;        // Oct 15, 2019
        Date obsDate = maturity - obsLag;          // Oct 15, 2029 -- future
        // For "growthOnly + flat observation", the swap stores I(0) and I(T).
        // The base fixing must be stored. Future fixings come from a forecast curve;
        // for this probe we build a flat zero-inflation curve at fixed-rate so that
        // forecast I(T)/I(base) matches (1+fixedRate)^T.
        // To keep this deterministic, instead set up a zero-inflation curve.

        // Use UKRPI past fixings: store base at 100.
        std::pair<Date,Date> basePeriod = inflationPeriod(baseDate, ukrpi->frequency());
        ukrpi->addFixing(basePeriod.first, 100.0, true);

        // Build a flat zero-inflation curve at 2.5%. Tied to UKRPI.
        // InterpolatedZeroInflationCurve<Linear>::build requires a pillar list.
        std::vector<Date> pillarDates = {baseDate, baseDate + Period(10, Years), baseDate + Period(20, Years)};
        std::vector<Rate> pillarRates = {0.025, 0.025, 0.025};
        auto zeroCurve = ext::make_shared<InterpolatedZeroInflationCurve<Linear>>(
            evalDate, baseDate, 0.025, ukrpi->frequency(),
            ActualActual(ActualActual::ISDA), TARGET(),
            pillarDates, pillarRates);
        Handle<ZeroInflationTermStructure> zeroHandle(zeroCurve);
        ukrpi = ext::make_shared<UKRPI>(false, zeroHandle);
        ukrpi->addFixing(basePeriod.first, 100.0, true);

        auto swap = ext::make_shared<ZeroCouponInflationSwap>(
            Swap::Payer, nominal, startDate, maturity,
            cal, bdc, ActualActual(ActualActual::ISDA),
            fixedRate, ukrpi, obsLag, CPI::AsIndex,
            false, Calendar(), BusinessDayConvention());

        auto eng = ext::make_shared<DiscountingSwapEngine>(nominalCurve, false);
        swap->setPricingEngine(eng);

        Real fairRate = swap->fairRate();
        Real fixedLegNPV = swap->fixedLegNPV();
        Real inflationLegNPV = swap->inflationLegNPV();
        Real totalNPV = swap->NPV();

        std::cout << "  \"zero_coupon_inflation_swap\": {\n";
        std::cout << "    \"nominal\": " << nominal << ",\n";
        std::cout << "    \"start_serial\": " << startDate.serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << maturity.serialNumber() << ",\n";
        std::cout << "    \"fixed_rate\": " << fixedRate << ",\n";
        std::cout << "    \"observation_lag_months\": 3,\n";
        std::cout << "    \"day_count_year_fraction\": " << T << ",\n";
        std::cout << "    \"expected_fixed_amount\": " << expectedFixedAmount << ",\n";
        std::cout << "    \"fair_rate\": " << fairRate << ",\n";
        std::cout << "    \"fixed_leg_npv\": " << fixedLegNPV << ",\n";
        std::cout << "    \"inflation_leg_npv\": " << inflationLegNPV << ",\n";
        std::cout << "    \"npv\": " << totalNPV << "\n";
        std::cout << "  },\n";
    }

    // ============================================================================
    // YoYInflationBlackCapFloorEngine + Bachelier + UnitDisplacedBlack on a
    // 5y YoY cap @ 2.5% strike, 1M EUR nominal, flat 2.5% YoY curve, flat
    // 3% nominal curve.
    //
    // We construct the cap leg manually using yoyInflationLeg(), wire a flat
    // YoY curve into YYEUHICP, then drive 3 engines.
    // ============================================================================
    {
        Real nominal = 1'000'000.0;
        Rate strike = 0.025;
        Period obsLag(3, Months);
        Date startDate = cal.advance(evalDate, 2, Days);
        Date endDate = cal.advance(startDate, 5, Years);
        Schedule yoySchedule(startDate, endDate, Period(1, Years), cal, bdc, bdc,
                             DateGeneration::Backward, false);

        auto yyeu = ext::make_shared<YYEUHICP>(false);

        // Build a flat YoY curve at 2.5%.
        Date baseDate = startDate - obsLag;
        std::vector<Date> pillarDates = {baseDate, baseDate + Period(10, Years), baseDate + Period(20, Years)};
        std::vector<Rate> pillarRates = {0.025, 0.025, 0.025};
        auto yoyCurve = ext::make_shared<InterpolatedYoYInflationCurve<Linear>>(
            evalDate, baseDate, 0.025, yyeu->frequency(),
            ActualActual(ActualActual::ISDA), TARGET(),
            pillarDates, pillarRates);
        Handle<YoYInflationTermStructure> yoyHandle(yoyCurve);

        auto yyeu2 = ext::make_shared<YYEUHICP>(false, yoyHandle);

        Leg yoyLeg = yoyInflationLeg(yoySchedule, cal, yyeu2, obsLag, CPI::AsIndex)
            .withNotionals(nominal)
            .withPaymentDayCounter(Actual360())
            .withPaymentAdjustment(bdc)
            .withSpreads(0.0)
            .withGearings(1.0);

        std::vector<Rate> capStrikes = {strike};

        // ---- Black engine
        {
            Volatility vol = 0.20;
            Handle<YoYOptionletVolatilitySurface> volSurface(
                ext::make_shared<ConstantYoYOptionletVolatility>(
                    vol, 0, cal, bdc, Actual360(), obsLag, yyeu2->frequency(),
                    yyeu2->interpolated(), -1.0, 100.0,
                    ShiftedLognormal, 0.0));
            auto cap = ext::make_shared<YoYInflationCap>(yoyLeg, capStrikes);
            auto eng = ext::make_shared<YoYInflationBlackCapFloorEngine>(
                yyeu2, volSurface, nominalCurve);
            cap->setPricingEngine(eng);
            Real npv = cap->NPV();
            std::cout << "  \"yoy_inflation_black_cap_5y_2_5pct\": {\n";
            std::cout << "    \"nominal\": " << nominal << ",\n";
            std::cout << "    \"strike\": " << strike << ",\n";
            std::cout << "    \"vol\": " << vol << ",\n";
            std::cout << "    \"yoy_rate\": 0.025,\n";
            std::cout << "    \"nominal_rate\": 0.03,\n";
            std::cout << "    \"npv\": " << npv << "\n";
            std::cout << "  },\n";
        }

        // ---- Bachelier engine
        {
            Volatility vol = 0.005;
            Handle<YoYOptionletVolatilitySurface> volSurface(
                ext::make_shared<ConstantYoYOptionletVolatility>(
                    vol, 0, cal, bdc, Actual360(), obsLag, yyeu2->frequency(),
                    yyeu2->interpolated(), -1.0, 100.0,
                    Normal, 0.0));
            auto cap = ext::make_shared<YoYInflationCap>(yoyLeg, capStrikes);
            auto eng = ext::make_shared<YoYInflationBachelierCapFloorEngine>(
                yyeu2, volSurface, nominalCurve);
            cap->setPricingEngine(eng);
            Real npv = cap->NPV();
            std::cout << "  \"yoy_inflation_bachelier_cap_5y_2_5pct\": {\n";
            std::cout << "    \"vol\": " << vol << ",\n";
            std::cout << "    \"npv\": " << npv << "\n";
            std::cout << "  },\n";
        }

        // ---- UnitDisplacedBlack engine
        {
            Volatility vol = 0.20;
            Handle<YoYOptionletVolatilitySurface> volSurface(
                ext::make_shared<ConstantYoYOptionletVolatility>(
                    vol, 0, cal, bdc, Actual360(), obsLag, yyeu2->frequency(),
                    yyeu2->interpolated(), -1.0, 100.0,
                    ShiftedLognormal, 1.0));
            auto cap = ext::make_shared<YoYInflationCap>(yoyLeg, capStrikes);
            auto eng = ext::make_shared<YoYInflationUnitDisplacedBlackCapFloorEngine>(
                yyeu2, volSurface, nominalCurve);
            cap->setPricingEngine(eng);
            Real npv = cap->NPV();
            std::cout << "  \"yoy_inflation_unit_displaced_black_cap_5y_2_5pct\": {\n";
            std::cout << "    \"vol\": " << vol << ",\n";
            std::cout << "    \"npv\": " << npv << "\n";
            std::cout << "  },\n";
        }

        // ---- Floor variant on Black engine
        {
            Volatility vol = 0.20;
            Handle<YoYOptionletVolatilitySurface> volSurface(
                ext::make_shared<ConstantYoYOptionletVolatility>(
                    vol, 0, cal, bdc, Actual360(), obsLag, yyeu2->frequency(),
                    yyeu2->interpolated(), -1.0, 100.0,
                    ShiftedLognormal, 0.0));
            auto floor = ext::make_shared<YoYInflationFloor>(yoyLeg, capStrikes);
            auto eng = ext::make_shared<YoYInflationBlackCapFloorEngine>(
                yyeu2, volSurface, nominalCurve);
            floor->setPricingEngine(eng);
            Real npv = floor->NPV();
            std::cout << "  \"yoy_inflation_black_floor_5y_2_5pct\": {\n";
            std::cout << "    \"vol\": " << vol << ",\n";
            std::cout << "    \"npv\": " << npv << "\n";
            std::cout << "  }\n";
        }
    }

    std::cout << "}\n";
    return 0;
}
