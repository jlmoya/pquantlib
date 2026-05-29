// W1-B cluster probe — Gaussian1d engines:
//
//   * Gaussian1dCapFloorEngine on a 5y cap@2.5% under Gsr(sigma=0.01,
//     reversion=0.05) and flat 3% curve.
//   * Gaussian1dCapFloorEngine sanity: same cap under Gsr(sigma≈0,
//     reversion=0.05) → matches the deterministic forward-curve cap
//     NPV (discounted intrinsic).
//   * Gaussian1dFloatFloatSwaptionEngine on a 5y-into-5y synthetic
//     float-float swap (Euribor6M payer vs Euribor3M with spread).
//   * Gaussian1dNonstandardSwaptionEngine on an amortizing payer
//     swaption (5y-into-5y, half-decay nominal schedule).
//
// C++ parity:
//   ql/pricingengines/capfloor/gaussian1dcapfloorengine.{hpp,cpp}
//   ql/pricingengines/swaption/gaussian1dfloatfloatswaptionengine.{hpp,cpp}
//   ql/pricingengines/swaption/gaussian1dnonstandardswaptionengine.{hpp,cpp}
//   ql/instruments/floatfloatswap.{hpp,cpp}
//   ql/instruments/floatfloatswaption.{hpp,cpp}
//   ql/instruments/nonstandardswap.{hpp,cpp}
//   ql/instruments/nonstandardswaption.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/instruments/capfloor.hpp>
#include <ql/instruments/floatfloatswap.hpp>
#include <ql/instruments/floatfloatswaption.hpp>
#include <ql/instruments/nonstandardswap.hpp>
#include <ql/instruments/nonstandardswaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/instruments/makevanillaswap.hpp>
#include <ql/instruments/makecapfloor.hpp>
#include <ql/math/array.hpp>
#include <ql/models/shortrate/onefactormodels/gsr.hpp>
#include <ql/pricingengines/capfloor/gaussian1dcapfloorengine.hpp>
#include <ql/pricingengines/swaption/gaussian1dfloatfloatswaptionengine.hpp>
#include <ql/pricingengines/swaption/gaussian1dnonstandardswaptionengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    Date today(15, May, 2026);
    Settings::instance().evaluationDate() = today;
    DayCounter dc365 = Actual365Fixed();
    DayCounter dc360 = Actual360();

    Handle<YieldTermStructure> yts(ext::make_shared<FlatForward>(
        today, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dc365));

    auto euribor3M = ext::make_shared<Euribor>(Period(3, Months), yts);
    auto euribor6M = ext::make_shared<Euribor>(Period(6, Months), yts);

    // -----------------------------------------------------------------
    // 1) Gaussian1dCapFloorEngine — 5y cap @ 2.5% on Euribor3M.
    //    Gsr(sigma=0.01, reversion=0.05), single piece, T=60Y.
    // -----------------------------------------------------------------
    {
        std::vector<Date> volstepdates;  // single piece
        std::vector<Real> volatilities = {0.01};
        Real reversion = 0.05;
        auto gsr = ext::make_shared<Gsr>(
            yts, volstepdates, volatilities, reversion, 60.0);

        // Build a 5y cap starting 3M-forward (avoid past-fixing complication).
        Calendar cal = TARGET();
        Date startDate = cal.advance(today, 3, Months);
        Date endDate = cal.advance(startDate, 5, Years);

        Schedule sched(
            startDate, endDate, Period(3, Months), cal,
            ModifiedFollowing, ModifiedFollowing,
            DateGeneration::Backward, false);

        Leg floatLeg = IborLeg(sched, euribor3M)
                         .withNotionals(1000000.0)
                         .withPaymentDayCounter(euribor3M->dayCounter())
                         .withPaymentAdjustment(ModifiedFollowing);

        auto cap = ext::make_shared<Cap>(floatLeg, std::vector<Rate>{0.025});

        auto engine = ext::make_shared<Gaussian1dCapFloorEngine>(
            gsr, 64, 7.0, true, false);
        cap->setPricingEngine(engine);

        Real cap_npv = cap->NPV();

        std::cout << "  \"gaussian1d_capfloor\": {\n";
        std::cout << "    \"cap_strike\": 0.025,\n";
        std::cout << "    \"cap_notional\": 1000000.0,\n";
        std::cout << "    \"gsr_sigma\": 0.01,\n";
        std::cout << "    \"gsr_reversion\": 0.05,\n";
        std::cout << "    \"flat_rate\": 0.03,\n";
        std::cout << "    \"integration_points\": 64,\n";
        std::cout << "    \"stddevs\": 7.0,\n";
        std::cout << "    \"cap_npv\": " << cap_npv << ",\n";
        std::cout << "    \"start_serial\": " << startDate.serialNumber() << ",\n";
        std::cout << "    \"end_serial\": " << endDate.serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // 2) Gaussian1dCapFloorEngine sanity: Gsr(sigma~=1e-8) → matches
    //    the deterministic per-coupon discounted intrinsic.
    // -----------------------------------------------------------------
    {
        std::vector<Date> volstepdates;
        std::vector<Real> volatilities = {1e-8};
        Real reversion = 0.05;
        auto gsr = ext::make_shared<Gsr>(
            yts, volstepdates, volatilities, reversion, 60.0);

        Calendar cal = TARGET();
        Date startDate = cal.advance(today, 3, Months);
        Date endDate = cal.advance(startDate, 5, Years);
        Schedule sched(
            startDate, endDate, Period(3, Months), cal,
            ModifiedFollowing, ModifiedFollowing,
            DateGeneration::Backward, false);

        // Use a low strike so cap is deep-ITM under flat-forward 3%
        // -> the deterministic forward intrinsic dominates.
        Real strike = 0.01;
        Leg floatLeg = IborLeg(sched, euribor3M)
                         .withNotionals(1000000.0)
                         .withPaymentDayCounter(euribor3M->dayCounter())
                         .withPaymentAdjustment(ModifiedFollowing);

        auto cap = ext::make_shared<Cap>(floatLeg, std::vector<Rate>{strike});
        auto engine = ext::make_shared<Gaussian1dCapFloorEngine>(
            gsr, 64, 7.0, true, false);
        cap->setPricingEngine(engine);
        Real cap_zero_vol = cap->NPV();

        // Reference: per-period intrinsic
        // sum_i (max(F_i - K, 0) * tau_i * P(t_pay_i)) * N
        Real intrinsic = 0.0;
        for (const auto& cf : floatLeg) {
            auto coupon = ext::dynamic_pointer_cast<IborCoupon>(cf);
            if (coupon == nullptr) continue;
            Real fwd = coupon->indexFixing();
            Real tau = coupon->accrualPeriod();
            Real disc = yts->discount(coupon->date());
            Real payoff = std::max(fwd - strike, 0.0);
            intrinsic += payoff * tau * disc * coupon->nominal();
        }

        std::cout << "  \"gaussian1d_capfloor_zerovol\": {\n";
        std::cout << "    \"strike\": " << strike << ",\n";
        std::cout << "    \"cap_npv\": " << cap_zero_vol << ",\n";
        std::cout << "    \"deterministic_intrinsic\": " << intrinsic << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // 3) Gaussian1dFloatFloatSwaptionEngine — 5y-into-5y synthetic
    //    float-float swap (Euribor3M receiver leg vs Euribor6M payer
    //    leg with negative spread). Notional 1M, equal payment day
    //    counters; european exercise at 5y.
    // -----------------------------------------------------------------
    {
        std::vector<Date> volstepdates;
        std::vector<Real> volatilities = {0.01};
        Real reversion = 0.05;
        auto gsr = ext::make_shared<Gsr>(
            yts, volstepdates, volatilities, reversion, 60.0);

        Calendar cal = TARGET();
        Date startDate = cal.advance(today, 5, Years);
        Date endDate = cal.advance(startDate, 5, Years);

        Schedule sched1(
            startDate, endDate, Period(6, Months), cal,
            ModifiedFollowing, ModifiedFollowing,
            DateGeneration::Backward, false);
        Schedule sched2(
            startDate, endDate, Period(3, Months), cal,
            ModifiedFollowing, ModifiedFollowing,
            DateGeneration::Backward, false);

        Real nominal = 1000000.0;
        auto floatFloat = ext::make_shared<FloatFloatSwap>(
            Swap::Payer,
            nominal, nominal,
            sched1, euribor6M, dc360,
            sched2, euribor3M, dc360,
            false, false,
            1.0, 0.0, Null<Real>(), Null<Real>(),
            1.0, -0.0050, Null<Real>(), Null<Real>(),
            ext::nullopt, ext::nullopt);

        Date exerciseDate = startDate;
        auto exercise = ext::make_shared<EuropeanExercise>(exerciseDate);
        auto swaption = ext::make_shared<FloatFloatSwaption>(
            floatFloat, exercise);

        auto engine = ext::make_shared<Gaussian1dFloatFloatSwaptionEngine>(
            gsr, 32, 5.0, true, false);
        swaption->setPricingEngine(engine);

        Real ff_npv = swaption->NPV();

        std::cout << "  \"gaussian1d_floatfloat_swaption\": {\n";
        std::cout << "    \"nominal\": " << nominal << ",\n";
        std::cout << "    \"spread2\": -0.0050,\n";
        std::cout << "    \"gsr_sigma\": 0.01,\n";
        std::cout << "    \"gsr_reversion\": 0.05,\n";
        std::cout << "    \"integration_points\": 32,\n";
        std::cout << "    \"stddevs\": 5.0,\n";
        std::cout << "    \"exercise_serial\": " << exerciseDate.serialNumber() << ",\n";
        std::cout << "    \"ff_npv\": " << ff_npv << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // 4) Gaussian1dNonstandardSwaptionEngine — 5y-into-5y amortizing
    //    payer swaption (fixed rate 3.0%, notional decays linearly to
    //    half), European exercise.
    // -----------------------------------------------------------------
    {
        std::vector<Date> volstepdates;
        std::vector<Real> volatilities = {0.01};
        Real reversion = 0.05;
        auto gsr = ext::make_shared<Gsr>(
            yts, volstepdates, volatilities, reversion, 60.0);

        Calendar cal = TARGET();
        Date startDate = cal.advance(today, 5, Years);
        Date endDate = cal.advance(startDate, 5, Years);

        Schedule fixedSched(
            startDate, endDate, Period(1, Years), cal,
            ModifiedFollowing, ModifiedFollowing,
            DateGeneration::Backward, false);
        Schedule floatSched(
            startDate, endDate, Period(6, Months), cal,
            ModifiedFollowing, ModifiedFollowing,
            DateGeneration::Backward, false);

        // Amortizing notional: 1M, 0.9M, 0.8M, 0.7M, 0.6M, 0.5M on the
        // fixed leg (one entry per fixed period). The floating leg
        // gets the corresponding per-fixed-period notional repeated
        // for each sub-period.
        Size nFixed = fixedSched.size() - 1;
        std::vector<Real> fixedNominal(nFixed), fixedRate(nFixed, 0.03);
        for (Size i = 0; i < nFixed; ++i) {
            fixedNominal[i] = 1000000.0 * (1.0 - 0.1 * static_cast<Real>(i));
        }
        Size nFloat = floatSched.size() - 1;
        std::vector<Real> floatNominal(nFloat), floatSpread(nFloat, 0.0),
                          floatGearing(nFloat, 1.0);
        // Float coupons span 6M each; fixed are 1Y. Two float coupons
        // per fixed bucket.
        for (Size i = 0; i < nFloat; ++i) {
            Size fixIdx = i / 2;
            if (fixIdx >= nFixed) fixIdx = nFixed - 1;
            floatNominal[i] = fixedNominal[fixIdx];
        }

        auto ns = ext::make_shared<NonstandardSwap>(
            Swap::Payer,
            fixedNominal, floatNominal,
            fixedSched, fixedRate, dc360,
            floatSched, euribor6M,
            floatGearing, floatSpread,
            dc360,
            false, false,
            ext::optional<BusinessDayConvention>(ModifiedFollowing));

        Date exerciseDate = startDate;
        auto exercise = ext::make_shared<EuropeanExercise>(exerciseDate);
        auto swaption = ext::make_shared<NonstandardSwaption>(
            ns, exercise);

        auto engine = ext::make_shared<Gaussian1dNonstandardSwaptionEngine>(
            gsr, 32, 5.0, true, false);
        swaption->setPricingEngine(engine);

        Real ns_npv = swaption->NPV();

        std::cout << "  \"gaussian1d_nonstandard_swaption\": {\n";
        std::cout << "    \"fixed_rate\": 0.03,\n";
        std::cout << "    \"initial_notional\": 1000000.0,\n";
        std::cout << "    \"gsr_sigma\": 0.01,\n";
        std::cout << "    \"gsr_reversion\": 0.05,\n";
        std::cout << "    \"integration_points\": 32,\n";
        std::cout << "    \"stddevs\": 5.0,\n";
        std::cout << "    \"exercise_serial\": " << exerciseDate.serialNumber() << ",\n";
        std::cout << "    \"ns_npv\": " << ns_npv << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
