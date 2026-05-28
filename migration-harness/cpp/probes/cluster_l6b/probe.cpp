// L6-B cluster probe: BatesEngine (analytic Bates model engine).
//
// Emits reference values for:
//   * BatesEngine call/put NPV at the Albrecher-Mayer-Schoutens-Tistaert
//     Heston canonical testbed (S=100, K=100, T=1, r=5%, q=0, kappa=2,
//     theta=0.04, sigma=0.3, rho=-0.7, v0=0.04) with Bates jumps off
//     (lambda=0) — must equal AnalyticHestonEngine NPV (reduction test).
//   * BatesEngine call/put NPV at the same testbed with Bates jumps on
//     (lambda=0.1, nu=-0.05, delta=0.1) — locks the Bates jump-CF
//     contribution. C++ uses Gauss-Laguerre order 144.
//   * Skew check: BatesEngine call at K in {90, 100, 110} with jumps on
//     so the implied skew is non-trivial vs the Heston-only counterpart.
//   * BatesEngine call NPV under the Bates 1996 paper's reference
//     scenario (S=100, K=100, T=1, r=5%, q=0, kappa=2, theta=0.04,
//     sigma=0.3, rho=-0.5, v0=0.04, lambda=0.1, nu=-0.05, delta=0.1).
//
// C++ parity: v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/models/equity/batesmodel.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/pricingengines/vanilla/batesengine.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {
    // The canonical Albrecher-Mayer-Schoutens-Tistaert testbed (rho=-0.7).
    struct AMSTSetup {
        Real spot = 100.0;
        Real r = 0.05;
        Real q = 0.0;
        Real v0 = 0.04;
        Real kappa = 2.0;
        Real theta = 0.04;
        Real sigma = 0.3;
        Real rho = -0.7;
    };

    // The Bates 1996 paper testbed (rho=-0.5).
    struct Bates96Setup {
        Real spot = 100.0;
        Real r = 0.05;
        Real q = 0.0;
        Real v0 = 0.04;
        Real kappa = 2.0;
        Real theta = 0.04;
        Real sigma = 0.3;
        Real rho = -0.5;
        Real lambda = 0.1;
        Real nu = -0.05;
        Real delta = 0.1;
    };
}

int main() {
    std::cout << std::setprecision(17);

    Date evalDate(15, June, 2026);
    Settings::instance().evaluationDate() = evalDate;

    Actual365Fixed dc;
    Date expiry = evalDate + 365; // T = 1.0 year exactly under Act/365 Fixed.

    Handle<Quote> spotQuote(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> rfTS(ext::make_shared<FlatForward>(
        evalDate, 0.05, dc, Continuous, Annual));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(
        evalDate, 0.0, dc, Continuous, Annual));

    auto priceUnder = [&](
        const ext::shared_ptr<PricingEngine>& engine,
        Real strike,
        Option::Type type) -> Real {
        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
        auto exercise = ext::make_shared<EuropeanExercise>(expiry);
        VanillaOption opt(payoff, exercise);
        opt.setPricingEngine(engine);
        return opt.NPV();
    };

    std::cout << "{\n";

    // ----------------------------------------------------------------
    // BatesEngine with lambda ~ 0 — reduces to Heston (no jumps).
    // BatesModel rejects lambda=0/delta=0 via PositiveConstraint, so
    // we use 1e-12 as a tractable proxy. The CF jump term scales as
    // t*lambda*(...) so the contribution at lambda=1e-12 is ~1e-12,
    // well under the LOOSE tolerance — for cross-validation, the
    // resulting NPV must match the AnalyticHestonEngine NPV at the
    // same (kappa, theta, sigma, rho, v0).
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        Real lambda_zero = 1e-12;
        Real nu_zero = 0.0;       // mean log-jump can be 0 (NoConstraint).
        Real delta_zero = 1e-12;  // must be > 0 (PositiveConstraint).

        auto batesProc = ext::make_shared<BatesProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho,
            lambda_zero, nu_zero, delta_zero);
        auto batesModel = ext::make_shared<BatesModel>(batesProc);
        auto batesEngine = ext::make_shared<BatesEngine>(batesModel, 144);

        Real call100 = priceUnder(batesEngine, 100.0, Option::Call);
        Real put100  = priceUnder(batesEngine, 100.0, Option::Put);
        Real call80  = priceUnder(batesEngine,  80.0, Option::Call);
        Real call120 = priceUnder(batesEngine, 120.0, Option::Call);

        std::cout << "  \"bates_engine_zero_jump\": {\n";
        std::cout << "    \"lambda\": " << lambda_zero << ",\n";
        std::cout << "    \"nu\": " << nu_zero << ",\n";
        std::cout << "    \"delta\": " << delta_zero << ",\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << ",\n";
        std::cout << "    \"call_otm_low\": " << call80 << ",\n";
        std::cout << "    \"call_otm_high\": " << call120 << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // BatesEngine on AMST testbed + jumps on. Locks the jump-CF
    // contribution against the C++ Gauss-Laguerre reference.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        Real lambda = 0.1;
        Real nu = -0.05;
        Real delta = 0.1;

        auto batesProc = ext::make_shared<BatesProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho,
            lambda, nu, delta);
        auto batesModel = ext::make_shared<BatesModel>(batesProc);
        auto batesEngine = ext::make_shared<BatesEngine>(batesModel, 144);

        Real call100 = priceUnder(batesEngine, 100.0, Option::Call);
        Real put100  = priceUnder(batesEngine, 100.0, Option::Put);
        Real call80  = priceUnder(batesEngine,  80.0, Option::Call);
        Real call120 = priceUnder(batesEngine, 120.0, Option::Call);
        Real call90  = priceUnder(batesEngine,  90.0, Option::Call);
        Real call110 = priceUnder(batesEngine, 110.0, Option::Call);

        std::cout << "  \"bates_engine_amst\": {\n";
        std::cout << "    \"lambda\": " << lambda << ",\n";
        std::cout << "    \"nu\": " << nu << ",\n";
        std::cout << "    \"delta\": " << delta << ",\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << ",\n";
        std::cout << "    \"call_otm_low\": " << call80 << ",\n";
        std::cout << "    \"call_otm_high\": " << call120 << ",\n";
        std::cout << "    \"call_skew_low\": " << call90 << ",\n";
        std::cout << "    \"call_skew_high\": " << call110 << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // BatesEngine on the Bates 1996 paper's reference scenario
    // (rho=-0.5 instead of -0.7).
    // ----------------------------------------------------------------
    {
        Bates96Setup s;
        auto batesProc = ext::make_shared<BatesProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho,
            s.lambda, s.nu, s.delta);
        auto batesModel = ext::make_shared<BatesModel>(batesProc);
        auto batesEngine = ext::make_shared<BatesEngine>(batesModel, 144);

        Real call100 = priceUnder(batesEngine, 100.0, Option::Call);
        Real put100  = priceUnder(batesEngine, 100.0, Option::Put);

        std::cout << "  \"bates_engine_b96\": {\n";
        std::cout << "    \"rho\": " << s.rho << ",\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
