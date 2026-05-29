// W1-C cluster probe: Bates variant models + analytic engines.
//
// Emits reference values for:
//
//   * BatesDetJumpModel — Bates with deterministic jump intensity
//     (kappaLambda, thetaLambda) layered on the lognormal-jump base.
//     Lock the calibration-arg vector (theta, kappa, sigma, rho, v0,
//     nu, delta, lambda, kappaLambda, thetaLambda) + the underlying
//     BatesProcess accessors after construction.
//
//   * BatesDoubleExpModel — double-exponential jump distribution.
//     Lock the calibration-arg vector (theta, kappa, sigma, rho, v0,
//     p, nuDown, nuUp, lambda).
//
//   * BatesDoubleExpDetJumpModel — double-exp + deterministic
//     intensity. Lock the arg vector (theta, kappa, sigma, rho, v0,
//     p, nuDown, nuUp, lambda, kappaLambda, thetaLambda).
//
//   * AnalyticBatesDetJumpEngine — call/put NPV at the AMST testbed
//     and Bates 1996 testbed; reduction check at kappaLambda=t large
//     should collapse to BatesEngine (deterministic intensity decays
//     to its mean fast).
//
//   * AnalyticBatesDoubleExpEngine — call/put NPV at the AMST testbed
//     under double-exponential jumps.
//
//   * Degenerate-jump reduction: BatesDetJumpEngine with lambda ~ 0
//     should collapse to pure Heston (NPV matches AnalyticHestonEngine).
//
// C++ parity: v1.42.1 (099987f0).
//   ql/models/equity/batesmodel.{hpp,cpp}
//   ql/pricingengines/vanilla/batesengine.{hpp,cpp}

#include <ql/exercise.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/models/equity/batesmodel.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/pricingengines/vanilla/batesengine.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {
    // Albrecher-Mayer-Schoutens-Tistaert testbed (rho=-0.7).
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

    // Lognormal-jump add-on (used by BatesDetJump).
    struct LogJumps {
        Real lambda = 0.1;
        Real nu = -0.05;
        Real delta = 0.1;
    };

    // Double-exponential-jump add-on.
    struct DoubleExpJumps {
        Real lambda = 0.1;
        Real nuUp = 0.05;
        Real nuDown = 0.05;
        Real p = 0.5;
    };

    // Deterministic-jump-intensity OU params.
    struct DetIntensity {
        Real kappaLambda = 1.0;
        Real thetaLambda = 0.1;
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
    // BatesDetJumpModel — calibration-arg vector + accessors.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        LogJumps j;
        DetIntensity d;

        auto batesProc = ext::make_shared<BatesProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho,
            j.lambda, j.nu, j.delta);
        auto detModel = ext::make_shared<BatesDetJumpModel>(
            batesProc, d.kappaLambda, d.thetaLambda);

        std::cout << "  \"bates_det_jump_model\": {\n";
        std::cout << "    \"n_args\": " << detModel->params().size() << ",\n";
        std::cout << "    \"theta\": " << detModel->theta() << ",\n";
        std::cout << "    \"kappa\": " << detModel->kappa() << ",\n";
        std::cout << "    \"sigma\": " << detModel->sigma() << ",\n";
        std::cout << "    \"rho\": " << detModel->rho() << ",\n";
        std::cout << "    \"v0\": " << detModel->v0() << ",\n";
        std::cout << "    \"nu\": " << detModel->nu() << ",\n";
        std::cout << "    \"delta\": " << detModel->delta() << ",\n";
        std::cout << "    \"lambda\": " << detModel->lambda() << ",\n";
        std::cout << "    \"kappaLambda\": " << detModel->kappaLambda() << ",\n";
        std::cout << "    \"thetaLambda\": " << detModel->thetaLambda() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // BatesDoubleExpModel — calibration-arg vector + accessors.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        DoubleExpJumps j;

        // BatesDoubleExpModel takes a HestonProcess (NOT a BatesProcess —
        // jumps are parameterized at the model level, not the process).
        auto hestonProc = ext::make_shared<HestonProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho);
        auto deModel = ext::make_shared<BatesDoubleExpModel>(
            hestonProc, j.lambda, j.nuUp, j.nuDown, j.p);

        std::cout << "  \"bates_double_exp_model\": {\n";
        std::cout << "    \"n_args\": " << deModel->params().size() << ",\n";
        std::cout << "    \"theta\": " << deModel->theta() << ",\n";
        std::cout << "    \"kappa\": " << deModel->kappa() << ",\n";
        std::cout << "    \"sigma\": " << deModel->sigma() << ",\n";
        std::cout << "    \"rho\": " << deModel->rho() << ",\n";
        std::cout << "    \"v0\": " << deModel->v0() << ",\n";
        std::cout << "    \"p\": " << deModel->p() << ",\n";
        std::cout << "    \"nuDown\": " << deModel->nuDown() << ",\n";
        std::cout << "    \"nuUp\": " << deModel->nuUp() << ",\n";
        std::cout << "    \"lambda\": " << deModel->lambda() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // BatesDoubleExpDetJumpModel — calibration-arg vector + accessors.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        DoubleExpJumps j;
        DetIntensity d;

        auto hestonProc = ext::make_shared<HestonProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho);
        auto dedModel = ext::make_shared<BatesDoubleExpDetJumpModel>(
            hestonProc, j.lambda, j.nuUp, j.nuDown, j.p,
            d.kappaLambda, d.thetaLambda);

        std::cout << "  \"bates_double_exp_det_jump_model\": {\n";
        std::cout << "    \"n_args\": " << dedModel->params().size() << ",\n";
        std::cout << "    \"theta\": " << dedModel->theta() << ",\n";
        std::cout << "    \"kappa\": " << dedModel->kappa() << ",\n";
        std::cout << "    \"sigma\": " << dedModel->sigma() << ",\n";
        std::cout << "    \"rho\": " << dedModel->rho() << ",\n";
        std::cout << "    \"v0\": " << dedModel->v0() << ",\n";
        std::cout << "    \"p\": " << dedModel->p() << ",\n";
        std::cout << "    \"nuDown\": " << dedModel->nuDown() << ",\n";
        std::cout << "    \"nuUp\": " << dedModel->nuUp() << ",\n";
        std::cout << "    \"lambda\": " << dedModel->lambda() << ",\n";
        std::cout << "    \"kappaLambda\": " << dedModel->kappaLambda() << ",\n";
        std::cout << "    \"thetaLambda\": " << dedModel->thetaLambda() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // AnalyticBatesDetJumpEngine on AMST testbed + jumps on.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        LogJumps j;
        DetIntensity d;

        auto batesProc = ext::make_shared<BatesProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho,
            j.lambda, j.nu, j.delta);
        auto detModel = ext::make_shared<BatesDetJumpModel>(
            batesProc, d.kappaLambda, d.thetaLambda);
        auto detEngine = ext::make_shared<BatesDetJumpEngine>(detModel, 144);

        Real call100 = priceUnder(detEngine, 100.0, Option::Call);
        Real put100  = priceUnder(detEngine, 100.0, Option::Put);
        Real call80  = priceUnder(detEngine,  80.0, Option::Call);
        Real call120 = priceUnder(detEngine, 120.0, Option::Call);
        Real call90  = priceUnder(detEngine,  90.0, Option::Call);
        Real call110 = priceUnder(detEngine, 110.0, Option::Call);

        std::cout << "  \"bates_det_jump_engine_amst\": {\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << ",\n";
        std::cout << "    \"call_otm_low\": " << call80 << ",\n";
        std::cout << "    \"call_otm_high\": " << call120 << ",\n";
        std::cout << "    \"call_skew_low\": " << call90 << ",\n";
        std::cout << "    \"call_skew_high\": " << call110 << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // AnalyticBatesDoubleExpEngine on AMST testbed + jumps on.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        DoubleExpJumps j;

        auto hestonProc = ext::make_shared<HestonProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho);
        auto deModel = ext::make_shared<BatesDoubleExpModel>(
            hestonProc, j.lambda, j.nuUp, j.nuDown, j.p);
        auto deEngine = ext::make_shared<BatesDoubleExpEngine>(deModel, 144);

        Real call100 = priceUnder(deEngine, 100.0, Option::Call);
        Real put100  = priceUnder(deEngine, 100.0, Option::Put);
        Real call80  = priceUnder(deEngine,  80.0, Option::Call);
        Real call120 = priceUnder(deEngine, 120.0, Option::Call);
        Real call90  = priceUnder(deEngine,  90.0, Option::Call);
        Real call110 = priceUnder(deEngine, 110.0, Option::Call);

        std::cout << "  \"bates_double_exp_engine_amst\": {\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << ",\n";
        std::cout << "    \"call_otm_low\": " << call80 << ",\n";
        std::cout << "    \"call_otm_high\": " << call120 << ",\n";
        std::cout << "    \"call_skew_low\": " << call90 << ",\n";
        std::cout << "    \"call_skew_high\": " << call110 << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // AnalyticBatesDoubleExpDetJumpEngine on AMST testbed + jumps on.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        DoubleExpJumps j;
        DetIntensity d;

        auto hestonProc = ext::make_shared<HestonProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho);
        auto dedModel = ext::make_shared<BatesDoubleExpDetJumpModel>(
            hestonProc, j.lambda, j.nuUp, j.nuDown, j.p,
            d.kappaLambda, d.thetaLambda);
        auto dedEngine = ext::make_shared<BatesDoubleExpDetJumpEngine>(dedModel, 144);

        Real call100 = priceUnder(dedEngine, 100.0, Option::Call);
        Real put100  = priceUnder(dedEngine, 100.0, Option::Put);
        Real call90  = priceUnder(dedEngine,  90.0, Option::Call);
        Real call110 = priceUnder(dedEngine, 110.0, Option::Call);

        std::cout << "  \"bates_double_exp_det_jump_engine_amst\": {\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << ",\n";
        std::cout << "    \"call_skew_low\": " << call90 << ",\n";
        std::cout << "    \"call_skew_high\": " << call110 << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // BatesDetJump degenerate reduction: lambda~0 → pure Heston.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;
        Real lambda_zero = 1e-12;
        Real nu_zero = 0.0;
        Real delta_zero = 1e-12;
        DetIntensity d;

        auto batesProc = ext::make_shared<BatesProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho,
            lambda_zero, nu_zero, delta_zero);
        auto detModel = ext::make_shared<BatesDetJumpModel>(
            batesProc, d.kappaLambda, d.thetaLambda);
        auto detEngine = ext::make_shared<BatesDetJumpEngine>(detModel, 144);

        Real call100 = priceUnder(detEngine, 100.0, Option::Call);
        Real put100  = priceUnder(detEngine, 100.0, Option::Put);

        std::cout << "  \"bates_det_jump_engine_zero_jump\": {\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // Heston-only reference for the zero-jump check above.
    // ----------------------------------------------------------------
    {
        AMSTSetup s;

        auto hestonProc = ext::make_shared<HestonProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho);
        auto hestonModel = ext::make_shared<HestonModel>(hestonProc);
        auto hestonEngine = ext::make_shared<AnalyticHestonEngine>(
            hestonModel, 144);

        Real call100 = priceUnder(hestonEngine, 100.0, Option::Call);
        Real put100  = priceUnder(hestonEngine, 100.0, Option::Put);

        std::cout << "  \"heston_engine_reference\": {\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": "  << put100  << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
