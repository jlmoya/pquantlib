// L4-C cluster probe: HestonProcess + HestonModel + AnalyticHestonEngine
// + BatesProcess + HestonModelHelper roundtrip.
//
// Emits reference values for:
//   * HestonProcess scalars (drift/diffusion at fixed (t,x)).
//   * HestonModel parameters after construction (theta/kappa/sigma/rho/v0).
//   * AnalyticHestonEngine call+put NPV at (S=100,K=100,T=1,r=5%,q=0,
//     kappa=2, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04) — the
//     Albrecher-Mayer-Schoutens-Tistaert standard test.
//   * AnalyticHestonEngine NPV smile at K in {80, 100, 120}.
//   * BatesProcess parameters (lambda, nu, delta, m).
//   * BatesProcess drift jump correction (drift[0] -= lambda*m).
//   * AnalyticHestonEngine call NPV under Bates parameters (no jump
//     CF — Heston engine on a Bates process simply ignores the jump
//     terms; full BatesEngine is out of scope. So this checks that the
//     "Heston with Bates params" reduces to plain Heston given lambda=0,
//     and also gives a divergent value with lambda>0 so we can lock the
//     Heston-only behaviour vs the BatesEngine reference deferred to L5).
//   * HestonModelHelper market value via Black formula at vol=0.20.
//
// C++ parity: v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/models/equity/batesmodel.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/models/equity/hestonmodelhelper.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {
    // The canonical Albrecher-Mayer-Schoutens-Tistaert testbed.
    struct Setup {
        Real spot = 100.0;
        Real r = 0.05;
        Real q = 0.0;
        Real v0 = 0.04;
        Real kappa = 2.0;
        Real theta = 0.04;
        Real sigma = 0.3;
        Real rho = -0.7;
        Real T = 1.0; // years; we'll set up a 365-day expiry.
    };
}

int main() {
    std::cout << std::setprecision(17);

    Date evalDate(15, June, 2026);
    Settings::instance().evaluationDate() = evalDate;

    Actual365Fixed dc;
    Calendar cal = NullCalendar();
    Date expiry = evalDate + 365; // T = 1.0 year exactly under Act/365 Fixed.

    Setup s;

    Handle<Quote> spotQuote(ext::make_shared<SimpleQuote>(s.spot));
    Handle<YieldTermStructure> rfTS(ext::make_shared<FlatForward>(
        evalDate, s.r, dc, Continuous, Annual));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(
        evalDate, s.q, dc, Continuous, Annual));

    auto process = ext::make_shared<HestonProcess>(
        rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho);

    std::cout << "{\n";

    // ----------------------------------------------------------------
    // HestonProcess scalars
    // ----------------------------------------------------------------
    {
        Array x(2);
        x[0] = s.spot;
        x[1] = s.v0;

        Array drift = process->drift(0.0, x);
        Matrix diff = process->diffusion(0.0, x);
        Array initial = process->initialValues();
        Array applied = process->apply(x, Array(2, 0.01)); // (0.01, 0.01) increment

        std::cout << "  \"process\": {\n";
        std::cout << "    \"size\": " << process->size() << ",\n";
        std::cout << "    \"factors\": " << process->factors() << ",\n";
        std::cout << "    \"initial_s\": " << initial[0] << ",\n";
        std::cout << "    \"initial_v\": " << initial[1] << ",\n";
        std::cout << "    \"drift_s\": " << drift[0] << ",\n";
        std::cout << "    \"drift_v\": " << drift[1] << ",\n";
        std::cout << "    \"diffusion_00\": " << diff[0][0] << ",\n";
        std::cout << "    \"diffusion_01\": " << diff[0][1] << ",\n";
        std::cout << "    \"diffusion_10\": " << diff[1][0] << ",\n";
        std::cout << "    \"diffusion_11\": " << diff[1][1] << ",\n";
        std::cout << "    \"apply_s\": " << applied[0] << ",\n";
        std::cout << "    \"apply_v\": " << applied[1] << ",\n";
        std::cout << "    \"time_to_expiry\": " << process->time(expiry) << ",\n";
        std::cout << "    \"v0\": " << process->v0() << ",\n";
        std::cout << "    \"kappa\": " << process->kappa() << ",\n";
        std::cout << "    \"theta\": " << process->theta() << ",\n";
        std::cout << "    \"sigma\": " << process->sigma() << ",\n";
        std::cout << "    \"rho\": " << process->rho() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // HestonModel
    // ----------------------------------------------------------------
    auto model = ext::make_shared<HestonModel>(process);
    {
        std::cout << "  \"heston_model\": {\n";
        std::cout << "    \"theta\": " << model->theta() << ",\n";
        std::cout << "    \"kappa\": " << model->kappa() << ",\n";
        std::cout << "    \"sigma\": " << model->sigma() << ",\n";
        std::cout << "    \"rho\": " << model->rho() << ",\n";
        std::cout << "    \"v0\": " << model->v0() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // AnalyticHestonEngine — Gauss-Laguerre integration, order 144.
    // Pin cpxLog to Gatheral so we have a deterministic, simple algorithm
    // for the Python port to match.
    // ----------------------------------------------------------------
    auto engine = ext::make_shared<AnalyticHestonEngine>(
        model,
        AnalyticHestonEngine::Gatheral,
        AnalyticHestonEngine::Integration::gaussLaguerre(144));

    auto priceVanilla = [&](Real strike, Option::Type type) -> Real {
        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
        auto exercise = ext::make_shared<EuropeanExercise>(expiry);
        VanillaOption opt(payoff, exercise);
        opt.setPricingEngine(engine);
        return opt.NPV();
    };

    {
        Real call100 = priceVanilla(100.0, Option::Call);
        Real put100 = priceVanilla(100.0, Option::Put);
        Real call80 = priceVanilla(80.0, Option::Call);
        Real call120 = priceVanilla(120.0, Option::Call);
        Real put80 = priceVanilla(80.0, Option::Put);
        Real put120 = priceVanilla(120.0, Option::Put);

        std::cout << "  \"heston_engine\": {\n";
        std::cout << "    \"call_atm\": " << call100 << ",\n";
        std::cout << "    \"put_atm\": " << put100 << ",\n";
        std::cout << "    \"call_otm_low\": " << call80 << ",\n";
        std::cout << "    \"call_otm_high\": " << call120 << ",\n";
        std::cout << "    \"put_otm_low\": " << put80 << ",\n";
        std::cout << "    \"put_otm_high\": " << put120 << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // HestonModelHelper — vol quote 0.20 at K=100, T=1.
    // Black market price for the helper's blackPrice(0.20).
    // ----------------------------------------------------------------
    {
        Period maturityPeriod = Period(12, Months);
        Handle<Quote> volQuote(ext::make_shared<SimpleQuote>(0.20));
        HestonModelHelper helper(
            maturityPeriod, cal, spotQuote, 100.0,
            volQuote, rfTS, qTS,
            BlackCalibrationHelper::RelativePriceError);
        helper.setPricingEngine(engine);

        // Trigger calculation cascade so market_value works.
        Real bp = helper.blackPrice(0.20);
        Real mv = helper.marketValue();
        Real model_val = helper.modelValue();
        Real err = helper.calibrationError();

        std::cout << "  \"heston_helper\": {\n";
        std::cout << "    \"black_price_v20\": " << bp << ",\n";
        std::cout << "    \"market_value\": " << mv << ",\n";
        std::cout << "    \"model_value\": " << model_val << ",\n";
        std::cout << "    \"calibration_error\": " << err << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // BatesProcess parameters + drift correction.
    // ----------------------------------------------------------------
    {
        Real lambda = 0.1;
        Real nu = -0.05;
        Real delta = 0.1;

        auto batesProc = ext::make_shared<BatesProcess>(
            rfTS, qTS, spotQuote, s.v0, s.kappa, s.theta, s.sigma, s.rho,
            lambda, nu, delta);

        Array x(2);
        x[0] = s.spot;
        x[1] = s.v0;
        Array drift = batesProc->drift(0.0, x);

        Real m_expected = std::exp(nu + 0.5 * delta * delta) - 1.0;

        std::cout << "  \"bates_process\": {\n";
        std::cout << "    \"size\": " << batesProc->size() << ",\n";
        std::cout << "    \"factors\": " << batesProc->factors() << ",\n";
        std::cout << "    \"lambda\": " << batesProc->lambda() << ",\n";
        std::cout << "    \"nu\": " << batesProc->nu() << ",\n";
        std::cout << "    \"delta\": " << batesProc->delta() << ",\n";
        std::cout << "    \"m\": " << m_expected << ",\n";
        std::cout << "    \"drift_s\": " << drift[0] << ",\n";
        std::cout << "    \"drift_v\": " << drift[1] << "\n";
        std::cout << "  },\n";

        // BatesModel — wraps the BatesProcess; parameters carry over.
        auto batesModel = ext::make_shared<BatesModel>(batesProc);

        std::cout << "  \"bates_model\": {\n";
        std::cout << "    \"theta\": " << batesModel->theta() << ",\n";
        std::cout << "    \"kappa\": " << batesModel->kappa() << ",\n";
        std::cout << "    \"sigma\": " << batesModel->sigma() << ",\n";
        std::cout << "    \"rho\": " << batesModel->rho() << ",\n";
        std::cout << "    \"v0\": " << batesModel->v0() << ",\n";
        std::cout << "    \"nu\": " << batesModel->nu() << ",\n";
        std::cout << "    \"delta\": " << batesModel->delta() << ",\n";
        std::cout << "    \"lambda\": " << batesModel->lambda() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
