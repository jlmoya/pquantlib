// L10-B cluster probe — Gaussian1d short-rate cluster.
//
// Captures reference values for the L10-B layer:
//
//   * GsrProcess (constant sigma + constant reversion variant) — checks
//     drift / diffusion / expectation / variance / G(t,T) / y(t) under
//     a degenerate one-step setup that reduces to OU + the
//     forward-measure-time correction.
//
//   * Gsr (constant reversion variant, T=60Y forward measure) on a flat
//     5% curve — zerobond(T, t, y) at multiple (t, T, y) triples;
//     numeraire(t, y) at multiple (t, y) pairs.
//
//   * Gsr.yGrid(stdDevs=4, gridPoints=8, T=1, t=0, y=0) — the AMC grid.
//
//   * Gaussian1dSwaptionVolatility(model, swaption_engine).volatility
//     at one (expiry, tenor) point. Black-implied vol of the engine's
//     ATM NPV (LOOSE — small inversion noise expected).
//
// C++ parity:
//   ql/models/shortrate/onefactormodels/gaussian1dmodel.{hpp,cpp}
//   ql/models/shortrate/onefactormodels/gsr.{hpp,cpp}
//   ql/processes/gsrprocess.{hpp,cpp}
//   ql/processes/gsrprocesscore.{hpp,cpp}
//   ql/termstructures/volatility/swaption/gaussian1dswaptionvolatility.{hpp,cpp}
//   ql/termstructures/volatility/gaussian1dsmilesection.{hpp,cpp}
//   ql/pricingengines/swaption/gaussian1dswaptionengine.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/instruments/makeswaption.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/math/array.hpp>
#include <ql/models/shortrate/onefactormodels/gsr.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/swaption/gaussian1dswaptionengine.hpp>
#include <ql/processes/gsrprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/gaussian1dsmilesection.hpp>
#include <ql/termstructures/volatility/swaption/gaussian1dswaptionvolatility.hpp>
#include <ql/termstructures/volatility/volatilitytype.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // -----------------------------------------------------------------
    // 1) GsrProcess in the affine limit (one step, constant sigma + rev)
    //    -> reduces to OU under the forward measure.
    //
    //    times = []  (no step dates -> only one volatility / reversion
    //                 entry needed). The forward measure horizon T=10.
    //                 sigma = 0.01, reversion = 0.05.
    // -----------------------------------------------------------------
    {
        Real T_fwd = 10.0;
        Array times(0);
        Array vols(1);
        vols[0] = 0.01;
        Array reversions(1);
        reversions[0] = 0.05;

        GsrProcess p(times, vols, reversions, T_fwd);

        std::cout << "  \"gsr_process_const\": {\n";
        std::cout << "    \"sigma\": " << p.sigma(0.5) << ",\n";
        std::cout << "    \"reversion\": " << p.reversion(0.5) << ",\n";
        std::cout << "    \"x0\": " << p.x0() << ",\n";
        std::cout << "    \"diffusion_t_0_5\": " << p.diffusion(0.5, 0.0) << ",\n";
        std::cout << "    \"variance_0_1\": " << p.variance(0.0, 0.0, 1.0) << ",\n";
        std::cout << "    \"variance_0_5\": " << p.variance(0.0, 0.0, 5.0) << ",\n";
        std::cout << "    \"std_deviation_0_5\": "
                  << p.stdDeviation(0.0, 0.0, 5.0) << ",\n";
        std::cout << "    \"y_0\": " << p.y(0.0) << ",\n";
        std::cout << "    \"y_1\": " << p.y(1.0) << ",\n";
        std::cout << "    \"y_5\": " << p.y(5.0) << ",\n";
        std::cout << "    \"G_0_1\": " << p.G(0.0, 1.0, 0.0) << ",\n";
        std::cout << "    \"G_0_5\": " << p.G(0.0, 5.0, 0.0) << ",\n";
        std::cout << "    \"G_1_5\": " << p.G(1.0, 5.0, 0.0) << ",\n";
        std::cout << "    \"expectation_0_0_1\": "
                  << p.expectation(0.0, 0.0, 1.0) << ",\n";
        std::cout << "    \"expectation_1_0_2\": "
                  << p.expectation(1.0, 0.0, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // 2) Gsr full model — constant reversion, flat curve, T=60Y forward
    //    measure (the default). Probe zerobond / numeraire on a grid.
    // -----------------------------------------------------------------
    {
        Date today(15, May, 2026);
        Settings::instance().evaluationDate() = today;

        DayCounter dc = Actual365Fixed();

        Handle<YieldTermStructure> yts(ext::make_shared<FlatForward>(
            today, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dc));

        std::vector<Date> volstepdates = {
            today + Period(1, Years),
            today + Period(2, Years),
            today + Period(5, Years),
        };
        std::vector<Real> volatilities = {0.01, 0.01, 0.01, 0.01};
        Real reversion = 0.05;
        Real T = 60.0;
        Gsr gsr(yts, volstepdates, volatilities, reversion, T);

        std::cout << "  \"gsr_model_const\": {\n";
        std::cout << "    \"sigma_first\": "
                  << gsr.volatility()[0] << ",\n";
        std::cout << "    \"reversion_first\": "
                  << gsr.reversion()[0] << ",\n";
        std::cout << "    \"numeraire_time\": "
                  << gsr.numeraireTime() << ",\n";
        // zerobond(T, t=0, y=0) ≈ curve.discount(T) (only equals it at t=0,
        // y=0 since the model is curve-fitted).
        std::cout << "    \"zerobond_5_0_0\": "
                  << gsr.zerobond(5.0, 0.0, 0.0) << ",\n";
        std::cout << "    \"zerobond_10_0_0\": "
                  << gsr.zerobond(10.0, 0.0, 0.0) << ",\n";
        // zerobond(T, t>0, y) depends on the state. We capture a few.
        std::cout << "    \"zerobond_5_1_0\": "
                  << gsr.zerobond(5.0, 1.0, 0.0) << ",\n";
        std::cout << "    \"zerobond_5_1_0p5\": "
                  << gsr.zerobond(5.0, 1.0, 0.5) << ",\n";
        std::cout << "    \"zerobond_5_1_m0p5\": "
                  << gsr.zerobond(5.0, 1.0, -0.5) << ",\n";
        std::cout << "    \"zerobond_10_2_0p5\": "
                  << gsr.zerobond(10.0, 2.0, 0.5) << ",\n";

        // numeraire(t, y)
        std::cout << "    \"numeraire_0_0\": "
                  << gsr.numeraire(0.0, 0.0) << ",\n";
        std::cout << "    \"numeraire_1_0\": "
                  << gsr.numeraire(1.0, 0.0) << ",\n";
        std::cout << "    \"numeraire_1_0p5\": "
                  << gsr.numeraire(1.0, 0.5) << ",\n";
        std::cout << "    \"numeraire_5_0\": "
                  << gsr.numeraire(5.0, 0.0) << ",\n";
        std::cout << "    \"curve_discount_5\": " << yts->discount(5.0) << ",\n";
        std::cout << "    \"curve_discount_10\": " << yts->discount(10.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // 3) Gaussian1dModel.yGrid (via Gsr; Gaussian1dModel is abstract).
    //
    //    stdDevs = 4, gridPoints = 8, T = 1, t = 0, y = 0
    //    -> 17-point grid.
    // -----------------------------------------------------------------
    {
        Date today(15, May, 2026);
        Settings::instance().evaluationDate() = today;
        DayCounter dc = Actual365Fixed();

        Handle<YieldTermStructure> yts(ext::make_shared<FlatForward>(
            today, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dc));

        std::vector<Date> volstepdates;  // single piece.
        std::vector<Real> volatilities = {0.01};
        Real reversion = 0.05;
        Gsr gsr(yts, volstepdates, volatilities, reversion);

        Array g = gsr.yGrid(4.0, 8, 1.0, 0.0, 0.0);

        std::cout << "  \"gsr_y_grid\": {\n";
        std::cout << "    \"size\": " << g.size() << ",\n";
        std::cout << "    \"values\": [";
        for (Size i = 0; i < g.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << g[i];
        }
        std::cout << "]\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // 4) Gaussian1dSwaptionVolatility — at one (expiry, tenor) point.
    //
    //    EuriborSwapIsdaFixA(10Y) on the same flat 3% curve, GSR with
    //    constant sigma=0.01 / reversion=0.05.
    //    Implied vol of ATM swaption at 5Y expiry.
    //
    //    NOTE: this returns the smile section's volatilityImpl at K=atm
    //          which is the Black implied vol of the model's swaption
    //          NPV (using newton-safe inversion).
    // -----------------------------------------------------------------
    {
        Date today(15, May, 2026);
        Settings::instance().evaluationDate() = today;
        DayCounter dc = Actual365Fixed();

        Handle<YieldTermStructure> yts(ext::make_shared<FlatForward>(
            today, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dc));

        std::vector<Date> volstepdates;
        // Match C++ test-suite gsr.cpp::testGsrModel parametrization
        // (sigma=0.01, reversion=0.01, T=50Y) — the swaption block at the
        // end of that test exercises the same engine pathway we're
        // back-out-inverting here.
        std::vector<Real> volatilities = {0.01};
        Real reversion = 0.01;
        auto gsr = ext::make_shared<Gsr>(
            yts, volstepdates, volatilities, reversion, 50.0);

        auto swapIndex = ext::make_shared<EuriborSwapIsdaFixA>(
            Period(10, Years), yts);

        // engine: 64 grid points, 7 stdevs (matches C++ default)
        auto engine = ext::make_shared<Gaussian1dSwaptionEngine>(
            gsr, 64, 7.0, true, false);

        Gaussian1dSwaptionVolatility svol(
            TARGET(), ModifiedFollowing, swapIndex, gsr, dc, engine);

        Date expiry = TARGET().advance(today, Period(5, Years));

        Real atm = gsr->swapRate(expiry, Period(10, Years), Date(), 0.0, swapIndex);
        Real annuity = gsr->swapAnnuity(expiry, Period(10, Years), Date(), 0.0, swapIndex);
        Real fwd_rate_3M = gsr->forwardRate(
            expiry, Date(), 0.0, ext::make_shared<Euribor>(Period(3, Months), yts));

        // Build the underlying via the swap index and run the engine
        // directly (matches the C++ test-suite gsr.cpp pattern).
        ext::shared_ptr<VanillaSwap> underlying =
            swapIndex->underlyingSwap(expiry);
        ext::shared_ptr<Exercise> ex(new EuropeanExercise(expiry));
        ext::shared_ptr<Swaption> stdSwaption(new Swaption(underlying, ex));
        stdSwaption->setPricingEngine(engine);
        Real swp_npv = stdSwaption->NPV();

        // Build the smile section directly to probe the components.
        auto smileSec = ext::make_shared<Gaussian1dSmileSection>(
            expiry, swapIndex, gsr, dc, engine);

        Real atm_section = smileSec->atmLevel();
        Real opt_price_atm = smileSec->optionPrice(atm, Option::Call);

        // OTM call slightly above atm.
        Real strike_otm = atm + 0.0050;
        Real opt_price_call = smileSec->optionPrice(strike_otm, Option::Call);
        Real vol_otm_call = svol.volatility(expiry, Period(10, Years), strike_otm);

        // OTM put slightly below atm.
        Real strike_itm = atm - 0.0050;
        Real opt_price_put = smileSec->optionPrice(strike_itm, Option::Put);
        Real vol_otm_put = svol.volatility(expiry, Period(10, Years), strike_itm);

        std::cout << "  \"gaussian1d_swaption_vol\": {\n";
        std::cout << "    \"atm_rate\": " << atm << ",\n";
        std::cout << "    \"atm_section\": " << atm_section << ",\n";
        std::cout << "    \"annuity\": " << annuity << ",\n";
        std::cout << "    \"fwd_rate_3m\": " << fwd_rate_3M << ",\n";
        std::cout << "    \"swp_npv_atm_payer\": " << swp_npv << ",\n";
        std::cout << "    \"opt_price_atm\": " << opt_price_atm << ",\n";
        std::cout << "    \"strike_otm_call\": " << strike_otm << ",\n";
        std::cout << "    \"opt_price_call\": " << opt_price_call << ",\n";
        std::cout << "    \"vol_otm_call\": " << vol_otm_call << ",\n";
        std::cout << "    \"strike_otm_put\": " << strike_itm << ",\n";
        std::cout << "    \"opt_price_put\": " << opt_price_put << ",\n";
        std::cout << "    \"vol_otm_put\": " << vol_otm_put << ",\n";
        std::cout << "    \"expiry_serial\": "
                  << expiry.serialNumber() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
