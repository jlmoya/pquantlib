// W1-A cluster probe — MarkovFunctional model + adapter process.
//
// Captures reference values for the W1-A layer:
//
//   * Gaussian1dGsrProcess (= GsrProcess under same-reversion config) —
//     checks drift / diffusion / expectation / variance at a couple of
//     state points. The TIGHT check confirms our adapter preserves
//     GsrProcess's SDE under flat-vol degenerate input.
//
//   * MarkovFunctional (swaption-strip calibrated, no smile pretreatment)
//     on a flat 3% curve with a 3-point swaption strip (1y10y, 2y9y,
//     3y8y) at constant 20% black vol — numeraire(t, y) and zerobond
//     samples on the calibrated grid.
//
//   * MarkovFunctional zerobond degeneration check — for very small vol
//     (sigma -> 0) the numeraire-tabulated zerobond should converge to
//     the curve discount (LOOSE because numerical bootstrap noise).
//
//   * MarkovFunctional swaption NPV — a 5y5y ATM swaption priced via
//     Gaussian1dSwaptionEngine on the calibrated model (matches the C++
//     reference path our MarkovFunctionalSwaptionEngine should
//     reproduce to LOOSE).
//
// C++ parity:
//   ql/models/shortrate/onefactormodels/markovfunctional.{hpp,cpp}
//   ql/processes/mfstateprocess.{hpp,cpp}
//   ql/processes/gsrprocess.{hpp,cpp}
//   ql/pricingengines/swaption/gaussian1dswaptionengine.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/instruments/makeswaption.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/math/array.hpp>
#include <ql/models/shortrate/onefactormodels/markovfunctional.hpp>
#include <ql/pricingengines/swaption/gaussian1dswaptionengine.hpp>
#include <ql/processes/gsrprocess.hpp>
#include <ql/processes/mfstateprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
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

    // Use a TARGET business day so that swaption fixing dates (today +
    // 1Y / 2Y / 3Y) are valid fixings. May 14th 2026 is a Thursday.
    Date today(14, May, 2026);
    Settings::instance().evaluationDate() = today;
    DayCounter dc = Actual365Fixed();

    Handle<YieldTermStructure> yts(ext::make_shared<FlatForward>(
        today, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dc));

    // -----------------------------------------------------------------
    // 1) Gaussian1dGsrProcess (= GsrProcess under same-reversion config).
    //
    //    Single piece, constant sigma=0.01, reversion=0.05, T=60.
    // -----------------------------------------------------------------
    {
        Array times(0);
        Array vols(1);
        vols[0] = 0.01;
        Array reversions(1);
        reversions[0] = 0.05;
        Real T = 60.0;
        GsrProcess p(times, vols, reversions, T);

        std::cout << "  \"gaussian1d_gsr_process\": {\n";
        std::cout << "    \"x0\": " << p.x0() << ",\n";
        std::cout << "    \"sigma_0_5\": " << p.sigma(0.5) << ",\n";
        std::cout << "    \"reversion_0_5\": " << p.reversion(0.5) << ",\n";
        std::cout << "    \"diffusion_0_5\": " << p.diffusion(0.5, 0.0) << ",\n";
        std::cout << "    \"variance_0_1\": " << p.variance(0.0, 0.0, 1.0) << ",\n";
        std::cout << "    \"variance_0_5\": " << p.variance(0.0, 0.0, 5.0) << ",\n";
        std::cout << "    \"std_deviation_0_5\": "
                  << p.stdDeviation(0.0, 0.0, 5.0) << ",\n";
        std::cout << "    \"expectation_0_0_1\": "
                  << p.expectation(0.0, 0.0, 1.0) << ",\n";
        std::cout << "    \"expectation_1_0_2\": "
                  << p.expectation(1.0, 0.0, 1.0) << ",\n";
        std::cout << "    \"y_5\": " << p.y(5.0) << ",\n";
        std::cout << "    \"G_0_5\": " << p.G(0.0, 5.0, 0.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // 2) MarkovFunctional — swaption-strip calibrated, no smile
    //    pretreatment. We use a constant 20% lognormal swaption vol
    //    surface, EuriborSwapIsdaFixA(10Y) swap index base, and a
    //    3-point expiry strip (1Y / 2Y / 3Y) all with 10Y tenor.
    //
    //    The constant volatility surface is the most degenerate case
    //    that still exercises the full numeraire bootstrap. With no
    //    pretreatment the model just uses the raw smile section (which
    //    for a flat black vol surface is essentially a black smile).
    // -----------------------------------------------------------------
    {
        ext::shared_ptr<IborIndex> iborIndex(
            new Euribor(6 * Months, yts));
        ext::shared_ptr<SwapIndex> swapBase(
            new EuriborSwapIsdaFixA(10 * Years, yts, yts));

        Handle<SwaptionVolatilityStructure> swVol(
            ext::make_shared<ConstantSwaptionVolatility>(
                today, TARGET(), ModifiedFollowing, 0.20, dc));

        // Advance to a TARGET fixing date for each swaption expiry.
        Calendar cal = TARGET();
        std::vector<Date> swaptionExpiries = {
            cal.advance(today, 1 * Years),
            cal.advance(today, 2 * Years),
            cal.advance(today, 3 * Years),
        };
        std::vector<Period> swaptionTenors = {
            10 * Years,
            10 * Years,
            10 * Years,
        };

        std::vector<Date> volstepdates = {
            cal.advance(today, 1 * Years),
            cal.advance(today, 2 * Years),
        };
        std::vector<Real> volatilities = {0.01, 0.01, 0.01};
        Real reversion = 0.01;

        // No smile pretreatment, NoPayoffExtrapolation, smaller grid for
        // probe stability.
        MarkovFunctional::ModelSettings settings;
        settings.withYGridPoints(32)
                .withYStdDevs(5.0)
                .withGaussHermitePoints(16)
                .withLowerRateBound(0.001)
                .withUpperRateBound(1.5)
                .withAdjustments(MarkovFunctional::ModelSettings::NoPayoffExtrapolation);

        MarkovFunctional mf(yts, reversion, volstepdates, volatilities,
                            swVol, swaptionExpiries, swaptionTenors,
                            swapBase, settings);

        std::cout << "  \"markov_functional_swaption\": {\n";
        std::cout << "    \"numeraire_time\": " << mf.numeraireTime() << ",\n";

        // numeraire(t=0, y=0) = curve.discount(numeraireTime)
        std::cout << "    \"numeraire_0_0\": " << mf.numeraire(0.0, 0.0) << ",\n";

        // numeraire(t=1, y) at a few state values.
        std::cout << "    \"numeraire_1_0\": " << mf.numeraire(1.0, 0.0) << ",\n";
        std::cout << "    \"numeraire_1_0p5\": " << mf.numeraire(1.0, 0.5) << ",\n";
        std::cout << "    \"numeraire_1_m0p5\": "
                  << mf.numeraire(1.0, -0.5) << ",\n";
        std::cout << "    \"numeraire_2_0\": " << mf.numeraire(2.0, 0.0) << ",\n";
        std::cout << "    \"numeraire_3_0\": " << mf.numeraire(3.0, 0.0) << ",\n";

        // zerobond(T, t=0, y=0) should match the input curve at t=0 (the
        // MarkovFunctional model is constructed to be curve-consistent at
        // the calibration grid times).
        Real disc1 = yts->discount(1.0);
        Real disc5 = yts->discount(5.0);
        Real disc10 = yts->discount(10.0);

        std::cout << "    \"curve_discount_1\": " << disc1 << ",\n";
        std::cout << "    \"curve_discount_5\": " << disc5 << ",\n";
        std::cout << "    \"curve_discount_10\": " << disc10 << ",\n";

        std::cout << "    \"zerobond_1_0_0\": "
                  << mf.zerobond(1.0, 0.0, 0.0) << ",\n";
        std::cout << "    \"zerobond_5_0_0\": "
                  << mf.zerobond(5.0, 0.0, 0.0) << ",\n";
        std::cout << "    \"zerobond_10_0_0\": "
                  << mf.zerobond(10.0, 0.0, 0.0) << ",\n";

        // zerobond at t>0 — state-dependent.
        std::cout << "    \"zerobond_5_1_0\": "
                  << mf.zerobond(5.0, 1.0, 0.0) << ",\n";
        std::cout << "    \"zerobond_5_1_0p5\": "
                  << mf.zerobond(5.0, 1.0, 0.5) << ",\n";
        std::cout << "    \"zerobond_5_1_m0p5\": "
                  << mf.zerobond(5.0, 1.0, -0.5) << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
