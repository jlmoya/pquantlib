// L6-A cluster probe — LongstaffSchwartz American MC closure.
//
// Captures reference values for:
//
//   * LsmBasisSystem::pathBasisSystem — Monomial / Laguerre / Hermite /
//     Chebyshev2nd at order 0..3, evaluated at a small set of test
//     points. These exercise the QL_GaussianOrthogonalPolynomial
//     weightedValue path that AmericanPathPricer uses. We capture the
//     basis values per polynomial type so the Python port (which
//     re-implements the recurrence rather than relying on
//     numpy.polynomial) can be TIGHT-tier asserted against these.
//
//   * AnalyticEuropeanEngine for an OTM-put baseline (S=36, K=40,
//     r=6%, q=0, sigma=20%, T=1y) — gives ~3.844. Used by tests as
//     the European-baseline cross-check (American > European by the
//     early-exercise premium).
//
//   * MCAmericanEngine via MakeMCAmericanEngine — Longstaff-Schwartz
//     1998 paper Table 1 American put (S=36, K=40, r=6%, q=0,
//     sigma=20%, T=1y, 50 timesteps, Monomial order 2). Calibration
//     and pricing both use PseudoRandom; calibration_samples=2048,
//     samples=4096, seed=42. The Longstaff-Schwartz paper itself
//     reports 4.478 (compared to a binomial benchmark of 4.486).
//
//   * The basis system has order 2 and uses Monomial. We probe a
//     deterministic seed so the Python tests can cross-check NPV ±
//     3-sigma against the C++ result rather than the paper value.
//
// All NPV values are LOOSE-tier (MC sampling + regression variance).
//
// C++ parity:
//   ql/methods/montecarlo/lsmbasissystem.{hpp,cpp},
//   ql/methods/montecarlo/longstaffschwartzpathpricer.hpp,
//   ql/pricingengines/mclongstaffschwartzengine.hpp,
//   ql/pricingengines/vanilla/mcamericanengine.{hpp,cpp},
//   ql/math/integrals/gaussianorthogonalpolynomial.cpp
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/methods/montecarlo/lsmbasissystem.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/mcamericanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

void emitArray(const std::vector<Real>& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

void emitBasisValues(LsmBasisSystem::PolynomialType type, Size order,
                     const std::vector<Real>& xs, const char* label) {
    // pathBasisSystem returns (order+1) basis fcts.
    auto basis = LsmBasisSystem::pathBasisSystem(order, type);
    std::cout << "  \"" << label << "\": {\n";
    std::cout << "    \"order\": " << order << ",\n";
    std::cout << "    \"basis_count\": " << basis.size() << ",\n";
    std::cout << "    \"xs\": ";
    emitArray(xs);
    std::cout << ",\n";
    // Emit a 2D array shape [order+1][xs.size()].
    std::cout << "    \"values\": [\n";
    for (Size i = 0; i < basis.size(); ++i) {
        std::cout << "      [";
        for (Size j = 0; j < xs.size(); ++j) {
            std::cout << basis[i](xs[j]);
            if (j + 1 < xs.size()) std::cout << ", ";
        }
        std::cout << "]";
        if (i + 1 < basis.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "    ]\n";
    std::cout << "  }";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ----------------------------------------------------------------
    // 1) LsmBasisSystem values per polynomial type at order 3
    //    (basis_count = 4: orders 0, 1, 2, 3).
    // ----------------------------------------------------------------
    std::vector<Real> xs_pos{0.5, 1.0, 1.5};
    std::vector<Real> xs_neg{-0.5, 0.0, 0.5};

    emitBasisValues(LsmBasisSystem::Monomial, 3, xs_pos, "lsm_monomial_order_3");
    std::cout << ",\n";
    emitBasisValues(LsmBasisSystem::Laguerre, 3, xs_pos, "lsm_laguerre_order_3");
    std::cout << ",\n";
    emitBasisValues(LsmBasisSystem::Hermite, 3, xs_pos, "lsm_hermite_order_3");
    std::cout << ",\n";
    emitBasisValues(LsmBasisSystem::Chebyshev2nd, 3, xs_neg,
                    "lsm_chebyshev2nd_order_3");
    std::cout << ",\n";

    // ----------------------------------------------------------------
    // 2) European baseline (analytic) — same setup as the LSM 1998 test
    //    S=36, K=40, r=6%, q=0, sigma=20%, T=1y.
    // ----------------------------------------------------------------
    Date today(15, May, 2026);
    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date expiry = today + 365;

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(36.0));
    Handle<YieldTermStructure> rf(
        ext::make_shared<FlatForward>(today, 0.06, dc));
    Handle<YieldTermStructure> q(
        ext::make_shared<FlatForward>(today, 0.0, dc));
    Handle<BlackVolTermStructure> vol(
        ext::make_shared<BlackConstantVol>(today, cal, 0.20, dc));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, q, rf, vol);

    auto putPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 40.0);
    auto euroExercise = ext::make_shared<EuropeanExercise>(expiry);
    auto amExercise =
        ext::make_shared<AmericanExercise>(today, expiry);

    {
        VanillaOption opt(putPayoff, euroExercise);
        opt.setPricingEngine(
            ext::make_shared<AnalyticEuropeanEngine>(process));
        std::cout << "  \"analytic_european_put_lsm1998_setup\": {\n";
        std::cout << "    \"npv\": " << opt.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // 3) MCAmericanEngine — Longstaff-Schwartz 1998 paper setup
    //    50 steps, calibration_samples=2048, samples=4096, seed=42.
    //    Monomial basis, order=2. The paper reports 4.478.
    // ----------------------------------------------------------------
    {
        VanillaOption opt(putPayoff, amExercise);
        opt.setPricingEngine(
            MakeMCAmericanEngine<PseudoRandom>(process)
                .withSteps(50)
                .withSamples(4096)
                .withSeed(42)
                .withCalibrationSamples(2048)
                .withPolynomialOrder(2)
                .withBasisSystem(LsmBasisSystem::Monomial));
        std::cout << "  \"mc_american_put_lsm1998_monomial_o2\": {\n";
        std::cout << "    \"npv\": " << opt.NPV() << ",\n";
        std::cout << "    \"error\": " << opt.errorEstimate() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // 4) Same setup, but with Laguerre order=2 — different basis.
    // ----------------------------------------------------------------
    {
        VanillaOption opt(putPayoff, amExercise);
        opt.setPricingEngine(
            MakeMCAmericanEngine<PseudoRandom>(process)
                .withSteps(50)
                .withSamples(4096)
                .withSeed(42)
                .withCalibrationSamples(2048)
                .withPolynomialOrder(2)
                .withBasisSystem(LsmBasisSystem::Laguerre));
        std::cout << "  \"mc_american_put_lsm1998_laguerre_o2\": {\n";
        std::cout << "    \"npv\": " << opt.NPV() << ",\n";
        std::cout << "    \"error\": " << opt.errorEstimate() << "\n";
        std::cout << "  },\n";
    }

    // ----------------------------------------------------------------
    // 5) Deep-OTM call (S=36, K=80) — should be ~0.
    // ----------------------------------------------------------------
    {
        auto callPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 80.0);
        VanillaOption opt(callPayoff, amExercise);
        opt.setPricingEngine(
            MakeMCAmericanEngine<PseudoRandom>(process)
                .withSteps(50)
                .withSamples(4096)
                .withSeed(42)
                .withCalibrationSamples(2048)
                .withPolynomialOrder(2)
                .withBasisSystem(LsmBasisSystem::Monomial));
        std::cout << "  \"mc_american_deep_otm_call\": {\n";
        std::cout << "    \"npv\": " << opt.NPV() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
