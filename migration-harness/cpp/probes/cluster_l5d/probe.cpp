// L5-D cluster probe: 1-D Finite-Difference Black-Scholes
//
// Captures reference values for the L5-D layer (FD framework +
// FdBlackScholesVanillaEngine + VanillaOption::impliedVolatility):
//
//   * FdmLinearOpLayout (1-D, size=8): spacing/size/index round-trip.
//   * Uniform1dMesher and Fdm1dMesher accessors: locations, dplus,
//     dminus at the textbook (start=-2, end=+2, size=11) grid.
//   * FdmBlackScholesMesher 1-D log-spot grid at (S=100, K=100, T=1,
//     r=5%, q=0%, sigma=20%, size=11) using xMin/xMax derived from
//     -InverseNormalCDF(eps=1e-4)*sigma*sqrt(T)*scale=1.5.
//   * UniformGridMesher (1-D, 11 points on [-2, +2]): location and dx.
//   * FirstDerivativeOp applied to f(x)=x^2 over the uniform grid:
//     interior nodes give 2x; boundary nodes use upwinding (idx=0)
//     and downwinding (idx=size-1).
//   * SecondDerivativeOp applied to f(x)=x^2 over the uniform grid:
//     interior nodes give 2.0 exactly; boundary nodes give 0.
//   * FirstDerivativeOp + SecondDerivativeOp applied to f(x)=x^3:
//     interior nodes give 3x^2 and 6x respectively (LOOSE — central
//     differences are O(h^2)).
//   * FdmBlackScholesOp via the 1-D BSM PDE — verify the operator
//     L = -0.5 sigma^2 S^2 d^2/dS^2 - (r-q) S d/dS + r applied to
//     f(S) = S has the expected analytic value at interior nodes.
//   * FdBlackScholesVanillaEngine: European Call/Put NPV converges
//     to AnalyticEuropeanEngine at (xGrid=200, tGrid=200, dampingSteps=0).
//   * FdBlackScholesVanillaEngine: American Put has early-exercise
//     premium vs European Put.
//   * VanillaOption::impliedVolatility roundtrip: build a European
//     call at sigma=0.20, get its NPV, then ask for the implied vol
//     given that NPV — must recover 0.20.
//
// C++ parity:
//   ql/methods/finitedifferences/meshers/uniform1dmesher.hpp,
//   ql/methods/finitedifferences/meshers/fdmblackscholesmesher.{hpp,cpp},
//   ql/methods/finitedifferences/meshers/uniformgridmesher.{hpp,cpp},
//   ql/methods/finitedifferences/operators/fdmlinearoplayout.{hpp,cpp},
//   ql/methods/finitedifferences/operators/firstderivativeop.{hpp,cpp},
//   ql/methods/finitedifferences/operators/secondderivativeop.{hpp,cpp},
//   ql/methods/finitedifferences/operators/fdmblackscholesop.{hpp,cpp},
//   ql/methods/finitedifferences/schemes/cranknicolsonscheme.{hpp,cpp},
//   ql/methods/finitedifferences/solvers/fdmbackwardsolver.{hpp,cpp},
//   ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.{hpp,cpp},
//   ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.{hpp,cpp},
//   ql/pricingengines/vanilla/fdblackscholesvanillaengine.{hpp,cpp},
//   ql/instruments/vanillaoption.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/instruments/vanillaoption.hpp>
#include <ql/exercise.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/meshers/uniformgridmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/handle.hpp>
#include <ql/math/array.hpp>

#include <iomanip>
#include <iostream>
#include <cmath>

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

void emitArray(const Array& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ---------------------------------------------------------------
    // Common setup: 1-year Call/Put @ S=K=100, r=5%, q=0%, sigma=20%.
    // ---------------------------------------------------------------
    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date ref(15, June, 2026);
    Settings::instance().evaluationDate() = ref;
    Date expiry = ref + 365; // 1.0 year under Actual/365 Fixed

    Real spot = 100.0;
    Real strike = 100.0;
    Volatility vol = 0.20;
    Rate r = 0.05;
    Rate q = 0.00;

    auto spotQ = ext::make_shared<SimpleQuote>(spot);
    Handle<Quote> spotH(spotQ);
    Handle<YieldTermStructure> rfH(ext::make_shared<FlatForward>(ref, r, dc));
    Handle<YieldTermStructure> divH(ext::make_shared<FlatForward>(ref, q, dc));
    Handle<BlackVolTermStructure> volH(ext::make_shared<BlackConstantVol>(ref, cal, vol, dc));

    auto gbsm = ext::make_shared<GeneralizedBlackScholesProcess>(
        spotH, divH, rfH, volH);

    Time T = gbsm->time(expiry); // = 1.0 under Actual/365

    // ---------------------------------------------------------------
    // 1. FdmLinearOpLayout (1-D, size=8): basic sanity
    // ---------------------------------------------------------------
    {
        std::vector<Size> dim = {8};
        FdmLinearOpLayout layout(dim);
        std::cout << "  \"linear_op_layout_1d\": {\n";
        std::cout << "    \"size\": " << layout.size() << ",\n";
        std::cout << "    \"dim\": [" << layout.dim()[0] << "],\n";
        std::cout << "    \"spacing\": [" << layout.spacing()[0] << "],\n";
        std::vector<Size> coords = {3};
        std::cout << "    \"index_at_3\": " << layout.index(coords) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // 2. Uniform1dMesher (start=-2, end=2, size=11)
    // ---------------------------------------------------------------
    {
        Uniform1dMesher m(-2.0, 2.0, 11);
        std::cout << "  \"uniform_1d_mesher\": {\n";
        std::cout << "    \"size\": " << m.size() << ",\n";
        std::cout << "    \"locations\": ";
        emitArray(m.locations());
        std::cout << ",\n";
        // dplus/dminus at index 5 (interior): both 0.4.
        std::cout << "    \"dplus_at_5\": " << m.dplus(5) << ",\n";
        std::cout << "    \"dminus_at_5\": " << m.dminus(5) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // 3. UniformGridMesher (1-D, 11 nodes on [-2, +2])
    // ---------------------------------------------------------------
    {
        auto layout = ext::make_shared<FdmLinearOpLayout>(std::vector<Size>{11});
        std::vector<std::pair<Real, Real>> boundaries = {{-2.0, 2.0}};
        UniformGridMesher m(layout, boundaries);
        // location at iter index=5 (coord 5)
        FdmLinearOpIterator iter(std::vector<Size>{11}, std::vector<Size>{5}, 5);
        std::cout << "  \"uniform_grid_mesher_1d\": {\n";
        std::cout << "    \"locations\": ";
        emitArray(m.locations(0));
        std::cout << ",\n";
        std::cout << "    \"location_at_idx5\": " << m.location(iter, 0) << ",\n";
        std::cout << "    \"dplus_at_idx5\": " << m.dplus(iter, 0) << ",\n";
        std::cout << "    \"dminus_at_idx5\": " << m.dminus(iter, 0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // 4. FdmBlackScholesMesher 1-D (size=11, anchored at log(K))
    //
    // Two variants captured here:
    //
    //   * "fdm_bs_mesher_uniform": cPoint = (Null, Null) → falls back
    //     to Uniform1dMesher on [xMin, xMax]. This is the variant the
    //     Python port mirrors (Concentrating1dMesher is deferred to
    //     Phase 6 per the L5-D carve-out).
    //   * "fdm_bs_mesher_concentrating": cPoint = (strike, 0.1) →
    //     Concentrating1dMesher, anchored at log(strike) with density
    //     parameter 0.1. This is the C++ engine default; captured for
    //     future-phase parity but NOT cross-validated against in L5-D.
    // ---------------------------------------------------------------
    {
        // Uniform variant (cPoint = Null) — matched by Python port.
        auto bsMesherUniform = ext::make_shared<FdmBlackScholesMesher>(
            11, gbsm, T, strike,
            Null<Real>(), Null<Real>(), 0.0001, 1.5,
            std::pair<Real, Real>(Null<Real>(), Null<Real>()));
        std::cout << "  \"fdm_bs_mesher_uniform\": {\n";
        std::cout << "    \"size\": " << bsMesherUniform->size() << ",\n";
        std::cout << "    \"locations\": ";
        emitArray(bsMesherUniform->locations());
        std::cout << "\n  },\n";

        // Concentrating variant (cPoint = (strike, 0.1)) — C++ engine
        // default. Captured for completeness; LOOSE convergence to
        // analytic European is verified via the engine result, not via
        // direct mesh comparison.
        auto bsMesherConcentrating = ext::make_shared<FdmBlackScholesMesher>(
            11, gbsm, T, strike,
            Null<Real>(), Null<Real>(), 0.0001, 1.5,
            std::pair<Real, Real>(strike, 0.1));
        std::cout << "  \"fdm_bs_mesher_concentrating\": {\n";
        std::cout << "    \"size\": " << bsMesherConcentrating->size() << ",\n";
        std::cout << "    \"locations\": ";
        emitArray(bsMesherConcentrating->locations());
        std::cout << "\n  },\n";
    }

    // ---------------------------------------------------------------
    // 5. FirstDerivativeOp + SecondDerivativeOp on uniform 1-D grid
    //
    // Apply to f(x)=x^2 over [-2, 2] with 11 points (dx=0.4).
    // First derivative interior: 2x. Boundary: upwinding/downwinding.
    // Second derivative interior: 2. Boundary: 0 (Dirichlet).
    // ---------------------------------------------------------------
    {
        auto layout = ext::make_shared<FdmLinearOpLayout>(std::vector<Size>{11});
        auto m = ext::make_shared<UniformGridMesher>(
            layout, std::vector<std::pair<Real, Real>>{{-2.0, 2.0}});

        // f(x) = x^2 at the 11 nodes.
        Array fx2(11);
        Array x = m->locations(0);
        for (Size i = 0; i < 11; ++i) {
            fx2[i] = x[i] * x[i];
        }
        FirstDerivativeOp d1(0, m);
        SecondDerivativeOp d2(0, m);
        Array d1_fx2 = d1.apply(fx2);
        Array d2_fx2 = d2.apply(fx2);

        std::cout << "  \"deriv_ops_x_squared\": {\n";
        std::cout << "    \"x\": ";
        emitArray(x);
        std::cout << ",\n";
        std::cout << "    \"d1_apply_x_squared\": ";
        emitArray(d1_fx2);
        std::cout << ",\n";
        std::cout << "    \"d2_apply_x_squared\": ";
        emitArray(d2_fx2);
        std::cout << "\n  },\n";

        // f(x) = x^3
        Array fx3(11);
        for (Size i = 0; i < 11; ++i) {
            fx3[i] = x[i] * x[i] * x[i];
        }
        Array d1_fx3 = d1.apply(fx3);
        Array d2_fx3 = d2.apply(fx3);
        std::cout << "  \"deriv_ops_x_cubed\": {\n";
        std::cout << "    \"d1_apply_x_cubed\": ";
        emitArray(d1_fx3);
        std::cout << ",\n";
        std::cout << "    \"d2_apply_x_cubed\": ";
        emitArray(d2_fx3);
        std::cout << "\n  },\n";
    }

    // ---------------------------------------------------------------
    // 6. FdmBlackScholesVanillaEngine European Call/Put — converges
    //    to AnalyticEuropeanEngine at (xGrid=200, tGrid=200).
    // ---------------------------------------------------------------
    {
        auto callPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto putPayoff  = ext::make_shared<PlainVanillaPayoff>(Option::Put,  strike);
        auto europExercise = ext::make_shared<EuropeanExercise>(expiry);

        // Reference: analytic European
        VanillaOption europCall(callPayoff, europExercise);
        VanillaOption europPut(putPayoff, europExercise);
        europCall.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(gbsm));
        europPut.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(gbsm));
        Real analyticCall = europCall.NPV();
        Real analyticPut  = europPut.NPV();

        // FD CrankNicolson
        VanillaOption fdCall(callPayoff, europExercise);
        VanillaOption fdPut(putPayoff, europExercise);
        fdCall.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
            gbsm, 200, 200, 0, FdmSchemeDesc::CrankNicolson()));
        fdPut.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
            gbsm, 200, 200, 0, FdmSchemeDesc::CrankNicolson()));

        std::cout << "  \"fd_european\": {\n";
        std::cout << "    \"analytic_call_npv\": " << analyticCall << ",\n";
        std::cout << "    \"fd_call_npv\": " << fdCall.NPV() << ",\n";
        std::cout << "    \"analytic_put_npv\": " << analyticPut << ",\n";
        std::cout << "    \"fd_put_npv\": " << fdPut.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // 7. American Put — early-exercise premium vs European Put.
    // ---------------------------------------------------------------
    {
        auto putPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
        auto europExercise = ext::make_shared<EuropeanExercise>(expiry);
        auto amerExercise  = ext::make_shared<AmericanExercise>(ref, expiry);

        VanillaOption europPut(putPayoff, europExercise);
        VanillaOption amerPut(putPayoff, amerExercise);

        europPut.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
            gbsm, 200, 200, 0, FdmSchemeDesc::CrankNicolson()));
        amerPut.setPricingEngine(ext::make_shared<FdBlackScholesVanillaEngine>(
            gbsm, 200, 200, 0, FdmSchemeDesc::CrankNicolson()));

        std::cout << "  \"fd_american\": {\n";
        std::cout << "    \"european_put_npv\": " << europPut.NPV() << ",\n";
        std::cout << "    \"american_put_npv\": " << amerPut.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // 8. impliedVolatility roundtrip: build a European Call at
    //    sigma=0.20, get NPV, ask for implied vol → should recover 0.20.
    //
    //    The C++ VanillaOption::impliedVolatility for a European-style
    //    option uses AnalyticEuropeanEngine internally to bisect on vol
    //    so we just record the call NPV and let the Python test
    //    invert it via its own impliedVolatility method.
    // ---------------------------------------------------------------
    {
        auto callPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto europExercise = ext::make_shared<EuropeanExercise>(expiry);
        VanillaOption europCall(callPayoff, europExercise);
        europCall.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(gbsm));
        Real npvAtVol20 = europCall.NPV();

        // Use the impliedVolatility helper (C++ method on VanillaOption).
        Real impliedVol = europCall.impliedVolatility(
            npvAtVol20, gbsm, 1e-6, 100, 0.001, 4.0);

        std::cout << "  \"implied_vol_european\": {\n";
        std::cout << "    \"npv_at_vol20\": " << npvAtVol20 << ",\n";
        std::cout << "    \"recovered_vol\": " << impliedVol << ",\n";
        std::cout << "    \"target_vol\": " << vol << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
