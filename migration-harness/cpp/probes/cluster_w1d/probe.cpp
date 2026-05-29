// Phase 11 W1-D cluster probe: Heston SLV + GjrGarch + time-dependent Heston.
//
// Captures reference values for:
//
//   * GJRGARCHProcess constants (CumNorm(lambda), drift / diffusion
//     entries at (V=v0, lambda=0.2, alpha=0.05, beta=0.85, gamma=0.05))
//     at t=0.5y.
//   * GJRGARCHModel.{omega, alpha, beta, gamma, lambda, v0}.
//   * AnalyticGJRGARCHEngine European call+put NPV at ATM 1y, with the
//     parameters from Duan et al. (2006).
//   * PiecewiseTimeDependentHestonModel: constant-parameter degenerate
//     case (theta/kappa/sigma/rho all equal across segments) NPV ATM
//     1y matches AnalyticHestonEngine NPV at the same parameters.
//   * AnalyticPTDHestonEngine NPV at a 2-segment piecewise setup
//     (long/short-vol break at 0.5y).
//
// C++ parity:
//   ql/processes/gjrgarchprocess.{hpp,cpp}
//   ql/models/equity/gjrgarchmodel.{hpp,cpp}
//   ql/pricingengines/vanilla/analyticgjrgarchengine.{hpp,cpp}
//   ql/models/equity/piecewisetimedependenthestonmodel.{hpp,cpp}
//   ql/pricingengines/vanilla/analyticptdhestonengine.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/models/equity/gjrgarchmodel.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/models/equity/piecewisetimedependenthestonmodel.hpp>
#include <ql/models/parameter.hpp>
#include <ql/pricingengines/vanilla/analyticgjrgarchengine.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/pricingengines/vanilla/analyticptdhestonengine.hpp>
#include <ql/processes/gjrgarchprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    Date evalDate(15, June, 2026);
    Settings::instance().evaluationDate() = evalDate;
    Actual365Fixed dc;

    // ===================================================================
    // 1) GJR-GARCH process + model + analytic engine.
    // ===================================================================
    {
        // Reference parameters (Duan et al. 2006 testbed, adapted to
        // daily constants annualized via daysPerYear=252).
        const Real spot = 100.0;
        const Real v0 = 0.000160;     // daily v0
        const Real omega = 0.000002;  // daily omega
        const Real alpha = 0.024;
        const Real beta = 0.93;
        const Real gamma = 0.059;
        const Real lambda = 0.2;
        const Real daysPerYear = 252.0;
        const Real r = 0.05;
        const Real q = 0.00;

        Handle<Quote> spotQuote(ext::make_shared<SimpleQuote>(spot));
        Handle<YieldTermStructure> rfTS(
            ext::make_shared<FlatForward>(evalDate, r, dc, Continuous, Annual));
        Handle<YieldTermStructure> qTS(
            ext::make_shared<FlatForward>(evalDate, q, dc, Continuous, Annual));

        auto process = ext::make_shared<GJRGARCHProcess>(
            rfTS, qTS, spotQuote, v0, omega, alpha, beta, gamma, lambda,
            daysPerYear, GJRGARCHProcess::FullTruncation);

        // Process drift / diffusion at (t=0.5y, x=(spot, daysPerYear*v0)).
        Array x(2);
        x[0] = spot;
        x[1] = daysPerYear * v0;
        Array d = process->drift(0.5, x);
        Matrix diff = process->diffusion(0.5, x);

        std::cout << "  \"gjr_garch_process\": {\n";
        std::cout << "    \"daysPerYear\": " << daysPerYear << ",\n";
        std::cout << "    \"v0\": " << v0 << ",\n";
        std::cout << "    \"omega\": " << omega << ",\n";
        std::cout << "    \"alpha\": " << alpha << ",\n";
        std::cout << "    \"beta\": " << beta << ",\n";
        std::cout << "    \"gamma\": " << gamma << ",\n";
        std::cout << "    \"lambda\": " << lambda << ",\n";
        std::cout << "    \"initial_values_S\": " << spot << ",\n";
        std::cout << "    \"initial_values_V\": " << (daysPerYear * v0) << ",\n";
        std::cout << "    \"drift_S\": " << d[0] << ",\n";
        std::cout << "    \"drift_V\": " << d[1] << ",\n";
        std::cout << "    \"diffusion_00\": " << diff[0][0] << ",\n";
        std::cout << "    \"diffusion_01\": " << diff[0][1] << ",\n";
        std::cout << "    \"diffusion_10\": " << diff[1][0] << ",\n";
        std::cout << "    \"diffusion_11\": " << diff[1][1] << "\n";
        std::cout << "  },\n";

        // Model parameter accessors.
        auto model = ext::make_shared<GJRGARCHModel>(process);

        std::cout << "  \"gjr_garch_model\": {\n";
        std::cout << "    \"omega\": " << model->omega() << ",\n";
        std::cout << "    \"alpha\": " << model->alpha() << ",\n";
        std::cout << "    \"beta\": " << model->beta() << ",\n";
        std::cout << "    \"gamma\": " << model->gamma() << ",\n";
        std::cout << "    \"lambda\": " << model->lambda() << ",\n";
        std::cout << "    \"v0\": " << model->v0() << "\n";
        std::cout << "  },\n";

        // AnalyticGJRGARCHEngine NPV at ATM 1y call + put.
        auto engine = ext::make_shared<AnalyticGJRGARCHEngine>(model);
        Date expiry = evalDate + 365;
        auto exercise = ext::make_shared<EuropeanExercise>(expiry);
        auto callPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, spot);
        auto putPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Put, spot);

        VanillaOption callOption(callPayoff, exercise);
        callOption.setPricingEngine(engine);

        VanillaOption putOption(putPayoff, exercise);
        putOption.setPricingEngine(engine);

        std::cout << "  \"analytic_gjr_garch_engine\": {\n";
        std::cout << "    \"call_atm_1y\": " << callOption.NPV() << ",\n";
        std::cout << "    \"put_atm_1y\": " << putOption.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ===================================================================
    // 2) PiecewiseTimeDependentHestonModel + AnalyticPTDHestonEngine.
    // ===================================================================
    {
        const Real spot = 100.0;
        const Real v0 = 0.04;
        const Real r = 0.05;
        const Real q = 0.00;

        Handle<Quote> spotQuote(ext::make_shared<SimpleQuote>(spot));
        Handle<YieldTermStructure> rfTS(
            ext::make_shared<FlatForward>(evalDate, r, dc, Continuous, Annual));
        Handle<YieldTermStructure> qTS(
            ext::make_shared<FlatForward>(evalDate, q, dc, Continuous, Annual));

        Date expiry = evalDate + 365;
        auto exercise = ext::make_shared<EuropeanExercise>(expiry);
        auto callPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, spot);
        auto putPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Put, spot);

        // --- (a) Degenerate "all-equal" piecewise → matches plain Heston NPV.
        {
            // Time grid 0 → 0.5 → 1.0.
            std::vector<Time> times = {0.5, 1.0};
            TimeGrid timeGrid(times.begin(), times.end());

            // PiecewiseConstantParameter expects times = break points
            // (vector of length n-1); arguments are length n.
            std::vector<Time> breaks = {0.5};
            PiecewiseConstantParameter thetaParam(breaks, PositiveConstraint());
            thetaParam.setParam(0, 0.04);
            thetaParam.setParam(1, 0.04);
            PiecewiseConstantParameter kappaParam(breaks, PositiveConstraint());
            kappaParam.setParam(0, 2.0);
            kappaParam.setParam(1, 2.0);
            PiecewiseConstantParameter sigmaParam(breaks, PositiveConstraint());
            sigmaParam.setParam(0, 0.3);
            sigmaParam.setParam(1, 0.3);
            PiecewiseConstantParameter rhoParam(
                breaks, BoundaryConstraint(-1.0, 1.0));
            rhoParam.setParam(0, -0.7);
            rhoParam.setParam(1, -0.7);

            auto ptdModel =
                ext::make_shared<PiecewiseTimeDependentHestonModel>(
                    rfTS, qTS, spotQuote, v0, thetaParam, kappaParam,
                    sigmaParam, rhoParam, timeGrid);

            auto ptdEngine =
                ext::make_shared<AnalyticPTDHestonEngine>(ptdModel, 144);

            VanillaOption call(callPayoff, exercise);
            call.setPricingEngine(ptdEngine);
            VanillaOption put(putPayoff, exercise);
            put.setPricingEngine(ptdEngine);

            std::cout << "  \"ptd_heston_degenerate\": {\n";
            std::cout << "    \"theta_seg0\": " << ptdModel->theta(0.25) << ",\n";
            std::cout << "    \"kappa_seg0\": " << ptdModel->kappa(0.25) << ",\n";
            std::cout << "    \"sigma_seg0\": " << ptdModel->sigma(0.25) << ",\n";
            std::cout << "    \"rho_seg0\": " << ptdModel->rho(0.25) << ",\n";
            std::cout << "    \"theta_seg1\": " << ptdModel->theta(0.75) << ",\n";
            std::cout << "    \"v0\": " << ptdModel->v0() << ",\n";
            std::cout << "    \"s0\": " << ptdModel->s0() << ",\n";
            std::cout << "    \"call_atm_1y\": " << call.NPV() << ",\n";
            std::cout << "    \"put_atm_1y\": " << put.NPV() << "\n";
            std::cout << "  },\n";

            // Reference: plain AnalyticHestonEngine NPV at same params.
            auto plainProcess = ext::make_shared<HestonProcess>(
                rfTS, qTS, spotQuote, v0, 2.0, 0.04, 0.3, -0.7);
            auto plainModel = ext::make_shared<HestonModel>(plainProcess);
            auto plainEngine =
                ext::make_shared<AnalyticHestonEngine>(plainModel, 144);

            VanillaOption plainCall(callPayoff, exercise);
            plainCall.setPricingEngine(plainEngine);
            VanillaOption plainPut(putPayoff, exercise);
            plainPut.setPricingEngine(plainEngine);

            std::cout << "  \"plain_heston_reference\": {\n";
            std::cout << "    \"call_atm_1y\": " << plainCall.NPV() << ",\n";
            std::cout << "    \"put_atm_1y\": " << plainPut.NPV() << "\n";
            std::cout << "  },\n";
        }

        // --- (b) 2-segment piecewise (short / long vol break at 0.5y).
        {
            std::vector<Time> times = {0.5, 1.0};
            TimeGrid timeGrid(times.begin(), times.end());

            std::vector<Time> breaks = {0.5};
            PiecewiseConstantParameter thetaParam(breaks, PositiveConstraint());
            thetaParam.setParam(0, 0.06);
            thetaParam.setParam(1, 0.04);
            PiecewiseConstantParameter kappaParam(breaks, PositiveConstraint());
            kappaParam.setParam(0, 2.5);
            kappaParam.setParam(1, 1.5);
            PiecewiseConstantParameter sigmaParam(breaks, PositiveConstraint());
            sigmaParam.setParam(0, 0.4);
            sigmaParam.setParam(1, 0.2);
            PiecewiseConstantParameter rhoParam(
                breaks, BoundaryConstraint(-1.0, 1.0));
            rhoParam.setParam(0, -0.8);
            rhoParam.setParam(1, -0.5);

            auto ptdModel =
                ext::make_shared<PiecewiseTimeDependentHestonModel>(
                    rfTS, qTS, spotQuote, v0, thetaParam, kappaParam,
                    sigmaParam, rhoParam, timeGrid);

            auto ptdEngine =
                ext::make_shared<AnalyticPTDHestonEngine>(ptdModel, 144);

            VanillaOption call(callPayoff, exercise);
            call.setPricingEngine(ptdEngine);
            VanillaOption put(putPayoff, exercise);
            put.setPricingEngine(ptdEngine);

            std::cout << "  \"ptd_heston_2segment\": {\n";
            std::cout << "    \"theta_seg0\": " << ptdModel->theta(0.25) << ",\n";
            std::cout << "    \"kappa_seg0\": " << ptdModel->kappa(0.25) << ",\n";
            std::cout << "    \"sigma_seg0\": " << ptdModel->sigma(0.25) << ",\n";
            std::cout << "    \"rho_seg0\": " << ptdModel->rho(0.25) << ",\n";
            std::cout << "    \"theta_seg1\": " << ptdModel->theta(0.75) << ",\n";
            std::cout << "    \"kappa_seg1\": " << ptdModel->kappa(0.75) << ",\n";
            std::cout << "    \"sigma_seg1\": " << ptdModel->sigma(0.75) << ",\n";
            std::cout << "    \"rho_seg1\": " << ptdModel->rho(0.75) << ",\n";
            std::cout << "    \"call_atm_1y\": " << call.NPV() << ",\n";
            std::cout << "    \"put_atm_1y\": " << put.NPV() << "\n";
            std::cout << "  }\n";
        }
    }

    std::cout << "}\n";
    return 0;
}
