// L3-D cluster probe: equity options + processes
//
// Captures reference values for the L3-D layer:
//
//   * GeneralizedBlackScholesProcess.x0 / drift / diffusion /
//     expectation / standard deviation / variance / evolve / apply.
//     Tested with constant Black vol → strike-independent local vol,
//     which triggers the analytic-closed-form branch.
//   * BlackScholesProcess (no dividends) — values match the GBSM
//     process when the dividend curve is zero-rate.
//   * BlackProcess (forwards): rate-equals-rate branch (drift = -sigma^2/2).
//   * BlackScholesMertonProcess: identical math to GBSM (just a
//     specialized constructor).
//   * EulerDiscretization: drift = mu * dt, diffusion = sigma * sqrt(dt),
//     variance = sigma^2 * dt.
//   * AnalyticEuropeanEngine: NPV + delta + gamma + vega + theta + rho
//     + dividend_rho at the textbook (S=100, K=100, T=1, r=5%, q=2%,
//     sigma=20%) parameters for both Call and Put.
//   * BinomialVanillaEngine: NPV at N=1000 for CRR / JR / Tian /
//     LeisenReimer on a European Call (compare to AnalyticEuropeanEngine).
//   * American Put via BinomialVanillaEngine (CRR, N=500) — exceeds
//     European Put (early-exercise premium).
//
// C++ parity:
//   ql/processes/blackscholesprocess.{hpp,cpp},
//   ql/processes/eulerdiscretization.{hpp,cpp},
//   ql/instruments/vanillaoption.{hpp,cpp},
//   ql/instruments/europeanoption.{hpp,cpp},
//   ql/pricingengines/blackcalculator.{hpp,cpp},
//   ql/pricingengines/vanilla/analyticeuropeanengine.{hpp,cpp},
//   ql/pricingengines/vanilla/binomialengine.hpp,
//   ql/methods/lattices/binomialtree.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/instruments/vanillaoption.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/exercise.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/binomialengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/eulerdiscretization.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/handle.hpp>

#include <iomanip>
#include <iostream>
#include <cmath>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // Common setup: 1-year Call/Put @ S=K=100, r=5%, q=2%, sigma=20%.
    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date ref(15, June, 2026);
    Settings::instance().evaluationDate() = ref;
    Date expiry = ref + 365; // 1.0 year under Actual/365 Fixed

    Real spot = 100.0;
    Real strike = 100.0;
    Volatility vol = 0.20;
    Rate r = 0.05;
    Rate q = 0.02;

    auto spotQ = ext::make_shared<SimpleQuote>(spot);
    Handle<Quote> spotH(spotQ);
    Handle<YieldTermStructure> rfH(ext::make_shared<FlatForward>(ref, r, dc));
    Handle<YieldTermStructure> divH(ext::make_shared<FlatForward>(ref, q, dc));
    Handle<BlackVolTermStructure> volH(ext::make_shared<BlackConstantVol>(ref, cal, vol, dc));

    auto gbsm = ext::make_shared<GeneralizedBlackScholesProcess>(
        spotH, divH, rfH, volH);

    // ---------------------------------------------------------------
    // Process: GeneralizedBlackScholesProcess
    // ---------------------------------------------------------------
    {
        std::cout << "  \"gbsm_process\": {\n";
        std::cout << "    \"x0\": " << gbsm->x0() << ",\n";
        // diffusion at t=0, x=spot: localVol = vol = 0.20 exactly.
        std::cout << "    \"diffusion_t0\": " << gbsm->diffusion(0.0, spot) << ",\n";
        // drift = (r_inst - q_inst) - 0.5 * sigma^2, where r/q_inst use
        // forward rate over [t, t+0.0001]. With constant FlatForward
        // continuous compounding the forward rate equals r/q exactly.
        std::cout << "    \"drift_t0\": " << gbsm->drift(0.0, spot) << ",\n";
        // expectation(0, S, 1): strike-independent branch -> S * exp((r-q)*1).
        std::cout << "    \"expectation_t0_dt1\": " << gbsm->expectation(0.0, spot, 1.0) << ",\n";
        // variance(0, S, 1): integral of sigma^2 dt = vol^2 * 1.
        std::cout << "    \"variance_t0_dt1\": " << gbsm->variance(0.0, spot, 1.0) << ",\n";
        // stdDeviation = sqrt(variance) = vol * sqrt(1) = 0.20.
        std::cout << "    \"std_deviation_t0_dt1\": " << gbsm->stdDeviation(0.0, spot, 1.0) << ",\n";
        // evolve with dw=0: deterministic drift only.
        // exact branch: x0 * exp((r-q)*dt - 0.5*var) when dw=0.
        Real ev0 = gbsm->evolve(0.0, spot, 1.0, 0.0);
        std::cout << "    \"evolve_t0_dt1_dw0\": " << ev0 << ",\n";
        // evolve with dw=1: x0 * exp((r-q)*dt - 0.5*var + sqrt(var) * 1).
        Real ev1 = gbsm->evolve(0.0, spot, 1.0, 1.0);
        std::cout << "    \"evolve_t0_dt1_dw1\": " << ev1 << ",\n";
        // apply: x0 * exp(dx).
        std::cout << "    \"apply_x100_dx0p2\": " << gbsm->apply(100.0, 0.2) << ",\n";
        // time(date): year fraction. expiry is ref + 365 → 1.0 under Actual/365.
        std::cout << "    \"time_at_expiry\": " << gbsm->time(expiry) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BlackScholesProcess (no dividends — embeds q=0 internally)
    // ---------------------------------------------------------------
    {
        auto bs = ext::make_shared<BlackScholesProcess>(spotH, rfH, volH);
        std::cout << "  \"bs_process\": {\n";
        std::cout << "    \"x0\": " << bs->x0() << ",\n";
        std::cout << "    \"diffusion_t0\": " << bs->diffusion(0.0, spot) << ",\n";
        std::cout << "    \"drift_t0\": " << bs->drift(0.0, spot) << ",\n";
        std::cout << "    \"expectation_t0_dt1\": " << bs->expectation(0.0, spot, 1.0) << ",\n";
        std::cout << "    \"variance_t0_dt1\": " << bs->variance(0.0, spot, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BlackProcess (Black 76, futures — r=q so drift = -sigma^2/2)
    // ---------------------------------------------------------------
    {
        auto bp = ext::make_shared<BlackProcess>(spotH, rfH, volH);
        std::cout << "  \"black_process\": {\n";
        std::cout << "    \"x0\": " << bp->x0() << ",\n";
        std::cout << "    \"diffusion_t0\": " << bp->diffusion(0.0, spot) << ",\n";
        // drift = -0.5 * sigma^2 = -0.02 exactly.
        std::cout << "    \"drift_t0\": " << bp->drift(0.0, spot) << ",\n";
        // expectation = spot * exp((r-r)*1) = spot exactly.
        std::cout << "    \"expectation_t0_dt1\": " << bp->expectation(0.0, spot, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BlackScholesMertonProcess (identical math to GBSM)
    // ---------------------------------------------------------------
    {
        auto bsm = ext::make_shared<BlackScholesMertonProcess>(spotH, divH, rfH, volH);
        std::cout << "  \"bsm_process\": {\n";
        std::cout << "    \"x0\": " << bsm->x0() << ",\n";
        std::cout << "    \"diffusion_t0\": " << bsm->diffusion(0.0, spot) << ",\n";
        std::cout << "    \"drift_t0\": " << bsm->drift(0.0, spot) << ",\n";
        std::cout << "    \"expectation_t0_dt1\": " << bsm->expectation(0.0, spot, 1.0) << ",\n";
        std::cout << "    \"variance_t0_dt1\": " << bsm->variance(0.0, spot, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // EulerDiscretization (1-D StochasticProcess1D branch)
    // ---------------------------------------------------------------
    {
        EulerDiscretization euler;
        // For BlackProcess, drift = -sigma^2/2 = -0.02, diffusion = vol = 0.20.
        auto bp = ext::make_shared<BlackProcess>(spotH, rfH, volH);
        Time dt = 0.25;
        std::cout << "  \"euler_discretization\": {\n";
        // drift * dt
        std::cout << "    \"drift_dt_quarter\": " << euler.drift(*bp, 0.0, spot, dt) << ",\n";
        // diffusion * sqrt(dt) = 0.20 * 0.5 = 0.10
        std::cout << "    \"diffusion_dt_quarter\": " << euler.diffusion(*bp, 0.0, spot, dt) << ",\n";
        // variance = sigma^2 * dt = 0.04 * 0.25 = 0.01
        std::cout << "    \"variance_dt_quarter\": " << euler.variance(*bp, 0.0, spot, dt) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // AnalyticEuropeanEngine — textbook BSM call + put + greeks
    // ---------------------------------------------------------------
    {
        auto callPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto putPayoff  = ext::make_shared<PlainVanillaPayoff>(Option::Put,  strike);
        auto exercise   = ext::make_shared<EuropeanExercise>(expiry);

        EuropeanOption call(callPayoff, exercise);
        EuropeanOption put(putPayoff, exercise);

        auto engine = ext::make_shared<AnalyticEuropeanEngine>(gbsm);
        call.setPricingEngine(engine);
        put.setPricingEngine(engine);

        std::cout << "  \"analytic_european\": {\n";
        std::cout << "    \"call_npv\": " << call.NPV() << ",\n";
        std::cout << "    \"call_delta\": " << call.delta() << ",\n";
        std::cout << "    \"call_gamma\": " << call.gamma() << ",\n";
        std::cout << "    \"call_vega\": " << call.vega() << ",\n";
        std::cout << "    \"call_theta\": " << call.theta() << ",\n";
        std::cout << "    \"call_rho\": " << call.rho() << ",\n";
        std::cout << "    \"call_dividend_rho\": " << call.dividendRho() << ",\n";
        std::cout << "    \"call_itm_cash_probability\": " << call.itmCashProbability() << ",\n";

        std::cout << "    \"put_npv\": " << put.NPV() << ",\n";
        std::cout << "    \"put_delta\": " << put.delta() << ",\n";
        std::cout << "    \"put_gamma\": " << put.gamma() << ",\n";
        std::cout << "    \"put_vega\": " << put.vega() << ",\n";
        std::cout << "    \"put_theta\": " << put.theta() << ",\n";
        std::cout << "    \"put_rho\": " << put.rho() << ",\n";
        std::cout << "    \"put_dividend_rho\": " << put.dividendRho() << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // BinomialVanillaEngine (CRR / JR / Tian / LR) for European Call
    // ---------------------------------------------------------------
    {
        auto callPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto exercise   = ext::make_shared<EuropeanExercise>(expiry);
        EuropeanOption call(callPayoff, exercise);

        std::cout << "  \"binomial_european_call\": {\n";

        // CRR @ N=1000
        call.setPricingEngine(ext::make_shared<BinomialVanillaEngine<CoxRossRubinstein>>(gbsm, 1000));
        std::cout << "    \"crr_n1000\": " << call.NPV() << ",\n";

        // Jarrow-Rudd @ N=1000
        call.setPricingEngine(ext::make_shared<BinomialVanillaEngine<JarrowRudd>>(gbsm, 1000));
        std::cout << "    \"jr_n1000\": " << call.NPV() << ",\n";

        // Tian @ N=1000
        call.setPricingEngine(ext::make_shared<BinomialVanillaEngine<Tian>>(gbsm, 1000));
        std::cout << "    \"tian_n1000\": " << call.NPV() << ",\n";

        // LeisenReimer @ N=1001 (odd; LR forces odd internally)
        call.setPricingEngine(ext::make_shared<BinomialVanillaEngine<LeisenReimer>>(gbsm, 1001));
        std::cout << "    \"lr_n1001\": " << call.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // American Put via BinomialVanillaEngine (CRR, N=500)
    //   Should exceed the European put NPV (early-exercise premium).
    // ---------------------------------------------------------------
    {
        auto putPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
        auto amExercise = ext::make_shared<AmericanExercise>(ref, expiry);
        VanillaOption amPut(putPayoff, amExercise);
        amPut.setPricingEngine(ext::make_shared<BinomialVanillaEngine<CoxRossRubinstein>>(gbsm, 500));

        // For comparison, also compute the European put with N=500
        auto euExercise = ext::make_shared<EuropeanExercise>(expiry);
        EuropeanOption euPut(putPayoff, euExercise);
        euPut.setPricingEngine(ext::make_shared<BinomialVanillaEngine<CoxRossRubinstein>>(gbsm, 500));

        std::cout << "  \"american_put_binomial\": {\n";
        std::cout << "    \"american_npv_n500\": " << amPut.NPV() << ",\n";
        std::cout << "    \"european_npv_n500\": " << euPut.NPV() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
