// Phase 11 W7-A cluster probe: experimental processes + variance-gamma.
//
// Captures reference values for:
//
//   * ExtendedBlackScholesMertonProcess: drift/diffusion at a known
//     (t, x); evolve (Euler discretization variant) reproduces the
//     GeneralizedBlackScholesProcess evolve for the default Euler
//     evol-discretization. Also Milstein + PredictorCorrector evolve.
//
//   * GemanRoncoroniProcess: drift(t,x), diffusion(t,x), stdDeviation
//     at a known state with the canonical vpp.cpp params.
//
//   * VarianceGammaProcess: x0 + accessor round-trips (sigma/nu/theta).
//
//   * VarianceGammaEngine (analytic): European call + put NPV at the
//     canonical variancegamma.cpp params/strikes (LOOSE).
//
//   * FFTVanillaEngine: European call NPV under a BS process vs the
//     AnalyticEuropeanEngine reference (LOOSE).
//
//   * FFTVarianceGammaEngine: European call NPV vs the analytic VG
//     engine (LOOSE).
//
// C++ parity:
//   ql/experimental/processes/extendedblackscholesprocess.hpp
//   ql/experimental/processes/gemanroncoroniprocess.hpp
//   ql/experimental/variancegamma/variancegammaprocess.hpp
//   ql/experimental/variancegamma/variancegammamodel.hpp
//   ql/experimental/variancegamma/analyticvariancegammaengine.hpp
//   ql/experimental/variancegamma/fftengine.hpp
//   ql/experimental/variancegamma/fftvanillaengine.hpp
//   ql/experimental/variancegamma/fftvariancegammaengine.hpp
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/experimental/processes/extendedblackscholesprocess.hpp>
#include <ql/experimental/processes/gemanroncoroniprocess.hpp>
#include <ql/experimental/variancegamma/analyticvariancegammaengine.hpp>
#include <ql/experimental/variancegamma/fftvanillaengine.hpp>
#include <ql/experimental/variancegamma/fftvariancegammaengine.hpp>
#include <ql/experimental/variancegamma/variancegammaprocess.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/eulerdiscretization.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/date.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

// ---------------------------------------------------------------------
// ExtendedBlackScholesMertonProcess
// ---------------------------------------------------------------------
void block_extended_bsm() {
    const DayCounter dc = Actual365Fixed();
    const Date today(15, January, 2024);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.02, dc));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), 0.25, dc));

    // Euler evol-discretization to match the plain GBSM evolve.
    ExtendedBlackScholesMertonProcess pEuler(
        spot, qTS, rTS, volTS,
        ext::shared_ptr<StochasticProcess1D::discretization>(new EulerDiscretization),
        ExtendedBlackScholesMertonProcess::Euler);

    ExtendedBlackScholesMertonProcess pMil(
        spot, qTS, rTS, volTS,
        ext::shared_ptr<StochasticProcess1D::discretization>(new EulerDiscretization),
        ExtendedBlackScholesMertonProcess::Milstein);

    ExtendedBlackScholesMertonProcess pPC(
        spot, qTS, rTS, volTS,
        ext::shared_ptr<StochasticProcess1D::discretization>(new EulerDiscretization),
        ExtendedBlackScholesMertonProcess::PredictorCorrector);

    // Plain GBSM with forced Euler discretization for the Euler-evolve match.
    GeneralizedBlackScholesProcess gbsm(
        spot, qTS, rTS, volTS,
        ext::shared_ptr<StochasticProcess1D::discretization>(new EulerDiscretization),
        true /* forceDiscretization */);

    const Time t = 0.5;
    const Real x = 105.0;
    const Time dt = 0.25;
    const Real dw = 0.5;

    emit("ebsm_drift", pEuler.drift(t, x));
    emit("ebsm_diffusion", pEuler.diffusion(t, x));
    emit("ebsm_x0", pEuler.x0());
    emit("ebsm_evolve_euler", pEuler.evolve(t, x, dt, dw));
    emit("ebsm_gbsm_evolve_euler", gbsm.evolve(t, x, dt, dw));
    emit("ebsm_evolve_milstein", pMil.evolve(t, x, dt, dw));
    emit("ebsm_evolve_predcorr", pPC.evolve(t, x, dt, dw));
}

// ---------------------------------------------------------------------
// GemanRoncoroniProcess (canonical vpp.cpp params)
// ---------------------------------------------------------------------
void block_geman_roncoroni() {
    const Real x0     = 3.3;
    const Real beta   = 0.05;
    const Real alpha  = 3.1;
    const Real gamma  = -0.09;
    const Real delta  = 0.07;
    const Real eps    = -0.40;
    const Real zeta   = -0.40;
    const Real d      = 1.6;
    const Real k      = 1.0;
    const Real tau    = 0.5;
    const Real sig2   = 10.0;
    const Real a      = -7.0;
    const Real b      = -0.3;
    const Real theta1 = 35.0;
    const Real theta2 = 9.0;
    const Real theta3 = 0.10;
    const Real psi    = 1.9;

    GemanRoncoroniProcess gr(x0, alpha, beta, gamma, delta,
                             eps, zeta, d, k, tau, sig2, a, b,
                             theta1, theta2, theta3, psi);

    const Time t = 0.4;
    const Real x = 3.0;
    const Time dt = 0.1;

    emit("gr_x0", gr.x0());
    emit("gr_drift", gr.drift(t, x));
    emit("gr_diffusion", gr.diffusion(t, x));
    emit("gr_std_deviation", gr.stdDeviation(t, x, dt));
    // deterministic evolve path: supply the jump-driver Array directly.
    Array du(3);
    du[0] = 0.4;  // interarrival driver
    du[1] = 0.6;  // jump-size driver
    du[2] = 0.0;
    emit("gr_evolve_below", gr.evolve(t, x, dt, 0.5, du));
    // x0 above mu+d branch (large x).
    emit("gr_evolve_above", gr.evolve(t, 50.0, dt, 0.5, du));
}

// ---------------------------------------------------------------------
// VarianceGammaProcess accessors
// ---------------------------------------------------------------------
void block_vg_process() {
    const DayCounter dc = Actual360();
    const Date today(15, January, 2024);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(6000.0));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.00, dc));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));

    VarianceGammaProcess vg(spot, qTS, rTS, 0.20, 0.05, -0.50);
    emit("vg_x0", vg.x0());
    emit("vg_sigma", vg.sigma());
    emit("vg_nu", vg.nu());
    emit("vg_theta", vg.theta());
}

// ---------------------------------------------------------------------
// VarianceGammaEngine (analytic) — canonical variancegamma.cpp case 0
// ---------------------------------------------------------------------
void block_vg_analytic() {
    const DayCounter dc = Actual360();
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(6000.0));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.00, dc));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));

    ext::shared_ptr<VarianceGammaProcess> vg(
        new VarianceGammaProcess(spot, qTS, rTS, 0.20, 0.05, -0.50));

    ext::shared_ptr<PricingEngine> engine(new VarianceGammaEngine(vg));

    // t = 1.0 (Actual360 → 360 days)
    const Date exDate = today + 360;
    ext::shared_ptr<Exercise> exercise(new EuropeanExercise(exDate));

    // a few representative strikes from the canonical grid
    const Real strikes[] = {5550.0, 6000.0, 6500.0};
    const char* names_c[] = {"vg_analytic_call_5550", "vg_analytic_call_6000",
                             "vg_analytic_call_6500"};
    for (int j = 0; j < 3; ++j) {
        ext::shared_ptr<StrikedTypePayoff> payoff(
            new PlainVanillaPayoff(Option::Call, strikes[j]));
        EuropeanOption option(payoff, exercise);
        option.setPricingEngine(engine);
        emit(names_c[j], option.NPV());
    }
    // one put
    ext::shared_ptr<StrikedTypePayoff> putPayoff(
        new PlainVanillaPayoff(Option::Put, 5550.0));
    EuropeanOption putOption(putPayoff, exercise);
    putOption.setPricingEngine(engine);
    emit("vg_analytic_put_5550", putOption.NPV());
}

// ---------------------------------------------------------------------
// FFTVarianceGammaEngine vs analytic VG (same canonical case)
// ---------------------------------------------------------------------
void block_vg_fft() {
    const DayCounter dc = Actual360();
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(6000.0));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.00, dc));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));

    ext::shared_ptr<VarianceGammaProcess> vg(
        new VarianceGammaProcess(spot, qTS, rTS, 0.20, 0.05, -0.50));

    ext::shared_ptr<FFTVarianceGammaEngine> fftEngine(
        new FFTVarianceGammaEngine(vg));

    const Date exDate = today + 360;
    ext::shared_ptr<Exercise> exercise(new EuropeanExercise(exDate));

    const Real strikes[] = {5550.0, 6000.0, 6500.0};
    const char* names_c[] = {"vg_fft_call_5550", "vg_fft_call_6000",
                             "vg_fft_call_6500"};

    // FFT engine wants a precalculated list; build options + precalculate.
    std::vector<ext::shared_ptr<Instrument>> optionList;
    std::vector<ext::shared_ptr<EuropeanOption>> options;
    for (int j = 0; j < 3; ++j) {
        ext::shared_ptr<StrikedTypePayoff> payoff(
            new PlainVanillaPayoff(Option::Call, strikes[j]));
        ext::shared_ptr<EuropeanOption> option(new EuropeanOption(payoff, exercise));
        option->setPricingEngine(fftEngine);
        options.push_back(option);
        optionList.push_back(option);
    }
    ext::shared_ptr<StrikedTypePayoff> putPayoff(
        new PlainVanillaPayoff(Option::Put, 5550.0));
    ext::shared_ptr<EuropeanOption> putOption(new EuropeanOption(putPayoff, exercise));
    putOption->setPricingEngine(fftEngine);
    optionList.push_back(putOption);

    fftEngine->precalculate(optionList);

    for (int j = 0; j < 3; ++j) {
        emit(names_c[j], options[j]->NPV());
    }
    emit("vg_fft_put_5550", putOption->NPV());
}

// ---------------------------------------------------------------------
// FFTVanillaEngine vs AnalyticEuropeanEngine (BS process)
// ---------------------------------------------------------------------
void block_fft_vanilla() {
    const DayCounter dc = Actual365Fixed();
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.02, dc));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), 0.25, dc));

    ext::shared_ptr<GeneralizedBlackScholesProcess> bs(
        new BlackScholesMertonProcess(spot, qTS, rTS, volTS));

    ext::shared_ptr<FFTVanillaEngine> fftEngine(new FFTVanillaEngine(bs));
    ext::shared_ptr<PricingEngine> anEngine(new AnalyticEuropeanEngine(bs));

    const Date exDate = today + 365;
    ext::shared_ptr<Exercise> exercise(new EuropeanExercise(exDate));

    const Real strikes[] = {90.0, 100.0, 110.0};
    const char* fft_names[] = {"fft_vanilla_call_90", "fft_vanilla_call_100",
                               "fft_vanilla_call_110"};
    const char* an_names[]  = {"an_vanilla_call_90", "an_vanilla_call_100",
                               "an_vanilla_call_110"};

    std::vector<ext::shared_ptr<Instrument>> optionList;
    std::vector<ext::shared_ptr<EuropeanOption>> options;
    for (int j = 0; j < 3; ++j) {
        ext::shared_ptr<StrikedTypePayoff> payoff(
            new PlainVanillaPayoff(Option::Call, strikes[j]));
        ext::shared_ptr<EuropeanOption> option(new EuropeanOption(payoff, exercise));
        option->setPricingEngine(fftEngine);
        options.push_back(option);
        optionList.push_back(option);
    }
    fftEngine->precalculate(optionList);

    for (int j = 0; j < 3; ++j) {
        emit(fft_names[j], options[j]->NPV());
        // analytic reference
        ext::shared_ptr<StrikedTypePayoff> payoff(
            new PlainVanillaPayoff(Option::Call, strikes[j]));
        EuropeanOption anOption(payoff, exercise);
        anOption.setPricingEngine(anEngine);
        emit(an_names[j], anOption.NPV());
    }

    // a put too
    ext::shared_ptr<StrikedTypePayoff> putPayoff(
        new PlainVanillaPayoff(Option::Put, 100.0));
    ext::shared_ptr<EuropeanOption> putOption(new EuropeanOption(putPayoff, exercise));
    putOption->setPricingEngine(fftEngine);
    std::vector<ext::shared_ptr<Instrument>> putList{putOption};
    fftEngine->precalculate(putList);
    emit("fft_vanilla_put_100", putOption->NPV());

    EuropeanOption anPut(putPayoff, exercise);
    anPut.setPricingEngine(anEngine);
    emit("an_vanilla_put_100", anPut.NPV(), false);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_extended_bsm();
    block_geman_roncoroni();
    block_vg_process();
    block_vg_analytic();
    block_vg_fft();
    block_fft_vanilla();

    std::cout << "}\n";
    return 0;
}
