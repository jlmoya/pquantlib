// Phase 11 W4-C cluster probe: barrier specialties + variance options.
//
// Captures reference values for:
//
//   * PartialTimeBarrierOption + AnalyticPartialTimeBarrierOptionEngine
//     (Heynen-Kat 1994 closed-form) — DownOut/UpOut Call and Put across
//     PartialBarrier::Start / EndB1 / EndB2 ranges. EndB2 is restricted
//     to strike<barrier (DownOut: K=90, B=100; UpOut: K=110, B=100 via
//     symmetric setup). Includes knock-in via Start range.
//
//   * BinomialDoubleBarrierEngine<CoxRossRubinstein> NPV at fine-step
//     limit (n=400) cross-checked against AnalyticDoubleBarrierEngine
//     for a non-degenerate KnockOut Call (S=100, K=100, B_lo=80,
//     B_hi=120, T=1y, r=5%, q=2%, sigma=25%).
//
//   * SoftBarrierOption + AnalyticSoftBarrierEngine (Hart-Ross 1994)
//     for DownIn / DownOut Call with a soft-range barrier
//     (S=100, X=100, U=95, L=85, T=1y, r=8%, q=4%, sigma=25%).
//
//   * VarianceOption + IntegralHestonVarianceOptionEngine (Bailey-
//     Swarztrauber 2-D oscillatory integral; specialised PlainVanilla
//     Call path) with Heston parameters (v0=0.04, kappa=4.0, theta=0.04,
//     sigma_v=0.25, rho=-0.5, r=4%, q=0%, T=0.5y, strike=0.04,
//     notional=10000).
//
// C++ parity:
//   ql/instruments/partialtimebarrieroption.hpp
//   ql/pricingengines/barrier/analyticpartialtimebarrieroptionengine.hpp
//   ql/experimental/barrieroption/binomialdoublebarrierengine.hpp
//   ql/instruments/softbarrieroption.hpp
//   ql/pricingengines/barrier/analyticsoftbarrierengine.hpp
//   ql/experimental/varianceoption/varianceoption.hpp
//   ql/experimental/varianceoption/integralhestonvarianceoptionengine.hpp
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/experimental/barrieroption/binomialdoublebarrierengine.hpp>
#include <ql/experimental/varianceoption/integralhestonvarianceoptionengine.hpp>
#include <ql/experimental/varianceoption/varianceoption.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/barriertype.hpp>
#include <ql/instruments/doublebarrieroption.hpp>
#include <ql/instruments/partialtimebarrieroption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/softbarrieroption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/pricingengines/barrier/analyticdoublebarrierengine.hpp>
#include <ql/pricingengines/barrier/analyticpartialtimebarrieroptionengine.hpp>
#include <ql/pricingengines/barrier/analyticsoftbarrierengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

ext::shared_ptr<GeneralizedBlackScholesProcess>
makeBsmProcess(const Date& today, Real spot, Rate r, Rate q, Volatility sigma) {
    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Handle<Quote> spotH(ext::make_shared<SimpleQuote>(spot));
    Handle<YieldTermStructure> qts(
        ext::make_shared<FlatForward>(today, q, dc));
    Handle<YieldTermStructure> rts(
        ext::make_shared<FlatForward>(today, r, dc));
    Handle<BlackVolTermStructure> vts(
        ext::make_shared<BlackConstantVol>(today, cal, sigma, dc));
    return ext::make_shared<GeneralizedBlackScholesProcess>(
        spotH, qts, rts, vts);
}

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;

    // ============================================================
    // 1) Partial-time barrier option (Heynen-Kat 1994).
    //    DownOut: S=100, K=90, B=100 (strike<barrier required for
    //    EndB2). UpOut variant uses K=110, B=100.
    //    T=1y, T1=0.5y (cover event), r=5%, q=0%, sigma=25%.
    // ============================================================
    {
        Real S = 100.0;
        Real B = 100.0;
        Real K_call = 90.0;   // strike < barrier for DownOut EndB2
        Rate r = 0.05;
        Rate q = 0.0;
        Volatility sigma = 0.25;
        Date exDate = today + 365 * Days;     // T=1y
        Date coverDate = today + 182 * Days;  // T1 ~ 0.5y

        auto process = makeBsmProcess(today, S, r, q, sigma);
        ext::shared_ptr<Exercise> exercise =
            ext::make_shared<EuropeanExercise>(exDate);
        ext::shared_ptr<StrikedTypePayoff> callPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, K_call);

        ext::shared_ptr<PricingEngine> engine =
            ext::make_shared<AnalyticPartialTimeBarrierOptionEngine>(process);

        // DownOut Call - Start range, K=90, B=100, S=100.
        PartialTimeBarrierOption opt1(
            Barrier::DownOut, PartialBarrier::Start, B, 0.0, coverDate,
            callPayoff, exercise);
        opt1.setPricingEngine(engine);
        emit("partial_time_downout_call_start", opt1.NPV());

        // DownOut Call - EndB1 range.
        PartialTimeBarrierOption opt2(
            Barrier::DownOut, PartialBarrier::EndB1, B, 0.0, coverDate,
            callPayoff, exercise);
        opt2.setPricingEngine(engine);
        emit("partial_time_downout_call_endb1", opt2.NPV());

        // DownOut Call - EndB2 range (requires strike<barrier).
        PartialTimeBarrierOption opt3(
            Barrier::DownOut, PartialBarrier::EndB2, B, 0.0, coverDate,
            callPayoff, exercise);
        opt3.setPricingEngine(engine);
        emit("partial_time_downout_call_endb2", opt3.NPV());

        // UpOut Call - Start range with B=120, K=110.
        Real B_up = 120.0;
        Real K_up = 110.0;
        ext::shared_ptr<StrikedTypePayoff> callPayoffUp =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, K_up);
        PartialTimeBarrierOption opt4(
            Barrier::UpOut, PartialBarrier::Start, B_up, 0.0, coverDate,
            callPayoffUp, exercise);
        opt4.setPricingEngine(engine);
        emit("partial_time_upout_call_start", opt4.NPV());

        // DownIn Call - Start range, K=90, B=80.
        Real B_di = 80.0;
        ext::shared_ptr<StrikedTypePayoff> callPayoffDi =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, K_call);
        PartialTimeBarrierOption opt5(
            Barrier::DownIn, PartialBarrier::Start, B_di, 0.0, coverDate,
            callPayoffDi, exercise);
        opt5.setPricingEngine(engine);
        emit("partial_time_downin_call_start", opt5.NPV());

        // UpIn Call - Start range, K=110, B=120.
        PartialTimeBarrierOption opt6(
            Barrier::UpIn, PartialBarrier::Start, B_up, 0.0, coverDate,
            callPayoffUp, exercise);
        opt6.setPricingEngine(engine);
        emit("partial_time_upin_call_start", opt6.NPV());

        // Put variant - UpOut Put with K=110, B=120 (driver maps to call
        // via reflection - symmetric barrier swap).
        ext::shared_ptr<StrikedTypePayoff> putPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Put, K_up);
        PartialTimeBarrierOption opt7(
            Barrier::UpOut, PartialBarrier::Start, B_up, 0.0, coverDate,
            putPayoff, exercise);
        opt7.setPricingEngine(engine);
        emit("partial_time_upout_put_start", opt7.NPV());
    }

    // ============================================================
    // 2) Binomial double-barrier engine cross-check against
    //    analytic double-barrier (Ikeda-Kunitomo 1992).
    //    Knock-out Call, S=100, K=100, B_lo=80, B_hi=120,
    //    T=1y, r=5%, q=2%, sigma=25%, rebate=0.
    // ============================================================
    {
        Real S = 100.0;
        Real K = 100.0;
        Real B_lo = 80.0;
        Real B_hi = 120.0;
        Rate r = 0.05;
        Rate q = 0.02;
        Volatility sigma = 0.25;
        Date exDate = today + 365 * Days;  // T=1y

        auto process = makeBsmProcess(today, S, r, q, sigma);
        ext::shared_ptr<Exercise> exercise =
            ext::make_shared<EuropeanExercise>(exDate);
        ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, K);

        DoubleBarrierOption analyticOpt(
            DoubleBarrier::KnockOut, B_lo, B_hi, 0.0, payoff, exercise);
        analyticOpt.setPricingEngine(
            ext::make_shared<AnalyticDoubleBarrierEngine>(process));
        emit("double_barrier_analytic_knockout_call",
             analyticOpt.NPV());

        // Binomial CRR engine, 400 steps for convergence.
        DoubleBarrierOption binomialOpt(
            DoubleBarrier::KnockOut, B_lo, B_hi, 0.0, payoff, exercise);
        binomialOpt.setPricingEngine(
            ext::make_shared<BinomialDoubleBarrierEngine<CoxRossRubinstein>>(
                process, 400));
        emit("double_barrier_binomial_crr_knockout_call_400",
             binomialOpt.NPV());

        // Knock-in variant as well.
        DoubleBarrierOption analyticIn(
            DoubleBarrier::KnockIn, B_lo, B_hi, 0.0, payoff, exercise);
        analyticIn.setPricingEngine(
            ext::make_shared<AnalyticDoubleBarrierEngine>(process));
        emit("double_barrier_analytic_knockin_call",
             analyticIn.NPV());

        DoubleBarrierOption binomialIn(
            DoubleBarrier::KnockIn, B_lo, B_hi, 0.0, payoff, exercise);
        binomialIn.setPricingEngine(
            ext::make_shared<BinomialDoubleBarrierEngine<CoxRossRubinstein>>(
                process, 400));
        emit("double_barrier_binomial_crr_knockin_call_400",
             binomialIn.NPV());
    }

    // ============================================================
    // 3) Soft-barrier option (Hart-Ross 1994).
    //    S=100, X=100, U=95, L=85, T=1y, r=8%, q=4%, sigma=25%.
    // ============================================================
    {
        Real S = 100.0;
        Real X = 100.0;
        Real U = 95.0;
        Real L = 85.0;
        Rate r = 0.08;
        Rate q = 0.04;
        Volatility sigma = 0.25;
        Date exDate = today + 365 * Days;  // T=1y

        auto process = makeBsmProcess(today, S, r, q, sigma);
        ext::shared_ptr<Exercise> exercise =
            ext::make_shared<EuropeanExercise>(exDate);
        ext::shared_ptr<StrikedTypePayoff> callPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, X);

        SoftBarrierOption sb1(
            Barrier::DownIn, L, U, callPayoff, exercise);
        sb1.setPricingEngine(
            ext::make_shared<AnalyticSoftBarrierEngine>(process));
        emit("soft_barrier_downin_call", sb1.NPV());

        SoftBarrierOption sb2(
            Barrier::DownOut, L, U, callPayoff, exercise);
        sb2.setPricingEngine(
            ext::make_shared<AnalyticSoftBarrierEngine>(process));
        emit("soft_barrier_downout_call", sb2.NPV());

        // UpIn at strike > spot with U=120, L=110.
        SoftBarrierOption sb3(
            Barrier::UpIn, 110.0, 120.0, callPayoff, exercise);
        sb3.setPricingEngine(
            ext::make_shared<AnalyticSoftBarrierEngine>(process));
        emit("soft_barrier_upin_call_high_band", sb3.NPV());

        // Put variant.
        ext::shared_ptr<StrikedTypePayoff> putPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Put, X);
        SoftBarrierOption sb4(
            Barrier::DownIn, L, U, putPayoff, exercise);
        sb4.setPricingEngine(
            ext::make_shared<AnalyticSoftBarrierEngine>(process));
        emit("soft_barrier_downin_put", sb4.NPV());
    }

    // ============================================================
    // 4) Variance option + IntegralHestonVarianceOptionEngine
    //    (Bailey-Swarztrauber 2-D integral, Recchioni et al.).
    //    Heston params chosen so Feller 2*kappa*theta > sigma_v^2:
    //    v0=0.04, kappa=4.0, theta=0.04, sigma_v=0.25, rho=-0.5,
    //    r=4%, q=0%, T=0.5y. Call on realised variance, strike=0.04,
    //    notional=10000.
    // ============================================================
    {
        Date startDate = today;
        Date maturity = today + 182 * Days;  // T ~ 0.5y

        Real v0 = 0.04;
        Real kappa = 4.0;
        Real theta = 0.04;
        Real sigma_v = 0.25;
        Real rho = -0.5;
        Rate r = 0.04;
        Real spot = 100.0;

        Handle<Quote> spotH(ext::make_shared<SimpleQuote>(spot));
        Handle<YieldTermStructure> rts(
            ext::make_shared<FlatForward>(today, r, dc));
        // engine asserts dividendYield empty - leave default handle.
        Handle<YieldTermStructure> qts(ext::shared_ptr<YieldTermStructure>{});

        auto hestonProcess = ext::make_shared<HestonProcess>(
            rts, qts, spotH, v0, kappa, theta, sigma_v, rho);

        Real strike = 0.04;
        Real notional = 10000.0;
        ext::shared_ptr<Payoff> callPayoff =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);

        VarianceOption vopt(callPayoff, notional, startDate, maturity);
        vopt.setPricingEngine(
            ext::make_shared<IntegralHestonVarianceOptionEngine>(
                hestonProcess));
        emit("variance_option_heston_call_npv", vopt.NPV());

        // Second strike for surface coverage.
        Real strike2 = 0.05;
        ext::shared_ptr<Payoff> callPayoff2 =
            ext::make_shared<PlainVanillaPayoff>(Option::Call, strike2);
        VarianceOption vopt2(callPayoff2, notional, startDate, maturity);
        vopt2.setPricingEngine(
            ext::make_shared<IntegralHestonVarianceOptionEngine>(
                hestonProcess));
        emit("variance_option_heston_call_npv_strike005", vopt2.NPV(), false);
    }

    std::cout << "}\n";
    return 0;
}
