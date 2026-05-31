// Phase 11 W11-C cluster probe: MarketModels (BGM) callability — exercise
// values + basis systems + swap-rate trigger logic. Deterministic per-object
// checks on a flat-forward LMMCurveState (rateTimes = {0.5,1.0,1.5,2.0},
// flat forward 0.05). The CANONICAL Bermudan-swaption MC test is structurally
// self-validating (payer+receiver==0, bermudan>=0, callable>=receiver,
// receiver+bermudan==callable) and needs no reference here.
//
//   * NothingExerciseValue.value(state) — always cf{timeIndex=currentIndex,
//     amount=0} after nextStep. TIGHT.
//   * BermudanSwaptionExerciseValue.value(state) — after nextStep at index k:
//     amount = max(coterminalSwapAnnuity(k,k) * payoff(coterminalSwapRate(k)),
//     0); timeIndex = k. We emit annuity, swap rate, and the resulting amount
//     for each exercise step with a fixed-strike (0.045) payer payoff. TIGHT.
//   * SwapBasisSystem.values(state) — {1, forwardRate(rateIndex),
//     coterminalSwapRate(rateIndex+1)} (or 2-vector for last). TIGHT.
//   * SwapForwardBasisSystem.values(state) — the 10/6/3 polynomial basis.
//     TIGHT.
//   * SwapRateTrigger.exercise(state) — swapTriggers[k] < coterminalSwapRate(
//     rateIndex). We emit the coterminal swap rates at each exercise step so
//     the Python boolean logic is cross-checked against a known trigger. TIGHT.
//   * numberOfFunctions() / numberOfExercises() shapes for both basis systems.
//
// C++ parity:
//   ql/models/marketmodels/callability/{nothingexercisevalue,
//     bermudanswaptionexercisevalue,swapbasissystem,swapforwardbasissystem,
//     swapratetrigger}.{hpp,cpp}
//   test-suite/marketmodel.cpp testCallableSwapNaif / testCallableSwapLS
//   @ v1.42.1 (099987f0).

#include <ql/instruments/payoffs.hpp>
#include <ql/models/marketmodels/callability/bermudanswaptionexercisevalue.hpp>
#include <ql/models/marketmodels/callability/nothingexercisevalue.hpp>
#include <ql/models/marketmodels/callability/swapbasissystem.hpp>
#include <ql/models/marketmodels/callability/swapforwardbasissystem.hpp>
#include <ql/models/marketmodels/callability/swapratetrigger.hpp>
#include <ql/models/marketmodels/curvestate.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace QuantLib;

namespace {

bool g_first = true;

void emit(const char* name, Real v) {
    if (!g_first) std::cout << ",\n";
    g_first = false;
    std::cout << "  \"" << name << "\": " << v;
}

void emit_arr(const char* name, const std::vector<Real>& v) {
    if (!g_first) std::cout << ",\n";
    g_first = false;
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

// rateTimes = {0.5,1.0,1.5,2.0}; flat forward 0.05 → 3 forward rates.
std::vector<Time> rateTimes4() { return {0.5, 1.0, 1.5, 2.0}; }
constexpr Real kF = 0.05;

LMMCurveState flatState() {
    LMMCurveState cs(rateTimes4());
    cs.setOnForwardRates(std::vector<Rate>(3, kF));
    return cs;
}

// Geometry sanity: emit the raw curve-state quantities the Python concretes
// rely on, so any divergence localizes to the concrete, not the curve state.
void block_geometry() {
    LMMCurveState cs = flatState();
    std::vector<Real> fwds = {cs.forwardRate(0), cs.forwardRate(1), cs.forwardRate(2)};
    emit_arr("geom_forwards", fwds);
    std::vector<Real> cotr = {cs.coterminalSwapRate(0), cs.coterminalSwapRate(1),
                              cs.coterminalSwapRate(2)};
    emit_arr("geom_cot_swap_rates", cotr);
    std::vector<Real> ann = {cs.coterminalSwapAnnuity(0, 0), cs.coterminalSwapAnnuity(1, 1),
                             cs.coterminalSwapAnnuity(2, 2)};
    emit_arr("geom_cot_annuities_self", ann);
    std::vector<Real> dr = {cs.discountRatio(0, 3), cs.discountRatio(1, 3),
                            cs.discountRatio(2, 3)};
    emit_arr("geom_disc_ratio_to_n", dr);
}

// NothingExerciseValue: after each nextStep, value()==cf{idx, 0.0}.
void block_nothing() {
    auto rt = rateTimes4();
    NothingExerciseValue nev(rt);
    LMMCurveState cs = flatState();
    emit("nev_num_exercises", (Real)nev.numberOfExercises());
    emit("nev_num_pcf", (Real)nev.possibleCashFlowTimes().size());
    auto ie = nev.isExerciseTime();
    emit("nev_is_exercise_size", (Real)ie.size());
    nev.reset();
    std::vector<Real> tis, amts;
    for (Size k = 0; k < ie.size(); ++k) {
        nev.nextStep(cs);
        MarketModelMultiProduct::CashFlow cf = nev.value(cs);
        tis.push_back((Real)cf.timeIndex);
        amts.push_back(cf.amount);
    }
    emit_arr("nev_time_indices", tis);
    emit_arr("nev_amounts", amts);
}

// BermudanSwaptionExerciseValue: payer payoff struck at 0.045.
void block_bermudan() {
    auto rt = rateTimes4();
    constexpr Real strike = 0.045;
    Size nExercises = rt.size() - 1;  // 3
    std::vector<ext::shared_ptr<Payoff> > payoffs;
    for (Size i = 0; i < nExercises; ++i)
        payoffs.emplace_back(new PlainVanillaPayoff(Option::Call, strike));
    BermudanSwaptionExerciseValue bev(rt, payoffs);
    LMMCurveState cs = flatState();

    emit("bev_num_exercises", (Real)bev.numberOfExercises());
    emit("bev_strike", strike);
    auto ie = bev.isExerciseTime();
    emit("bev_is_exercise_size", (Real)ie.size());

    bev.reset();
    std::vector<Real> tis, amts;
    for (Size k = 0; k < nExercises; ++k) {
        bev.nextStep(cs);
        MarketModelMultiProduct::CashFlow cf = bev.value(cs);
        tis.push_back((Real)cf.timeIndex);
        amts.push_back(cf.amount);
    }
    emit_arr("bev_time_indices", tis);
    emit_arr("bev_amounts", amts);
}

// SwapBasisSystem on exerciseTimes = {0.5,1.0,1.5}.
void block_swap_basis() {
    auto rt = rateTimes4();
    std::vector<Time> ex = {0.5, 1.0, 1.5};
    SwapBasisSystem sbs(rt, ex);
    LMMCurveState cs = flatState();

    emit("sbs_num_exercises", (Real)sbs.numberOfExercises());
    std::vector<Real> nf;
    for (Size s : sbs.numberOfFunctions())
        nf.push_back((Real)s);
    emit_arr("sbs_number_of_functions", nf);

    sbs.reset();
    // step through, emit each exercise step's basis-function vector flattened.
    std::vector<Real> flat;
    std::vector<Real> sizes;
    for (Size k = 0; k < ex.size(); ++k) {
        sbs.nextStep(cs);
        std::vector<Real> r;
        sbs.values(cs, r);
        sizes.push_back((Real)r.size());
        for (Real x : r)
            flat.push_back(x);
    }
    emit_arr("sbs_values_sizes", sizes);
    emit_arr("sbs_values_flat", flat);
}

// SwapForwardBasisSystem on exerciseTimes = {0.5,1.0,1.5}.
void block_swap_forward_basis() {
    auto rt = rateTimes4();
    std::vector<Time> ex = {0.5, 1.0, 1.5};
    SwapForwardBasisSystem sfbs(rt, ex);
    LMMCurveState cs = flatState();

    emit("sfbs_num_exercises", (Real)sfbs.numberOfExercises());
    std::vector<Real> nf;
    for (Size s : sfbs.numberOfFunctions())
        nf.push_back((Real)s);
    emit_arr("sfbs_number_of_functions", nf);

    sfbs.reset();
    std::vector<Real> flat;
    std::vector<Real> sizes;
    for (Size k = 0; k < ex.size(); ++k) {
        sfbs.nextStep(cs);
        std::vector<Real> r;
        sfbs.values(cs, r);
        sizes.push_back((Real)r.size());
        for (Real x : r)
            flat.push_back(x);
    }
    emit_arr("sfbs_values_sizes", sizes);
    emit_arr("sfbs_values_flat", flat);
}

// SwapRateTrigger on exerciseTimes = {0.5,1.0,1.5}, triggers all 0.04.
// exercise(state) returns swapTriggers[k] < coterminalSwapRate(rateIndex[k]).
void block_swap_rate_trigger() {
    auto rt = rateTimes4();
    std::vector<Time> ex = {0.5, 1.0, 1.5};
    constexpr Real trigger = 0.04;
    std::vector<Rate> triggers(ex.size(), trigger);
    SwapRateTrigger srt(rt, triggers, ex);
    LMMCurveState cs = flatState();

    emit("srt_trigger", trigger);
    std::vector<Real> et;
    for (Real t : srt.exerciseTimes())
        et.push_back(t);
    emit_arr("srt_exercise_times", et);

    srt.reset();
    std::vector<Real> rates;
    std::vector<Real> exFlags;
    for (Size k = 0; k < ex.size(); ++k) {
        srt.nextStep(cs);
        // mirror SwapRateTrigger::exercise: rateIndex[currentIndex-1] then
        // coterminalSwapRate(rateIndex). For ex={0.5,1.0,1.5}, rateIndex={0,1,2}.
        // We record the boolean it returns.
        bool ex_k = srt.exercise(cs);
        exFlags.push_back(ex_k ? 1.0 : 0.0);
    }
    // also emit the coterminal swap rates at the relevant indices (0,1,2).
    LMMCurveState cs2 = flatState();
    rates = {cs2.coterminalSwapRate(0), cs2.coterminalSwapRate(1), cs2.coterminalSwapRate(2)};
    emit_arr("srt_cot_swap_rates", rates);
    emit_arr("srt_exercise_flags", exFlags);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_geometry();
    block_nothing();
    block_bermudan();
    block_swap_basis();
    block_swap_forward_basis();
    block_swap_rate_trigger();

    std::cout << "\n}\n";
    return 0;
}
