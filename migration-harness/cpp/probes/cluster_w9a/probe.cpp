// Phase 11 W9-A cluster probe: MarketModels (LIBOR Market Model / BGM)
// core/foundations — the abstract spine + curve-state geometry +
// swap<->forward mappings + numeraire discounters + grid utilities.
//
// This is the FIRST marketmodels cluster; all downstream W9-B/C, W10, W11
// build against the abstract spine ported here. The probe pins the pure
// algebra of the *concrete* curve-state classes and the static mapping
// helpers — the abstract bases themselves carry no reference values.
//
//   * mergeTimes(times) — merged sorted-unique time grid + per-input
//     presence flags (EXACT — deterministic set merge).
//
//   * EvolutionDescription(rateTimes) — default evolutionTimes = rateTimes
//     minus the last; rateTaus; firstAliveRate bookkeeping; numberOfRates /
//     numberOfSteps (TIGHT — pure index/grid algebra).
//
//   * LMMCurveState.setOnForwardRates → discountRatio / forwardRate /
//     coterminalSwapRate / coterminalSwapAnnuity / cmSwapRate (TIGHT —
//     LIBOR-market-model discounting algebra).
//
//   * CMSwapCurveState.setOnCMSwapRates → discountRatio / coterminalSwapRate
//     round-trip (TIGHT).
//
//   * CoterminalSwapCurveState.setOnCoterminalSwapRates → discountRatio /
//     forwardRate / coterminalSwapAnnuity (TIGHT).
//
//   * SwapForwardMappings::coterminalSwapForwardJacobian at a known LMM
//     curve state (TIGHT — dsr[i]/df[j] analytic jacobian).
//
//   * MarketModelDiscounter.numeraireBonds — log-linear-interpolated
//     numeraire-rebased discount of a payment time (TIGHT).
//
// C++ parity:
//   ql/models/marketmodels/utilities.hpp
//   ql/models/marketmodels/evolutiondescription.hpp
//   ql/models/marketmodels/curvestates/lmmcurvestate.hpp
//   ql/models/marketmodels/curvestates/cmswapcurvestate.hpp
//   ql/models/marketmodels/curvestates/coterminalswapcurvestate.hpp
//   ql/models/marketmodels/swapforwardmappings.hpp
//   ql/models/marketmodels/discounter.hpp
//   @ v1.42.1 (099987f0).

#include <ql/models/marketmodels/curvestates/cmswapcurvestate.hpp>
#include <ql/models/marketmodels/curvestates/coterminalswapcurvestate.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/discounter.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/swapforwardmappings.hpp>
#include <ql/models/marketmodels/utilities.hpp>
#include <ql/math/matrix.hpp>

#include <iomanip>
#include <iostream>
#include <valarray>
#include <vector>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

// rateTimes: t0..t5, evenly spaced 0.5y apart starting at 0.5 -> 6 times,
// 5 forward rates. Used by all curve-state blocks.
std::vector<Time> rateTimes6() {
    std::vector<Time> rt(6);
    for (Size i = 0; i < 6; ++i)
        rt[i] = 0.5 * (i + 1);  // 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    return rt;
}

void block_utilities() {
    // mergeTimes: three overlapping increasing time vectors.
    std::vector<std::vector<Time>> times(3);
    times[0] = {1.0, 2.0, 3.0};
    times[1] = {2.0, 4.0};
    times[2] = {1.0, 5.0};
    std::vector<Time> merged;
    std::vector<std::valarray<bool>> isPresent;
    mergeTimes(times, merged, isPresent);
    // merged = {1,2,3,4,5}
    emit("merge_n", (Real)merged.size());
    emit("merge_t0", merged[0]);
    emit("merge_t1", merged[1]);
    emit("merge_t2", merged[2]);
    emit("merge_t3", merged[3]);
    emit("merge_t4", merged[4]);
    // presence: input 0 = {1,2,3} -> [1,1,1,0,0]; sum over each row
    Size s0 = 0, s1 = 0, s2 = 0;
    for (Size j = 0; j < merged.size(); ++j) {
        s0 += isPresent[0][j] ? 1 : 0;
        s1 += isPresent[1][j] ? 1 : 0;
        s2 += isPresent[2][j] ? 1 : 0;
    }
    emit("merge_present_row0_sum", (Real)s0);
    emit("merge_present_row1_sum", (Real)s1);
    emit("merge_present_row2_sum", (Real)s2);
    // isInSubset({1,2,3,4,5}, {2,4}) -> [0,1,0,1,0]
    std::valarray<bool> sub = isInSubset(merged, std::vector<Time>{2.0, 4.0});
    Size subsum = 0;
    for (Size j = 0; j < merged.size(); ++j)
        subsum += sub[j] ? 1 : 0;
    emit("subset_sum", (Real)subsum);
    emit("subset_at1", sub[1] ? 1.0 : 0.0);
    emit("subset_at3", sub[3] ? 1.0 : 0.0);
}

void block_evolution_description() {
    std::vector<Time> rt = rateTimes6();
    // Default evolutionTimes = rateTimes minus the last = {0.5,1,1.5,2,2.5}
    EvolutionDescription evo(rt);
    emit("evo_n_rates", (Real)evo.numberOfRates());
    emit("evo_n_steps", (Real)evo.numberOfSteps());
    // rateTaus all 0.5
    emit("evo_tau0", evo.rateTaus()[0]);
    emit("evo_tau4", evo.rateTaus()[4]);
    // evolutionTimes back == rateTimes[n-2] == 2.5
    emit("evo_evoltime_back", evo.evolutionTimes().back());
    // firstAliveRate: for step j with evolution time t, count of rateTimes<=prevEvolTime.
    // currentEvolutionTime starts at 0 -> step0 alive=0; then it advances.
    const std::vector<Size>& far = evo.firstAliveRate();
    emit("evo_far0", (Real)far[0]);
    emit("evo_far1", (Real)far[1]);
    emit("evo_far2", (Real)far[2]);
    emit("evo_far3", (Real)far[3]);
    emit("evo_far4", (Real)far[4]);
    // relevanceRates default = (0, n) for each step
    emit("evo_relevance0_first", (Real)evo.relevanceRates()[0].first);
    emit("evo_relevance0_second", (Real)evo.relevanceRates()[0].second);
}

// Common forward-rate set for curve-state blocks.
std::vector<Rate> fwds5() {
    return {0.04, 0.045, 0.05, 0.055, 0.06};
}

void block_lmm_curve_state() {
    std::vector<Time> rt = rateTimes6();
    LMMCurveState cs(rt);
    cs.setOnForwardRates(fwds5());

    emit("lmm_n_rates", (Real)cs.numberOfRates());
    emit("lmm_fwd0", cs.forwardRate(0));
    emit("lmm_fwd4", cs.forwardRate(4));
    // discountRatio(i, j) = P(i)/P(j); P built from forwards.
    emit("lmm_dr_0_5", cs.discountRatio(0, 5));  // P(t0)/P(t5)
    emit("lmm_dr_2_5", cs.discountRatio(2, 5));
    emit("lmm_dr_0_3", cs.discountRatio(0, 3));
    // coterminal swap rate / annuity
    emit("lmm_cot_swap_rate_0", cs.coterminalSwapRate(0));
    emit("lmm_cot_swap_rate_2", cs.coterminalSwapRate(2));
    emit("lmm_cot_annuity_num5_0", cs.coterminalSwapAnnuity(5, 0));
    emit("lmm_cot_annuity_num5_2", cs.coterminalSwapAnnuity(5, 2));
    // cm swap rate spanning 2
    emit("lmm_cm_swap_rate_sp2_0", cs.cmSwapRate(0, 2));
    emit("lmm_cm_swap_rate_sp2_2", cs.cmSwapRate(2, 2));
    // swapRate(begin,end) from base CurveState
    emit("lmm_swap_rate_0_5", cs.swapRate(0, 5));
    emit("lmm_swap_rate_1_4", cs.swapRate(1, 4));
}

void block_cm_swap_curve_state() {
    std::vector<Time> rt = rateTimes6();
    Size spanning = 2;
    // First build an LMM state to derive consistent cm swap rates, then
    // round-trip through CMSwapCurveState.
    LMMCurveState lmm(rt);
    lmm.setOnForwardRates(fwds5());
    std::vector<Rate> cmRates(5);
    for (Size i = 0; i < 5; ++i)
        cmRates[i] = lmm.cmSwapRate(i, spanning);

    CMSwapCurveState cs(rt, spanning);
    cs.setOnCMSwapRates(cmRates);

    emit("cm_n_rates", (Real)cs.numberOfRates());
    emit("cm_cm_rate_0", cs.cmSwapRate(0, spanning));
    emit("cm_cm_rate_2", cs.cmSwapRate(2, spanning));
    // discount ratios should reconstruct the original LMM ones (TIGHT round-trip).
    emit("cm_dr_0_5", cs.discountRatio(0, 5));
    emit("cm_dr_2_5", cs.discountRatio(2, 5));
    emit("cm_fwd0", cs.forwardRate(0));
    emit("cm_cot_swap_rate_0", cs.coterminalSwapRate(0));
}

void block_coterminal_swap_curve_state() {
    std::vector<Time> rt = rateTimes6();
    LMMCurveState lmm(rt);
    lmm.setOnForwardRates(fwds5());
    std::vector<Rate> cotRates(5);
    for (Size i = 0; i < 5; ++i)
        cotRates[i] = lmm.coterminalSwapRate(i);

    CoterminalSwapCurveState cs(rt);
    cs.setOnCoterminalSwapRates(cotRates);

    emit("cot_n_rates", (Real)cs.numberOfRates());
    emit("cot_cot_rate_0", cs.coterminalSwapRate(0));
    emit("cot_cot_rate_2", cs.coterminalSwapRate(2));
    emit("cot_dr_0_5", cs.discountRatio(0, 5));
    emit("cot_dr_2_5", cs.discountRatio(2, 5));
    emit("cot_fwd0", cs.forwardRate(0));
    emit("cot_fwd4", cs.forwardRate(4));
    emit("cot_annuity_num5_0", cs.coterminalSwapAnnuity(5, 0));
}

void block_swap_forward_jacobian() {
    std::vector<Time> rt = rateTimes6();
    LMMCurveState cs(rt);
    cs.setOnForwardRates(fwds5());
    Matrix jac = SwapForwardMappings::coterminalSwapForwardJacobian(cs);
    // Upper-triangular dsr[i]/df[j]; emit a few representative entries.
    emit("jac_0_0", jac[0][0]);
    emit("jac_0_4", jac[0][4]);
    emit("jac_2_2", jac[2][2]);
    emit("jac_2_4", jac[2][4]);
    emit("jac_4_4", jac[4][4]);
    // Below-diagonal entry must be zero (swap rate i does not depend on fwd j<i).
    emit("jac_4_0", jac[4][0]);
    // annuity + swapDerivative direct helpers
    emit("jac_annuity_0_5_5", SwapForwardMappings::annuity(cs, 0, 5, 5));
    emit("jac_swapderiv_0_5_2", SwapForwardMappings::swapDerivative(cs, 0, 5, 2));
}

void block_discounter() {
    std::vector<Time> rt = rateTimes6();
    LMMCurveState cs(rt);
    cs.setOnForwardRates(fwds5());
    // Payment exactly at a rate time -> beforeWeight handling.
    MarketModelDiscounter d1(2.0, rt);  // = t3
    emit("disc_pay2_num5", d1.numeraireBonds(cs, 5));
    // Payment between rate times -> log-linear interpolation.
    MarketModelDiscounter d2(1.75, rt);
    emit("disc_pay175_num5", d2.numeraireBonds(cs, 5));
    // Payment after the last rate time -> clamps to last period.
    MarketModelDiscounter d3(3.5, rt);
    emit("disc_pay35_num5", d3.numeraireBonds(cs, 5));
    // Different numeraire.
    MarketModelDiscounter d4(1.75, rt);
    emit("disc_pay175_num0", d4.numeraireBonds(cs, 0), false);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_utilities();
    block_evolution_description();
    block_lmm_curve_state();
    block_cm_swap_curve_state();
    block_coterminal_swap_curve_state();
    block_swap_forward_jacobian();
    block_discounter();

    std::cout << "}\n";
    return 0;
}
