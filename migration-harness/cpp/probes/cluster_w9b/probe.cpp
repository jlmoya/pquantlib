// Phase 11 W9-B cluster probe: MarketModels (LIBOR Market Model / BGM)
// correlations + drift calculators + historical analysis enabler.
//
// Builds on the W9-A core spine (curve states + swap<->forward mappings +
// PiecewiseConstantCorrelation abstract). Pins:
//
//   (a) correlations/
//   * exponentialCorrelations(rateTimes, L, beta, gamma, t) — exponential
//     instantaneous-correlation matrix (TIGHT — closed-form exp formula).
//   * ExponentialForwardCorrelation(rateTimes, L, beta) — gamma=1 path goes
//     through TimeHomogeneousForwardCorrelation::evolvedMatrices; gamma!=1
//     path integrates over interval midpoints (TIGHT).
//   * TimeHomogeneousForwardCorrelation::evolvedMatrices(fwdCorr) — the
//     time-homogeneous lower-right-shifted family of per-step matrices (TIGHT).
//   * CotSwapFromFwdCorrelation(fwdCorr, curveState, displacement) —
//     coterminal-swap correlation = corr(Z C Z^T) with expired-rate zeroing
//     (TIGHT — Z-matrix sandwich + CovarianceDecomposition).
//
//   (b) driftcomputation/
//   * LMMDriftCalculator.compute / computePlain / computeReduced at a known
//     LMMCurveState + pseudoRoot (TIGHT — pure linear algebra, Joshi 2003).
//   * LMMNormalDriftCalculator.compute — normal (not lognormal) variant.
//   * SMMDriftCalculator.compute at a CoterminalSwapCurveState.
//   * CMSMMDriftCalculator.compute at a CMSwapCurveState.
//
//   (c) historical analysis enabler
//   * SequenceStatistics covariance() / correlation() on a known 3-d sample
//     set (TIGHT — multi-dim moment + outer-product accumulation).
//
// C++ parity:
//   ql/models/marketmodels/correlations/expcorrelations.hpp
//   ql/models/marketmodels/correlations/timehomogeneousforwardcorrelation.hpp
//   ql/models/marketmodels/correlations/cotswapfromfwdcorrelation.hpp
//   ql/models/marketmodels/driftcomputation/lmmdriftcalculator.hpp
//   ql/models/marketmodels/driftcomputation/lmmnormaldriftcalculator.hpp
//   ql/models/marketmodels/driftcomputation/smmdriftcalculator.hpp
//   ql/models/marketmodels/driftcomputation/cmsmmdriftcalculator.hpp
//   ql/math/statistics/sequencestatistics.hpp
//   @ v1.42.1 (099987f0).

#include <ql/math/matrix.hpp>
#include <ql/math/statistics/sequencestatistics.hpp>
#include <ql/models/marketmodels/correlations/cotswapfromfwdcorrelation.hpp>
#include <ql/models/marketmodels/correlations/expcorrelations.hpp>
#include <ql/models/marketmodels/correlations/timehomogeneousforwardcorrelation.hpp>
#include <ql/models/marketmodels/curvestates/cmswapcurvestate.hpp>
#include <ql/models/marketmodels/curvestates/coterminalswapcurvestate.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/driftcomputation/cmsmmdriftcalculator.hpp>
#include <ql/models/marketmodels/driftcomputation/lmmdriftcalculator.hpp>
#include <ql/models/marketmodels/driftcomputation/lmmnormaldriftcalculator.hpp>
#include <ql/models/marketmodels/driftcomputation/smmdriftcalculator.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

// rateTimes: t0..t5, evenly spaced 0.5y apart starting at 0.5 -> 6 times,
// 5 forward rates. Same grid as W9-A.
std::vector<Time> rateTimes6() {
    std::vector<Time> rt(6);
    for (Size i = 0; i < 6; ++i)
        rt[i] = 0.5 * (i + 1);  // 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    return rt;
}

std::vector<Rate> fwds5() {
    return {0.04, 0.045, 0.05, 0.055, 0.06};
}

// --- (a) correlations -------------------------------------------------------

void block_exponential_correlations() {
    std::vector<Time> rt = rateTimes6();
    // exponentialCorrelations(rateTimes, L=0.5, beta=0.2, gamma=1, t=0)
    Matrix c = exponentialCorrelations(rt, 0.5, 0.2, 1.0, 0.0);
    emit("expc_rows", (Real)c.rows());
    emit("expc_0_0", c[0][0]);
    emit("expc_0_1", c[0][1]);
    emit("expc_0_4", c[0][4]);
    emit("expc_2_4", c[2][4]);
    emit("expc_4_4", c[4][4]);
    // with a positive evaluation time t=0.75 (some rates expired/aliveness)
    Matrix ct = exponentialCorrelations(rt, 0.5, 0.2, 1.0, 0.75);
    emit("expct_1_1", ct[1][1]);
    emit("expct_1_4", ct[1][4]);
    // gamma != 1 (time-inhomogeneous)
    Matrix cg = exponentialCorrelations(rt, 0.5, 0.2, 0.8, 0.0);
    emit("expcg_0_4", cg[0][4]);
    emit("expcg_2_4", cg[2][4]);
}

void block_time_homogeneous_forward_correlation() {
    // Build a forward correlation matrix (exp, gamma=1) and evolve it.
    std::vector<Time> rt = rateTimes6();
    Matrix fwd = exponentialCorrelations(rt, 0.5, 0.2, 1.0, 0.0);
    std::vector<Matrix> evolved =
        TimeHomogeneousForwardCorrelation::evolvedMatrices(fwd);
    emit("thfc_count", (Real)evolved.size());
    // step 0 == fwd
    emit("thfc_k0_0_0", evolved[0][0][0]);
    emit("thfc_k0_0_4", evolved[0][0][4]);
    emit("thfc_k0_2_4", evolved[0][2][4]);
    // step 1: lower-right-shifted: evolved[1][i][j] = fwd[i-1][j-1] for i,j>=1
    emit("thfc_k1_1_1", evolved[1][1][1]);
    emit("thfc_k1_1_4", evolved[1][1][4]);
    emit("thfc_k1_0_0", evolved[1][0][0]);  // = 0 (expired in step 1)
    // step 2
    emit("thfc_k2_2_4", evolved[2][2][4]);
    emit("thfc_k2_4_4", evolved[2][4][4]);
    // class wrapper: TimeHomogeneousForwardCorrelation
    TimeHomogeneousForwardCorrelation thfc(fwd, rt);
    emit("thfc_obj_n", (Real)thfc.numberOfRates());
    emit("thfc_obj_corr0_0_4", thfc.correlations()[0][0][4]);
    emit("thfc_obj_times0", thfc.times()[0]);
    emit("thfc_obj_corr_via_accessor_2_4", thfc.correlation(0)[2][4]);
}

void block_exponential_forward_correlation() {
    std::vector<Time> rt = rateTimes6();
    // gamma=1 path (default) -> time-homogeneous evolved family
    ExponentialForwardCorrelation efc(rt, 0.5, 0.2);
    emit("efc_n", (Real)efc.numberOfRates());
    emit("efc_count", (Real)efc.correlations().size());
    emit("efc_k0_0_4", efc.correlations()[0][0][4]);
    emit("efc_k1_1_4", efc.correlations()[1][1][4]);
    emit("efc_times0", efc.times()[0]);
    emit("efc_times_back", efc.times().back());
    // gamma != 1 path -> midpoint integration
    ExponentialForwardCorrelation efcg(rt, 0.5, 0.2, 0.8);
    emit("efcg_count", (Real)efcg.correlations().size());
    emit("efcg_k0_0_4", efcg.correlations()[0][0][4]);
    emit("efcg_k1_1_4", efcg.correlations()[1][1][4]);
}

void block_cot_swap_from_fwd_correlation() {
    std::vector<Time> rt = rateTimes6();
    LMMCurveState cs(rt);
    cs.setOnForwardRates(fwds5());
    auto fwdCorr = ext::make_shared<ExponentialForwardCorrelation>(
        rt, 0.5, 0.2);
    Spread displacement = 0.02;
    CotSwapFromFwdCorrelation csfc(fwdCorr, cs, displacement);
    emit("csfc_n", (Real)csfc.numberOfRates());
    emit("csfc_count", (Real)csfc.correlations().size());
    // diagonal must be 1 on the first step (no expired rates)
    emit("csfc_k0_0_0", csfc.correlations()[0][0][0]);
    emit("csfc_k0_0_4", csfc.correlations()[0][0][4]);
    emit("csfc_k0_2_4", csfc.correlations()[0][2][4]);
    emit("csfc_k0_4_4", csfc.correlations()[0][4][4]);
    // later step has expired-rate zeroing
    emit("csfc_k2_2_4", csfc.correlations()[2][2][4]);
    emit("csfc_k2_0_0", csfc.correlations()[2][0][0]);  // expired -> 0
    emit("csfc_times_back", csfc.times().back());
    emit("csfc_ratetimes_back", csfc.rateTimes().back());
}

// --- (b) drift calculators --------------------------------------------------

// A deterministic pseudo-root: full-factor lower-triangular-ish 5x5.
// vol scaled so the covariance is realistic for a 0.5y step.
Matrix pseudoRoot5() {
    Matrix p(5, 5, 0.0);
    // simple, reproducible: p[i][r] = 0.10*(1 + 0.1*i) * f(r)
    // make it full rank by a banded structure
    for (Size i = 0; i < 5; ++i) {
        for (Size r = 0; r <= i; ++r) {
            p[i][r] = 0.10 * (1.0 + 0.05 * i) / std::sqrt(Real(i + 1));
        }
    }
    return p;
}

// A rank-2 reduced pseudo-root 5x2.
Matrix pseudoRoot5x2() {
    Matrix p(5, 2, 0.0);
    for (Size i = 0; i < 5; ++i) {
        p[i][0] = 0.10 * (1.0 + 0.05 * i);
        p[i][1] = 0.02 * (1.0 - 0.1 * i);
    }
    return p;
}

void block_lmm_drift_calculator() {
    std::vector<Time> rt = rateTimes6();
    LMMCurveState cs(rt);
    cs.setOnForwardRates(fwds5());
    std::vector<Time> taus = cs.rateTaus();
    std::vector<Spread> disp(5, 0.0);
    Matrix p = pseudoRoot5();

    // numeraire = 5 (terminal), alive = 0
    LMMDriftCalculator dc(p, disp, taus, 5, 0);
    std::vector<Real> drifts(5, 0.0);
    dc.computePlain(cs, drifts);
    emit("lmm_drift_plain_0", drifts[0]);
    emit("lmm_drift_plain_1", drifts[1]);
    emit("lmm_drift_plain_2", drifts[2]);
    emit("lmm_drift_plain_3", drifts[3]);
    emit("lmm_drift_plain_4", drifts[4]);

    // full-factor: compute() should dispatch to computePlain
    std::vector<Real> drifts2(5, 0.0);
    dc.compute(cs, drifts2);
    emit("lmm_drift_compute_0", drifts2[0]);
    emit("lmm_drift_compute_4", drifts2[4]);

    // reduced (full-factor pseudo-root still valid for computeReduced)
    std::vector<Real> driftsR(5, 0.0);
    dc.computeReduced(cs, driftsR);
    emit("lmm_drift_reduced_0", driftsR[0]);
    emit("lmm_drift_reduced_4", driftsR[4]);

    // displaced + non-terminal numeraire=3, alive=1
    std::vector<Spread> disp2(5, 0.01);
    LMMDriftCalculator dc2(p, disp2, taus, 3, 1);
    std::vector<Real> drifts3(5, 0.0);
    dc2.computePlain(cs, drifts3);
    emit("lmm_drift_num3_alive1_1", drifts3[1]);
    emit("lmm_drift_num3_alive1_2", drifts3[2]);  // numeraire-1 -> 0
    emit("lmm_drift_num3_alive1_4", drifts3[4]);

    // rank-2 reduced
    Matrix p2 = pseudoRoot5x2();
    LMMDriftCalculator dc3(p2, disp, taus, 5, 0);
    std::vector<Real> drifts4(5, 0.0);
    dc3.compute(cs, drifts4);  // not full factor -> reduced
    emit("lmm_drift_rank2_0", drifts4[0]);
    emit("lmm_drift_rank2_4", drifts4[4]);
}

void block_lmm_normal_drift_calculator() {
    std::vector<Time> rt = rateTimes6();
    LMMCurveState cs(rt);
    cs.setOnForwardRates(fwds5());
    std::vector<Time> taus = cs.rateTaus();
    Matrix p = pseudoRoot5();

    LMMNormalDriftCalculator dc(p, taus, 5, 0);
    std::vector<Real> drifts(5, 0.0);
    dc.computePlain(cs, drifts);
    emit("lmmn_drift_plain_0", drifts[0]);
    emit("lmmn_drift_plain_2", drifts[2]);
    emit("lmmn_drift_plain_4", drifts[4]);

    std::vector<Real> driftsR(5, 0.0);
    dc.computeReduced(cs, driftsR);
    emit("lmmn_drift_reduced_0", driftsR[0]);
    emit("lmmn_drift_reduced_4", driftsR[4]);

    // non-terminal numeraire
    LMMNormalDriftCalculator dc2(p, taus, 3, 1);
    std::vector<Real> drifts3(5, 0.0);
    dc2.computePlain(cs, drifts3);
    emit("lmmn_drift_num3_alive1_1", drifts3[1]);
    emit("lmmn_drift_num3_alive1_4", drifts3[4]);
}

void block_smm_drift_calculator() {
    std::vector<Time> rt = rateTimes6();
    // build a CoterminalSwapCurveState consistent with the LMM forwards
    LMMCurveState lmm(rt);
    lmm.setOnForwardRates(fwds5());
    std::vector<Rate> cotRates(5);
    for (Size i = 0; i < 5; ++i)
        cotRates[i] = lmm.coterminalSwapRate(i);
    CoterminalSwapCurveState cs(rt);
    cs.setOnCoterminalSwapRates(cotRates);

    std::vector<Time> taus = cs.rateTaus();
    std::vector<Spread> disp(5, 0.0);
    Matrix p = pseudoRoot5();

    SMMDriftCalculator dc(p, disp, taus, 5, 0);
    std::vector<Real> drifts(5, 0.0);
    dc.compute(cs, drifts);
    emit("smm_drift_0", drifts[0]);
    emit("smm_drift_1", drifts[1]);
    emit("smm_drift_2", drifts[2]);
    emit("smm_drift_3", drifts[3]);
    emit("smm_drift_4", drifts[4]);

    // displaced
    std::vector<Spread> disp2(5, 0.01);
    SMMDriftCalculator dc2(p, disp2, taus, 5, 0);
    std::vector<Real> drifts2(5, 0.0);
    dc2.compute(cs, drifts2);
    emit("smm_drift_disp_0", drifts2[0]);
    emit("smm_drift_disp_4", drifts2[4]);
}

void block_cmsmm_drift_calculator() {
    std::vector<Time> rt = rateTimes6();
    Size spanning = 2;
    LMMCurveState lmm(rt);
    lmm.setOnForwardRates(fwds5());
    std::vector<Rate> cmRates(5);
    for (Size i = 0; i < 5; ++i)
        cmRates[i] = lmm.cmSwapRate(i, spanning);
    CMSwapCurveState cs(rt, spanning);
    cs.setOnCMSwapRates(cmRates);

    std::vector<Time> taus = cs.rateTaus();
    std::vector<Spread> disp(5, 0.0);
    Matrix p = pseudoRoot5();

    CMSMMDriftCalculator dc(p, disp, taus, 5, 0, spanning);
    std::vector<Real> drifts(5, 0.0);
    dc.compute(cs, drifts);
    emit("cmsmm_drift_0", drifts[0]);
    emit("cmsmm_drift_1", drifts[1]);
    emit("cmsmm_drift_2", drifts[2]);
    emit("cmsmm_drift_3", drifts[3]);
    emit("cmsmm_drift_4", drifts[4]);

    // displaced
    std::vector<Spread> disp2(5, 0.01);
    CMSMMDriftCalculator dc2(p, disp2, taus, 5, 0, spanning);
    std::vector<Real> drifts2(5, 0.0);
    dc2.compute(cs, drifts2);
    emit("cmsmm_drift_disp_0", drifts2[0]);
    emit("cmsmm_drift_disp_4", drifts2[4]);
}

// --- (c) historical analysis enabler: SequenceStatistics -------------------

void block_sequence_statistics() {
    SequenceStatistics stats(3);
    // Five 3-d samples (deterministic).
    std::vector<std::vector<Real>> samples = {
        {1.0, 2.0, 3.0},
        {2.0, 1.0, 4.0},
        {3.0, 4.0, 2.0},
        {4.0, 3.0, 5.0},
        {5.0, 6.0, 1.0},
    };
    for (const auto& s : samples)
        stats.add(s.begin(), s.end());

    emit("seqstat_samples", (Real)stats.samples());
    emit("seqstat_size", (Real)stats.size());
    std::vector<Real> m = stats.mean();
    emit("seqstat_mean0", m[0]);
    emit("seqstat_mean1", m[1]);
    emit("seqstat_mean2", m[2]);
    std::vector<Real> v = stats.variance();
    emit("seqstat_var0", v[0]);
    emit("seqstat_var2", v[2]);
    Matrix cov = stats.covariance();
    emit("seqstat_cov_0_0", cov[0][0]);
    emit("seqstat_cov_0_1", cov[0][1]);
    emit("seqstat_cov_0_2", cov[0][2]);
    emit("seqstat_cov_1_2", cov[1][2]);
    emit("seqstat_cov_2_2", cov[2][2]);
    Matrix corr = stats.correlation();
    emit("seqstat_corr_0_0", corr[0][0]);
    emit("seqstat_corr_0_1", corr[0][1]);
    emit("seqstat_corr_0_2", corr[0][2]);
    emit("seqstat_corr_1_2", corr[1][2], false);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_exponential_correlations();
    block_time_homogeneous_forward_correlation();
    block_exponential_forward_correlation();
    block_cot_swap_from_fwd_correlation();
    block_lmm_drift_calculator();
    block_lmm_normal_drift_calculator();
    block_smm_drift_calculator();
    block_cmsmm_drift_calculator();
    block_sequence_statistics();

    std::cout << "}\n";
    return 0;
}
