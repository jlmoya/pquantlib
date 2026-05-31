// Phase 11 W10-A cluster probe: MarketModels (BGM) concrete volatility models.
//
// Builds on the W9 core spine (MarketModel abstract + curve states +
// swap<->forward mappings + PiecewiseConstantCorrelation + correlations).
// Pins:
//
//   (a) concrete MarketModels + piecewise-constant variances
//   * FlatVol pseudoRoot(i) at a flat vol vector + exponential correlation
//     (TIGHT — pure linear algebra; spectral rank-reduced pseudo-sqrt).
//   * FlatVol covariance(i) and totalCovariance == sum of per-step covs (TIGHT).
//   * AbcdVol pseudoRoot(i) at known abcd params (TIGHT).
//   * PiecewiseConstantAbcdVariance.variance(i)/volatility(i)/totalVolatility(i)
//     integral vs C++ closed form (TIGHT).
//
//   (b) PseudoRootFacade + adapters
//   * PseudoRootFacade wraps a set of pseudo-roots; pseudoRoot/covariance (TIGHT).
//   * FwdToCotSwapAdapter / CotSwapToFwdAdapter round-trip: adapt
//     fwd->cotswap->fwd recovers initial forward rates (LOOSE).
//   * CotSwapToFwdAdapter initialRates == original forward rates (TIGHT).
//   * SwapCovariance via FwdToCotSwapAdapter at a known forward model (TIGHT).
//
// C++ parity:
//   ql/models/marketmodels/models/flatvol.hpp
//   ql/models/marketmodels/models/abcdvol.hpp
//   ql/models/marketmodels/models/piecewiseconstantabcdvariance.hpp
//   ql/models/marketmodels/models/pseudorootfacade.hpp
//   ql/models/marketmodels/models/fwdtocotswapadapter.hpp
//   ql/models/marketmodels/models/cotswaptofwdadapter.hpp
//   @ v1.42.1 (099987f0).

#include <ql/math/matrix.hpp>
#include <ql/models/marketmodels/correlations/expcorrelations.hpp>
#include <ql/models/marketmodels/correlations/timehomogeneousforwardcorrelation.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/models/abcdvol.hpp>
#include <ql/models/marketmodels/models/cotswaptofwdadapter.hpp>
#include <ql/models/marketmodels/models/flatvol.hpp>
#include <ql/models/marketmodels/models/fwdtocotswapadapter.hpp>
#include <ql/models/marketmodels/models/piecewiseconstantabcdvariance.hpp>
#include <ql/models/marketmodels/models/pseudorootfacade.hpp>
#include <ql/models/marketmodels/models/volatilityinterpolationspecifierabcd.hpp>
#include <ql/models/marketmodels/piecewiseconstantcorrelation.hpp>
#include <ql/models/marketmodels/swapforwardmappings.hpp>

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
// 5 forward rates. Same grid as W9-A/B.
std::vector<Time> rateTimes6() {
    std::vector<Time> rt(6);
    for (Size i = 0; i < 6; ++i)
        rt[i] = 0.5 * (i + 1);  // 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    return rt;
}

std::vector<Rate> fwds5() {
    return {0.04, 0.045, 0.05, 0.055, 0.06};
}

// --- (a) FlatVol ------------------------------------------------------------

// FlatVol on a flat 20% vol, exponential correlation (L=0.5, beta=0.2),
// numberOfFactors=5 (full rank). evolution = default (rateTimes minus last).
ext::shared_ptr<FlatVol> makeFlatVol(Size factors) {
    std::vector<Time> rt = rateTimes6();
    EvolutionDescription evolution(rt);
    Matrix fwdCorr = exponentialCorrelations(rt, 0.5, 0.2, 1.0, 0.0);
    ext::shared_ptr<PiecewiseConstantCorrelation> corr(
        new TimeHomogeneousForwardCorrelation(fwdCorr, rt));
    std::vector<Volatility> vols(5, 0.20);
    std::vector<Rate> rates = fwds5();
    std::vector<Spread> disp(5, 0.0);
    return ext::shared_ptr<FlatVol>(
        new FlatVol(vols, corr, evolution, factors, rates, disp));
}

void block_flat_vol() {
    ext::shared_ptr<FlatVol> fv = makeFlatVol(5);
    emit("fv_n_rates", (Real)fv->numberOfRates());
    emit("fv_n_factors", (Real)fv->numberOfFactors());
    emit("fv_n_steps", (Real)fv->numberOfSteps());

    // pseudoRoot at step 0 (full grid -- all rates alive)
    const Matrix& pr0 = fv->pseudoRoot(0);
    emit("fv_pr0_rows", (Real)pr0.rows());
    emit("fv_pr0_cols", (Real)pr0.columns());
    emit("fv_pr0_0_0", pr0[0][0]);
    emit("fv_pr0_1_0", pr0[1][0]);
    emit("fv_pr0_4_0", pr0[4][0]);
    emit("fv_pr0_4_4", pr0[4][4]);

    // covariance at step 0 = pr0 * pr0^T (diagonal = per-rate variance over [0,t0])
    const Matrix& cov0 = fv->covariance(0);
    emit("fv_cov0_0_0", cov0[0][0]);
    emit("fv_cov0_0_1", cov0[0][1]);
    emit("fv_cov0_4_4", cov0[4][4]);

    // covariance at step 2
    const Matrix& cov2 = fv->covariance(2);
    emit("fv_cov2_2_2", cov2[2][2]);
    emit("fv_cov2_2_3", cov2[2][3]);

    // totalCovariance(4) = sum of covariance(0..4)
    const Matrix& tc = fv->totalCovariance(4);
    emit("fv_totcov4_4_4", tc[4][4]);
    emit("fv_totcov4_0_4", tc[0][4]);

    // timeDependentVolatility: sqrt(cov[i][i]/tau) per step for rate 4
    std::vector<Volatility> tdv = fv->timeDependentVolatility(4);
    emit("fv_tdv4_step0", tdv[0]);
    emit("fv_tdv4_step4", tdv[4]);

    // rank-reduced (3 factors): pseudoRoot has 3 columns; covariance diagonal
    // is approximately preserved by normalizePseudoRoot (exactly, in fact).
    ext::shared_ptr<FlatVol> fv3 = makeFlatVol(3);
    const Matrix& pr0_3 = fv3->pseudoRoot(0);
    emit("fv3_pr0_cols", (Real)pr0_3.columns());
    const Matrix& cov0_3 = fv3->covariance(0);
    emit("fv3_cov0_0_0", cov0_3[0][0]);   // diagonal preserved by normalization
    emit("fv3_cov0_4_4", cov0_3[4][4]);
}

// --- (a) AbcdVol ------------------------------------------------------------

void block_abcd_vol() {
    std::vector<Time> rt = rateTimes6();
    EvolutionDescription evolution(rt);
    Matrix fwdCorr = exponentialCorrelations(rt, 0.5, 0.2, 1.0, 0.0);
    ext::shared_ptr<PiecewiseConstantCorrelation> corr(
        new TimeHomogeneousForwardCorrelation(fwdCorr, rt));
    // abcd params (Rebonato defaults-ish): a=-0.02, b=0.5, c=1.0, d=0.14
    Real a = -0.02, b = 0.5, c = 1.0, d = 0.14;
    std::vector<Real> ks(5, 1.0);
    std::vector<Rate> rates = fwds5();
    std::vector<Spread> disp(5, 0.0);
    AbcdVol av(a, b, c, d, ks, corr, evolution, 5, rates, disp);

    emit("av_n_rates", (Real)av.numberOfRates());
    emit("av_n_steps", (Real)av.numberOfSteps());
    const Matrix& pr0 = av.pseudoRoot(0);
    emit("av_pr0_rows", (Real)pr0.rows());
    emit("av_pr0_cols", (Real)pr0.columns());
    const Matrix& cov0 = av.covariance(0);
    emit("av_cov0_0_0", cov0[0][0]);
    emit("av_cov0_4_4", cov0[4][4]);
    const Matrix& cov3 = av.covariance(3);
    emit("av_cov3_4_4", cov3[4][4]);
    emit("av_cov3_3_4", cov3[3][4]);
    // total covariance for rate 4 across all steps
    const Matrix& tc = av.totalCovariance(4);
    emit("av_totcov4_4_4", tc[4][4]);
}

// --- (a) PiecewiseConstantAbcdVariance --------------------------------------

void block_piecewise_constant_abcd_variance() {
    std::vector<Time> rt = rateTimes6();
    Real a = -0.02, b = 0.5, c = 1.0, d = 0.14;
    // resetIndex 4 -> the last (rate 4, fixing at rt[4]=2.5)
    PiecewiseConstantAbcdVariance pv(a, b, c, d, 4, rt);
    emit("pcav_var0", pv.variance(0));
    emit("pcav_var2", pv.variance(2));
    emit("pcav_var4", pv.variance(4));
    emit("pcav_vol0", pv.volatility(0));
    emit("pcav_vol4", pv.volatility(4));
    emit("pcav_totvar4", pv.totalVariance(4));
    emit("pcav_totvol4", pv.totalVolatility(4));
    // a different reset index
    PiecewiseConstantAbcdVariance pv2(a, b, c, d, 2, rt);
    emit("pcav2_var0", pv2.variance(0));
    emit("pcav2_var2", pv2.variance(2));
    emit("pcav2_totvol2", pv2.totalVolatility(2));
}

// --- (b) PseudoRootFacade ---------------------------------------------------

void block_pseudo_root_facade() {
    // Build pseudo-roots from a FlatVol, then re-wrap them in a facade and
    // check the facade reproduces the same pseudoRoot/covariance.
    ext::shared_ptr<FlatVol> fv = makeFlatVol(5);
    std::vector<Matrix> prs;
    for (Size k = 0; k < fv->numberOfSteps(); ++k)
        prs.push_back(fv->pseudoRoot(k));
    std::vector<Time> rt = rateTimes6();
    std::vector<Rate> rates = fwds5();
    std::vector<Spread> disp(5, 0.0);
    PseudoRootFacade facade(prs, rt, rates, disp);
    emit("prf_n_rates", (Real)facade.numberOfRates());
    emit("prf_n_factors", (Real)facade.numberOfFactors());
    emit("prf_n_steps", (Real)facade.numberOfSteps());
    emit("prf_init0", facade.initialRates()[0]);
    emit("prf_init4", facade.initialRates()[4]);
    const Matrix& pr0 = facade.pseudoRoot(0);
    emit("prf_pr0_0_0", pr0[0][0]);
    emit("prf_pr0_4_4", pr0[4][4]);
    const Matrix& cov0 = facade.covariance(0);
    emit("prf_cov0_4_4", cov0[4][4]);
}

// --- (b) Adapters: Fwd->Cot->Fwd round trip + SwapCovariance ----------------

void block_adapters() {
    ext::shared_ptr<FlatVol> fwdModel = makeFlatVol(5);

    // Fwd -> coterminal swap
    ext::shared_ptr<FwdToCotSwapAdapter> cot(
        new FwdToCotSwapAdapter(fwdModel));
    emit("f2c_n_rates", (Real)cot->numberOfRates());
    emit("f2c_init0", cot->initialRates()[0]);  // first coterminal swap rate
    emit("f2c_init4", cot->initialRates()[4]);

    // SwapCovariance via the adapter at step 0 (TIGHT linear algebra)
    const Matrix& swapCov0 = cot->covariance(0);
    emit("f2c_swapcov0_0_0", swapCov0[0][0]);
    emit("f2c_swapcov0_4_4", swapCov0[4][4]);
    emit("f2c_swapcov0_3_4", swapCov0[3][4]);

    // coterminal swap -> fwd round trip (should recover initial fwd rates)
    ext::shared_ptr<CotSwapToFwdAdapter> back(
        new CotSwapToFwdAdapter(cot));
    emit("c2f_init0", back->initialRates()[0]);
    emit("c2f_init1", back->initialRates()[1]);
    emit("c2f_init4", back->initialRates()[4]);

    // pseudoRoot round trip at step 0 (should ~recover fwdModel's pseudoRoot)
    const Matrix& pr0 = back->pseudoRoot(0);
    const Matrix& orig0 = fwdModel->pseudoRoot(0);
    emit("c2f_pr0_0_0", pr0[0][0]);
    emit("orig_pr0_0_0", orig0[0][0]);
    emit("c2f_pr0_4_4", pr0[4][4]);
    emit("orig_pr0_4_4", orig0[4][4]);
}

// --- (c) VolatilityInterpolationSpecifierabcd -------------------------------

void block_vol_interpolation_specifier() {
    // period=2, offset=1, noBigRates=2 -> noSmallRates=5 (timesForSmallRates
    // has 6 entries). Big-rate times = small[offset + j*period].
    std::vector<Time> timesSmall(6);
    for (Size i = 0; i < 6; ++i)
        timesSmall[i] = 0.5 * (i + 1);  // 0.5..3.0
    std::vector<Time> bigRt = {timesSmall[1], timesSmall[3], timesSmall[5]};  // 1,2,3
    std::vector<PiecewiseConstantAbcdVariance> origVars;
    origVars.emplace_back(-0.02, 0.5, 1.0, 0.14, 0, bigRt);
    origVars.emplace_back(-0.02, 0.5, 1.0, 0.14, 1, bigRt);

    VolatilityInterpolationSpecifierabcd spec(2, 1, origVars, timesSmall, 0.0);
    emit("vis_period", (Real)spec.getPeriod());
    emit("vis_offset", (Real)spec.getOffset());
    emit("vis_no_big", (Real)spec.getNoBigRates());
    emit("vis_no_small", (Real)spec.getNoSmallRates());

    const auto& iv = spec.interpolatedVariances();
    for (Size k = 0; k < iv.size(); ++k) {
        std::string nm = "vis_small_totvol_" + std::to_string(k);
        emit(nm.c_str(), iv[k]->totalVolatility(k));
    }
    // a couple of per-step variances of the interpolated curves
    emit("vis_small0_var0", iv[0]->variance(0));
    emit("vis_small4_var0", iv[4]->variance(0));

    // after setScalingFactors (scale big rate 0 by 1.1, big rate 1 by 0.9)
    spec.setScalingFactors({1.1, 0.9});
    const auto& iv2 = spec.interpolatedVariances();
    emit("vis_scaled_small0_totvol", iv2[0]->totalVolatility(0));
    emit("vis_scaled_small2_totvol", iv2[2]->totalVolatility(2));

    // after setLastCapletVol (force terminal vol to 0.25)
    spec.setLastCapletVol(0.25);
    const auto& iv3 = spec.interpolatedVariances();
    emit("vis_lastvol_small4_totvol",
         iv3[spec.getNoSmallRates() - 1]->totalVolatility(spec.getNoSmallRates() - 1),
         false);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_flat_vol();
    block_abcd_vol();
    block_piecewise_constant_abcd_variance();
    block_pseudo_root_facade();
    block_adapters();
    block_vol_interpolation_specifier();

    std::cout << "}\n";
    return 0;
}
