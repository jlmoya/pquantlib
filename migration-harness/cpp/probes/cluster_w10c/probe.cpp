// Phase 11 W10-C cluster probe: MarketModels (BGM) caplet / coterminal-swaption
// CALIBRATION.
//
// Builds on the W9 core spine + W10-A concrete vol models (FlatVol /
// PseudoRootFacade / VolatilityInterpolationSpecifier). Pins:
//
//   (a) AlphaForm concretes + AlphaFinder
//   * AlphaFormInverseLinear(t,alpha) = 1/(1+alpha*t) at known params (TIGHT).
//   * AlphaFormLinearHyperbolic(t,alpha) = sqrt(1 + a*t*(atan(a*t)-pi/2)) (TIGHT).
//   * AlphaFinder.solve(...) on a synthetic two-rate target: alpha + a + b +
//     the resulting putative rate-two vols (TIGHT — deterministic root-find).
//
//   (b) SphereCylinderOptimizer (the inner solver of max-homogeneity)
//   * findClosest + findByProjection on the two canonical C++ test fixtures
//     (one EXACT-ish, one LOOSE — matches marketmodel_smmcaplethomocalibration).
//
//   (c) CTSMMCapletMaxHomogeneityCalibration.calibrate(...) — the canonical
//       marketmodel.cpp calibration: 10-rate semiannual coterminal swap market
//       model, abcd swap variances, exponential fwd correlation. We check the
//       calibration succeeds (failures==0) and that the resulting swapPseudoRoots
//       reprice the input swaption vols (perfect, TIGHT) and the caplet vols
//       (1bp, LOOSE — iterative). Reports per-rate model caplet vols + errors.
//
//   (d) CTSMMCapletOriginalCalibration.calibrate(...) — joint caplet+swaption
//       Joshi-original calibration: per-rate model caplet/swaption vols + errors
//       (LOOSE — iterative deviation metrics).
//
// C++ parity:
//   ql/models/marketmodels/models/alphaform.hpp
//   ql/models/marketmodels/models/alphaformconcrete.hpp
//   ql/models/marketmodels/models/alphafinder.hpp
//   ql/math/optimization/spherecylinder.hpp
//   ql/models/marketmodels/models/ctsmmcapletcalibration.hpp
//   ql/models/marketmodels/models/capletcoterminalmaxhomogeneity.hpp
//   ql/models/marketmodels/models/capletcoterminalswaptioncalibration.hpp
//   @ v1.42.1 (099987f0).

#include <ql/math/matrix.hpp>
#include <ql/math/optimization/spherecylinder.hpp>
#include <ql/models/marketmodels/correlations/cotswapfromfwdcorrelation.hpp>
#include <ql/models/marketmodels/correlations/expcorrelations.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/models/alphafinder.hpp>
#include <ql/models/marketmodels/models/alphaformconcrete.hpp>
#include <ql/models/marketmodels/models/capletcoterminalmaxhomogeneity.hpp>
#include <ql/models/marketmodels/models/capletcoterminalswaptioncalibration.hpp>
#include <ql/models/marketmodels/models/cotswaptofwdadapter.hpp>
#include <ql/models/marketmodels/models/piecewiseconstantabcdvariance.hpp>
#include <ql/models/marketmodels/models/pseudorootfacade.hpp>
#include <ql/models/marketmodels/piecewiseconstantcorrelation.hpp>

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

// === (a) AlphaForm concretes + AlphaFinder =================================

void block_alpha_form() {
    std::vector<Time> times = {0.5, 1.0, 1.5, 2.0, 2.5};

    AlphaFormInverseLinear ail(times, 0.3);
    emit("ail_a0_i0", ail(0));  // 1/(1+0.3*0.5)
    emit("ail_a0_i2", ail(2));  // 1/(1+0.3*1.5)
    emit("ail_a0_i4", ail(4));  // 1/(1+0.3*2.5)
    ail.setAlpha(-0.2);
    emit("ail_a1_i3", ail(3));  // 1/(1-0.2*2.0)

    AlphaFormLinearHyperbolic alh(times, 0.4);
    emit("alh_a0_i0", alh(0));
    emit("alh_a0_i2", alh(2));
    emit("alh_a0_i4", alh(4));
    alh.setAlpha(-0.15);
    emit("alh_a1_i3", alh(3));
}

void block_alpha_finder() {
    // a synthetic 2-rate (stepindex=0) calibration: rate-one vol on [0,t0],
    // rate-two homogeneous vols on [0,t0] and [t0,t1]. The finder modifies the
    // rate-two vols by alpha-form (a,b) so the swap variance hits the target.
    std::vector<Time> times = {0.5, 1.0};
    auto form = ext::make_shared<AlphaFormLinearHyperbolic>(times);
    AlphaFinder finder(form);

    std::vector<Volatility> rateonevols = {0.20};
    std::vector<Volatility> ratetwohomog = {0.18, 0.22};
    std::vector<Real> correlations = {0.85};
    Real w0 = 0.6, w1 = 0.4;
    // total homogeneous var = 0.18^2 + 0.22^2 = 0.0324 + 0.0484 = 0.0808
    // pick a target swap variance achievable near alpha=0
    Real targetVariance = 0.02;
    Real tolerance = 1e-12;
    Real alphaMax = 1.0, alphaMin = -1.0;
    Integer steps = 100;

    Real alpha = 0.0, a = 0.0, b = 0.0;
    std::vector<Volatility> ratetwovols(2);
    bool ok = finder.solve(0.0, 0, rateonevols, ratetwohomog, correlations,
                           w0, w1, targetVariance, tolerance, alphaMax, alphaMin,
                           steps, alpha, a, b, ratetwovols);
    emit("af_solve_ok", ok ? 1.0 : 0.0);
    emit("af_alpha", alpha);
    emit("af_a", a);
    emit("af_b", b);
    emit("af_v0", ratetwovols[0]);
    emit("af_v1", ratetwovols[1]);

    // max-homogeneity variant on the same fixture
    Real alpha2 = 0.0, a2 = 0.0, b2 = 0.0;
    std::vector<Volatility> ratetwovols2(2);
    bool ok2 = finder.solveWithMaxHomogeneity(
        0.0, 0, rateonevols, ratetwohomog, correlations, w0, w1, targetVariance,
        tolerance, alphaMax, alphaMin, steps, alpha2, a2, b2, ratetwovols2);
    emit("afh_solve_ok", ok2 ? 1.0 : 0.0);
    emit("afh_alpha", alpha2);
    emit("afh_a", a2);
    emit("afh_b", b2);
    emit("afh_v0", ratetwovols2[0]);
    emit("afh_v1", ratetwovols2[1]);
}

// === (b) SphereCylinderOptimizer ===========================================

void block_sphere_cylinder() {
    {
        Real R = 1.0, S = 0.5, alpha = 1.5;
        Real Z = 1.0 / std::sqrt(3.0);
        SphereCylinderOptimizer opt(R, S, alpha, Z, Z, Z);
        emit("sc1_nonempty", opt.isIntersectionNonEmpty() ? 1.0 : 0.0);
        Real y1, y2, y3;
        opt.findClosest(100, 1e-8, y1, y2, y3);
        emit("sc1_close_y1", y1);
        emit("sc1_close_y2", y2);
        emit("sc1_close_y3", y3);
        opt.findByProjection(y1, y2, y3);
        emit("sc1_proj_y1", y1);
        emit("sc1_proj_y2", y2);
        emit("sc1_proj_y3", y3);
    }
    {
        Real R = 5.0, S = 1.0, alpha = 1.0;
        Real Z1 = 1.0, Z2 = 2.0, Z3 = std::sqrt(20.0);
        SphereCylinderOptimizer opt(R, S, alpha, Z1, Z2, Z3);
        emit("sc2_nonempty", opt.isIntersectionNonEmpty() ? 1.0 : 0.0);
        Real y1, y2, y3;
        opt.findClosest(100, 1e-8, y1, y2, y3);
        emit("sc2_close_y1", y1);
        emit("sc2_close_y2", y2);
        emit("sc2_close_y3", y3);
        opt.findByProjection(y1, y2, y3);
        emit("sc2_proj_y1", y1);
        emit("sc2_proj_y2", y2);
        emit("sc2_proj_y3", y3);
    }
}

// === shared calibration fixture (marketmodel_smmcaplethomocalibration) =====

struct Fixture {
    std::vector<Time> rateTimes;
    std::vector<Rate> forwards;
    ext::shared_ptr<LMMCurveState> cs;
    ext::shared_ptr<PiecewiseConstantCorrelation> corr;
    std::vector<ext::shared_ptr<PiecewiseConstantVariance>> swapVariances;
    std::vector<Volatility> capletVols;
    Spread displacement = 0.0;
    Size numberOfRates;
};

// rate times = the C++ test grid: semiannual schedule over 66 months gives
// 12 dates -> 11 rate times -> but accruals drop the last so numberOfRates=10.
// We reproduce the exact yearFractions with SimpleDayCounter (act/360-ish flat
// 0.5y steps): t_i = 0.5*(i+1) for i=0..10, then numberOfRates=10 forwards.
// SimpleDayCounter yearFraction over a semiannual schedule is exactly months/12.
Fixture make_fixture() {
    Fixture f;
    // 11 rate times: 0.5 .. 5.5 (months 6..66 in steps of 6 -> /12)
    f.rateTimes.resize(11);
    for (Size i = 0; i < 11; ++i)
        f.rateTimes[i] = 0.5 * (i + 1);
    f.numberOfRates = 10;

    f.forwards.resize(10);
    for (Size i = 0; i < 10; ++i)
        f.forwards[i] = 0.03 + 0.0025 * i;

    f.cs = ext::make_shared<LMMCurveState>(f.rateTimes);
    f.cs->setOnForwardRates(f.forwards);

    Real longTermCorrelation = 0.5, beta = 0.2;
    auto fwdCorr = ext::make_shared<ExponentialForwardCorrelation>(
        f.rateTimes, longTermCorrelation, beta);
    f.corr = ext::make_shared<CotSwapFromFwdCorrelation>(fwdCorr, *f.cs,
                                                         f.displacement);

    Real a = 0.0, b = 0.17, c = 1.0, d = 0.10;
    f.swapVariances.resize(10);
    for (Size i = 0; i < 10; ++i)
        f.swapVariances[i] = ext::make_shared<PiecewiseConstantAbcdVariance>(
            a, b, c, d, i, f.rateTimes);

    f.capletVols = {0.1640, 0.1740, 0.1840, 0.1940, 0.1840,
                    0.1740, 0.1640, 0.1540, 0.1440, 0.1340376439125532};
    return f;
}

// === (c) max-homogeneity full calibration ==================================

void block_max_homogeneity() {
    Fixture f = make_fixture();
    EvolutionDescription evolution(f.rateTimes);
    Real caplet0Swaption1Priority = 1.0;
    CTSMMCapletMaxHomogeneityCalibration calibrator(
        evolution, f.corr, f.swapVariances, f.capletVols, f.cs, f.displacement,
        caplet0Swaption1Priority);

    Size numberOfFactors = 3;
    Natural maxIterations = 10;
    Real capletTolerance = 1e-4;
    Natural innerMaxIterations = 100;
    Real innerTolerance = 1e-8;
    bool result = calibrator.calibrate(numberOfFactors, maxIterations,
                                       capletTolerance, innerMaxIterations,
                                       innerTolerance);
    emit("mh_result", result ? 1.0 : 0.0);
    emit("mh_failures", (Real)calibrator.failures());
    emit("mh_caplet_rms", calibrator.capletRmsError());
    emit("mh_caplet_max", calibrator.capletMaxError());
    emit("mh_swaption_rms", calibrator.swaptionRmsError());
    emit("mh_swaption_max", calibrator.swaptionMaxError());

    const std::vector<Matrix>& swapPseudoRoots = calibrator.swapPseudoRoots();
    auto smm = ext::make_shared<PseudoRootFacade>(
        swapPseudoRoots, f.rateTimes, f.cs->coterminalSwapRates(),
        std::vector<Spread>(f.numberOfRates, f.displacement));
    CotSwapToFwdAdapter flmm(smm);
    Matrix capletTotCov = flmm.totalCovariance(f.numberOfRates - 1);
    for (Size i = 0; i < f.numberOfRates; ++i) {
        Volatility v = std::sqrt(capletTotCov[i][i] / f.rateTimes[i]);
        emit(("mh_caplet_vol_" + std::to_string(i)).c_str(), v);
    }

    // perfect swaption fit: rebuild swap terminal covariance
    Matrix swapTermCov(f.numberOfRates, f.numberOfRates, 0.0);
    for (Size i = 0; i < f.numberOfRates; ++i) {
        swapTermCov += swapPseudoRoots[i] * transpose(swapPseudoRoots[i]);
        Volatility sv = std::sqrt(swapTermCov[i][i] / f.rateTimes[i]);
        emit(("mh_swaption_vol_" + std::to_string(i)).c_str(), sv);
    }
}

// === (d) original (joint) calibration ======================================

void block_original_calibration() {
    Fixture f = make_fixture();
    EvolutionDescription evolution(f.rateTimes);
    std::vector<Real> alpha(f.numberOfRates, 0.0);
    bool lowestRoot = false;
    bool useFullApprox = true;
    CTSMMCapletOriginalCalibration calibrator(
        evolution, f.corr, f.swapVariances, f.capletVols, f.cs, f.displacement,
        alpha, lowestRoot, useFullApprox);

    Size numberOfFactors = 3;
    Natural maxIterations = 10;
    Real capletTolerance = 1e-4;
    bool result = calibrator.calibrate(numberOfFactors, maxIterations,
                                       capletTolerance);
    emit("oc_result", result ? 1.0 : 0.0);
    emit("oc_failures", (Real)calibrator.failures());
    emit("oc_caplet_rms", calibrator.capletRmsError());
    emit("oc_caplet_max", calibrator.capletMaxError());
    emit("oc_swaption_rms", calibrator.swaptionRmsError());
    emit("oc_swaption_max", calibrator.swaptionMaxError());

    const std::vector<Matrix>& swapPseudoRoots = calibrator.swapPseudoRoots();
    auto smm = ext::make_shared<PseudoRootFacade>(
        swapPseudoRoots, f.rateTimes, f.cs->coterminalSwapRates(),
        std::vector<Spread>(f.numberOfRates, f.displacement));
    CotSwapToFwdAdapter flmm(smm);
    Matrix capletTotCov = flmm.totalCovariance(f.numberOfRates - 1);
    for (Size i = 0; i < f.numberOfRates; ++i) {
        Volatility v = std::sqrt(capletTotCov[i][i] / f.rateTimes[i]);
        emit(("oc_caplet_vol_" + std::to_string(i)).c_str(), v);
    }
    Matrix swapTermCov(f.numberOfRates, f.numberOfRates, 0.0);
    for (Size i = 0; i < f.numberOfRates; ++i) {
        swapTermCov += swapPseudoRoots[i] * transpose(swapPseudoRoots[i]);
        Volatility sv = std::sqrt(swapTermCov[i][i] / f.rateTimes[i]);
        bool last = (i == f.numberOfRates - 1);
        emit(("oc_swaption_vol_" + std::to_string(i)).c_str(), sv, !last);
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_alpha_form();
    block_alpha_finder();
    block_sphere_cylinder();
    block_max_homogeneity();
    block_original_calibration();

    std::cout << "}\n";
    return 0;
}
