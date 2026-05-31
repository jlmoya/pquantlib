// Phase 11 W10-B cluster probe: MarketModels (BGM) forward-/swap-rate EVOLVERS.
//
// Builds on the W9 core (MarketModelEvolver abstract + curve states + drift
// calculators + MT/Sobol BrownianGenerators) and the W10-A concrete models
// (FlatVol). Each evolver advances the rates step-by-step under a MarketModel
// + a BrownianGenerator, applying the drift + pseudo-root diffusion.
//
// Cross-validation strategy (Brownian-stream note): the C++ MTBrownianGenerator
// uses RandomSequenceGenerator<MersenneTwisterUniformRng> + InverseCumulative
// Normal, which pquantlib ports BIT-IDENTICALLY. So driving every evolver from
// MTBrownianGeneratorFactory(seed) makes the Gaussian increments match C++
// exactly, and the evolved rates are cross-validated TIGHT (the entire
// RNG -> drift -> diffusion pipeline is probed, not just the algebra). We use
// MT rather than Sobol precisely because Sobol's stream (scipy Joe-Kuo)
// diverges from C++ (Jaeckel) beyond ~2 factors (known W9-C finding).
//
// Pins (all with MTBrownianGeneratorFactory(seed=42), one path):
//   * LogNormalFwdRatePc      — evolved forward rates after a full path (TIGHT).
//   * LogNormalFwdRateEuler   — evolved forward rates (TIGHT).
//   * LogNormalFwdRateIpc     — terminal measure, evolved forward rates (TIGHT).
//   * LogNormalFwdRateBalland — evolved forward rates (TIGHT).
//   * LogNormalFwdRateiBalland— terminal measure, evolved forward rates (TIGHT).
//   * NormalFwdRatePc         — normal model, evolved forward rates (TIGHT).
//   * LogNormalCotSwapRatePc  — evolved coterminal swap rates (TIGHT).
//   * LogNormalCmSwapRatePc   — evolved CM swap rates, span=2 (TIGHT).
//   * SVDDFwdRatePc           — SquareRootAndersen vol process, fwd rates (TIGHT).
//   * SquareRootAndersen      — QE sub-step draws + stepSd, fixed variates (TIGHT).
//   * 2-factor multi-step (initialStep>0) sanity (TIGHT).
//
// C++ parity:
//   ql/models/marketmodels/evolvers/lognormalfwdratepc.hpp
//   ql/models/marketmodels/evolvers/lognormalfwdrateeuler.hpp
//   ql/models/marketmodels/evolvers/lognormalfwdrateeulerconstrained.hpp
//   ql/models/marketmodels/evolvers/lognormalfwdrateipc.hpp
//   ql/models/marketmodels/evolvers/lognormalfwdrateballand.hpp
//   ql/models/marketmodels/evolvers/lognormalfwdrateiballand.hpp
//   ql/models/marketmodels/evolvers/normalfwdratepc.hpp
//   ql/models/marketmodels/evolvers/lognormalcotswapratepc.hpp
//   ql/models/marketmodels/evolvers/lognormalcmswapratepc.hpp
//   ql/models/marketmodels/evolvers/svddfwdratepc.hpp
//   ql/models/marketmodels/evolvers/marketmodelvolprocess.hpp
//   ql/models/marketmodels/evolvers/volprocesses/squarerootandersen.hpp
//   @ v1.42.1 (099987f0).

#include <ql/math/matrix.hpp>
#include <ql/models/marketmodels/browniangenerators/mtbrowniangenerator.hpp>
#include <ql/models/marketmodels/correlations/expcorrelations.hpp>
#include <ql/models/marketmodels/correlations/timehomogeneousforwardcorrelation.hpp>
#include <ql/models/marketmodels/curvestates/cmswapcurvestate.hpp>
#include <ql/models/marketmodels/curvestates/coterminalswapcurvestate.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/evolvers/lognormalcmswapratepc.hpp>
#include <ql/models/marketmodels/evolvers/lognormalcotswapratepc.hpp>
#include <ql/models/marketmodels/evolvers/lognormalfwdrateballand.hpp>
#include <ql/models/marketmodels/evolvers/lognormalfwdrateeuler.hpp>
#include <ql/models/marketmodels/evolvers/lognormalfwdrateiballand.hpp>
#include <ql/models/marketmodels/evolvers/lognormalfwdrateipc.hpp>
#include <ql/models/marketmodels/evolvers/lognormalfwdratepc.hpp>
#include <ql/models/marketmodels/evolvers/normalfwdratepc.hpp>
#include <ql/models/marketmodels/evolvers/svddfwdratepc.hpp>
#include <ql/models/marketmodels/evolvers/volprocesses/squarerootandersen.hpp>
#include <ql/models/marketmodels/models/flatvol.hpp>
#include <ql/models/marketmodels/models/fwdtocotswapadapter.hpp>
#include <ql/models/marketmodels/models/pseudorootfacade.hpp>
#include <ql/models/marketmodels/piecewiseconstantcorrelation.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

// rateTimes: t0..t5, evenly spaced 0.5y apart -> 6 times, 5 forward rates.
// Same grid as W9 / W10-A.
std::vector<Time> rateTimes6() {
    std::vector<Time> rt(6);
    for (Size i = 0; i < 6; ++i)
        rt[i] = 0.5 * (i + 1);  // 0.5..3.0
    return rt;
}

std::vector<Rate> fwds5() {
    return {0.04, 0.045, 0.05, 0.055, 0.06};
}

std::vector<Real> zeroDisplacements5() {
    return std::vector<Real>(5, 0.0);
}

// FlatVol on a flat 12% vol, exponential correlation (L=0.5, beta=0.1),
// at the requested factor count. evolution = default (rateTimes minus last).
ext::shared_ptr<FlatVol> makeFlatVol(Size factors) {
    std::vector<Time> rt = rateTimes6();
    EvolutionDescription evolution(rt);
    std::vector<Volatility> vols(5, 0.12);
    Matrix corr =
        exponentialCorrelations(evolution.rateTimes(), 0.5, 0.1, 1.0, 0.0);
    ext::shared_ptr<PiecewiseConstantCorrelation> pcc(
        new TimeHomogeneousForwardCorrelation(corr, rt));
    return ext::make_shared<FlatVol>(vols, pcc, evolution, factors, fwds5(),
                                     zeroDisplacements5());
}

// Emit the per-step pseudo-roots of a market model as a flat JSON block so the
// Python side can build a byte-identical PseudoRootFacade. This isolates the
// drift+diffusion evolution math from the eigensolver sign convention: the
// spectral pseudo-root is only unique up to an orthogonal rotation / sign of
// each eigenvector, so FlatVol's raw A differs between C++ (Jacobi) and Python
// (LAPACK) for expired-rate (zero-row) steps even though A @ A^T agrees. By
// feeding both sides the SAME A via PseudoRootFacade, the evolved rates match
// TIGHT. (Recommendation (b) of the Brownian-stream note, applied to the
// pseudo-root instead of the RNG stream.)
void emitPseudoRoots(const char* prefix, const ext::shared_ptr<MarketModel>& mm) {
    Size steps = mm->numberOfSteps();
    Size n = mm->numberOfRates();
    Size f = mm->numberOfFactors();
    for (Size s = 0; s < steps; ++s) {
        const Matrix& A = mm->pseudoRoot(s);
        for (Size i = 0; i < n; ++i)
            for (Size k = 0; k < f; ++k) {
                std::string nm = std::string(prefix) + "_pr_" +
                                 std::to_string(s) + "_" + std::to_string(i) +
                                 "_" + std::to_string(k);
                emit(nm.c_str(), A[i][k]);
            }
    }
    // also emit the model's initial rates so the Python facade matches exactly
    // (FlatVol -> forward rates; FwdToCotSwapAdapter -> coterminal swap rates).
    const std::vector<Rate>& init = mm->initialRates();
    for (Size i = 0; i < init.size(); ++i) {
        std::string nm = std::string(prefix) + "_init_" + std::to_string(i);
        emit(nm.c_str(), init[i]);
    }
}

// Build a PseudoRootFacade over a market model's own pseudo-roots. The facade
// is itself a MarketModel, so every evolver runs against the identical A's that
// emitPseudoRoots wrote to JSON.
ext::shared_ptr<MarketModel> facadeOf(const ext::shared_ptr<MarketModel>& mm) {
    std::vector<Matrix> prs;
    for (Size s = 0; s < mm->numberOfSteps(); ++s)
        prs.push_back(mm->pseudoRoot(s));
    return ext::shared_ptr<MarketModel>(new PseudoRootFacade(
        prs, mm->evolution().rateTimes(), mm->initialRates(),
        mm->displacements()));
}

// Run a single path through an evolver and emit the final forward rates.
template <class Evolver>
void emitFinalForwards(const char* prefix, Evolver& evolver, bool lastBlock = false) {
    evolver.startNewPath();
    Size steps = evolver.numeraires().size();
    for (Size s = 0; s < steps; ++s)
        evolver.advanceStep();
    const std::vector<Rate>& f = evolver.currentState().forwardRates();
    for (Size i = 0; i < f.size(); ++i) {
        std::string nm = std::string(prefix) + "_" + std::to_string(i);
        bool comma = !(lastBlock && i + 1 == f.size());
        emit(nm.c_str(), f[i], comma);
    }
}

// --- (a) LMM forward-rate evolvers ------------------------------------------

void block_pc() {
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    LogNormalFwdRatePc evolver(mm, gf, num, 0);
    emitFinalForwards("pc_fwd", evolver);
}

void block_euler() {
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    LogNormalFwdRateEuler evolver(mm, gf, num, 0);
    emitFinalForwards("euler_fwd", evolver);
}

void block_ipc() {
    // ipc requires terminal measure.
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    LogNormalFwdRateIpc evolver(mm, gf, num, 0);
    emitFinalForwards("ipc_fwd", evolver);
}

void block_balland() {
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    LogNormalFwdRateBalland evolver(mm, gf, num, 0);
    emitFinalForwards("balland_fwd", evolver);
}

void block_iballand() {
    // iBalland requires terminal measure.
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    LogNormalFwdRateiBalland evolver(mm, gf, num, 0);
    emitFinalForwards("iballand_fwd", evolver);
}

void block_pc_mm() {
    // spot (money-market) measure PC — exercises the non-terminal drift branch.
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(7UL);
    auto num = moneyMarketMeasure(mm->evolution());
    LogNormalFwdRatePc evolver(mm, gf, num, 0);
    emitFinalForwards("pc_mm_fwd", evolver);
}

void block_pc_2factor() {
    // 2-factor reduced model — exercises computeReduced drift path.
    auto mm = facadeOf(makeFlatVol(2));
    MTBrownianGeneratorFactory gf(99UL);
    auto num = terminalMeasure(mm->evolution());
    LogNormalFwdRatePc evolver(mm, gf, num, 0);
    emitFinalForwards("pc_2f_fwd", evolver);
}

// --- (b) Normal PC ----------------------------------------------------------

void block_normal_pc() {
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    NormalFwdRatePc evolver(mm, gf, num, 0);
    emitFinalForwards("normal_pc_fwd", evolver);
}

// --- (b) Coterminal swap-rate PC --------------------------------------------

void block_cotswap_pc() {
    // Build a forward FlatVol model, then re-express as a swap-rate model via
    // FwdToCotSwapAdapter (its initialRates are coterminal swap rates). The
    // CotSwap evolver's currentState() is a CoterminalSwapCurveState; emit the
    // evolved coterminal swap rates.
    auto fwdmm = makeFlatVol(5);
    ext::shared_ptr<MarketModel> swapmm0(new FwdToCotSwapAdapter(fwdmm));
    auto swapmm = facadeOf(swapmm0);
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(swapmm->evolution());
    LogNormalCotSwapRatePc evolver(swapmm, gf, num, 0);
    evolver.startNewPath();
    Size steps = num.size();
    for (Size s = 0; s < steps; ++s)
        evolver.advanceStep();
    const std::vector<Rate>& sr = evolver.currentState().coterminalSwapRates();
    for (Size i = 0; i < sr.size(); ++i) {
        std::string nm = std::string("cotswap_sr_") + std::to_string(i);
        emit(nm.c_str(), sr[i]);
    }
    // also emit the implied forward rates of the final state
    const std::vector<Rate>& f = evolver.currentState().forwardRates();
    for (Size i = 0; i < f.size(); ++i) {
        std::string nm = std::string("cotswap_fwd_") + std::to_string(i);
        emit(nm.c_str(), f[i]);
    }
}

// --- (b) CM swap-rate PC ----------------------------------------------------

void block_cmswap_pc() {
    // CM-swap model directly from a FlatVol forward model: the CM evolver
    // interprets the model's initialRates as CM swap rates (span=2) and evolves
    // them. We probe the evolved CM swap rates + implied forwards.
    Size span = 2;
    auto mm = facadeOf(makeFlatVol(5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    LogNormalCmSwapRatePc evolver(span, mm, gf, num, 0);
    evolver.startNewPath();
    Size steps = num.size();
    for (Size s = 0; s < steps; ++s)
        evolver.advanceStep();
    const std::vector<Rate>& sr = evolver.currentState().cmSwapRates(span);
    for (Size i = 0; i < sr.size(); ++i) {
        std::string nm = std::string("cmswap_sr_") + std::to_string(i);
        emit(nm.c_str(), sr[i]);
    }
    const std::vector<Rate>& f = evolver.currentState().forwardRates();
    for (Size i = 0; i < f.size(); ++i) {
        std::string nm = std::string("cmswap_fwd_") + std::to_string(i);
        emit(nm.c_str(), f[i]);
    }
}

// --- (c) SquareRootAndersen vol process (standalone) ------------------------

void block_square_root_andersen() {
    // QE discretization of a square-root variance process. Feed a fixed set of
    // variates (so the draws are deterministic) and probe the sub-step states +
    // stepSd. Parameters: meanLevel=0.04, reversionSpeed=1.0, volVar=0.3,
    // v0=0.04, 5 evolution times, 2 sub-steps, w1=0.5, w2=0.5, cut=1.5.
    std::vector<Time> evolTimes(5);
    for (Size i = 0; i < 5; ++i)
        evolTimes[i] = 0.5 * (i + 1);
    SquareRootAndersen sra(0.04, 1.0, 0.3, 0.04, evolTimes, 2, 0.5, 0.5, 1.5);
    emit("sra_variates_per_step", static_cast<Real>(sra.variatesPerStep()));
    emit("sra_number_state_vars", static_cast<Real>(sra.numberStateVariables()));

    sra.nextPath();
    // step 1: variates {0.3, -0.5}
    {
        std::vector<Real> v = {0.3, -0.5};
        Real w = sra.nextstep(v);
        emit("sra_step1_weight", w);
        emit("sra_step1_sd", sra.stepSd());
        emit("sra_step1_state", sra.stateVariables()[0]);
    }
    // step 2: variates {-1.0, 0.8}
    {
        std::vector<Real> v = {-1.0, 0.8};
        Real w = sra.nextstep(v);
        emit("sra_step2_weight", w);
        emit("sra_step2_sd", sra.stepSd());
        emit("sra_step2_state", sra.stateVariables()[0]);
    }
    // step 3: large negative variate to exercise the psi>cut (exponential)
    // branch and the u<p -> v=0 branch.
    {
        std::vector<Real> v = {-3.0, -3.0};
        Real w = sra.nextstep(v);
        emit("sra_step3_weight", w);
        emit("sra_step3_sd", sra.stepSd());
        emit("sra_step3_state", sra.stateVariables()[0]);
    }
}

// --- (c) SVDDFwdRatePc (displaced-diffusion with vol process) ---------------

void block_svdd() {
    // SVD-reduced-factor displaced-diffusion PC evolver with an external
    // SquareRootAndersen vol process. The vol process draws variatesPerStep
    // extra Gaussians per step which the evolver interleaves into the generator
    // stream (isVolVariate schedule). Probe the evolved forward rates.
    auto mm = facadeOf(makeFlatVol(5));
    std::vector<Time> evolTimes = mm->evolution().evolutionTimes();
    ext::shared_ptr<MarketModelVolProcess> vol(
        new SquareRootAndersen(1.0, 1.0, 0.3, 1.0, evolTimes, 2, 0.5, 0.5, 1.5));
    MTBrownianGeneratorFactory gf(42UL);
    auto num = terminalMeasure(mm->evolution());
    // firstVolatilityFactor=5 (after the 5 rate factors), volFactorStep=1.
    SVDDFwdRatePc evolver(mm, gf, vol, 5, 1, num, 0);
    emitFinalForwards("svdd_fwd", evolver, /*lastBlock=*/true);
}

// Emit the shared pseudo-root matrices the Python tests rebuild into
// PseudoRootFacades. pr5 = 5-factor FlatVol (used by pc/euler/ipc/balland/
// iballand/normal/cmswap/svdd); pr2 = 2-factor FlatVol (pc_2f); cotswap_pr =
// FwdToCotSwapAdapter over the 5-factor FlatVol (cotswap).
void block_pseudo_roots() {
    emitPseudoRoots("pr5", facadeOf(makeFlatVol(5)));
    emitPseudoRoots("pr2", facadeOf(makeFlatVol(2)));
    ext::shared_ptr<MarketModel> swap0(new FwdToCotSwapAdapter(makeFlatVol(5)));
    emitPseudoRoots("cotswap_pr", facadeOf(swap0));
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_pseudo_roots();
    block_pc();
    block_euler();
    block_ipc();
    block_balland();
    block_iballand();
    block_pc_mm();
    block_pc_2factor();
    block_normal_pc();
    block_cotswap_pc();
    block_cmswap_pc();
    block_square_root_andersen();
    block_svdd();

    std::cout << "}\n";
    return 0;
}
