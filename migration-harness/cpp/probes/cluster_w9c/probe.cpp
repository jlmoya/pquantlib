// Phase 11 W9-C cluster probe: MarketModels (LIBOR Market Model / BGM)
// Brownian generators + accounting / greek engines. Builds on the W9-A
// core spine.
//
//   * MTBrownianGenerator(factors,steps,seed) — Mersenne-Twister Brownian
//     generator. nextPath() then nextStep() first draws. The underlying
//     uniform stream is RandomSequenceGenerator<MersenneTwisterUniformRng>
//     and the Gaussian transform is InverseCumulativeNormal (Acklam), both
//     of which pquantlib ports bit-identically -> EXACT.
//
//   * SobolBrownianGeneratorBase orderedIndices() for the three orderings
//     (Factors / Steps / Diagonal) at (factors=3, steps=4) -> EXACT integer
//     index schema (independent of the Sobol direction-integer family).
//
//   * SobolBrownianGeneratorBase::transform(variates) — the Brownian-bridge
//     transform test interface on a FIXED deterministic input matrix
//     (factors=2, steps=3). This is pure bridge algebra over unit-time
//     steps, independent of the RNG -> TIGHT. (The Sobol *stream* diverges
//     between C++ Jaeckel and scipy Joe-Kuo for dim>2, so we pin the
//     deterministic algebra rather than the stream.)
//
//   * AccountingEngine — drive a deterministic stub evolver + stub
//     MultiProduct over a flat 1-step LMM world; check the discounted
//     cash-flow accumulation -> TIGHT (deterministic degenerate path).
//
//   * PathwiseAccountingEngine is NOT probed here: its C++ constructor
//     requires a concrete LogNormalFwdRateEuler (W10), so it cannot be
//     instantiated against a stub evolver. The pquantlib port is exercised
//     against a self-consistent degenerate-path analytic reference (see the
//     test) and a full cross-validation lands once W10 evolvers exist.
//
//   * ProxyGreekEngine::singlePathValues original-evolver leg on the
//     deterministic path -> TIGHT (the constrained-evolver legs need a
//     concrete ConstrainedEvolver from W10/W11, so we probe the
//     unconstrained leg only).
//
// C++ parity:
//   ql/models/marketmodels/browniangenerators/mtbrowniangenerator.hpp
//   ql/models/marketmodels/browniangenerators/sobolbrowniangenerator.hpp
//   ql/models/marketmodels/accountingengine.hpp
//   ql/models/marketmodels/pathwiseaccountingengine.hpp
//   ql/models/marketmodels/proxygreekengine.hpp
//   @ v1.42.1 (099987f0).

#include <ql/models/marketmodels/accountingengine.hpp>
#include <ql/models/marketmodels/browniangenerators/mtbrowniangenerator.hpp>
#include <ql/models/marketmodels/browniangenerators/sobolbrowniangenerator.hpp>
#include <ql/models/marketmodels/curvestate.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/evolver.hpp>
#include <ql/models/marketmodels/multiproduct.hpp>
#include <ql/models/marketmodels/proxygreekengine.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/statistics/sequencestatistics.hpp>

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

// ---------------------------------------------------------------------------
// A flat, deterministic single-step LMM world used by the accounting probes.
// rateTimes = {0.5, 1.0, 1.5, 2.0}: 3 forward rates, flat at kForward.
// The stub evolver "advances" once to a curve state set on those forwards,
// uses the TERMINAL bond (index 3) as numeraire, and the product pays a unit
// cash flow at t = 1.5 (cash-flow-time index 0). The terminal numeraire makes
// numeraireBonds = P(1.5)/P(2.0) != 1, so the loop result is non-trivial.
// The path is single-step (done after the first nextTimeStep).
// ---------------------------------------------------------------------------

std::vector<Time> rateTimes4() { return {0.5, 1.0, 1.5, 2.0}; }

constexpr Real kForward = 0.05;       // flat forward rate for all 3 rates
constexpr Size kNumeraire = 3;        // terminal bond (rateTimes index 3)
constexpr Real kInitNumeraire = 100.0;
constexpr Real kCashAmount = 1.0;     // unit cash flow at t = 1.5
constexpr Real kCashFlowTime = 1.5;

// --- stub MarketModelEvolver: one step to an LMM state on flat forwards -----
class StubEvolver : public MarketModelEvolver {
  public:
    StubEvolver()
    : numeraires_(1, kNumeraire), state_(rateTimes4()), step_(0) {
        state_.setOnForwardRates(std::vector<Rate>(3, kForward));
    }
    const std::vector<Size>& numeraires() const override { return numeraires_; }
    Real startNewPath() override { step_ = 0; return 1.0; }
    Real advanceStep() override { ++step_; return 1.0; }
    Size currentStep() const override { return step_; }
    const CurveState& currentState() const override { return state_; }
    void setInitialState(const CurveState&) override {}
  private:
    std::vector<Size> numeraires_;
    LMMCurveState state_;
    Size step_;
};

// --- stub MultiProduct: pays kCashAmount once at t = 1.5 --------------------
class StubProduct : public MarketModelMultiProduct {
  public:
    StubProduct() : evolution_(rateTimes4()), done_(false) {}
    std::vector<Time> possibleCashFlowTimes() const override { return {kCashFlowTime}; }
    Size numberOfProducts() const override { return 1; }
    Size maxNumberOfCashFlowsPerProductPerStep() const override { return 1; }
    void reset() override { done_ = false; }
    std::vector<Size> suggestedNumeraires() const override { return {kNumeraire}; }
    const EvolutionDescription& evolution() const override { return evolution_; }
    bool nextTimeStep(
        const CurveState&,
        std::vector<Size>& numberCashFlowsThisStep,
        std::vector<std::vector<CashFlow> >& cashFlowsGenerated) override {
        if (!done_) {
            numberCashFlowsThisStep[0] = 1;
            cashFlowsGenerated[0][0].timeIndex = 0;
            cashFlowsGenerated[0][0].amount = kCashAmount;
            done_ = true;
            return true;  // single-step path: done after the first flow
        }
        numberCashFlowsThisStep[0] = 0;
        return true;
    }
    std::unique_ptr<MarketModelMultiProduct> clone() const override {
        return std::unique_ptr<MarketModelMultiProduct>(new StubProduct(*this));
    }
  private:
    EvolutionDescription evolution_;
    bool done_;
};

void block_mt_brownian_generator() {
    // factors=2, steps=2, seed=42. Deterministic MT stream.
    MTBrownianGenerator gen(2, 2, 42);
    emit("mt_factors", (Real)gen.numberOfFactors());
    emit("mt_steps", (Real)gen.numberOfSteps());
    Real pathWeight = gen.nextPath();
    emit("mt_path_weight", pathWeight);
    std::vector<Real> step0(2), step1(2);
    Real w0 = gen.nextStep(step0);
    Real w1 = gen.nextStep(step1);
    emit("mt_step_weight", w0);
    emit("mt_s0_0", step0[0]);
    emit("mt_s0_1", step0[1]);
    emit("mt_s1_0", step1[0]);
    emit("mt_s1_1", step1[1]);
    (void)w1;
}

void block_sobol_ordered_indices() {
    // factors=3, steps=4. Probe the integer ordering schema for each
    // ordering. We read orderedIndices() via the public test interface.
    const Size factors = 3, steps = 4;

    SobolBrownianGenerator byFactor(factors, steps, SobolBrownianGenerator::Factors, 42);
    const auto& f = byFactor.orderedIndices();
    // Factors: counter increments row-major (i outer, j inner).
    emit("sob_factors_00", (Real)f[0][0]);
    emit("sob_factors_03", (Real)f[0][3]);
    emit("sob_factors_10", (Real)f[1][0]);
    emit("sob_factors_23", (Real)f[2][3]);

    SobolBrownianGenerator byStep(factors, steps, SobolBrownianGenerator::Steps, 42);
    const auto& s = byStep.orderedIndices();
    // Steps: counter increments column-major (j outer, i inner).
    emit("sob_steps_00", (Real)s[0][0]);
    emit("sob_steps_10", (Real)s[1][0]);
    emit("sob_steps_01", (Real)s[0][1]);
    emit("sob_steps_23", (Real)s[2][3]);

    SobolBrownianGenerator byDiag(factors, steps, SobolBrownianGenerator::Diagonal, 42);
    const auto& d = byDiag.orderedIndices();
    emit("sob_diag_00", (Real)d[0][0]);
    emit("sob_diag_10", (Real)d[1][0]);
    emit("sob_diag_01", (Real)d[0][1]);
    emit("sob_diag_23", (Real)d[2][3]);
}

void block_sobol_transform() {
    // factors=2, steps=3, Factors ordering. Feed a FIXED variate matrix
    // (dimension = factors*steps = 6, nPaths = 1) and read the
    // Brownian-bridged output. Pure bridge algebra over unit-time steps.
    const Size factors = 2, steps = 3;
    SobolBrownianGenerator gen(factors, steps, SobolBrownianGenerator::Factors, 42);

    // variates[k][path]; k = 0..5. Fixed deterministic values.
    std::vector<std::vector<Real> > variates(factors * steps,
                                             std::vector<Real>(1));
    Real vals[6] = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6};
    for (Size k = 0; k < factors * steps; ++k)
        variates[k][0] = vals[k];

    std::vector<std::vector<Real> > out = gen.transform(variates);
    // out[factor][path*steps + step]
    emit("sob_xf_f0_s0", out[0][0]);
    emit("sob_xf_f0_s1", out[0][1]);
    emit("sob_xf_f0_s2", out[0][2]);
    emit("sob_xf_f1_s0", out[1][0]);
    emit("sob_xf_f1_s1", out[1][1]);
    emit("sob_xf_f1_s2", out[1][2]);
}

void block_accounting_engine() {
    auto evolver = ext::make_shared<StubEvolver>();
    StubProduct product;
    AccountingEngine engine(evolver, Clone<MarketModelMultiProduct>(product),
                            kInitNumeraire);
    SequenceStatisticsInc stats(1);
    engine.multiplePathValues(stats, 4);  // deterministic -> all paths equal
    // value = amount * numeraireBonds(state, numeraire=1) * initialNumeraire.
    // For payment at t=2.0 == rateTime[1], numeraire 1, discounter clamps to
    // the single period; numeraireBonds = P(before=0)/P(1)-rebased.
    emit("acc_mean", stats.mean()[0]);
    emit("acc_samples", (Real)stats.samples());
}

void block_proxy_greek_engine() {
    auto evolver = ext::make_shared<StubEvolver>();
    StubProduct product;
    // No constrained evolvers / diff weights -> probe the original-evolver
    // leg only (the constrained legs need a concrete ConstrainedEvolver).
    std::vector<std::vector<ext::shared_ptr<ConstrainedEvolver> > > noConstr;
    std::vector<std::vector<std::vector<Real> > > noWeights;
    // one (start,end) swap-rate constraint range per evolution step (3 here).
    std::vector<Size> startIdx(3, 0), endIdx(3, 3);
    ProxyGreekEngine engine(evolver, noConstr, noWeights, startIdx, endIdx,
                            Clone<MarketModelMultiProduct>(product),
                            kInitNumeraire);
    std::vector<Real> values(1);
    std::vector<std::vector<std::vector<Real> > > modified;
    engine.singlePathValues(values, modified);
    emit("proxy_value", values[0], false);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_mt_brownian_generator();
    block_sobol_ordered_indices();
    block_sobol_transform();
    block_accounting_engine();
    block_proxy_greek_engine();

    std::cout << "}\n";
    return 0;
}
