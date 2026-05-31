// Phase 11 W11-D cluster probe: MarketModels (BGM) pathwise-greeks Jacobians.
//
//   Builds on W9 (CurveState/EvolutionDescription/SwapForwardMappings),
//   W10 (FlatVol/PseudoRootFacade), W11-A (MultiProduct framework). The classes
//   probed are the analytic building blocks of pathwise market vegas:
//
//   * RatePseudoRootJacobian / RatePseudoRootJacobianAllElements — the analytic
//     Jacobian dR_j / d(pseudoRoot element) for the one-step log-Euler forward
//     evolution (GG paper page 95 "B" matrix). Driven by a fully deterministic
//     (pseudoRoot, oldRates, oneStepDFs, newRates, gaussians) tuple so the pure
//     linear algebra cross-validates TIGHT. We also emit the FINITE-DIFFERENCE
//     bump of the same one-step evolution (the RatePseudoRootJacobianNumerical
//     class) so the Python test can cross-check analytic vs numerical (LOOSE).
//
//   * SwaptionPseudoDerivative — d(swaption implied vol / variance) /
//     d(pseudoRoot) for a coterminal swaption on a FlatVol model. variance,
//     impliedVolatility, expiry + per-step variance/volatility derivative
//     matrices, all TIGHT.
//
//   * CapPseudoDerivative — d(cap implied vol / price) / d(pseudoRoot). The cap
//     implied vol is recovered by a Brent solve over the sum-of-caplets Black
//     price. impliedVolatility + per-step price/vol derivative matrices TIGHT.
//
//   * VegaBumpCluster / VegaBumpCollection — the deterministic clustering of
//     pseudo-root elements into bump groups. doesIntersect / isCompatible /
//     numberBumps / isFull / isNonOverlapping / isSensible schema, TIGHT.
//
//   * VolatilityBumpInstrumentJacobian — instrument-level vega-bump Jacobian:
//     derivativesVolatility + onePercentBump + getAllOnePercentBumps, TIGHT.
//
//   * OrthogonalizedBumpFinder — orthogonalised pseudo-root bump directions
//     (drives PathwiseVegasAccountingEngine). The flattened bump magnitudes
//     reproduce the target instrument vegas, LOOSE.
//
// C++ parity:
//   ql/models/marketmodels/pathwisegreeks/ratepseudorootjacobian.{hpp,cpp}
//   ql/models/marketmodels/pathwisegreeks/swaptionpseudojacobian.{hpp,cpp}
//   ql/models/marketmodels/pathwisegreeks/bumpinstrumentjacobian.{hpp,cpp}
//   ql/models/marketmodels/pathwisegreeks/vegabumpcluster.{hpp,cpp}
//   test-suite/marketmodel.cpp testPathwiseVegas / testVegaBumpInstrumentJacobian
//   @ v1.42.1 (099987f0).

#include <ql/math/matrix.hpp>
#include <ql/models/marketmodels/correlations/expcorrelations.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/models/flatvol.hpp>
#include <ql/models/marketmodels/models/pseudorootfacade.hpp>
#include <ql/models/marketmodels/pathwisegreeks/bumpinstrumentjacobian.hpp>
#include <ql/models/marketmodels/pathwisegreeks/ratepseudorootjacobian.hpp>
#include <ql/models/marketmodels/pathwisegreeks/swaptionpseudojacobian.hpp>
#include <ql/models/marketmodels/pathwisegreeks/vegabumpcluster.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace QuantLib;

namespace {

bool g_first = true;

void sep() {
    if (!g_first) std::cout << ",\n";
    g_first = false;
}

void emit(const char* name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
}

void emit_int(const char* name, long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_bool(const char* name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_arr(const char* name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const char* name, const std::vector<long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

// flatten a Matrix row-major
void emit_mat(const char* name, const Matrix& m) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < m.rows(); ++i) {
        for (Size j = 0; j < m.columns(); ++j) {
            if (i || j) std::cout << ", ";
            std::cout << std::setprecision(17) << m[i][j];
        }
    }
    std::cout << "]";
}

// flatten a vector<Matrix> all row-major concatenated
void emit_mats(const char* name, const std::vector<Matrix>& ms) {
    sep();
    std::cout << "  \"" << name << "\": [";
    bool first = true;
    for (const auto& m : ms)
        for (Size i = 0; i < m.rows(); ++i)
            for (Size j = 0; j < m.columns(); ++j) {
                if (!first) std::cout << ", ";
                first = false;
                std::cout << std::setprecision(17) << m[i][j];
            }
    std::cout << "]";
}

// ----------------------------------------------------------------------------
// Shared deterministic FlatVol market model.
// 6 rates, 3 factors, semiannual rate times, exponential correlation.
// ----------------------------------------------------------------------------

struct Setup {
    std::vector<Time> rateTimes;
    std::vector<Real> accruals;
    std::vector<Rate> forwards;
    std::vector<Spread> displacements;
    std::vector<Volatility> volatilities;
    Size factors;
    EvolutionDescription evolution;
    ext::shared_ptr<MarketModel> model;
};

Setup makeSetup() {
    Setup s;
    Size n = 6;
    s.factors = 3;
    // semiannual: rateTimes has n+1 entries
    s.rateTimes.resize(n + 1);
    for (Size i = 0; i <= n; ++i)
        s.rateTimes[i] = 0.5 * (i + 1);  // 0.5, 1.0, ..., 3.5
    s.accruals.resize(n);
    for (Size i = 0; i < n; ++i)
        s.accruals[i] = s.rateTimes[i + 1] - s.rateTimes[i];

    s.forwards.resize(n);
    s.displacements.resize(n);
    s.volatilities.resize(n);
    for (Size i = 0; i < n; ++i) {
        s.forwards[i] = 0.03 + 0.002 * i;
        s.displacements[i] = 0.0;
        s.volatilities[i] = 0.12 + 0.005 * i;
    }

    s.evolution = EvolutionDescription(s.rateTimes);

    Real longTermCorr = 0.5;
    Real beta = 0.2;
    ext::shared_ptr<PiecewiseConstantCorrelation> corr(
        new ExponentialForwardCorrelation(s.rateTimes, longTermCorr, beta));

    s.model = ext::shared_ptr<MarketModel>(new FlatVol(
        s.volatilities, corr, s.evolution, s.factors, s.forwards, s.displacements));
    return s;
}

void block_setup(const Setup& s) {
    emit_int("n_rates", (long)s.forwards.size());
    emit_int("n_factors", (long)s.factors);
    emit_int("n_steps", (long)s.evolution.numberOfSteps());
    emit_arr("rate_times", s.rateTimes);
    emit_arr("rate_taus", s.evolution.rateTaus());
    emit_arr("accruals", s.accruals);
    emit_arr("forwards", s.forwards);
    emit_arr("displacements", s.displacements);
    emit_arr("volatilities", s.volatilities);
    std::vector<long> alive(s.evolution.numberOfSteps());
    for (Size i = 0; i < alive.size(); ++i)
        alive[i] = (long)s.evolution.firstAliveRate()[i];
    emit_iarr("first_alive_rate", alive);
    // emit each step pseudoRoot so Python can build a PseudoRootFacade-free
    // deterministic fixture if needed
    std::vector<Matrix> pseudos;
    for (Size i = 0; i < s.evolution.numberOfSteps(); ++i)
        pseudos.push_back(s.model->pseudoRoot(i));
    emit_mats("pseudo_roots", pseudos);
}

// ----------------------------------------------------------------------------
// RatePseudoRootJacobian + AllElements + Numerical.
// Deterministic single-step getBumps at step `stepIndex`.
// ----------------------------------------------------------------------------

void block_jacobian(const Setup& s) {
    Size n = s.forwards.size();
    Size step = 0;  // first step => aliveIndex 0 => numeraire 0 (money market)
    Size aliveIndex = s.evolution.firstAliveRate()[step];
    Size numeraire = aliveIndex;  // discretely compounding MM account requirement

    const Matrix& pseudoRoot = s.model->pseudoRoot(step);
    emit_mat("jac_pseudo_root", pseudoRoot);

    // deterministic state: oldRates = forwards, newRates a perturbation,
    // gaussians a fixed vector, oneStepDFs from oldRates.
    std::vector<Rate> oldRates = s.forwards;
    std::vector<Rate> newRates(n);
    for (Size j = 0; j < n; ++j)
        newRates[j] = oldRates[j] * (1.0 + 0.01 * (j + 1));
    std::vector<Real> gaussians(s.factors);
    for (Size f = 0; f < s.factors; ++f)
        gaussians[f] = 0.3 - 0.2 * f;  // 0.3, 0.1, -0.1

    // oneStepDFs[0]=1; oneStepDFs[i] = 1/(1+oldRates[i-1]*tau[i-1])
    std::vector<Real> oneStepDFs(n + 1);
    oneStepDFs[0] = 1.0;
    for (Size i = 1; i <= n; ++i)
        oneStepDFs[i] = 1.0 / (1.0 + oldRates[i - 1] * s.evolution.rateTaus()[i - 1]);

    emit_arr("jac_old_rates", oldRates);
    emit_arr("jac_new_rates", newRates);
    emit_arr("jac_gaussians", gaussians);
    emit_arr("jac_one_step_dfs", oneStepDFs);

    // pseudoBumps: a small set of unit-ish bumps (each a full Matrix the size of
    // the pseudo-root). We use 4 bumps each hitting one (rate,factor) element.
    std::vector<Matrix> pseudoBumps;
    struct RF { Size r, f; };
    std::vector<RF> spots = {{0, 0}, {2, 1}, {4, 2}, {5, 0}};
    for (auto& rf : spots) {
        Matrix b(n, s.factors, 0.0);
        b[rf.r][rf.f] = 1.0;
        pseudoBumps.push_back(b);
    }
    emit_int("jac_n_bumps", (long)pseudoBumps.size());

    // analytic
    RatePseudoRootJacobian jac(pseudoRoot, aliveIndex, numeraire,
                               s.evolution.rateTaus(), pseudoBumps,
                               s.displacements);
    Matrix B(pseudoBumps.size(), n);
    jac.getBumps(oldRates, oneStepDFs, newRates, gaussians, B);
    emit_mat("jac_B", B);

    // all-elements
    RatePseudoRootJacobianAllElements jacAll(pseudoRoot, aliveIndex, numeraire,
                                             s.evolution.rateTaus(),
                                             s.displacements);
    std::vector<Matrix> Ball;
    for (Size j = 0; j < n; ++j)
        Ball.emplace_back(n, s.factors, 0.0);
    jacAll.getBumps(oldRates, oneStepDFs, newRates, gaussians, Ball);
    emit_mats("jac_B_all", Ball);

    // numerical (finite difference) — the same B with the Numerical class.
    // Note: it uses `gaussians` not `oneStepDFs`.
    RatePseudoRootJacobianNumerical num(pseudoRoot, aliveIndex, numeraire,
                                        s.evolution.rateTaus(), pseudoBumps,
                                        s.displacements);
    Matrix Bnum(pseudoBumps.size(), n);
    num.getBumps(oldRates, oneStepDFs, newRates, gaussians, Bnum);
    emit_mat("jac_B_numerical", Bnum);
}

// ----------------------------------------------------------------------------
// SwaptionPseudoDerivative.
// ----------------------------------------------------------------------------

void block_swaption(const Setup& s) {
    Size startIndex = 1;
    Size endIndex = 5;
    SwaptionPseudoDerivative deriv(s.model, startIndex, endIndex);

    emit_int("swpt_start", (long)startIndex);
    emit_int("swpt_end", (long)endIndex);
    emit("swpt_variance", deriv.variance());
    emit("swpt_implied_vol", deriv.impliedVolatility());
    emit("swpt_expiry", deriv.expiry());

    std::vector<Matrix> varDerivs, volDerivs;
    for (Size i = 0; i < s.evolution.numberOfSteps(); ++i) {
        varDerivs.push_back(deriv.varianceDerivative(i));
        volDerivs.push_back(deriv.volatilityDerivative(i));
    }
    emit_mats("swpt_variance_derivs", varDerivs);
    emit_mats("swpt_volatility_derivs", volDerivs);
}

// ----------------------------------------------------------------------------
// CapPseudoDerivative.
// ----------------------------------------------------------------------------

void block_cap(const Setup& s) {
    Size startIndex = 1;
    Size endIndex = 5;
    Real strike = 0.04;
    Real firstDF = 0.97;
    CapPseudoDerivative deriv(s.model, strike, startIndex, endIndex, firstDF);

    emit_int("cap_start", (long)startIndex);
    emit_int("cap_end", (long)endIndex);
    emit("cap_strike", strike);
    emit("cap_first_df", firstDF);
    emit("cap_implied_vol", deriv.impliedVolatility());

    std::vector<Matrix> priceDerivs, volDerivs;
    for (Size i = 0; i < s.evolution.numberOfSteps(); ++i) {
        priceDerivs.push_back(deriv.priceDerivative(i));
        volDerivs.push_back(deriv.volatilityDerivative(i));
    }
    emit_mats("cap_price_derivs", priceDerivs);
    emit_mats("cap_volatility_derivs", volDerivs);
}

// ----------------------------------------------------------------------------
// VegaBumpCluster + VegaBumpCollection.
// ----------------------------------------------------------------------------

void block_cluster(const Setup& s) {
    // two clusters: check doesIntersect, isCompatible
    VegaBumpCluster a(0, 2, 1, 3, 0, 1);
    VegaBumpCluster b(1, 3, 2, 4, 0, 1);  // overlaps a in factor 1, rate 2
    VegaBumpCluster c(0, 1, 4, 5, 2, 3);  // disjoint from a

    emit_bool("clus_a_intersect_b", a.doesIntersect(b));
    emit_bool("clus_a_intersect_c", a.doesIntersect(c));
    emit_bool("clus_b_intersect_c", b.doesIntersect(c));
    emit_bool("clus_a_compatible", a.isCompatible(s.model));
    // a cluster whose rateEnd exceeds the rate count -> incompatible
    VegaBumpCluster big(0, 1, 0, 100, 0, 1);
    emit_bool("clus_big_compatible", big.isCompatible(s.model));

    // factor-wise collection
    VegaBumpCollection collFactor(s.model, true);
    emit_int("coll_factor_numberBumps", (long)collFactor.numberBumps());
    emit_bool("coll_factor_isFull", collFactor.isFull());
    emit_bool("coll_factor_isNonOverlapping", collFactor.isNonOverlapping());
    emit_bool("coll_factor_isSensible", collFactor.isSensible());

    // non-factor-wise collection
    VegaBumpCollection collWhole(s.model, false);
    emit_int("coll_whole_numberBumps", (long)collWhole.numberBumps());

    // manually-built collection from a vector of clusters -> exercise the
    // unchecked ctor + isFull / isNonOverlapping on it.
    std::vector<VegaBumpCluster> bumps;
    for (Size st = 0; st < s.evolution.numberOfSteps(); ++st)
        for (Size r = s.evolution.firstAliveRate()[st]; r < s.forwards.size(); ++r)
            bumps.emplace_back(0, s.factors, r, r + 1, st, st + 1);
    VegaBumpCollection collManual(bumps, s.model);
    emit_int("coll_manual_numberBumps", (long)collManual.numberBumps());
    emit_bool("coll_manual_isFull", collManual.isFull());
    emit_bool("coll_manual_isNonOverlapping", collManual.isNonOverlapping());
}

// ----------------------------------------------------------------------------
// VolatilityBumpInstrumentJacobian + OrthogonalizedBumpFinder.
// ----------------------------------------------------------------------------

void block_instrument_jacobian(const Setup& s) {
    // whole-factor collection (one cluster per alive (rate,step))
    VegaBumpCollection bumps(s.model, false);
    emit_int("inst_numberBumps", (long)bumps.numberBumps());

    std::vector<VolatilityBumpInstrumentJacobian::Swaption> swaptions;
    VolatilityBumpInstrumentJacobian::Swaption sw;
    sw.startIndex_ = 1;
    sw.endIndex_ = 5;
    swaptions.push_back(sw);

    std::vector<VolatilityBumpInstrumentJacobian::Cap> caps;
    VolatilityBumpInstrumentJacobian::Cap cp;
    cp.startIndex_ = 1;
    cp.endIndex_ = 5;
    cp.strike_ = 0.04;
    caps.push_back(cp);

    VolatilityBumpInstrumentJacobian jac(bumps, swaptions, caps);

    std::vector<Real> dv0 = jac.derivativesVolatility(0);  // swaption
    std::vector<Real> dv1 = jac.derivativesVolatility(1);  // cap
    emit_arr("inst_deriv_swaption", dv0);
    emit_arr("inst_deriv_cap", dv1);

    std::vector<Real> op0 = jac.onePercentBump(0);
    std::vector<Real> op1 = jac.onePercentBump(1);
    emit_arr("inst_onepct_swaption", op0);
    emit_arr("inst_onepct_cap", op1);

    emit_mat("inst_all_onepct", jac.getAllOnePercentBumps());

    // OrthogonalizedBumpFinder: the orthogonalised bumps. We flatten the result
    // into [step][instrument] -> Matrix(rate,factor).
    OrthogonalizedBumpFinder finder(bumps, swaptions, caps,
                                    /*multiplierCutOff*/ 100.0,
                                    /*tolerance*/ 1e-8);
    std::vector<std::vector<Matrix>> theBumps;
    finder.GetVegaBumps(theBumps);
    emit_int("orth_n_steps", (long)theBumps.size());
    emit_int("orth_n_bumps", (long)(theBumps.empty() ? 0 : theBumps[0].size()));
    // flatten: for each step, for each bump, the matrix
    std::vector<Matrix> flat;
    for (auto& perStep : theBumps)
        for (auto& m : perStep)
            flat.push_back(m);
    emit_mats("orth_bumps", flat);
}

}  // namespace

int main() {
    std::cout << "{\n";
    Setup s = makeSetup();
    block_setup(s);
    block_jacobian(s);
    block_swaption(s);
    block_cap(s);
    block_cluster(s);
    block_instrument_jacobian(s);
    std::cout << "\n}" << std::endl;
    return 0;
}
