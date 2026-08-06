// migration-harness/cpp/probes/v143_models_marketmodels/probe.cpp
//
// Reference values for the ql/models/marketmodels/ coverage tail (v1.43):
//
//   PathwiseVegasAccountingEngine        (pathwiseaccountingengine.hpp:115 + .cpp)
//   PathwiseVegasOuterAccountingEngine   (pathwiseaccountingengine.hpp:196 + .cpp)
//   FlatVolFactory                       (models/flatvol.hpp:66 + flatvol.cpp)
//   rankReducedSqrt / SymmetricSchurDecomposition tie-break
//                                        (math/matrixutilities/pseudosqrt.cpp,
//                                         symmetricschurdecomposition.cpp:116-137)
//
// NO EVALUATION DATE IS SET. The two engines and rankReducedSqrt are pure
// numerics with no date input at all, and the FlatVolFactory block builds its
// FlatForward with an EXPLICIT reference date, so nothing here reads
// Settings::instance().evaluationDate(). The pytest module therefore needs no
// ObservableSettings fixture, and that absence is verified rather than assumed.
//
// What is pinned and why:
//
//   * DETERMINISTIC DRIVING NOISE. Both engines are Monte-Carlo, so the probe
//     replaces the Brownian generator with FixedBrownianGenerator: a
//     hard-coded (path x step*factor) table of Gaussian increments, emitted in
//     the JSON so the Python test reads the SAME numbers rather than
//     transcribing them. That removes the one known obstacle to path-for-path
//     reproduction — pquantlib's InverseCumulativeNormal omits the C++ Halley
//     refinement, so an MT- or Sobol-driven engine would agree only to ~1e-16
//     PER VARIATE, and three exp()-compounded steps of that is not a
//     defensible TIGHT comparison. With fixed variates the whole
//     drift -> diffusion -> cash-flow -> backward-sweep pipeline is exercised
//     at TIGHT with no statistical fudge factor.
//
//   * The pseudo-roots are hard-coded literals fed through PseudoRootFacade
//     (and emitted). The spectral pseudo-root of a covariance matrix is only
//     unique up to an orthogonal rotation within each eigenspace, so deriving
//     A from FlatVol would confound an engine bug with an eigensolver
//     difference. Displacements are NON-ZERO (0.02): a port that dropped the
//     displacement would still reproduce a zero-displacement reference.
//
//   * PER-PATH accumulation as well as the final means. singlePathValues is
//     private, so the engines are run at numberOfPaths = 1, 2 and 5 from
//     FRESH evolvers; the 1-path means ARE path 1's values, and the 2-path
//     means pin path 2. The standard errors are pinned too — they are the only
//     witness to the sum-of-squares accumulation.
//
//   * BOTH deflation arms. MarketModelPathwiseMultiCaplet is undeflated
//     (doDeflation_ = true, so MarketModelPathwiseDiscounter::getFactors runs
//     and its derivatives enter fullDerivatives_);
//     MarketModelPathwiseMultiDeflatedCaplet is already deflated
//     (doDeflation_ = false, the else-branch of the same if).
//
//   * A product whose numberOfProducts() differs from numberOfRates:
//     MarketModelPathwiseMultiDeflatedCap with 2 caps over 3 rates. Both
//     engines index values[] by `i*entriesPerProduct + ...`, so a port that
//     conflated products with rates would still pass on the caplet products.
//
//   * The inner and outer engines on the SAME product, so the Python side can
//     reproduce the C++ test-suite's own cross-check that early bump
//     contraction (PathwiseVegasAccountingEngine) and late bump contraction
//     (PathwiseVegasOuterAccountingEngine::multiplePathValues) agree — and the
//     outer engine's un-contracted multiplePathValuesElementary means, which
//     are the only witness to the elementary-vega index arithmetic
//     `m + l*factors + k*rates*factors`.
//
//   * FlatVolFactory: initialRates and displacements (the displaced-vol
//     formula f*vol/(f+displacement) and the LinearInterpolation lookup), plus
//     each step's covariance A*A^T rather than A itself. A*A^T is invariant
//     under the eigensolver's choice of basis; A is not, and pquantlib uses
//     LAPACK where C++ uses Jacobi. Both a full-rank and a rank-reduced factor
//     count are emitted (the retained eigen-subspace, hence A*A^T, is still
//     uniquely determined when the eigenvalues are distinct).
//
//   * rankReducedSqrt on matrices with TIED eigenvalues, full matrix pinned.
//     SymmetricSchurDecomposition sorts std::pair<Real, std::vector<Real> >
//     with std::greater<>, which is lexicographic: eigenvalue descending and
//     THEN eigenvector descending. Reversing an ascending eigensolver output
//     reproduces the first key only. The 2x2 identity is the minimal witness
//     (C++ returns I; a plain reversal returns the exchange matrix), and
//     A*A^T = I either way — which is exactly why the FULL matrix is pinned.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/models/marketmodels.json.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <ql/math/matrix.hpp>
#include <ql/math/matrixutilities/pseudosqrt.hpp>
#include <ql/models/marketmodels/browniangenerator.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/evolvers/lognormalfwdrateeuler.hpp>
#include <ql/models/marketmodels/marketmodel.hpp>
#include <ql/models/marketmodels/models/flatvol.hpp>
#include <ql/models/marketmodels/models/pseudorootfacade.hpp>
#include <ql/models/marketmodels/pathwiseaccountingengine.hpp>
#include <ql/models/marketmodels/products/pathwise/pathwiseproductcaplet.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/utilities/clone.hpp>

using namespace QuantLib;

namespace {

// Python's json module accepts the non-standard NaN / Infinity literals, which
// is what reference_reader.load() uses. A NaN really does occur here: at
// numberOfPaths == 1 the C++ standard error is sqrt(sumsq/n - mean*mean), zero
// in exact arithmetic, and the QuantLib build contracts `mean*mean` into an
// FMA — so the subtraction lands a fraction of an ulp BELOW zero and sqrt()
// returns NaN. That value is a property of how libQuantLib was compiled, not of
// the algorithm, so the pytest module must not compare it (see the module
// docstring); it is emitted only so the artifact is visible in the reference.
void emit(const std::string& name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": ";
    if (std::isnan(v))
        std::cout << "NaN";
    else if (std::isinf(v))
        std::cout << (v > 0 ? "Infinity" : "-Infinity");
    else
        std::cout << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

void emitVector(const std::string& prefix, const std::vector<Real>& v) {
    for (Size i = 0; i < v.size(); ++i)
        emit(prefix + "_" + std::to_string(i), v[i]);
}

void emitMatrix(const std::string& prefix, const Matrix& m) {
    for (Size i = 0; i < m.rows(); ++i)
        for (Size j = 0; j < m.columns(); ++j)
            emit(prefix + "_" + std::to_string(i) + "_" + std::to_string(j),
                 m[i][j]);
}

// ---------------------------------------------------------------------------
// deterministic Brownian generator
// ---------------------------------------------------------------------------

// kNumberSteps * kFactors variates per path, kMaxPaths paths. Chosen by hand:
// a mix of signs and magnitudes (including a near-zero and a >2 sigma move) so
// no rate stays close to its initial value and every caplet is in the money on
// some paths and out on others.
const Size kFactors = 2;
const Size kNumberRates = 3;
const Size kNumberSteps = 3;
const Size kMaxPaths = 5;

const Real kGaussians[kMaxPaths][kNumberSteps * kFactors] = {
    { 0.35,  -1.20,   0.80,   0.15,  -0.45,   1.05},
    {-0.90,   0.25,  -1.75,   0.60,   1.30,  -0.20},
    { 2.10,   0.05,   0.40,  -0.85,  -1.10,   0.70},
    {-0.30,  -0.65,   1.55,   1.20,   0.20,  -1.40},
    { 0.75,   1.85,  -0.55,  -0.10,   0.95,   0.50},
};

class FixedBrownianGenerator : public BrownianGenerator {
  public:
    FixedBrownianGenerator(Size factors, Size steps)
    : factors_(factors), steps_(steps) {}

    Real nextStep(std::vector<Real>& out) override {
        for (Size f = 0; f < factors_; ++f)
            out[f] = kGaussians[path_ % kMaxPaths][step_ * factors_ + f];
        ++step_;
        return 1.0;
    }
    Real nextPath() override {
        if (started_)
            ++path_;
        else
            started_ = true;
        step_ = 0;
        return 1.0;
    }
    Size numberOfFactors() const override { return factors_; }
    Size numberOfSteps() const override { return steps_; }

  private:
    Size factors_, steps_;
    Size path_ = 0, step_ = 0;
    bool started_ = false;
};

class FixedBrownianGeneratorFactory : public BrownianGeneratorFactory {
  public:
    ext::shared_ptr<BrownianGenerator> create(Size factors,
                                              Size steps) const override {
        return ext::shared_ptr<BrownianGenerator>(
            new FixedBrownianGenerator(factors, steps));
    }
};

// ---------------------------------------------------------------------------
// the market-model world
// ---------------------------------------------------------------------------

std::vector<Time> rateTimes() { return {0.5, 1.0, 1.5, 2.0}; }
std::vector<Real> accruals() { return {0.5, 0.5, 0.5}; }
std::vector<Time> paymentTimes() { return {1.0, 1.5, 2.0}; }
std::vector<Rate> initialRates() { return {0.05, 0.055, 0.06}; }
std::vector<Spread> displacements() { return {0.02, 0.02, 0.02}; }
std::vector<Rate> strikes() { return {0.04, 0.04, 0.04}; }

const Real kStrike = 0.04;
const Real kInitialNumeraireValue = 0.95;

// Hand-picked pseudo-roots: row i is rate i, column f is factor f. Row norms
// grow with the rate index (a plausible term structure of instantaneous vols)
// and the second factor changes sign across rates so the two factors are not
// interchangeable.
std::vector<Matrix> pseudoRoots() {
    std::vector<Matrix> prs;
    const Real data[kNumberSteps][kNumberRates][kFactors] = {
        {{0.11, 0.030}, {0.09, -0.020}, {0.07, 0.015}},
        {{0.00, 0.000}, {0.12, 0.040}, {0.08, -0.025}},
        {{0.00, 0.000}, {0.00, 0.000}, {0.13, 0.050}},
    };
    for (Size s = 0; s < kNumberSteps; ++s) {
        Matrix m(kNumberRates, kFactors);
        for (Size i = 0; i < kNumberRates; ++i)
            for (Size f = 0; f < kFactors; ++f)
                m[i][f] = data[s][i][f];
        prs.push_back(m);
    }
    return prs;
}

ext::shared_ptr<MarketModel> makeModel() {
    return ext::shared_ptr<MarketModel>(new PseudoRootFacade(
        pseudoRoots(), rateTimes(), initialRates(), displacements()));
}

// vegaBumps[step][bump]: exactly the construction the C++ test-suite uses in
// MarketModelTest::testPathwiseVegas, with bumpIncrement = 1 + steps/3 = 2 and
// factorsToTest = min(2, factors) = 2, so the bump index enumerates
// (rate k in {0,2}) x (factor f in {0,1}) x (step m in {0,1,2}) = 12 bumps and
// vegaBumps[l][bump] is non-zero only where l == m and k >= l.
std::vector<std::vector<Matrix> > makeVegaBumps() {
    const Real vegaBumpSize = 1e-2;
    const Size bumpIncrement = 1 + kNumberSteps / 3;
    const Size factorsToTest = kFactors < 2 ? kFactors : 2;

    std::vector<std::vector<Matrix> > vegaBumps;
    Matrix modelBump(kNumberRates, kFactors, 0.0);
    for (Size l = 0; l < kNumberSteps; ++l) {
        vegaBumps.emplace_back();
        for (Size k = 0; k < kNumberRates; k = k + bumpIncrement) {
            for (Size f = 0; f < factorsToTest; ++f) {
                for (Size m = 0; m < kNumberSteps; ++m) {
                    if (l == m && k >= l)
                        modelBump[k][f] = vegaBumpSize;
                    vegaBumps[l].push_back(modelBump);
                    modelBump[k][f] = 0.0;
                }
            }
        }
    }
    return vegaBumps;
}

// ---------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------

void block_world() {
    std::vector<Matrix> prs = pseudoRoots();
    for (Size s = 0; s < prs.size(); ++s)
        emitMatrix("pr_" + std::to_string(s), prs[s]);
    emitVector("init", initialRates());
    emitVector("disp", displacements());
    for (Size p = 0; p < kMaxPaths; ++p)
        for (Size i = 0; i < kNumberSteps * kFactors; ++i)
            emit("gauss_" + std::to_string(p) + "_" + std::to_string(i),
                 kGaussians[p][i]);
    emit("initial_numeraire_value", kInitialNumeraireValue);
    emit("strike", kStrike);

    // the derived bookkeeping every port has to get right
    EvolutionDescription evolution(rateTimes());
    std::vector<Size> numeraires = moneyMarketMeasure(evolution);
    for (Size i = 0; i < numeraires.size(); ++i)
        emit("numeraire_" + std::to_string(i), Real(numeraires[i]));
    const std::vector<Size>& alive = evolution.firstAliveRate();
    for (Size i = 0; i < alive.size(); ++i)
        emit("first_alive_" + std::to_string(i), Real(alive[i]));

    std::vector<std::vector<Matrix> > vb = makeVegaBumps();
    emit("number_bumps", Real(vb[0].size()));
    for (Size l = 0; l < vb.size(); ++l)
        for (Size b = 0; b < vb[l].size(); ++b)
            emitMatrix("vb_" + std::to_string(l) + "_" + std::to_string(b),
                       vb[l][b]);
}

// Run PathwiseVegasAccountingEngine on `product` for `paths` paths from a
// FRESH evolver (hence a fresh FixedBrownianGenerator at path 0) and emit the
// means and errors.
void runInner(const std::string& tag,
              const Clone<MarketModelPathwiseMultiProduct>& product,
              Size paths) {
    FixedBrownianGeneratorFactory factory;
    ext::shared_ptr<MarketModel> model = makeModel();
    EvolutionDescription evolution(rateTimes());
    std::vector<Size> numeraires = moneyMarketMeasure(evolution);

    LogNormalFwdRateEuler evolver(model, factory, numeraires);
    PathwiseVegasAccountingEngine engine(
        ext::make_shared<LogNormalFwdRateEuler>(evolver), product, model,
        makeVegaBumps(), kInitialNumeraireValue);

    std::vector<Real> means, errors;
    engine.multiplePathValues(means, errors, paths);
    emit(tag + "_size", Real(means.size()));
    emitVector(tag + "_mean", means);
    emitVector(tag + "_err", errors);
}

void runOuter(const std::string& tag,
              const Clone<MarketModelPathwiseMultiProduct>& product,
              Size paths) {
    EvolutionDescription evolution(rateTimes());
    std::vector<Size> numeraires = moneyMarketMeasure(evolution);

    {
        FixedBrownianGeneratorFactory factory;
        ext::shared_ptr<MarketModel> model = makeModel();
        LogNormalFwdRateEuler evolver(model, factory, numeraires);
        PathwiseVegasOuterAccountingEngine engine(
            ext::make_shared<LogNormalFwdRateEuler>(evolver), product, model,
            makeVegaBumps(), kInitialNumeraireValue);
        std::vector<Real> means, errors;
        engine.multiplePathValuesElementary(means, errors, paths);
        emit(tag + "_elem_size", Real(means.size()));
        emitVector(tag + "_elem_mean", means);
        emitVector(tag + "_elem_err", errors);
    }
    {
        FixedBrownianGeneratorFactory factory;
        ext::shared_ptr<MarketModel> model = makeModel();
        LogNormalFwdRateEuler evolver(model, factory, numeraires);
        PathwiseVegasOuterAccountingEngine engine(
            ext::make_shared<LogNormalFwdRateEuler>(evolver), product, model,
            makeVegaBumps(), kInitialNumeraireValue);
        std::vector<Real> means, errors;
        engine.multiplePathValues(means, errors, paths);
        emit(tag + "_size", Real(means.size()));
        emitVector(tag + "_mean", means);
        emitVector(tag + "_err", errors);
    }
}

void block_engines() {
    MarketModelPathwiseMultiCaplet caplets(rateTimes(), accruals(),
                                           paymentTimes(), strikes());
    MarketModelPathwiseMultiDeflatedCaplet capletsDeflated(
        rateTimes(), accruals(), paymentTimes(), strikes());
    std::vector<std::pair<Size, Size> > startsAndEnds;
    startsAndEnds.emplace_back(0, 2);
    startsAndEnds.emplace_back(1, 3);
    MarketModelPathwiseMultiDeflatedCap capsDeflated(
        rateTimes(), accruals(), paymentTimes(), kStrike, startsAndEnds);

    // undeflated caplets: doDeflation_ == true. One path, two paths, five.
    runInner("inner_undefl_p1", Clone<MarketModelPathwiseMultiProduct>(caplets), 1);
    runInner("inner_undefl_p2", Clone<MarketModelPathwiseMultiProduct>(caplets), 2);
    runInner("inner_undefl_p5", Clone<MarketModelPathwiseMultiProduct>(caplets), 5);

    // deflated caplets: doDeflation_ == false.
    runInner("inner_defl_p5",
             Clone<MarketModelPathwiseMultiProduct>(capletsDeflated), 5);

    // deflated caps: numberOfProducts (2) != numberOfRates (3).
    runInner("inner_caps_p5",
             Clone<MarketModelPathwiseMultiProduct>(capsDeflated), 5);
    runOuter("outer_caps_p5",
             Clone<MarketModelPathwiseMultiProduct>(capsDeflated), 5);
    runOuter("outer_undefl_p5",
             Clone<MarketModelPathwiseMultiProduct>(caplets), 5);
}

void block_flat_vol_factory() {
    const Date refDate(15, June, 2026);
    const Actual365Fixed dc;
    Handle<YieldTermStructure> curve(
        ext::make_shared<FlatForward>(refDate, 0.04, dc));

    // volatility term structure: LinearInterpolation over (times, vols).
    // rateTimes()[0..2] = 0.5, 1.0, 1.5 all lie strictly inside [0, 2] so the
    // interpolation never extrapolates (C++ would throw if it did).
    std::vector<Time> times = {0.0, 0.5, 1.0, 1.5, 2.0};
    std::vector<Volatility> vols = {0.22, 0.20, 0.18, 0.16, 0.15};

    const Real longTermCorrelation = 0.5;
    const Real beta = 0.2;
    const Spread displacement = 0.01;

    FlatVolFactory factory(longTermCorrelation, beta, times, vols, curve,
                           displacement);
    emit("fvf_ref_date_serial", Real(refDate.serialNumber()));
    emit("fvf_displacement", displacement);
    emit("fvf_long_term_correlation", longTermCorrelation);
    emit("fvf_beta", beta);
    for (Size i = 0; i < times.size(); ++i) {
        emit("fvf_time_" + std::to_string(i), times[i]);
        emit("fvf_vol_" + std::to_string(i), vols[i]);
    }

    EvolutionDescription evolution(rateTimes());
    for (Size factors = 1; factors <= kNumberRates; ++factors) {
        std::string tag = "fvf_f" + std::to_string(factors);
        ext::shared_ptr<MarketModel> mm = factory.create(evolution, factors);
        emit(tag + "_n_rates", Real(mm->numberOfRates()));
        emit(tag + "_n_factors", Real(mm->numberOfFactors()));
        emit(tag + "_n_steps", Real(mm->numberOfSteps()));
        emitVector(tag + "_init", mm->initialRates());
        emitVector(tag + "_disp", mm->displacements());
        // covariance = A * A^T is invariant under the eigensolver's choice of
        // basis; the raw A is not (Jacobi vs LAPACK). Both are emitted: the
        // covariance is what the pytest asserts, the raw pseudo-root is
        // emitted so the residual eigenbasis divergence stays measurable.
        for (Size s = 0; s < mm->numberOfSteps(); ++s) {
            emitMatrix(tag + "_cov_" + std::to_string(s), mm->covariance(s));
            emitMatrix(tag + "_pr_" + std::to_string(s), mm->pseudoRoot(s));
        }
        emitMatrix(tag + "_totcov",
                   mm->totalCovariance(mm->numberOfSteps() - 1));
    }
}

void block_rank_reduced_sqrt() {
    // (a) 2x2 identity: TIED eigenvalues, canonical eigenvectors. C++ returns
    //     the identity because std::greater<> on pair<Real, vector<Real> >
    //     puts (1, {1,0}) before (1, {0,1}). Reversing an ascending
    //     eigensolver's output returns the exchange matrix instead. A*A^T is I
    //     for both, so only the FULL matrix exposes the difference.
    Matrix i2(2, 2, 0.0);
    i2[0][0] = 1.0;
    i2[1][1] = 1.0;
    emitMatrix("rrs_i2_none", rankReducedSqrt(i2, 2, 1.0, SalvagingAlgorithm::None));
    emitMatrix("rrs_i2_spectral",
               rankReducedSqrt(i2, 2, 1.0, SalvagingAlgorithm::Spectral));

    // (b) diagonal with a repeated eigenvalue and a distinct one.
    Matrix d3(3, 3, 0.0);
    d3[0][0] = 2.0;
    d3[1][1] = 2.0;
    d3[2][2] = 1.0;
    emitMatrix("rrs_d3_none", rankReducedSqrt(d3, 3, 1.0, SalvagingAlgorithm::None));

    // (c) repeated eigenvalue with NON-TRIVIAL eigenvectors: the 2x2 block
    //     [[1.5,0.5],[0.5,1.5]] has eigenvalues 2 and 1, so the full matrix has
    //     eigenvalues {2, 2, 1} and the eigenvalue-2 eigenspace is spanned by
    //     e0 and (0,1,1)/sqrt(2) — a genuine tie between two non-parallel
    //     vectors, one of which is not a coordinate axis.
    Matrix b3(3, 3, 0.0);
    b3[0][0] = 2.0;
    b3[1][1] = 1.5;
    b3[1][2] = 0.5;
    b3[2][1] = 0.5;
    b3[2][2] = 1.5;
    emitMatrix("rrs_b3_none", rankReducedSqrt(b3, 3, 1.0, SalvagingAlgorithm::None));

    // (d) generic non-degenerate control: no ties, so the ordering rule is
    //     irrelevant and only the eigen-decomposition itself is under test.
    Matrix g3(3, 3);
    g3[0][0] = 1.0;  g3[0][1] = 0.3;  g3[0][2] = 0.1;
    g3[1][0] = 0.3;  g3[1][1] = 2.0;  g3[1][2] = 0.2;
    g3[2][0] = 0.1;  g3[2][1] = 0.2;  g3[2][2] = 3.0;
    emitMatrix("rrs_g3_none", rankReducedSqrt(g3, 3, 1.0, SalvagingAlgorithm::None));
    emitMatrix("rrs_g3_rank2", rankReducedSqrt(g3, 2, 1.0, SalvagingAlgorithm::None));

    // (e) a semi-definite matrix with a zero eigenvalue, salvaged spectrally.
    Matrix s3(3, 3);
    s3[0][0] = 1.0;  s3[0][1] = 1.0;  s3[0][2] = 0.0;
    s3[1][0] = 1.0;  s3[1][1] = 1.0;  s3[1][2] = 0.0;
    s3[2][0] = 0.0;  s3[2][1] = 0.0;  s3[2][2] = 4.0;
    emitMatrix("rrs_s3_spectral",
               rankReducedSqrt(s3, 3, 1.0, SalvagingAlgorithm::Spectral));
}

}  // namespace

int main() {
    try {
        std::cout << std::setprecision(17);
        std::cout << "{\n";
        block_world();
        block_engines();
        block_flat_vol_factory();
        block_rank_reduced_sqrt();
        // final entry without a trailing comma
        emit("probe_ok", 1.0, false);
        std::cout << "}\n";
        return 0;
    } catch (std::exception& e) {
        std::cerr << "probe failed: " << e.what() << "\n";
        return 1;
    }
}
