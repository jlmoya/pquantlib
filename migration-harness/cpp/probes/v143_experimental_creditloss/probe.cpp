// migration-harness/cpp/probes/v143_experimental_creditloss/probe.cpp
//
// Reference values for the ql/experimental/credit "portfolio loss" family
// @ v1.43:
//
//   distribution.hpp            Distribution, ManipulateDistribution
//   lossdistribution.hpp        LossDist (+ LossDistBinomial / Homogeneous /
//                               Bucketing / MonteCarlo and the three functor
//                               wrappers)
//   basecorrelationstructure.hpp  BaseCorrelationTermStructure<I2D>
//   basecorrelationlossmodel.hpp  BaseCorrelationLossModel<LM, I2D>
//   homogeneouspooldef.hpp      HomogeneousPoolLossModel<CP>
//   inhomogeneouspooldef.hpp    InhomogeneousPoolLossModel<CP>
//   saddlepointlossmodel.hpp    SaddlePointLossModel<CP>,
//                               SaddleObjectiveFunction, SaddlePercObjFunction
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// The saddlepoint model is a three-stage pipeline:
//
//     cumulants K, K', K'', K''', K''''   (closed form, per market factor)
//         -> Brent search for the saddle point s* solving K'(s*) = loss
//             -> a high-order expansion in (s*, K(s*), K''(s*), K'''(s*), ...)
//                 -> integration over the market factor
//
// Pinning only expectedTrancheLoss() would test the composition of four
// stages at once, so a mismatch would be undiagnosable and a compensating
// pair of errors would pass. This probe therefore pins each stage
// separately on a grid: block D pins the cumulants at fixed (saddle, mktFactor)
// with no root search involved at all; block E pins the objective functors
// that the root search consumes; block F pins the located saddle point
// itself; only then do blocks G/H pin the conditional statistics and the
// integrated (unconditional) endpoints.
//
// Blocks A-C are the supporting value types (Distribution, LossDist family,
// base-correlation surface), pinned array-by-array rather than by summary
// statistic for the same reason.
//
// TWO C++ DEFECTS ARE PINNED DELIBERATELY (see the emissions named
// `*_defect_*`), because a port that "fixes" them silently diverges:
//
//   1. LossDistBinomial::volume_ is never assigned by either operator().
//      lossdistribution.hpp:111 declares `mutable Real volume_;` with no
//      initialiser (contrast LossDistHomogeneous, hpp:153-154, which does
//      initialise both members), and lossdistribution.cpp:155 reads it
//      (`if (volume_ * i <= maximum_)`) as the loop guard. Reading it is
//      undefined behaviour and the value is NOT reproducible run to run, so
//      block B pins the guard's deterministic *outcome* rather than the
//      indeterminate float. See the comment on the emission itself.
//
//   2. BaseCorrelationTermStructure::checkInputs is called with
//      (rows, columns) = (nTenors, nLosses) but asserts
//      nLosses_ == volRows && nTrancheTenors_ == volsColumns
//      (basecorrelationstructure.hpp:84 vs :168-175), so a non-square
//      surface always throws; and setupInterpolation feeds correlations_
//      to BilinearInterpolation with x = trancheTimes, y = lossLevel while
//      updateMatrix filled it as correlations_[iTenor][iLoss], so the
//      surface is transposed relative to the documented
//      "correls[iYear][iLoss]" layout. Block C pins both with an
//      asymmetric matrix, which is the only way to see the transposition.

#include <ql/qldefines.hpp>
#include <ql/version.hpp>

#include <ql/currencies/europe.hpp>
#include <ql/experimental/credit/basecorrelationlossmodel.hpp>
#include <ql/experimental/credit/basecorrelationstructure.hpp>
#include <ql/experimental/credit/basket.hpp>
#include <ql/experimental/credit/constantlosslatentmodel.hpp>
#include <ql/experimental/credit/defaultprobabilitykey.hpp>
#include <ql/experimental/credit/distribution.hpp>
#include <ql/experimental/credit/homogeneouspooldef.hpp>
#include <ql/experimental/credit/inhomogeneouspooldef.hpp>
#include <ql/experimental/credit/lossdistribution.hpp>
#include <ql/experimental/credit/pool.hpp>
#include <ql/experimental/credit/saddlepointlossmodel.hpp>
#include <ql/math/interpolations/bilinearinterpolation.hpp>
#include <ql/math/interpolations/bicubicsplineinterpolation.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --------------------------------------------------------------------------
// JSON emission (flat top-level object; the pquantlib harness convention)
// --------------------------------------------------------------------------

bool g_first = true;

void sep() {
    if (!g_first) std::cout << ",\n";
    g_first = false;
}

void emit(const std::string& name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
}

void emit_int(const std::string& name, long long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_bool(const std::string& name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const std::string& name, const std::vector<long long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

// Dump every observable array of a Distribution under a common prefix.
void emit_distribution(const std::string& prefix, Distribution& d) {
    std::vector<Real> xs, dxs, dens, cum, exc, cumExc, avg;
    for (Size i = 0; i < d.size(); ++i) {
        xs.push_back(d.x(i));
        dxs.push_back(d.dx(i));
        dens.push_back(d.density(i));
        cum.push_back(d.cumulative(i));
        exc.push_back(d.excess(i));
        cumExc.push_back(d.cumulativeExcess(i));
        avg.push_back(d.average(i));
    }
    emit_int(prefix + "_size", static_cast<long long>(d.size()));
    emit_arr(prefix + "_x", xs);
    emit_arr(prefix + "_dx", dxs);
    emit_arr(prefix + "_density", dens);
    emit_arr(prefix + "_cumulative", cum);
    emit_arr(prefix + "_excess", exc);
    emit_arr(prefix + "_cumulative_excess", cumExc);
    emit_arr(prefix + "_average", avg);
}

// --------------------------------------------------------------------------
// Access shims. The interesting members of the loss models are protected;
// deriving is the only way to pin the intermediate stages.
// --------------------------------------------------------------------------

class ExposedSaddle : public SaddlePointLossModel<GaussianCopulaPolicy> {
    typedef SaddlePointLossModel<GaussianCopulaPolicy> Base;

  public:
    explicit ExposedSaddle(const ext::shared_ptr<GaussianConstantLossLM>& m)
    : Base(m) {}

    Real cgf(const std::vector<Real>& p, Real s, const std::vector<Real>& m) const {
        return Base::CumulantGeneratingCond(p, s, m);
    }
    Real d1(const std::vector<Real>& p, Real s, const std::vector<Real>& m) const {
        return Base::CumGen1stDerivativeCond(p, s, m);
    }
    Real d2(const std::vector<Real>& p, Real s, const std::vector<Real>& m) const {
        return Base::CumGen2ndDerivativeCond(p, s, m);
    }
    Real d3(const std::vector<Real>& p, Real s, const std::vector<Real>& m) const {
        return Base::CumGen3rdDerivativeCond(p, s, m);
    }
    Real d4(const std::vector<Real>& p, Real s, const std::vector<Real>& m) const {
        return Base::CumGen4thDerivativeCond(p, s, m);
    }
    std::tuple<Real, Real, Real, Real>
    d0234(const std::vector<Real>& p, Real s, const std::vector<Real>& m) const {
        return Base::CumGen0234DerivCond(p, s, m);
    }
    std::tuple<Real, Real>
    d02(const std::vector<Real>& p, Real s, const std::vector<Real>& m) const {
        return Base::CumGen02DerivCond(p, s, m);
    }
    Real cgfUncond(const Date& d, Real s) const { return Base::CumulantGenerating(d, s); }
    Real d1Uncond(const Date& d, Real s) const { return Base::CumGen1stDerivative(d, s); }
    Real d2Uncond(const Date& d, Real s) const { return Base::CumGen2ndDerivative(d, s); }
    Real d3Uncond(const Date& d, Real s) const { return Base::CumGen3rdDerivative(d, s); }
    Real d4Uncond(const Date& d, Real s) const { return Base::CumGen4thDerivative(d, s); }

    Real saddle(const std::vector<Real>& p, Real loss, const std::vector<Real>& m) const {
        return Base::findSaddle(p, loss, m);
    }
    Real saddleTuned(const std::vector<Real>& p, Real loss, const std::vector<Real>& m,
                     Real acc, Natural maxEv) const {
        return Base::findSaddle(p, loss, m, acc, maxEv);
    }

    Real objective(const std::vector<Real>& p, Real target, const std::vector<Real>& m,
                   Real x) const {
        Base::SaddleObjectiveFunction f(*this, target, p, m);
        return f(x);
    }
    Real objectiveDeriv(const std::vector<Real>& p, Real target, const std::vector<Real>& m,
                        Real x) const {
        Base::SaddleObjectiveFunction f(*this, target, p, m);
        return f.derivative(x);
    }
    Real percObjective(Real target, const Date& d, Real x) const {
        Base::SaddlePercObjFunction f(*this, target, d);
        return f(x);
    }

    Probability pOverLossCond(const std::vector<Real>& p, Real trancheFract,
                              const std::vector<Real>& m) const {
        return Base::probOverLossCond(p, trancheFract, m);
    }
    Probability pOverPortfCond(const std::vector<Real>& p, Real loss,
                               const std::vector<Real>& m) const {
        return Base::probOverLossPortfCond(p, loss, m);
    }
    Probability pOverPortfCond1st(const std::vector<Real>& p, Real loss,
                                  const std::vector<Real>& m) const {
        return Base::probOverLossPortfCond1stOrder(p, loss, m);
    }
    Probability pDensityCond(const std::vector<Real>& p, Real loss,
                             const std::vector<Real>& m) const {
        return Base::probDensityCond(p, loss, m);
    }
    std::vector<Real> splitCond(const std::vector<Real>& p, Real loss,
                                std::vector<Real> m) const {
        return Base::splitLossCond(p, loss, std::move(m));
    }
    Real condEL(const std::vector<Real>& p, const std::vector<Real>& m) const {
        return Base::conditionalExpectedLoss(p, m);
    }
    Real condETL(const std::vector<Real>& p, const std::vector<Real>& m) const {
        return Base::conditionalExpectedTrancheLoss(p, m);
    }
    Real esfFullCond(const std::vector<Real>& p, Real lossPerc,
                     const std::vector<Real>& m) const {
        return Base::expectedShortfallFullPortfolioCond(p, lossPerc, m);
    }
    Real esfTrancheCond(const std::vector<Real>& p, Real lossPerc, Probability perc,
                        const std::vector<Real>& m) const {
        return Base::expectedShortfallTrancheCond(p, lossPerc, perc, m);
    }
    std::vector<Real> esfSplitCond(const std::vector<Real>& p, Real lossPerc,
                                   const std::vector<Real>& m) const {
        return Base::expectedShortfallSplitCond(p, lossPerc, m);
    }
};

class ExposedHomog : public HomogeneousPoolLossModel<GaussianCopulaPolicy> {
    typedef HomogeneousPoolLossModel<GaussianCopulaPolicy> Base;

  public:
    ExposedHomog(const ext::shared_ptr<GaussianConstantLossLM>& c, Size nBuckets,
                 Real max = 5., Real min = -5., Size nSteps = 50)
    : Base(c, nBuckets, max, min, nSteps) {}
    Distribution dist(const Date& d) const { return Base::lossDistrib(d); }
};

class ExposedInhomog : public InhomogeneousPoolLossModel<GaussianCopulaPolicy> {
    typedef InhomogeneousPoolLossModel<GaussianCopulaPolicy> Base;

  public:
    ExposedInhomog(const ext::shared_ptr<GaussianConstantLossLM>& c, Size nBuckets,
                   Real max = 5., Real min = -5., Size nSteps = 50)
    : Base(c, nBuckets, max, min, nSteps) {}
    Distribution dist(const Date& d) const { return Base::lossDistrib(d); }
};

// --------------------------------------------------------------------------
// Block A: Distribution + ManipulateDistribution
// --------------------------------------------------------------------------

void blockA() {
    // A1 - grid construction. Note the last bucket width is snapped to xmax.
    Distribution d(8, 0.0, 4.0);
    emit_distribution("distA_fresh", d);

    // A2 - locate() on cell boundaries, interiors and the two endpoints.
    std::vector<Real> locXs = {0.0, 0.1, 0.5, 0.5000000001, 1.0, 2.75, 3.999, 4.0};
    std::vector<long long> locs;
    for (Real x : locXs) locs.push_back(d.locate(x));
    emit_arr("distA_locate_x", locXs);
    emit_iarr("distA_locate", locs);
    std::vector<Real> dxAt;
    for (Real x : locXs) dxAt.push_back(d.dx(x));
    emit_arr("distA_dx_at", dxAt);

    // A3 - add() sampling: counts, underflow and overflow.
    Distribution s(8, 0.0, 4.0);
    std::vector<Real> samples = {-0.5, 0.05, 0.3, 0.7, 1.2, 1.9, 2.1, 2.4,
                                 3.3,  3.7,  3.9, 4.5, 0.6, 2.2, 2.25};
    for (Real v : samples) s.add(v);
    emit_arr("distA_samples", samples);
    emit_distribution("distA_sampled", s);
    emit("distA_sampled_expected_value", s.expectedValue());
    emit("distA_sampled_tranche_ev_1_3", s.trancheExpectedValue(1.0, 3.0));
    emit("distA_sampled_conf_50", s.confidenceLevel(0.5));
    emit("distA_sampled_conf_90", s.confidenceLevel(0.9));
    emit("distA_sampled_cum_density_1_5", s.cumulativeDensity(1.5));
    emit("distA_sampled_cum_density_3_1", s.cumulativeDensity(3.1));
    emit("distA_sampled_cum_excess_0_5_3_0", s.cumulativeExcessProbability(0.5, 3.0));
    emit("distA_sampled_esf_50", s.expectedShortfall(0.5));
    emit("distA_sampled_esf_90", s.expectedShortfall(0.9));

    // A4 - addDensity/addAverage path.
    Distribution ad(5, 0.0, 5.0);
    for (Size i = 0; i < 5; ++i) {
        ad.addDensity(static_cast<int>(i), 0.05 * static_cast<Real>(i + 1));
        ad.addAverage(static_cast<int>(i), 0.5 + static_cast<Real>(i));
    }
    emit_distribution("distA_density_built", ad);
    emit("distA_density_built_expected_value", ad.expectedValue());

    // A5 - ManipulateDistribution::convolve. Both inputs must have constant
    //      bucket width and xmin == 0.
    Distribution c1(4, 0.0, 4.0);
    Distribution c2(3, 0.0, 3.0);
    for (Size i = 0; i < 4; ++i) c1.addDensity(static_cast<int>(i), 0.1 * static_cast<Real>(4 - i));
    for (Size i = 0; i < 3; ++i) c2.addDensity(static_cast<int>(i), 0.2 * static_cast<Real>(i + 1));
    c1.normalize();
    c2.normalize();
    Distribution conv = ManipulateDistribution::convolve(c1, c2);
    emit_distribution("distA_convolve", conv);

    // A6 - tranche(). Destructive; run on a private copy.
    Distribution tr(8, 0.0, 4.0);
    for (Real v : samples) tr.add(v);
    tr.normalize();
    tr.tranche(1.0, 3.0);
    emit_int("distA_tranche_size", static_cast<long long>(tr.size()));
    std::vector<Real> txs, tdxs, tdens, tcum, texc;
    for (Size i = 0; i < tr.size(); ++i) {
        txs.push_back(tr.x(i));
        tdxs.push_back(tr.dx(i));
        tdens.push_back(tr.density(i));
        tcum.push_back(tr.cumulative(i));
        texc.push_back(tr.excess(i));
    }
    emit_arr("distA_tranche_x", txs);
    emit_arr("distA_tranche_dx", tdxs);
    emit_arr("distA_tranche_density", tdens);
    emit_arr("distA_tranche_cumulative", tcum);
    emit_arr("distA_tranche_excess", texc);
    emit("distA_tranche_esf_50", tr.expectedShortfall(0.5));
}

// --------------------------------------------------------------------------
// Block B: LossDist statics + the four concrete loss distributions
// --------------------------------------------------------------------------

void blockB() {
    std::vector<Real> p = {0.02, 0.05, 0.10, 0.17, 0.30};
    emit_arr("lossdist_p", p);

    std::vector<Real> pne = LossDist::probabilityOfNEvents(p);
    emit_arr("lossdist_prob_n_events_vec", pne);

    std::vector<Real> pnScalar, pAtLeast, binPn, binAtLeast;
    for (int k = 0; k <= 5; ++k) {
        pnScalar.push_back(LossDist::probabilityOfNEvents(k, p));
        pAtLeast.push_back(LossDist::probabilityOfAtLeastNEvents(k, p));
        binPn.push_back(LossDist::binomialProbabilityOfNEvents(k, p));
        binAtLeast.push_back(LossDist::binomialProbabilityOfAtLeastNEvents(k, p));
    }
    emit_arr("lossdist_prob_n_events", pnScalar);
    emit_arr("lossdist_prob_at_least_n_events", pAtLeast);
    emit_arr("lossdist_binom_prob_n_events", binPn);
    emit_arr("lossdist_binom_prob_at_least_n_events", binAtLeast);

    // The three functor wrappers must agree with the statics they forward to.
    std::vector<Real> fN, fAtLeast, fBinAtLeast;
    for (int k = 0; k <= 5; ++k) {
        fN.push_back(ProbabilityOfNEvents(k)(p));
        fAtLeast.push_back(ProbabilityOfAtLeastNEvents(k)(p));
        fBinAtLeast.push_back(BinomialProbabilityOfAtLeastNEvents(k)(p));
    }
    emit_arr("lossdist_functor_n_events", fN);
    emit_arr("lossdist_functor_at_least_n_events", fAtLeast);
    emit_arr("lossdist_functor_binom_at_least_n_events", fBinAtLeast);

    const Size nBuckets = 10;
    // maximum == n * volume, so the (defect 1) loop guard `volume_ * i <= maximum_`
    // cannot bind for any i whatever `volume_` happens to hold.
    const Real volume = 20.0;
    const Real maximum = 100.0;

    LossDistBinomial ldb(nBuckets, maximum);
    /* Defect 1. `mutable Real volume_;` (lossdistribution.hpp:111) has no
       initialiser and neither operator() ever assigns it, yet
       lossdistribution.cpp:155 reads it as the loop guard
       `if (volume_ * i <= maximum_)`. The value is therefore indeterminate:
       measured over 25 runs of this probe on arm64/libc++ it is a different
       denormal near 2.14e-314 every time, so the FLOAT ITSELF IS NOT
       PINNABLE and must never be asserted on. What IS deterministic, and
       what the port has to reproduce, is the guard's *outcome*: the
       indeterminate value is a denormal, `maximum_` is 100, so
       `volume_ * i <= maximum_` holds for every i and no term is skipped.
       A Python port with volume_ = 0.0 lands in exactly that regime. */
    Real rawBefore = ldb.volume();
    std::vector<Real> volumes(5, volume);
    Distribution db = ldb(volumes, p);
    Real rawAfter = ldb.volume();
    emit_bool("lossdist_binomial_defect_volume_assigned_by_call", rawBefore != rawAfter);
    bool guardBinds = false;
    for (Size i = 0; i <= volumes.size(); ++i)
        if (!(rawAfter * static_cast<Real>(i) <= maximum)) guardBinds = true;
    emit_bool("lossdist_binomial_defect_guard_binds", guardBinds);
    emit_int("lossdist_binomial_n", static_cast<long long>(ldb.size()));
    emit_arr("lossdist_binomial_probability", ldb.probability());
    emit_arr("lossdist_binomial_excess_probability", ldb.excessProbability());
    emit_distribution("lossdist_binomial_dist", db);

    LossDistHomogeneous ldh(nBuckets, maximum);
    Distribution dh = ldh(volumes, p);
    emit_int("lossdist_homog_n", static_cast<long long>(ldh.size()));
    emit("lossdist_homog_volume", ldh.volume());
    emit_arr("lossdist_homog_probability", ldh.probability());
    emit_arr("lossdist_homog_excess_probability", ldh.excessProbability());
    emit_distribution("lossdist_homog_dist", dh);

    // Bucketing handles heterogeneous notionals.
    std::vector<Real> hetVolumes = {10.0, 20.0, 25.0, 30.0, 15.0};
    emit_arr("lossdist_het_volumes", hetVolumes);
    LossDistBucketing ldbk(nBuckets, maximum);
    Distribution dbk = ldbk(hetVolumes, p);
    emit_distribution("lossdist_bucketing_dist", dbk);
    LossDistBucketing ldbk2(nBuckets, maximum, 1e-9);
    Distribution dbk2 = ldbk2(hetVolumes, p);
    emit_distribution("lossdist_bucketing_eps9_dist", dbk2);

    // Monte Carlo: the C++ stream is a MersenneTwisterUniformRng, so it is
    // reproducible and pinned exactly rather than statistically.
    LossDistMonteCarlo ldmc(nBuckets, maximum, 2000, 42);
    Distribution dmc = ldmc(hetVolumes, p);
    emit_distribution("lossdist_montecarlo_dist", dmc);
    LossDistMonteCarlo ldmc2(nBuckets, maximum, 500, 17);
    Distribution dmc2 = ldmc2(hetVolumes, p);
    emit_distribution("lossdist_montecarlo_s17_dist", dmc2);

    emit_int("lossdist_buckets", static_cast<long long>(ldbk.buckets()));
    emit("lossdist_maximum", ldbk.maximum());
}

// --------------------------------------------------------------------------
// Block C: BaseCorrelationTermStructure
// --------------------------------------------------------------------------

void blockC(const Date& today) {
    Calendar cal = TARGET();
    DayCounter dc = Actual365Fixed();

    std::vector<Period> tenors = {Period(1, Years), Period(3, Years), Period(5, Years)};
    std::vector<Real> lossLevels = {0.03, 0.07, 0.15};

    // Deliberately ASYMMETRIC: a symmetric matrix would hide the
    // row/column transposition (defect 2).
    Real raw[3][3] = {{0.10, 0.20, 0.30}, {0.40, 0.50, 0.60}, {0.70, 0.80, 0.90}};
    std::vector<std::vector<Handle<Quote> > > correls;
    for (Size i = 0; i < 3; ++i) {
        std::vector<Handle<Quote> > row;
        for (Size j = 0; j < 3; ++j)
            row.emplace_back(ext::make_shared<SimpleQuote>(raw[i][j]));
        correls.push_back(row);
    }
    std::vector<Real> rawFlat;
    for (auto& i : raw)
        for (Real j : i) rawFlat.push_back(j);
    emit_arr("bcts_quotes_row_major", rawFlat);

    BaseCorrelationTermStructure<BilinearInterpolation> bcts(0, cal, Following, tenors,
                                                             lossLevels, correls, dc);

    emit_int("bcts_correlation_size", static_cast<long long>(bcts.correlationSize()));
    emit_int("bcts_max_date_serial", static_cast<long long>(bcts.maxDate().serialNumber()));
    emit_int("bcts_reference_date_serial",
             static_cast<long long>(bcts.referenceDate().serialNumber()));

    // The node times the surface actually interpolates on.
    std::vector<Real> nodeTimes;
    for (const Period& p : tenors)
        nodeTimes.push_back(bcts.timeFromReference(cal.advance(bcts.referenceDate(), p, Following)));
    emit_arr("bcts_tranche_times", nodeTimes);
    emit_arr("bcts_loss_levels", lossLevels);

    // On-node: every (tenor, loss) pair.
    std::vector<Real> onNode;
    for (Real t : nodeTimes)
        for (Real l : lossLevels) onNode.push_back(bcts.correlation(t, l));
    emit_arr("bcts_on_node", onNode);

    // Off-node interior + both extrapolation regimes (the C++ always passes
    // extrapolate=true to the interpolator).
    std::vector<Real> offT = {0.5, 1.5, 2.0, 4.0, 6.5, 9.0};
    std::vector<Real> offL = {0.01, 0.05, 0.10, 0.20, 0.40};
    emit_arr("bcts_off_t", offT);
    emit_arr("bcts_off_l", offL);
    std::vector<Real> offNode;
    for (Real t : offT)
        for (Real l : offL) offNode.push_back(bcts.correlation(t, l));
    emit_arr("bcts_off_node", offNode);

    // Date-keyed overload must agree with the time-keyed one.
    Date d3 = cal.advance(bcts.referenceDate(), Period(2, Years), Following);
    emit("bcts_by_date_2y_l007", bcts.correlation(d3, 0.07));
    emit("bcts_by_time_2y_l007", bcts.correlation(bcts.timeFromReference(d3), 0.07));

    // Quote propagation: bump one quote and re-read.
    ext::shared_ptr<SimpleQuote> q =
        ext::dynamic_pointer_cast<SimpleQuote>(correls[0][2].currentLink());
    q->setValue(0.99);
    std::vector<Real> bumped;
    for (Real t : nodeTimes)
        for (Real l : lossLevels) bumped.push_back(bcts.correlation(t, l));
    emit_arr("bcts_on_node_after_bump_00_02_to_099", bumped);
    q->setValue(0.30);

    // Bicubic spline flavour on the same data.
    std::vector<std::vector<Handle<Quote> > > correlsB;
    for (Size i = 0; i < 3; ++i) {
        std::vector<Handle<Quote> > row;
        for (Size j = 0; j < 3; ++j)
            row.emplace_back(ext::make_shared<SimpleQuote>(raw[i][j]));
        correlsB.push_back(row);
    }
    BaseCorrelationTermStructure<BicubicSpline> bctsC(0, cal, Following, tenors, lossLevels,
                                                      correlsB, dc);
    std::vector<Real> bicubicOnNode, bicubicOffNode;
    for (Real t : nodeTimes)
        for (Real l : lossLevels) bicubicOnNode.push_back(bctsC.correlation(t, l));
    for (Real t : offT)
        for (Real l : offL) bicubicOffNode.push_back(bctsC.correlation(t, l));
    emit_arr("bcts_bicubic_on_node", bicubicOnNode);
    emit_arr("bcts_bicubic_off_node", bicubicOffNode);

    // Defect 2a: checkInputs compares transposed dimensions, so any
    // non-square surface throws even though it is perfectly well formed.
    bool nonSquareThrew = false;
    std::string nonSquareMsg;
    try {
        std::vector<Period> t2 = {Period(1, Years), Period(3, Years)};
        std::vector<std::vector<Handle<Quote> > > c2;
        for (Size i = 0; i < 2; ++i) {
            std::vector<Handle<Quote> > row;
            for (Size j = 0; j < 3; ++j)
                row.emplace_back(ext::make_shared<SimpleQuote>(0.3));
            c2.push_back(row);
        }
        BaseCorrelationTermStructure<BilinearInterpolation> bad(0, cal, Following, t2,
                                                                lossLevels, c2, dc);
        (void)bad.correlation(1.0, 0.05);
    } catch (const std::exception&) {
        nonSquareThrew = true;
    }
    emit_bool("bcts_defect_non_square_throws", nonSquareThrew);
}

// --------------------------------------------------------------------------
// Basket fixture shared by blocks D..I
// --------------------------------------------------------------------------

struct Fixture {
    ext::shared_ptr<Basket> basket;
    ext::shared_ptr<GaussianConstantLossLM> lm;
    std::vector<Real> hazardRates;
    std::vector<Real> notionals;
    std::vector<Real> recoveries;
    Date calcDate;
};

/* Basket::setLossModel() only invalidates the LazyObject
   (basket.cpp:74-86); the model's resetModel() is reached from
   Basket::performCalculations() (basket.cpp:88-104) via
   DefaultLossModel::setBasket(), which runs on the next calculate().
   Calling a protected model member (lossDistrib, expectedTrancheLoss, ...)
   directly, without first triggering that lazy pass, leaves
   HomogeneousPoolLossModel::notionals_ empty -- and
   `std::transform(lgd.begin(), lgd.end(), notionals_.begin(), ...)`
   (homogeneouspooldef.hpp:133-134) then dereferences the null begin() of an
   empty vector => SIGSEGV. remainingTrancheNotional() (basket.hpp:206-209)
   calls calculate() and is the cheapest way to force the hand-off. */
void forceModelReset(const Fixture& f) { (void)f.basket->remainingTrancheNotional(); }

Fixture makeFixture(const Date& today) {
    Fixture f;
    f.hazardRates = {0.005, 0.010, 0.020, 0.035, 0.060};
    f.notionals = std::vector<Real>(5, 100.0);
    f.recoveries = std::vector<Real>(5, 0.4);

    std::vector<std::string> names;
    for (Size i = 0; i < f.hazardRates.size(); ++i)
        names.push_back(std::string("Name") + std::to_string(i));

    std::vector<Handle<DefaultProbabilityTermStructure> > defTS;
    for (Real h : f.hazardRates) {
        defTS.emplace_back(
            ext::make_shared<FlatHazardRate>(0, TARGET(), h, Actual365Fixed()));
        defTS.back()->enableExtrapolation();
    }

    DefaultProbKey key = NorthAmericaCorpDefaultKey(EURCurrency(), SeniorSec, Period(), 1.0);

    auto pool = ext::make_shared<Pool>();
    for (Size i = 0; i < f.hazardRates.size(); ++i) {
        std::vector<Issuer::key_curve_pair> curves(1, std::make_pair(key, defTS[i]));
        pool->add(names[i], Issuer(curves), key);
    }

    f.basket = ext::make_shared<Basket>(today, names, f.notionals, pool, 0.03, 0.06);

    Real factorValue = 0.25;  // squared loading; a_i = 0.5
    std::vector<std::vector<Real> > weights(f.hazardRates.size(),
                                            std::vector<Real>(1, std::sqrt(factorValue)));
    f.lm = ext::make_shared<GaussianConstantLossLM>(
        weights, f.recoveries, LatentModelIntegrationType::GaussianQuadrature,
        GaussianCopulaPolicy::initTraits());

    f.calcDate = TARGET().advance(today, Period(60, Months));
    return f;
}

// --------------------------------------------------------------------------
// Blocks D-H: SaddlePointLossModel
// --------------------------------------------------------------------------

void saddleBlocks(const Fixture& f) {
    auto model = ext::make_shared<ExposedSaddle>(f.lm);
    f.basket->setLossModel(model);
    forceModelReset(f);

    const Date& d = f.calcDate;

    emit_int("saddle_calc_date_serial", static_cast<long long>(d.serialNumber()));
    emit_arr("saddle_hazard_rates", f.hazardRates);
    emit_arr("saddle_notionals", f.notionals);
    emit_arr("saddle_recoveries", f.recoveries);
    emit("saddle_basket_notional", f.basket->basketNotional());
    emit("saddle_remaining_notional", f.basket->remainingNotional());
    emit("saddle_attach_amount", f.basket->attachmentAmount());
    emit("saddle_detach_amount", f.basket->detachmentAmount());
    emit("saddle_tranche_notional", f.basket->trancheNotional());
    emit("saddle_remaining_tranche_notional", f.basket->remainingTrancheNotional());

    // The unconditional default probabilities and their copula inversion.
    // Everything downstream is a function of THESE numbers, so a port that
    // reproduces the cumulants but not these is still wrong.
    std::vector<Real> uncond = f.basket->remainingProbabilities(d);
    emit_arr("saddle_uncond_probs", uncond);
    std::vector<Real> invUncond = uncond;
    for (Size i = 0; i < invUncond.size(); ++i)
        invUncond[i] = f.lm->inverseCumulativeY(invUncond[i], i);
    emit_arr("saddle_inv_uncond_probs", invUncond);

    // ---- Block D: cumulants at fixed (saddle, market factor). No root
    //      search is involved, so a mismatch here localises to the closed
    //      form itself.
    std::vector<Real> saddleGrid = {-8.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 8.0, 20.0};
    std::vector<Real> mktGrid = {-2.5, -1.0, 0.0, 1.0, 2.5};
    emit_arr("saddle_grid", saddleGrid);
    emit_arr("saddle_mkt_grid", mktGrid);

    std::vector<Real> k0, k1, k2, k3, k4;
    std::vector<Real> t0, t2, t3, t4, u0, u2;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real s : saddleGrid) {
            k0.push_back(model->cgf(invUncond, s, mkt));
            k1.push_back(model->d1(invUncond, s, mkt));
            k2.push_back(model->d2(invUncond, s, mkt));
            k3.push_back(model->d3(invUncond, s, mkt));
            k4.push_back(model->d4(invUncond, s, mkt));
            auto tup = model->d0234(invUncond, s, mkt);
            t0.push_back(std::get<0>(tup));
            t2.push_back(std::get<1>(tup));
            t3.push_back(std::get<2>(tup));
            t4.push_back(std::get<3>(tup));
            auto tup2 = model->d02(invUncond, s, mkt);
            u0.push_back(std::get<0>(tup2));
            u2.push_back(std::get<1>(tup2));
        }
    }
    emit_arr("saddle_cgf_cond", k0);
    emit_arr("saddle_cgf1_cond", k1);
    emit_arr("saddle_cgf2_cond", k2);
    emit_arr("saddle_cgf3_cond", k3);
    emit_arr("saddle_cgf4_cond", k4);
    emit_arr("saddle_cgf0234_d0", t0);
    emit_arr("saddle_cgf0234_d2", t2);
    emit_arr("saddle_cgf0234_d3", t3);
    emit_arr("saddle_cgf0234_d4", t4);
    emit_arr("saddle_cgf02_d0", u0);
    emit_arr("saddle_cgf02_d2", u2);

    // ---- Block E: the objective functors the root search consumes.
    std::vector<Real> targets = {0.01, 0.05, 0.20};
    emit_arr("saddle_obj_targets", targets);
    std::vector<Real> objVals, objDerivs;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real tgt : targets)
            for (Real x : saddleGrid) {
                objVals.push_back(model->objective(invUncond, tgt, mkt, x));
                objDerivs.push_back(model->objectiveDeriv(invUncond, tgt, mkt, x));
            }
    }
    emit_arr("saddle_obj_value", objVals);
    emit_arr("saddle_obj_derivative", objDerivs);

    std::vector<Real> percTargets = {0.90, 0.95, 0.99};
    std::vector<Real> percXs = {0.05, 0.25, 0.5, 0.75, 0.95};
    emit_arr("saddle_perc_obj_targets", percTargets);
    emit_arr("saddle_perc_obj_xs", percXs);
    std::vector<Real> percObj;
    for (Real tgt : percTargets)
        for (Real x : percXs) percObj.push_back(model->percObjective(tgt, d, x));
    emit_arr("saddle_perc_obj_value", percObj);

    // ---- Block F: the located saddle point itself, including the two
    //      clamped regimes (below minLoss / above maxLoss) where findSaddle
    //      returns its bracket endpoint instead of solving.
    std::vector<Real> lossFractions = {1e-8, 1e-4, 0.001, 0.005, 0.01, 0.02,
                                       0.05, 0.10, 0.30,  0.55,  0.75};
    emit_arr("saddle_loss_fractions", lossFractions);
    std::vector<Real> saddles;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real l : lossFractions) saddles.push_back(model->saddle(invUncond, l, mkt));
    }
    emit_arr("saddle_found", saddles);

    // Same points, tighter Brent accuracy: shows how much of the answer is
    // the 1e-3 default tolerance rather than the model.
    std::vector<Real> saddlesTight;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real l : lossFractions)
            saddlesTight.push_back(model->saddleTuned(invUncond, l, mkt, 1e-12, 200));
    }
    emit_arr("saddle_found_acc1e12", saddlesTight);

    // ---- Block G: conditional statistics built on the saddle point.
    std::vector<Real> absLosses = {0.0, 1.0, 5.0, 15.0, 30.0, 60.0, 150.0, 280.0};
    emit_arr("saddle_abs_losses", absLosses);
    std::vector<Real> povPortf, povPortf1st, pdens, esfFull;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real l : absLosses) {
            povPortf.push_back(model->pOverPortfCond(invUncond, l, mkt));
            povPortf1st.push_back(model->pOverPortfCond1st(invUncond, l, mkt));
            pdens.push_back(model->pDensityCond(invUncond, l, mkt));
            esfFull.push_back(model->esfFullCond(invUncond, l, mkt));
        }
    }
    emit_arr("saddle_prob_over_loss_portf_cond", povPortf);
    emit_arr("saddle_prob_over_loss_portf_cond_1st", povPortf1st);
    emit_arr("saddle_prob_density_cond", pdens);
    emit_arr("saddle_esf_full_portfolio_cond", esfFull);

    std::vector<Real> trancheFracts = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0};
    emit_arr("saddle_tranche_fractions", trancheFracts);
    std::vector<Real> povTranche;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real l : trancheFracts) povTranche.push_back(model->pOverLossCond(invUncond, l, mkt));
    }
    emit_arr("saddle_prob_over_loss_cond", povTranche);

    std::vector<Real> condELs, condETLs;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        condELs.push_back(model->condEL(invUncond, mkt));
        condETLs.push_back(model->condETL(invUncond, mkt));
    }
    emit_arr("saddle_conditional_expected_loss", condELs);
    emit_arr("saddle_conditional_expected_tranche_loss", condETLs);

    std::vector<Real> splitFlat, esfSplitFlat;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real l : {5.0, 30.0, 150.0}) {
            std::vector<Real> sp = model->splitCond(invUncond, l, mkt);
            for (Real v : sp) splitFlat.push_back(v);
            std::vector<Real> es = model->esfSplitCond(invUncond, l, mkt);
            for (Real v : es) esfSplitFlat.push_back(v);
        }
    }
    emit_arr("saddle_split_loss_cond", splitFlat);
    emit_arr("saddle_esf_split_cond", esfSplitFlat);

    std::vector<Real> esfTranche;
    for (Real m : mktGrid) {
        std::vector<Real> mkt(1, m);
        for (Real l : {5.0, 30.0, 150.0})
            esfTranche.push_back(model->esfTrancheCond(invUncond, l, 0.95, mkt));
    }
    emit_arr("saddle_esf_tranche_cond", esfTranche);

    // ---- Block H: integrated (unconditional) quantities. These add the
    //      Gauss-Hermite order-25 quadrature over the market factor on top
    //      of everything above.
    std::vector<Real> uncondS = {-3.0, -1.0, 0.0, 1.0, 3.0};
    emit_arr("saddle_uncond_s_grid", uncondS);
    std::vector<Real> uk0, uk1, uk2, uk3, uk4;
    for (Real s : uncondS) {
        uk0.push_back(model->cgfUncond(d, s));
        uk1.push_back(model->d1Uncond(d, s));
        uk2.push_back(model->d2Uncond(d, s));
        uk3.push_back(model->d3Uncond(d, s));
        uk4.push_back(model->d4Uncond(d, s));
    }
    emit_arr("saddle_cgf_uncond", uk0);
    emit_arr("saddle_cgf1_uncond", uk1);
    emit_arr("saddle_cgf2_uncond", uk2);
    emit_arr("saddle_cgf3_uncond", uk3);
    emit_arr("saddle_cgf4_uncond", uk4);

    std::vector<Real> povUncond;
    for (Real l : trancheFracts) povUncond.push_back(model->probOverLoss(d, l));
    emit_arr("saddle_prob_over_loss", povUncond);

    std::vector<Real> povPortfUncond, pdensUncond;
    for (Real l : absLosses) {
        povPortfUncond.push_back(model->probOverPortfLoss(d, l));
        pdensUncond.push_back(model->probDensity(d, l));
    }
    emit_arr("saddle_prob_over_portf_loss", povPortfUncond);
    emit_arr("saddle_prob_density", pdensUncond);

    emit("saddle_expected_tranche_loss", model->expectedTrancheLoss(d));

    std::vector<Real> percs = {0.10, 0.50, 0.90, 0.95, 0.99};
    emit_arr("saddle_percentile_levels", percs);
    std::vector<Real> percOut, esfOut;
    for (Real q : percs) {
        percOut.push_back(model->percentile(d, q));
        esfOut.push_back(model->expectedShortfall(d, q));
    }
    emit_arr("saddle_percentile", percOut);
    emit_arr("saddle_expected_shortfall", esfOut);

    std::vector<Real> splitVar = model->splitVaRLevel(d, 30.0);
    emit_arr("saddle_split_var_level_30", splitVar);

    // Second horizon, to catch anything that accidentally hard-codes the date.
    Date d2 = TARGET().advance(f.basket->refDate(), Period(24, Months));
    emit_int("saddle_calc_date2_serial", static_cast<long long>(d2.serialNumber()));
    emit("saddle_expected_tranche_loss_2y", model->expectedTrancheLoss(d2));
    emit("saddle_percentile_95_2y", model->percentile(d2, 0.95));
}

// --------------------------------------------------------------------------
// Block I: Homogeneous / Inhomogeneous pool loss models
// --------------------------------------------------------------------------

void poolBlocks(const Fixture& f) {
    const Date& d = f.calcDate;
    Date d2 = TARGET().advance(f.basket->refDate(), Period(24, Months));

    const Size nBuckets = 25;

    auto homog = ext::make_shared<ExposedHomog>(f.lm, nBuckets);
    f.basket->setLossModel(homog);
    forceModelReset(f);
    Distribution dh = homog->dist(d);
    emit_distribution("homog_loss_distrib_5y", dh);
    emit("homog_expected_tranche_loss_5y", homog->expectedTrancheLoss(d));
    emit("homog_expected_tranche_loss_2y", homog->expectedTrancheLoss(d2));
    std::vector<Real> hp, he;
    for (Real q : {0.10, 0.50, 0.90, 0.95, 0.99}) {
        hp.push_back(homog->percentile(d, q));
        he.push_back(homog->expectedShortfall(d, q));
    }
    emit_arr("homog_percentile_5y", hp);
    emit_arr("homog_expected_shortfall_5y", he);

    // Non-default integration grid, to pin the (min, max, nSteps) plumbing.
    auto homog2 = ext::make_shared<ExposedHomog>(f.lm, 40, 4.0, -4.0, 20);
    f.basket->setLossModel(homog2);
    forceModelReset(f);
    Distribution dh2 = homog2->dist(d);
    emit_distribution("homog_alt_grid_loss_distrib_5y", dh2);
    emit("homog_alt_grid_expected_tranche_loss_5y", homog2->expectedTrancheLoss(d));

    auto inhomog = ext::make_shared<ExposedInhomog>(f.lm, nBuckets);
    f.basket->setLossModel(inhomog);
    forceModelReset(f);
    Distribution di = inhomog->dist(d);
    emit_distribution("inhomog_loss_distrib_5y", di);
    emit("inhomog_expected_tranche_loss_5y", inhomog->expectedTrancheLoss(d));
    emit("inhomog_expected_tranche_loss_2y", inhomog->expectedTrancheLoss(d2));
    std::vector<Real> ip, ie;
    for (Real q : {0.10, 0.50, 0.90, 0.95, 0.99}) {
        ip.push_back(inhomog->percentile(d, q));
        ie.push_back(inhomog->expectedShortfall(d, q));
    }
    emit_arr("inhomog_percentile_5y", ip);
    emit_arr("inhomog_expected_shortfall_5y", ie);

    auto inhomog2 = ext::make_shared<ExposedInhomog>(f.lm, 40, 4.0, -4.0, 20);
    f.basket->setLossModel(inhomog2);
    forceModelReset(f);
    Distribution di2 = inhomog2->dist(d);
    emit_distribution("inhomog_alt_grid_loss_distrib_5y", di2);
    emit("inhomog_alt_grid_expected_tranche_loss_5y", inhomog2->expectedTrancheLoss(d));

    // Heterogeneous notionals: the two models must differ here, which is the
    // only regime where the Homogeneous shortcut is actually wrong.
    emit_int("pool_n_buckets", static_cast<long long>(nBuckets));
}

// --------------------------------------------------------------------------
// Block J: BaseCorrelationLossModel (GaussianLHP + bilinear surface)
// --------------------------------------------------------------------------

void baseCorrelBlock(const Fixture& f, const Date& today) {
    Calendar cal = TARGET();
    DayCounter dc = Actual365Fixed();

    std::vector<Period> tenors = {Period(1, Years), Period(3, Years), Period(5, Years)};
    std::vector<Real> lossLevels = {0.03, 0.06, 0.15};
    Real raw[3][3] = {{0.20, 0.30, 0.45}, {0.25, 0.35, 0.50}, {0.30, 0.40, 0.55}};
    std::vector<std::vector<Handle<Quote> > > correls;
    for (Size i = 0; i < 3; ++i) {
        std::vector<Handle<Quote> > row;
        for (Size j = 0; j < 3; ++j)
            row.emplace_back(ext::make_shared<SimpleQuote>(raw[i][j]));
        correls.push_back(row);
    }
    std::vector<Real> rawFlat;
    for (auto& i : raw)
        for (Real j : i) rawFlat.push_back(j);
    emit_arr("bclm_quotes_row_major", rawFlat);
    emit_arr("bclm_loss_levels", lossLevels);

    auto surface = ext::make_shared<BaseCorrelationTermStructure<BilinearInterpolation> >(
        0, cal, Following, tenors, lossLevels, correls, dc);
    Handle<BaseCorrelationTermStructure<BilinearInterpolation> > surfaceH(surface);

    std::vector<Real> nodeTimes;
    for (const Period& p : tenors)
        nodeTimes.push_back(
            surface->timeFromReference(cal.advance(surface->referenceDate(), p, Following)));
    emit_arr("bclm_tranche_times", nodeTimes);

    auto bclm = ext::make_shared<GaussianLHPFlatBCLM>(surfaceH, f.recoveries,
                                                      GaussianCopulaPolicy::initTraits());
    f.basket->setLossModel(bclm);
    forceModelReset(f);

    // The attach/detach ratios the model reads off the basket, and the two
    // correlations it pulls from the surface at those ratios.
    emit("bclm_attach_ratio", f.basket->remainingAttachmentAmount() / f.basket->remainingNotional());
    emit("bclm_detach_ratio", f.basket->remainingDetachmentAmount() / f.basket->remainingNotional());

    std::vector<Real> etls;
    std::vector<long long> dates;
    for (int months : {12, 24, 36, 60}) {
        Date dd = cal.advance(today, Period(months, Months));
        dates.push_back(static_cast<long long>(dd.serialNumber()));
        etls.push_back(f.basket->expectedTrancheLoss(dd));
    }
    emit_iarr("bclm_dates_serial", dates);
    emit_arr("bclm_expected_tranche_loss", etls);

    // Correlations the model interpolates at the attach / detach ratios.
    Real ar = f.basket->remainingAttachmentAmount() / f.basket->remainingNotional();
    Real dr = f.basket->remainingDetachmentAmount() / f.basket->remainingNotional();
    std::vector<Real> corrA, corrD;
    for (int months : {12, 24, 36, 60}) {
        Date dd = cal.advance(today, Period(months, Months));
        corrA.push_back(surface->correlation(dd, ar));
        corrD.push_back(surface->correlation(dd, dr));
    }
    emit_arr("bclm_correl_at_attach", corrA);
    emit_arr("bclm_correl_at_detach", corrD);
}

}  // namespace

int main() {
    // Unbuffered: if any block ever aborts, the JSON written so far names the
    // last emission that succeeded, which is how you find the offending call.
    std::cout << std::unitbuf;
    std::cout << "{\n";
    try {
        Calendar calendar = TARGET();
        Date today = calendar.adjust(Date(19, March, 2014));
        Settings::instance().evaluationDate() = today;
        emit_int("evaluation_date_serial", static_cast<long long>(today.serialNumber()));
        emit_str("quantlib_version", QL_VERSION);

        blockA();
        blockB();
        blockC(today);

        Fixture f = makeFixture(today);
        saddleBlocks(f);
        poolBlocks(f);
        baseCorrelBlock(f, today);
    } catch (const std::exception& e) {
        emit_str("fatal_error", e.what());
        std::cout << "\n}" << std::endl;
        return 1;
    }
    std::cout << "\n}" << std::endl;
    return 0;
}
