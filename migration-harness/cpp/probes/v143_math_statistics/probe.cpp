// migration-harness/cpp/probes/v143_math_statistics/probe.cpp
//
// Reference values for ql/math/statistics/** at C++ QuantLib v1.43.
//
// What is pinned and why:
//
//  * GeneralStatistics — the moment surface plus the two quantile lookups.
//    percentile()/topPercentile() walk the *weight* integral and stop at the
//    first sample whose cumulative weight reaches the target; the sample set
//    below deliberately contains ties and a target that lands exactly on a
//    cumulative-weight boundary, which is where an off-by-one convention
//    shows up.  expectationValue() is pinned directly (value + count) since
//    every risk measure is expressed through it.
//
//  * GenericRiskStatistics (== RiskStatistics == Statistics) — semiVariance,
//    downsideVariance, regret, potentialUpside, valueAtRisk,
//    expectedShortfall, shortfall, averageShortfall over four data sets:
//    one with ties and exact hits on the target, one wide symmetric spread,
//    one strictly positive (so every below-target measure must *throw*), and
//    one with exactly a single sample below target (regret needs N > 1).
//    Failures are recorded as the string "__error__" so the port has to
//    reproduce the throw, not just the happy path.
//
//  * GenericGaussianStatistics — the closed forms, probed over
//    GeneralStatistics, over IncrementalStatistics and over StatsHolder
//    (the precomputed-moments holder), which is the only reason the class is
//    a template rather than a plain base.
//
//  * ConvergenceStatistics + DoublingConvergenceSteps — the whole
//    convergenceTable(), before and after reset(), for both underlying
//    statistics types.  The doubling rule itself is pinned as a raw step
//    sequence.
//
//  * DiscrepancyStatistics — discrepancy() after *every* add.  The class
//    updates adiscr_/cdiscr_ incrementally; a batch recomputation agrees on
//    the final value and diverges on the intermediate ones.
//
//  * Histogram — breaks, counts and frequencies for all four bin rules
//    (Sturges, FD, Scott, explicit bin count) plus the explicit-breaks
//    constructor, whose near-duplicate break points exercise the
//    close_enough() de-duplication (and the resulting bins_/breaks_ mismatch,
//    which is a real quirk of the C++ code).
//
//  * GenericSequenceStatistics — the lifted per-dimension vectors and the
//    covariance/correlation matrices, over both Statistics and
//    IncrementalStatistics elements, with non-uniform weights.
//
//  * IncrementalStatistics — mean/variance/skewness/kurtosis/downside on a
//    weighted set, to pin the boost-accumulator formulas exactly.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/statistics.json.

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/statistics/convergencestatistics.hpp>
#include <ql/math/statistics/discrepancystatistics.hpp>
#include <ql/math/statistics/gaussianstatistics.hpp>
#include <ql/math/statistics/histogram.hpp>
#include <ql/math/statistics/incrementalstatistics.hpp>
#include <ql/math/statistics/riskstatistics.hpp>
#include <ql/math/statistics/sequencestatistics.hpp>
#include <ql/math/statistics/statistics.hpp>

using namespace QuantLib;

namespace {

void num(Real x) {
    if (!std::isfinite(x))
        std::cout << "\"__nonfinite__\"";
    else if (x == 0.0 && std::signbit(x))
        // The sign of a negative zero is load-bearing here: it is what
        // distinguishes ``-min(x, 0)`` from ``max(-x, 0)`` in valueAtRisk and
        // friends.  Printed as "-0.0" rather than "-0" because a JSON reader
        // parses "-0" as the integer zero and drops the sign.
        std::cout << "-0.0";
    else
        std::cout << x;
}

void arr(const std::vector<Real>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        num(v[i]);
    }
    std::cout << "]";
}

void sizeArr(const std::vector<Size>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

void matrix(const Matrix& m) {
    std::cout << "[";
    for (Size i = 0; i < m.rows(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << "[";
        for (Size j = 0; j < m.columns(); ++j) {
            if (j != 0)
                std::cout << ", ";
            num(m[i][j]);
        }
        std::cout << "]";
    }
    std::cout << "]";
}

// Evaluates f() and emits either the number or the marker "__error__" when
// the C++ side throws.  Reproducing the throw is part of the contract.
void guarded(const std::string& key, const std::function<Real()>& f, bool comma = true) {
    std::cout << "    \"" << key << "\": ";
    try {
        num(f());
    } catch (const std::exception&) {
        std::cout << "\"__error__\"";
    }
    std::cout << (comma ? ",\n" : "\n");
}

void guardedVec(const std::string& key,
                const std::function<std::vector<Real>()>& f,
                bool comma = true) {
    std::cout << "    \"" << key << "\": ";
    try {
        arr(f());
    } catch (const std::exception&) {
        std::cout << "\"__error__\"";
    }
    std::cout << (comma ? ",\n" : "\n");
}

void guardedMatrix(const std::string& key,
                   const std::function<Matrix()>& f,
                   bool comma = true) {
    std::cout << "    \"" << key << "\": ";
    try {
        matrix(f());
    } catch (const std::exception&) {
        std::cout << "\"__error__\"";
    }
    std::cout << (comma ? ",\n" : "\n");
}

// --- data sets --------------------------------------------------------------
//
// Every value is a dyadic rational so the decimal text in the JSON round-trips
// to the identical double on the Python side.

// Ties (-1 twice, 0 twice, 2 three times, 5 twice), samples sitting exactly on
// the natural targets (0 and the percentile boundary), both signs.
const std::vector<Real> kTies = {-3.0, -1.0, -1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 5.0, 5.0};

// Wide symmetric spread: -8.0 + 0.25*k, k = 0..63.
std::vector<Real> spreadData() {
    std::vector<Real> v(64);
    for (Size k = 0; k < 64; ++k)
        v[k] = -8.0 + 0.25 * Real(k);
    return v;
}

// Strictly positive: every below-zero measure must throw.
const std::vector<Real> kPositive = {1.5, 2.0, 2.0, 3.25, 4.0, 6.5};

// Exactly one sample below 0.0: regret()'s "samples under target <= 1" guard.
const std::vector<Real> kOneBelow = {5.0, 6.0, 7.0, -1.0};

// Non-uniform weights over a tied value set, to pin the weighted percentile
// walk and the weighted moments.
const std::vector<Real> kWeightedValues = {2.0, 1.0, 3.0, 1.0, 4.0, -2.0, 1.0};
const std::vector<Real> kWeights = {0.5, 2.0, 1.0, 1.5, 3.0, 0.25, 0.75};

// 50 points in [0, 5.75], all multiples of 1/8.
std::vector<Real> histogramData() {
    std::vector<Real> v(50);
    for (Size k = 0; k < 50; ++k)
        v[k] = Real((k * k) % 47) / 8.0;
    return v;
}

// 3-dimensional sequence samples, all multiples of 1/4, with at least two
// negatives per dimension so downsideVariance()/semiVariance() are defined.
const std::vector<std::vector<Real> > kSequence = {
    {-1.50, 2.00, 0.50},  {0.25, -1.00, 3.00},  {2.00, 0.50, -2.50},
    {-0.75, 1.25, 1.00},  {3.50, -2.00, 0.25},  {-2.25, 0.75, -1.50},
    {1.00, 3.00, 2.00},   {0.50, -0.50, -0.75}};
const std::vector<Real> kSequenceWeights = {1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.25, 2.0};

// Points in [0,1]^d for the discrepancy walk (dyadic, deterministic).
std::vector<std::vector<Real> > discrepancyPoints(Size dimension, Size n) {
    // A shifted-lattice style sequence: r_{i,k} = frac(0.125 + i*(k+1)/16).
    std::vector<std::vector<Real> > pts(n, std::vector<Real>(dimension));
    for (Size i = 0; i < n; ++i)
        for (Size k = 0; k < dimension; ++k) {
            Real t = 0.125 + Real(i) * Real(k + 1) / 16.0;
            pts[i][k] = t - std::floor(t);
        }
    return pts;
}

// --- emitters ---------------------------------------------------------------

void emitStatisticsBlock(const std::string& key,
                         const std::vector<Real>& data,
                         const std::vector<Real>& weights,
                         Real target,
                         Real centile,
                         bool comma) {
    Statistics s;
    if (weights.empty())
        s.addSequence(data.begin(), data.end());
    else
        s.addSequence(data.begin(), data.end(), weights.begin());

    std::cout << "  \"" << key << "\": {\n";
    std::cout << "    \"data\": ";
    arr(data);
    std::cout << ",\n";
    std::cout << "    \"weights\": ";
    arr(weights);
    std::cout << ",\n";
    std::cout << "    \"target\": ";
    num(target);
    std::cout << ",\n";
    std::cout << "    \"centile\": ";
    num(centile);
    std::cout << ",\n";
    std::cout << "    \"samples\": " << s.samples() << ",\n";

    guarded("weight_sum", [&] { return s.weightSum(); });
    guarded("mean", [&] { return s.mean(); });
    guarded("variance", [&] { return s.variance(); });
    guarded("standard_deviation", [&] { return s.standardDeviation(); });
    guarded("error_estimate", [&] { return s.errorEstimate(); });
    guarded("skewness", [&] { return s.skewness(); });
    guarded("kurtosis", [&] { return s.kurtosis(); });
    guarded("min", [&] { return s.min(); });
    guarded("max", [&] { return s.max(); });

    // percentile / topPercentile across the whole (0,1] range, including the
    // exact cumulative-weight boundaries.
    const Real ys[] = {0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 1.0};
    std::cout << "    \"percentile_ys\": [";
    for (Size i = 0; i < std::size(ys); ++i) {
        if (i != 0)
            std::cout << ", ";
        num(ys[i]);
    }
    std::cout << "],\n";
    std::cout << "    \"percentile\": [";
    for (Size i = 0; i < std::size(ys); ++i) {
        if (i != 0)
            std::cout << ", ";
        try {
            num(s.percentile(ys[i]));
        } catch (const std::exception&) {
            std::cout << "\"__error__\"";
        }
    }
    std::cout << "],\n";
    std::cout << "    \"top_percentile\": [";
    for (Size i = 0; i < std::size(ys); ++i) {
        if (i != 0)
            std::cout << ", ";
        try {
            num(s.topPercentile(ys[i]));
        } catch (const std::exception&) {
            std::cout << "\"__error__\"";
        }
    }
    std::cout << "],\n";

    // empirical risk measures
    guarded("semi_variance", [&] { return s.semiVariance(); });
    guarded("semi_deviation", [&] { return s.semiDeviation(); });
    guarded("downside_variance", [&] { return s.downsideVariance(); });
    guarded("downside_deviation", [&] { return s.downsideDeviation(); });
    guarded("regret", [&] { return s.regret(target); });
    guarded("potential_upside", [&] { return s.potentialUpside(centile); });
    guarded("value_at_risk", [&] { return s.valueAtRisk(centile); });
    guarded("expected_shortfall", [&] { return s.expectedShortfall(centile); });
    guarded("shortfall", [&] { return s.shortfall(target); });
    guarded("average_shortfall", [&] { return s.averageShortfall(target); });

    // gaussian closed forms
    guarded("gaussian_downside_variance", [&] { return s.gaussianDownsideVariance(); });
    guarded("gaussian_downside_deviation", [&] { return s.gaussianDownsideDeviation(); });
    guarded("gaussian_regret", [&] { return s.gaussianRegret(target); });
    guarded("gaussian_percentile", [&] { return s.gaussianPercentile(0.75); });
    guarded("gaussian_top_percentile", [&] { return s.gaussianTopPercentile(0.75); });
    guarded("gaussian_potential_upside", [&] { return s.gaussianPotentialUpside(centile); });
    guarded("gaussian_value_at_risk", [&] { return s.gaussianValueAtRisk(centile); });
    guarded("gaussian_expected_shortfall", [&] { return s.gaussianExpectedShortfall(centile); });
    guarded("gaussian_shortfall", [&] { return s.gaussianShortfall(target); });
    guarded("gaussian_average_shortfall", [&] { return s.gaussianAverageShortfall(target); });

    // out-of-range guards: all four must throw
    guarded("potential_upside_below_range", [&] { return s.potentialUpside(0.5); });
    guarded("value_at_risk_at_one", [&] { return s.valueAtRisk(1.0); });
    guarded("gaussian_percentile_at_zero", [&] { return s.gaussianPercentile(0.0); });
    guarded("gaussian_percentile_at_one", [&] { return s.gaussianPercentile(1.0); });

    // expectationValue, pinned directly: E[x^2 | x > target] and its count.
    Real evValue = 0.0;
    Size evCount = 0;
    {
        std::pair<Real, Size> r = s.expectationValue(
            [](Real x) { return x * x; }, [=](Real x) { return x > target; });
        evValue = r.first;
        evCount = r.second;
    }
    std::cout << "    \"expectation_value_sq_above_target\": ";
    num(evValue);
    std::cout << ",\n";
    std::cout << "    \"expectation_value_sq_above_target_count\": " << evCount << ",\n";

    std::pair<Real, Size> rAll = s.expectationValue([](Real x) { return x * x; });
    std::cout << "    \"expectation_value_sq_all\": ";
    num(rAll.first);
    std::cout << ",\n";
    std::cout << "    \"expectation_value_sq_all_count\": " << rAll.second << "\n";

    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitGaussianBlock(bool comma) {
    const std::vector<Real> data = spreadData();

    Statistics s;
    s.addSequence(data.begin(), data.end());

    GenericGaussianStatistics<IncrementalStatistics> igs;
    igs.addSequence(data.begin(), data.end());

    StatsHolder holder(s.mean(), s.standardDeviation());
    GenericGaussianStatistics<StatsHolder> held(holder);

    const Real target = -0.5;
    const Real centile = 0.95;

    std::cout << "  \"gaussian\": {\n";
    std::cout << "    \"data\": ";
    arr(data);
    std::cout << ",\n";
    std::cout << "    \"target\": ";
    num(target);
    std::cout << ",\n";
    std::cout << "    \"centile\": ";
    num(centile);
    std::cout << ",\n";
    std::cout << "    \"holder_mean\": ";
    num(holder.mean());
    std::cout << ",\n";
    std::cout << "    \"holder_standard_deviation\": ";
    num(holder.standardDeviation());
    std::cout << ",\n";

    guarded("general_gaussian_regret", [&] { return s.gaussianRegret(target); });
    guarded("general_gaussian_downside_variance", [&] { return s.gaussianDownsideVariance(); });
    guarded("general_gaussian_downside_deviation", [&] { return s.gaussianDownsideDeviation(); });
    guarded("general_gaussian_percentile", [&] { return s.gaussianPercentile(0.3); });
    guarded("general_gaussian_top_percentile", [&] { return s.gaussianTopPercentile(0.3); });
    guarded("general_gaussian_potential_upside", [&] { return s.gaussianPotentialUpside(centile); });
    guarded("general_gaussian_value_at_risk", [&] { return s.gaussianValueAtRisk(centile); });
    guarded("general_gaussian_expected_shortfall",
            [&] { return s.gaussianExpectedShortfall(centile); });
    guarded("general_gaussian_shortfall", [&] { return s.gaussianShortfall(target); });
    guarded("general_gaussian_average_shortfall",
            [&] { return s.gaussianAverageShortfall(target); });

    guarded("incremental_gaussian_regret", [&] { return igs.gaussianRegret(target); });
    guarded("incremental_gaussian_percentile", [&] { return igs.gaussianPercentile(0.3); });
    guarded("incremental_gaussian_potential_upside",
            [&] { return igs.gaussianPotentialUpside(centile); });
    guarded("incremental_gaussian_value_at_risk", [&] { return igs.gaussianValueAtRisk(centile); });
    guarded("incremental_gaussian_expected_shortfall",
            [&] { return igs.gaussianExpectedShortfall(centile); });
    guarded("incremental_gaussian_shortfall", [&] { return igs.gaussianShortfall(target); });
    guarded("incremental_gaussian_average_shortfall",
            [&] { return igs.gaussianAverageShortfall(target); });
    guarded("incremental_gaussian_downside_variance",
            [&] { return igs.gaussianDownsideVariance(); });

    guarded("holder_gaussian_regret", [&] { return held.gaussianRegret(target); });
    guarded("holder_gaussian_percentile", [&] { return held.gaussianPercentile(0.3); });
    guarded("holder_gaussian_potential_upside",
            [&] { return held.gaussianPotentialUpside(centile); });
    guarded("holder_gaussian_value_at_risk", [&] { return held.gaussianValueAtRisk(centile); });
    guarded("holder_gaussian_expected_shortfall",
            [&] { return held.gaussianExpectedShortfall(centile); });
    guarded("holder_gaussian_shortfall", [&] { return held.gaussianShortfall(target); });
    guarded("holder_gaussian_average_shortfall",
            [&] { return held.gaussianAverageShortfall(target); }, false);

    std::cout << "  }" << (comma ? "," : "") << "\n";
}

template <class S>
void emitConvergenceTable(const std::string& key, bool comma) {
    ConvergenceStatistics<S> stats;
    for (Size i = 1; i <= 8; ++i)
        stats.add(Real(i));

    std::cout << "    \"" << key << "\": {\n";
    std::cout << "      \"first_sizes\": [";
    for (Size i = 0; i < stats.convergenceTable().size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << stats.convergenceTable()[i].first;
    }
    std::cout << "],\n";
    std::cout << "      \"first_means\": [";
    for (Size i = 0; i < stats.convergenceTable().size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        num(stats.convergenceTable()[i].second);
    }
    std::cout << "],\n";

    stats.reset();
    for (Size i = 1; i <= 4; ++i)
        stats.add(Real(i));

    std::cout << "      \"after_reset_sizes\": [";
    for (Size i = 0; i < stats.convergenceTable().size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << stats.convergenceTable()[i].first;
    }
    std::cout << "],\n";
    std::cout << "      \"after_reset_means\": [";
    for (Size i = 0; i < stats.convergenceTable().size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        num(stats.convergenceTable()[i].second);
    }
    std::cout << "],\n";

    // weighted addSequence path, 20 samples, weight = 1 + i/4
    ConvergenceStatistics<S> wstats;
    std::vector<Real> vals(20), wts(20);
    for (Size i = 0; i < 20; ++i) {
        vals[i] = 1.0 + 0.5 * Real(i);
        wts[i] = 1.0 + 0.25 * Real(i);
    }
    wstats.addSequence(vals.begin(), vals.end(), wts.begin());
    std::cout << "      \"weighted_values\": ";
    arr(vals);
    std::cout << ",\n";
    std::cout << "      \"weighted_weights\": ";
    arr(wts);
    std::cout << ",\n";
    std::cout << "      \"weighted_sizes\": [";
    for (Size i = 0; i < wstats.convergenceTable().size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << wstats.convergenceTable()[i].first;
    }
    std::cout << "],\n";
    std::cout << "      \"weighted_means\": [";
    for (Size i = 0; i < wstats.convergenceTable().size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        num(wstats.convergenceTable()[i].second);
    }
    std::cout << "]\n";
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

void emitConvergenceBlock(bool comma) {
    std::cout << "  \"convergence\": {\n";

    DoublingConvergenceSteps rule;
    std::vector<Size> steps;
    Size n = rule.initialSamples();
    for (Size i = 0; i < 12; ++i) {
        steps.push_back(n);
        n = rule.nextSamples(n);
    }
    std::cout << "    \"doubling_steps\": ";
    sizeArr(steps);
    std::cout << ",\n";

    emitConvergenceTable<Statistics>("statistics", true);
    emitConvergenceTable<IncrementalStatistics>("incremental", false);

    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitDiscrepancyCase(Size dimension, Size n, bool comma) {
    const std::vector<std::vector<Real> > pts = discrepancyPoints(dimension, n);
    DiscrepancyStatistics stat(dimension);

    std::vector<Real> running;
    for (Size i = 0; i < n; ++i) {
        stat.add(pts[i]);
        running.push_back(stat.discrepancy());
    }

    std::cout << "    \"dim" << dimension << "\": {\n";
    std::cout << "      \"dimension\": " << dimension << ",\n";
    std::cout << "      \"points\": [";
    for (Size i = 0; i < n; ++i) {
        if (i != 0)
            std::cout << ", ";
        arr(pts[i]);
    }
    std::cout << "],\n";
    std::cout << "      \"running_discrepancy\": ";
    arr(running);
    std::cout << ",\n";
    std::cout << "      \"samples\": " << stat.samples() << "\n";
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

void emitDiscrepancyBlock(bool comma) {
    std::cout << "  \"discrepancy\": {\n";
    emitDiscrepancyCase(2, 12, true);
    emitDiscrepancyCase(3, 10, true);
    emitDiscrepancyCase(5, 8, false);
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitHistogram(const std::string& key, const Histogram& h, Size dataSize, bool comma) {
    std::cout << "    \"" << key << "\": {\n";
    std::cout << "      \"bins\": " << h.bins() << ",\n";
    std::cout << "      \"algorithm\": " << static_cast<int>(h.algorithm()) << ",\n";
    std::cout << "      \"empty\": " << (h.empty() ? "true" : "false") << ",\n";
    std::cout << "      \"breaks\": ";
    arr(h.breaks());
    std::cout << ",\n";
    std::vector<Size> counts;
    std::vector<Real> freqs;
    for (Size i = 0; i < h.bins(); ++i) {
        counts.push_back(h.counts(i));
        freqs.push_back(h.frequency(i));
    }
    std::cout << "      \"counts\": ";
    sizeArr(counts);
    std::cout << ",\n";
    std::cout << "      \"frequency\": ";
    arr(freqs);
    std::cout << ",\n";
    std::cout << "      \"data_size\": " << dataSize << "\n";
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

void emitHistogramBlock(bool comma) {
    const std::vector<Real> data = histogramData();

    std::cout << "  \"histogram\": {\n";
    std::cout << "    \"data\": ";
    arr(data);
    std::cout << ",\n";

    emitHistogram("sturges",
                  Histogram(data.begin(), data.end(), Histogram::Sturges),
                  data.size(), true);
    emitHistogram("fd", Histogram(data.begin(), data.end(), Histogram::FD), data.size(), true);
    emitHistogram("scott", Histogram(data.begin(), data.end(), Histogram::Scott), data.size(),
                  true);
    emitHistogram("breaks_count_4", Histogram(data.begin(), data.end(), Size(4)), data.size(),
                  true);
    emitHistogram("breaks_count_1", Histogram(data.begin(), data.end(), Size(1)), data.size(),
                  true);

    // Explicit break points, deliberately unsorted and containing a pair that
    // close_enough() collapses.  bins_ is fixed at breaks.size()+1 *before*
    // the de-duplication, so the trailing bins stay empty — pinned as-is.
    const std::vector<Real> explicitBreaks = {3.0, 1.0, 2.0, 2.0 + 5e-16};
    std::cout << "    \"explicit_breaks_input\": ";
    arr(explicitBreaks);
    std::cout << ",\n";
    emitHistogram("explicit_breaks",
                  Histogram(data.begin(), data.end(), explicitBreaks.begin(),
                            explicitBreaks.end()),
                  data.size(), true);

    Histogram defaulted;
    std::cout << "    \"default_constructed\": {\n";
    std::cout << "      \"bins\": " << defaulted.bins() << ",\n";
    std::cout << "      \"algorithm\": " << static_cast<int>(defaulted.algorithm()) << ",\n";
    std::cout << "      \"empty\": " << (defaulted.empty() ? "true" : "false") << "\n";
    std::cout << "    },\n";

    // The None algorithm must fail.
    std::cout << "    \"none_algorithm_throws\": ";
    try {
        Histogram bad(data.begin(), data.end(), Histogram::None);
        std::cout << "false";
    } catch (const std::exception&) {
        std::cout << "true";
    }
    std::cout << "\n";

    std::cout << "  }" << (comma ? "," : "") << "\n";
}

template <class S>
void emitSequenceBlock(const std::string& key, bool comma) {
    GenericSequenceStatistics<S> ss(3);
    for (Size i = 0; i < kSequence.size(); ++i)
        ss.add(kSequence[i], kSequenceWeights[i]);

    std::cout << "    \"" << key << "\": {\n";
    std::cout << "      \"size\": " << ss.size() << ",\n";
    std::cout << "      \"samples\": " << ss.samples() << ",\n";
    guarded("weight_sum", [&] { return ss.weightSum(); });
    guardedVec("mean", [&] { return ss.mean(); });
    guardedVec("variance", [&] { return ss.variance(); });
    guardedVec("standard_deviation", [&] { return ss.standardDeviation(); });
    guardedVec("error_estimate", [&] { return ss.errorEstimate(); });
    guardedVec("skewness", [&] { return ss.skewness(); });
    guardedVec("kurtosis", [&] { return ss.kurtosis(); });
    guardedVec("min", [&] { return ss.min(); });
    guardedVec("max", [&] { return ss.max(); });
    guardedVec("downside_variance", [&] { return ss.downsideVariance(); });
    guardedVec("downside_deviation", [&] { return ss.downsideDeviation(); });
    guardedMatrix("covariance", [&] { return ss.covariance(); });
    guardedMatrix("correlation", [&] { return ss.correlation(); }, false);
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

void emitSequenceStatisticsOnly(bool comma) {
    // The single-argument lifted methods only exist for element types that
    // carry the risk surface, i.e. SequenceStatistics (== over Statistics).
    SequenceStatistics ss(3);
    for (Size i = 0; i < kSequence.size(); ++i)
        ss.add(kSequence[i], kSequenceWeights[i]);

    const Real target = 0.0;
    const Real centile = 0.9;

    std::cout << "    \"statistics_risk\": {\n";
    std::cout << "      \"target\": ";
    num(target);
    std::cout << ",\n";
    std::cout << "      \"centile\": ";
    num(centile);
    std::cout << ",\n";
    guardedVec("semi_variance", [&] { return ss.semiVariance(); });
    guardedVec("semi_deviation", [&] { return ss.semiDeviation(); });
    guardedVec("percentile", [&] { return ss.percentile(0.5); });
    guardedVec("gaussian_percentile", [&] { return ss.gaussianPercentile(0.5); });
    guardedVec("potential_upside", [&] { return ss.potentialUpside(centile); });
    guardedVec("gaussian_potential_upside", [&] { return ss.gaussianPotentialUpside(centile); });
    guardedVec("value_at_risk", [&] { return ss.valueAtRisk(centile); });
    guardedVec("gaussian_value_at_risk", [&] { return ss.gaussianValueAtRisk(centile); });
    guardedVec("expected_shortfall", [&] { return ss.expectedShortfall(centile); });
    guardedVec("gaussian_expected_shortfall", [&] { return ss.gaussianExpectedShortfall(centile); });
    guardedVec("regret", [&] { return ss.regret(target); });
    guardedVec("shortfall", [&] { return ss.shortfall(target); });
    guardedVec("gaussian_shortfall", [&] { return ss.gaussianShortfall(target); });
    guardedVec("average_shortfall", [&] { return ss.averageShortfall(target); });
    guardedVec("gaussian_average_shortfall", [&] { return ss.gaussianAverageShortfall(target); },
               false);
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

void emitSequenceStatisticsBlock(bool comma) {
    std::cout << "  \"sequence\": {\n";
    std::cout << "    \"samples_input\": [";
    for (Size i = 0; i < kSequence.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        arr(kSequence[i]);
    }
    std::cout << "],\n";
    std::cout << "    \"weights\": ";
    arr(kSequenceWeights);
    std::cout << ",\n";
    emitSequenceBlock<Statistics>("statistics", true);
    emitSequenceBlock<IncrementalStatistics>("incremental", true);
    emitSequenceStatisticsOnly(false);
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

void emitIncrementalBlock(bool comma) {
    IncrementalStatistics inc;
    inc.addSequence(kWeightedValues.begin(), kWeightedValues.end(), kWeights.begin());

    // A second set with a large offset: this is where the lazy weighted_mean
    // (weighted_sum / sum_of_weights) and the iterative weighted_variance
    // pull apart from a naive implementation.
    //
    // Only mean and variance are pinned for this set.  skewness/kurtosis go
    // through boost's *lazy* estimators, whose (m2 - m1^2) and
    // (m3 - 3*m2*m1 + 2*m1^3) terms cancel to nothing here: with x ~ 1e8 the
    // exact numerator is -1.63e-2 while the double-precision one is +5.37e8,
    // a relative error of 3e10.  The result carries zero significant digits,
    // so whatever this compiler prints is rounding noise, not a reference
    // value, and pinning it would only force a port to reproduce one
    // particular compiler's FP contraction.
    IncrementalStatistics shifted;
    std::vector<Real> shiftedValues(64), shiftedWeights(64);
    for (Size i = 0; i < 64; ++i) {
        shiftedValues[i] = 1.0e8 + 0.125 * Real(i) - 4.0;
        shiftedWeights[i] = 0.5 + 0.125 * Real(i % 7);
    }
    shifted.addSequence(shiftedValues.begin(), shiftedValues.end(), shiftedWeights.begin());

    // Zero-weight sample: boost still counts it.
    IncrementalStatistics zeroW;
    zeroW.add(1.0, 1.0);
    zeroW.add(9.0, 0.0);
    zeroW.add(3.0, 1.0);

    std::cout << "  \"incremental\": {\n";
    std::cout << "    \"values\": ";
    arr(kWeightedValues);
    std::cout << ",\n";
    std::cout << "    \"weights\": ";
    arr(kWeights);
    std::cout << ",\n";
    std::cout << "    \"samples\": " << inc.samples() << ",\n";
    guarded("weight_sum", [&] { return inc.weightSum(); });
    guarded("mean", [&] { return inc.mean(); });
    guarded("variance", [&] { return inc.variance(); });
    guarded("standard_deviation", [&] { return inc.standardDeviation(); });
    guarded("error_estimate", [&] { return inc.errorEstimate(); });
    guarded("skewness", [&] { return inc.skewness(); });
    guarded("kurtosis", [&] { return inc.kurtosis(); });
    guarded("min", [&] { return inc.min(); });
    guarded("max", [&] { return inc.max(); });
    std::cout << "    \"downside_samples\": " << inc.downsideSamples() << ",\n";
    guarded("downside_weight_sum", [&] { return inc.downsideWeightSum(); });
    guarded("downside_variance", [&] { return inc.downsideVariance(); });
    guarded("downside_deviation", [&] { return inc.downsideDeviation(); });

    std::cout << "    \"shifted_values\": ";
    arr(shiftedValues);
    std::cout << ",\n";
    std::cout << "    \"shifted_weights\": ";
    arr(shiftedWeights);
    std::cout << ",\n";
    guarded("shifted_mean", [&] { return shifted.mean(); });
    guarded("shifted_variance", [&] { return shifted.variance(); });

    std::cout << "    \"zero_weight_samples\": " << zeroW.samples() << ",\n";
    guarded("zero_weight_weight_sum", [&] { return zeroW.weightSum(); });
    guarded("zero_weight_mean", [&] { return zeroW.mean(); });
    guarded("zero_weight_max", [&] { return zeroW.max(); }, false);

    std::cout << "  }" << (comma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const std::vector<Real> noWeights;
    emitStatisticsBlock("ties", kTies, noWeights, 0.0, 0.95, true);
    emitStatisticsBlock("spread", spreadData(), noWeights, 0.0, 0.95, true);
    emitStatisticsBlock("positive", kPositive, noWeights, 0.0, 0.9, true);
    emitStatisticsBlock("one_below", kOneBelow, noWeights, 0.0, 0.9, true);
    emitStatisticsBlock("weighted", kWeightedValues, kWeights, 1.0, 0.9, true);

    emitGaussianBlock(true);
    emitConvergenceBlock(true);
    emitDiscrepancyBlock(true);
    emitHistogramBlock(true);
    emitSequenceStatisticsBlock(true);
    emitIncrementalBlock(false);

    std::cout << "}\n";
    return 0;
}
