// migration-harness/cpp/probes/v143_legacy_lmm/probe.cpp
//
// Reference values for the LIBOR-Market-Model coverage tail that C++ keeps
// under ql/legacy/libormarketmodels/ (v1.43) — volatility models, correlation
// models, covariance parameterizations and the LFM stochastic process:
//
//   LmVolatilityModel                   lmvolmodel.hpp
//   LmFixedVolatilityModel              lmfixedvolmodel.hpp
//   LmLinearExponentialVolatilityModel  lmlinexpvolmodel.hpp
//   LmExtLinearExponentialVolModel      lmextlinexpvolmodel.hpp
//   LmConstWrapperVolatilityModel       lmconstwrappervolmodel.hpp
//   LmCorrelationModel                  lmcorrmodel.hpp
//   LmExponentialCorrelationModel       lmexpcorrmodel.hpp
//   LmLinearExponentialCorrelationModel lmlinexpcorrmodel.hpp
//   LmConstWrapperCorrelationModel      lmconstwrappercorrmodel.hpp
//   LfmCovarianceParameterization       lfmcovarparam.hpp
//   LfmCovarianceProxy                  lfmcovarproxy.hpp
//   LfmHullWhiteParameterization        lfmhullwhiteparam.hpp
//   LiborForwardModelProcess            lfmprocess.hpp
//
// LiborForwardModel and LfmSwaptionEngine are pinned by the sibling probe
// v143_legacy_lmmswaption.
//
// What is pinned and why:
//
//   * EVERY constructor argument at a NON-DEFAULT value. The defect class this
//     port is prone to is an argument that is accepted and then silently
//     dropped, so:
//       - LmLinearExponentialCorrelationModel is built BOTH with an explicit
//         `factors = 3` (< size) and with the Null<Size> default. The two
//         differ in `factors()`, in the shape of `pseudoSqrt`, AND in
//         `correlation()` itself, because generateArguments() overwrites
//         corrMatrix_ with pseudoSqrt * transpose(pseudoSqrt) — a rank-3
//         reconstruction is NOT the raw rho + (1-rho) exp(-beta |i-j|)
//         matrix. A port that ignored `factors` would fail on the matrix,
//         not only on the inspector.
//       - the four (a, b, c, d) of the linear-exponential volatility model
//         are mutually distinct and none is 0 or 1.
//       - LmExtLinearExponentialVolModel is emitted with its default k_i == 1
//         AND after setParams() installs k_i = 0.8 + 0.05 i, which is the only
//         thing separating it from its base class.
//       - LmFixedVolatilityModel gets a volatility array whose entries are all
//         distinct and start times that are neither uniform nor starting at 0.
//
//   * setParams() round-trips on both correlation models and on the extended
//     volatility model. setParams() is what wires the LmXxxModel into
//     CalibratedModel calibration; a port whose setParams forgot to call
//     generateArguments() would still reproduce every constructor-time value.
//
//   * The FULL matrices (correlation, pseudoSqrt, diffusion, covariance,
//     integratedCovariance), not just a norm or a diagonal — an error in one
//     entry and an error in another must not be able to cancel.
//
//     pseudoSqrt is sign-sensitive: SymmetricSchurDecomposition pins each
//     eigenvector so its first component is >= 0, and the resulting
//     pseudo-root feeds LfmCovarianceProxy::diffusion, hence the process
//     evolution. B B^T alone would not catch a sign flip, so B itself is
//     pinned entry by entry.
//
//   * BOTH branches of LfmCovarianceProxy::integratedCovariance(i, j, t):
//       - the analytic fast path (time-independent correlation AND a
//         volatility model that implements integratedVariance);
//       - the GaussKronrodAdaptive fallback, reached here by pairing a
//         LmFixedVolatilityModel (whose integratedVariance inherits the
//         base-class QL_FAIL) with the same correlation model. The C++
//         try/catch swallowing that failure is load-bearing behaviour.
//     Plus LfmCovarianceParameterization::integratedCovariance(t), the
//     64-segment numerical double loop, on a deliberately small (size 3)
//     proxy so the Python side can afford to run it.
//
//   * LiborForwardModelProcess: fixing times/dates, accrual start/end times,
//     initial values, nextIndexReset around a fixing (the strict `upper_bound`
//     boundary is exactly what the C++ test-suite testInitialisation checks),
//     discountBond, drift, diffusion, covariance(t0, x0, dt) — note dt is
//     applied as a scale factor — apply, evolve (the predictor-corrector step),
//     and cashFlows at a NON-UNIT notional.
//
//   * LfmHullWhiteParameterization in both its 1-factor form (no correlation
//     matrix given) and its 3-factor form, with the Hull-White factor loadings
//     the C++ test-suite uses. The lambda bootstrapping is pinned through
//     sqrt(covariance(0)[i][i]), the same quantity testLambdaBootstrapping
//     checks, plus the raw diffusion/covariance/integratedCovariance matrices
//     at several times.
//
// The market setup mirrors test-suite/libormarketmodel.cpp (Euribor6M over a
// two-pillar ZeroCurve) and test-suite/libormarketmodelprocess.cpp (Euribor1Y
// over a steeper curve, with a CapletVarianceCurve built from the process's
// own fixing dates).
//
// EVALUATION DATE: this probe sets Settings::instance().evaluationDate(). The
// value is emitted as "evaluation_date_serial" at the top level and the pytest
// module MUST pin the same date, or every date-derived number below drifts
// with the wall clock.
//
// Emits JSON on stdout; redirect to references/v143/legacy/lmm.json.

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/cashflows/coupon.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/legacy/libormarketmodels/lfmcovarproxy.hpp>
#include <ql/legacy/libormarketmodels/lfmhullwhiteparam.hpp>
#include <ql/legacy/libormarketmodels/lfmprocess.hpp>
#include <ql/legacy/libormarketmodels/lmconstwrappercorrmodel.hpp>
#include <ql/legacy/libormarketmodels/lmconstwrappervolmodel.hpp>
#include <ql/legacy/libormarketmodels/lmexpcorrmodel.hpp>
#include <ql/legacy/libormarketmodels/lmextlinexpvolmodel.hpp>
#include <ql/legacy/libormarketmodels/lmfixedvolmodel.hpp>
#include <ql/legacy/libormarketmodels/lmlinexpcorrmodel.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/models/parameter.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/optionlet/capletvariancecurve.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actualactual.hpp>

using namespace QuantLib;

namespace {

// --- JSON helpers ----------------------------------------------------------

void emitArray(const char* key, const std::vector<Real>& v, int indent, bool comma) {
    std::string pad(indent, ' ');
    std::cout << pad << "\"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitArray(const char* key, const Array& a, int indent, bool comma) {
    emitArray(key, std::vector<Real>(a.begin(), a.end()), indent, comma);
}

void emitIntArray(const char* key, const std::vector<long>& v, int indent, bool comma) {
    std::string pad(indent, ' ');
    std::cout << pad << "\"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitMatrix(const char* key, const Matrix& m, int indent, bool comma) {
    std::string pad(indent, ' ');
    std::cout << pad << "\"" << key << "\": [";
    for (Size i = 0; i < m.rows(); ++i) {
        if (i != 0)
            std::cout << ",";
        std::cout << "\n" << pad << "  [";
        for (Size j = 0; j < m.columns(); ++j) {
            if (j != 0)
                std::cout << ", ";
            std::cout << m[i][j];
        }
        std::cout << "]";
    }
    std::cout << "\n" << pad << "]" << (comma ? "," : "") << "\n";
}

void emitScalar(const char* key, Real x, int indent, bool comma) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": " << x << (comma ? "," : "")
              << "\n";
}

void emitInt(const char* key, long x, int indent, bool comma) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": " << x << (comma ? "," : "")
              << "\n";
}

void emitBool(const char* key, bool b, int indent, bool comma) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": " << (b ? "true" : "false")
              << (comma ? "," : "") << "\n";
}

// Flatten every param of an Lm*Model. All of them are ConstantParameter, so
// evaluating at t = 0 is the whole content (that is exactly how the C++
// models read them back: arguments_[k](0.0)).
std::vector<Real> paramValues(const std::vector<Parameter>& p) {
    std::vector<Real> out;
    out.reserve(p.size());
    for (const auto& q : p)
        out.push_back(q(0.0));
    return out;
}

// --- market setup ----------------------------------------------------------

// test-suite/libormarketmodel.cpp makeIndex(): Euribor6M over a two-pillar
// ZeroCurve whose first pillar is the spot date implied by the evaluation date.
const Date kAnchor(4, September, 2005);

ext::shared_ptr<IborIndex> makeIndex6M(std::vector<Date> dates,
                                       const std::vector<Rate>& rates) {
    DayCounter dayCounter = Actual360();
    RelinkableHandle<YieldTermStructure> termStructure;
    ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));

    Date todaysDate = index->fixingCalendar().adjust(kAnchor);
    Settings::instance().evaluationDate() = todaysDate;

    dates[0] = index->fixingCalendar().advance(todaysDate, index->fixingDays(), Days);
    termStructure.linkTo(
        ext::shared_ptr<YieldTermStructure>(new ZeroCurve(dates, rates, dayCounter)));
    return index;
}

ext::shared_ptr<IborIndex> makeIndex6M() {
    return makeIndex6M({{4, September, 2005}, {4, September, 2018}}, {0.039, 0.041});
}

// test-suite/libormarketmodelprocess.cpp makeIndex(): Euribor1Y, steeper curve.
ext::shared_ptr<IborIndex> makeIndex1Y() {
    DayCounter dayCounter = Actual360();
    std::vector<Date> dates = {{4, September, 2005}, {4, September, 2018}};
    std::vector<Rate> rates = {0.01, 0.08};

    RelinkableHandle<YieldTermStructure> termStructure(
        ext::shared_ptr<YieldTermStructure>(new ZeroCurve(dates, rates, dayCounter)));
    ext::shared_ptr<IborIndex> index(new Euribor1Y(termStructure));

    Date todaysDate = index->fixingCalendar().adjust(kAnchor);
    Settings::instance().evaluationDate() = todaysDate;

    dates[0] = index->fixingCalendar().advance(todaysDate, index->fixingDays(), Days);
    termStructure.linkTo(
        ext::shared_ptr<YieldTermStructure>(new ZeroCurve(dates, rates, dayCounter)));
    return index;
}

const Size kHwLen = 10;

// test-suite/libormarketmodelprocess.cpp makeCapVolCurve().
ext::shared_ptr<CapletVarianceCurve> makeCapVolCurve(const Date& todaysDate) {
    Volatility vols[] = {14.40, 17.15, 16.81, 16.64, 16.17, 15.78, 15.40, 15.21, 14.86, 14.54};

    std::vector<Date> dates;
    std::vector<Volatility> capletVols;
    ext::shared_ptr<LiborForwardModelProcess> process(
        new LiborForwardModelProcess(kHwLen + 1, makeIndex1Y()));

    for (Size i = 0; i < kHwLen; ++i) {
        capletVols.push_back(vols[i] / 100);
        dates.push_back(process->fixingDates()[i + 1]);
    }
    return ext::make_shared<CapletVarianceCurve>(todaysDate, dates, capletVols,
                                                 ActualActual(ActualActual::ISDA));
}

// --- shared model parameters ----------------------------------------------
//
// Deliberately all distinct and none equal to 0 or 1.
const Real kA = 0.23;
const Real kB = 0.17;
const Real kC = 0.11;
const Real kD = 0.29;
const Size kSize = 6;

std::vector<Time> fixingTimeGrid() {
    // Non-uniform, does NOT start at 0 — a port that hard-coded 0.5 * i or
    // assumed fixingTimes_[0] == 0 fails here.
    return {0.25, 0.80, 1.35, 2.10, 3.05, 4.40};
}

const std::vector<Time> kEvalTimes = {0.0, 0.37, 1.10, 2.55, 3.90};

} // namespace

int main() {
    try {
        std::cout << std::setprecision(17);

        // Both makeIndex helpers set the evaluation date to the same adjusted
        // anchor; build one now so the emitted serial is authoritative.
        ext::shared_ptr<IborIndex> index6m = makeIndex6M();
        const Date evalDate = Settings::instance().evaluationDate();

        std::cout << "{\n";
        emitInt("evaluation_date_serial", evalDate.serialNumber(), 2, true);

        // ============================================================
        // LmExponentialCorrelationModel  (+ LmCorrelationModel base)
        // ============================================================
        const Size expCorrSize = 7;
        const Real expCorrRho = 0.13;
        ext::shared_ptr<LmCorrelationModel> expCorr(
            new LmExponentialCorrelationModel(expCorrSize, expCorrRho));

        std::cout << "  \"lm_exponential_correlation\": {\n";
        emitInt("size", static_cast<long>(expCorr->size()), 4, true);
        emitInt("factors", static_cast<long>(expCorr->factors()), 4, true);
        emitBool("is_time_independent", expCorr->isTimeIndependent(), 4, true);
        emitArray("params", paramValues(expCorr->params()), 4, true);
        emitMatrix("correlation_t0", expCorr->correlation(0.0), 4, true);
        // Time-independent: the same matrix must come back at any t.
        emitMatrix("correlation_t3_7", expCorr->correlation(3.7), 4, true);
        emitMatrix("pseudo_sqrt_t0", expCorr->pseudoSqrt(0.0), 4, true);
        {
            // scalar accessor, full grid
            Matrix scalarGrid(expCorrSize, expCorrSize);
            for (Size i = 0; i < expCorrSize; ++i)
                for (Size j = 0; j < expCorrSize; ++j)
                    scalarGrid[i][j] = expCorr->correlation(i, j, 1.9, Array());
            emitMatrix("correlation_scalar_t1_9", scalarGrid, 4, true);
        }
        {
            // setParams round-trip: generateArguments() must rebuild both
            // corrMatrix_ and pseudoSqrt_.
            std::vector<Parameter> np(1);
            np[0] = ConstantParameter(0.47, PositiveConstraint());
            expCorr->setParams(np);
            emitArray("params_after_set", paramValues(expCorr->params()), 4, true);
            emitMatrix("correlation_after_set", expCorr->correlation(0.0), 4, true);
            emitMatrix("pseudo_sqrt_after_set", expCorr->pseudoSqrt(0.0), 4, false);
            // restore
            std::vector<Parameter> back(1);
            back[0] = ConstantParameter(expCorrRho, PositiveConstraint());
            expCorr->setParams(back);
        }
        std::cout << "  },\n";

        // ============================================================
        // LmLinearExponentialCorrelationModel
        // ============================================================
        const Size linExpCorrSize = 8;
        const Real linExpCorrRho = 0.42;
        const Real linExpCorrBeta = 0.73;
        {
            // explicit, NON-DEFAULT factor count (3 < size)
            LmLinearExponentialCorrelationModel m(linExpCorrSize, linExpCorrRho, linExpCorrBeta, 3);
            std::cout << "  \"lm_linexp_correlation_3factors\": {\n";
            emitInt("size", static_cast<long>(m.size()), 4, true);
            emitInt("factors", static_cast<long>(m.factors()), 4, true);
            emitBool("is_time_independent", m.isTimeIndependent(), 4, true);
            emitArray("params", paramValues(m.params()), 4, true);
            emitMatrix("correlation_t0", m.correlation(0.0), 4, true);
            emitMatrix("correlation_t2_2", m.correlation(2.2), 4, true);
            emitMatrix("pseudo_sqrt_t0", m.pseudoSqrt(0.0), 4, true);
            {
                Matrix scalarGrid(linExpCorrSize, linExpCorrSize);
                for (Size i = 0; i < linExpCorrSize; ++i)
                    for (Size j = 0; j < linExpCorrSize; ++j)
                        scalarGrid[i][j] = m.correlation(i, j, 0.4, Array());
                emitMatrix("correlation_scalar_t0_4", scalarGrid, 4, true);
            }
            {
                std::vector<Parameter> np(2);
                np[0] = ConstantParameter(-0.15, BoundaryConstraint(-1.0, 1.0));
                np[1] = ConstantParameter(1.35, PositiveConstraint());
                m.setParams(np);
                emitArray("params_after_set", paramValues(m.params()), 4, true);
                emitMatrix("correlation_after_set", m.correlation(0.0), 4, true);
                emitMatrix("pseudo_sqrt_after_set", m.pseudoSqrt(0.0), 4, false);
            }
            std::cout << "  },\n";
        }
        {
            // Null<Size> default -> factors() == size, full-rank reconstruction
            LmLinearExponentialCorrelationModel m(linExpCorrSize, linExpCorrRho, linExpCorrBeta);
            std::cout << "  \"lm_linexp_correlation_default_factors\": {\n";
            emitInt("factors", static_cast<long>(m.factors()), 4, true);
            emitMatrix("correlation_t0", m.correlation(0.0), 4, true);
            emitMatrix("pseudo_sqrt_t0", m.pseudoSqrt(0.0), 4, false);
            std::cout << "  },\n";
        }

        // ============================================================
        // LmLinearExponentialVolatilityModel (+ LmVolatilityModel base)
        // ============================================================
        const std::vector<Time> fixingTimes = fixingTimeGrid();
        ext::shared_ptr<LmVolatilityModel> linExpVol(
            new LmLinearExponentialVolatilityModel(fixingTimes, kA, kB, kC, kD));

        std::cout << "  \"lm_linexp_volatility\": {\n";
        emitInt("size", static_cast<long>(linExpVol->size()), 4, true);
        emitArray("fixing_times", std::vector<Real>(fixingTimes.begin(), fixingTimes.end()), 4,
                  true);
        emitArray("params", paramValues(linExpVol->params()), 4, true);
        {
            Matrix volGrid(kEvalTimes.size(), kSize);
            Matrix volScalarGrid(kEvalTimes.size(), kSize);
            for (Size k = 0; k < kEvalTimes.size(); ++k) {
                Array v = linExpVol->volatility(kEvalTimes[k]);
                for (Size i = 0; i < kSize; ++i) {
                    volGrid[k][i] = v[i];
                    volScalarGrid[k][i] = linExpVol->volatility(i, kEvalTimes[k], Array());
                }
            }
            emitArray("eval_times", std::vector<Real>(kEvalTimes.begin(), kEvalTimes.end()), 4,
                      true);
            emitMatrix("volatility_vector", volGrid, 4, true);
            emitMatrix("volatility_scalar", volScalarGrid, 4, true);
        }
        {
            // integratedVariance over the full (i, j) grid at u = 1.7
            Matrix iv(kSize, kSize);
            for (Size i = 0; i < kSize; ++i)
                for (Size j = 0; j < kSize; ++j)
                    iv[i][j] = linExpVol->integratedVariance(i, j, 1.7, Array());
            emitMatrix("integrated_variance_u1_7", iv, 4, true);
            Matrix iv2(kSize, kSize);
            for (Size i = 0; i < kSize; ++i)
                for (Size j = 0; j < kSize; ++j)
                    iv2[i][j] = linExpVol->integratedVariance(i, j, 4.0, Array());
            emitMatrix("integrated_variance_u4_0", iv2, 4, false);
        }
        std::cout << "  },\n";

        // ============================================================
        // LmExtLinearExponentialVolModel
        // ============================================================
        {
            ext::shared_ptr<LmVolatilityModel> extVol(
                new LmExtLinearExponentialVolModel(fixingTimes, kA, kB, kC, kD));

            std::cout << "  \"lm_ext_linexp_volatility\": {\n";
            emitInt("size", static_cast<long>(extVol->size()), 4, true);
            emitInt("n_params", static_cast<long>(extVol->params().size()), 4, true);
            emitArray("params_default", paramValues(extVol->params()), 4, true);
            {
                // with default k_i == 1 it must equal the base model exactly
                Matrix volGrid(kEvalTimes.size(), kSize);
                for (Size k = 0; k < kEvalTimes.size(); ++k) {
                    Array v = extVol->volatility(kEvalTimes[k]);
                    for (Size i = 0; i < kSize; ++i)
                        volGrid[k][i] = v[i];
                }
                emitMatrix("volatility_vector_default", volGrid, 4, true);
            }
            {
                // k_i = 0.8 + 0.05 i — all distinct, none equal to 1
                std::vector<Parameter> np(4 + kSize);
                np[0] = ConstantParameter(kA, PositiveConstraint());
                np[1] = ConstantParameter(kB, PositiveConstraint());
                np[2] = ConstantParameter(kC, PositiveConstraint());
                np[3] = ConstantParameter(kD, PositiveConstraint());
                for (Size i = 0; i < kSize; ++i)
                    np[4 + i] = ConstantParameter(0.8 + 0.05 * Real(i), PositiveConstraint());
                extVol->setParams(np);
                emitArray("params_scaled", paramValues(extVol->params()), 4, true);

                Matrix volGrid(kEvalTimes.size(), kSize);
                Matrix volScalarGrid(kEvalTimes.size(), kSize);
                for (Size k = 0; k < kEvalTimes.size(); ++k) {
                    Array v = extVol->volatility(kEvalTimes[k]);
                    for (Size i = 0; i < kSize; ++i) {
                        volGrid[k][i] = v[i];
                        volScalarGrid[k][i] = extVol->volatility(i, kEvalTimes[k], Array());
                    }
                }
                emitMatrix("volatility_vector_scaled", volGrid, 4, true);
                emitMatrix("volatility_scalar_scaled", volScalarGrid, 4, true);

                Matrix iv(kSize, kSize);
                for (Size i = 0; i < kSize; ++i)
                    for (Size j = 0; j < kSize; ++j)
                        iv[i][j] = extVol->integratedVariance(i, j, 1.7, Array());
                emitMatrix("integrated_variance_scaled_u1_7", iv, 4, false);
            }
            std::cout << "  },\n";
        }

        // ============================================================
        // LmFixedVolatilityModel
        // ============================================================
        {
            // Distinct volatilities; start times non-uniform and not from 0.
            Array vols(kSize);
            for (Size i = 0; i < kSize; ++i)
                vols[i] = 0.12 + 0.017 * Real(i);
            const std::vector<Time> startTimes = fixingTimes;

            LmFixedVolatilityModel m(vols, startTimes);
            std::cout << "  \"lm_fixed_volatility\": {\n";
            emitInt("size", static_cast<long>(m.size()), 4, true);
            emitInt("n_params", static_cast<long>(m.params().size()), 4, true);
            emitArray("volatilities", vols, 4, true);
            emitArray("start_times", std::vector<Real>(startTimes.begin(), startTimes.end()), 4,
                      true);
            {
                // Times chosen to straddle every start-time bucket, including
                // both endpoints of the allowed [front, back] range.
                std::vector<Time> ts = {0.25, 0.5, 0.80, 1.34, 1.35, 2.7, 4.39, 4.40};
                emitArray("eval_times", std::vector<Real>(ts.begin(), ts.end()), 4, true);
                Matrix volGrid(ts.size(), kSize);
                Matrix volScalarGrid(ts.size(), kSize);
                for (Size k = 0; k < ts.size(); ++k) {
                    Array v = m.volatility(ts[k]);
                    for (Size i = 0; i < kSize; ++i) {
                        volGrid[k][i] = v[i];
                        // volatility(i, t) has no bounds check on i - ti; only
                        // probe the indices the vector form also fills.
                        volScalarGrid[k][i] = v[i] != 0.0 ? m.volatility(i, ts[k], Array()) : 0.0;
                    }
                }
                emitMatrix("volatility_vector", volGrid, 4, true);
                emitMatrix("volatility_scalar_where_nonzero", volScalarGrid, 4, true);
            }
            // integratedVariance is NOT implemented -> base class QL_FAIL
            std::cout << "    \"integrated_variance_raises\": ";
            try {
                m.integratedVariance(0, 0, 1.0, Array());
                std::cout << "false\n";
            } catch (const std::exception&) {
                std::cout << "true\n";
            }
            std::cout << "  },\n";
        }

        // ============================================================
        // LmConstWrapperVolatilityModel / LmConstWrapperCorrelationModel
        // ============================================================
        {
            LmConstWrapperVolatilityModel wrapVol(linExpVol);
            std::cout << "  \"lm_const_wrapper_volatility\": {\n";
            emitInt("size", static_cast<long>(wrapVol.size()), 4, true);
            emitInt("n_params", static_cast<long>(wrapVol.params().size()), 4, true);
            Matrix volGrid(kEvalTimes.size(), kSize);
            for (Size k = 0; k < kEvalTimes.size(); ++k) {
                Array v = wrapVol.volatility(kEvalTimes[k]);
                for (Size i = 0; i < kSize; ++i)
                    volGrid[k][i] = v[i];
            }
            emitMatrix("volatility_vector", volGrid, 4, true);
            Matrix scalarGrid(kEvalTimes.size(), kSize);
            for (Size k = 0; k < kEvalTimes.size(); ++k)
                for (Size i = 0; i < kSize; ++i)
                    scalarGrid[k][i] = wrapVol.volatility(i, kEvalTimes[k], Array());
            emitMatrix("volatility_scalar", scalarGrid, 4, true);
            Matrix iv(kSize, kSize);
            for (Size i = 0; i < kSize; ++i)
                for (Size j = 0; j < kSize; ++j)
                    iv[i][j] = wrapVol.integratedVariance(i, j, 1.7, Array());
            emitMatrix("integrated_variance_u1_7", iv, 4, false);
            std::cout << "  },\n";

            LmConstWrapperCorrelationModel wrapCorr(expCorr);
            std::cout << "  \"lm_const_wrapper_correlation\": {\n";
            emitInt("size", static_cast<long>(wrapCorr.size()), 4, true);
            emitInt("factors", static_cast<long>(wrapCorr.factors()), 4, true);
            emitInt("n_params", static_cast<long>(wrapCorr.params().size()), 4, true);
            emitBool("is_time_independent", wrapCorr.isTimeIndependent(), 4, true);
            emitMatrix("correlation_t0", wrapCorr.correlation(0.0), 4, true);
            emitMatrix("pseudo_sqrt_t0", wrapCorr.pseudoSqrt(0.0), 4, true);
            emitScalar("correlation_scalar_2_5_t1_1", wrapCorr.correlation(2, 5, 1.1, Array()), 4, false);
            std::cout << "  },\n";
        }

        // ============================================================
        // LfmCovarianceProxy — analytic branch
        // ============================================================
        {
            ext::shared_ptr<LmCorrelationModel> corr(
                new LmExponentialCorrelationModel(kSize, expCorrRho));
            LfmCovarianceProxy proxy(linExpVol, corr);

            std::cout << "  \"lfm_covariance_proxy\": {\n";
            emitInt("size", static_cast<long>(proxy.size()), 4, true);
            emitInt("factors", static_cast<long>(proxy.factors()), 4, true);
            emitMatrix("diffusion_t0_37", proxy.diffusion(0.37), 4, true);
            emitMatrix("diffusion_t2_55", proxy.diffusion(2.55), 4, true);
            emitMatrix("covariance_t0_37", proxy.covariance(0.37), 4, true);
            emitMatrix("covariance_t2_55", proxy.covariance(2.55), 4, true);
            {
                // analytic fast path: corr is time-independent AND the vol
                // model implements integratedVariance
                Matrix ic(kSize, kSize);
                for (Size i = 0; i < kSize; ++i)
                    for (Size j = 0; j < kSize; ++j)
                        ic[i][j] = proxy.integratedCovariance(i, j, 2.3, Array());
                emitMatrix("integrated_covariance_analytic_t2_3", ic, 4, false);
            }
            std::cout << "  },\n";
        }

        // ============================================================
        // LfmCovarianceProxy — numerical fallback branch, and
        // LfmCovarianceParameterization::integratedCovariance
        // ============================================================
        {
            // LmFixedVolatilityModel has no integratedVariance: the C++
            // try/catch in LfmCovarianceProxy::integratedCovariance swallows
            // the QL_FAIL and drops into GaussKronrodAdaptive. Size 3 keeps
            // the 64-segment double loop affordable on the Python side.
            // The numerical routines integrate from 0, and
            // LmFixedVolatilityModel rejects t outside [front, back], so the
            // start-time grid must begin at 0 here.
            const Size n = 3;
            const std::vector<Time> startTimes = {0.0, 0.55, 1.10};
            Array vols(n);
            vols[0] = 0.14;
            vols[1] = 0.21;
            vols[2] = 0.17;
            ext::shared_ptr<LmVolatilityModel> fixedVol(
                new LmFixedVolatilityModel(vols, startTimes));
            ext::shared_ptr<LmCorrelationModel> corr(
                new LmExponentialCorrelationModel(n, 0.31));
            LfmCovarianceProxy proxy(fixedVol, corr);

            std::cout << "  \"lfm_covariance_proxy_numeric\": {\n";
            emitInt("size", static_cast<long>(proxy.size()), 4, true);
            emitArray("start_times", std::vector<Real>(startTimes.begin(), startTimes.end()), 4,
                      true);
            emitArray("volatilities", vols, 4, true);
            {
                // t = 0.5 stays inside the FIRST start-time bucket, so ti == 0
                // for the whole integration domain and every (i, j) is safe.
                //
                // Beyond the first bucket ti >= 1, and C++
                // LmFixedVolatilityModel::volatility(Size i, Time t, ...)
                // evaluates volatilities_[i - ti] with UNSIGNED Size and no
                // bounds check: for i < ti that is an out-of-bounds read, i.e.
                // undefined behaviour. (The vector-valued overload is fine —
                // it zero-fills those entries — which is why the base-class
                // integratedCovariance below can safely use a larger t.)
                // Only the pairs with min(i, j) >= 1 are therefore emitted at
                // t = 1.05, where the integrand really is piecewise.
                Matrix ic(n, n);
                for (Size i = 0; i < n; ++i)
                    for (Size j = 0; j < n; ++j)
                        ic[i][j] = proxy.integratedCovariance(i, j, 0.5, Array());
                emitMatrix("integrated_covariance_numeric_t0_5", ic, 4, true);

                Matrix icHigh(n - 1, n - 1);
                for (Size i = 1; i < n; ++i)
                    for (Size j = 1; j < n; ++j)
                        icHigh[i - 1][j - 1] = proxy.integratedCovariance(i, j, 1.05, Array());
                emitMatrix("integrated_covariance_numeric_t1_05_upper_block", icHigh, 4, true);
            }
            // The base-class 64-segment double loop over diffusion(t).
            emitMatrix("base_integrated_covariance_t1_05",
                       proxy.LfmCovarianceParameterization::integratedCovariance(1.05), 4, false);
            std::cout << "  },\n";
        }

        // ============================================================
        // LiborForwardModelProcess
        // ============================================================
        {
            ext::shared_ptr<IborIndex> index = makeIndex6M();
            const Size n = 6;
            ext::shared_ptr<LiborForwardModelProcess> process(
                new LiborForwardModelProcess(n, index));

            ext::shared_ptr<LmVolatilityModel> vol(new LmLinearExponentialVolatilityModel(
                process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001));
            ext::shared_ptr<LmCorrelationModel> corr(new LmExponentialCorrelationModel(n, 0.5));
            process->setCovarParam(
                ext::shared_ptr<LfmCovarianceParameterization>(new LfmCovarianceProxy(vol, corr)));

            std::cout << "  \"lfm_process\": {\n";
            emitInt("size", static_cast<long>(process->size()), 4, true);
            emitInt("factors", static_cast<long>(process->factors()), 4, true);
            emitArray("initial_values", process->initialValues(), 4, true);
            emitArray("fixing_times",
                      std::vector<Real>(process->fixingTimes().begin(),
                                        process->fixingTimes().end()),
                      4, true);
            {
                std::vector<long> serials;
                for (const auto& d : process->fixingDates())
                    serials.push_back(d.serialNumber());
                emitIntArray("fixing_date_serials", serials, 4, true);
            }
            emitArray("accrual_start_times",
                      std::vector<Real>(process->accrualStartTimes().begin(),
                                        process->accrualStartTimes().end()),
                      4, true);
            emitArray("accrual_end_times",
                      std::vector<Real>(process->accrualEndTimes().begin(),
                                        process->accrualEndTimes().end()),
                      4, true);
            {
                // nextIndexReset is a strict upper_bound: at t exactly on a
                // fixing time it must already return the NEXT index.
                std::vector<long> resets;
                std::vector<Real> queried;
                const std::vector<Time>& ft = process->fixingTimes();
                for (Size i = 0; i < ft.size(); ++i) {
                    queried.push_back(ft[i] - 1e-6);
                    resets.push_back(static_cast<long>(process->nextIndexReset(ft[i] - 1e-6)));
                    queried.push_back(ft[i]);
                    resets.push_back(static_cast<long>(process->nextIndexReset(ft[i])));
                    queried.push_back(ft[i] + 1e-6);
                    resets.push_back(static_cast<long>(process->nextIndexReset(ft[i] + 1e-6)));
                }
                emitArray("next_index_reset_times", queried, 4, true);
                emitIntArray("next_index_reset", resets, 4, true);
            }
            {
                std::vector<Rate> rates(n);
                for (Size i = 0; i < n; ++i)
                    rates[i] = 0.03 + 0.004 * Real(i);
                emitArray("discount_bond_rates", std::vector<Real>(rates.begin(), rates.end()), 4,
                          true);
                std::vector<DiscountFactor> dfs = process->discountBond(rates);
                emitArray("discount_bond", std::vector<Real>(dfs.begin(), dfs.end()), 4, true);
            }
            {
                Array x(n);
                for (Size i = 0; i < n; ++i)
                    x[i] = 0.035 + 0.003 * Real(i);
                emitArray("state_x", x, 4, true);
                emitArray("drift_t0_9", process->drift(0.9, x), 4, true);
                emitArray("drift_t2_4", process->drift(2.4, x), 4, true);
                emitMatrix("diffusion_t0_9", process->diffusion(0.9, x), 4, true);
                // covariance(t0, x0, dt): dt scales the parameterization
                emitMatrix("covariance_t0_9_dt0_25", process->covariance(0.9, x, 0.25), 4, true);
                Array dx(n);
                for (Size i = 0; i < n; ++i)
                    dx[i] = 0.01 * Real(i) - 0.02;
                emitArray("apply_dx", dx, 4, true);
                emitArray("apply", process->apply(x, dx), 4, true);
                Array dw(process->factors());
                for (Size i = 0; i < process->factors(); ++i)
                    dw[i] = 0.3 - 0.11 * Real(i);
                emitArray("evolve_dw", dw, 4, true);
                emitArray("evolve_t0_9_dt0_25", process->evolve(0.9, x, 0.25, dw), 4, true);
            }
            {
                // cashFlows at a NON-UNIT notional
                Leg flows = process->cashFlows(2.5);
                std::vector<long> serials;
                std::vector<Real> amounts;
                std::vector<Real> nominals;
                for (const auto& cf : flows) {
                    serials.push_back(cf->date().serialNumber());
                    amounts.push_back(cf->amount());
                    auto cpn = ext::dynamic_pointer_cast<Coupon>(cf);
                    nominals.push_back(cpn != nullptr ? cpn->nominal() : 0.0);
                }
                emitIntArray("cashflow_date_serials", serials, 4, true);
                emitArray("cashflow_amounts", amounts, 4, true);
                emitArray("cashflow_nominals", nominals, 4, false);
            }
            std::cout << "  },\n";
        }

        // ============================================================
        // LfmHullWhiteParameterization
        // ============================================================
        {
            ext::shared_ptr<IborIndex> index1y = makeIndex1Y();
            ext::shared_ptr<LiborForwardModelProcess> process(
                new LiborForwardModelProcess(kHwLen, index1y));
            ext::shared_ptr<CapletVarianceCurve> capVol =
                makeCapVolCurve(Settings::instance().evaluationDate());

            // 1-factor: no correlation matrix, factors defaults to 1
            LfmHullWhiteParameterization hw1(process, capVol);

            std::cout << "  \"lfm_hull_white_1factor\": {\n";
            emitInt("size", static_cast<long>(hw1.size()), 4, true);
            emitInt("factors", static_cast<long>(hw1.factors()), 4, true);
            emitArray("process_fixing_times",
                      std::vector<Real>(process->fixingTimes().begin(),
                                        process->fixingTimes().end()),
                      4, true);
            {
                std::vector<long> serials;
                for (const auto& d : process->fixingDates())
                    serials.push_back(d.serialNumber());
                emitIntArray("process_fixing_date_serials", serials, 4, true);
            }
            {
                // testLambdaBootstrapping: sqrt of the diagonal of
                // covariance(0) reproduces the bootstrapped lambdas.
                Matrix cov0 = hw1.covariance(0.0);
                std::vector<Real> lambdas;
                for (Size i = 1; i < kHwLen; ++i)
                    lambdas.push_back(std::sqrt(cov0[i][i]));
                emitArray("lambdas", lambdas, 4, true);
            }
            emitMatrix("diffusion_t0", hw1.diffusion(0.0), 4, true);
            emitMatrix("diffusion_t2_5", hw1.diffusion(2.5), 4, true);
            emitMatrix("covariance_t0", hw1.covariance(0.0), 4, true);
            emitMatrix("covariance_t2_5", hw1.covariance(2.5), 4, true);
            emitMatrix("integrated_covariance_t3_3", hw1.integratedCovariance(3.3), 4, false);
            std::cout << "  },\n";

            // 3-factor: Hull & White factor loadings from the C++ test-suite
            Real compValues[] = {0.85549771,  0.46707264,  0.22353259,  0.91915359,  0.37716089,
                                 0.11360610,  0.96438280,  0.26413316,  -0.01412414, 0.97939148,
                                 0.13492952,  -0.15028753, 0.95970595,  -0.00000000, -0.28100621,
                                 0.97939148,  -0.13492952, -0.15028753, 0.96438280,  -0.26413316,
                                 -0.01412414, 0.91915359,  -0.37716089, 0.11360610,  0.85549771,
                                 -0.46707264, 0.22353259};
            Matrix volaComp(9, 3);
            std::copy(compValues, compValues + 27, volaComp.begin());
            Matrix correlation = volaComp * transpose(volaComp);

            LfmHullWhiteParameterization hw3(process, capVol, correlation, 3);
            std::cout << "  \"lfm_hull_white_3factor\": {\n";
            emitInt("size", static_cast<long>(hw3.size()), 4, true);
            emitInt("factors", static_cast<long>(hw3.factors()), 4, true);
            emitMatrix("correlation_input", correlation, 4, true);
            emitMatrix("diffusion_t0", hw3.diffusion(0.0), 4, true);
            emitMatrix("diffusion_t2_5", hw3.diffusion(2.5), 4, true);
            emitMatrix("covariance_t0", hw3.covariance(0.0), 4, true);
            emitMatrix("integrated_covariance_t3_3", hw3.integratedCovariance(3.3), 4, false);
            std::cout << "  }\n";
        }

        std::cout << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
}
