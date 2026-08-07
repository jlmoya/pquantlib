// migration-harness/cpp/probes/v143_ts_abcd/probe.cpp
//
// Reference values from C++ QuantLib v1.43 for:
//
//   * AbcdSquared                      (ql/termstructures/volatility/abcd.hpp:93)
//   * AbcdCalibration::AbcdError       (ql/termstructures/volatility/abcdcalibration.hpp:44)
//   * AbcdCalibration::AbcdParametersTransformation
//                                      (ql/termstructures/volatility/abcdcalibration.hpp:69)
//   * AbcdCalibration::error/maxError/errors/k/value
//                                      (ql/termstructures/volatility/abcdcalibration.cpp:179-205)
//   * XABRInterpolationImpl::interpolationError / interpolationMaxError
//                                      (ql/math/interpolations/xabrinterpolation.hpp:245-285)
//     reached through SABRInterpolation::rmsError() / maxError()
//                                      (ql/math/interpolations/sabrinterpolation.hpp:183-184)
//
// Emits ONE JSON object on stdout and nothing else.
//
// -----------------------------------------------------------------------------
// WHY TWO OF THESE ARE TRANSCRIBED RATHER THAN CONSTRUCTED
// -----------------------------------------------------------------------------
// `AbcdError` and `AbcdParametersTransformation` are declared in the PRIVATE
// section of `class AbcdCalibration` (abcdcalibration.hpp:42 opens `private:`;
// the two classes follow at :44 and :69, and `public:` only reopens at :80).
// No translation unit outside the class can name them, so a probe cannot
// instantiate them directly.
//
// This probe therefore does two things:
//
//   (a) it TRANSCRIBES the four-line bodies of `direct` / `inverse` verbatim
//       from abcdcalibration.cpp:37-52 into file-local functions, and evaluates
//       them with C++'s own std::exp / std::log on C++'s own Array. That pins
//       the arithmetic under the reference toolchain, and catches any
//       transcription error in the Python port; it does NOT independently
//       re-derive the formula, and that limitation is deliberate and stated.
//
//   (b) it measures `AbcdError` END TO END through the public API. `AbcdError::
//       value(x)` is exactly {a,b,c,d} := direct(x); return calibration.error();
//       and `AbcdError::values(x)` the same with errors(). Both `error()` and
//       `errors()` ARE public (abcdcalibration.hpp:97-101), and an
//       AbcdCalibration constructed with all four parameters fixed leaves
//       a_,b_,c_,d_ exactly as passed (compute() short-circuits at
//       abcdcalibration.cpp:117-122). So constructing the calibration at
//       direct(x) and reading error()/errors() reproduces AbcdError's output
//       through code the probe did not write.
//
// -----------------------------------------------------------------------------
// THE DIAGNOSTIC-DENOMINATOR FAMILY
// -----------------------------------------------------------------------------
// Both `AbcdCalibration::error()` (abcdcalibration.cpp:179-187) and
// `XABRInterpolationImpl::interpolationError()` (xabrinterpolation.hpp:270-274)
// compute
//
//     sqrt( n * SUM_i w_i * e_i^2 / (n - 1) )
//
// with weights that default to a UNIFORM 1/n (abcdcalibration.cpp:72,
// xabrinterpolation.hpp:178-179), so in the unweighted case the value reduces
// to the SAMPLE standard deviation sqrt(SUM e_i^2 / (n-1)).  Both `maxError()`
// implementations are UNWEIGHTED: max |value(x_i) - y_i|
// (abcdcalibration.cpp:189-196, xabrinterpolation.hpp:276-285).
//
// Both are pinned here for n = 2, 3 and 6 so the (n-1) denominator is separable
// from the leading n, and in vega-weighted and unweighted flavours so the
// weight enters the RMS but not the max.
//
// The SABR case is constructed with ALL FOUR parameters fixed. C++ then
// short-circuits the optimisation entirely (xabrinterpolation.hpp:161-168:
// "there is nothing to optimize") and sets error_/maxError_ straight from the
// closed-form expressions. That keeps every number below independent of
// LevenbergMarquardt, which matters because PQuantLib's LevenbergMarquardt is a
// scipy delegation whose C++-parity tests are xfailed.
//
// -----------------------------------------------------------------------------
// DETERMINISM
// -----------------------------------------------------------------------------
// No evaluation date is set: every quantity here is a pure function of its
// arguments.  Nothing is read before it is written; the SABR interpolation is
// update()d before rmsError()/maxError() are queried.

#include <ql/version.hpp>

#include <ql/math/array.hpp>
#include <ql/math/interpolations/sabrinterpolation.hpp>
#include <ql/termstructures/volatility/abcd.hpp>
#include <ql/termstructures/volatility/abcdcalibration.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Verbatim transcription of AbcdCalibration::AbcdParametersTransformation,
// abcdcalibration.cpp:37-52.  See the header comment for why this is copied
// rather than constructed.
// ---------------------------------------------------------------------------
Array abcdTransformDirect(const Array& x) {
    Array y(4);
    y[1] = x[1];
    y[2] = std::exp(x[2]);
    y[3] = std::exp(x[3]);
    y[0] = std::exp(x[0]) - y[3];
    return y;
}

Array abcdTransformInverse(const Array& x) {
    Array y(4);
    y[1] = x[1];
    y[2] = std::log(x[2]);
    y[3] = std::log(x[3]);
    y[0] = std::log(x[0] + x[3]);
    return y;
}

std::string num(Real x) {
    if (x == Null<Real>())
        return "null";
    if (std::isnan(x))
        return "\"nan\"";
    if (std::isinf(x))
        return x > 0 ? "\"inf\"" : "\"-inf\"";
    std::ostringstream os;
    os << std::setprecision(17) << x;
    return os.str();
}

std::string arr(const std::vector<Real>& v) {
    std::ostringstream os;
    os << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i)
            os << ", ";
        os << num(v[i]);
    }
    os << "]";
    return os.str();
}

std::string arr(const Array& v) {
    return arr(std::vector<Real>(v.begin(), v.end()));
}

} // namespace

int main() {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"quantlib_version\": \"" << QL_VERSION << "\",\n";

    // =====================================================================
    // PART 1 -- AbcdSquared (abcd.hpp:93, abcd.cpp:99-105)
    //
    //   AbcdSquared(a,b,c,d,T,S)(t) == AbcdFunction(a,b,c,d).covariance(t,T,S)
    //                               == f(T-t) * f(S-t)          (abcd.cpp:43-45)
    //
    // Note it is NOT the square of f(t): the name refers to the product of two
    // evaluations, and T != S makes that visible.  Both the AbcdSquared value
    // and the AbcdFunction::covariance / instantaneousCovariance it delegates
    // to are emitted so a port cannot satisfy one and miss the other.
    // =====================================================================
    {
        const Real a = -0.06, b = 0.17, c = 0.54, d = 0.17;
        const Real Ts[] = {2.0, 5.0, 5.0};
        const Real Ss[] = {2.0, 5.0, 7.5};
        const Real ts[] = {0.0, 0.25, 1.0, 1.9999, 2.0, 3.5, 5.0};

        out << "  \"abcd_squared\": [\n";
        bool first = true;
        for (Size k = 0; k < 3; ++k) {
            AbcdSquared sq(a, b, c, d, Ts[k], Ss[k]);
            AbcdFunction fn(a, b, c, d);
            for (Real t : ts) {
                if (!first)
                    out << ",\n";
                first = false;
                out << "    {\"a\": " << num(a) << ", \"b\": " << num(b)
                    << ", \"c\": " << num(c) << ", \"d\": " << num(d)
                    << ", \"T\": " << num(Ts[k]) << ", \"S\": " << num(Ss[k])
                    << ", \"t\": " << num(t)
                    << ", \"value\": " << num(sq(t))
                    << ", \"covariance\": " << num(fn.covariance(t, Ts[k], Ss[k]))
                    << ", \"instantaneousCovariance\": "
                    << num(fn.instantaneousCovariance(t, Ts[k], Ss[k])) << "}";
            }
        }
        out << "\n  ],\n";
    }

    // =====================================================================
    // PART 2 -- AbcdParametersTransformation (abcdcalibration.cpp:37-52)
    //
    // direct:  unconstrained x -> constrained (a,b,c,d)
    //            b = x1;  c = exp(x2);  d = exp(x3);  a = exp(x0) - d
    //          which is exactly the AbcdMathFunction feasibility set
    //          c > 0, d > 0, a + d > 0.
    // inverse: the other way, and NOTE the asymmetry: inverse reads x[3] (the
    //          CONSTRAINED d) in the a-slot expression log(x[0] + x[3]), so
    //          inverse(direct(u)) == u only because direct wrote d into y[3]
    //          before computing y[0].  A port that reorders those two lines
    //          still round-trips for some inputs and not others; the
    //          round-trip residual is emitted per case to catch that.
    // =====================================================================
    {
        // Unconstrained points, deliberately including negatives and a large
        // magnitude so exp/log cannot be confused with identity.
        const Real xs[][4] = {
            {0.0, 0.0, 0.0, 0.0},
            {-1.0, 0.17, -0.61618613602, -1.77195684193},
            {2.5, -0.4, 1.25, -3.0},
            {-0.30538, 0.17, 0.5, 0.25},
        };
        out << "  \"abcd_transformation\": [\n";
        for (Size k = 0; k < 4; ++k) {
            if (k)
                out << ",\n";
            Array x(4);
            for (Size i = 0; i < 4; ++i)
                x[i] = xs[k][i];
            const Array y = abcdTransformDirect(x);
            const Array back = abcdTransformInverse(y);
            std::vector<Real> resid(4);
            for (Size i = 0; i < 4; ++i)
                resid[i] = back[i] - x[i];
            out << "    {\"x\": " << arr(x) << ", \"direct\": " << arr(y)
                << ", \"inverse_of_direct\": " << arr(back)
                << ", \"roundtrip_residual\": " << arr(resid) << "}";
        }
        out << "\n  ],\n";

        // inverse() applied to CONSTRAINED points, i.e. the way compute() uses
        // it on the initial guess (abcdcalibration.cpp:132).
        const Real cs[][4] = {
            {-0.06, 0.17, 0.54, 0.17},
            {0.10, -0.02, 1.00, 0.05},
            {0.00, 0.00, 1.00, 1.00},
        };
        out << "  \"abcd_transformation_inverse\": [\n";
        for (Size k = 0; k < 3; ++k) {
            if (k)
                out << ",\n";
            Array x(4);
            for (Size i = 0; i < 4; ++i)
                x[i] = cs[k][i];
            const Array y = abcdTransformInverse(x);
            const Array back = abcdTransformDirect(y);
            std::vector<Real> resid(4);
            for (Size i = 0; i < 4; ++i)
                resid[i] = back[i] - x[i];
            out << "    {\"x\": " << arr(x) << ", \"inverse\": " << arr(y)
                << ", \"direct_of_inverse\": " << arr(back)
                << ", \"roundtrip_residual\": " << arr(resid) << "}";
        }
        out << "\n  ],\n";
    }

    // =====================================================================
    // PART 3 -- AbcdCalibration diagnostics, and AbcdError through them.
    //
    // Every case fixes all four parameters, so compute() short-circuits
    // (abcdcalibration.cpp:117-122) and no optimiser runs.  `error()`,
    // `maxError()` and `errors()` are then pure functions of (a,b,c,d),
    // times, blackVols and weights.
    //
    // `abcd_error_via_public_api` additionally feeds an UNCONSTRAINED x
    // through direct() first, which is precisely AbcdError::value/values
    // (abcdcalibration.hpp:47-63).
    //
    // n takes the values 2, 3 and 6: with the leading n and the trailing
    // (n-1) both present, three different n are needed to separate them from
    // any other denominator convention.
    // =====================================================================
    {
        const std::vector<Real> times6 = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
        const std::vector<Real> vols6 = {0.222, 0.208, 0.196, 0.181, 0.166, 0.155};

        struct Case {
            const char* name;
            std::vector<Real> t;
            std::vector<Real> v;
            bool vegaWeighted;
            Real a, b, c, d;
        };
        const std::vector<Case> cases = {
            {"n6_unweighted", times6, vols6, false, -0.06, 0.17, 0.54, 0.17},
            {"n6_vegaweighted", times6, vols6, true, -0.06, 0.17, 0.54, 0.17},
            {"n3_unweighted",
             {0.5, 1.0, 2.0},
             {0.208, 0.196, 0.181},
             false,
             -0.06,
             0.17,
             0.54,
             0.17},
            {"n3_vegaweighted",
             {0.5, 1.0, 2.0},
             {0.208, 0.196, 0.181},
             true,
             -0.06,
             0.17,
             0.54,
             0.17},
            {"n2_unweighted", {1.0, 2.0}, {0.196, 0.181}, false, -0.06, 0.17, 0.54, 0.17},
            // A parameter set that reproduces the data far better, so the
            // residuals are small and the (n-1) scaling is still resolvable.
            {"n6_near_fit", times6, vols6, false, -0.0592, 0.1524, 0.5000, 0.1660},
        };

        out << "  \"abcd_calibration\": {\n";
        for (Size k = 0; k < cases.size(); ++k) {
            const Case& cs = cases[k];
            if (k)
                out << ",\n";
            AbcdCalibration calib(cs.t, cs.v, cs.a, cs.b, cs.c, cs.d, true, true, true, true,
                                  cs.vegaWeighted);
            calib.compute();
            std::vector<Real> vals;
            vals.reserve(cs.t.size());
            for (Real t : cs.t)
                vals.push_back(calib.value(t));
            out << "    \"" << cs.name << "\": {"
                << "\"times\": " << arr(cs.t) << ", \"blackVols\": " << arr(cs.v)
                << ", \"vegaWeighted\": " << (cs.vegaWeighted ? "true" : "false")
                << ", \"a\": " << num(calib.a()) << ", \"b\": " << num(calib.b())
                << ", \"c\": " << num(calib.c()) << ", \"d\": " << num(calib.d())
                << ", \"values\": " << arr(vals) << ", \"error\": " << num(calib.error())
                << ", \"maxError\": " << num(calib.maxError())
                << ", \"errors\": " << arr(calib.errors())
                << ", \"k\": " << arr(calib.k(cs.t, cs.v)) << "}";
        }
        out << "\n  },\n";

        // AbcdError::value(x) / values(x): direct(x) then error()/errors().
        const Real xs[][4] = {
            {-0.30538, 0.17, -0.61618613602, -1.77195684193},
            {0.0, 0.10, -0.5, -2.0},
            {-1.0, -0.05, 0.25, -1.5},
        };
        out << "  \"abcd_error_via_public_api\": [\n";
        for (Size k = 0; k < 3; ++k) {
            if (k)
                out << ",\n";
            Array x(4);
            for (Size i = 0; i < 4; ++i)
                x[i] = xs[k][i];
            const Array p = abcdTransformDirect(x);
            AbcdCalibration calib(times6, vols6, p[0], p[1], p[2], p[3], true, true, true, true,
                                  false);
            calib.compute();
            out << "    {\"x\": " << arr(x) << ", \"direct\": " << arr(p)
                << ", \"value\": " << num(calib.error())
                << ", \"values\": " << arr(calib.errors()) << "}";
        }
        out << "\n  ],\n";
    }

    // =====================================================================
    // PART 4 -- XABRInterpolationImpl::interpolationError / MaxError,
    //           reached through SABRInterpolation::rmsError() / maxError().
    //
    // All four SABR parameters are fixed, so xabrinterpolation.hpp:161-168
    // short-circuits and no optimiser runs.  weights_ default to a uniform
    // 1/n (xabrinterpolation.hpp:178-179); vegaWeighted recomputes them as
    // blackFormulaStdDevDerivative normalised to sum 1 (hpp:142-159 and
    // sabrinterpolation.hpp:131-135).
    //
    // Emitted per case: the model volatilities (so the port's SABR formula is
    // separable from its error formula), the raw errors, rmsError and maxError.
    // =====================================================================
    {
        struct SCase {
            const char* name;
            std::vector<Real> strikes;
            std::vector<Real> vols;
            Real t, forward, alpha, beta, nu, rho;
            bool vegaWeighted;
        };

        const std::vector<Real> k6 = {0.02, 0.03, 0.04, 0.05, 0.06, 0.07};
        const std::vector<Real> v6 = {0.31, 0.27, 0.25, 0.245, 0.25, 0.26};

        const std::vector<SCase> cases = {
            {"sabr_n6_unweighted", k6, v6, 5.0, 0.04, 0.06, 0.5, 0.4, -0.1, false},
            {"sabr_n6_vegaweighted", k6, v6, 5.0, 0.04, 0.06, 0.5, 0.4, -0.1, true},
            {"sabr_n3_unweighted",
             {0.03, 0.04, 0.05},
             {0.27, 0.25, 0.245},
             5.0,
             0.04,
             0.06,
             0.5,
             0.4,
             -0.1,
             false},
            {"sabr_n3_vegaweighted",
             {0.03, 0.04, 0.05},
             {0.27, 0.25, 0.245},
             5.0,
             0.04,
             0.06,
             0.5,
             0.4,
             -0.1,
             true},
            {"sabr_n2_unweighted",
             {0.03, 0.05},
             {0.27, 0.245},
             5.0,
             0.04,
             0.06,
             0.5,
             0.4,
             -0.1,
             false},
        };

        out << "  \"sabr_interpolation_error\": {\n";
        for (Size k = 0; k < cases.size(); ++k) {
            const SCase& cs = cases[k];
            if (k)
                out << ",\n";
            std::vector<Real> x = cs.strikes;
            std::vector<Real> y = cs.vols;
            SABRInterpolation interp(x.begin(), x.end(), y.begin(), cs.t, cs.forward, cs.alpha,
                                     cs.beta, cs.nu, cs.rho,
                                     /*alphaIsFixed*/ true, /*betaIsFixed*/ true,
                                     /*nuIsFixed*/ true, /*rhoIsFixed*/ true, cs.vegaWeighted);
            interp.update();
            std::vector<Real> model, errs;
            model.reserve(x.size());
            errs.reserve(x.size());
            for (Size i = 0; i < x.size(); ++i) {
                const Real m = interp(x[i], true);
                model.push_back(m);
                errs.push_back(m - y[i]);
            }
            out << "    \"" << cs.name << "\": {"
                << "\"strikes\": " << arr(cs.strikes) << ", \"vols\": " << arr(cs.vols)
                << ", \"t\": " << num(cs.t) << ", \"forward\": " << num(cs.forward)
                << ", \"alpha\": " << num(cs.alpha) << ", \"beta\": " << num(cs.beta)
                << ", \"nu\": " << num(cs.nu) << ", \"rho\": " << num(cs.rho)
                << ", \"vegaWeighted\": " << (cs.vegaWeighted ? "true" : "false")
                << ", \"model\": " << arr(model) << ", \"errors\": " << arr(errs)
                << ", \"rmsError\": " << num(interp.rmsError())
                << ", \"maxError\": " << num(interp.maxError()) << "}";
        }
        out << "\n  }\n";
    }

    out << "}\n";
    std::cout << out.str();
    return 0;
}
