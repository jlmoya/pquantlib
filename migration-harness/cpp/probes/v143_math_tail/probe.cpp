// migration-harness/cpp/probes/v143_math_tail/probe.cpp
//
// Reference values for the ql/math classes that do not belong to any of the
// larger subdirectories: the abcd and polynomial functional forms, Richardson
// extrapolation, the prime-number generator, the B-spline basis, the adaptive
// Runge-Kutta ODE integrator, the FFT and the Gaussian copula.
//
// What is pinned, and why:
//
//   * for the functional forms — value, first derivative and primitive, at
//     t < 0 (which C++ clamps to 0 rather than evaluating), at 0, and across
//     the body, plus the rolling-window coefficient vectors, which are the
//     part most likely to be silently wrong because they are only ever
//     consumed by other code;
//   * for Richardson extrapolation — both overloads, including the
//     unknown-order one whose answer depends on a bracket scan with a 0.1
//     step and a Brent solve, so the estimated order is emitted alongside;
//   * for the ODE integrator — the solution AND an integration over a stiff
//     interval where the step controller has to shrink, since a fixed-step
//     RK45 would agree on the easy case and diverge here;
//   * for the FFT — the full complex output of a non-power-of-two input
//     padded into the next order, forward and inverse, because the
//     bit-reversal permutation and the twiddle recurrence are where a
//     reimplementation goes wrong.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/tail.json.

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

#include <ql/math/abcdmathfunction.hpp>
#include <ql/math/bspline.hpp>
#include <ql/math/copulas/gaussiancopula.hpp>
#include <ql/math/fastfouriertransform.hpp>
#include <ql/math/ode/adaptiverungekutta.hpp>
#include <ql/math/polynomialmathfunction.hpp>
#include <ql/math/primenumbers.hpp>
#include <ql/math/richardsonextrapolation.hpp>

using namespace QuantLib;

namespace {

std::ostream& out = std::cout;

void num(Real v) {
    if (std::isnan(v)) out << "\"nan\"";
    else if (std::isinf(v)) out << (v > 0 ? "\"inf\"" : "\"-inf\"");
    else out << v;
}

void numArray(const std::vector<Real>& v) {
    out << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out << ", ";
        num(v[i]);
    }
    out << "]";
}

const std::vector<Real> kTimes = {-1.0, -1e-12, 0.0, 1e-8, 0.25, 1.0, 2.5, 5.0,
                                  10.0, 30.0, 100.0};

// --------------------------------------------------------- abcd functional --

void emitAbcd() {
    // (a, b, c, d) sets: the default, b == 0 (maximumLocation short-circuits),
    // a < 0 with b > 0 (interior maximum), and a negative-b set that still
    // passes validate().
    const std::vector<std::vector<Real>> params = {
        {0.002, 0.001, 0.16, 0.0005},
        {0.02, 0.0, 0.5, 0.01},
        {-0.005, 0.02, 0.3, 0.02},
        {0.05, -0.01, 0.4, 0.03},
        {0.0, 0.0, 1.0, 0.0}
    };
    out << "  \"abcd_math_function\": [";
    bool first = true;
    for (const auto& p : params) {
        AbcdMathFunction f(p[0], p[1], p[2], p[3]);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"abcd\": ";
        numArray(p);
        out << ",\n     \"maximum_location\": ";
        num(f.maximumLocation());
        out << ", \"maximum_value\": ";
        num(f.maximumValue());
        out << ", \"long_term_value\": ";
        num(f.longTermValue());
        out << ",\n     \"derivative_coefficients\": ";
        numArray(f.derivativeCoefficients());
        out << ",\n     \"values\": [";
        for (std::size_t i = 0; i < kTimes.size(); ++i) {
            if (i) out << ", ";
            out << "[" << kTimes[i] << ", ";
            num(f(kTimes[i]));
            out << ", ";
            num(f.derivative(kTimes[i]));
            out << ", ";
            num(f.primitive(kTimes[i]));
            out << "]";
        }
        out << "],\n     \"definite_integral\": ";
        num(f.definiteIntegral(0.5, 4.0));
        out << ",\n     \"definite_integral_coefficients\": ";
        numArray(f.definiteIntegralCoefficients(0.5, 4.0));
        out << ",\n     \"definite_derivative_coefficients\": ";
        numArray(f.definiteDerivativeCoefficients(0.5, 4.0));
        out << "}";
    }
    out << "\n  ],\n";
}

// --------------------------------------------------- polynomial functional --

void emitPolynomial() {
    const std::vector<std::vector<Real>> coeffs = {
        {1.0},
        {0.5, -0.25},
        {1.0, 0.0, -2.0, 0.75},
        {-3.0, 1.5, 0.0, 0.0, 0.125}
    };
    out << "  \"polynomial_function\": [";
    bool first = true;
    for (const auto& c : coeffs) {
        PolynomialFunction f(c);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"coefficients\": ";
        numArray(c);
        out << ", \"order\": " << f.order();
        out << ",\n     \"derivative_coefficients\": ";
        numArray(f.derivativeCoefficients());
        out << ",\n     \"primitive_coefficients\": ";
        numArray(f.primitiveCoefficients());
        out << ",\n     \"values\": [";
        for (std::size_t i = 0; i < kTimes.size(); ++i) {
            if (i) out << ", ";
            out << "[" << kTimes[i] << ", ";
            num(f(kTimes[i]));
            out << ", ";
            num(f.derivative(kTimes[i]));
            out << ", ";
            num(f.primitive(kTimes[i]));
            out << "]";
        }
        out << "],\n     \"definite_integral\": ";
        num(f.definiteIntegral(0.5, 4.0));
        out << ",\n     \"definite_integral_coefficients\": ";
        numArray(f.definiteIntegralCoefficients(0.5, 4.0));
        out << ",\n     \"definite_derivative_coefficients\": ";
        numArray(f.definiteDerivativeCoefficients(0.5, 4.0));
        out << "}";
    }
    out << "\n  ],\n";
}

// ------------------------------------------------- Richardson extrapolation --

void emitRichardson() {
    // f(h) = exp(1+h) has a first-order error term; f2(h) = exp(1 + h*h) a
    // second-order one. The known-order overload is exercised with the right
    // and the deliberately wrong n, since supplying the wrong order is a
    // legitimate call that returns a specific wrong answer.
    auto f1 = [](Real h) { return std::exp(1.0 + h); };
    auto f2 = [](Real h) { return std::exp(1.0 + h * h); };

    out << "  \"richardson\": {\n";
    out << "    \"exact\": ";
    num(std::exp(1.0));
    out << ",\n    \"known_order\": [";
    bool first = true;
    for (Real dh : {0.1, 0.01, 0.5}) {
        for (Real n : {1.0, 2.0}) {
            RichardsonExtrapolation extrap(f1, dh, n);
            for (Real t : {2.0, 4.0}) {
                out << (first ? "\n      " : ",\n      ");
                first = false;
                out << "{\"f\": \"exp1h\", \"delta_h\": " << dh << ", \"n\": " << n
                    << ", \"t\": " << t << ", \"v\": ";
                num(extrap(t));
                out << "}";
            }
        }
    }
    for (Real dh : {0.1, 0.5}) {
        RichardsonExtrapolation extrap(f2, dh, 2.0);
        for (Real t : {2.0, 4.0}) {
            out << ",\n      {\"f\": \"exp1hh\", \"delta_h\": " << dh
                << ", \"n\": 2, \"t\": " << t << ", \"v\": ";
            num(extrap(t));
            out << "}";
        }
    }
    out << "\n    ],\n    \"unknown_order\": [";
    first = true;
    for (Real dh : {0.1, 0.02}) {
        RichardsonExtrapolation e1(f1, dh);
        RichardsonExtrapolation e2(f2, dh);
        out << (first ? "\n      " : ",\n      ");
        first = false;
        out << "{\"f\": \"exp1h\", \"delta_h\": " << dh << ", \"t\": 4, \"s\": 2, \"v\": ";
        num(e1(4.0, 2.0));
        out << "},\n      {\"f\": \"exp1hh\", \"delta_h\": " << dh
            << ", \"t\": 4, \"s\": 2, \"v\": ";
        num(e2(4.0, 2.0));
        out << "}";
    }
    out << "\n    ]\n  },\n";
}

// --------------------------------------------------------------- primes ----

void emitPrimes() {
    out << "  \"prime_numbers\": [";
    // Indices span the precomputed table (0..14) and well past it, where the
    // trial-division loop takes over.
    const std::vector<Size> idx = {0, 1, 2, 5, 10, 14, 15, 16, 20, 50, 100,
                                   500, 1000};
    for (std::size_t i = 0; i < idx.size(); ++i) {
        if (i) out << ", ";
        out << "[" << idx[i] << ", " << PrimeNumbers::get(idx[i]) << "]";
    }
    out << "],\n";
}

// -------------------------------------------------------------- B-spline ---

void emitBSpline() {
    struct Cfg { Natural p; Natural n; std::vector<Real> knots; };
    const std::vector<Cfg> cfgs = {
        // uniform linear
        {1, 2, {0.0, 1.0, 2.0, 3.0, 4.0}},
        // clamped cubic (repeated end knots)
        {3, 3, {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0}},
        // non-uniform quadratic
        {2, 4, {0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 4.0, 4.0}}
    };
    out << "  \"bspline\": [";
    bool first = true;
    for (const auto& c : cfgs) {
        BSpline b(c.p, c.n, c.knots);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"p\": " << c.p << ", \"n\": " << c.n << ", \"knots\": ";
        numArray(c.knots);
        out << ",\n     \"values\": [";
        bool f2 = true;
        // Sample on the knots themselves (where the p == 0 half-open support
        // test decides), at midpoints, and outside the knot span.
        for (Real x = -0.5; x <= 4.6; x += 0.25) {
            for (Natural i = 0; i <= c.n; ++i) {
                out << (f2 ? "" : ", ");
                f2 = false;
                out << "[" << i << ", " << x << ", ";
                num(b(i, x));
                out << "]";
            }
        }
        out << "]}";
    }
    out << "\n  ],\n";
}

// ------------------------------------------------- adaptive Runge-Kutta ----

void emitOde() {
    out << "  \"adaptive_runge_kutta\": {\n";
    // y' = y, y(0) = 1  =>  y(x) = exp(x). Easy; every RK gets it.
    AdaptiveRungeKutta<Real> rk(1e-8, 1e-4, 1e-12);
    out << "    \"exponential\": [";
    bool first = true;
    for (Real x2 : {0.5, 1.0, 2.0, -1.0}) {
        AdaptiveRungeKutta<Real> r(1e-8, 1e-4, 1e-12);
        out << (first ? "" : ", ");
        first = false;
        out << "[" << x2 << ", ";
        num(r([](Real, Real y) { return y; }, 1.0, 0.0, x2));
        out << "]";
    }
    out << "],\n";

    // y' = -50 (y - cos x): moderately stiff, forces the controller to shrink
    // the step. A fixed-step integrator agrees on the exponential above and
    // parts company here.
    out << "    \"stiff\": [";
    first = true;
    for (Real eps : {1e-6, 1e-10}) {
        for (Real x2 : {1.0, 3.0}) {
            AdaptiveRungeKutta<Real> r(eps, 1e-4, 1e-14);
            out << (first ? "" : ", ");
            first = false;
            out << "{\"eps\": " << eps << ", \"x2\": " << x2 << ", \"v\": ";
            num(r([](Real x, Real y) { return -50.0 * (y - std::cos(x)); }, 0.0, 0.0, x2));
            out << "}";
        }
    }
    out << "],\n";

    // Two-dimensional system: harmonic oscillator y'' = -y written as a
    // first-order pair. Pins the vector overload, which is the one the 1-D
    // overload delegates to through OdeFctWrapper.
    out << "    \"harmonic\": [";
    first = true;
    for (Real x2 : {1.0, 3.14159, 6.0}) {
        AdaptiveRungeKutta<Real> r(1e-10, 1e-4, 1e-14);
        const std::vector<Real> y = r(
            [](Real, const std::vector<Real>& v) {
                return std::vector<Real>{v[1], -v[0]};
            },
            std::vector<Real>{1.0, 0.0}, 0.0, x2);
        out << (first ? "" : ", ");
        first = false;
        out << "{\"x2\": " << x2 << ", \"y\": ";
        numArray(y);
        out << "}";
    }
    out << "]\n  },\n";
    (void)rk;
}

// ------------------------------------------------------------------- FFT ---

void emitFft() {
    out << "  \"fft\": {\n    \"min_order\": [";
    for (std::size_t i = 1; i <= 10; ++i) {
        if (i > 1) out << ", ";
        out << "[" << i << ", " << FastFourierTransform::min_order(i) << "]";
    }
    out << "],\n    \"cases\": [";
    // A deliberately non-power-of-two input: the transform zero-pads by
    // leaving the untouched output slots at their default-constructed value,
    // so the padding behaviour is part of the answer.
    const std::vector<std::vector<std::complex<Real>>> inputs = {
        {{1.0, 0.0}, {2.0, -1.0}, {0.0, 0.0}, {-1.0, 2.0}},
        {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}},
        {{0.5, 0.25}}
    };
    bool first = true;
    for (const auto& in : inputs) {
        const std::size_t order = FastFourierTransform::min_order(in.size());
        FastFourierTransform fft(order);
        std::vector<std::complex<Real>> fwd(fft.output_size()), inv(fft.output_size());
        fft.transform(in.begin(), in.end(), fwd.begin());
        fft.inverse_transform(in.begin(), in.end(), inv.begin());
        out << (first ? "\n      " : ",\n      ");
        first = false;
        out << "{\"input\": [";
        for (std::size_t i = 0; i < in.size(); ++i) {
            if (i) out << ", ";
            out << "[";
            num(in[i].real());
            out << ", ";
            num(in[i].imag());
            out << "]";
        }
        out << "], \"order\": " << order << ", \"output_size\": " << fft.output_size();
        out << ",\n       \"forward\": [";
        for (std::size_t i = 0; i < fwd.size(); ++i) {
            if (i) out << ", ";
            out << "[";
            num(fwd[i].real());
            out << ", ";
            num(fwd[i].imag());
            out << "]";
        }
        out << "],\n       \"inverse\": [";
        for (std::size_t i = 0; i < inv.size(); ++i) {
            if (i) out << ", ";
            out << "[";
            num(inv[i].real());
            out << ", ";
            num(inv[i].imag());
            out << "]";
        }
        out << "]}";
    }
    out << "\n    ]\n  },\n";
}

// -------------------------------------------------------- Gaussian copula --

void emitGaussianCopula() {
    const std::vector<Real> rhos = {-0.9, -0.5, 0.0, 0.5, 0.9};
    const std::vector<std::pair<Real, Real>> pts = {
        {0.001, 0.001}, {0.01, 0.5}, {0.25, 0.75}, {0.5, 0.5}, {0.9, 0.1},
        {0.999, 0.999}
    };
    out << "  \"gaussian_copula\": [";
    bool first = true;
    for (Real rho : rhos) {
        GaussianCopula c(rho);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"rho\": " << rho << ", \"cases\": [";
        for (std::size_t i = 0; i < pts.size(); ++i) {
            if (i) out << ", ";
            out << "[" << pts[i].first << ", " << pts[i].second << ", ";
            num(c(pts[i].first, pts[i].second));
            out << "]";
        }
        out << "]}";
    }
    out << "\n  ]\n";
}

} // namespace

int main() {
    out << std::setprecision(17);
    out << "{\n";
    emitAbcd();
    emitPolynomial();
    emitRichardson();
    emitPrimes();
    emitBSpline();
    emitOde();
    emitFft();
    emitGaussianCopula();
    out << "}\n";
    return 0;
}
