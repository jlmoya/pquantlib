// L5-A foundations mega-probe.
//
// Captures reference values for the L5-A foundations layer (the
// Phase 5 pilot):
//
//   * SobolRsg first 5 vectors at dimension 2 with default direction
//     integers (C++ default = Jaeckel) and with JoeKuoD5 — the latter
//     matches scipy's qmc.Sobol implementation, which is what the
//     Python port wraps. Captures both so the Python tests can pick
//     the matching set.
//   * GammaFunction(x) and GammaFunction::logValue(x) at
//     x = 0.5, 1.0, 1.5, 2.0, 5.0, 10.0. logValue requires x > 0;
//     Gamma(0.5) = sqrt(pi); Gamma(1) = 1; Gamma(5) = 24; Gamma(10) = 362880.
//   * AkimaCubicInterpolation at 5 (x, y) pairs evaluated at
//     intermediate x positions. The known-good Akima reference
//     comes from C++ directly.
//
// C++ parity:
//   ql/math/randomnumbers/sobolrsg.{hpp,cpp},
//   ql/math/distributions/gammadistribution.{hpp,cpp},
//   ql/math/interpolations/cubicinterpolation.hpp (AkimaCubicInterpolation)
//   @ v1.42.1 (099987f0).

#include <ql/math/distributions/gammadistribution.hpp>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/randomnumbers/sobolrsg.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

void emitArray(const std::vector<Real>& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

void emitSobolFirst5(const std::string& key,
                     SobolRsg::DirectionIntegers di) {
    // C++ parity: seed = 42, dim = 2, useGrayCode = true (default).
    SobolRsg rsg(2, 42, di, true);
    std::cout << "  \"" << key << "\": [\n";
    for (int k = 0; k < 5; ++k) {
        const auto& s = rsg.nextSequence();
        std::cout << "    ";
        emitArray(s.value);
        if (k != 4) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ],\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // -----------------------------------------------------------------
    // SobolRsg (dim=2, seed=42, useGrayCode=true) — capture two
    // direction-integer sets so the Python port can pick the one
    // matching scipy.
    // -----------------------------------------------------------------
    emitSobolFirst5("sobol_jaeckel_d2_s42", SobolRsg::Jaeckel);
    emitSobolFirst5("sobol_joekuod5_d2_s42", SobolRsg::JoeKuoD5);
    emitSobolFirst5("sobol_joekuod6_d2_s42", SobolRsg::JoeKuoD6);
    emitSobolFirst5("sobol_joekuod7_d2_s42", SobolRsg::JoeKuoD7);

    // -----------------------------------------------------------------
    // GammaFunction (C++ Lanczos approximation). value() recurses for
    // x < 1 via Gamma(x+1)/x and for x <= -20 via the reflection
    // formula. We probe x = 0.5, 1, 1.5, 2, 5, 10.
    // -----------------------------------------------------------------
    {
        GammaFunction g;
        std::cout << "  \"gamma_function\": {\n";
        const double xs[6] = {0.5, 1.0, 1.5, 2.0, 5.0, 10.0};
        std::cout << "    \"value\": [";
        for (int i = 0; i < 6; ++i) {
            std::cout << g.value(xs[i]);
            if (i + 1 < 6) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"log_value\": [";
        for (int i = 0; i < 6; ++i) {
            std::cout << g.logValue(xs[i]);
            if (i + 1 < 6) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"xs\": [";
        for (int i = 0; i < 6; ++i) {
            std::cout << xs[i];
            if (i + 1 < 6) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // AkimaCubicInterpolation: 5 (x, y) knots, evaluated at 5
    // intermediate x positions. Akima 1970 is local & non-monotonic.
    // -----------------------------------------------------------------
    {
        std::vector<Real> x = {0.0, 1.0, 2.0, 3.0, 4.0};
        std::vector<Real> y = {0.0, 1.0, 4.0, 9.0, 16.0}; // y = x^2 sample
        AkimaCubicInterpolation interp(x.begin(), x.end(), y.begin());
        interp.update();
        std::vector<Real> xs_eval = {0.25, 0.75, 1.5, 2.5, 3.75};
        std::cout << "  \"akima_cubic\": {\n";
        std::cout << "    \"xs\": ";
        emitArray(x);
        std::cout << ",\n";
        std::cout << "    \"ys\": ";
        emitArray(y);
        std::cout << ",\n";
        std::cout << "    \"xs_eval\": ";
        emitArray(xs_eval);
        std::cout << ",\n";
        std::cout << "    \"values\": [";
        for (Size i = 0; i < xs_eval.size(); ++i) {
            std::cout << interp(xs_eval[i]);
            if (i + 1 < xs_eval.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"derivatives\": [";
        for (Size i = 0; i < xs_eval.size(); ++i) {
            std::cout << interp.derivative(xs_eval[i]);
            if (i + 1 < xs_eval.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"second_derivatives\": [";
        for (Size i = 0; i < xs_eval.size(); ++i) {
            std::cout << interp.secondDerivative(xs_eval[i]);
            if (i + 1 < xs_eval.size()) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
