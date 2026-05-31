// Phase 11 W6-D cluster probe: experimental heuristic optimizers + RNG.
//
// Captures reference values for:
//
//   * ZigguratRng — first-N standard-normal draws for a fixed seed.
//     This is the EXACT-tier reproducibility anchor: the Python port
//     reuses the already-bit-exact MersenneTwisterUniformRng stream and
//     the same tabulated ziggurat constants, so it must reproduce these
//     doubles to the bit.  Also emits the sample mean/variance over a
//     large draw count for a LOOSE statistical cross-check.
//
//   * LevyFlightDistribution — pdf(x) at known x for known (xm, alpha),
//     plus inverse-CDF draws operator()(eng) for a deterministic stream
//     of uniforms.  The inverse transform xm * u^{-1/alpha} is captured
//     given the exact uniforms mt.nextReal() emits, anchoring value
//     parity (the C++ std::uniform_real_distribution draw differs from
//     QuantLib's mt.nextReal(), so the Python port documents that it
//     drives the inverse transform off a QuantLib MT uniform — the
//     reference here is the closed-form transform, not the C++ engine
//     draw, so both sides agree on the math).
//
//   * IsotropicRandomWalk — analytic isotropy invariant: for a unit
//     radius the squared norm of the step equals radius^2 * sum of the
//     per-dimension weight^2 contributions only in the 1-D case; for
//     d>1 the construction is a recursive spherical parametrisation.
//     We capture the deterministic step vector for a fixed MT seed and
//     a unit (degenerate) distribution so the Python port can match the
//     spherical-coordinate recurrence bit-for-bit against the same MT
//     uniform stream.
//
//   * ParticleSwarmOptimization — global minimum of the sphere and of
//     the (shifted) Rosenbrock test function.  The probe emits only the
//     analytic global optima; the Python test runs PSO with a fixed
//     seed and asserts convergence into that basin (LOOSE).
//
//   * FireflyAlgorithm — analytic global optimum of the same test
//     functions (region-convergence contract, LOOSE).
//
//   * HybridSimulatedAnnealing — analytic global optimum of a 1-D /
//     2-D test function (region-convergence contract, LOOSE).
//
// C++ parity:
//   ql/experimental/math/zigguratrng.hpp
//   ql/experimental/math/levyflightdistribution.hpp
//   ql/experimental/math/isotropicrandomwalk.hpp
//   ql/experimental/math/particleswarmoptimization.hpp
//   ql/experimental/math/fireflyalgorithm.hpp
//   ql/experimental/math/hybridsimulatedannealing.hpp
//   @ v1.42.1 (099987f0).

#include <ql/experimental/math/isotropicrandomwalk.hpp>
#include <ql/experimental/math/levyflightdistribution.hpp>
#include <ql/experimental/math/zigguratrng.hpp>
#include <ql/math/array.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

    void printArray(const std::vector<double>& a) {
        std::cout << "[";
        for (Size i = 0; i < a.size(); ++i) {
            std::cout << a[i];
            if (i + 1 < a.size()) std::cout << ", ";
        }
        std::cout << "]";
    }

}

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ---- ZigguratRng: first-N draws (EXACT anchor) ------------------
    {
        const unsigned long seed = 42UL;
        const Size n = 50;
        ZigguratRng zig(seed);
        std::vector<double> draws;
        draws.reserve(n);
        for (Size i = 0; i < n; ++i)
            draws.push_back(zig.next().value);

        std::cout << "  \"ziggurat\": {\n";
        std::cout << "    \"seed\": " << seed << ",\n";
        std::cout << "    \"n\": " << n << ",\n";
        std::cout << "    \"draws\": ";
        printArray(draws);
        std::cout << ",\n";

        // Large-sample statistical moments (LOOSE).
        const unsigned long seed2 = 1234UL;
        const Size big = 2000000;
        ZigguratRng zig2(seed2);
        double sum = 0.0, sumSq = 0.0;
        for (Size i = 0; i < big; ++i) {
            double x = zig2.next().value;
            sum += x;
            sumSq += x * x;
        }
        double mean = sum / big;
        double var = sumSq / big - mean * mean;
        std::cout << "    \"stat_seed\": " << seed2 << ",\n";
        std::cout << "    \"stat_n\": " << big << ",\n";
        std::cout << "    \"stat_mean\": " << mean << ",\n";
        std::cout << "    \"stat_var\": " << var << "\n";
        std::cout << "  },\n";
    }

    // ---- LevyFlightDistribution: pdf + inverse-transform draws -------
    {
        // pdf(x) at known x for (xm=1, alpha=1.5) and (xm=0.5, alpha=2.0)
        LevyFlightDistribution d1(1.0, 1.5);
        LevyFlightDistribution d2(0.5, 2.0);
        std::vector<double> xs = {0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0};
        std::vector<double> pdf1, pdf2;
        for (double x : xs) {
            pdf1.push_back(d1(x));
            pdf2.push_back(d2(x));
        }

        std::cout << "  \"levy_flight\": {\n";
        std::cout << "    \"xm1\": 1.0, \"alpha1\": 1.5,\n";
        std::cout << "    \"xm2\": 0.5, \"alpha2\": 2.0,\n";
        std::cout << "    \"xs\": ";
        printArray(xs);
        std::cout << ",\n";
        std::cout << "    \"pdf1\": ";
        printArray(pdf1);
        std::cout << ",\n";
        std::cout << "    \"pdf2\": ";
        printArray(pdf2);
        std::cout << ",\n";

        // Inverse-transform: given uniforms u, the variate is
        // xm * u^{-1/alpha}.  Emit the closed-form transform values for
        // a deterministic uniform grid so the Python port (which drives
        // the inverse transform off a QuantLib MT uniform) can match the
        // math exactly.
        std::vector<double> us = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};
        std::vector<double> var1, var2;
        for (double u : us) {
            var1.push_back(1.0 * std::pow(u, -1.0 / 1.5));
            var2.push_back(0.5 * std::pow(u, -1.0 / 2.0));
        }
        std::cout << "    \"us\": ";
        printArray(us);
        std::cout << ",\n";
        std::cout << "    \"variate1\": ";
        printArray(var1);
        std::cout << ",\n";
        std::cout << "    \"variate2\": ";
        printArray(var2);
        std::cout << "\n";
        std::cout << "  },\n";
    }

    // ---- IsotropicRandomWalk: deterministic step (MT-driven) --------
    {
        // Use a degenerate (constant-radius) distribution so the only
        // randomness is the MT-driven spherical angles.  We capture the
        // step vector for a 1-D and a 3-D walk so the Python port can
        // match the recursive spherical parametrisation against the
        // same MT uniform stream.
        //
        // The distribution functor must satisfy operator()(Engine&)
        // returning the radius.  We model a constant radius 1.0 by
        // using a Levy distribution with parameters that... no — we
        // need an exact constant.  Instead we emit, for a fixed MT seed,
        // the sequence of mt.nextReal() angles the walk would consume,
        // and the resulting step assuming radius == 1.0.  The Python
        // test reproduces this with its own constant-radius distribution.
        const unsigned long seed = 7UL;

        // 1-D: walk consumes one mt.nextReal() to pick sign.
        {
            MersenneTwisterUniformRng mt(seed);
            double u = mt.nextReal();
            double radius = 1.0;
            double step = (u < 0.5) ? -radius : radius;
            std::cout << "  \"isotropic_1d\": {\n";
            std::cout << "    \"seed\": " << seed << ",\n";
            std::cout << "    \"radius\": " << radius << ",\n";
            std::cout << "    \"u\": " << u << ",\n";
            std::cout << "    \"step\": " << step << "\n";
            std::cout << "  },\n";
        }

        // 3-D: dim=3 -> loop runs for i in [0, dim-2) == [0,1), i.e.
        // one iteration, consuming phi = pi*mt.nextReal() before the
        // loop, then the loop consumes one more phi.  Replicate the
        // exact recurrence from isotropicrandomwalk.hpp.
        {
            MersenneTwisterUniformRng mt(seed);
            const Size dim = 3;
            std::vector<double> weights(dim, 1.0);
            std::vector<double> out(dim, 0.0);
            double radius = 1.0;
            Size widx = 0;
            // phi = M_PI * rng.nextReal();
            double phi = M_PI * mt.nextReal();
            for (Size i = 0; i < dim - 2; ++i) {
                out[widx] = radius * std::cos(phi) * weights[widx];
                ++widx;
                radius *= std::sin(phi);
                phi = M_PI * mt.nextReal();
            }
            out[widx] = radius * std::cos(2.0 * phi) * weights[widx];
            // last element uses the *same* weight index (C++ does not
            // advance the weight iterator for the final sin term).
            out[widx + 1] = radius * std::sin(2.0 * phi) * weights[widx];

            std::cout << "  \"isotropic_3d\": {\n";
            std::cout << "    \"seed\": " << seed << ",\n";
            std::cout << "    \"dim\": " << dim << ",\n";
            std::cout << "    \"radius\": 1.0,\n";
            std::cout << "    \"step\": ";
            printArray(out);
            std::cout << "\n";
            std::cout << "  },\n";
        }
    }

    // ---- Optimizer global optima (LOOSE region-convergence) ---------
    {
        // Sphere: f(x) = sum x_i^2, global min 0 at origin.
        // Rosenbrock (2D): f = (a-x)^2 + b(y-x^2)^2, a=1, b=100,
        //   global min 0 at (1, 1).
        std::cout << "  \"optimizers\": {\n";
        std::cout << "    \"sphere_min_value\": 0.0,\n";
        std::cout << "    \"sphere_min_x\": [0.0, 0.0],\n";
        std::cout << "    \"rosenbrock_a\": 1.0,\n";
        std::cout << "    \"rosenbrock_b\": 100.0,\n";
        std::cout << "    \"rosenbrock_min_value\": 0.0,\n";
        std::cout << "    \"rosenbrock_min_x\": [1.0, 1.0]\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
