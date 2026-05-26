// L1-C mega-probe: 9 Solver1D concretes + 6 simple integrals.

#include <ql/math/integrals/simpsonintegral.hpp>
#include <ql/math/integrals/trapezoidintegral.hpp>
#include <ql/math/integrals/segmentintegral.hpp>
#include <ql/math/integrals/kronrodintegral.hpp>
#include <ql/math/integrals/gausslobattointegral.hpp>
#include <ql/math/solvers1d/bisection.hpp>
#include <ql/math/solvers1d/brent.hpp>
#include <ql/math/solvers1d/falseposition.hpp>
#include <ql/math/solvers1d/finitedifferencenewtonsafe.hpp>
#include <ql/math/solvers1d/halley.hpp>
#include <ql/math/solvers1d/newton.hpp>
#include <ql/math/solvers1d/newtonsafe.hpp>
#include <ql/math/solvers1d/ridder.hpp>
#include <ql/math/solvers1d/secant.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

// Test function with simple closed-form roots: f(x) = (x - 2)(x - 5).
// Roots at x=2 and x=5. f'(x) = 2x - 7.
struct Quadratic {
    Real operator()(Real x) const { return (x - 2.0) * (x - 5.0); }
    Real derivative(Real x) const { return 2.0 * x - 7.0; }
    Real secondDerivative(Real /*x*/) const { return 2.0; }
};

}

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    Quadratic f;
    Real accuracy = 1e-12;

    std::cout << "  \"solvers\": {\n";
    // Bracketed solvers: bracket = [3.0, 7.0] guarantees the root at x=5.
    Real x_min = 3.0, x_max = 7.0;
    Real guess = 4.0;

    std::cout << "    \"bisection\":           " << Bisection().solve(f, accuracy, guess, x_min, x_max) << ",\n";
    std::cout << "    \"brent\":               " << Brent().solve(f, accuracy, guess, x_min, x_max) << ",\n";
    std::cout << "    \"false_position\":      " << FalsePosition().solve(f, accuracy, guess, x_min, x_max) << ",\n";
    std::cout << "    \"ridder\":              " << Ridder().solve(f, accuracy, guess, x_min, x_max) << ",\n";
    std::cout << "    \"secant\":              " << Secant().solve(f, accuracy, guess, x_min, x_max) << ",\n";
    // Unbracketed (Newton-style): start from guess + step.
    std::cout << "    \"newton\":              " << Newton().solve(f, accuracy, guess, 0.1) << ",\n";
    std::cout << "    \"newton_safe\":         " << NewtonSafe().solve(f, accuracy, guess, x_min, x_max) << ",\n";
    std::cout << "    \"halley\":              " << Halley().solve(f, accuracy, guess, 0.1) << ",\n";
    std::cout << "    \"fd_newton_safe\":      " << FiniteDifferenceNewtonSafe().solve(f, accuracy, guess, x_min, x_max) << "\n";
    std::cout << "  },\n";

    // --- integrals -------------------------------------------------------
    std::cout << "  \"integrals\": {\n";
    // Integrand: x^2 over [0, 1] = 1/3.
    auto f_x_squared = [](Real x) { return x * x; };

    SimpsonIntegral simpson(1e-10, 100);
    std::cout << "    \"simpson_x_squared\":    " << simpson(f_x_squared, 0.0, 1.0) << ",\n";

    TrapezoidIntegral<Default> trapezoid(1e-10, 100);
    std::cout << "    \"trapezoid_x_squared\":  " << trapezoid(f_x_squared, 0.0, 1.0) << ",\n";

    SegmentIntegral segment(100);
    std::cout << "    \"segment_x_squared\":    " << segment(f_x_squared, 0.0, 1.0) << ",\n";

    GaussKronrodAdaptive kronrod(1e-10, 100);
    std::cout << "    \"kronrod_x_squared\":    " << kronrod(f_x_squared, 0.0, 1.0) << ",\n";

    GaussLobattoIntegral lobatto(100, 1e-10);
    std::cout << "    \"lobatto_x_squared\":    " << lobatto(f_x_squared, 0.0, 1.0) << ",\n";

    // Integrand: sin(x) over [0, π] = 2.
    auto f_sin = [](Real x) { return std::sin(x); };
    Real pi = std::acos(-1.0);
    std::cout << "    \"simpson_sin\":          " << simpson(f_sin, 0.0, pi) << ",\n";
    std::cout << "    \"trapezoid_sin\":        " << trapezoid(f_sin, 0.0, pi) << ",\n";
    std::cout << "    \"kronrod_sin\":          " << kronrod(f_sin, 0.0, pi) << "\n";
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
