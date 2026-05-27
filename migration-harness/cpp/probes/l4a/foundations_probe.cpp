// L4-A foundations mega-probe.
//
// Captures reference values for the L4-A foundations layer (the
// Phase 4 pilot):
//
//   * LevenbergMarquardt minimum on Rosenbrock f(x,y)=(1-x)^2+100(y-x^2)^2,
//     residuals form r=(1-x, 10(y-x^2)).
//   * Simplex minimum on the same Rosenbrock (scalar f directly).
//   * ConstantParameter round-trips (value, testParams).
//   * PiecewiseConstantParameter round-trips at tenor grid {1,3,5}
//     with values {0.10, 0.20, 0.30, 0.40} (one extra trailing slot).
//   * Constraint transforms — NoConstraint and PositiveConstraint —
//     report ``test`` outcomes on representative arrays and the bounds
//     they expose (used by ProjectedConstraint in scipy wrappers).
//
// C++ parity:
//   ql/math/optimization/levenbergmarquardt.{hpp,cpp},
//   ql/math/optimization/simplex.{hpp,cpp},
//   ql/math/optimization/constraint.{hpp,cpp},
//   ql/models/parameter.hpp
//   @ v1.42.1 (099987f0).

#include <ql/math/array.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/costfunction.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/math/optimization/simplex.hpp>
#include <ql/models/parameter.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

// Rosenbrock f(x,y) = (1-x)^2 + 100*(y - x^2)^2
// As a sum-of-squares: r1 = 1-x, r2 = 10*(y-x^2); f = r1^2 + r2^2.
class RosenbrockResiduals : public CostFunction {
  public:
    Array values(const Array& x) const override {
        Array r(2);
        r[0] = 1.0 - x[0];
        r[1] = 10.0 * (x[1] - x[0] * x[0]);
        return r;
    }
    // Inherits default ``value`` (sqrt(mean(values(x)^2))). Override
    // here to explicitly return Rosenbrock f, which Simplex needs as
    // a scalar (it ignores ``values``).
    Real value(const Array& x) const override {
        Real r1 = 1.0 - x[0];
        Real r2 = 10.0 * (x[1] - x[0] * x[0]);
        return r1 * r1 + r2 * r2;
    }
};

void emitArray(const Array& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // -----------------------------------------------------------------
    // LevenbergMarquardt on Rosenbrock
    // -----------------------------------------------------------------
    {
        RosenbrockResiduals cf;
        NoConstraint c;
        Array x0(2);
        x0[0] = -1.2;
        x0[1] = 1.0;
        Problem p(cf, c, x0);
        EndCriteria ec(2000, 100, 1e-12, 1e-12, 1e-12);
        LevenbergMarquardt lm(1e-8, 1e-8, 1e-8);
        EndCriteria::Type rc = lm.minimize(p, ec);
        std::cout << "  \"levenberg_marquardt\": {\n";
        std::cout << "    \"end_criteria\": " << static_cast<int>(rc) << ",\n";
        std::cout << "    \"x\": ";
        emitArray(p.currentValue());
        std::cout << ",\n";
        std::cout << "    \"f\": " << cf.value(p.currentValue()) << ",\n";
        std::cout << "    \"function_evaluation\": " << p.functionEvaluation() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // Simplex on Rosenbrock
    // -----------------------------------------------------------------
    {
        RosenbrockResiduals cf;
        NoConstraint c;
        Array x0(2);
        x0[0] = -1.2;
        x0[1] = 1.0;
        Problem p(cf, c, x0);
        EndCriteria ec(10000, 1000, 1e-12, 1e-12, 1e-12);
        Simplex s(0.1);
        EndCriteria::Type rc = s.minimize(p, ec);
        std::cout << "  \"simplex\": {\n";
        std::cout << "    \"end_criteria\": " << static_cast<int>(rc) << ",\n";
        std::cout << "    \"x\": ";
        emitArray(p.currentValue());
        std::cout << ",\n";
        std::cout << "    \"f\": " << cf.value(p.currentValue()) << ",\n";
        std::cout << "    \"function_evaluation\": " << p.functionEvaluation() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // ConstantParameter round-trip
    // -----------------------------------------------------------------
    {
        ConstantParameter cp(0.05, NoConstraint());
        std::cout << "  \"constant_parameter\": {\n";
        std::cout << "    \"size\": " << cp.size() << ",\n";
        std::cout << "    \"value_at_0\": " << cp(0.0) << ",\n";
        std::cout << "    \"value_at_3_25\": " << cp(3.25) << ",\n";
        std::cout << "    \"value_at_100\": " << cp(100.0) << ",\n";
        std::cout << "    \"params\": ";
        emitArray(cp.params());
        std::cout << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // PiecewiseConstantParameter round-trip
    // -----------------------------------------------------------------
    {
        // C++ parity (parameter.hpp:119-142): with times {1, 3, 5}, an
        // array of size 4 is allocated; index i = upper_bound(times, t)
        // returns 0,1,2,3 for t in (-inf,1], (1,3], (3,5], (5,+inf).
        std::vector<Time> times = {1.0, 3.0, 5.0};
        PiecewiseConstantParameter pcp(times, NoConstraint());
        pcp.setParam(0, 0.10);
        pcp.setParam(1, 0.20);
        pcp.setParam(2, 0.30);
        pcp.setParam(3, 0.40);
        std::cout << "  \"piecewise_constant_parameter\": {\n";
        std::cout << "    \"size\": " << pcp.size() << ",\n";
        std::cout << "    \"value_at_0\": " << pcp(0.0) << ",\n";
        std::cout << "    \"value_at_1\": " << pcp(1.0) << ",\n";
        std::cout << "    \"value_at_1_5\": " << pcp(1.5) << ",\n";
        std::cout << "    \"value_at_3\": " << pcp(3.0) << ",\n";
        std::cout << "    \"value_at_3_5\": " << pcp(3.5) << ",\n";
        std::cout << "    \"value_at_5\": " << pcp(5.0) << ",\n";
        std::cout << "    \"value_at_6\": " << pcp(6.0) << ",\n";
        std::cout << "    \"params\": ";
        emitArray(pcp.params());
        std::cout << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // Constraint transforms — NoConstraint & PositiveConstraint
    // -----------------------------------------------------------------
    {
        NoConstraint nc;
        Array a(3);
        a[0] = -1.0; a[1] = 0.0; a[2] = 1e300;
        Array b(2);
        b[0] = 1.0; b[1] = 0.0;
        Array c(2);
        c[0] = 1.0; c[1] = -1.0;
        std::cout << "  \"no_constraint\": {\n";
        std::cout << "    \"test_neg_zero_huge\": " << (nc.test(a) ? "true" : "false") << ",\n";
        Array probe(3);
        probe[0] = 0.0; probe[1] = 1.0; probe[2] = 2.0;
        Array up = nc.upperBound(probe);
        Array lo = nc.lowerBound(probe);
        std::cout << "    \"upper_bound_at_0_is_pos_max\": " << (up[0] > 1e300 ? "true" : "false") << ",\n";
        std::cout << "    \"lower_bound_at_0_is_neg_max\": " << (lo[0] < -1e300 ? "true" : "false") << "\n";
        std::cout << "  },\n";

        PositiveConstraint pc;
        std::cout << "  \"positive_constraint\": {\n";
        std::cout << "    \"test_pos_zero\": " << (pc.test(b) ? "true" : "false") << ",\n";
        std::cout << "    \"test_pos_neg\": " << (pc.test(c) ? "true" : "false") << ",\n";
        Array probe2(2);
        probe2[0] = 1.0; probe2[1] = 2.0;
        std::cout << "    \"lower_bound\": ";
        emitArray(pc.lowerBound(probe2));
        std::cout << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
