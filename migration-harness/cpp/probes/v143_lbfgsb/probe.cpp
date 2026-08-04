// migration-harness/cpp/probes/v143_lbfgsb/probe.cpp
//
// Reference values for LBFGSB — the limited-memory bound-constrained
// quasi-Newton optimizer introduced in C++ QuantLib v1.43
// (ql/math/optimization/lbfgsb.{hpp,cpp}, Byrd/Lu/Nocedal/Zhu 1995).
//
// WHAT NEEDS PINNING, AND WHY
// ---------------------------
// The two places a port silently diverges are the *generalized Cauchy point*
// (Algorithm CP: walking the breakpoints of the projected steepest-descent
// path and pinning variables as they hit a bound) and the *subspace
// minimization* (direct primal method over the free variables, then truncation
// back into the box). Both live in an anonymous namespace inside lbfgsb.cpp,
// so neither is directly observable from outside the translation unit.
//
// Compensating strategy — three independent layers:
//
//   1. MANY DISTINCT BOUND/START CONFIGURATIONS. Interior optimum, optimum
//      clamped on one bound, all-bounds-active corner (empty free set), two
//      bounds reached through *distinct* breakpoints (so the Cauchy walk must
//      traverse more than one), a pinned coordinate (lower == upper), an
//      infeasible start that must be clipped, a start sitting exactly on the
//      bound the gradient pushes into (the `t_i <= 0` branch, for both the
//      upper and the lower bound), a start already at the constrained optimum
//      (zero iterations), and mixed bounded/unbounded coordinates (the
//      +/-0.5*DBL_MAX "no bound" sentinel test).
//
//   2. TRUNCATED-ITERATION TRAJECTORY. EndCriteria::maxIterations caps the
//      outer loop, and the iterate is published via Problem::setCurrentValue
//      at the end of every iteration. Running the same problem with
//      maxIterations = 3, 4, ... 12 therefore exposes the iterate sequence one
//      step at a time, pinning the Cauchy point + subspace step indirectly but
//      tightly.
//
//   3. FULL OBJECTIVE-EVALUATION TRACE. LBFGSB reaches the cost function only
//      through Problem::valueAndGradient, so a recording CostFunction captures
//      every trial point in order: the initial point, then every line-search
//      probe. This pins the search direction AND the Wolfe line search
//      (c1 = 1e-4, c2 = 0.9, doubling expansion capped at the largest feasible
//      step, then bisection, then the "accept best sufficient decrease"
//      fallback).
//
// Every case also records the terminal EndCriteria::Type, the projected
// gradient infinity norm (the KKT residual of a box-constrained problem),
// Problem::gradientNormValue() (which LBFGSB sets to pgInf^2, NOT |g|^2) and
// the function/gradient evaluation counters. The counters are structural
// diagnostics: if x matches but the counts do not, the inner loop differs and
// the agreement is luck.
//
// The cost functions and canonical setups mirror test-suite/optimizers.cpp
// @ v1.43 (testLBFGSB / testLBFGSBActiveBounds / testLBFGSBCoverage).
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/lbfgsb.json.

#include <ql/errors.hpp>
#include <ql/math/array.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/costfunction.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/lbfgsb.hpp>
#include <ql/math/optimization/problem.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
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

void emit(const char* name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
}

void emit_int(const char* name, long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_bool(const char* name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_str(const char* name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
}

void emit_arr(const char* name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const char* name, const std::vector<long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

std::vector<Real> toVec(const Array& a) {
    return std::vector<Real>(a.begin(), a.end());
}

std::string key(const std::string& prefix, const char* field) {
    return prefix + "_" + field;
}

std::string ecName(EndCriteria::Type t) {
    std::ostringstream os;
    os << t; // QuantLib::operator<<(std::ostream&, EndCriteria::Type)
    return os.str();
}

// --------------------------------------------------------------------------
// Cost functions (verbatim from test-suite/optimizers.cpp @ v1.43)
// --------------------------------------------------------------------------

// Extended Rosenbrock with analytic gradient; global minimum 0 at (1,...,1).
class RosenbrockFunction : public CostFunction {
  public:
    Real value(const Array& x) const override {
        Real f = 0.0;
        for (Size i = 0; i + 1 < x.size(); ++i)
            f += 100.0 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1.0 - x[i], 2);
        return f;
    }
    Array values(const Array& x) const override { return Array(1, value(x)); }
    void gradient(Array& grad, const Array& x) const override {
        std::fill(grad.begin(), grad.end(), 0.0);
        for (Size i = 0; i + 1 < x.size(); ++i) {
            grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
            grad[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
        }
    }
    Real valueAndGradient(Array& grad, const Array& x) const override {
        gradient(grad, x);
        return value(x);
    }
};

// Separable quadratic sum_i w_i (x_i - c_i)^2; unconstrained minimum at c.
class WeightedQuadratic : public CostFunction {
  public:
    WeightedQuadratic(Array center, Array weight)
    : center_(std::move(center)), weight_(std::move(weight)) {}
    Real value(const Array& x) const override {
        Real f = 0.0;
        for (Size i = 0; i < x.size(); ++i)
            f += weight_[i] * std::pow(x[i] - center_[i], 2);
        return f;
    }
    Array values(const Array& x) const override { return Array(1, value(x)); }
    void gradient(Array& grad, const Array& x) const override {
        for (Size i = 0; i < x.size(); ++i)
            grad[i] = 2.0 * weight_[i] * (x[i] - center_[i]);
    }
    Real valueAndGradient(Array& grad, const Array& x) const override {
        gradient(grad, x);
        return value(x);
    }
  private:
    Array center_, weight_;
};

// Same quadratic with no analytic gradient, forcing the optimizer onto
// CostFunction's central-difference gradient (finiteDifferenceEpsilon 1e-8).
// Those inner evaluations bypass Problem, so they do NOT bump its counters.
class WeightedQuadraticValueOnly : public CostFunction {
  public:
    WeightedQuadraticValueOnly(Array center, Array weight)
    : center_(std::move(center)), weight_(std::move(weight)) {}
    Real value(const Array& x) const override {
        Real f = 0.0;
        for (Size i = 0; i < x.size(); ++i)
            f += weight_[i] * std::pow(x[i] - center_[i], 2);
        return f;
    }
    Array values(const Array& x) const override { return Array(1, value(x)); }
  private:
    Array center_, weight_;
};

// Decorator recording every Problem::valueAndGradient call, in order.
template <class Base>
class Traced : public Base {
  public:
    using Base::Base;
    Real valueAndGradient(Array& grad, const Array& x) const override {
        const Real f = Base::valueAndGradient(grad, x);
        xs_.push_back(x);
        fs_.push_back(f);
        return f;
    }
    const std::vector<Array>& tracedX() const { return xs_; }
    const std::vector<Real>& tracedF() const { return fs_; }
  private:
    mutable std::vector<Array> xs_;
    mutable std::vector<Real> fs_;
};

// --------------------------------------------------------------------------
// Shared setup
// --------------------------------------------------------------------------

EndCriteria stdEndCriteria() {
    return EndCriteria(1000, 100, 1e-12, 1e-12, 1e-10);
}

const Real INF_BOUND = QL_MAX_REAL; // the "no bound" sentinel, == DBL_MAX

// The tight optimizer settings used by almost every upstream case.
const Size kMem = 10;
const Real kPgTol = 1e-10;

Real kFTol() { return 1e1 * QL_EPSILON; }

Array qCenter() { return Array{3.0, -2.0, 0.5}; }
Array qWeight() { return Array{1.0, 4.0, 0.25}; }

// Infinity norm of the projected gradient P(x - g, l, u) - x, the quantity
// that vanishes at a KKT point of a box-constrained problem.
Real projectedGradientNorm(const Array& x, const Array& g, const Array& lo, const Array& hi) {
    Real norm = 0.0;
    for (Size i = 0; i < x.size(); ++i) {
        const Real proj = std::min(std::max(x[i] - g[i], lo[i]), hi[i]) - x[i];
        norm = std::max(norm, std::fabs(proj));
    }
    return norm;
}

void emitResult(const std::string& prefix,
                Problem& p,
                EndCriteria::Type ec,
                const CostFunction& f,
                const Array& lo,
                const Array& hi) {
    const Array x = p.currentValue();
    Array g(x.size(), 0.0);
    f.gradient(g, x); // direct call: does not disturb the Problem counters

    emit_arr(key(prefix, "x").c_str(), toVec(x));
    emit(key(prefix, "f").c_str(), p.functionValue());
    emit(key(prefix, "grad_norm_value").c_str(), p.gradientNormValue()); // == pgInf^2
    emit(key(prefix, "pg_inf_norm").c_str(), projectedGradientNorm(x, g, lo, hi));
    emit_arr(key(prefix, "gradient").c_str(), toVec(g));
    emit_int(key(prefix, "ec").c_str(), static_cast<long>(ec));
    emit_str(key(prefix, "ec_name").c_str(), ecName(ec));
    emit_int(key(prefix, "nfev").c_str(), static_cast<long>(p.functionEvaluation()));
    emit_int(key(prefix, "ngev").c_str(), static_cast<long>(p.gradientEvaluation()));
}

void runBounded(const std::string& prefix,
                CostFunction& f,
                const Array& lo,
                const Array& hi,
                const Array& x0,
                Size memory,
                Real pgTol,
                Real fTol,
                const EndCriteria& endCriteria) {
    NonhomogeneousBoundaryConstraint c(lo, hi);
    Problem problem(f, c, x0);
    LBFGSB optimizer(memory, pgTol, fTol);
    const EndCriteria::Type ec = optimizer.minimize(problem, endCriteria);
    emitResult(prefix, problem, ec, f, lo, hi);
}

void runUnconstrained(const std::string& prefix,
                      CostFunction& f,
                      const Array& x0,
                      Size memory,
                      Real pgTol,
                      Real fTol,
                      const EndCriteria& endCriteria) {
    NoConstraint c;
    Problem problem(f, c, x0);
    LBFGSB optimizer(memory, pgTol, fTol);
    const EndCriteria::Type ec = optimizer.minimize(problem, endCriteria);
    emitResult(prefix, problem, ec, f, c.lowerBound(x0), c.upperBound(x0));
}

// --------------------------------------------------------------------------
// Blocks
// --------------------------------------------------------------------------

// Unconstrained equivalence: with no active bounds LBFGSB degrades to plain
// limited-memory BFGS and must find (1,...,1).
void block_unconstrained() {
    const EndCriteria ec = stdEndCriteria();
    {
        RosenbrockFunction f;
        runUnconstrained("rosenbrock_2d_unconstrained", f, Array(2, -1.0),
                         kMem, kPgTol, kFTol(), ec);
    }
    {
        RosenbrockFunction f;
        runUnconstrained("rosenbrock_10d_unconstrained", f, Array(10, -1.0),
                         kMem, kPgTol, kFTol(), ec);
    }
    {
        // memory (3) < dimension (20): forces eviction of correction pairs and
        // repeated rebuilds of the compact representation.
        RosenbrockFunction f;
        runUnconstrained("rosenbrock_20d_small_memory", f, Array(20, -1.0),
                         3, 1e-8, kFTol(), ec);
    }
    {
        // memory == 1: the minimal non-empty compact representation (col = 1,
        // so W is n x 2 and M is 2 x 2).
        RosenbrockFunction f;
        runUnconstrained("rosenbrock_2d_memory_one", f, Array(2, -1.0),
                         1, kPgTol, kFTol(), ec);
    }
}

void block_bounded() {
    const EndCriteria ec = stdEndCriteria();

    // Interior optimum: the box encloses the unconstrained minimizer.
    {
        WeightedQuadratic f(qCenter(), qWeight());
        runBounded("quadratic_interior_optimum", f, Array(3, -10.0), Array(3, 10.0),
                   Array(3, 0.0), kMem, kPgTol, kFTol(), ec);
    }

    // Active-bound optimum with the DEFAULT constructor
    // LBFGSB() == LBFGSB(10, 1e-8, 1e7 * QL_EPSILON). [0,1]^3 clips
    // (3,-2,0.5) to (1,0,0.5). Upstream asserts the operative stop is
    // ZeroGradientNorm (the KKT test), not the factr fallback.
    {
        WeightedQuadratic f(qCenter(), qWeight());
        const Array lo(3, 0.0), hi(3, 1.0), x0(3, 0.5);
        NonhomogeneousBoundaryConstraint c(lo, hi);
        Problem problem(f, c, x0);
        LBFGSB optimizer; // defaults
        const EndCriteria::Type t = optimizer.minimize(problem, ec);
        emitResult("quadratic_active_bounds_default_ctor", problem, t, f, lo, hi);
    }

    // Bound-constrained Rosenbrock: [-2,0.5]^2 clips the optimum; SciPy's
    // L-BFGS-B converges to (0.5, 0.25) on the boundary.
    {
        RosenbrockFunction f;
        runBounded("rosenbrock_2d_bounded", f, Array(2, -2.0), Array(2, 0.5),
                   Array(2, -1.0), kMem, kPgTol, kFTol(), ec);
    }

    // All bounds active at a corner: the free set is EMPTY, exercising the
    // nf == 0 early return of the subspace minimization.
    {
        WeightedQuadratic f(Array{5.0, 5.0, 5.0}, Array{1.0, 1.0, 1.0});
        runBounded("quadratic_all_active_corner", f, Array(3, 0.0), Array(3, 1.0),
                   Array(3, 0.5), kMem, kPgTol, kFTol(), ec);
    }

    // Two bounds reached through DISTINCT breakpoints: the disparate weights
    // make the projected steepest-descent path hit the two upper bounds at
    // different step lengths, so the Cauchy walk must traverse more than one.
    {
        WeightedQuadratic f(Array{10.0, 10.0}, Array{1.0, 100.0});
        runBounded("quadratic_two_active_distinct_breakpoints", f, Array(2, 0.0),
                   Array(2, 1.0), Array{0.9, 0.1}, kMem, kPgTol, kFTol(), ec);
    }

    // n = 1 with an active bound: min (x-5)^2 over [0,1] is x = 1.
    {
        WeightedQuadratic f(Array{5.0}, Array{1.0});
        runBounded("quadratic_1d_active_bound", f, Array(1, 0.0), Array(1, 1.0),
                   Array(1, 0.5), kMem, kPgTol, kFTol(), ec);
    }

    // Pinned coordinate (lower == upper).
    {
        WeightedQuadratic f(qCenter(), qWeight());
        runBounded("quadratic_pinned_coordinate", f, Array{-10.0, -10.0, 0.25},
                   Array{10.0, 10.0, 0.25}, Array(3, 0.0), kMem, kPgTol, kFTol(), ec);
    }

    // Infeasible start: x0 = (5,5,5) is outside [0,1]^3 and must be clipped
    // into the box BEFORE the first objective evaluation.
    {
        WeightedQuadratic f(qCenter(), qWeight());
        runBounded("quadratic_infeasible_start", f, Array(3, 0.0), Array(3, 1.0),
                   Array(3, 5.0), kMem, kPgTol, kFTol(), ec);
    }

    // Finite-difference gradient with active bounds.
    {
        WeightedQuadraticValueOnly f(Array{3.0, -2.0, 0.7}, Array{1.0, 4.0, 0.25});
        runBounded("quadratic_finite_difference_gradient", f, Array(3, 0.0),
                   Array(3, 1.0), Array(3, 0.5), kMem, 1e-6, 1e7 * QL_EPSILON, ec);
    }

    // Start ON the bound the gradient pushes into: the Cauchy breakpoint is
    // t_i = 0 (or -0.0), so the `t_i <= 0` branch pins the coordinate before
    // the walk starts. Upper- and lower-bound variants.
    {
        // x0[0] = 1.0 == hi[0], g[0] = 2*(1-5) = -8 < 0 => t_0 = -0.0 <= 0.
        WeightedQuadratic f(Array{5.0, 5.0}, Array{1.0, 1.0});
        runBounded("quadratic_start_on_upper_bound_gradient_pushes_out", f,
                   Array(2, 0.0), Array(2, 1.0), Array{1.0, 0.5},
                   kMem, kPgTol, kFTol(), ec);
    }
    {
        // x0[0] = 0.0 == lo[0], g[0] = 2*(0+5) = 10 > 0 => t_0 = 0 <= 0.
        WeightedQuadratic f(Array{-5.0, -5.0}, Array{1.0, 1.0});
        runBounded("quadratic_start_on_lower_bound_gradient_pushes_out", f,
                   Array(2, 0.0), Array(2, 1.0), Array{0.0, 0.5},
                   kMem, kPgTol, kFTol(), ec);
    }

    // Start already AT the constrained optimum: pgInf == 0 on the very first
    // check, so the loop exits with ZeroGradientNorm after exactly one
    // objective evaluation and zero iterations.
    {
        WeightedQuadratic f(Array{5.0, 5.0}, Array{1.0, 1.0});
        runBounded("quadratic_start_at_optimum_corner", f, Array(2, 0.0),
                   Array(2, 1.0), Array(2, 1.0), kMem, kPgTol, kFTol(), ec);
    }

    // Mixed bounded / unbounded coordinates. Pins the noUpper/noLower
    // sentinel test, which is `u >= 0.5*DBL_MAX` / `l <= -0.5*DBL_MAX` (the
    // 0.5 guards against overflow in x - bound), NOT `u == DBL_MAX`.
    // Expected optimum: (3, 0, 0.25).
    {
        WeightedQuadratic f(qCenter(), qWeight());
        runBounded("quadratic_mixed_bounded_unbounded", f,
                   Array{-INF_BOUND, 0.0, -INF_BOUND}, Array{INF_BOUND, 1.0, 0.25},
                   Array{0.0, 0.5, 0.0}, kMem, kPgTol, kFTol(), ec);
    }
}

// The two-Array constructor OVERRIDES the problem's constraint — it does not
// intersect with it. The second case WIDENS a [0,1]^3 constraint back to
// [-10,10]^3 and recovers the interior optimum, which an intersecting
// implementation could never do.
void block_explicit_bounds() {
    const EndCriteria ec = stdEndCriteria();
    {
        WeightedQuadratic f(qCenter(), qWeight());
        const Array lo(3, 0.0), hi(3, 1.0), x0(3, 0.5);
        NoConstraint c; // says "unbounded"
        Problem problem(f, c, x0);
        LBFGSB optimizer(lo, hi, kMem, kPgTol, kFTol());
        const EndCriteria::Type t = optimizer.minimize(problem, ec);
        emitResult("explicit_bounds_ctor_overrides_constraint", problem, t, f, lo, hi);
    }
    {
        WeightedQuadratic f(qCenter(), qWeight());
        const Array ctorLo(3, -10.0), ctorHi(3, 10.0), x0(3, 0.5);
        NonhomogeneousBoundaryConstraint c(Array(3, 0.0), Array(3, 1.0)); // ignored
        Problem problem(f, c, x0);
        LBFGSB optimizer(ctorLo, ctorHi, kMem, kPgTol, kFTol());
        const EndCriteria::Type t = optimizer.minimize(problem, ec);
        emitResult("explicit_bounds_ctor_widens_constraint", problem, t, f, ctorLo, ctorHi);
    }
}

void block_stopping_criteria() {
    // EndCriteria::maxIterations stop. maxStationaryStateIterations must be
    // > 1 and < maxIterations, so 3 is the smallest usable cap.
    {
        RosenbrockFunction f;
        const EndCriteria capped(3, 2, 1e-12, 1e-12, 1e-10);
        runUnconstrained("rosenbrock_10d_max_iterations_stop", f, Array(10, -1.0),
                         kMem, kPgTol, kFTol(), capped);
    }

    // EndCriteria::checkZeroGradientNorm stop, distinct from LBFGSB's own
    // pgTol_ test: pgTol_ = 1e-12 while gradientNormEpsilon = 1e-2, so the
    // EndCriteria branch fires first. Both report ZeroGradientNorm, so only
    // the resulting x distinguishes them.
    {
        WeightedQuadratic f(qCenter(), qWeight());
        const EndCriteria loose(1000, 100, 1e-12, 1e-12, 1e-2);
        runBounded("quadratic_endcriteria_gradient_norm_stop", f, Array(3, -10.0),
                   Array(3, 10.0), Array(3, 0.0), kMem, 1e-12, kFTol(), loose);
    }
}

// Truncated-iteration trajectory on the bound-constrained Rosenbrock: running
// the same problem with maxIterations = 3..12 exposes the iterate sequence
// step by step. Strongest available proxy for the private Cauchy point +
// subspace minimization; a wrong inner loop diverges at the first iterate.
void block_trajectory() {
    const Array lo(2, -2.0), hi(2, 0.5), x0(2, -1.0);
    std::vector<long> ks;
    std::vector<Real> xs, fs;
    std::vector<long> ecs, nfevs;

    for (Size k = 3; k <= 12; ++k) {
        RosenbrockFunction f;
        NonhomogeneousBoundaryConstraint c(lo, hi);
        Problem problem(f, c, x0);
        LBFGSB optimizer(kMem, kPgTol, kFTol());
        const EndCriteria capped(k, 2, 1e-12, 1e-12, 1e-10);
        const EndCriteria::Type t = optimizer.minimize(problem, capped);

        const Array x = problem.currentValue();
        ks.push_back(static_cast<long>(k));
        for (Size i = 0; i < x.size(); ++i)
            xs.push_back(x[i]);
        fs.push_back(problem.functionValue());
        ecs.push_back(static_cast<long>(t));
        nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
    }

    emit_iarr("rosenbrock_2d_bounded_iterate_trajectory_k", ks);
    emit_int("rosenbrock_2d_bounded_iterate_trajectory_n", 2);
    emit_arr("rosenbrock_2d_bounded_iterate_trajectory_x", xs); // row-major, n per step
    emit_arr("rosenbrock_2d_bounded_iterate_trajectory_f", fs);
    emit_iarr("rosenbrock_2d_bounded_iterate_trajectory_ec", ecs);
    emit_iarr("rosenbrock_2d_bounded_iterate_trajectory_nfev", nfevs);
}

// Full objective-evaluation traces. Entry 0 is the initial evaluation at the
// (clipped) start point; every later entry is a Wolfe line-search trial point,
// in call order.
void block_traces() {
    const EndCriteria ec = stdEndCriteria();
    {
        Traced<WeightedQuadratic> f(Array{10.0, 10.0}, Array{1.0, 100.0});
        const Array lo(2, 0.0), hi(2, 1.0), x0{0.9, 0.1};
        NonhomogeneousBoundaryConstraint c(lo, hi);
        Problem problem(f, c, x0);
        LBFGSB optimizer(kMem, kPgTol, kFTol());
        const EndCriteria::Type t = optimizer.minimize(problem, ec);

        emitResult("quadratic_two_active_eval_trace", problem, t, f, lo, hi);

        std::vector<Real> flat;
        for (const Array& a : f.tracedX())
            for (Size i = 0; i < a.size(); ++i)
                flat.push_back(a[i]);
        emit_int("quadratic_two_active_eval_trace_count",
                 static_cast<long>(f.tracedX().size()));
        emit_int("quadratic_two_active_eval_trace_n", 2);
        emit_arr("quadratic_two_active_eval_trace_points", flat); // row-major
        emit_arr("quadratic_two_active_eval_trace_values", f.tracedF());
    }
    {
        Traced<RosenbrockFunction> f;
        const Array lo(2, -2.0), hi(2, 0.5), x0(2, -1.0);
        NonhomogeneousBoundaryConstraint c(lo, hi);
        Problem problem(f, c, x0);
        LBFGSB optimizer(kMem, kPgTol, kFTol());
        const EndCriteria::Type t = optimizer.minimize(problem, ec);

        emitResult("rosenbrock_2d_bounded_eval_trace", problem, t, f, lo, hi);

        std::vector<Real> flat;
        for (const Array& a : f.tracedX())
            for (Size i = 0; i < a.size(); ++i)
                flat.push_back(a[i]);
        emit_int("rosenbrock_2d_bounded_eval_trace_count",
                 static_cast<long>(f.tracedX().size()));
        emit_int("rosenbrock_2d_bounded_eval_trace_n", 2);
        emit_arr("rosenbrock_2d_bounded_eval_trace_points", flat); // row-major
        emit_arr("rosenbrock_2d_bounded_eval_trace_values", f.tracedF());
    }
}

// Constructor / minimize argument validation. Message text is checked by
// substring so the result is independent of whether the build prefixes errors
// with file and line.
void block_validation() {
    bool zeroMemoryThrows = false, zeroMemoryMessageMatches = false;
    try {
        LBFGSB bad(0);
        (void)bad;
    } catch (const std::exception& e) {
        zeroMemoryThrows = true;
        zeroMemoryMessageMatches =
            std::string(e.what()).find("memory must be positive") != std::string::npos;
    }

    bool mismatchedBoundsThrows = false, mismatchedBoundsMessageMatches = false;
    try {
        LBFGSB bad(Array(2, 0.0), Array(3, 1.0), 10);
        (void)bad;
    } catch (const std::exception& e) {
        mismatchedBoundsThrows = true;
        mismatchedBoundsMessageMatches =
            std::string(e.what()).find("lower and upper bound sizes are inconsistent") !=
            std::string::npos;
    }

    // Bounds whose size disagrees with the problem dimension are rejected
    // inside minimize(), not by the constructor.
    bool wrongDimensionThrows = false, wrongDimensionMessageMatches = false;
    try {
        WeightedQuadratic f(Array{1.0, 2.0}, Array{1.0, 1.0});
        NoConstraint c;
        Problem problem(f, c, Array(2, 0.0));
        LBFGSB bad(Array(3, 0.0), Array(3, 1.0));
        bad.minimize(problem, stdEndCriteria());
    } catch (const std::exception& e) {
        wrongDimensionThrows = true;
        wrongDimensionMessageMatches =
            std::string(e.what()).find("bounds size does not match the number of variables") !=
            std::string::npos;
    }

    emit_bool("validation_zero_memory_throws", zeroMemoryThrows);
    emit_bool("validation_zero_memory_message_matches", zeroMemoryMessageMatches);
    emit_bool("validation_mismatched_bounds_throws", mismatchedBoundsThrows);
    emit_bool("validation_mismatched_bounds_message_matches", mismatchedBoundsMessageMatches);
    emit_bool("validation_wrong_dimension_throws", wrongDimensionThrows);
    emit_bool("validation_wrong_dimension_message_matches", wrongDimensionMessageMatches);
}

// Constants a port has to reproduce exactly.
void block_constants() {
    emit("default_pg_tol", 1e-8);
    emit("default_f_tol", 1e7 * QL_EPSILON);
    emit_int("default_memory", 10);
    // LBFGSB stores factr_ = fTol / QL_EPSILON and later tests
    // (fOld - f) <= factr_ * QL_EPSILON * denom. A port must reproduce that
    // round-trip literally, not collapse it to fTol * denom.
    emit("default_factr", (1e7 * QL_EPSILON) / QL_EPSILON);
    emit("tight_f_tol", kFTol());
    emit("tight_factr", kFTol() / QL_EPSILON);
    emit("ql_epsilon", QL_EPSILON);
    emit("ql_max_real", QL_MAX_REAL);
    emit("no_bound_threshold", 0.5 * QL_MAX_REAL);
    emit("line_search_c1", 1e-4);
    emit("line_search_c2", 0.9);
    emit_int("line_search_max_iterations", 30);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    block_constants();
    block_unconstrained();
    block_bounded();
    block_explicit_bounds();
    block_stopping_criteria();
    block_trajectory();
    block_traces();
    block_validation();
    std::cout << "\n}" << std::endl;
    return 0;
}
