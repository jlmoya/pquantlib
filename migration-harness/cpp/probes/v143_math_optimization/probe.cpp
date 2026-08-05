// migration-harness/cpp/probes/v143_math_optimization/probe.cpp
//
// Reference values for ql/math/optimization/** @ v1.43 (6b57206e).
//
// WHAT NEEDS PINNING, AND WHY
// ---------------------------
// Every optimizer in this package is an *iteration scheme*. Two implementations
// can converge to the same minimizer while taking completely different paths,
// evaluating the objective a different number of times, and returning a
// different EndCriteria::Type. A port that only matches the converged x is
// unverified. So for every optimizer this probe pins, in order of increasing
// strength:
//
//   1. the converged point x and the published Problem::functionValue();
//   2. the terminal EndCriteria::Type (an int AND its spelled-out name);
//   3. Problem::functionEvaluation() / gradientEvaluation() — the structural
//      diagnostic: same x with different counts means a different inner loop;
//   4. the ITERATE TRAJECTORY under a truncated iteration cap. Running the same
//      problem with EndCriteria::maxIterations = 3, 4, ... exposes the iterate
//      sequence one step at a time. This is what separates "converged to the
//      same place" from "took the same path".
//
// Note on (4) for the line-search based methods: LineSearchBasedMethod::minimize
// hands the SAME EndCriteria to the inner line search, and ArmijoLineSearch /
// GoldsteinLineSearch call endCriteria.checkMaxIterations(loopNumber, ...) with
// their own backtracking counter. So a small maxIterations also starves the
// inner backtracking loop and flips succeed() to false. That coupling is real
// v1.43 behaviour and the trajectory block pins it deliberately.
//
// Blocks
// ------
//   A  Constraint::update + LineSearch::update      (the halving loops)
//   B  SteepestDescent / ConjugateGradient / BFGS x Armijo / Goldstein
//   C  Simplex (incl. the exact setup pquantlib's existing test uses)
//   D  LevenbergMarquardt (MINPACK lmdif; incl. useCostFunctionsJacobian)
//   E  DifferentialEvolution (all 7 strategies x 3 crossover types, seeded)
//   F  SimulatedAnnealing (both cooling schemes, seeded)
//   G  LeastSquareFunction + NonLinearLeastSquare
//   H  Projection / ProjectedCostFunction / ProjectedConstraint
//   I  CompositeConstraint
//   J  EndCriteria checkers (the statStateIterations in-out counter) + ctor
//   K  CostFunction::jacobian / valuesAndJacobian / SimpleCostFunction
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/optimization.json.

#include <ql/errors.hpp>
#include <ql/math/array.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/optimization/armijo.hpp>
#include <ql/math/optimization/bfgs.hpp>
#include <ql/math/optimization/conjugategradient.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/costfunction.hpp>
#include <ql/math/optimization/differentialevolution.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/goldstein.hpp>
#include <ql/math/optimization/leastsquare.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/math/optimization/linesearch.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/math/optimization/projectedconstraint.hpp>
#include <ql/math/optimization/projectedcostfunction.hpp>
#include <ql/math/optimization/projection.hpp>
#include <ql/math/optimization/simplex.hpp>
#include <ql/math/optimization/simulatedannealing.hpp>
#include <ql/math/optimization/steepestdescent.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/utilities/null.hpp>

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

void emit(const std::string& name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
}

void emit_int(const std::string& name, long v) {
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

void emit_iarr(const std::string& name, const std::vector<long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

void emit_sarr(const std::string& name, const std::vector<std::string>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << "\"" << v[i] << "\"";
    }
    std::cout << "]";
}

std::vector<Real> toVec(const Array& a) { return std::vector<Real>(a.begin(), a.end()); }

std::string key(const std::string& prefix, const char* field) { return prefix + "_" + field; }

std::string ecName(EndCriteria::Type t) {
    std::ostringstream os;
    os << t; // QuantLib::operator<<(std::ostream&, EndCriteria::Type)
    return os.str();
}

// Row-major flattening of a Matrix, so the Python test can reshape.
std::vector<Real> flatten(const Matrix& m) {
    std::vector<Real> out;
    out.reserve(m.rows() * m.columns());
    for (Size i = 0; i < m.rows(); ++i)
        for (Size j = 0; j < m.columns(); ++j)
            out.push_back(m[i][j]);
    return out;
}

// --------------------------------------------------------------------------
// Cost functions. Every one of these is trivially reproducible in Python so
// the test does not have to restate literals -- it rebuilds the same objective.
// --------------------------------------------------------------------------

// Extended Rosenbrock with an ANALYTIC gradient; global minimum 0 at (1,...,1).
// value() is overridden, so `value` is the Rosenbrock function itself (NOT the
// CostFunction default sqrt(mean(values^2))).
class RosenbrockAnalytic : public CostFunction {
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

// Separable quadratic sum_i w_i (x_i - c_i)^2, analytic gradient.
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

// Rosenbrock as a RESIDUAL VECTOR, with NO overrides: this exercises the
// CostFunction defaults -- value() == sqrt(mean(values^2)), gradient() ==
// central difference with finiteDifferenceEpsilon() == 1e-8, jacobian()
// likewise. Levenberg-Marquardt and pquantlib's existing Simplex test both
// use exactly this shape.
class RosenbrockResiduals : public CostFunction {
  public:
    Array values(const Array& x) const override {
        Array r(2);
        r[0] = 1.0 - x[0];
        r[1] = 10.0 * (x[1] - x[0] * x[0]);
        return r;
    }
};

// Exponential fit: residual_i = a*exp(b*t_i) - y_i, m = 8 residuals, n = 2.
// The data is generated from (a,b) = (2.5, -0.35) with a fixed perturbation
// so the least-squares minimum is near, but not at, the generating parameters.
const Real kExpT[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
const Real kExpY[8] = {2.55, 2.05, 1.79, 1.44, 1.26, 1.02, 0.88, 0.71};

class ExpFitResiduals : public CostFunction {
  public:
    Array values(const Array& x) const override {
        Array r(8);
        for (Size i = 0; i < 8; ++i)
            r[i] = x[0] * std::exp(x[1] * kExpT[i]) - kExpY[i];
        return r;
    }
};

// The same exponential fit expressed as a LeastSquareProblem: target b_i = y_i,
// fct2fit_i = a*exp(b*t_i), gradient d fct2fit_i / d x_j.
class ExpFitLSP : public LeastSquareProblem {
  public:
    Size size() override { return 8; }
    void targetAndValue(const Array& x, Array& target, Array& fct2fit) override {
        for (Size i = 0; i < 8; ++i) {
            target[i] = kExpY[i];
            fct2fit[i] = x[0] * std::exp(x[1] * kExpT[i]);
        }
    }
    void targetValueAndGradient(const Array& x,
                                Matrix& grad_fct2fit,
                                Array& target,
                                Array& fct2fit) override {
        for (Size i = 0; i < 8; ++i) {
            const Real e = std::exp(x[1] * kExpT[i]);
            target[i] = kExpY[i];
            fct2fit[i] = x[0] * e;
            grad_fct2fit[i][0] = e;
            grad_fct2fit[i][1] = x[0] * kExpT[i] * e;
        }
    }
};

// Concrete LineSearch used only to reach the (non-virtual, but protected-state)
// LineSearch::update halving loop from outside.
class ProbeLineSearch : public LineSearch {
  public:
    Real operator()(Problem&, EndCriteria::Type&, const EndCriteria&, Real t_ini) override {
        return t_ini;
    }
};

// --------------------------------------------------------------------------
// Shared emission helper for an optimizer run
// --------------------------------------------------------------------------

void emitRun(const std::string& prefix, Problem& p, EndCriteria::Type ec) {
    emit_arr(key(prefix, "x"), toVec(p.currentValue()));
    emit(key(prefix, "f"), p.functionValue());
    emit(key(prefix, "grad_norm_value"), p.gradientNormValue());
    emit_int(key(prefix, "ec"), static_cast<long>(ec));
    emit_str(key(prefix, "ec_name"), ecName(ec));
    emit_int(key(prefix, "nfev"), static_cast<long>(p.functionEvaluation()));
    emit_int(key(prefix, "ngev"), static_cast<long>(p.gradientEvaluation()));
}

EndCriteria stdEC() { return EndCriteria(1000, 100, 1e-12, 1e-12, 1e-10); }

// ==========================================================================
// Block A -- Constraint::update and LineSearch::update
// ==========================================================================

void block_a() {
    std::vector<std::string> cases;

    // (1) NoConstraint: never halves, diff == beta exactly.
    {
        const std::string p = "cupdate_noconstraint";
        cases.push_back(p);
        NoConstraint c;
        Array params{1.0, 2.0};
        Array dir{-1.0, 0.5};
        const Real diff = c.update(params, dir, 3.0);
        emit(key(p, "diff"), diff);
        emit_arr(key(p, "params"), toVec(params));
    }
    // (2) PositiveConstraint, direction that overshoots into negatives:
    //     beta must be halved until every component stays > 0.
    {
        const std::string p = "cupdate_positive_halved";
        cases.push_back(p);
        PositiveConstraint c;
        Array params{1.0, 4.0};
        Array dir{-1.0, -1.0};
        const Real diff = c.update(params, dir, 8.0);
        emit(key(p, "diff"), diff);
        emit_arr(key(p, "params"), toVec(params));
    }
    // (3) BoundaryConstraint [0,1], overshoot on the upper side.
    {
        const std::string p = "cupdate_boundary_halved";
        cases.push_back(p);
        BoundaryConstraint c(0.0, 1.0);
        Array params{0.25, 0.25};
        Array dir{1.0, 0.0};
        const Real diff = c.update(params, dir, 5.0);
        emit(key(p, "diff"), diff);
        emit_arr(key(p, "params"), toVec(params));
    }
    // (4) LineSearch::update -- same halving loop, but reached through the
    //     LineSearch base class (it is a separate copy of the code in v1.43).
    {
        const std::string p = "lsupdate_positive_halved";
        cases.push_back(p);
        ProbeLineSearch ls;
        PositiveConstraint c;
        Array params{1.0, 4.0};
        Array dir{-1.0, -1.0};
        const Real diff = ls.update(params, dir, 8.0, c);
        emit(key(p, "diff"), diff);
        emit_arr(key(p, "params"), toVec(params));
    }
    // (5) LineSearch::update with NoConstraint (identity path).
    {
        const std::string p = "lsupdate_noconstraint";
        cases.push_back(p);
        ProbeLineSearch ls;
        NoConstraint c;
        Array params{-1.0, 0.0, 2.0};
        Array dir{0.5, -0.25, 1.0};
        const Real diff = ls.update(params, dir, 1.0, c);
        emit(key(p, "diff"), diff);
        emit_arr(key(p, "params"), toVec(params));
    }

    emit_sarr("block_a_cases", cases);
}

// ==========================================================================
// Block B -- LineSearchBasedMethod family
// ==========================================================================

enum class Method { SteepestDescent_, ConjugateGradient_, BFGS_ };
enum class LS { ArmijoDefault, ArmijoCustom, GoldsteinDefault, GoldsteinCustom };

ext::shared_ptr<LineSearch> makeLS(LS which) {
    switch (which) {
      case LS::ArmijoDefault:
        return ext::shared_ptr<LineSearch>(new ArmijoLineSearch);
      case LS::ArmijoCustom:
        return ext::shared_ptr<LineSearch>(new ArmijoLineSearch(1e-8, 0.2, 0.5));
      case LS::GoldsteinDefault:
        return ext::shared_ptr<LineSearch>(new GoldsteinLineSearch);
      case LS::GoldsteinCustom:
        return ext::shared_ptr<LineSearch>(new GoldsteinLineSearch(1e-8, 0.1, 0.7, 2.0));
    }
    QL_FAIL("unreachable");
}

ext::shared_ptr<OptimizationMethod> makeMethod(Method m, const ext::shared_ptr<LineSearch>& ls) {
    switch (m) {
      case Method::SteepestDescent_:
        return ext::shared_ptr<OptimizationMethod>(new SteepestDescent(ls));
      case Method::ConjugateGradient_:
        return ext::shared_ptr<OptimizationMethod>(new ConjugateGradient(ls));
      case Method::BFGS_:
        return ext::shared_ptr<OptimizationMethod>(new BFGS(ls));
    }
    QL_FAIL("unreachable");
}

void runLSBM(const std::string& prefix,
             Method m,
             LS lsKind,
             CostFunction& f,
             Constraint& c,
             const Array& x0,
             const EndCriteria& ec) {
    Problem problem(f, c, x0);
    ext::shared_ptr<OptimizationMethod> om = makeMethod(m, makeLS(lsKind));
    const EndCriteria::Type t = om->minimize(problem, ec);
    emitRun(prefix, problem, t);
}

void block_b() {
    std::vector<std::string> cases;
    const EndCriteria ec = stdEC();

    struct MEntry { const char* name; Method m; };
    struct LEntry { const char* name; LS ls; };
    const MEntry methods[] = {{"sd", Method::SteepestDescent_},
                              {"cg", Method::ConjugateGradient_},
                              {"bfgs", Method::BFGS_}};
    const LEntry searches[] = {{"armijo", LS::ArmijoDefault},
                               {"armijoc", LS::ArmijoCustom},
                               {"goldstein", LS::GoldsteinDefault},
                               {"goldsteinc", LS::GoldsteinCustom}};

    for (const auto& me : methods) {
        for (const auto& le : searches) {
            // (1) 2-D Rosenbrock, analytic gradient, unconstrained.
            {
                const std::string p = std::string("lsbm_") + me.name + "_" + le.name + "_rosen2d";
                cases.push_back(p);
                RosenbrockAnalytic f;
                NoConstraint c;
                runLSBM(p, me.m, le.ls, f, c, Array{-1.2, 1.0}, ec);
            }
            // (2) 3-D weighted quadratic, analytic gradient, unconstrained.
            {
                const std::string p = std::string("lsbm_") + me.name + "_" + le.name + "_quad3d";
                cases.push_back(p);
                WeightedQuadratic f(Array{3.0, -2.0, 0.5}, Array{1.0, 4.0, 0.25});
                NoConstraint c;
                runLSBM(p, me.m, le.ls, f, c, Array(3, 0.0), ec);
            }
            // (3) residual-only Rosenbrock: objective is sqrt(mean(r^2)) and
            //     the gradient is the CostFunction central difference. This is
            //     the branch where finiteDifferenceEpsilon() matters.
            {
                const std::string p = std::string("lsbm_") + me.name + "_" + le.name + "_rosenres";
                cases.push_back(p);
                RosenbrockResiduals f;
                NoConstraint c;
                runLSBM(p, me.m, le.ls, f, c, Array{-1.2, 1.0}, ec);
            }
            // (4) constrained: PositiveConstraint forces the update() halving
            //     loop inside the line search.
            {
                const std::string p = std::string("lsbm_") + me.name + "_" + le.name + "_quadpos";
                cases.push_back(p);
                WeightedQuadratic f(Array{3.0, 2.0}, Array{1.0, 4.0});
                PositiveConstraint c;
                runLSBM(p, me.m, le.ls, f, c, Array{0.5, 0.5}, ec);
            }
        }
    }
    emit_sarr("block_b_cases", cases);

    // ---- truncated-iteration trajectories -------------------------------
    // maxIterations = 3..12 with maxStationaryStateIterations = 2. Note this
    // also caps the inner backtracking loop of the line search.
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs, ngevs;
        for (long k = 3; k <= 12; ++k) {
            RosenbrockAnalytic f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            ConjugateGradient om(makeLS(LS::ArmijoDefault));
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-10);
            const EndCriteria::Type t = om.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
            ngevs.push_back(static_cast<long>(problem.gradientEvaluation()));
        }
        emit_iarr("lsbm_cg_traj_k", ks);
        emit_int("lsbm_cg_traj_n", 2);
        emit_arr("lsbm_cg_traj_x", xs);
        emit_arr("lsbm_cg_traj_f", fs);
        emit_iarr("lsbm_cg_traj_ec", ecs);
        emit_iarr("lsbm_cg_traj_nfev", nfevs);
        emit_iarr("lsbm_cg_traj_ngev", ngevs);
    }
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs, ngevs;
        for (long k = 3; k <= 12; ++k) {
            RosenbrockAnalytic f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            BFGS om(makeLS(LS::ArmijoDefault));
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-10);
            const EndCriteria::Type t = om.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
            ngevs.push_back(static_cast<long>(problem.gradientEvaluation()));
        }
        emit_iarr("lsbm_bfgs_traj_k", ks);
        emit_int("lsbm_bfgs_traj_n", 2);
        emit_arr("lsbm_bfgs_traj_x", xs);
        emit_arr("lsbm_bfgs_traj_f", fs);
        emit_iarr("lsbm_bfgs_traj_ec", ecs);
        emit_iarr("lsbm_bfgs_traj_nfev", nfevs);
        emit_iarr("lsbm_bfgs_traj_ngev", ngevs);
    }
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs, ngevs;
        for (long k = 3; k <= 12; ++k) {
            WeightedQuadratic f(Array{3.0, -2.0, 0.5}, Array{1.0, 4.0, 0.25});
            NoConstraint c;
            Problem problem(f, c, Array(3, 0.0));
            SteepestDescent om(makeLS(LS::GoldsteinDefault));
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-10);
            const EndCriteria::Type t = om.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
            ngevs.push_back(static_cast<long>(problem.gradientEvaluation()));
        }
        emit_iarr("lsbm_sd_gold_traj_k", ks);
        emit_int("lsbm_sd_gold_traj_n", 3);
        emit_arr("lsbm_sd_gold_traj_x", xs);
        emit_arr("lsbm_sd_gold_traj_f", fs);
        emit_iarr("lsbm_sd_gold_traj_ec", ecs);
        emit_iarr("lsbm_sd_gold_traj_nfev", nfevs);
        emit_iarr("lsbm_sd_gold_traj_ngev", ngevs);
    }
    // Two DENSE trajectories (k = 3 .. 60) over the configurations whose
    // fully converged runs are the most chaotic. Their purpose is to
    // separate "the port took a different path" (which shows up within the
    // first few iterations) from "the same path amplified a rounding
    // difference over a thousand evaluations" (which only shows up late).
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs, ngevs;
        for (long k = 3; k <= 60; ++k) {
            RosenbrockAnalytic f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            ConjugateGradient om(makeLS(LS::ArmijoCustom));
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-10);
            const EndCriteria::Type t = om.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
            ngevs.push_back(static_cast<long>(problem.gradientEvaluation()));
        }
        emit_iarr("lsbm_cg_armijoc_rosen2d_dense_k", ks);
        emit_int("lsbm_cg_armijoc_rosen2d_dense_n", 2);
        emit_arr("lsbm_cg_armijoc_rosen2d_dense_x", xs);
        emit_arr("lsbm_cg_armijoc_rosen2d_dense_f", fs);
        emit_iarr("lsbm_cg_armijoc_rosen2d_dense_ec", ecs);
        emit_iarr("lsbm_cg_armijoc_rosen2d_dense_nfev", nfevs);
        emit_iarr("lsbm_cg_armijoc_rosen2d_dense_ngev", ngevs);
    }
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs, ngevs;
        for (long k = 3; k <= 60; ++k) {
            RosenbrockResiduals f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            BFGS om(makeLS(LS::ArmijoDefault));
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-10);
            const EndCriteria::Type t = om.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
            ngevs.push_back(static_cast<long>(problem.gradientEvaluation()));
        }
        emit_iarr("lsbm_bfgs_armijo_rosenres_dense_k", ks);
        emit_int("lsbm_bfgs_armijo_rosenres_dense_n", 2);
        emit_arr("lsbm_bfgs_armijo_rosenres_dense_x", xs);
        emit_arr("lsbm_bfgs_armijo_rosenres_dense_f", fs);
        emit_iarr("lsbm_bfgs_armijo_rosenres_dense_ec", ecs);
        emit_iarr("lsbm_bfgs_armijo_rosenres_dense_nfev", nfevs);
        emit_iarr("lsbm_bfgs_armijo_rosenres_dense_ngev", ngevs);
    }
}

// ==========================================================================
// Block C -- Simplex
// ==========================================================================

void block_c() {
    std::vector<std::string> cases;

    // (1) THE EXACT SETUP pquantlib's existing (scipy-backed) Simplex test
    //     uses: residual Rosenbrock, x0 = (-1.2, 1), lambda = 0.1,
    //     EndCriteria(10000, 1000, 1e-12, 1e-12, 1e-12).
    {
        const std::string p = "simplex_rosenres_lambda0p1";
        cases.push_back(p);
        RosenbrockResiduals f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        Simplex s(0.1);
        const EndCriteria ec(10000, 1000, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, s.minimize(problem, ec));
    }
    // (2) analytic Rosenbrock, lambda = 0.1.
    {
        const std::string p = "simplex_rosen2d_lambda0p1";
        cases.push_back(p);
        RosenbrockAnalytic f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        Simplex s(0.1);
        emitRun(p, problem, s.minimize(problem, stdEC()));
    }
    // (3) lambda = 1.0 -- a different initial simplex, hence a different path.
    {
        const std::string p = "simplex_rosen2d_lambda1";
        cases.push_back(p);
        RosenbrockAnalytic f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        Simplex s(1.0);
        emitRun(p, problem, s.minimize(problem, stdEC()));
    }
    // (4) 1-D quadratic, the degenerate n = 1 simplex (2 vertices).
    {
        const std::string p = "simplex_quad1d";
        cases.push_back(p);
        WeightedQuadratic f(Array{3.0}, Array{1.0});
        NoConstraint c;
        Problem problem(f, c, Array(1, 0.0));
        Simplex s(0.5);
        emitRun(p, problem, s.minimize(problem, stdEC()));
    }
    // (5) 3-D quadratic.
    {
        const std::string p = "simplex_quad3d";
        cases.push_back(p);
        WeightedQuadratic f(Array{3.0, -2.0, 0.5}, Array{1.0, 4.0, 0.25});
        NoConstraint c;
        Problem problem(f, c, Array(3, 0.0));
        Simplex s(1.0);
        emitRun(p, problem, s.minimize(problem, stdEC()));
    }
    // (6) PositiveConstraint: the initial-simplex construction goes through
    //     Constraint::update, and extrapolate() rejects infeasible trials by
    //     halving `factor`.
    {
        const std::string p = "simplex_quadpos";
        cases.push_back(p);
        WeightedQuadratic f(Array{3.0, 2.0}, Array{1.0, 4.0});
        PositiveConstraint c;
        Problem problem(f, c, Array{0.5, 0.5});
        Simplex s(1.0);
        emitRun(p, problem, s.minimize(problem, stdEC()));
    }
    // (7) BoundaryConstraint where the optimum is OUTSIDE the box, so the
    //     simplex is driven onto the boundary and extrapolation eventually
    //     collapses (the `fabs(factor) <= QL_EPSILON` exit that returns
    //     StationaryFunctionValue).
    {
        const std::string p = "simplex_quad_boxed";
        cases.push_back(p);
        WeightedQuadratic f(Array{5.0, 5.0}, Array{1.0, 1.0});
        BoundaryConstraint c(0.0, 1.0);
        Problem problem(f, c, Array{0.5, 0.5});
        Simplex s(0.25);
        emitRun(p, problem, s.minimize(problem, stdEC()));
    }
    // (8) loose rootEpsilon: stops on simplexSize < xtol well before machine
    //     precision, so the returned x is NOT the exact minimizer.
    {
        const std::string p = "simplex_rosen2d_loose_xtol";
        cases.push_back(p);
        RosenbrockAnalytic f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        Simplex s(0.1);
        const EndCriteria ec(10000, 1000, 1e-4, 1e-12, 1e-12);
        emitRun(p, problem, s.minimize(problem, ec));
    }
    emit_sarr("block_c_cases", cases);

    // ---- truncated-iteration trajectory ---------------------------------
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs;
        for (long k = 3; k <= 20; ++k) {
            RosenbrockAnalytic f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            Simplex s(0.1);
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-10);
            const EndCriteria::Type t = s.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
        }
        emit_iarr("simplex_traj_k", ks);
        emit_int("simplex_traj_n", 2);
        emit_arr("simplex_traj_x", xs);
        emit_arr("simplex_traj_f", fs);
        emit_iarr("simplex_traj_ec", ecs);
        emit_iarr("simplex_traj_nfev", nfevs);
    }
}

// ==========================================================================
// Block D -- LevenbergMarquardt
// ==========================================================================

void block_d() {
    std::vector<std::string> cases;

    // (1) THE EXACT SETUP pquantlib's existing (scipy-backed) LM test uses.
    //     NOTE: LevenbergMarquardt publishes functionValue() as
    //     costFunction().value(x), i.e. sqrt(mean(r^2)) -- NOT r.r.
    {
        const std::string p = "lm_rosenres";
        cases.push_back(p);
        RosenbrockResiduals f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        LevenbergMarquardt lm(1e-8, 1e-8, 1e-8);
        const EndCriteria ec(2000, 100, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, lm.minimize(problem, ec));
    }
    // (2) default ctor arguments.
    {
        const std::string p = "lm_rosenres_default";
        cases.push_back(p);
        RosenbrockResiduals f;
        NoConstraint c;
        Problem problem(f, c, Array{0.5, 0.5});
        LevenbergMarquardt lm;
        const EndCriteria ec(2000, 100, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, lm.minimize(problem, ec));
    }
    // (3) exponential fit, m = 8 > n = 2.
    {
        const std::string p = "lm_expfit";
        cases.push_back(p);
        ExpFitResiduals f;
        NoConstraint c;
        Problem problem(f, c, Array{1.0, -1.0});
        LevenbergMarquardt lm(1e-8, 1e-8, 1e-8);
        const EndCriteria ec(2000, 100, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, lm.minimize(problem, ec));
    }
    // (4) exponential fit with the cost function's own (central-difference)
    //     jacobian instead of lmdif's forward-difference fdjac2.
    {
        const std::string p = "lm_expfit_costjac";
        cases.push_back(p);
        ExpFitResiduals f;
        NoConstraint c;
        Problem problem(f, c, Array{1.0, -1.0});
        LevenbergMarquardt lm(1e-8, 1e-8, 1e-8, true);
        const EndCriteria ec(2000, 100, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, lm.minimize(problem, ec));
    }
    // (5) PositiveConstraint: exercises the fcn() penalty branch (1e10 in
    //     every residual) when lmdif probes an infeasible point.
    {
        const std::string p = "lm_expfit_positive";
        cases.push_back(p);
        ExpFitResiduals f;
        PositiveConstraint c;
        Problem problem(f, c, Array{1.0, 0.5});
        LevenbergMarquardt lm(1e-8, 1e-8, 1e-8);
        const EndCriteria ec(2000, 100, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, lm.minimize(problem, ec));
    }
    // (6) a starved maxIterations -> maxfev = maxIterations*(n+1) is hit and
    //     lmdif returns info == 5 -> EndCriteria::MaxIterations.
    {
        const std::string p = "lm_expfit_maxiter";
        cases.push_back(p);
        ExpFitResiduals f;
        NoConstraint c;
        Problem problem(f, c, Array{1.0, -1.0});
        LevenbergMarquardt lm(1e-8, 1e-8, 1e-8);
        const EndCriteria ec(3, 2, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, lm.minimize(problem, ec));
    }
    // (7) loose functionEpsilon: lmdif stops on ftol, giving a DIFFERENT
    //     converged point from case (3).
    {
        const std::string p = "lm_expfit_loose_ftol";
        cases.push_back(p);
        ExpFitResiduals f;
        NoConstraint c;
        Problem problem(f, c, Array{1.0, -1.0});
        LevenbergMarquardt lm(1e-8, 1e-8, 1e-8);
        const EndCriteria ec(2000, 100, 1e-12, 1e-3, 1e-12);
        emitRun(p, problem, lm.minimize(problem, ec));
    }
    emit_sarr("block_d_cases", cases);

    // ---- truncated-iteration trajectory ---------------------------------
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs;
        for (long k = 3; k <= 12; ++k) {
            ExpFitResiduals f;
            NoConstraint c;
            Problem problem(f, c, Array{1.0, -1.0});
            LevenbergMarquardt lm(1e-8, 1e-8, 1e-8);
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-12);
            const EndCriteria::Type t = lm.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
        }
        emit_iarr("lm_traj_k", ks);
        emit_int("lm_traj_n", 2);
        emit_arr("lm_traj_x", xs);
        emit_arr("lm_traj_f", fs);
        emit_iarr("lm_traj_ec", ecs);
        emit_iarr("lm_traj_nfev", nfevs);
    }
}

// ==========================================================================
// Block E -- DifferentialEvolution
// ==========================================================================

// A fixed initial population removes the fillInitialPopulation RNG draws from
// the comparison, isolating the generation loop; the "random init" cases put
// them back in.
std::vector<Array> fixedPopulation2D(Size n) {
    std::vector<Array> pop;
    pop.reserve(n);
    for (Size i = 0; i < n; ++i) {
        const Real a = -2.0 + 4.0 * Real(i) / Real(n - 1);
        const Real b = 2.0 - 4.0 * Real((i * 7) % n) / Real(n - 1);
        pop.emplace_back(Array{a, b});
    }
    return pop;
}

void block_e() {
    std::vector<std::string> cases;

    struct SEntry { const char* name; DifferentialEvolution::Strategy s; };
    const SEntry strategies[] = {
        {"rand1standard", DifferentialEvolution::Rand1Standard},
        {"bestjitter", DifferentialEvolution::BestMemberWithJitter},
        {"currtobest2", DifferentialEvolution::CurrentToBest2Diffs},
        {"pervectordither", DifferentialEvolution::Rand1DiffWithPerVectorDither},
        {"dither", DifferentialEvolution::Rand1DiffWithDither},
        {"eitheror", DifferentialEvolution::EitherOrWithOptimalRecombination},
        {"selfadaptrot", DifferentialEvolution::Rand1SelfadaptiveWithRotation}};

    struct CEntry { const char* name; DifferentialEvolution::CrossoverType c; };
    const CEntry crossovers[] = {{"normal", DifferentialEvolution::Normal},
                                 {"binomial", DifferentialEvolution::Binomial},
                                 {"exponential", DifferentialEvolution::Exponential}};

    const Size kPop = 20;
    const std::vector<Array> fixedPop = fixedPopulation2D(kPop);

    // Emit the fixed initial population so the Python test rebuilds it exactly.
    {
        std::vector<Real> flat;
        for (const auto& a : fixedPop)
            for (Real v : a) flat.push_back(v);
        emit_arr("de_fixed_population", flat);
        emit_int("de_fixed_population_members", static_cast<long>(kPop));
        emit_int("de_fixed_population_dim", 2);
    }

    for (const auto& se : strategies) {
        for (const auto& ce : crossovers) {
            const std::string p =
                std::string("de_") + se.name + "_" + ce.name + "_rosen2d";
            cases.push_back(p);
            RosenbrockAnalytic f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            DifferentialEvolution::Configuration conf =
                DifferentialEvolution::Configuration()
                    .withStrategy(se.s)
                    .withCrossoverType(ce.c)
                    .withSeed(42)
                    .withStepsizeWeight(0.4)
                    .withCrossoverProbability(0.9)
                    .withBounds(true)
                    .withLowerBound(Array{-5.0, -5.0})
                    .withUpperBound(Array{5.0, 5.0})
                    .withInitialPopulation(fixedPop);
            DifferentialEvolution de(conf);
            const EndCriteria ec(30, 20, 1e-12, 1e-12, 1e-12);
            emitRun(p, problem, de.minimize(problem, ec));
        }
    }

    // Random initial population (exercises fillInitialPopulation), bounds from
    // the constraint rather than from the Configuration.
    {
        const std::string p = "de_randominit_boundaryconstraint";
        cases.push_back(p);
        RosenbrockAnalytic f;
        BoundaryConstraint c(-5.0, 5.0);
        Problem problem(f, c, Array{-1.2, 1.0});
        DifferentialEvolution::Configuration conf =
            DifferentialEvolution::Configuration()
                .withStrategy(DifferentialEvolution::BestMemberWithJitter)
                .withCrossoverType(DifferentialEvolution::Normal)
                .withSeed(7)
                .withPopulationMembers(16)
                .withStepsizeWeight(0.2)
                .withCrossoverProbability(0.9)
                .withBounds(true);
        DifferentialEvolution de(conf);
        const EndCriteria ec(25, 20, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, de.minimize(problem, ec));
    }
    // Adaptive crossover (adaptCrossover consumes extra RNG draws).
    {
        const std::string p = "de_adaptive_crossover";
        cases.push_back(p);
        WeightedQuadratic f(Array{1.5, -0.5}, Array{1.0, 3.0});
        BoundaryConstraint c(-5.0, 5.0);
        Problem problem(f, c, Array{0.0, 0.0});
        DifferentialEvolution::Configuration conf =
            DifferentialEvolution::Configuration()
                .withStrategy(DifferentialEvolution::Rand1Standard)
                .withCrossoverType(DifferentialEvolution::Binomial)
                .withSeed(11)
                .withStepsizeWeight(0.5)
                .withCrossoverProbability(0.8)
                .withAdaptiveCrossover(true)
                .withBounds(true)
                .withInitialPopulation(fixedPop);
        DifferentialEvolution de(conf);
        const EndCriteria ec(25, 20, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, de.minimize(problem, ec));
    }
    // applyBounds == false: the mirror/clamp branch is skipped entirely.
    {
        const std::string p = "de_no_bounds";
        cases.push_back(p);
        WeightedQuadratic f(Array{1.5, -0.5}, Array{1.0, 3.0});
        NoConstraint c;
        Problem problem(f, c, Array{0.0, 0.0});
        DifferentialEvolution::Configuration conf =
            DifferentialEvolution::Configuration()
                .withStrategy(DifferentialEvolution::Rand1Standard)
                .withCrossoverType(DifferentialEvolution::Normal)
                .withSeed(13)
                .withStepsizeWeight(0.5)
                .withCrossoverProbability(0.9)
                .withBounds(false)
                .withLowerBound(Array{-5.0, -5.0})
                .withUpperBound(Array{5.0, 5.0})
                .withInitialPopulation(fixedPop);
        DifferentialEvolution de(conf);
        const EndCriteria ec(25, 20, 1e-12, 1e-12, 1e-12);
        emitRun(p, problem, de.minimize(problem, ec));
    }
    emit_sarr("block_e_cases", cases);

    // ---- truncated-generation trajectory --------------------------------
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs;
        for (long k = 3; k <= 12; ++k) {
            RosenbrockAnalytic f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            DifferentialEvolution::Configuration conf =
                DifferentialEvolution::Configuration()
                    .withStrategy(DifferentialEvolution::Rand1Standard)
                    .withCrossoverType(DifferentialEvolution::Normal)
                    .withSeed(42)
                    .withStepsizeWeight(0.4)
                    .withCrossoverProbability(0.9)
                    .withBounds(true)
                    .withLowerBound(Array{-5.0, -5.0})
                    .withUpperBound(Array{5.0, 5.0})
                    .withInitialPopulation(fixedPop);
            DifferentialEvolution de(conf);
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-12, 1e-12, 1e-12);
            const EndCriteria::Type t = de.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
        }
        emit_iarr("de_traj_k", ks);
        emit_int("de_traj_n", 2);
        emit_arr("de_traj_x", xs);
        emit_arr("de_traj_f", fs);
        emit_iarr("de_traj_ec", ecs);
        emit_iarr("de_traj_nfev", nfevs);
    }
}

// ==========================================================================
// Block F -- SimulatedAnnealing
// ==========================================================================

void block_f() {
    std::vector<std::string> cases;

    // ConstantFactor cooling: T *= (1 - epsilon) every m moves.
    {
        const std::string p = "sa_constantfactor_rosen2d";
        cases.push_back(p);
        RosenbrockAnalytic f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        SimulatedAnnealing<> sa(0.1, 10.0, Real(0.99), Size(10), MersenneTwisterUniformRng(42));
        const EndCriteria ec(500, 100, 1e-8, 1e-12, 1e-12);
        emitRun(p, problem, sa.minimize(problem, ec));
    }
    // ConstantBudget cooling: T = T0 * (1 - k/K)^alpha, zero after K moves.
    {
        const std::string p = "sa_constantbudget_rosen2d";
        cases.push_back(p);
        RosenbrockAnalytic f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        SimulatedAnnealing<> sa(0.1, 10.0, Size(200), 1.0, MersenneTwisterUniformRng(42));
        const EndCriteria ec(500, 100, 1e-8, 1e-12, 1e-12);
        emitRun(p, problem, sa.minimize(problem, ec));
    }
    // 3-D quadratic, ConstantFactor.
    {
        const std::string p = "sa_constantfactor_quad3d";
        cases.push_back(p);
        WeightedQuadratic f(Array{3.0, -2.0, 0.5}, Array{1.0, 4.0, 0.25});
        NoConstraint c;
        Problem problem(f, c, Array(3, 0.0));
        SimulatedAnnealing<> sa(1.0, 5.0, Real(0.95), Size(5), MersenneTwisterUniformRng(7));
        const EndCriteria ec(400, 100, 1e-8, 1e-12, 1e-12);
        emitRun(p, problem, sa.minimize(problem, ec));
    }
    // Constrained: PositiveConstraint drives both the initial-simplex
    // Constraint::update AND the QL_MAX_REAL rejection inside amotsa.
    {
        const std::string p = "sa_constantbudget_quadpos";
        cases.push_back(p);
        WeightedQuadratic f(Array{3.0, 2.0}, Array{1.0, 4.0});
        PositiveConstraint c;
        Problem problem(f, c, Array{0.5, 0.5});
        SimulatedAnnealing<> sa(1.0, 5.0, Size(150), 2.0, MersenneTwisterUniformRng(3));
        const EndCriteria ec(400, 100, 1e-8, 1e-12, 1e-12);
        emitRun(p, problem, sa.minimize(problem, ec));
    }
    // Zero initial temperature -> the log-perturbation term vanishes and the
    // method degenerates to a deterministic simplex (RNG still consumed).
    {
        const std::string p = "sa_zero_temperature";
        cases.push_back(p);
        RosenbrockAnalytic f;
        NoConstraint c;
        Problem problem(f, c, Array{-1.2, 1.0});
        SimulatedAnnealing<> sa(0.1, 0.0, Real(0.9), Size(4), MersenneTwisterUniformRng(5));
        const EndCriteria ec(300, 100, 1e-8, 1e-12, 1e-12);
        emitRun(p, problem, sa.minimize(problem, ec));
    }
    emit_sarr("block_f_cases", cases);

    // ---- truncated-iteration trajectory ---------------------------------
    {
        std::vector<long> ks;
        std::vector<Real> xs, fs;
        std::vector<long> ecs, nfevs;
        for (long k = 5; k <= 40; k += 5) {
            RosenbrockAnalytic f;
            NoConstraint c;
            Problem problem(f, c, Array{-1.2, 1.0});
            SimulatedAnnealing<> sa(0.1, 10.0, Real(0.99), Size(10), MersenneTwisterUniformRng(42));
            const EndCriteria kec(static_cast<Size>(k), 2, 1e-8, 1e-12, 1e-12);
            const EndCriteria::Type t = sa.minimize(problem, kec);
            ks.push_back(k);
            for (Real v : problem.currentValue()) xs.push_back(v);
            fs.push_back(problem.functionValue());
            ecs.push_back(static_cast<long>(t));
            nfevs.push_back(static_cast<long>(problem.functionEvaluation()));
        }
        emit_iarr("sa_traj_k", ks);
        emit_int("sa_traj_n", 2);
        emit_arr("sa_traj_x", xs);
        emit_arr("sa_traj_f", fs);
        emit_iarr("sa_traj_ec", ecs);
        emit_iarr("sa_traj_nfev", nfevs);
    }
}

// ==========================================================================
// Block G -- LeastSquareFunction + NonLinearLeastSquare
// ==========================================================================

void block_g() {
    // LeastSquareFunction pointwise: value == diff.diff, values == diff*diff
    // (ELEMENTWISE SQUARE -- not the raw residual), gradient ==
    // -2 * J^T diff.
    {
        ExpFitLSP lsp;
        LeastSquareFunction lsf(lsp);
        const Array x{2.0, -0.3};
        Array g(2, 0.0);
        const Real v = lsf.value(x);
        const Array vs = lsf.values(x);
        lsf.gradient(g, x);
        Array g2(2, 0.0);
        const Real v2 = lsf.valueAndGradient(g2, x);
        emit_arr("lsf_x", toVec(x));
        emit("lsf_value", v);
        emit_arr("lsf_values", toVec(vs));
        emit_arr("lsf_gradient", toVec(g));
        emit("lsf_value_and_gradient_value", v2);
        emit_arr("lsf_value_and_gradient_grad", toVec(g2));
    }
    // NonLinearLeastSquare with the DEFAULT optimizer (ConjugateGradient with
    // a default ArmijoLineSearch) and the default accuracy/maxiter.
    {
        ExpFitLSP lsp;
        NoConstraint c;
        NonLinearLeastSquare nlls(c);
        nlls.setInitialValue(Array{1.0, -1.0});
        const Array& res = nlls.perform(lsp);
        emit_arr("nlls_default_x", toVec(res));
        emit("nlls_default_resnorm", nlls.residualNorm());
        emit("nlls_default_lastvalue", nlls.lastValue());
        emit_int("nlls_default_exitflag", static_cast<long>(nlls.exitFlag()));
    }
    // NonLinearLeastSquare with an explicit accuracy/maxiter.
    {
        ExpFitLSP lsp;
        NoConstraint c;
        NonLinearLeastSquare nlls(c, 1e-8, 200);
        nlls.setInitialValue(Array{1.0, -1.0});
        const Array& res = nlls.perform(lsp);
        emit_arr("nlls_tight_x", toVec(res));
        emit("nlls_tight_resnorm", nlls.residualNorm());
        emit("nlls_tight_lastvalue", nlls.lastValue());
        emit_int("nlls_tight_exitflag", static_cast<long>(nlls.exitFlag()));
    }
    // NonLinearLeastSquare with an injected optimizer (Simplex).
    {
        ExpFitLSP lsp;
        NoConstraint c;
        NonLinearLeastSquare nlls(c, 1e-8, 200,
                                  ext::shared_ptr<OptimizationMethod>(new Simplex(0.1)));
        nlls.setInitialValue(Array{1.0, -1.0});
        const Array& res = nlls.perform(lsp);
        emit_arr("nlls_simplex_x", toVec(res));
        emit("nlls_simplex_resnorm", nlls.residualNorm());
        emit("nlls_simplex_lastvalue", nlls.lastValue());
        emit_int("nlls_simplex_exitflag", static_cast<long>(nlls.exitFlag()));
    }
    // ... and with LevenbergMarquardt, which reaches LeastSquareFunction
    // through values() -- i.e. it minimises the sum of SQUARED squares.
    {
        ExpFitLSP lsp;
        NoConstraint c;
        NonLinearLeastSquare nlls(c, 1e-8, 200,
                                  ext::shared_ptr<OptimizationMethod>(new LevenbergMarquardt));
        nlls.setInitialValue(Array{1.0, -1.0});
        const Array& res = nlls.perform(lsp);
        emit_arr("nlls_lm_x", toVec(res));
        emit("nlls_lm_resnorm", nlls.residualNorm());
        emit("nlls_lm_lastvalue", nlls.lastValue());
        emit_int("nlls_lm_exitflag", static_cast<long>(nlls.exitFlag()));
    }
}

// ==========================================================================
// Block H -- Projection / ProjectedCostFunction / ProjectedConstraint
// ==========================================================================

void block_h() {
    const Array full{1.0, 2.0, 3.0, 4.0};
    const std::vector<bool> fix{false, true, false, true}; // free: 0 and 2

    {
        Projection proj(full, fix);
        const Array projected = proj.project(Array{10.0, 20.0, 30.0, 40.0});
        const Array included = proj.include(Array{7.0, 8.0});
        emit_arr("projection_parameter_values", toVec(full));
        emit_iarr("projection_fix", std::vector<long>{0, 1, 0, 1});
        emit_arr("projection_project_in", std::vector<Real>{10.0, 20.0, 30.0, 40.0});
        emit_arr("projection_project_out", toVec(projected));
        emit_arr("projection_include_in", std::vector<Real>{7.0, 8.0});
        emit_arr("projection_include_out", toVec(included));
    }
    // Default (empty) fixParameters -> every parameter free.
    {
        Projection proj(full);
        emit_arr("projection_allfree_project_out",
                 toVec(proj.project(Array{10.0, 20.0, 30.0, 40.0})));
        emit_arr("projection_allfree_include_out",
                 toVec(proj.include(Array{5.0, 6.0, 7.0, 8.0})));
    }
    // ProjectedCostFunction over the residual Rosenbrock (n = 2), fixing x[1].
    {
        RosenbrockResiduals f;
        const Array base{0.3, 0.7};
        const std::vector<bool> fix2{false, true};
        ProjectedCostFunction pcf(f, base, fix2);
        const Array free{-1.2};
        emit_arr("pcf_base", toVec(base));
        emit_iarr("pcf_fix", std::vector<long>{0, 1});
        emit_arr("pcf_free", toVec(free));
        emit("pcf_value", pcf.value(free));
        emit_arr("pcf_values", toVec(pcf.values(free)));
        emit_arr("pcf_include", toVec(pcf.include(free)));
        emit_arr("pcf_project", toVec(pcf.project(Array{9.0, 8.0})));
        Array g(1, 0.0);
        pcf.gradient(g, free); // inherited CostFunction central difference
        emit_arr("pcf_gradient", toVec(g));
    }
    // ProjectedCostFunction built from an existing Projection object.
    {
        RosenbrockResiduals f;
        const Array base{0.3, 0.7};
        const std::vector<bool> fix2{true, false};
        Projection proj(base, fix2);
        ProjectedCostFunction pcf(f, proj);
        const Array free{1.4};
        emit_arr("pcf2_free", toVec(free));
        emit("pcf2_value", pcf.value(free));
        emit_arr("pcf2_values", toVec(pcf.values(free)));
    }
    // ProjectedConstraint over a NonhomogeneousBoundaryConstraint.
    {
        const Array lo{-1.0, -2.0, -3.0};
        const Array hi{1.0, 2.0, 3.0};
        NonhomogeneousBoundaryConstraint inner(lo, hi);
        const Array base{0.0, 5.0, 0.0}; // x[1] fixed OUTSIDE the box
        const std::vector<bool> fix3{false, true, false};
        ProjectedConstraint pc(inner, base, fix3);
        emit_arr("projconstraint_lo", toVec(lo));
        emit_arr("projconstraint_hi", toVec(hi));
        emit_arr("projconstraint_base", toVec(base));
        emit_iarr("projconstraint_fix", std::vector<long>{0, 1, 0});
        // base[1] = 5 > hi[1] = 2, so include() lands outside -> test false.
        emit_bool("projconstraint_test_infeasible_fixed", pc.test(Array{0.5, 0.5}));
        emit_arr("projconstraint_upper", toVec(pc.upperBound(Array{0.5, 0.5})));
        emit_arr("projconstraint_lower", toVec(pc.lowerBound(Array{0.5, 0.5})));
    }
    // ProjectedConstraint with a feasible fixed value + the Projection ctor.
    {
        const Array lo{-1.0, -2.0, -3.0};
        const Array hi{1.0, 2.0, 3.0};
        NonhomogeneousBoundaryConstraint inner(lo, hi);
        const Array base{0.0, 1.0, 0.0};
        const std::vector<bool> fix3{false, true, false};
        Projection proj(base, fix3);
        ProjectedConstraint pc(inner, proj);
        emit_bool("projconstraint2_test_inside", pc.test(Array{0.5, 0.5}));
        emit_bool("projconstraint2_test_outside", pc.test(Array{0.5, 9.0}));
        emit_arr("projconstraint2_upper", toVec(pc.upperBound(Array{0.5, 0.5})));
        emit_arr("projconstraint2_lower", toVec(pc.lowerBound(Array{0.5, 0.5})));
    }
}

// ==========================================================================
// Block I -- CompositeConstraint
// ==========================================================================

void block_i() {
    PositiveConstraint pos;
    BoundaryConstraint box(-1.0, 2.0);
    CompositeConstraint comp(pos, box);
    const Array probe{0.5, 1.5};

    emit_bool("composite_test_both_ok", comp.test(Array{0.5, 1.5}));
    emit_bool("composite_test_violates_positive", comp.test(Array{-0.5, 1.5}));
    emit_bool("composite_test_violates_boundary", comp.test(Array{0.5, 3.0}));
    // upper = min(+DBL_MAX, 2.0) = 2.0 ; lower = max(0.0, -1.0) = 0.0
    emit_arr("composite_upper", toVec(comp.upperBound(probe)));
    emit_arr("composite_lower", toVec(comp.lowerBound(probe)));

    // Nested composite: (Positive AND [-1,2]) AND [0.25, 5]
    BoundaryConstraint box2(0.25, 5.0);
    CompositeConstraint nested(comp, box2);
    emit_bool("composite_nested_test_ok", nested.test(Array{0.5, 1.5}));
    emit_bool("composite_nested_test_low", nested.test(Array{0.1, 1.5}));
    emit_arr("composite_nested_upper", toVec(nested.upperBound(probe)));
    emit_arr("composite_nested_lower", toVec(nested.lowerBound(probe)));

    // Composite over NoConstraint -> the +/-DBL_MAX defaults survive.
    NoConstraint none;
    CompositeConstraint openComp(none, none);
    emit_arr("composite_open_upper", toVec(openComp.upperBound(probe)));
    emit_arr("composite_open_lower", toVec(openComp.lowerBound(probe)));
}

// ==========================================================================
// Block J -- EndCriteria (ctor defaults + the statStateIterations checkers)
// ==========================================================================

void block_j() {
    // ---- ctor: Null<Size>() maxStationaryStateIterations ------------------
    // Null<Size>() == numeric_limits<int>::max(); the ctor replaces it with
    // min(maxIterations/2, 100).
    {
        EndCriteria a(1000, Null<Size>(), 1e-8, 1e-9, 1e-7);
        emit_int("ec_null_stat_1000", static_cast<long>(a.maxStationaryStateIterations()));
        EndCriteria b(30, Null<Size>(), 1e-8, 1e-9, 1e-7);
        emit_int("ec_null_stat_30", static_cast<long>(b.maxStationaryStateIterations()));
    }
    // ---- ctor: Null<Real>() gradientNormEpsilon -> functionEpsilon --------
    {
        EndCriteria a(1000, 100, 1e-8, 1.25e-9, Null<Real>());
        emit("ec_null_gradeps", a.gradientNormEpsilon());
        emit("ec_null_gradeps_source", a.functionEpsilon());
    }
    // ---- the Null sentinels themselves, so Python can pin them ------------
    emit("ec_null_real_sentinel", Real(Null<Real>()));
    emit_int("ec_null_size_sentinel", static_cast<long>(Size(Null<Size>())));

    // ---- checkStationaryPoint: the in-out counter -------------------------
    // rootEpsilon = 0.1, maxStationaryStateIterations = 3.
    {
        const EndCriteria ec(100, 3, 0.1, 0.01, 1e-7);
        // A run of steps: big move, then five tiny moves. The counter resets
        // on the big move and only trips once it EXCEEDS the max.
        const Real xs[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5};
        Size stat = 0;
        std::vector<long> fired, counters, types;
        EndCriteria::Type t = EndCriteria::None;
        for (Real x : xs) {
            const bool hit = ec.checkStationaryPoint(0.0, x, stat, t);
            fired.push_back(hit ? 1 : 0);
            counters.push_back(static_cast<long>(stat));
            types.push_back(static_cast<long>(t));
        }
        emit_arr("ec_csp_x", std::vector<Real>(xs, xs + 7));
        emit_iarr("ec_csp_fired", fired);
        emit_iarr("ec_csp_counter", counters);
        emit_iarr("ec_csp_type", types);
    }
    // ---- checkStationaryFunctionValue -------------------------------------
    {
        const EndCriteria ec(100, 2, 0.1, 0.05, 1e-7);
        const Real fs[6] = {1.0, 0.01, 0.001, 0.0, 0.9, 0.001};
        Size stat = 0;
        std::vector<long> fired, counters, types;
        EndCriteria::Type t = EndCriteria::None;
        for (Real fv : fs) {
            const bool hit = ec.checkStationaryFunctionValue(0.0, fv, stat, t);
            fired.push_back(hit ? 1 : 0);
            counters.push_back(static_cast<long>(stat));
            types.push_back(static_cast<long>(t));
        }
        emit_arr("ec_csfv_f", std::vector<Real>(fs, fs + 6));
        emit_iarr("ec_csfv_fired", fired);
        emit_iarr("ec_csfv_counter", counters);
        emit_iarr("ec_csfv_type", types);
    }
    // ---- the "seed the counter with the max" idiom the optimizers use -----
    // Simplex and LineSearchBasedMethod both pass maxStationaryStateIterations
    // AS the counter, which makes the check fire unconditionally.
    {
        const EndCriteria ec(100, 7, 0.1, 0.05, 1e-7);
        Size stat = ec.maxStationaryStateIterations();
        EndCriteria::Type t = EndCriteria::None;
        const bool hit = ec.checkStationaryPoint(0.0, 0.0, stat, t);
        emit_bool("ec_seeded_csp_fired", hit);
        emit_int("ec_seeded_csp_type", static_cast<long>(t));
        Size stat2 = ec.maxStationaryStateIterations();
        EndCriteria::Type t2 = EndCriteria::None;
        const bool hit2 = ec.checkStationaryFunctionValue(0.0, 0.0, stat2, t2);
        emit_bool("ec_seeded_csfv_fired", hit2);
        emit_int("ec_seeded_csfv_type", static_cast<long>(t2));
    }
    // ---- checkStationaryFunctionAccuracy ---------------------------------
    {
        const EndCriteria ec(100, 3, 0.1, 0.05, 1e-7);
        EndCriteria::Type t1 = EndCriteria::None;
        const bool a = ec.checkStationaryFunctionAccuracy(0.01, false, t1);
        EndCriteria::Type t2 = EndCriteria::None;
        const bool b = ec.checkStationaryFunctionAccuracy(0.01, true, t2);
        EndCriteria::Type t3 = EndCriteria::None;
        const bool c = ec.checkStationaryFunctionAccuracy(0.05, true, t3);
        emit_bool("ec_csfa_negative_optim", a);
        emit_int("ec_csfa_negative_optim_type", static_cast<long>(t1));
        emit_bool("ec_csfa_below", b);
        emit_int("ec_csfa_below_type", static_cast<long>(t2));
        emit_bool("ec_csfa_at_epsilon", c);
        emit_int("ec_csfa_at_epsilon_type", static_cast<long>(t3));
    }
    // ---- operator() -------------------------------------------------------
    {
        const EndCriteria ec(10, 2, 0.1, 0.05, 1e-3);
        Size stat = 0;
        std::vector<long> fired, counters, types;
        // (iteration, fold, fnew, normgnew) tuples chosen so that each branch
        // of the || chain fires in turn.
        const Real folds[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
        const Real fnews[5] = {0.5, 0.99, 0.99, 0.99, 0.99};
        const Real gnews[5] = {1.0, 1.0, 1.0, 1.0, 1e-6};
        const long iters[5] = {0, 1, 2, 3, 4};
        for (int i = 0; i < 5; ++i) {
            EndCriteria::Type t = EndCriteria::None;
            const bool hit = ec(static_cast<Size>(iters[i]), stat, false, folds[i], 0.0,
                                fnews[i], gnews[i], t);
            fired.push_back(hit ? 1 : 0);
            counters.push_back(static_cast<long>(stat));
            types.push_back(static_cast<long>(t));
        }
        emit_iarr("ec_call_fired", fired);
        emit_iarr("ec_call_counter", counters);
        emit_iarr("ec_call_type", types);
        // maxIterations branch
        EndCriteria::Type t = EndCriteria::None;
        Size s2 = 0;
        const bool hit = ec(10, s2, false, 1.0, 0.0, 0.5, 1.0, t);
        emit_bool("ec_call_maxiter_fired", hit);
        emit_int("ec_call_maxiter_type", static_cast<long>(t));
        // positiveOptimization branch (checkStationaryFunctionAccuracy)
        EndCriteria::Type t3 = EndCriteria::None;
        Size s3 = 0;
        const bool hit3 = ec(0, s3, true, 1.0, 0.0, 0.01, 1.0, t3);
        emit_bool("ec_call_accuracy_fired", hit3);
        emit_int("ec_call_accuracy_type", static_cast<long>(t3));
    }
    // ---- succeeded() ------------------------------------------------------
    {
        std::vector<long> flags;
        for (int i = 0; i <= 7; ++i)
            flags.push_back(EndCriteria::succeeded(static_cast<EndCriteria::Type>(i)) ? 1 : 0);
        emit_iarr("ec_succeeded_by_type", flags);
        std::vector<std::string> names;
        for (int i = 0; i <= 7; ++i)
            names.push_back(ecName(static_cast<EndCriteria::Type>(i)));
        emit_sarr("ec_type_names", names);
    }
}

// ==========================================================================
// Block K -- CostFunction defaults + SimpleCostFunction
// ==========================================================================

Array simpleValues(const Array& x) {
    Array r(3);
    r[0] = x[0] - 1.0;
    r[1] = 2.0 * x[1] + x[0];
    r[2] = x[0] * x[1] - 0.5;
    return r;
}

void block_k() {
    // Default value/gradient/jacobian on the residual Rosenbrock.
    {
        RosenbrockResiduals f;
        const Array x{0.3, -0.7};
        Array g(2, 0.0);
        Matrix jac(2, 2, 0.0);
        const Array vs = f.values(x);
        const Real v = f.value(x);
        f.gradient(g, x);
        f.jacobian(jac, x);
        Matrix jac2(2, 2, 0.0);
        const Array vs2 = f.valuesAndJacobian(jac2, x);
        Array g2(2, 0.0);
        const Real v2 = f.valueAndGradient(g2, x);
        emit_arr("cf_x", toVec(x));
        emit_arr("cf_values", toVec(vs));
        emit("cf_value", v);
        emit_arr("cf_gradient", toVec(g));
        emit_arr("cf_jacobian", flatten(jac));
        emit_arr("cf_values_and_jacobian_values", toVec(vs2));
        emit_arr("cf_values_and_jacobian_jac", flatten(jac2));
        emit("cf_value_and_gradient_value", v2);
        emit_arr("cf_value_and_gradient_grad", toVec(g2));
        emit("cf_finite_difference_epsilon", f.finiteDifferenceEpsilon());
    }
    // SimpleCostFunction wrapping a free function: value/gradient/jacobian all
    // come from the CostFunction defaults.
    {
        SimpleCostFunction<Array (*)(const Array&)> scf(&simpleValues);
        const Array x{0.4, 1.1};
        Array g(2, 0.0);
        Matrix jac(3, 2, 0.0);
        scf.gradient(g, x);
        scf.jacobian(jac, x);
        emit_arr("scf_x", toVec(x));
        emit_arr("scf_values", toVec(scf.values(x)));
        emit("scf_value", scf.value(x));
        emit_arr("scf_gradient", toVec(g));
        emit_arr("scf_jacobian", flatten(jac));
        emit_int("scf_jacobian_rows", 3);
        emit_int("scf_jacobian_cols", 2);
    }
}

} // anonymous namespace

int main() {
    std::cout << "{\n";
    block_a();
    block_b();
    block_c();
    block_d();
    block_e();
    block_f();
    block_g();
    block_h();
    block_i();
    block_j();
    block_k();
    std::cout << "\n}\n";
    return 0;
}
