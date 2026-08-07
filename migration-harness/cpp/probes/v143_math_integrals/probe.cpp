// migration-harness/cpp/probes/v143_math_integrals/probe.cpp
//
// Reference values for ql/math/integrals/** in C++ QuantLib v1.43.
//
// What is pinned and why:
//
//   * gaussianorthogonalpolynomial.{hpp,cpp} — the three-term recurrence is
//     the whole content of these classes, so mu_0, alpha(0..7), beta(0..7),
//     the weight w(x) and the recursive value()/weightedValue() are pinned
//     on a domain-appropriate x-grid for every polynomial. Jacobi(0,0) and
//     Jacobi(-0.5,-0.5) additionally drive the two l'Hospital branches in
//     GaussJacobiPolynomial::alpha/beta (denominator close_enough to zero).
//
//   * gaussianquadratures.{hpp,cpp} — the node and weight ARRAYS are pinned
//     element by element, not just the quadrature sum, because the ordering
//     is observable API (TqrEigenDecomposition sorts eigenvalues DEscending
//     and normalises the eigenvector sign on its first component) and a port
//     that delegates the eigenproblem to LAPACK gets ASCending order for
//     free. Integrals of several integrands follow, including |x| and
//     sqrt(|x|), which a polynomial rule cannot integrate well — a rule that
//     was silently replaced by a different rule shows up there first.
//
//   * MultiDimGaussianIntegration — the tensor-product weight vector and the
//     full node table, which is where the spacing/partial_sum index algebra
//     lives, plus a sum.
//
//   * detail::GaussianQuadratureIntegrator<> — the [a,b] affine remap.
//
//   * gausslaguerrecosinepolynomial.hpp — the moment recursion m_(n) and the
//     factorial cache, then the recurrence coefficients that the moment-based
//     Golub-Welsch derives from them, then the nodes/weights/integral.
//
//   * discreteintegrals.{hpp,cpp} — non-uniform grids with both an odd and an
//     even number of points (the even case takes the trailing trapezoid
//     correction in DiscreteSimpsonIntegral), plus both Integrator wrappers.
//
//   * filonintegral.{hpp,cpp} — alpha/beta/gamma are catastrophically
//     cancelling for small theta, so several t are pinned.
//
//   * kronrodintegral.cpp GaussKronrodNonAdaptive — the 10/21/43/87 Patterson
//     ladder. Three integrands chosen so the rule returns after 21, after 43
//     and after 87 points respectively; the absolute error estimate
//     (rescaleError) is pinned too since it decides the ladder.
//
//   * twodimensionalintegral.hpp — nested Integrator composition.
//
//   * trapezoidintegral.hpp — the Default and MidPoint policies, which differ
//     in both the refinement stencil and nbEvalutions().
//
//   * expsinhintegral.hpp / tanhsinhintegral.hpp — thin wrappers over boost's
//     double-exponential quadrature. Pinned at several relative tolerances so
//     a Python re-implementation can be compared against a bound derived from
//     the stopping criterion rather than from luck.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/integrals.json.

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <ql/math/array.hpp>
#include <ql/math/integrals/discreteintegrals.hpp>
#include <ql/math/integrals/expsinhintegral.hpp>
#include <ql/math/integrals/filonintegral.hpp>
#include <ql/math/integrals/gaussianorthogonalpolynomial.hpp>
#include <ql/math/integrals/gaussianquadratures.hpp>
#include <ql/math/integrals/gausslaguerrecosinepolynomial.hpp>
#include <ql/math/integrals/kronrodintegral.hpp>
#include <ql/math/integrals/segmentintegral.hpp>
#include <ql/math/integrals/simpsonintegral.hpp>
#include <ql/math/integrals/tanhsinhintegral.hpp>
#include <ql/math/integrals/trapezoidintegral.hpp>
#include <ql/math/integrals/twodimensionalintegral.hpp>
#include <ql/shared_ptr.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------- JSON ----

void num(Real v) {
    if (std::isnan(v) || std::isinf(v))
        std::cout << "null";
    else
        std::cout << v;
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

void arr(const Array& v) {
    arr(std::vector<Real>(v.begin(), v.end()));
}

void strs(const std::vector<std::string>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << "\"" << v[i] << "\"";
    }
    std::cout << "]";
}

// ---------------------------------------------------- shared integrands ----

typedef std::function<Real(Real)> Fn;

struct NamedFn {
    std::string name;
    Fn f;
};

// Named once, referenced by key from the JSON, so the Python test walks the
// reference rather than restating the case list.
const std::vector<NamedFn>& integrands() {
    static const std::vector<NamedFn> v = {
        {"one", [](Real) { return 1.0; }},
        {"x", [](Real x) { return x; }},
        {"x2", [](Real x) { return x * x; }},
        {"x4", [](Real x) { return x * x * x * x; }},
        {"inv_1px2", [](Real x) { return 1.0 / (1.0 + x * x); }},
        {"sin_x", [](Real x) { return std::sin(x); }},
        {"abs_x", [](Real x) { return std::fabs(x); }},
        {"sqrt_abs_x", [](Real x) { return std::sqrt(std::fabs(x)); }},
    };
    return v;
}

Fn integrandByName(const std::string& name) {
    for (const auto& e : integrands())
        if (e.name == name)
            return e.f;
    QL_FAIL("unknown integrand " << name);
}

// ------------------------------------------------------- polynomials -------

const std::vector<Size>& polyOrders() {
    static const std::vector<Size> v = {0, 1, 2, 3, 4, 5, 6, 7};
    return v;
}

// GaussJacobiPolynomial::alpha/beta raise for the degenerate index where both
// the numerator and the l'Hospital-corrected denominator vanish (alpha(i) and
// beta(i) for i == 0 when alpha_ + beta_ == 0, i.e. Legendre and
// Gegenbauer(0.5)). GaussianQuadrature never asks for beta(0), so the throw is
// unreachable there — but it is observable API, so it is pinned as a null
// entry and the Python port must raise in exactly the same places.
void emitMaybeThrowing(const std::function<Real(Size)>& g) {
    std::cout << "[";
    for (Size k = 0; k < polyOrders().size(); ++k) {
        if (k != 0)
            std::cout << ", ";
        try {
            const Real v = g(polyOrders()[k]);
            num(v);
        } catch (const std::exception&) {
            std::cout << "null";
        }
    }
    std::cout << "]";
}

void emitPolynomial(const std::string& name,
                    const std::string& kind,
                    const GaussianOrthogonalPolynomial& p,
                    const std::vector<Real>& xs,
                    bool comma) {
    std::cout << "    {\"name\": \"" << name << "\", \"kind\": \"" << kind << "\",\n";
    std::cout << "     \"mu_0\": ";
    num(p.mu_0());
    std::cout << ",\n     \"indices\": ";
    {
        std::vector<Real> idx;
        for (Size i : polyOrders())
            idx.push_back(Real(i));
        arr(idx);
    }
    std::cout << ",\n     \"alpha\": ";
    emitMaybeThrowing([&p](Size i) { return p.alpha(i); });
    std::cout << ",\n     \"beta\": ";
    emitMaybeThrowing([&p](Size i) { return p.beta(i); });
    std::cout << ",\n     \"xs\": ";
    arr(xs);
    std::cout << ",\n     \"w\": ";
    {
        std::vector<Real> w;
        for (Real x : xs)
            w.push_back(p.w(x));
        arr(w);
    }
    std::cout << ",\n     \"value\": [";
    for (Size k = 0; k < polyOrders().size(); ++k) {
        if (k != 0)
            std::cout << ",\n                ";
        std::vector<Real> row;
        for (Real x : xs)
            row.push_back(p.value(polyOrders()[k], x));
        arr(row);
    }
    std::cout << "],\n     \"weighted_value\": [";
    for (Size k = 0; k < polyOrders().size(); ++k) {
        if (k != 0)
            std::cout << ",\n                         ";
        std::vector<Real> row;
        for (Real x : xs)
            row.push_back(p.weightedValue(polyOrders()[k], x));
        arr(row);
    }
    std::cout << "]}" << (comma ? "," : "") << "\n";
}

// ------------------------------------------------------- quadratures -------

void emitQuadrature(const std::string& name,
                    const std::string& kind,
                    GaussianQuadrature& q,
                    const std::vector<std::string>& fnames,
                    bool comma) {
    std::cout << "    {\"name\": \"" << name << "\", \"kind\": \"" << kind << "\",\n";
    std::cout << "     \"order\": " << q.order() << ",\n";
    std::cout << "     \"x\": ";
    arr(q.x());
    std::cout << ",\n     \"weights\": ";
    arr(q.weights());
    std::cout << ",\n     \"integrand_names\": ";
    strs(fnames);
    std::cout << ",\n     \"integrals\": ";
    {
        std::vector<Real> v;
        for (const auto& fn : fnames)
            v.push_back(q(integrandByName(fn)));
        arr(v);
    }
    // L1 of the quadrature sum, sum_i |w_i f(x_i)|. Several of these sums
    // cancel from O(1) terms down to ~1e-14 (any odd integrand on a symmetric
    // rule), where a *relative* comparison against the C++ result is
    // meaningless; the meaningful bound is relative to this L1.
    std::cout << ",\n     \"integrals_l1\": ";
    {
        std::vector<Real> v;
        for (const auto& fn : fnames) {
            const Fn f = integrandByName(fn);
            Real s = 0.0;
            for (Size i = 0; i < q.order(); ++i)
                s += std::fabs(q.weights()[i] * f(q.x()[i]));
            v.push_back(s);
        }
        arr(v);
    }
    std::cout << "}" << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------- 2-D helper -----

Real f2dProduct(Real x, Real y) {
    return std::exp(-x * x) * std::sin(y) + x * y;
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ================================================== 1. polynomials ====
    {
        const std::vector<Real> xsUnit = {-0.9, -0.5, -0.125, 0.0, 0.25, 0.5, 0.875};
        const std::vector<Real> xsPos = {0.0, 0.125, 0.5, 1.0, 2.5, 7.0, 13.25};
        const std::vector<Real> xsFull = {-3.5, -1.25, -0.5, 0.0, 0.5, 1.25, 3.5};

        std::cout << "  \"orthogonal_polynomials\": [\n";
        emitPolynomial("laguerre_s0", "GaussLaguerrePolynomial",
                       GaussLaguerrePolynomial(0.0), xsPos, true);
        emitPolynomial("laguerre_s1_5", "GaussLaguerrePolynomial",
                       GaussLaguerrePolynomial(1.5), xsPos, true);
        emitPolynomial("hermite_mu0", "GaussHermitePolynomial",
                       GaussHermitePolynomial(0.0), xsFull, true);
        emitPolynomial("hermite_mu0_3", "GaussHermitePolynomial",
                       GaussHermitePolynomial(0.3), xsFull, true);
        emitPolynomial("jacobi_0_5_1_5", "GaussJacobiPolynomial",
                       GaussJacobiPolynomial(0.5, 1.5), xsUnit, true);
        emitPolynomial("jacobi_m0_25_0_75", "GaussJacobiPolynomial",
                       GaussJacobiPolynomial(-0.25, 0.75), xsUnit, true);
        emitPolynomial("legendre", "GaussLegendrePolynomial",
                       GaussLegendrePolynomial(), xsUnit, true);
        emitPolynomial("chebyshev", "GaussChebyshevPolynomial",
                       GaussChebyshevPolynomial(), xsUnit, true);
        emitPolynomial("chebyshev2nd", "GaussChebyshev2ndPolynomial",
                       GaussChebyshev2ndPolynomial(), xsUnit, true);
        emitPolynomial("gegenbauer_1_5", "GaussGegenbauerPolynomial",
                       GaussGegenbauerPolynomial(1.5), xsUnit, true);
        emitPolynomial("gegenbauer_0_5", "GaussGegenbauerPolynomial",
                       GaussGegenbauerPolynomial(0.5), xsUnit, true);
        emitPolynomial("hyperbolic", "GaussHyperbolicPolynomial",
                       GaussHyperbolicPolynomial(), xsFull, false);
        std::cout << "  ],\n";
    }

    // ================================================== 2. quadratures ====
    {
        const std::vector<std::string> all = {"one",      "x",      "x2",
                                              "x4",       "inv_1px2", "sin_x",
                                              "abs_x",    "sqrt_abs_x"};
        std::cout << "  \"quadratures\": [\n";
        {
            GaussLaguerreIntegration q(8);
            emitQuadrature("laguerre_8", "GaussLaguerreIntegration", q, all, true);
        }
        {
            GaussLaguerreIntegration q(16, 1.5);
            emitQuadrature("laguerre_16_s1_5", "GaussLaguerreIntegration", q, all, true);
        }
        {
            // High order on purpose. GaussianQuadrature's weight is
            // mu_0 * v0_i^2 / w(x_i), and for Laguerre w(x) = x^s e^{-x}, so
            // the division multiplies by e^{x_i}. At order 64 the largest node
            // is ~2.3e2 and at order 144 it is ~5.5e2, which means the first
            // eigenvector component there is ~e^{-x/2} — 1e-50 and 1e-119
            // below the unit norm respectively. A norm-wise backward-stable
            // eigensolver (LAPACK) returns noise or an exact zero in that
            // position; squaring it and multiplying by e^{x} yields weights
            // that are 0 or ~1e126 where C++ has O(10). Only an
            // implementation that keeps *relative* accuracy in tiny
            // components — C++'s TqrEigenDecomposition, which multiplies
            // Givens rotations into the first row and never forms a
            // cancelling difference — reproduces these.
            //
            // Orders 8 and 16 above cannot detect that: their largest nodes
            // are ~22 and ~51, so e^{-x/2} is still 1e-5 / 1e-12 and survives
            // in double. 64 and 144 are the orders the Heston engines
            // actually run (Integration::gaussLaguerre defaults to 128, the
            // engine to 144, HestonBlackVolSurface to 160).
            GaussLaguerreIntegration q(64);
            emitQuadrature("laguerre_64", "GaussLaguerreIntegration", q, all, true);
        }
        {
            GaussHermiteIntegration q(8);
            emitQuadrature("hermite_8", "GaussHermiteIntegration", q, all, true);
        }
        {
            // Even order deliberately. With mu != 0 and an *odd* order the
            // central node is a computed zero (~1e-17) and its weight is
            // mu_0 * v^2 / (|x|^{2 mu} e^{-x^2}) — dividing by |x|^0.6 turns a
            // 1-ulp node difference into an O(1) weight difference, so that
            // configuration is not reproducible across eigen-solvers at all
            // (it is not reproducible across compilers in C++ either).
            GaussHermiteIntegration q(6, 0.3);
            emitQuadrature("hermite_6_mu0_3", "GaussHermiteIntegration", q, all, true);
        }
        {
            GaussHermiteIntegration q(5);
            emitQuadrature("hermite_5", "GaussHermiteIntegration", q, all, true);
        }
        {
            GaussJacobiIntegration q(8, 0.5, 1.5);
            emitQuadrature("jacobi_8", "GaussJacobiIntegration", q, all, true);
        }
        {
            GaussLegendreIntegration q(8);
            emitQuadrature("legendre_8", "GaussLegendreIntegration", q, all, true);
        }
        {
            GaussLegendreIntegration q(16);
            emitQuadrature("legendre_16", "GaussLegendreIntegration", q, all, true);
        }
        {
            GaussChebyshevIntegration q(8);
            emitQuadrature("chebyshev_8", "GaussChebyshevIntegration", q, all, true);
        }
        {
            GaussChebyshev2ndIntegration q(8);
            emitQuadrature("chebyshev2nd_8", "GaussChebyshev2ndIntegration", q, all, true);
        }
        {
            GaussGegenbauerIntegration q(8, 1.5);
            emitQuadrature("gegenbauer_8", "GaussGegenbauerIntegration", q, all, true);
        }
        {
            GaussHyperbolicIntegration q(8);
            emitQuadrature("hyperbolic_8", "GaussHyperbolicIntegration", q, all, true);
        }
        {
            GaussLegendreIntegration q(3);
            emitQuadrature("legendre_3", "GaussLegendreIntegration", q, all, false);
        }
        std::cout << "  ],\n";
    }

    // ======================================= 3. multi-dim quadrature ======
    {
        std::cout << "  \"multi_dim\": [\n";
        struct Case {
            std::string name;
            std::vector<Size> ns;
        };
        const std::vector<Case> cases = {{"legendre_3x4", {3, 4}},
                                         {"legendre_2x3x2", {2, 3, 2}},
                                         {"legendre_5", {5}}};
        for (Size c = 0; c < cases.size(); ++c) {
            const MultiDimGaussianIntegration mdi(
                cases[c].ns, [](Size n) {
                    return ext::make_shared<GaussLegendreIntegration>(n);
                });
            std::cout << "    {\"name\": \"" << cases[c].name << "\", \"ns\": [";
            for (Size j = 0; j < cases[c].ns.size(); ++j)
                std::cout << (j ? ", " : "") << cases[c].ns[j];
            std::cout << "],\n     \"weights\": ";
            arr(mdi.weights());
            std::cout << ",\n     \"x\": [";
            for (Size i = 0; i < mdi.x().size(); ++i) {
                if (i != 0)
                    std::cout << ",\n           ";
                arr(mdi.x()[i]);
            }
            std::cout << "],\n     \"integral_sum_sq\": ";
            num(mdi([](const Array& v) {
                Real s = 0.0;
                for (Real e : v)
                    s += e * e;
                return s;
            }));
            std::cout << ",\n     \"integral_exp_prod\": ";
            num(mdi([](const Array& v) {
                Real s = 0.0;
                for (Real e : v)
                    s += e;
                return std::exp(s);
            }));
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ================================ 4. GaussianQuadratureIntegrator =====
    {
        std::cout << "  \"quadrature_integrators\": [\n";
        struct Case {
            std::string name;
            std::string kind;
            Size n;
            std::string fn;
            Real a;
            Real b;
        };
        const std::vector<Case> cases = {
            {"legendre_16_x2_0_1", "GaussLegendreIntegrator", 16, "x2", 0.0, 1.0},
            {"legendre_16_sin_0_pi", "GaussLegendreIntegrator", 16, "sin_x", 0.0, M_PI},
            {"legendre_16_sqrt_0_1", "GaussLegendreIntegrator", 16, "sqrt_abs_x", 0.0, 1.0},
            {"legendre_4_x4_m2_3", "GaussLegendreIntegrator", 4, "x4", -2.0, 3.0},
            {"chebyshev_16_x2_0_1", "GaussChebyshevIntegrator", 16, "x2", 0.0, 1.0},
            {"chebyshev_16_sin_0_pi", "GaussChebyshevIntegrator", 16, "sin_x", 0.0, M_PI},
            {"chebyshev2nd_16_x2_0_1", "GaussChebyshev2ndIntegrator", 16, "x2", 0.0, 1.0},
            {"chebyshev2nd_16_inv_m1_1", "GaussChebyshev2ndIntegrator", 16, "inv_1px2", -1.0,
             1.0},
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            ext::shared_ptr<Integrator> integrator;
            if (k.kind == "GaussLegendreIntegrator")
                integrator = ext::make_shared<GaussLegendreIntegrator>(k.n);
            else if (k.kind == "GaussChebyshevIntegrator")
                integrator = ext::make_shared<GaussChebyshevIntegrator>(k.n);
            else
                integrator = ext::make_shared<GaussChebyshev2ndIntegrator>(k.n);
            const Real v = (*integrator)(integrandByName(k.fn), k.a, k.b);
            std::cout << "    {\"name\": \"" << k.name << "\", \"kind\": \"" << k.kind
                      << "\", \"n\": " << k.n << ", \"integrand\": \"" << k.fn
                      << "\", \"a\": ";
            num(k.a);
            std::cout << ", \"b\": ";
            num(k.b);
            std::cout << ", \"value\": ";
            num(v);
            std::cout << ", \"max_evaluations\": " << integrator->maxEvaluations()
                      << ", \"absolute_accuracy\": ";
            num(integrator->absoluteAccuracy());
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ==================================== 5. Laguerre trigonometric =======
    {
        std::cout << "  \"laguerre_trigonometric\": [\n";
        struct Case {
            std::string name;
            std::string kind;
            Real u;
            Size n;
        };
        // Two orders per (kind, u). The n = 5 rules only ever touch alpha(0..4)
        // and beta(1..4), where the Hankel-determinant recursion is still well
        // conditioned in double precision; the n = 8 rules reach into the
        // indices where it is not — which is exactly why the C++ class is a
        // template over mp_real. Both are pinned so the difference is
        // measurable rather than assumed.
        const std::vector<Case> cases = {
            {"cosine_u0_5_n5", "GaussLaguerreCosinePolynomial", 0.5, 5},
            {"cosine_u0_5_n8", "GaussLaguerreCosinePolynomial", 0.5, 8},
            {"cosine_u2_0_n5", "GaussLaguerreCosinePolynomial", 2.0, 5},
            {"cosine_u2_0_n8", "GaussLaguerreCosinePolynomial", 2.0, 8},
            {"cosine_u0_0_n6", "GaussLaguerreCosinePolynomial", 0.0, 6},
            {"sine_u0_5_n5", "GaussLaguerreSinePolynomial", 0.5, 5},
            {"sine_u0_5_n8", "GaussLaguerreSinePolynomial", 0.5, 8},
            {"sine_u2_0_n5", "GaussLaguerreSinePolynomial", 2.0, 5},
            {"sine_u2_0_n8", "GaussLaguerreSinePolynomial", 2.0, 8},
        };
        const std::vector<Real> xs = {0.0, 0.25, 1.0, 2.5, 6.0, 11.5};
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            ext::shared_ptr<GaussianOrthogonalPolynomial> poly;
            std::vector<Real> moments;
            if (k.kind == "GaussLaguerreCosinePolynomial") {
                const auto p = ext::make_shared<GaussLaguerreCosinePolynomial<Real> >(k.u);
                for (Size i = 0; i < 10; ++i)
                    moments.push_back(p->moment(i));
                poly = p;
            } else {
                const auto p = ext::make_shared<GaussLaguerreSinePolynomial<Real> >(k.u);
                for (Size i = 0; i < 10; ++i)
                    moments.push_back(p->moment(i));
                poly = p;
            }
            std::cout << "    {\"name\": \"" << k.name << "\", \"kind\": \"" << k.kind
                      << "\", \"u\": ";
            num(k.u);
            std::cout << ", \"n\": " << k.n << ",\n     \"moments\": ";
            arr(moments);
            std::cout << ",\n     \"mu_0\": ";
            num(poly->mu_0());
            std::cout << ",\n     \"alpha\": ";
            {
                std::vector<Real> a;
                for (Size i = 0; i < k.n; ++i)
                    a.push_back(poly->alpha(i));
                arr(a);
            }
            std::cout << ",\n     \"beta\": ";
            {
                std::vector<Real> b;
                for (Size i = 0; i < k.n; ++i)
                    b.push_back(poly->beta(i));
                arr(b);
            }
            std::cout << ",\n     \"xs\": ";
            arr(xs);
            std::cout << ",\n     \"w\": ";
            {
                std::vector<Real> w;
                for (Real x : xs)
                    w.push_back(poly->w(x));
                arr(w);
            }
            GaussianQuadrature q(k.n, *poly);
            std::cout << ",\n     \"nodes\": ";
            arr(q.x());
            std::cout << ",\n     \"quad_weights\": ";
            arr(q.weights());
            std::cout << ",\n     \"integral_one\": ";
            num(q([](Real) { return 1.0; }));
            std::cout << ",\n     \"integral_inv_1px2\": ";
            num(q([](Real x) { return 1.0 / (1.0 + x * x); }));
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ====================================== 6. discrete integrals =========
    {
        std::cout << "  \"discrete_integrals\": [\n";
        struct Case {
            std::string name;
            std::vector<Real> x;
            std::vector<Real> f;
        };
        const std::vector<Case> cases = {
            // odd count, non-uniform
            {"odd_nonuniform",
             {0.0, 0.3, 1.1, 1.4, 2.9, 3.0, 5.5},
             {1.0, 1.09, 3.21, 2.96, 8.41, 9.0, 30.25}},
            // even count, non-uniform -> DiscreteSimpsonIntegral takes the
            // trailing trapezoid correction
            {"even_nonuniform",
             {0.0, 0.3, 1.1, 1.4, 2.9, 3.0},
             {1.0, 1.09, 3.21, 2.96, 8.41, 9.0}},
            // uniform, odd
            {"uniform_odd", {0.0, 0.5, 1.0, 1.5, 2.0}, {0.0, 0.25, 1.0, 2.25, 4.0}},
            // uniform, even
            {"uniform_even", {0.0, 0.5, 1.0, 1.5}, {0.0, 0.25, 1.0, 2.25}},
            // degenerate
            {"single_point", {1.0}, {2.0}},
            {"two_points", {1.0, 3.0}, {2.0, 4.0}},
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Array x(cases[c].x.begin(), cases[c].x.end());
            const Array f(cases[c].f.begin(), cases[c].f.end());
            std::cout << "    {\"name\": \"" << cases[c].name << "\", \"x\": ";
            arr(cases[c].x);
            std::cout << ", \"f\": ";
            arr(cases[c].f);
            std::cout << ",\n     \"trapezoid\": ";
            num(DiscreteTrapezoidIntegral()(x, f));
            std::cout << ", \"simpson\": ";
            num(DiscreteSimpsonIntegral()(x, f));
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ==================================== 7. discrete integrators =========
    {
        std::cout << "  \"discrete_integrators\": [\n";
        struct Case {
            std::string name;
            Size evaluations;
            std::string fn;
            Real a;
            Real b;
        };
        const std::vector<Case> cases = {
            {"n11_x2_0_1", 11, "x2", 0.0, 1.0},
            {"n10_x2_0_1", 10, "x2", 0.0, 1.0},
            {"n21_sin_0_pi", 21, "sin_x", 0.0, M_PI},
            {"n20_sin_0_pi", 20, "sin_x", 0.0, M_PI},
            {"n51_sqrt_0_1", 51, "sqrt_abs_x", 0.0, 1.0},
            {"n11_inv_m1_1", 11, "inv_1px2", -1.0, 1.0},
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            const DiscreteTrapezoidIntegrator ti(k.evaluations);
            const DiscreteSimpsonIntegrator si(k.evaluations);
            const Real tv = ti(integrandByName(k.fn), k.a, k.b);
            const Size tn = ti.numberOfEvaluations();
            const Real sv = si(integrandByName(k.fn), k.a, k.b);
            const Size sn = si.numberOfEvaluations();
            std::cout << "    {\"name\": \"" << k.name
                      << "\", \"evaluations\": " << k.evaluations << ", \"integrand\": \""
                      << k.fn << "\", \"a\": ";
            num(k.a);
            std::cout << ", \"b\": ";
            num(k.b);
            std::cout << ",\n     \"trapezoid\": ";
            num(tv);
            std::cout << ", \"trapezoid_evaluations\": " << tn << ", \"simpson\": ";
            num(sv);
            std::cout << ", \"simpson_evaluations\": " << sn;
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ================================================ 8. Filon ============
    {
        std::cout << "  \"filon\": [\n";
        struct Case {
            std::string name;
            std::string type;
            Real t;
            Size intervals;
            std::string fn;
            Real a;
            Real b;
        };
        const std::vector<Case> cases = {
            {"cos_t1_n16_x2", "Cosine", 1.0, 16, "x2", 0.0, 1.0},
            {"sin_t1_n16_x2", "Sine", 1.0, 16, "x2", 0.0, 1.0},
            {"cos_t10_n50_exp", "Cosine", 10.0, 50, "inv_1px2", 0.0, 2.0},
            {"sin_t10_n50_exp", "Sine", 10.0, 50, "inv_1px2", 0.0, 2.0},
            {"cos_t0_1_n20_x4", "Cosine", 0.1, 20, "x4", -1.0, 1.0},
            {"sin_t0_1_n20_x4", "Sine", 0.1, 20, "x4", -1.0, 1.0},
            {"cos_t100_n200", "Cosine", 100.0, 200, "one", 0.0, 1.0},
            {"sin_t100_n200", "Sine", 100.0, 200, "one", 0.0, 1.0},
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            const FilonIntegral fi(k.type == "Cosine" ? FilonIntegral::Cosine
                                                      : FilonIntegral::Sine,
                                   k.t, k.intervals);
            const Real v = fi(integrandByName(k.fn), k.a, k.b);
            std::cout << "    {\"name\": \"" << k.name << "\", \"type\": \"" << k.type
                      << "\", \"t\": ";
            num(k.t);
            std::cout << ", \"intervals\": " << k.intervals << ", \"integrand\": \"" << k.fn
                      << "\", \"a\": ";
            num(k.a);
            std::cout << ", \"b\": ";
            num(k.b);
            std::cout << ", \"value\": ";
            num(v);
            std::cout << ", \"max_evaluations\": " << fi.maxEvaluations();
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ============================= 9. GaussKronrodNonAdaptive =============
    {
        std::cout << "  \"kronrod_non_adaptive\": [\n";
        struct Case {
            std::string name;
            std::string fn; // local, not from integrands()
            Real a;
            Real b;
            Real absAcc;
            Real relAcc;
        };
        const std::vector<Case> cases = {
            {"x2_0_1", "x2", 0.0, 1.0, 1e-10, 1e-10},
            {"sin_0_pi", "sin_x", 0.0, M_PI, 1e-10, 1e-10},
            {"osc50_0_1", "sin_50x", 0.0, 1.0, 1e-10, 1e-10},
            {"osc200_0_1", "sin_200x", 0.0, 1.0, 1e-10, 1e-10},
            {"inv_sqrt_0_1", "inv_sqrt_x", 0.0, 1.0, 1e-10, 1e-10},
            {"runge_m1_1", "runge", -1.0, 1.0, 1e-10, 1e-10},
            {"x2_loose_0_1", "x2", 0.0, 1.0, 1e-2, 1e-2},
            {"osc50_reversed", "sin_50x", 1.0, 0.0, 1e-10, 1e-10},
        };
        auto local = [](const std::string& n) -> Fn {
            if (n == "x2")
                return [](Real x) { return x * x; };
            if (n == "sin_x")
                return [](Real x) { return std::sin(x); };
            if (n == "sin_50x")
                return [](Real x) { return std::sin(50.0 * x); };
            if (n == "sin_200x")
                return [](Real x) { return std::sin(200.0 * x); };
            if (n == "inv_sqrt_x")
                return [](Real x) { return 1.0 / std::sqrt(x); };
            if (n == "runge")
                return [](Real x) { return 1.0 / (1.0 + 25.0 * x * x); };
            QL_FAIL("unknown local integrand " << n);
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            const GaussKronrodNonAdaptive gk(k.absAcc, 100, k.relAcc);
            const Real v = gk(local(k.fn), k.a, k.b);
            std::cout << "    {\"name\": \"" << k.name << "\", \"integrand\": \"" << k.fn
                      << "\", \"a\": ";
            num(k.a);
            std::cout << ", \"b\": ";
            num(k.b);
            std::cout << ", \"abs_accuracy\": ";
            num(k.absAcc);
            std::cout << ", \"rel_accuracy\": ";
            num(k.relAcc);
            std::cout << ",\n     \"value\": ";
            num(v);
            std::cout << ", \"absolute_error\": ";
            num(gk.absoluteError());
            std::cout << ", \"evaluations\": " << gk.numberOfEvaluations()
                      << ", \"success\": " << (gk.integrationSuccess() ? "true" : "false");
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ================================== 10. two-dimensional ===============
    {
        std::cout << "  \"two_dimensional\": [\n";
        struct Case {
            std::string name;
            std::string integratorX;
            std::string integratorY;
            Real ax, ay, bx, by;
        };
        // At least one deterministic fixed-node rule per pair: nesting two
        // *adaptive* rules at the same tolerance never terminates, because the
        // inner rule's own error floors the outer rule's refinement increment.
        const std::vector<Case> cases = {
            {"segment_unit_box", "segment_50", "segment_50", 0.0, 0.0, 1.0, 1.0},
            {"legendre_wide_box", "legendre_16", "legendre_16", -1.5, 0.5, 2.0, 3.0},
            {"mixed_box", "simpson_1e-6", "legendre_16", -1.0, 0.0, 1.0, 2.0},
            {"degenerate_x", "segment_50", "segment_50", 1.0, 0.0, 1.0, 1.0},
            {"reversed_x", "legendre_16", "legendre_16", 2.0, 0.5, -1.5, 3.0},
        };
        auto makeIntegrator = [](const std::string& n) -> ext::shared_ptr<Integrator> {
            if (n == "segment_50")
                return ext::make_shared<SegmentIntegral>(50);
            if (n == "legendre_16")
                return ext::make_shared<GaussLegendreIntegrator>(16);
            if (n == "simpson_1e-6")
                return ext::make_shared<SimpsonIntegral>(1e-6, 40);
            QL_FAIL("unknown integrator " << n);
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            const TwoDimensionalIntegral tdi(makeIntegrator(k.integratorX),
                                             makeIntegrator(k.integratorY));
            const Real v =
                tdi(f2dProduct, std::make_pair(k.ax, k.ay), std::make_pair(k.bx, k.by));
            std::cout << "    {\"name\": \"" << k.name << "\", \"integrator_x\": \""
                      << k.integratorX << "\", \"integrator_y\": \"" << k.integratorY
                      << "\", \"a\": [";
            num(k.ax);
            std::cout << ", ";
            num(k.ay);
            std::cout << "], \"b\": [";
            num(k.bx);
            std::cout << ", ";
            num(k.by);
            std::cout << "], \"value\": ";
            num(v);
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ================================ 11. trapezoid policies ==============
    {
        std::cout << "  \"trapezoid_policies\": [\n";
        struct Case {
            std::string name;
            std::string policy;
            Real accuracy;
            Size maxIter;
            std::string fn;
            Real a;
            Real b;
        };
        // MidPoint accuracies are deliberately looser than the Default ones.
        // The loop seeds I with the *trapezoid* T_1 = (f(a)+f(b))(b-a)/2 and
        // then applies the midpoint trisection recursion I <- (I + dx*S)/3, so
        // the seed error decays only by a factor 3 per iteration while N grows
        // by 3 — asking for 1e-10 costs O(3^20) evaluations.
        const std::vector<Case> cases = {
            {"default_x2_0_1", "Default", 1e-10, 100, "x2", 0.0, 1.0},
            {"midpoint_x2_0_1", "MidPoint", 1e-5, 100, "x2", 0.0, 1.0},
            {"default_sin_0_pi", "Default", 1e-10, 100, "sin_x", 0.0, M_PI},
            {"midpoint_sin_0_pi", "MidPoint", 1e-3, 100, "sin_x", 0.0, M_PI},
            {"default_sqrt_0_1", "Default", 1e-6, 100, "sqrt_abs_x", 0.0, 1.0},
            {"midpoint_sqrt_0_1", "MidPoint", 1e-5, 100, "sqrt_abs_x", 0.0, 1.0},
            {"midpoint_inv_m1_1", "MidPoint", 1e-4, 100, "inv_1px2", -1.0, 1.0},
            {"default_x4_m2_3", "Default", 1e-7, 100, "x4", -2.0, 3.0},
            {"midpoint_x4_m2_3", "MidPoint", 1e-2, 100, "x4", -2.0, 3.0},
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            Real v;
            Size n;
            if (k.policy == "Default") {
                const TrapezoidIntegral<Default> ti(k.accuracy, k.maxIter);
                v = ti(integrandByName(k.fn), k.a, k.b);
                n = ti.numberOfEvaluations();
            } else {
                const TrapezoidIntegral<MidPoint> ti(k.accuracy, k.maxIter);
                v = ti(integrandByName(k.fn), k.a, k.b);
                n = ti.numberOfEvaluations();
            }
            std::cout << "    {\"name\": \"" << k.name << "\", \"policy\": \"" << k.policy
                      << "\", \"accuracy\": ";
            num(k.accuracy);
            std::cout << ", \"max_iterations\": " << k.maxIter << ", \"integrand\": \""
                      << k.fn << "\", \"a\": ";
            num(k.a);
            std::cout << ", \"b\": ";
            num(k.b);
            std::cout << ", \"value\": ";
            num(v);
            std::cout << ", \"evaluations\": " << n;
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // =============================== 12. tanh-sinh (boost) ================
    {
        std::cout << "  \"tanh_sinh\": [\n";
        struct Case {
            std::string name;
            std::string fn;
            Real a;
            Real b;
            Real relTolerance;
        };
        const Real defTol = std::sqrt(std::numeric_limits<Real>::epsilon());
        const std::vector<Case> cases = {
            {"x2_0_1_default", "x2", 0.0, 1.0, defTol},
            {"x2_0_1_tight", "x2", 0.0, 1.0, 1e-13},
            {"sin_0_pi_tight", "sin_x", 0.0, M_PI, 1e-13},
            {"inv_sqrt_0_1_tight", "inv_sqrt_x", 0.0, 1.0, 1e-13},
            {"log_0_1_tight", "log_x", 0.0, 1.0, 1e-13},
            {"sqrt_1mx2_m1_1_tight", "sqrt_1mx2", -1.0, 1.0, 1e-13},
            {"runge_m1_1_tight", "runge", -1.0, 1.0, 1e-13},
            {"gauss_m5_5_tight", "gauss", -5.0, 5.0, 1e-13},
            {"osc50_0_1_tight", "sin_50x", 0.0, 1.0, 1e-13},
            {"x2_1_0_reversed", "x2", 1.0, 0.0, 1e-13},
        };
        auto local = [](const std::string& n) -> Fn {
            if (n == "x2")
                return [](Real x) { return x * x; };
            if (n == "sin_x")
                return [](Real x) { return std::sin(x); };
            if (n == "sin_50x")
                return [](Real x) { return std::sin(50.0 * x); };
            if (n == "inv_sqrt_x")
                return [](Real x) { return 1.0 / std::sqrt(x); };
            if (n == "log_x")
                return [](Real x) { return std::log(x); };
            if (n == "sqrt_1mx2")
                return [](Real x) { return std::sqrt(1.0 - x * x); };
            if (n == "runge")
                return [](Real x) { return 1.0 / (1.0 + 25.0 * x * x); };
            if (n == "gauss")
                return [](Real x) { return std::exp(-x * x); };
            QL_FAIL("unknown local integrand " << n);
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            const TanhSinhIntegral ts(k.relTolerance);
            const Real v = ts(local(k.fn), k.a, k.b);
            std::cout << "    {\"name\": \"" << k.name << "\", \"integrand\": \"" << k.fn
                      << "\", \"a\": ";
            num(k.a);
            std::cout << ", \"b\": ";
            num(k.b);
            std::cout << ", \"rel_tolerance\": ";
            num(k.relTolerance);
            std::cout << ",\n     \"value\": ";
            num(v);
            std::cout << ", \"absolute_error\": ";
            num(ts.absoluteError());
            std::cout << ", \"evaluations\": " << ts.numberOfEvaluations();
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ================================ 13. exp-sinh (boost) ================
    {
        std::cout << "  \"exp_sinh\": [\n";
        struct Case {
            std::string name;
            std::string fn;
            Real a; // finite endpoint; the other one is +/- infinity
            bool lowerIsFinite; // true -> [a, +inf); false -> (-inf, a]
            Real relTolerance;
            bool useHalfInfiniteOverload; // integrate(f) == [0, inf)
        };
        const Real defTol = std::sqrt(std::numeric_limits<Real>::epsilon());
        const std::vector<Case> cases = {
            {"exp_neg_x_default", "exp_neg_x", 0.0, true, defTol, true},
            {"exp_neg_x_tight", "exp_neg_x", 0.0, true, 1e-13, true},
            {"inv_1px2_tight", "inv_1px2", 0.0, true, 1e-13, true},
            {"x_exp_neg_x2_tight", "x_exp_neg_x2", 0.0, true, 1e-13, true},
            {"inv_sqrt_exp_tight", "inv_sqrt_exp", 0.0, true, 1e-13, true},
            {"exp_neg_x_from_2", "exp_neg_x", 2.0, true, 1e-13, false},
            {"inv_1px2_from_1", "inv_1px2", 1.0, true, 1e-13, false},
            {"exp_neg_x_from_0_ab", "exp_neg_x", 0.0, true, 1e-13, false},
            {"exp_x_to_0", "exp_x", 0.0, false, 1e-13, false},
            {"inv_1px2_to_m1", "inv_1px2", -1.0, false, 1e-13, false},
        };
        auto local = [](const std::string& n) -> Fn {
            if (n == "exp_neg_x")
                return [](Real x) { return std::exp(-x); };
            if (n == "exp_x")
                return [](Real x) { return std::exp(x); };
            if (n == "inv_1px2")
                return [](Real x) { return 1.0 / (1.0 + x * x); };
            if (n == "x_exp_neg_x2")
                return [](Real x) { return x * std::exp(-x * x); };
            if (n == "inv_sqrt_exp")
                return [](Real x) { return std::exp(-x) / std::sqrt(x); };
            QL_FAIL("unknown local integrand " << n);
        };
        for (Size c = 0; c < cases.size(); ++c) {
            const Case& k = cases[c];
            const ExpSinhIntegral es(k.relTolerance);
            const Real inf = std::numeric_limits<Real>::infinity();
            Real v;
            if (k.useHalfInfiniteOverload)
                v = es.integrate(local(k.fn));
            else if (k.lowerIsFinite)
                v = es(local(k.fn), k.a, inf);
            else
                v = es(local(k.fn), -inf, k.a);
            std::cout << "    {\"name\": \"" << k.name << "\", \"integrand\": \"" << k.fn
                      << "\", \"a\": ";
            num(k.a);
            std::cout << ", \"lower_is_finite\": " << (k.lowerIsFinite ? "true" : "false");
            std::cout << ", \"rel_tolerance\": ";
            num(k.relTolerance);
            std::cout << ", \"half_infinite_overload\": "
                      << (k.useHalfInfiniteOverload ? "true" : "false");
            std::cout << ",\n     \"value\": ";
            num(v);
            std::cout << ", \"absolute_error\": ";
            num(es.absoluteError());
            std::cout << ", \"evaluations\": " << es.numberOfEvaluations();
            std::cout << "}" << (c + 1 < cases.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";
    }

    // ============================= 14. environment / sanity ===============
    std::cout << "  \"meta\": {\"null_real\": ";
    num(Null<Real>());
    std::cout << ", \"null_size\": " << Size(Null<Size>()) << "}\n";

    std::cout << "}\n";
    return 0;
}
