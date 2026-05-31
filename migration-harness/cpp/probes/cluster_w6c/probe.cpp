// Phase 11 W6-C cluster probe: experimental math foundations.
//
// Captures reference values for:
//
//   * ConvolvedStudentT (CumulativeBehrensFisher) — cumulative of a sum
//     of n iid odd-order Student-t variables at known x.
//   * GaussNonCentralChiSquaredPolynomial — moments + GaussianQuadrature
//     node sums (Gautschi rule-of-sum test, nu=4 lambda=1).
//   * moorePenroseInverse — MATLAB pinv reference (8x6 magic-ish matrix),
//     minimal-norm solution check.
//   * GaussianQuadMultidimIntegrator — separable Gaussian-weighted
//     integrand factorises into a product of 1-D Gauss-Hermite results.
//   * MultidimIntegral — tensor-product trapezoid integration of a
//     separable integrand over a box.
//   * QL_PIECEWISE_FUNCTION — RCLL step lookup.
//   * PiecewiseIntegral — integrator split at critical points.
//   * GaussianCopulaPolicy / TCopulaPolicy — cumulative/inverse round-trip.
//   * LatentModel — latentVariableCorrel + cumulativeY/inverseCumulativeY
//     + integratedExpectedValue of a one-factor Gaussian model.
//   * laplaceInterpolation — Laplace in-fill of Null entries (matches the
//     interpolations.cpp reference grids).
//   * PolarStudentTRng — first sample from a seeded MT-backed generator.
//
// C++ parity:
//   ql/experimental/math/convolvedstudentt.hpp
//   ql/experimental/math/gaussiannoncentralchisquaredpolynomial.hpp
//   ql/experimental/math/moorepenroseinverse.hpp
//   ql/experimental/math/multidimquadrature.hpp
//   ql/experimental/math/multidimintegrator.hpp
//   ql/experimental/math/piecewisefunction.hpp
//   ql/experimental/math/piecewiseintegral.hpp
//   ql/experimental/math/gaussiancopulapolicy.hpp
//   ql/experimental/math/tcopulapolicy.hpp
//   ql/experimental/math/latentmodel.hpp
//   ql/experimental/math/laplaceinterpolation.hpp
//   ql/experimental/math/polarstudenttrng.hpp
//   @ v1.42.1 (099987f0).

#include <ql/errors.hpp>
#include <ql/experimental/math/convolvedstudentt.hpp>
#include <ql/experimental/math/gaussiancopulapolicy.hpp>
#include <ql/experimental/math/gaussiannoncentralchisquaredpolynomial.hpp>
#include <ql/experimental/math/laplaceinterpolation.hpp>
#include <ql/experimental/math/latentmodel.hpp>
#include <ql/experimental/math/moorepenroseinverse.hpp>
#include <ql/experimental/math/multidimintegrator.hpp>
#include <ql/experimental/math/multidimquadrature.hpp>
#include <ql/experimental/math/piecewisefunction.hpp>
#include <ql/experimental/math/piecewiseintegral.hpp>
#include <ql/experimental/math/polarstudenttrng.hpp>
#include <ql/experimental/math/tcopulapolicy.hpp>
#include <ql/math/array.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/integrals/gaussianquadratures.hpp>
#include <ql/math/integrals/trapezoidintegral.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

using namespace QuantLib;

namespace {
    std::string j(const std::string& key, Real v) {
        std::ostringstream os;
        os << "  \"" << key << "\": " << std::setprecision(17) << v;
        return os.str();
    }
}

// A trivial one-factor Gaussian latent model.  The LatentModel template is
// abstract on an Impl that supplies the random sampler; for the probe we only
// exercise the (Impl-independent) inspector + integration surface, so a
// minimal Impl whose nested templates are never instantiated suffices.
class TestLatentModel : public LatentModel<GaussianCopulaPolicy> {
  public:
    explicit TestLatentModel(const std::vector<std::vector<Real>>& w)
    : LatentModel<GaussianCopulaPolicy>(w, GaussianCopulaPolicy::initTraits()) {}
    // forced by the abstract integration() pure-virtual in the base
    const ext::shared_ptr<LMIntegration>& integration() const override {
        static ext::shared_ptr<LMIntegration> p;
        return p;
    }
};

int main() {
    std::vector<std::string> out;

    // ---- ConvolvedStudentT: sum of 2 iid t_3 with unit factors ----
    {
        std::vector<Integer> df = {3, 3};
        std::vector<Real> fac = {1.0, 1.0};
        CumulativeBehrensFisher cbf(df, fac);
        out.push_back(j("convolved_t3_t3_cum_at_0", cbf(0.0)));
        out.push_back(j("convolved_t3_t3_cum_at_1", cbf(1.0)));
        out.push_back(j("convolved_t3_t3_cum_at_2_5", cbf(2.5)));
        out.push_back(j("convolved_t3_t3_dens_at_0", cbf.density(0.0)));
        out.push_back(j("convolved_t3_t3_dens_at_1", cbf.density(1.0)));
        // single t_5 (degenerate convolution of one variable)
        std::vector<Integer> df1 = {5};
        std::vector<Real> fac1 = {1.0};
        CumulativeBehrensFisher cbf1(df1, fac1);
        out.push_back(j("convolved_t5_cum_at_1_5", cbf1(1.5)));
        // inverse round trip on a narrower (sub-unit factors) convolution so
        // the Brent bracket (xMax driven by sum-of-squares) stays valid.
        std::vector<Real> facHalf = {0.5, 0.5};
        CumulativeBehrensFisher cbfH(df, facHalf);
        out.push_back(j("convolved_t3_t3_half_cum_at_1", cbfH(1.0)));
        InverseCumulativeBehrensFisher inv(df, facHalf);
        out.push_back(j("convolved_t3_t3_half_inv_of_cum_at_1", inv(cbfH(1.0))));
    }

    // ---- GaussNonCentralChiSquaredPolynomial ----
    {
        GaussNonCentralChiSquaredPolynomial p(4.0, 1.0);
        out.push_back(j("ncchisq_4_1_moment_0", p.moment(0)));
        out.push_back(j("ncchisq_4_1_moment_1", p.moment(1)));
        out.push_back(j("ncchisq_4_1_moment_2", p.moment(2)));
        out.push_back(j("ncchisq_4_1_moment_5", p.moment(5)));
        out.push_back(j("ncchisq_4_1_w_at_3", p.w(3.0)));
        out.push_back(j("ncchisq_4_1_alpha_0", p.alpha(0)));
        out.push_back(j("ncchisq_4_1_alpha_2", p.alpha(2)));
        out.push_back(j("ncchisq_4_1_beta_1", p.beta(1)));
        out.push_back(j("ncchisq_4_1_beta_3", p.beta(3)));
        // Gautschi rule-of-sum: sum of nodes for n=4..9
        for (Size n = 4; n < 10; ++n) {
            const Array x = GaussianQuadrature(n, p).x();
            const Real s = std::accumulate(x.begin(), x.end(), Real(0.0));
            out.push_back(j("ncchisq_4_1_nodesum_n" + std::to_string(n), s));
        }
    }

    // ---- moorePenroseInverse (MATLAB pinv reference) ----
    {
        Real tmp[8][6] = {{64, 2, 3, 61, 60, 6},    {9, 55, 54, 12, 13, 51},
                          {17, 47, 46, 20, 21, 43}, {40, 26, 27, 37, 36, 30},
                          {32, 34, 35, 29, 28, 38}, {41, 23, 22, 44, 45, 19},
                          {49, 15, 14, 52, 53, 11}, {8, 58, 59, 5, 4, 62}};
        Matrix A(8, 6);
        for (Size i = 0; i < 8; ++i)
            for (Size k = 0; k < 6; ++k)
                A(i, k) = tmp[i][k];
        Matrix P = moorePenroseInverse(A);
        Array b(8, 260.0);
        Array x = P * b;
        for (Size i = 0; i < 6; ++i)
            out.push_back(j("mpinv_minnorm_x" + std::to_string(i), x[i]));
        // a couple of raw pseudo-inverse entries
        out.push_back(j("mpinv_P_0_0", P(0, 0)));
        out.push_back(j("mpinv_P_2_5", P(2, 5)));
    }

    // ---- GaussianQuadMultidimIntegrator: separable factorisation ----
    // f(x,y) = x^2 * y^2  integrated with Gauss-Hermite weight exp(-x^2)
    // factorises into (1-D x^2 integral)^2.
    {
        GaussianQuadMultidimIntegrator integ2(2, 12);
        std::function<Real(const std::vector<Real>&)> f2 =
            [](const std::vector<Real>& v) { return v[0] * v[0] * v[1] * v[1]; };
        out.push_back(j("multidim_quad_x2y2_dim2", integ2(f2)));
        // 1-D reference: same Gauss-Hermite, single dimension x^2
        GaussianQuadMultidimIntegrator integ1(1, 12);
        std::function<Real(const std::vector<Real>&)> f1 =
            [](const std::vector<Real>& v) { return v[0] * v[0]; };
        Real oneD = integ1(f1);
        out.push_back(j("multidim_quad_x2_dim1", oneD));
        out.push_back(j("multidim_quad_product_ref", oneD * oneD));
        // a 3-D one: f = x^2 y^2 z^2
        GaussianQuadMultidimIntegrator integ3(3, 12);
        std::function<Real(const std::vector<Real>&)> f3 =
            [](const std::vector<Real>& v) {
                return v[0] * v[0] * v[1] * v[1] * v[2] * v[2];
            };
        out.push_back(j("multidim_quad_x2y2z2_dim3", integ3(f3)));
    }

    // ---- MultidimIntegral: tensor-product trapezoid over a box ----
    // f(x,y) = x * y over [0,1]x[0,2] -> (1/2)*(2) = 1
    {
        auto t = ext::make_shared<TrapezoidIntegral<Default>>(1e-10, 100000);
        std::vector<ext::shared_ptr<Integrator>> integrators = {t, t};
        MultidimIntegral mdi(integrators);
        std::function<Real(const std::vector<Real>&)> f =
            [](const std::vector<Real>& v) { return v[0] * v[1]; };
        std::vector<Real> a = {0.0, 0.0};
        std::vector<Real> b = {1.0, 2.0};
        out.push_back(j("multidim_trap_xy_box", mdi(f, a, b)));
    }

    // ---- QL_PIECEWISE_FUNCTION (RCLL step lookup) ----
    {
        std::vector<Real> X = {1.0, 2.0, 3.0};
        std::vector<Real> Y = {10.0, 20.0, 30.0, 40.0};
        out.push_back(j("piecewise_at_0_5", QL_PIECEWISE_FUNCTION(X, Y, 0.5)));
        out.push_back(j("piecewise_at_1_0", QL_PIECEWISE_FUNCTION(X, Y, 1.0)));
        out.push_back(j("piecewise_at_1_5", QL_PIECEWISE_FUNCTION(X, Y, 1.5)));
        out.push_back(j("piecewise_at_2_0", QL_PIECEWISE_FUNCTION(X, Y, 2.0)));
        out.push_back(j("piecewise_at_3_0", QL_PIECEWISE_FUNCTION(X, Y, 3.0)));
        out.push_back(j("piecewise_at_5_0", QL_PIECEWISE_FUNCTION(X, Y, 5.0)));
    }

    // ---- PiecewiseIntegral: split at critical points ----
    // integrate f(x)=x over [0,4] with critical point at 2; result still 8.
    {
        auto base = ext::make_shared<TrapezoidIntegral<Default>>(1e-12, 100000);
        std::vector<Real> crit = {2.0};
        PiecewiseIntegral pwi(base, crit, true);
        std::function<Real(Real)> f = [](Real x) { return x; };
        out.push_back(j("piecewise_integral_x_0_4", pwi(f, 0.0, 4.0)));
    }

    // ---- GaussianCopulaPolicy: cumulative / inverse round trip ----
    {
        std::vector<std::vector<Real>> w = {{0.5}, {0.4}};
        GaussianCopulaPolicy gcp(w);
        out.push_back(j("gauss_copula_numFactors", Real(gcp.numFactors())));
        Real cy = gcp.cumulativeY(0.7, 0);
        out.push_back(j("gauss_copula_cumY_0_7", cy));
        out.push_back(j("gauss_copula_invY_of_cumY", gcp.inverseCumulativeY(cy, 0)));
        out.push_back(j("gauss_copula_cumZ_0_3", gcp.cumulativeZ(0.3)));
        std::vector<Real> m = {0.1, -0.2};
        out.push_back(j("gauss_copula_density", gcp.density(m)));
    }

    // ---- TCopulaPolicy: cumulative / inverse round trip ----
    {
        std::vector<std::vector<Real>> w = {{0.5}, {0.4}};
        TCopulaPolicy::initTraits tr;
        // tOrders size must equal nFactors (1) + 1 idiosyncratic = 2; all odd.
        tr.tOrders = {3, 3};
        TCopulaPolicy tcp(w, tr);
        out.push_back(j("t_copula_numFactors", Real(tcp.numFactors())));
        Real cy = tcp.cumulativeY(0.7, 0);
        out.push_back(j("t_copula_cumY_0_7", cy));
        out.push_back(j("t_copula_invY_of_cumY", tcp.inverseCumulativeY(cy, 0)));
        out.push_back(j("t_copula_cumZ_0_3", tcp.cumulativeZ(0.3)));
        out.push_back(j("t_copula_varfac_0", tcp.varianceFactors()[0]));
    }

    // ---- LatentModel: one-factor Gaussian inspectors + correl ----
    {
        std::vector<std::vector<Real>> w = {{0.5}, {0.4}, {0.3}};
        TestLatentModel lm(w);
        out.push_back(j("latent_size", Real(lm.size())));
        out.push_back(j("latent_numFactors", Real(lm.numFactors())));
        out.push_back(j("latent_numTotalFactors", Real(lm.numTotalFactors())));
        out.push_back(j("latent_idiosync_0", lm.idiosyncFctrs()[0]));
        out.push_back(j("latent_correl_0_1", lm.latentVariableCorrel(0, 1)));
        out.push_back(j("latent_correl_0_0", lm.latentVariableCorrel(0, 0)));
        out.push_back(j("latent_cumY_0", lm.cumulativeY(0.7, 0)));
        out.push_back(j("latent_invY_0",
                        lm.inverseCumulativeY(lm.cumulativeY(0.7, 0), 0)));
        // latentVarValue: allFactors = [M, Z_0, Z_1, Z_2]; Y_0 = 0.5*M + idio*Z_0
        std::vector<Real> allF = {1.0, 0.5, -0.5, 0.25};
        out.push_back(j("latent_varValue_0", lm.latentVarValue(allF, 0)));
    }

    // ---- laplaceInterpolation: in-fill of Null entries ----
    {
        Real na = Null<Real>();
        Matrix m2 = {{1.0, 2.0, 4.0}, {6.0, na, 7.0}, {5.0, 3.0, 2.0}};
        laplaceInterpolation(m2);
        out.push_back(j("laplace_inner_1_1", m2(1, 1)));

        Matrix m3 = {{1.0, na, 4.0}, {6.0, 6.5, 7.0}, {5.0, 3.0, 2.0}};
        laplaceInterpolation(m3);
        out.push_back(j("laplace_boundary_0_1", m3(0, 1)));

        Matrix m7 = {{na, 2.0, 4.0}, {6.0, 6.5, 7.0}, {5.0, 3.0, 2.0}};
        laplaceInterpolation(m7);
        out.push_back(j("laplace_corner_0_0", m7(0, 0)));
    }

    // ---- PolarStudentTRng: first sample ----
    {
        MersenneTwisterUniformRng mt(42UL);
        PolarStudentTRng<MersenneTwisterUniformRng> rng(5.0, mt);
        out.push_back(j("polar_t5_seed42_sample0", rng.next().value));
        out.push_back(j("polar_t5_seed42_sample1", rng.next().value));
    }

    std::cout << "{\n";
    for (Size i = 0; i < out.size(); ++i)
        std::cout << out[i] << (i + 1 < out.size() ? ",\n" : "\n");
    std::cout << "}\n";
    return 0;
}
