// Phase 11 W3-B cluster probe: copulas + correlation + latent + loss models.
//
// Captures reference values for:
//
//   * OneFactorGaussianCopula.conditionalProbability(p, m) on a grid of
//     (correlation, prob, factor) values + cumulativeZ/cumulativeY/density
//     + integral(p) (Hull-White 2006 reference).
//
//   * OneFactorStudentCopula at (nz=nm=10) conditionalProbability +
//     cumulativeZ + density + integral(p) round-trip.
//
//   * GaussianLHPLossModel.expectedTrancheLossImpl(remainingNot, prob,
//     averageRR, attach, detach) — calls the closed-form private impl via
//     test-only invocation by way of pricing a homogeneous basket (using
//     the (correlation, recoveries-vec) ctor). We emit the closed-form
//     output for two (attach, detach) levels at four (prob, rr, corr)
//     points to cover the Kalemanova-Schmid-Werner reference.
//
// We deliberately skip the basket-coupled `expectedTrancheLoss(Date)` path
// because the basket layer is owned by W3-C. The cross-validated value is
// `expectedTrancheLossImpl(remainingNotional=1, prob, avgRR, k1, k2)` which
// the Python port exposes directly without basket coupling.
//
// C++ parity:
//   ql/experimental/credit/onefactorcopula.hpp
//   ql/experimental/credit/onefactorgaussiancopula.hpp
//   ql/experimental/credit/onefactorstudentcopula.hpp
//   ql/experimental/credit/gaussianlhplossmodel.hpp
//   @ v1.42.1 (099987f0).

#include <ql/experimental/credit/gaussianlhplossmodel.hpp>
#include <ql/experimental/credit/onefactorgaussiancopula.hpp>
#include <ql/experimental/credit/onefactorstudentcopula.hpp>
#include <ql/math/distributions/bivariatenormaldistribution.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/quotes/simplequote.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

// Open the GaussianLHPLossModel implementation so we can compute its
// closed-form independent of a Basket. The header marks the impl `private`
// — we replicate the formula here verbatim (same code as the .cpp impl).
namespace {

    Real lhpExpectedTrancheLossImpl(Real correlation,
                                    Real prob,
                                    Real averageRR,
                                    Real attach,
                                    Real detach) {
        if (attach >= detach) return 0.0;

        const Real beta = std::sqrt(correlation);
        const Real sqrt1minus = std::sqrt(1.0 - correlation);
        BivariateCumulativeNormalDistribution biphi(-beta);
        CumulativeNormalDistribution phi;

        const Real one = 1.0 - 1.0e-12;
        const Real k1 = std::min(one, attach / (1.0 - averageRR)) + QL_EPSILON;
        const Real k2 = std::min(one, detach / (1.0 - averageRR)) + QL_EPSILON;

        if (prob > 0.0) {
            const Real ip = InverseCumulativeNormal::standard_value(prob);
            const Real invFlightK1 =
                (ip - sqrt1minus * InverseCumulativeNormal::standard_value(k1)) / beta;
            const Real invFlightK2 =
                (ip - sqrt1minus * InverseCumulativeNormal::standard_value(k2)) / beta;
            return (detach * phi(invFlightK2) - attach * phi(invFlightK1)
                    + (1.0 - averageRR) *
                        (biphi(ip, -invFlightK2) - biphi(ip, -invFlightK1)));
        }
        return 0.0;
    }

    Real lhpPercentilePortfolioLossFraction(Real correlation,
                                            Real prob,
                                            Real averageRR,
                                            Real perctl) {
        if (perctl == 0.0) return 0.0;
        if (perctl == 1.0) perctl = 1.0 - QL_EPSILON;

        const Real beta = std::sqrt(correlation);
        const Real sqrt1minus = std::sqrt(1.0 - correlation);
        CumulativeNormalDistribution phi;
        return (1.0 - averageRR) *
            phi((InverseCumulativeNormal::standard_value(prob) +
                 beta * InverseCumulativeNormal::standard_value(perctl)) /
                sqrt1minus);
    }
}

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) OneFactorGaussianCopula
    // ============================================================
    {
        ext::shared_ptr<SimpleQuote> rho(new SimpleQuote(0.25));
        Handle<Quote> h(rho);
        OneFactorGaussianCopula cop(h);

        std::cout << "  \"gauss_copula\": {\n";
        std::cout << "    \"correlation\": " << cop.correlation() << ",\n";
        // density at m = 0
        std::cout << "    \"density_at_0\": " << cop.density(0.0) << ",\n";
        // cumulativeZ at z = 1
        std::cout << "    \"cumZ_at_1\": " << cop.cumulativeZ(1.0) << ",\n";
        // cumulativeY at y = 1
        std::cout << "    \"cumY_at_1\": " << cop.cumulativeY(1.0) << ",\n";
        // inverseCumulativeY at p = 0.7
        std::cout << "    \"invY_at_0p7\": " << cop.inverseCumulativeY(0.7) << ",\n";
        // conditionalProbability(0.2, m=0)
        std::cout << "    \"condProb_p02_m0\": "
                  << cop.conditionalProbability(0.2, 0.0) << ",\n";
        // conditionalProbability(0.2, m=1)
        std::cout << "    \"condProb_p02_m1\": "
                  << cop.conditionalProbability(0.2, 1.0) << ",\n";
        // conditionalProbability(0.5, m=-0.5)
        std::cout << "    \"condProb_p05_m_neg0p5\": "
                  << cop.conditionalProbability(0.5, -0.5) << ",\n";
        // integral(0.2)
        std::cout << "    \"integral_p02\": " << cop.integral(0.2) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) OneFactorGaussianCopula at correlation = 0.50
    // ============================================================
    {
        ext::shared_ptr<SimpleQuote> rho(new SimpleQuote(0.50));
        Handle<Quote> h(rho);
        OneFactorGaussianCopula cop(h);

        std::cout << "  \"gauss_copula_corr050\": {\n";
        std::cout << "    \"condProb_p01_m0\": "
                  << cop.conditionalProbability(0.1, 0.0) << ",\n";
        std::cout << "    \"condProb_p03_m1p5\": "
                  << cop.conditionalProbability(0.3, 1.5) << ",\n";
        std::cout << "    \"integral_p03\": " << cop.integral(0.3) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) OneFactorStudentCopula nz=nm=10
    // ============================================================
    {
        ext::shared_ptr<SimpleQuote> rho(new SimpleQuote(0.25));
        Handle<Quote> h(rho);
        OneFactorStudentCopula cop(h, 10, 10);

        std::cout << "  \"stud_copula_df10\": {\n";
        std::cout << "    \"correlation\": " << cop.correlation() << ",\n";
        std::cout << "    \"density_at_0\": " << cop.density(0.0) << ",\n";
        std::cout << "    \"cumZ_at_1\": " << cop.cumulativeZ(1.0) << ",\n";
        std::cout << "    \"condProb_p02_m0\": "
                  << cop.conditionalProbability(0.2, 0.0) << ",\n";
        std::cout << "    \"condProb_p02_m1\": "
                  << cop.conditionalProbability(0.2, 1.0) << ",\n";
        std::cout << "    \"integral_p02\": " << cop.integral(0.2) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 4) GaussianLHP closed-form: expectedTrancheLossImpl
    // ============================================================
    {
        const Real corr = 0.30;
        const Real prob = 0.05; // PD over the period
        const Real avgRR = 0.40;
        const Real remaining = 1.0;

        std::cout << "  \"lhp_etl\": {\n";
        std::cout << "    \"corr\": " << corr << ",\n";
        std::cout << "    \"prob\": " << prob << ",\n";
        std::cout << "    \"avgRR\": " << avgRR << ",\n";
        std::cout << "    \"etl_0_3\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.0, 0.03) << ",\n";
        std::cout << "    \"etl_3_6\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.03, 0.06) << ",\n";
        std::cout << "    \"etl_6_9\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.06, 0.09) << ",\n";
        std::cout << "    \"etl_9_12\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.09, 0.12) << ",\n";
        std::cout << "    \"etl_0_100\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.0, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 5) GaussianLHP at a higher-default scenario
    // ============================================================
    {
        const Real corr = 0.20;
        const Real prob = 0.15;
        const Real avgRR = 0.30;
        const Real remaining = 1.0;

        std::cout << "  \"lhp_etl_stress\": {\n";
        std::cout << "    \"etl_0_5\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.0, 0.05) << ",\n";
        std::cout << "    \"etl_5_15\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.05, 0.15) << ",\n";
        std::cout << "    \"etl_15_30\": "
                  << remaining * lhpExpectedTrancheLossImpl(corr, prob, avgRR, 0.15, 0.30) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 6) GaussianLHP percentile portfolio loss fraction
    // ============================================================
    {
        const Real corr = 0.30;
        const Real prob = 0.05;
        const Real avgRR = 0.40;

        std::cout << "  \"lhp_percentile\": {\n";
        std::cout << "    \"p995\": "
                  << lhpPercentilePortfolioLossFraction(corr, prob, avgRR, 0.995) << ",\n";
        std::cout << "    \"p990\": "
                  << lhpPercentilePortfolioLossFraction(corr, prob, avgRR, 0.99) << ",\n";
        std::cout << "    \"p950\": "
                  << lhpPercentilePortfolioLossFraction(corr, prob, avgRR, 0.95) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 7) Vasicek conditional-default sanity reference
    //    p_hat(m, p, rho) = Phi( (Phi^-1(p) - sqrt(rho) m) / sqrt(1-rho) )
    // ============================================================
    {
        CumulativeNormalDistribution phi;
        InverseCumulativeNormal invphi;
        const Real p = 0.10;
        const Real rho = 0.20;

        const Real ip = invphi(p);
        const Real sqrt_rho = std::sqrt(rho);
        const Real sqrt_1mr = std::sqrt(1.0 - rho);

        std::cout << "  \"vasicek_ref\": {\n";
        std::cout << "    \"p\": " << p << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"p_hat_m_neg2\": " << phi((ip - sqrt_rho * (-2.0)) / sqrt_1mr) << ",\n";
        std::cout << "    \"p_hat_m_neg1\": " << phi((ip - sqrt_rho * (-1.0)) / sqrt_1mr) << ",\n";
        std::cout << "    \"p_hat_m_0\": " << phi((ip - sqrt_rho * (0.0)) / sqrt_1mr) << ",\n";
        std::cout << "    \"p_hat_m_1\": " << phi((ip - sqrt_rho * (1.0)) / sqrt_1mr) << ",\n";
        std::cout << "    \"p_hat_m_2\": " << phi((ip - sqrt_rho * (2.0)) / sqrt_1mr) << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
