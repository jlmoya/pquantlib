// migration-harness/cpp/probes/v143_math_distributions/probe.cpp
//
// Reference values for the ql/math/distributions classes PQuantLib was missing,
// plus an audit of the two distributions PQuantLib already had but implemented
// by delegating to scipy (the non-central chi-square CDF and the bivariate
// cumulative normal). Those two are emitted here precisely so the Python side
// can be held to the C++ numbers rather than to a docstring claiming the
// delegation is equivalent.
//
// What is pinned, and why:
//
//   * every distribution is sampled across the body AND both tails — a CDF
//     approximation that is fine in the body can be several orders out at
//     1e-12, and that is exactly where an inverted CDF gets used;
//   * the inverse routines are sampled at probabilities close to 0 and 1, and
//     their C++ iteration parameters (accuracy / max-evaluations) are pinned
//     alongside, because the iteration cap is part of the answer;
//   * TabulatedGaussLegendre is exercised on a polynomial it must integrate
//     exactly (degree <= 2n-1), a polynomial one order too high, an analytic
//     non-polynomial, and a function with a kink at 0 which the rule cannot
//     resolve — the last one discriminates between "same nodes and weights"
//     and "some other quadrature that happens to be accurate".
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/distributions.json.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/beta.hpp>
#include <ql/math/incompletegamma.hpp>
#include <ql/math/distributions/binomialdistribution.hpp>
#include <ql/math/distributions/bivariatenormaldistribution.hpp>
#include <ql/math/distributions/bivariatestudenttdistribution.hpp>
#include <ql/math/distributions/chisquaredistribution.hpp>
#include <ql/math/distributions/gammadistribution.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/distributions/poissondistribution.hpp>
#include <ql/math/distributions/studenttdistribution.hpp>
#include <ql/math/integrals/gaussianquadratures.hpp>

using namespace QuantLib;

namespace {

std::ostream& out = std::cout;

void num(Real v) {
    if (std::isnan(v)) {
        out << "\"nan\"";
    } else if (std::isinf(v)) {
        out << (v > 0 ? "\"inf\"" : "\"-inf\"");
    } else {
        out << v;
    }
}

// Several of the inverse routines have a hard evaluation cap and throw rather
// than return a degraded answer. Throwing IS the pinned behaviour, so it is
// recorded as the sentinel "raises" and the Python test asserts the port
// raises too.
template <class F>
void numOrRaise(const F& f) {
    Real v;
    try {
        v = f();
    } catch (const std::exception&) {
        out << "\"raises\"";
        return;
    }
    num(v);
}

// ---------------------------------------------------------------- samples --

// Body and both tails. The extreme entries are where an approximation that is
// "good enough" in the body stops being good enough.
const std::vector<Real> kNormalArgs = {
    -37.0, -20.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5,
    -1e-8, 0.0, 1e-8, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, 37.0
};

// Probabilities spanning 1e-15 .. 1-1e-15.
const std::vector<Real> kProbs = {
    1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 0.001, 0.01, 0.02424, 0.02425, 0.02426,
    0.05, 0.1, 0.25, 0.4, 0.49999, 0.5, 0.50001, 0.6, 0.75, 0.9, 0.95,
    0.97574, 0.97575, 0.97576, 0.99, 0.999, 1.0 - 1e-4, 1.0 - 1e-6,
    1.0 - 1e-9, 1.0 - 1e-12
};

// ------------------------------------------------------------- gamma block --

void emitGamma() {
    const std::vector<Real> xs = {
        0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 7.0, 20.0,
        50.0, 171.0
    };
    GammaFunction g;
    out << "  \"gamma_function\": {\n    \"log_value\": [";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        out << (i ? ", " : "") << "[" << xs[i] << ", ";
        num(g.logValue(xs[i]));
        out << "]";
    }
    out << "],\n    \"value\": [";
    // value() also exercises the x<1 recurrence and the x<=-20 reflection.
    const std::vector<Real> vs = {
        0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 0.1, 0.001, -0.5, -1.5, -10.5, -20.5,
        -25.5
    };
    for (std::size_t i = 0; i < vs.size(); ++i) {
        out << (i ? ", " : "") << "[" << vs[i] << ", ";
        num(g.value(vs[i]));
        out << "]";
    }
    out << "]\n  },\n";
}

void emitIncompleteGamma() {
    // (a, x) pairs straddling the x < a+1 series / continued-fraction switch.
    const std::vector<std::pair<Real, Real>> cases = {
        {0.5, 0.0}, {0.5, 0.1}, {0.5, 1.0}, {0.5, 5.0},
        {1.0, 0.5}, {1.0, 2.0}, {1.0, 20.0},
        {2.5, 1.0}, {2.5, 3.5}, {2.5, 3.4999999}, {2.5, 3.5000001},
        {5.0, 2.0}, {5.0, 6.0}, {5.0, 30.0},
        {20.0, 10.0}, {20.0, 21.0}, {20.0, 60.0},
        {100.0, 90.0}, {100.0, 101.0}, {100.0, 200.0}
    };
    out << "  \"incomplete_gamma\": [";
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const Real a = cases[i].first, x = cases[i].second;
        out << (i ? ",\n    " : "\n    ") << "{\"a\": " << a << ", \"x\": " << x
            << ", \"P\": ";
        num(incompleteGammaFunction(a, x));
        // The two representations are only valid on their own side of the
        // switch; call each where it converges.
        out << ", \"series\": ";
        if (x < a + 1.0) num(incompleteGammaFunctionSeriesRepr(a, x)); else out << "null";
        out << ", \"continued_fraction\": ";
        if (x >= a + 1.0) num(incompleteGammaFunctionContinuedFractionRepr(a, x)); else out << "null";
        out << "}";
    }
    out << "\n  ],\n";
}

void emitCumulativeGamma() {
    const std::vector<Real> as = {0.25, 0.5, 1.0, 2.5, 5.0, 20.0, 100.0};
    const std::vector<Real> xs = {-1.0, 0.0, 1e-6, 0.1, 0.5, 1.0, 2.0, 5.0,
                                  10.0, 25.0, 60.0, 150.0, 400.0};
    out << "  \"cumulative_gamma\": [";
    bool first = true;
    for (Real a : as) {
        CumulativeGammaDistribution f(a);
        for (Real x : xs) {
            out << (first ? "\n    " : ",\n    ");
            first = false;
            out << "{\"a\": " << a << ", \"x\": " << x << ", \"v\": ";
            num(f(x));
            out << "}";
        }
    }
    out << "\n  ],\n";
}

// ----------------------------------------------------------- poisson block --

void emitPoisson() {
    const std::vector<Real> mus = {0.0, 0.001, 0.5, 1.0, 3.7, 10.0, 50.0};
    const std::vector<BigNatural> ks = {0, 1, 2, 3, 5, 10, 20, 40, 80, 150};
    out << "  \"poisson\": [";
    bool first = true;
    for (Real mu : mus) {
        PoissonDistribution pdf(mu);
        CumulativePoissonDistribution cdf(mu);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"mu\": " << mu << ", \"pdf\": [";
        for (std::size_t i = 0; i < ks.size(); ++i) {
            out << (i ? ", " : "") << "[" << ks[i] << ", ";
            num(pdf(ks[i]));
            out << "]";
        }
        out << "], \"cdf\": [";
        for (std::size_t i = 0; i < ks.size(); ++i) {
            // CumulativePoissonDistribution is 1 - P(k+1, mu); mu == 0 makes
            // incompleteGammaFunction's x == 0 branch return 0 => cdf == 1.
            out << (i ? ", " : "") << "[" << ks[i] << ", ";
            num(cdf(ks[i]));
            out << "]";
        }
        out << "]}";
    }
    out << "\n  ],\n";

    // InverseCumulativePoisson requires lambda > 0 and x in [0, 1]; x == 1.0
    // returns QL_MAX_REAL, which is pinned as a string sentinel.
    const std::vector<Real> lambdas = {0.5, 1.0, 4.0, 12.0};
    const std::vector<Real> xs = {0.0, 1e-9, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9,
                                  0.99, 0.999999, 1.0};
    out << "  \"inverse_cumulative_poisson\": [";
    first = true;
    for (Real lam : lambdas) {
        InverseCumulativePoisson inv(lam);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"lambda\": " << lam << ", \"cases\": [";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            out << (i ? ", " : "") << "[" << xs[i] << ", ";
            numOrRaise([&] { return inv(xs[i]); });
            out << "]";
        }
        out << "]}";
    }
    out << "\n  ],\n";
}

// ---------------------------------------------------------- binomial block --

void emitBinomial() {
    const std::vector<std::pair<BigNatural, BigNatural>> nk = {
        {0, 0}, {1, 0}, {1, 1}, {5, 2}, {10, 5}, {20, 7}, {50, 25},
        {100, 3}, {100, 50}, {170, 85}
    };
    out << "  \"binomial_coefficient\": [";
    for (std::size_t i = 0; i < nk.size(); ++i) {
        out << (i ? ",\n    " : "\n    ") << "{\"n\": " << nk[i].first
            << ", \"k\": " << nk[i].second << ", \"ln\": ";
        num(binomialCoefficientLn(nk[i].first, nk[i].second));
        out << ", \"value\": ";
        num(binomialCoefficient(nk[i].first, nk[i].second));
        out << "}";
    }
    out << "\n  ],\n";

    const std::vector<Real> ps = {0.0, 1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6, 1.0};
    const std::vector<BigNatural> ns = {1, 5, 20, 101};
    out << "  \"binomial\": [";
    bool first = true;
    for (Real p : ps) {
        for (BigNatural n : ns) {
            BinomialDistribution pdf(p, n);
            CumulativeBinomialDistribution cdf(p, n);
            out << (first ? "\n    " : ",\n    ");
            first = false;
            out << "{\"p\": " << p << ", \"n\": " << n << ", \"pdf\": [";
            bool f2 = true;
            for (BigNatural k = 0; k <= n + 1; ++k) {
                if (n > 20 && k % 7 != 0 && k != n && k != n + 1) continue;
                out << (f2 ? "" : ", ") << "[" << k << ", ";
                f2 = false;
                num(pdf(k));
                out << "]";
            }
            out << "], \"cdf\": [";
            f2 = true;
            for (BigNatural k = 0; k <= n + 1; ++k) {
                if (n > 20 && k % 7 != 0 && k != n && k != n + 1) continue;
                out << (f2 ? "" : ", ") << "[" << k << ", ";
                f2 = false;
                num(cdf(k));
                out << "]";
            }
            out << "]}";
        }
    }
    out << "\n  ],\n";

    // PeizerPrattMethod2Inversion already exists in Python; pinned as a
    // regression anchor for the odd-n precondition.
    out << "  \"peizer_pratt\": [";
    const std::vector<std::pair<Real, BigNatural>> pp = {
        {-3.0, 51}, {-1.0, 51}, {0.0, 51}, {1.0, 51}, {3.0, 51},
        {-2.0, 5}, {2.0, 5}, {0.5, 1001}
    };
    for (std::size_t i = 0; i < pp.size(); ++i) {
        out << (i ? ", " : "") << "[" << pp[i].first << ", " << pp[i].second
            << ", ";
        num(PeizerPrattMethod2Inversion(pp[i].first, pp[i].second));
        out << "]";
    }
    out << "],\n";
}

// --------------------------------------------------------- chi-square block --

void emitChiSquare() {
    const std::vector<Real> dfs = {0.5, 1.0, 2.0, 4.0, 7.5, 30.0};
    const std::vector<Real> xs = {-1.0, 0.0, 1e-8, 0.1, 0.5, 1.0, 2.0, 4.0,
                                  7.0, 15.0, 40.0, 100.0};
    out << "  \"cumulative_chi_square\": [";
    bool first = true;
    for (Real df : dfs) {
        CumulativeChiSquareDistribution f(df);
        for (Real x : xs) {
            out << (first ? "\n    " : ",\n    ");
            first = false;
            out << "{\"df\": " << df << ", \"x\": " << x << ", \"v\": ";
            num(f(x));
            out << "}";
        }
    }
    out << "\n  ],\n";

    // The non-central CDF: this is the audit of PQuantLib's scipy ncx2
    // delegation. Sampled into both tails on purpose.
    const std::vector<std::pair<Real, Real>> dfncp = {
        {1.0, 0.0}, {1.0, 0.5}, {2.0, 1.0}, {4.0, 2.0}, {4.0, 20.0},
        {0.5, 0.25}, {10.0, 5.0}, {30.0, 100.0}, {2.5, 0.001}
    };
    const std::vector<Real> ncxs = {0.0, 1e-6, 0.01, 0.1, 0.5, 1.0, 3.0, 6.0,
                                    12.0, 30.0, 80.0, 200.0};
    out << "  \"noncentral_chi_square\": [";
    first = true;
    for (const auto& dn : dfncp) {
        NonCentralCumulativeChiSquareDistribution f(dn.first, dn.second);
        NonCentralCumulativeChiSquareSankaranApprox s(dn.first, dn.second);
        for (Real x : ncxs) {
            out << (first ? "\n    " : ",\n    ");
            first = false;
            out << "{\"df\": " << dn.first << ", \"ncp\": " << dn.second
                << ", \"x\": " << x << ", \"cdf\": ";
            num(f(x));
            out << ", \"sankaran\": ";
            // Sankaran is a closed-form approximation and is only defined for
            // x > 0 (it takes pow(x/(df+ncp), h)).
            if (x > 0.0) num(s(x)); else out << "null";
            out << "}";
        }
    }
    out << "\n  ],\n";

    // Inverse non-central chi-square: the doubling search + Brent. The
    // max-evaluations and accuracy are part of the answer, so pin them.
    out << "  \"inverse_noncentral_chi_square\": {\n"
        << "    \"max_evaluations\": 10,\n    \"accuracy\": 1e-8,\n"
        << "    \"cases\": [";
    const std::vector<Real> invxs = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};
    first = true;
    for (const auto& dn : dfncp) {
        InverseNonCentralCumulativeChiSquareDistribution inv(dn.first, dn.second);
        for (Real x : invxs) {
            out << (first ? "\n      " : ",\n      ");
            first = false;
            out << "{\"df\": " << dn.first << ", \"ncp\": " << dn.second
                << ", \"x\": " << x << ", \"v\": ";
            numOrRaise([&] { return inv(x); });
            out << "}";
        }
    }
    out << "\n    ]\n  },\n";
}

// ------------------------------------------------------------ student block --

void emitStudent() {
    const std::vector<Integer> ns = {1, 2, 3, 4, 7, 30, 200};
    const std::vector<Real> xs = {-50.0, -10.0, -4.0, -2.0, -1.0, -0.25, 0.0,
                                  0.25, 1.0, 2.0, 4.0, 10.0, 50.0};
    out << "  \"student\": [";
    bool first = true;
    for (Integer n : ns) {
        StudentDistribution pdf(n);
        CumulativeStudentDistribution cdf(n);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"n\": " << n << ", \"pdf\": [";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            out << (i ? ", " : "") << "[" << xs[i] << ", ";
            num(pdf(xs[i]));
            out << "]";
        }
        out << "], \"cdf\": [";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            out << (i ? ", " : "") << "[" << xs[i] << ", ";
            num(cdf(xs[i]));
            out << "]";
        }
        out << "]}";
    }
    out << "\n  ],\n";

    // InverseCumulativeStudent is a bare Newton iteration started at x = 0
    // with a fixed accuracy and a 50-step cap. Both are pinned; the sample
    // probabilities stay where the iteration converges within the cap.
    out << "  \"inverse_student\": {\n    \"accuracy\": 1e-6,\n"
        << "    \"max_iterations\": 50,\n    \"cases\": [";
    const std::vector<Real> ys = {0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95,
                                  0.99, 0.999};
    first = true;
    for (Integer n : ns) {
        InverseCumulativeStudent inv(n);
        for (Real y : ys) {
            out << (first ? "\n      " : ",\n      ");
            first = false;
            out << "{\"n\": " << n << ", \"y\": " << y << ", \"v\": ";
            numOrRaise([&] { return inv(y); });
            out << "}";
        }
    }
    out << "\n    ]\n  },\n";

    // Bivariate Student-t (Dunnett & Sobel 1954): the even-n and odd-n
    // branches are structurally different formulas, so both are sampled, and
    // rho is taken to both signs and close to the +/-1 limit where the
    // f_x epsilon guard fires.
    const std::vector<Natural> bns = {1, 2, 3, 4, 5, 8, 15};
    const std::vector<Real> rhos = {-0.999999, -0.75, -0.3, 0.0, 0.3, 0.75,
                                    0.999999};
    const std::vector<std::pair<Real, Real>> pts = {
        {-4.0, -4.0}, {-2.0, 1.0}, {-1.0, -0.5}, {0.0, 0.0}, {0.5, -1.5},
        {1.0, 2.0}, {3.0, 3.0}
    };
    out << "  \"bivariate_student\": [";
    first = true;
    for (Natural n : bns) {
        for (Real rho : rhos) {
            BivariateCumulativeStudentDistribution f(n, rho);
            out << (first ? "\n    " : ",\n    ");
            first = false;
            out << "{\"n\": " << n << ", \"rho\": " << rho << ", \"cases\": [";
            for (std::size_t i = 0; i < pts.size(); ++i) {
                out << (i ? ", " : "") << "[" << pts[i].first << ", "
                    << pts[i].second << ", ";
                num(f(pts[i].first, pts[i].second));
                out << "]";
            }
            out << "]}";
        }
    }
    out << "\n  ],\n";
}

// ------------------------------------------------------------ normal block --

void emitNormal() {
    MoroInverseCumulativeNormal moro;
    MoroInverseCumulativeNormal moroShifted(0.25, 2.5);
    out << "  \"moro_inverse_cumulative_normal\": {\n    \"standard\": [";
    for (std::size_t i = 0; i < kProbs.size(); ++i) {
        out << (i ? ", " : "") << "[" << kProbs[i] << ", ";
        num(moro(kProbs[i]));
        out << "]";
    }
    out << "],\n    \"average\": 0.25, \"sigma\": 2.5,\n    \"shifted\": [";
    for (std::size_t i = 0; i < kProbs.size(); ++i) {
        out << (i ? ", " : "") << "[" << kProbs[i] << ", ";
        num(moroShifted(kProbs[i]));
        out << "]";
    }
    out << "]\n  },\n";

    MaddockCumulativeNormal mcdf;
    MaddockCumulativeNormal mcdfShifted(-1.5, 0.75);
    out << "  \"maddock_cumulative_normal\": {\n    \"standard\": [";
    for (std::size_t i = 0; i < kNormalArgs.size(); ++i) {
        out << (i ? ", " : "") << "[" << kNormalArgs[i] << ", ";
        num(mcdf(kNormalArgs[i]));
        out << "]";
    }
    out << "],\n    \"average\": -1.5, \"sigma\": 0.75,\n    \"shifted\": [";
    for (std::size_t i = 0; i < kNormalArgs.size(); ++i) {
        out << (i ? ", " : "") << "[" << kNormalArgs[i] << ", ";
        num(mcdfShifted(kNormalArgs[i]));
        out << "]";
    }
    out << "]\n  },\n";

    MaddockInverseCumulativeNormal minv;
    MaddockInverseCumulativeNormal minvShifted(-1.5, 0.75);
    out << "  \"maddock_inverse_cumulative_normal\": {\n    \"standard\": [";
    for (std::size_t i = 0; i < kProbs.size(); ++i) {
        out << (i ? ", " : "") << "[" << kProbs[i] << ", ";
        num(minv(kProbs[i]));
        out << "]";
    }
    out << "],\n    \"average\": -1.5, \"sigma\": 0.75,\n    \"shifted\": [";
    for (std::size_t i = 0; i < kProbs.size(); ++i) {
        out << (i ? ", " : "") << "[" << kProbs[i] << ", ";
        num(minvShifted(kProbs[i]));
        out << "]";
    }
    out << "]\n  },\n";
}

void emitBivariateNormal() {
    // rho covers every branch of Dr78's case analysis (signs of a, b, rho and
    // the a*b*rho > 0 rotation) and every branch of We04DP (|rho| < 0.3,
    // < 0.75, < 0.925, >= 0.925, and rho < 0 inside the tail branch).
    const std::vector<Real> rhos = {-0.999, -0.95, -0.9, -0.8, -0.5, -0.2, 0.0,
                                    0.2, 0.5, 0.8, 0.9, 0.95, 0.999};
    const std::vector<std::pair<Real, Real>> pts = {
        {-8.0, -8.0}, {-6.0, 2.0}, {-3.0, -3.0}, {-2.0, 1.0}, {-1.0, -1.0},
        {-0.5, 0.5}, {0.0, 0.0}, {0.5, -0.5}, {1.0, 1.0}, {2.0, -1.0},
        {3.0, 3.0}, {6.0, -2.0}, {8.0, 8.0}
    };
    out << "  \"bivariate_normal_dr78\": [";
    bool first = true;
    for (Real rho : rhos) {
        BivariateCumulativeNormalDistributionDr78 f(rho);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"rho\": " << rho << ", \"cases\": [";
        for (std::size_t i = 0; i < pts.size(); ++i) {
            out << (i ? ", " : "") << "[" << pts[i].first << ", "
                << pts[i].second << ", ";
            num(f(pts[i].first, pts[i].second));
            out << "]";
        }
        out << "]}";
    }
    out << "\n  ],\n";

    out << "  \"bivariate_normal_we04dp\": [";
    first = true;
    for (Real rho : rhos) {
        BivariateCumulativeNormalDistributionWe04DP f(rho);
        out << (first ? "\n    " : ",\n    ");
        first = false;
        out << "{\"rho\": " << rho << ", \"cases\": [";
        for (std::size_t i = 0; i < pts.size(); ++i) {
            out << (i ? ", " : "") << "[" << pts[i].first << ", "
                << pts[i].second << ", ";
            num(f(pts[i].first, pts[i].second));
            out << "]";
        }
        out << "]}";
    }
    out << "\n  ],\n";

    // rho == +/-1 exactly: the degenerate case both classes must still accept.
    out << "  \"bivariate_normal_degenerate\": [";
    const std::vector<Real> deg = {-1.0, 1.0};
    first = true;
    for (Real rho : deg) {
        BivariateCumulativeNormalDistributionWe04DP w(rho);
        for (const auto& p : pts) {
            out << (first ? "\n    " : ",\n    ");
            first = false;
            out << "{\"rho\": " << rho << ", \"a\": " << p.first
                << ", \"b\": " << p.second << ", \"we04dp\": ";
            num(w(p.first, p.second));
            out << "}";
        }
    }
    out << "\n  ],\n";
}

// -------------------------------------------- tabulated Gauss-Legendre rule --

void emitTabulatedGaussLegendre() {
    // f1: degree 11 polynomial — exact for order >= 12, not for order 6.
    // f2: degree 5 polynomial — exact for every supported order.
    // f3: analytic non-polynomial.
    // f4: |x| — a kink at 0 the rule cannot resolve at any order. This is the
    //     entry that discriminates "same nodes and weights" from "some other
    //     quadrature that is merely accurate".
    struct Case { const char* name; Real (*f)(Real); };
    const Case cases[] = {
        {"poly5",  [](Real x) { return 3.0*x*x*x*x*x - 2.0*x*x*x + x - 0.5; }},
        {"poly11", [](Real x) { return std::pow(x, 11) + 4.0*std::pow(x, 8) - x*x; }},
        {"exp",    [](Real x) { return std::exp(x); }},
        {"runge",  [](Real x) { return 1.0/(1.0 + 25.0*x*x); }},
        {"abs",    [](Real x) { return std::fabs(x); }}
    };
    const std::vector<Size> orders = {6, 7, 12, 20};
    out << "  \"tabulated_gauss_legendre\": [";
    bool first = true;
    for (Size o : orders) {
        TabulatedGaussLegendre q(o);
        for (const auto& c : cases) {
            out << (first ? "\n    " : ",\n    ");
            first = false;
            out << "{\"order\": " << o << ", \"f\": \"" << c.name
                << "\", \"v\": ";
            num(q(c.f));
            out << "}";
        }
    }
    out << "\n  ]\n";
}

} // namespace

int main() {
    out << std::setprecision(17);
    out << "{\n";
    emitGamma();
    emitIncompleteGamma();
    emitCumulativeGamma();
    emitPoisson();
    emitBinomial();
    emitChiSquare();
    emitStudent();
    emitNormal();
    emitBivariateNormal();
    emitTabulatedGaussLegendre();
    out << "}\n";
    return 0;
}
