// migration-harness/cpp/probes/v143_experimental_latent/probe.cpp
//
// Reference values for the ql/experimental/math latent-model integration and
// sampling machinery @ v1.43. Five gap classes:
//
//   IntegrationBase        latentmodel.hpp:89 (primary) + the two full
//                          specialisations at :114 and :132
//   IntegrationFactory     latentmodel.hpp:446 (nested in LatentModel)
//   FactorSampler          latentmodel.hpp:407 (nested in LatentModel) plus
//                          the two partial specialisations at :742 and :768
//   VectorIntegrator       multidimquadrature.hpp:52 (private, nested in
//                          GaussianQuadMultidimIntegrator)
//   Ziggurat               zigguratrng.hpp:64 (RNG traits struct)
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// Three things here cannot be checked by "does the number look right":
//
// 1. VectorIntegrator is `private:` inside GaussianQuadMultidimIntegrator, so
//    it is only reachable through integrate<std::vector<Real>>(). And it has a
//    DEFECT: the leading term
//
//        std::vector<Real> term = f(x_[i]);                       // i = n-1
//        std::for_each(term.begin(), term.end(),
//                      [&](Real x) -> Real { return x * w_[i]; }); // NO-OP
//        std::vector<Real> sum = term;
//
//    passes `Real x` BY VALUE and discards the lambda's return, so `term` is
//    never scaled. The highest-index node therefore enters the sum with weight
//    1 instead of w_[n-1]. Block B pins this as arithmetic: the observed
//    integrateV output, the "bugged" reconstruction f(x[n-1]) + sum_{i<n-1}
//    w_i f(x_i), and the textbook sum_i w_i f(x_i). The port must reproduce
//    the bugged one.
//
// 2. The FactorSampler specialisation for PolarStudentTRng hands each
//    per-factor generator a COPY of its own just-seeded urng_
//    (latentmodel.hpp:786). Every factor therefore consumes the *same* uniform
//    stream, so the "independent" systemic factors come out perfectly
//    dependent. Block H pins the raw samples so a port cannot accidentally
//    "fix" it into independence.
//
// 3. IntegrationFactory hard-codes its parameters (quadrature order 25;
//    TrapezoidIntegral<Default>(1e-4, 20) over [-35, 35]). Block F pins those
//    by *behaviour*: the same integrand through the factory and through a
//    hand-built IntegrationBase must agree bit-for-bit.
//
// Block A pins GaussHermiteIntegration(25) nodes/weights first, because every
// quadrature number below is a function of them; if A disagrees, nothing else
// in the quadrature blocks means anything.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/latent.json.

#include <ql/experimental/math/gaussiancopulapolicy.hpp>
#include <ql/experimental/math/latentmodel.hpp>
#include <ql/experimental/math/multidimintegrator.hpp>
#include <ql/experimental/math/multidimquadrature.hpp>
#include <ql/experimental/math/polarstudenttrng.hpp>
#include <ql/experimental/math/tcopulapolicy.hpp>
#include <ql/experimental/math/zigguratrng.hpp>
#include <ql/math/integrals/gaussianquadratures.hpp>
#include <ql/math/integrals/trapezoidintegral.hpp>
#include <ql/math/randomnumbers/boxmullergaussianrng.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/math/randomnumbers/randomsequencegenerator.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
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

void emit_int(const std::string& name, long long v) {
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

void emit_iarr(const std::string& name, const std::vector<long long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

// --------------------------------------------------------------------------
// Integrands. FP contract is switched OFF: at -O3 Clang fuses `a*b + c` into
// an FMA and the fused value differs from the separately-rounded one in the
// last bit, which would surface as a "port bug" in a pinned number that is
// really an artefact of how the probe's own scaffolding was compiled.
// --------------------------------------------------------------------------

#pragma clang fp contract(off)

Real f_one(const std::vector<Real>&) { return 1.0; }
Real f_lin(const std::vector<Real>& x) { return x[0]; }
Real f_sqr(const std::vector<Real>& x) { return x[0] * x[0]; }
Real f_gauss(const std::vector<Real>& x) { return std::exp(-0.25 * x[0] * x[0]); }
Real f_cos(const std::vector<Real>& x) { return std::cos(x[0]); }
Real f_prod2(const std::vector<Real>& x) { return x[0] * x[1]; }
Real f_sqr2(const std::vector<Real>& x) { return x[0] * x[0] + x[1] * x[1]; }

// Vector integrand: four components with very different magnitudes, so the
// missing weight on the top node cannot hide behind a small term. NOTE the
// components do NOT decay at the extreme Hermite nodes (|x| ~ 6.16 at order
// 25), which makes them maximally sensitive to the ~6e-8 relative uncertainty
// the Golub-Welsch construction has in the outermost weight at that order.
std::vector<Real> fv(const std::vector<Real>& x) {
    return {1.0, x[0], x[0] * x[0], std::cos(x[0])};
}

// Same shape, but Gaussian-damped: every component is O(1e-9) at the extreme
// node, so the outer-weight uncertainty cannot reach the result and this one
// can be pinned at the tight tier.
std::vector<Real> fvd(const std::vector<Real>& x) {
    Real d = std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
    return {d, d * x[0], d * x[0] * x[0], d * std::cos(x[0])};
}

// --------------------------------------------------------------------------
// LatentModel is only integrable through a protected virtual integration();
// IntegrationFactory is a protected nested class. Derive to reach both.
// --------------------------------------------------------------------------

template <class Copula>
class ProbeLM : public LatentModel<Copula> {
  public:
    ProbeLM(const std::vector<std::vector<Real> >& w,
            LatentModelIntegrationType::LatentModelIntegrationType type,
            const typename Copula::initTraits& ini = typename Copula::initTraits())
    : LatentModel<Copula>(w, ini),
      integration_(LatentModel<Copula>::IntegrationFactory::createLMIntegration(
          w[0].size(), type)) {}

    // published so the probe can also drive the facility directly
    const ext::shared_ptr<LMIntegration>& integration() const override { return integration_; }

  private:
    ext::shared_ptr<LMIntegration> integration_;
};

}  // namespace

int main() {
    std::cout << "{\n";

    // ======================================================================
    // A. GaussHermiteIntegration(25, mu=0) — the 1-D rule every quadrature
    //    number below is built from. Note TqrEigenDecomposition sorts with
    //    std::greater<>, so x_ comes out DESCENDING; the port must keep that
    //    order because VectorIntegrator indexes it.
    // ======================================================================
    {
        GaussHermiteIntegration gh(25, 0.0);
        std::vector<Real> nodes, weights;
        for (Size i = 0; i < gh.order(); ++i) {
            nodes.push_back(gh.x()[i]);
            weights.push_back(gh.weights()[i]);
        }
        emit_int("A_gh25_order", static_cast<long long>(gh.order()));
        emit_arr("A_gh25_nodes", nodes);
        emit_arr("A_gh25_weights", weights);
        emit_bool("A_gh25_descending", nodes.front() > nodes.back());
        // scalar accumulation: sum runs i = order-1 .. 0
        emit("A_gh25_int_gaussdens",
             gh([](Real x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI); }));
        emit("A_gh25_int_x2gaussdens", gh([](Real x) {
                 return x * x * std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
             }));
    }

    // ======================================================================
    // B. VectorIntegrator (multidimquadrature.hpp:52). Only reachable via
    //    GaussianQuadMultidimIntegrator::integrate<std::vector<Real>>; in
    //    dimension 1 that call IS VectorIntegrator::operator() applied to
    //    x -> fv({x}).
    // ======================================================================
    {
        const Size order = 25;
        GaussianQuadMultidimIntegrator q1(1, order);
        std::vector<Real> got = q1.integrate<std::vector<Real> >(
            std::function<std::vector<Real>(const std::vector<Real>&)>(fv));
        emit_arr("B_vecint_dim1_order25", got);

        // Reconstruct both readings of the C++ source, from the same nodes.
        GaussHermiteIntegration gh(order, 0.0);
        std::vector<Real> bugged, textbook;
        {
            std::vector<Real> arg(1);
            Integer i = static_cast<Integer>(order) - 1;
            arg[0] = gh.x()[i];
            bugged = fv(arg);  // <- unweighted leading term (the defect)
            for (i--; i >= 0; --i) {
                arg[0] = gh.x()[i];
                std::vector<Real> term = fv(arg);
                for (Size j = 0; j < bugged.size(); ++j)
                    bugged[j] = gh.weights()[i] * term[j] + bugged[j];
            }
            textbook.assign(4, 0.0);
            for (Integer k = static_cast<Integer>(order) - 1; k >= 0; --k) {
                arg[0] = gh.x()[k];
                std::vector<Real> term = fv(arg);
                for (Size j = 0; j < textbook.size(); ++j)
                    textbook[j] += gh.weights()[k] * term[j];
            }
        }
        emit_arr("B_vecint_dim1_bugged_reconstruction", bugged);
        emit_arr("B_vecint_dim1_textbook_reconstruction", textbook);
        // Relative distance rather than == : the library's accumulation and
        // the probe-local reconstruction are the same expression but are not
        // guaranteed to be contracted into FMAs the same way, so they can
        // differ in the last bit. The point of these two numbers is the
        // ORDER OF MAGNITUDE: ~1e-16 for the bugged reading, ~1e-2 for the
        // textbook one.
        {
            std::vector<Real> rb, rt;
            for (Size j = 0; j < got.size(); ++j) {
                Real den = std::max(std::fabs(got[j]), 1e-300);
                rb.push_back(std::fabs(got[j] - bugged[j]) / den);
                rt.push_back(std::fabs(got[j] - textbook[j]) / den);
            }
            emit_arr("B_vecint_dim1_reldiff_bugged", rb);
            emit_arr("B_vecint_dim1_reldiff_textbook", rt);
        }
        // Sanity: with weight w_[24] the leading term would have contributed
        // w_[24]*fv[0] = w_[24]; instead it contributed 1.
        emit("B_vecint_top_weight", gh.weights()[order - 1]);
        // Gaussian-damped vector integrand: the same defect, but now every
        // term is O(1) and the result is pinnable at the tight tier.
        emit_arr("B_vecintd_dim1_order25",
                 q1.integrate<std::vector<Real> >(
                     std::function<std::vector<Real>(const std::vector<Real>&)>(fvd)));
        // mu != 0 exercises the generalized-Hermite weight passthrough.
        GaussianQuadMultidimIntegrator q1mu(1, 8, 0.3);
        emit_int("B_vecint_mu_order", static_cast<long long>(q1mu.order()));
        emit_arr("B_vecint_dim1_order8_mu0_3",
                 q1mu.integrate<std::vector<Real> >(
                     std::function<std::vector<Real>(const std::vector<Real>&)>(fv)));
        GaussHermiteIntegration gh8mu(8, 0.3);
        std::vector<Real> n8, w8;
        for (Size i = 0; i < 8; ++i) {
            n8.push_back(gh8mu.x()[i]);
            w8.push_back(gh8mu.weights()[i]);
        }
        emit_arr("B_gh8_mu0_3_nodes", n8);
        emit_arr("B_gh8_mu0_3_weights", w8);

        // Dimensions 2 and 3: the defect compounds once per recursion level.
        // Order 9 for the undamped 2-D case: at order 25 the extreme-node
        // weight is only good to ~6e-8 (see block A), which would swamp the
        // comparison for an integrand that does not decay.
        GaussianQuadMultidimIntegrator q2o9(2, 9);
        emit_arr("B_vecint_dim2_order9",
                 q2o9.integrate<std::vector<Real> >(
                     std::function<std::vector<Real>(const std::vector<Real>&)>(
                         [](const std::vector<Real>& x) -> std::vector<Real> {
                             return {1.0, x[0], x[1], x[0] * x[1]};
                         })));
        // ...and the damped 2-D case at the order the factory actually uses.
        GaussianQuadMultidimIntegrator q2(2, order);
        emit_arr("B_vecintd_dim2_order25",
                 q2.integrate<std::vector<Real> >(
                     std::function<std::vector<Real>(const std::vector<Real>&)>(
                         [](const std::vector<Real>& x) -> std::vector<Real> {
                             Real d = std::exp(-0.5 * (x[0] * x[0] + x[1] * x[1])) / (2.0 * M_PI);
                             return {d, d * x[0], d * x[1], d * x[0] * x[1]};
                         })));
        // The order-9 rule itself, so the Python side can bound the
        // dim-2 / dim-3 disagreement term by term instead of guessing a tier.
        GaussHermiteIntegration gh9(9, 0.0);
        std::vector<Real> n9, w9;
        for (Size i = 0; i < 9; ++i) {
            n9.push_back(gh9.x()[i]);
            w9.push_back(gh9.weights()[i]);
        }
        emit_arr("B_gh9_nodes", n9);
        emit_arr("B_gh9_weights", w9);

        GaussianQuadMultidimIntegrator q3(3, 9);
        emit_arr("B_vecint_dim3_order9",
                 q3.integrate<std::vector<Real> >(
                     std::function<std::vector<Real>(const std::vector<Real>&)>(
                         [](const std::vector<Real>& x) -> std::vector<Real> {
                             return {1.0, x[0] + x[1] + x[2], x[0] * x[1] * x[2]};
                         })));
        // Order 4 keeps the whole thing small enough to check by hand.
        GaussianQuadMultidimIntegrator q1o4(1, 4);
        emit_arr("B_vecint_dim1_order4",
                 q1o4.integrate<std::vector<Real> >(
                     std::function<std::vector<Real>(const std::vector<Real>&)>(fv)));
        GaussHermiteIntegration gh4(4, 0.0);
        std::vector<Real> n4, w4;
        for (Size i = 0; i < 4; ++i) {
            n4.push_back(gh4.x()[i]);
            w4.push_back(gh4.weights()[i]);
        }
        emit_arr("B_gh4_nodes", n4);
        emit_arr("B_gh4_weights", w4);
    }

    // ======================================================================
    // C. GaussianQuadMultidimIntegrator scalar path (already ported; pinned
    //    here because B and D sit on top of it).
    // ======================================================================
    {
        GaussianQuadMultidimIntegrator q1(1, 25);
        emit_int("C_quad_order", static_cast<long long>(q1.order()));
        emit("C_quad_dim1_one",
             q1.integrate<Real>(std::function<Real(const std::vector<Real>&)>(
                 [](const std::vector<Real>& x) { return std::exp(-0.5 * x[0] * x[0]); })));
        emit("C_quad_dim1_sqr",
             q1.integrate<Real>(std::function<Real(const std::vector<Real>&)>(
                 [](const std::vector<Real>& x) {
                     return x[0] * x[0] * std::exp(-0.5 * x[0] * x[0]);
                 })));
        GaussianQuadMultidimIntegrator q2(2, 25);
        emit("C_quad_dim2_prod",
             q2.integrate<Real>(std::function<Real(const std::vector<Real>&)>(
                 [](const std::vector<Real>& x) {
                     return x[0] * x[1] * std::exp(-0.5 * (x[0] * x[0] + x[1] * x[1]));
                 })));
        emit("C_quad_dim2_sqr",
             q2.integrate<Real>(std::function<Real(const std::vector<Real>&)>(
                 [](const std::vector<Real>& x) {
                     return (x[0] * x[0] + x[1] * x[1]) *
                            std::exp(-0.5 * (x[0] * x[0] + x[1] * x[1]));
                 })));
    }

    // ======================================================================
    // D. IntegrationBase<GaussianQuadMultidimIntegrator> (latentmodel.hpp:114)
    // ======================================================================
    {
        IntegrationBase<GaussianQuadMultidimIntegrator> ib(1, 25);
        emit_int("D_ibquad_order", static_cast<long long>(ib.order()));
        emit("D_ibquad_integrate_gaussdens", ib.integrate([](const std::vector<Real>& x) {
            return std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
        }));
        emit_arr("D_ibquad_integrateV",
                 ib.integrateV([](const std::vector<Real>& x) -> std::vector<Real> {
                     Real d = std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
                     return {d, d * x[0], d * x[0] * x[0]};
                 }));
        // an LMIntegration& dispatches to the same code through the vtable
        LMIntegration& lmi = ib;
        emit("D_ibquad_via_base", lmi.integrate([](const std::vector<Real>& x) {
            return std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
        }));

        IntegrationBase<GaussianQuadMultidimIntegrator> ib2(2, 25);
        emit("D_ibquad_dim2", ib2.integrate([](const std::vector<Real>& x) {
            return std::exp(-0.5 * (x[0] * x[0] + x[1] * x[1])) / (2.0 * M_PI);
        }));
    }

    // ======================================================================
    // E. IntegrationBase<MultidimIntegral> (latentmodel.hpp:132)
    // ======================================================================
    {
        std::vector<ext::shared_ptr<Integrator> > ints;
        ints.push_back(ext::make_shared<TrapezoidIntegral<Default> >(1.e-4, 20));
        IntegrationBase<MultidimIntegral> ib(ints, -35., 35.);
        emit_arr("E_ibtrap_a", ib.a_);
        emit_arr("E_ibtrap_b", ib.b_);
        emit("E_ibtrap_gaussdens", ib.integrate([](const std::vector<Real>& x) {
            return std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
        }));
        emit("E_ibtrap_x2gaussdens", ib.integrate([](const std::vector<Real>& x) {
            return x[0] * x[0] * std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
        }));
        // |x| and cos(x) against the normal density: sqrt(2/pi) and exp(-1/2),
        // neither of which is 1 (unlike the two above, which coincide).
        emit("E_ibtrap_absxgaussdens", ib.integrate([](const std::vector<Real>& x) {
            return std::fabs(x[0]) * std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
        }));
        emit("E_ibtrap_cosgaussdens", ib.integrate([](const std::vector<Real>& x) {
            return std::cos(x[0]) * std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
        }));
        // integrateV is NOT overridden here -> LMIntegration's QL_FAIL default.
        bool threw = false;
        std::string msg;
        try {
            LMIntegration& lmi = ib;
            (void)lmi.integrateV(
                [](const std::vector<Real>&) -> std::vector<Real> { return {1.0}; });
        } catch (const std::exception& e) {
            threw = true;
            msg = e.what();
        }
        emit_bool("E_ibtrap_integrateV_throws", threw);
        emit_bool("E_ibtrap_integrateV_msg_has_novector",
                  msg.find("No vector integration provided") != std::string::npos);

        // 2-D, smaller domain so the nested trapezoid stays cheap.
        std::vector<ext::shared_ptr<Integrator> > ints2;
        for (Size i = 0; i < 2; ++i)
            ints2.push_back(ext::make_shared<TrapezoidIntegral<Default> >(1.e-4, 20));
        IntegrationBase<MultidimIntegral> ib2(ints2, -8., 8.);
        emit_arr("E_ibtrap2_a", ib2.a_);
        emit_arr("E_ibtrap2_b", ib2.b_);
        emit("E_ibtrap2_gaussdens", ib2.integrate([](const std::vector<Real>& x) {
            return std::exp(-0.5 * (x[0] * x[0] + x[1] * x[1])) / (2.0 * M_PI);
        }));
    }

    // ======================================================================
    // F. IntegrationFactory::createLMIntegration (latentmodel.hpp:446).
    //    Pinned by behaviour: the factory's hard-coded parameters (order 25;
    //    trapezoid 1e-4/20 over [-35,35]) are proved by making the factory
    //    product agree bit-for-bit with a hand-built IntegrationBase.
    // ======================================================================
    {
        std::vector<std::vector<Real> > w1(3, std::vector<Real>(1, 0.0));
        w1[0][0] = 0.5;
        w1[1][0] = 0.4;
        w1[2][0] = 0.3;

        ProbeLM<GaussianCopulaPolicy> lmq(w1, LatentModelIntegrationType::GaussianQuadrature);
        ProbeLM<GaussianCopulaPolicy> lmt(w1, LatentModelIntegrationType::Trapezoid);

        auto dens = [](const std::vector<Real>& x) {
            return std::exp(-0.5 * x[0] * x[0]) / std::sqrt(2.0 * M_PI);
        };
        Real fq = lmq.integration()->integrate(dens);
        Real ft = lmt.integration()->integrate(dens);
        emit("F_factory_quad_gaussdens", fq);
        emit("F_factory_trap_gaussdens", ft);

        IntegrationBase<GaussianQuadMultidimIntegrator> handQuad(1, 25);
        std::vector<ext::shared_ptr<Integrator> > ints;
        ints.push_back(ext::make_shared<TrapezoidIntegral<Default> >(1.e-4, 20));
        IntegrationBase<MultidimIntegral> handTrap(ints, -35., 35.);
        emit_bool("F_factory_quad_is_order25", fq == handQuad.integrate(dens));
        emit_bool("F_factory_trap_is_1e4_20_m35_35", ft == handTrap.integrate(dens));

        // the quadrature branch is the documented default argument
        ProbeLM<GaussianCopulaPolicy> lmdflt(w1, LatentModelIntegrationType::GaussianQuadrature);
        emit_bool("F_factory_default_is_quadrature",
                  lmdflt.integration()->integrate(dens) == fq);

        // enum values (a port that renumbers them breaks the switch)
        emit_int("F_enum_GaussianQuadrature",
                 static_cast<long long>(LatentModelIntegrationType::GaussianQuadrature));
        emit_int("F_enum_Trapezoid",
                 static_cast<long long>(LatentModelIntegrationType::Trapezoid));

        // the default: case
        bool badThrew = false;
        std::string badMsg;
        try {
            ProbeLM<GaussianCopulaPolicy> bad(
                w1,
                static_cast<LatentModelIntegrationType::LatentModelIntegrationType>(99));
            (void)bad.integration();
        } catch (const std::exception& e) {
            badThrew = true;
            badMsg = e.what();
        }
        emit_bool("F_factory_unknown_type_throws", badThrew);
        emit_bool("F_factory_unknown_type_msg",
                  badMsg.find("Unknown latent model integration type") != std::string::npos);

        // dimension 2 factory product
        std::vector<std::vector<Real> > w2(3, std::vector<Real>(2, 0.0));
        w2[0][0] = 0.5; w2[0][1] = 0.3;
        w2[1][0] = 0.4; w2[1][1] = 0.2;
        w2[2][0] = 0.3; w2[2][1] = 0.1;
        ProbeLM<GaussianCopulaPolicy> lmq2(w2, LatentModelIntegrationType::GaussianQuadrature);
        emit("F_factory_quad_dim2_gaussdens",
             lmq2.integration()->integrate([](const std::vector<Real>& x) {
                 return std::exp(-0.5 * (x[0] * x[0] + x[1] * x[1])) / (2.0 * M_PI);
             }));
    }

    // ======================================================================
    // G. LatentModel::integratedExpectedValue / ...V, end to end, both
    //    integration types x both copula policies.
    // ======================================================================
    {
        std::vector<std::vector<Real> > w1(3, std::vector<Real>(1, 0.0));
        w1[0][0] = 0.5;
        w1[1][0] = 0.4;
        w1[2][0] = 0.3;

        ProbeLM<GaussianCopulaPolicy> gq(w1, LatentModelIntegrationType::GaussianQuadrature);
        ProbeLM<GaussianCopulaPolicy> gt(w1, LatentModelIntegrationType::Trapezoid);

        emit("G_gauss_quad_one", gq.integratedExpectedValue(f_one));
        emit("G_gauss_quad_lin", gq.integratedExpectedValue(f_lin));
        emit("G_gauss_quad_sqr", gq.integratedExpectedValue(f_sqr));
        emit("G_gauss_quad_gauss", gq.integratedExpectedValue(f_gauss));
        emit("G_gauss_quad_cos", gq.integratedExpectedValue(f_cos));

        emit("G_gauss_trap_one", gt.integratedExpectedValue(f_one));
        emit("G_gauss_trap_lin", gt.integratedExpectedValue(f_lin));
        emit("G_gauss_trap_sqr", gt.integratedExpectedValue(f_sqr));
        emit("G_gauss_trap_gauss", gt.integratedExpectedValue(f_gauss));
        emit("G_gauss_trap_cos", gt.integratedExpectedValue(f_cos));

        // analytic values, for the record (NOT what the port is checked
        // against — the port is checked against the numbers above).
        emit("G_analytic_gauss", std::sqrt(2.0 / 3.0));
        emit("G_analytic_cos", std::exp(-0.5));

        // vector version: only the quadrature facility implements integrateV,
        // and it carries the VectorIntegrator defect from block B.
        emit_arr("G_gauss_quad_V", gq.integratedExpectedValueV(
                                       [](const std::vector<Real>& x) -> std::vector<Real> {
                                           return {1.0, x[0], x[0] * x[0], std::cos(x[0])};
                                       }));
        bool threwV = false;
        try {
            (void)gt.integratedExpectedValueV(
                [](const std::vector<Real>&) -> std::vector<Real> { return {1.0}; });
        } catch (const std::exception&) {
            threwV = true;
        }
        emit_bool("G_gauss_trap_V_throws", threwV);

        // two systemic factors
        std::vector<std::vector<Real> > w2(3, std::vector<Real>(2, 0.0));
        w2[0][0] = 0.5; w2[0][1] = 0.3;
        w2[1][0] = 0.4; w2[1][1] = 0.2;
        w2[2][0] = 0.3; w2[2][1] = 0.1;
        ProbeLM<GaussianCopulaPolicy> gq2(w2, LatentModelIntegrationType::GaussianQuadrature);
        emit("G_gauss2_quad_one", gq2.integratedExpectedValue(f_one));
        emit("G_gauss2_quad_prod", gq2.integratedExpectedValue(f_prod2));
        emit("G_gauss2_quad_sqr", gq2.integratedExpectedValue(f_sqr2));

        // Student-t copula, one systemic factor. tOrders has numFactors+1
        // entries (systemic factors + the shared idiosyncratic one).
        std::vector<std::vector<Real> > wt(2, std::vector<Real>(1, 0.0));
        wt[0][0] = 0.5;
        wt[1][0] = 0.4;
        TCopulaPolicy::initTraits ini;
        ini.tOrders.push_back(3);
        ini.tOrders.push_back(5);
        ProbeLM<TCopulaPolicy> tq(wt, LatentModelIntegrationType::GaussianQuadrature, ini);
        ProbeLM<TCopulaPolicy> tt(wt, LatentModelIntegrationType::Trapezoid, ini);
        emit("G_t_quad_one", tq.integratedExpectedValue(f_one));
        emit("G_t_quad_sqr", tq.integratedExpectedValue(f_sqr));
        emit("G_t_trap_one", tt.integratedExpectedValue(f_one));
        emit("G_t_trap_sqr", tt.integratedExpectedValue(f_sqr));
        emit("G_t_trap_gauss", tt.integratedExpectedValue(f_gauss));
        // the T density itself, at a few points (cross-checks the scipy
        // delegation the Python TCopulaPolicy makes)
        {
            std::vector<Real> pts, dens;
            for (Real x = -3.0; x <= 3.0001; x += 1.5) {
                std::vector<Real> m(1, x);
                pts.push_back(x);
                dens.push_back(tq.copula().density(m));
            }
            emit_arr("G_t_density_pts", pts);
            emit_arr("G_t_density_vals", dens);
            emit_arr("G_t_varianceFactors", tq.copula().varianceFactors());
        }
    }

    // ======================================================================
    // H. FactorSampler — the generic template (latentmodel.hpp:407) and both
    //    partial specialisations (:742 Box-Muller, :768 polar Student-t).
    // ======================================================================
    {
        std::vector<std::vector<Real> > w1(3, std::vector<Real>(1, 0.0));
        w1[0][0] = 0.5;
        w1[1][0] = 0.4;
        w1[2][0] = 0.3;
        GaussianCopulaPolicy gcp(w1);
        emit_int("H_gauss_numFactors", static_cast<long long>(gcp.numFactors()));

        // H1: generic — RandomSequenceGenerator<MersenneTwisterUniformRng>
        //     fed through allFactorCumulInverter.
        {
            typedef LatentModel<GaussianCopulaPolicy>::FactorSampler<
                RandomSequenceGenerator<MersenneTwisterUniformRng> >
                Generic;
            Generic fs(gcp, 42UL);
            std::vector<Real> flat;
            for (Size k = 0; k < 5; ++k) {
                const Generic::sample_type& s = fs.nextSequence();
                for (Real v : s.value) flat.push_back(v);
                if (k == 0) emit("H1_generic_weight0", s.weight);
            }
            emit_arr("H1_generic_seed42_5x4", flat);
        }

        // H2: specialisation for RandomSequenceGenerator<BoxMullerGaussianRng>
        {
            typedef LatentModel<GaussianCopulaPolicy>::FactorSampler<
                RandomSequenceGenerator<BoxMullerGaussianRng<MersenneTwisterUniformRng> > >
                BoxM;
            BoxM fs(gcp, 42UL);
            std::vector<Real> flat;
            for (Size k = 0; k < 5; ++k) {
                const BoxM::sample_type& s = fs.nextSequence();
                for (Real v : s.value) flat.push_back(v);
                if (k == 0) emit("H2_boxmuller_weight0", s.weight);
            }
            emit_arr("H2_boxmuller_seed42_5x4", flat);
        }

        // H3: specialisation for RandomSequenceGenerator<PolarStudentTRng>.
        //     Every per-factor generator gets a COPY of the same just-seeded
        //     urng_, so all factors share one uniform stream.
        {
            std::vector<std::vector<Real> > wt(2, std::vector<Real>(1, 0.0));
            wt[0][0] = 0.5;
            wt[1][0] = 0.4;
            TCopulaPolicy::initTraits ini;
            ini.tOrders.push_back(3);
            ini.tOrders.push_back(5);
            TCopulaPolicy tcp(wt, ini);
            emit_int("H3_t_numFactors", static_cast<long long>(tcp.numFactors()));
            emit_arr("H3_t_varianceFactors", tcp.varianceFactors());

            typedef LatentModel<TCopulaPolicy>::FactorSampler<
                RandomSequenceGenerator<PolarStudentTRng<MersenneTwisterUniformRng> > >
                PolarT;
            PolarT fs(tcp, 42UL);
            std::vector<Real> flat;
            for (Size k = 0; k < 5; ++k) {
                const PolarT::sample_type& s = fs.nextSequence();
                for (Real v : s.value) flat.push_back(v);
                if (k == 0) emit("H3_polart_weight0", s.weight);
            }
            emit_arr("H3_polart_seed42_5x3", flat);

            // The shared-stream defect, isolated: two standalone
            // PolarStudentTRng built from copies of one seeded MT with the
            // degrees of freedom the sampler derives, 2/(1-vf^2) == nu.
            {
                MersenneTwisterUniformRng u(42UL);
                const std::vector<Real>& vf = tcp.varianceFactors();
                std::vector<Real> dfs;
                for (Real v : vf) dfs.push_back(2. / (1. - v * v));
                emit_arr("H3_derived_dfs", dfs);
                PolarStudentTRng<MersenneTwisterUniformRng> a(dfs[0], u);
                PolarStudentTRng<MersenneTwisterUniformRng> b(dfs[1], u);
                std::vector<Real> as, bs;
                for (Size k = 0; k < 4; ++k) {
                    as.push_back(a.next().value);
                    bs.push_back(b.next().value);
                }
                emit_arr("H3_standalone_df3", as);
                emit_arr("H3_standalone_df5", bs);
            }
            // Same two standalone generators, run long enough to pin the
            // consequence of the shared stream (the k-th draws of the two
            // never disagree about their sign, because both read the same
            // (v,u) pair and the polar rejection test r^2 >= 1 does not
            // depend on nu) AND long enough to decompose what
            // nextSequence() does: trng_[0] advances once per row while
            // trng_.back() advances twice, so the sign agreement does NOT
            // survive into the sample vector.
            {
                MersenneTwisterUniformRng u(42UL);
                const std::vector<Real>& vf = tcp.varianceFactors();
                std::vector<Real> dfs;
                for (Real v : vf) dfs.push_back(2. / (1. - v * v));
                PolarStudentTRng<MersenneTwisterUniformRng> a(dfs[0], u);
                PolarStudentTRng<MersenneTwisterUniformRng> b(dfs[1], u);
                std::vector<Real> as, bs;
                for (Size k = 0; k < 32; ++k) {
                    as.push_back(a.next().value);
                    bs.push_back(b.next().value);
                }
                emit_arr("H3_standalone_df3_32", as);
                emit_arr("H3_standalone_df5_32", bs);
            }

            // H4: generic sampler over the T copula, for contrast — this one
            //     DOES normalise (it goes through allFactorCumulInverter).
            typedef LatentModel<TCopulaPolicy>::FactorSampler<
                RandomSequenceGenerator<MersenneTwisterUniformRng> >
                GenericT;
            GenericT gfs(tcp, 42UL);
            std::vector<Real> gflat;
            for (Size k = 0; k < 3; ++k) {
                const GenericT::sample_type& s = gfs.nextSequence();
                for (Real v : s.value) gflat.push_back(v);
            }
            emit_arr("H4_generic_t_seed42_3x3", gflat);
        }
    }

    // ======================================================================
    // I. Ziggurat RNG traits (zigguratrng.hpp:64).
    // ======================================================================
    {
        emit_int("I_ziggurat_allowsErrorEstimate",
                 static_cast<long long>(Ziggurat::allowsErrorEstimate));
        Ziggurat::rsg_type rsg = Ziggurat::make_sequence_generator(4, 42UL);
        emit_int("I_ziggurat_rsg_dimension", static_cast<long long>(rsg.dimension()));
        std::vector<Real> flat;
        for (Size k = 0; k < 5; ++k) {
            const Ziggurat::rsg_type::sample_type& s = rsg.nextSequence();
            for (Real v : s.value) flat.push_back(v);
            if (k == 0) emit("I_ziggurat_weight0", s.weight);
        }
        emit_arr("I_ziggurat_seed42_5x4", flat);

        // the traits factory is exactly RandomSequenceGenerator<ZigguratRng>
        RandomSequenceGenerator<ZigguratRng> hand(4, 42UL);
        std::vector<Real> handFlat;
        for (Size k = 0; k < 5; ++k) {
            const RandomSequenceGenerator<ZigguratRng>::sample_type& s = hand.nextSequence();
            for (Real v : s.value) handFlat.push_back(v);
        }
        emit_bool("I_ziggurat_factory_is_rsg_of_zigguratrng", handFlat == flat);

        // scalar ZigguratRng stream, for the seam
        ZigguratRng z(42UL);
        std::vector<Real> zs;
        for (Size k = 0; k < 8; ++k) zs.push_back(z.next().value);
        emit_arr("I_zigguratrng_seed42_8", zs);
    }

    emit_str("probe", "v143_experimental_latent");
    std::cout << "\n}\n";
    return 0;
}
