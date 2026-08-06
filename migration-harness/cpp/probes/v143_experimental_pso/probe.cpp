// migration-harness/cpp/probes/v143_experimental_pso/probe.cpp
//
// Reference values for the ql/experimental/math swarm optimizers @ v1.43
// (6b57206e): the ParticleSwarmOptimization Inertia/Topology strategy family
// and the FireflyAlgorithm RandomWalk family.
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// Three of the four gap classes here (ClubsTopology, LevyFlightInertia,
// DecreasingGaussianWalk) do NOT run off QuantLib's MersenneTwisterUniformRng.
// They run off <random>: std::mt19937 driving std::uniform_int_distribution,
// std::uniform_real_distribution and std::normal_distribution. Those
// distributions are implementation-defined, so "port the algorithm" is not
// enough — the *stream* has to be pinned too, or a Python port can only be
// checked distributionally, which is not cross-validation.
//
// So block A pins the <random> primitives themselves against this toolchain's
// libc++, in isolation, before any QuantLib class uses them:
//
//   A1  std::mt19937 raw outputs vs QuantLib MersenneTwisterUniformRng
//       nextInt32() for the same seed. (Claim under test: they are the same
//       engine. If A1 ever disagrees, everything below it is void.)
//   A2  std::generate_canonical<double,53> / uniform_real_distribution
//   A3  std::uniform_int_distribution<Size>  (the __independent_bits_engine path)
//   A4  std::normal_distribution<double>     (libc++ Marsaglia polar, 2-at-a-time)
//
// Blocks B..F then pin the QuantLib classes that consume them:
//
//   B  LevyFlightDistribution variates off std::mt19937
//   C  IsotropicRandomWalk::nextReal (dim 1..4, weighted and unweighted)
//   D  ParticleSwarmOptimization x {Trivial, SimpleRandom, Decreasing,
//      Adaptive, LevyFlight} inertia x {Global, KNeighbors, Clubs} topology.
//      Pinned as the EVALUATION TRAJECTORY: a cost function that records
//      every x it is handed. Two implementations that return the same optimum
//      by a different route disagree on the trajectory, which is the point.
//   E  FireflyAlgorithm x {Gaussian, LevyFlight, DecreasingGaussian} walk,
//      same trajectory treatment.
//   G  the ClubsTopology out-of-range defect (arithmetic only)
//   H  the DecreasingInertia uninitialised-member defect (arithmetic only)
//   F  AdaptiveInertia's c0 schedule, isolated: a hand-rolled replica of
//      AdaptiveInertia::setValues driven by a scripted pBF_ sequence, so the
//      inertia state machine is pinned independently of the swarm dynamics.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/pso.json.

#include <ql/experimental/math/fireflyalgorithm.hpp>
#include <ql/experimental/math/isotropicrandomwalk.hpp>
#include <ql/experimental/math/levyflightdistribution.hpp>
#include <ql/experimental/math/particleswarmoptimization.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/costfunction.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
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
// Test cost functions. Both record every point they are handed, which is what
// turns "same answer" into "same path".
//
// FP_CONTRACT is switched OFF inside both. At -O3 Clang fuses `s += d*d` into
// an FMA, and the fused sum differs from the separately-rounded one in the
// last bit — which would show up as a port bug in the pinned trajectory when
// it is really an artefact of how the *probe's own* objective was compiled.
// The objective is scaffolding, not the thing under test, so it is pinned to
// plain IEEE double arithmetic that Python reproduces exactly.
// --------------------------------------------------------------------------

#pragma clang fp contract(off)

class RecordingSphere : public CostFunction {
  public:
    explicit RecordingSphere(Size maxRecord) : maxRecord_(maxRecord) {}
    Array values(const Array& x) const override { return x; }
    Real value(const Array& x) const override {
        Real s = 0.0;
        for (Real xi : x) s += xi * xi;
        if (log_.size() < maxRecord_) {
            for (Real xi : x) log_.push_back(xi);
            log_.push_back(s);
        }
        ++calls_;
        return s;
    }
    const std::vector<Real>& log() const { return log_; }
    Size calls() const { return calls_; }

  private:
    Size maxRecord_;
    mutable std::vector<Real> log_;
    mutable Size calls_ = 0;
};

// f(x) = sum_i (x_i - shift_i)^4 + (x_i - shift_i)^2 : smooth, single basin,
// deliberately not symmetric about the box centre so a sign error shows up.
class RecordingQuartic : public CostFunction {
  public:
    RecordingQuartic(Array shift, Size maxRecord)
    : shift_(std::move(shift)), maxRecord_(maxRecord) {}
    Array values(const Array& x) const override { return x; }
    Real value(const Array& x) const override {
        Real s = 0.0;
        for (Size i = 0; i < x.size(); ++i) {
            Real d = x[i] - shift_[i];
            s += d * d * d * d + d * d;
        }
        if (log_.size() < maxRecord_) {
            for (Real xi : x) log_.push_back(xi);
            log_.push_back(s);
        }
        ++calls_;
        return s;
    }
    const std::vector<Real>& log() const { return log_; }
    Size calls() const { return calls_; }

  private:
    Array shift_;
    Size maxRecord_;
    mutable std::vector<Real> log_;
    mutable Size calls_ = 0;
};

// --------------------------------------------------------------------------
// Block A — the <random> primitives, pinned in isolation.
// --------------------------------------------------------------------------

void blockA() {
    // A1: std::mt19937 vs QuantLib MersenneTwisterUniformRng, same seed.
    for (unsigned long seed : {1UL, 42UL, 12345UL}) {
        std::mt19937 g(static_cast<std::mt19937::result_type>(seed));
        MersenneTwisterUniformRng ql(seed);
        std::vector<long long> stdOut, qlOut;
        for (int i = 0; i < 8; ++i) {
            stdOut.push_back(static_cast<long long>(g()));
            qlOut.push_back(static_cast<long long>(ql.nextInt32()));
        }
        emit_iarr("A1_std_mt19937_seed" + std::to_string(seed), stdOut);
        emit_iarr("A1_ql_mt_nextInt32_seed" + std::to_string(seed), qlOut);
        emit_bool("A1_same_engine_seed" + std::to_string(seed), stdOut == qlOut);
    }

    // A2: generate_canonical<double,53> and uniform_real_distribution.
    {
        std::mt19937 g(42);
        std::vector<Real> gc;
        for (int i = 0; i < 8; ++i)
            gc.push_back(std::generate_canonical<double, 53>(g));
        emit_arr("A2_generate_canonical_53_seed42", gc);
    }
    {
        std::mt19937 g(42);
        std::uniform_real_distribution<Real> u01(0.0, 1.0);
        std::vector<Real> v;
        for (int i = 0; i < 8; ++i) v.push_back(u01(g));
        emit_arr("A2_uniform_real_0_1_seed42", v);
    }
    {
        std::mt19937 g(7);
        std::uniform_real_distribution<Real> um11(-1.0, 1.0);
        std::vector<Real> v;
        for (int i = 0; i < 8; ++i) v.push_back(um11(g));
        emit_arr("A2_uniform_real_m1_1_seed7", v);
    }

    // A3: uniform_int_distribution<Size> — the ranges ClubsTopology uses
    // (distribution_(1, totalClubs_) and param_type(1, k) reparametrisations).
    {
        const std::pair<Size, Size> ranges[] = {
            {1, 8}, {1, 5}, {1, 4}, {1, 3}, {1, 2}, {0, 9}, {1, 1}, {2, 17}, {0, 1000000}
        };
        for (auto r : ranges) {
            std::mt19937 g(1234);
            std::uniform_int_distribution<Size> d(r.first, r.second);
            std::vector<long long> v;
            for (int i = 0; i < 12; ++i) v.push_back(static_cast<long long>(d(g)));
            emit_iarr("A3_uniform_int_" + std::to_string(r.first) + "_" +
                          std::to_string(r.second) + "_seed1234",
                      v);
        }
        // param_type reparametrisation on a shared engine, exactly as
        // ClubsTopology::leaveRandomClub / joinRandomClub do it.
        std::mt19937 g(99);
        std::uniform_int_distribution<Size> d(1, 8);
        using pt = std::uniform_int_distribution<Size>::param_type;
        std::vector<long long> v;
        for (int i = 0; i < 12; ++i) v.push_back(static_cast<long long>(d(g, pt(1, 1 + (i % 5)))));
        emit_iarr("A3_uniform_int_paramtype_seed99", v);
    }

    // A4: normal_distribution<double> — note libc++ generates in PAIRS, so an
    // odd number of draws leaves a cached value; the sequence below exposes it.
    for (Real sigma : {1.0, 0.25}) {
        std::mt19937 g(5);
        std::normal_distribution<Real> nd(0.0, sigma);
        std::vector<Real> v;
        for (int i = 0; i < 10; ++i) v.push_back(nd(g));
        std::ostringstream key;
        key << "A4_normal_sigma" << (sigma == 1.0 ? "1" : "0p25") << "_seed5";
        emit_arr(key.str(), v);
    }
}

// --------------------------------------------------------------------------
// Block B — LevyFlightDistribution off std::mt19937.
// --------------------------------------------------------------------------

void blockB() {
    struct Case { Real xm, alpha; unsigned seed; const char* tag; };
    const Case cases[] = {
        {1.0, 1.0, 42, "xm1_a1"},
        {1.0, 1.5, 42, "xm1_a1p5"},
        {0.5, 0.7, 11, "xm0p5_a0p7"},
    };
    for (const auto& c : cases) {
        std::mt19937 g(c.seed);
        LevyFlightDistribution d(c.xm, c.alpha);
        std::vector<Real> v;
        for (int i = 0; i < 8; ++i) v.push_back(d(g));
        emit_arr(std::string("B_levy_variates_") + c.tag, v);
        // pdf is deterministic and already ported; pin a few points anyway so
        // the two halves of the class travel together.
        std::vector<Real> pdf;
        for (Real x : {0.1, 0.5, 1.0, 2.0, 10.0}) pdf.push_back(d(x));
        emit_arr(std::string("B_levy_pdf_") + c.tag, pdf);
    }
}

// --------------------------------------------------------------------------
// Block C — IsotropicRandomWalk over a std::mt19937-driven Levy radius.
// --------------------------------------------------------------------------

void blockC() {
    for (Size dim : {1u, 2u, 3u, 4u}) {
        std::mt19937 g(2024);
        IsotropicRandomWalk<LevyFlightDistribution, std::mt19937> walk(
            g, LevyFlightDistribution(1.0, 1.3), dim, Array(dim, 1.0), 7777);
        std::vector<Real> out;
        Array step(dim, 0.0);
        for (int k = 0; k < 5; ++k) {
            walk.nextReal<Real*>(&step[0]);
            for (Size j = 0; j < dim; ++j) out.push_back(step[j]);
        }
        emit_arr("C_isowalk_dim" + std::to_string(dim) + "_seed2024_ang7777", out);
    }
    // Weighted (box-aspect) variant, as the firefly/PSO consumers configure it.
    {
        Size dim = 3;
        std::mt19937 g(2024);
        IsotropicRandomWalk<LevyFlightDistribution, std::mt19937> walk(
            g, LevyFlightDistribution(1.0, 1.3), 1, Array(1, 1.0), 7777);
        Array lo(dim), hi(dim);
        lo[0] = -1.0; hi[0] = 3.0;
        lo[1] = 0.0;  hi[1] = 1.0;
        lo[2] = -5.0; hi[2] = 5.0;
        walk.setDimension(dim, lo, hi);
        std::vector<Real> out;
        Array step(dim, 0.0);
        for (int k = 0; k < 5; ++k) {
            walk.nextReal<Real*>(&step[0]);
            for (Size j = 0; j < dim; ++j) out.push_back(step[j]);
        }
        emit_arr("C_isowalk_boxweighted_dim3", out);
    }
}

// --------------------------------------------------------------------------
// Block D — ParticleSwarmOptimization: inertia x topology, trajectory pinned.
// --------------------------------------------------------------------------

// The PSO-In (explicit omega) constructor is used throughout, NOT the
// phi-derived PSO-Co one. The PSO-Co constructor computes
//     c0_ = 2 / |2 - phi - sqrt(phi*phi - 4*phi)|
// and on this ARM64 -O3 build Clang contracts `phi*phi - 4*phi` into an FMA,
// which moves c0_ by 3 ULP relative to the same expression evaluated with
// separate rounding. That is a property of how libQuantLib happened to be
// compiled, not of the Inertia/Topology classes these cases exist to pin, and
// it would make the whole trajectory architecture-dependent. Passing omega as
// a literal removes it. Block F pins the constriction factor separately, both
// ways, so nothing is swept under the rug.
const Real kOmega = 0.72984378812835763;
const Real kC1 = 1.4961797656631331;
const Real kC2 = 1.4961797656631331;

void runPso(const std::string& tag,
            const ext::shared_ptr<ParticleSwarmOptimization::Topology>& topo,
            const ext::shared_ptr<ParticleSwarmOptimization::Inertia>& inertia,
            Size M,
            Size maxIter) {
    Array shift(2);
    shift[0] = 0.7;
    shift[1] = -1.3;
    RecordingQuartic f(shift, 120);
    BoundaryConstraint c(-4.0, 4.0);
    Array x0(2, 0.0);
    Problem P(f, c, x0);
    EndCriteria ec(maxIter, maxIter - 1, 1e-12, 1e-12, 1e-12);

    ParticleSwarmOptimization pso(M, topo, inertia, kOmega, kC1, kC2, 31415UL);
    EndCriteria::Type t = pso.minimize(P, ec);

    emit_arr("D_" + tag + "_trajectory", f.log());
    emit_int("D_" + tag + "_calls", static_cast<long long>(f.calls()));
    std::vector<Real> xs;
    for (Real v : P.currentValue()) xs.push_back(v);
    emit_arr("D_" + tag + "_x", xs);
    emit("D_" + tag + "_f", P.functionValue());
    emit_int("D_" + tag + "_ecType", static_cast<long long>(t));
}

void blockD() {
    // Deterministic inertias against the deterministic topologies: these are
    // pure QuantLib-MT / Sobol paths and must match bit-for-bit.
    runPso("global_trivial",
           ext::make_shared<GlobalTopology>(),
           ext::make_shared<TrivialInertia>(), 8, 12);
    runPso("global_adaptive",
           ext::make_shared<GlobalTopology>(),
           ext::make_shared<AdaptiveInertia>(0.2, 0.9, 5, 2), 8, 12);
    runPso("kneighbors_adaptive",
           ext::make_shared<KNeighbors>(2),
           ext::make_shared<AdaptiveInertia>(0.1, 1.2, 3, 1), 8, 12);
    runPso("global_simplerandom",
           ext::make_shared<GlobalTopology>(),
           ext::make_shared<SimpleRandomInertia>(0.4, 555UL), 8, 12);
    // DecreasingInertia is deliberately NOT pinned here — see blockH: its
    // setValues() loop bound is an uninitialised member, so its behaviour is
    // indeterminate and nothing about it can be a reference value.

    // ClubsTopology, totalClubs == defaultClubs ("everyone in every club").
    // This is the ONLY configuration v1.43 can actually run — see blockG.
    runPso("clubsfull_trivial",
           ext::make_shared<ClubsTopology>(4, 4, 4, 2, 3, 909UL),
           ext::make_shared<TrivialInertia>(), 6, 10);
    runPso("clubsfull_adaptive",
           ext::make_shared<ClubsTopology>(3, 3, 3, 1, 2, 909UL),
           ext::make_shared<AdaptiveInertia>(0.2, 0.9, 5, 2), 8, 12);

    // LevyFlightInertia: std::mt19937 + IsotropicRandomWalk inside.
    runPso("global_levyflight",
           ext::make_shared<GlobalTopology>(),
           ext::make_shared<LevyFlightInertia>(1.5, 2, 4242UL), 8, 12);
    runPso("clubsfull_levyflight",
           ext::make_shared<ClubsTopology>(5, 5, 5, 1, 3, 909UL),
           ext::make_shared<LevyFlightInertia>(1.2, 1, 4242UL), 8, 12);
}

// --------------------------------------------------------------------------
// Block E — FireflyAlgorithm random walks.
// --------------------------------------------------------------------------

void runFa(const std::string& tag,
           const ext::shared_ptr<FireflyAlgorithm::Intensity>& intensity,
           const ext::shared_ptr<FireflyAlgorithm::RandomWalk>& walk,
           Size M,
           Size Mde,
           Size maxIter) {
    RecordingSphere f(120);
    BoundaryConstraint c(-3.0, 3.0);
    Array x0(2, 0.0);
    Problem P(f, c, x0);
    EndCriteria ec(maxIter, maxIter - 1, 1e-12, 1e-12, 1e-12);

    FireflyAlgorithm fa(M, intensity, walk, Mde, 1.0, 0.5, 271828UL);
    EndCriteria::Type t = fa.minimize(P, ec);

    emit_arr("E_" + tag + "_trajectory", f.log());
    emit_int("E_" + tag + "_calls", static_cast<long long>(f.calls()));
    std::vector<Real> xs;
    for (Real v : P.currentValue()) xs.push_back(v);
    emit_arr("E_" + tag + "_x", xs);
    emit("E_" + tag + "_f", P.functionValue());
    emit_int("E_" + tag + "_ecType", static_cast<long long>(t));
}

void blockE() {
    runFa("exp_gaussianwalk",
          ext::make_shared<ExponentialIntensity>(1.0, 1e-8, 1.0),
          ext::make_shared<GaussianWalk>(0.3, 0.9, 8080UL), 8, 0, 8);
    runFa("exp_levywalk",
          ext::make_shared<ExponentialIntensity>(1.0, 1e-8, 1.0),
          ext::make_shared<LevyFlightWalk>(1.5, 0.5, 0.9, 8080UL), 8, 0, 8);
    runFa("exp_decreasinggaussian",
          ext::make_shared<ExponentialIntensity>(1.0, 1e-8, 1.0),
          ext::make_shared<DecreasingGaussianWalk>(0.3, 0.9, 8080UL), 8, 0, 8);
    runFa("invsq_decreasinggaussian",
          ext::make_shared<InverseLawSquareIntensity>(1.0, 1e-8),
          ext::make_shared<DecreasingGaussianWalk>(0.5, 0.8, 8080UL), 8, 0, 8);
    // With a DE subpopulation, so the delta^2 schedule spans several
    // iterations with Mfa < M.
    runFa("exp_decreasinggaussian_de",
          ext::make_shared<ExponentialIntensity>(1.0, 1e-8, 1.0),
          ext::make_shared<DecreasingGaussianWalk>(0.3, 0.9, 8080UL), 8, 3, 8);
    // Pure DE (Mde == M, so Mfa == 0 and isFA is false). This is the branch
    // where `Array& xBest` is assigned through, copying the drawn array into
    // x_[values_[0].second] instead of rebinding — pinned so the port cannot
    // "clean up" the aliasing.
    runFa("pure_de",
          ext::make_shared<ExponentialIntensity>(1.0, 1e-8, 1.0),
          ext::make_shared<GaussianWalk>(0.3, 0.9, 8080UL), 6, 6, 8);
}

// --------------------------------------------------------------------------
// Block F — AdaptiveInertia's c0 state machine, isolated from swarm dynamics.
//
// AdaptiveInertia::setValues is 25 lines of pure state machine over the
// personal-best array. Replicating it here against a scripted pBF_ sequence
// pins the schedule itself (the counter, the doubling/halving, the clamp, and
// the "first iteration leaves inertia unchanged" special case) without the
// swarm in the way.
// --------------------------------------------------------------------------

void blockF() {
    struct Rep {
        Real c0, best, minI, maxI;
        Size sh, sl, counter;
        bool started = false;
        Real step(const std::vector<Real>& pbf) {
            Real currBest = pbf[0];
            for (Size i = 1; i < pbf.size(); ++i)
                if (currBest > pbf[i]) currBest = pbf[i];
            if (started) {
                if (currBest < best) { best = currBest; counter--; }
                else { counter++; }
                if (counter > sh) c0 = std::max(minI, std::min(maxI, c0 * 0.5));
                else if (counter < sl) c0 = std::max(minI, std::min(maxI, c0 * 2.0));
            } else {
                best = currBest;
                started = true;
            }
            return c0;
        }
    };

    // A scripted personal-best history: improve, improve, stall x6, improve.
    const std::vector<std::vector<Real>> history = {
        {5.0, 6.0, 7.0}, {4.0, 6.0, 7.0}, {3.0, 6.0, 7.0}, {3.0, 6.0, 7.0},
        {3.0, 6.0, 7.0}, {3.0, 6.0, 7.0}, {3.0, 6.0, 7.0}, {3.0, 6.0, 7.0},
        {3.0, 6.0, 7.0}, {2.5, 6.0, 7.0}, {2.5, 6.0, 7.0}, {1.0, 6.0, 7.0},
    };

    {
        Rep r{0.7298437881283576, QL_MAX_REAL, 0.2, 0.9, 5, 2, 0, false};
        std::vector<Real> c0s;
        for (const auto& h : history) c0s.push_back(r.step(h));
        emit_arr("F_adaptive_c0_schedule_sh5_sl2", c0s);
    }
    {
        Rep r{0.5, QL_MAX_REAL, 0.1, 1.2, 3, 1, 0, false};
        std::vector<Real> c0s;
        for (const auto& h : history) c0s.push_back(r.step(h));
        emit_arr("F_adaptive_c0_schedule_sh3_sl1", c0s);
    }
    // The PSO constriction factor for the default c1 = c2 = 2.05, emitted BOTH
    // ways. This block is compiled with contraction off (the file-scope pragma
    // above), so the first value is the expression as written in the source.
    // The second forces the fusion that libQuantLib's own -O3 ARM64 build
    // applies to `phi*phi - 4*phi` inside the PSO-Co constructor. The two
    // differ by 3 ULP, and the fused one is what a PSO built via that
    // constructor actually uses on this machine.
    {
        Real phi = 2.05 + 2.05;
        emit("F_pso_constriction_c0_unfused",
             2.0 / std::abs(2.0 - phi - std::sqrt(phi * phi - 4 * phi)));
        emit("F_pso_constriction_c0_fused",
             2.0 / std::abs(2.0 - phi - std::sqrt(std::fma(phi, phi, -(4 * phi)))));
        emit("F_pso_probe_omega", kOmega);
        emit("F_pso_probe_c1", kC1);
    }
}

// --------------------------------------------------------------------------
// Block H — DecreasingInertia reads an uninitialised member (v1.43 defect).
//
//   class DecreasingInertia : public ParticleSwarmOptimization::Inertia {
//     DecreasingInertia(Real threshold = 0.5) : threshold_(threshold) {}
//     void setSize(Size M, Size N, Real c0, const EndCriteria& e) override {
//         N_ = N; c0_ = c0; iteration_ = 0; maxIterations_ = e.maxIterations();
//     }                                   // <-- M_ is never assigned
//     void setValues() override {
//         Real c0 = c0_*(threshold_ + (1.0 - threshold_)*
//                        (maxIterations_ - iteration_) / maxIterations_);
//         for (Size i = 0; i < M_; i++) (*V_)[i] *= c0;   // <-- M_ indeterminate
//     }
//     Size M_, N_, maxIterations_, iteration_;            // no initialisers
//   };
//
// The constructor initialises only threshold_, and setSize assigns N_, c0_,
// iteration_ and maxIterations_ but not M_. So the loop bound in setValues is
// whatever memory the object happened to be built on. On this build it comes
// out 0, i.e. DecreasingInertia silently applies NO inertia at all, which is
// how it was identified: modelling the Python port as a no-op setValues drops
// the trajectory mismatch against C++ from 67/120 to 10/120 (the remainder
// being ordinary FP contraction).
//
// Nothing about DecreasingInertia's *dynamics* can therefore be a reference
// value. What is emitted is the c0 schedule it would produce if M_ were set —
// pure arithmetic over threshold_, maxIterations_ and iteration_, which the
// port can be held to — plus the observed loop bound.
// --------------------------------------------------------------------------

void blockH() {
    const Size maxIterations = 12;
    for (Real threshold : {0.3, 0.5}) {
        std::vector<Real> sched;
        const Real c0 = kOmega;
        for (Size iteration = 0; iteration < 6; ++iteration) {
            sched.push_back(c0 * (threshold + (1.0 - threshold) *
                                                  (maxIterations - iteration) /
                                                  static_cast<Real>(maxIterations)));
        }
        emit_arr(std::string("H_decreasing_c0_schedule_thr") +
                     (threshold == 0.3 ? "0p3" : "0p5"),
                 sched);
    }
    // C++ never advances iteration_ inside setValues either, so in practice
    // only the iteration_ == 0 entry above is ever used.
    emit_int("H_decreasing_iteration_is_never_advanced", 1);
    emit_str("H_decreasing_defect",
             "v1.43 DecreasingInertia::setSize does not assign M_, so setValues loops to an "
             "uninitialised bound. Observed as 0 on this build (inertia never applied). "
             "Behaviour is indeterminate and is not pinned.");
}

// --------------------------------------------------------------------------
// Block G — the ClubsTopology out-of-range defect in v1.43, pinned as
// arithmetic rather than as behaviour (because the behaviour is a segfault).
//
// ClubsTopology's ctor initialises
//     distribution_(1, totalClubs_)
// i.e. std::uniform_int_distribution<Size> over the CLOSED range
// [1, totalClubs_]. setSize() then indexes containers of length totalClubs_
// with that draw:
//     clubSet[index] = true;              // clubSet.size() == totalClubs_
//     particles4clubs_[index][i] = true;  // outer size == totalClubs_
// so index 0 is unreachable and index == totalClubs_ is one past the end.
// With totalClubs_ == defaultClubs_ setSize takes the other branch and never
// draws, which is why every ClubsTopology case in block D is configured that
// way. Any defaultClubs_ < totalClubs_ configuration reads a garbage
// std::vector<bool> out of particles4clubs_ and segfaults; verified with
// ClubsTopology(3, 6, 5, 2, 4, 909).setSize(8) on this toolchain.
//
// What is emitted here is the draw sequence itself, which needs no UB to
// produce, and shows totalClubs_ being drawn.
// --------------------------------------------------------------------------

void blockG() {
    const Size totalClubs = 6;
    std::mt19937 g(909);
    std::uniform_int_distribution<Size> d(1, totalClubs);
    std::vector<long long> draws;
    long long maxSeen = 0, minSeen = 1 << 30;
    for (int i = 0; i < 24; ++i) {
        auto v = static_cast<long long>(d(g));
        draws.push_back(v);
        maxSeen = std::max(maxSeen, v);
        minSeen = std::min(minSeen, v);
    }
    emit_iarr("G_clubs_setSize_draws_total6_seed909", draws);
    emit_int("G_clubs_totalClubs", static_cast<long long>(totalClubs));
    emit_int("G_clubs_valid_index_max", static_cast<long long>(totalClubs) - 1);
    emit_int("G_clubs_draw_max", maxSeen);
    emit_int("G_clubs_draw_min", minSeen);
    emit_bool("G_clubs_draw_can_exceed_valid_index", maxSeen > static_cast<long long>(totalClubs) - 1);
    emit_bool("G_clubs_index_zero_unreachable", minSeen > 0);
    emit_str("G_clubs_defect",
             "v1.43 ClubsTopology::setSize indexes length-totalClubs_ containers with a "
             "uniform_int_distribution(1, totalClubs_) draw; index==totalClubs_ is out of "
             "range and segfaults. Only defaultClubs_==totalClubs_ is runnable.");
}

}  // namespace

int main() {
    std::cout << "{\n";
    blockA();
    blockB();
    blockC();
    blockD();
    blockE();
    blockF();
    blockG();
    blockH();
    std::cout << "\n}\n";
    return 0;
}
