// migration-harness/cpp/probes/v143_math_expint/probe.cpp
//
// Reference values for ql/math/integrals/exponentialintegrals.{hpp,cpp} in
// C++ QuantLib v1.43 — namespace QuantLib::ExponentialIntegral:
//
//     Real Si(Real x);                      Real Ci(Real x);
//     std::complex<Real> Si(const std::complex<Real>&);
//     std::complex<Real> Ci(const std::complex<Real>&);
//     std::complex<Real> E1(const std::complex<Real>&);
//     std::complex<Real> Ei(const std::complex<Real>&);
//
// The whole content of this translation unit is a pile of *branch selections*
// on the argument, and every one of them is a different algorithm — a port that
// only reproduces the "happy" branch is wrong everywhere else. So every guard
// clause below gets cases on BOTH sides of its threshold.
//
// Branches pinned, in source order (.cpp line numbers are v1.43):
//
//   exponential_integrals_helper::f / g (.cpp:39-73)
//       Rational approximations in 1/x^2 used only by the real x > 4 branches.
//       Not directly callable (they live in a named-but-internal namespace), so
//       they are pinned indirectly through Si/Ci above 4.
//
//   Si(Real x) (.cpp:77-98)
//       B1  x < 0            -> -Si(-x)                    [`si_real_neg_*`]
//       B2  0 <= x <= 4.0    -> Pade-type rational series   [`si_real_*`]
//       B3  x > 4.0          -> M_PI_2 - f(x)cos x - g(x)sin x
//       The `x <= 4.0` boundary is straddled with nextafter(4,0) / 4.0 /
//       nextafter(4,inf) so a port that writes `<` instead of `<=` fails.
//
//   Ci(Real x) (.cpp:100-121)
//       G1  QL_REQUIRE(x >= 0) -> throws for x < 0     [expected == "raises"]
//       B4  0 <= x <= 4.0    -> M_EULER_MASCHERONI + log x + rational
//                               (x == 0 is the log pole: -inf, emitted "-inf")
//       B5  x > 4.0          -> f(x)sin x - g(x)cos x
//
//   Ei(z, acc) (.cpp:123-197) — the engine under everything complex.
//       G2  z == 0                    -> -inf + 0i
//       G3  QL_REQUIRE(Re z < z_inf)  (z_inf = log(0.01*QL_MAX_REAL)+log(100)
//                                      ~ 709.78; not probed, it needs |Im z|
//                                      > 709 which no Si/Ci caller produces)
//       B6  |z| > 1.1*z_asym          -> asymptotic series in k!/z^k, stopped
//                                        by the `match` predicate at
//                                        5*QL_EPSILON relative. z_asym =
//                                        2 - 1.035*log(5*QL_EPSILON), so the
//                                        threshold is 1.1*z_asym ~ 41.4046 and
//                                        the r41p0 / r41p5 magnitudes straddle
//                                        it.
//       B7  |z| > 4.5 AND (Re z < 0 OR |Im z| > 4.5)
//                                     -> 47-level continued fraction. DIST=4.5
//                                        is straddled by r4p4 / r4p5 / r4p6
//                                        (the test is strict `>`, so |z| == 4.5
//                                        must still take B8).
//       B8  otherwise                 -> power series with the half-harmonic
//                                        accumulator nn, terminated by exact
//                                        FP equality `s + sn*nn == s`.
//       B9  the imag()==0 tail of B8 replaces the computed imaginary part with
//           acc.imag() — visible only on the real axis, hence the `a180`/`a000`
//           directions.
//       boost::math::sign(Im z) is (z==0) ? 0 : signbit ? -1 : 1, so the
//       +/- i*pi accumulator VANISHES on the real axis. Real-axis cases pin it.
//
//   E1(z) (.cpp:203-213)
//       B10 Im z < 0                     -> -Ei(-z, -i*pi)
//       B11 Im z > 0 OR Re z < 0         -> -Ei(-z, +i*pi)
//       B12 otherwise (Im z == 0, Re z >= 0) -> -Ei(-z, 0)
//
//   Si(complex z) (.cpp:217-235)
//       B13 |z| <= 0.2 -> Taylor series (straddled by r0p19/r0p2/r0p21)
//       B14 else       -> 0.5i(E1(-iz) - E1(iz) -/+ i*pi), where the sign of
//                         the pi is +1 iff (Re>=0 && Im>=0) || (Re>0 && Im<0).
//                         All four quadrants AND all four half-axes are probed
//                         because the predicate is asymmetric on the axes:
//                         (Re==0, Im<0) takes -pi while (Re>0, Im<0) takes +pi.
//
//   Ci(complex z) (.cpp:237-247)
//       B15 Re z < 0 && Im z >= 0  -> acc = +i*pi
//       B16 Re z <= 0 && Im z <= 0 -> acc = -i*pi
//       B17 otherwise              -> acc = 0
//       Again the axes matter: (Re==0, Im>0) takes B17, (Re==0, Im<0) takes B16.
//
//   heston_* cases
//       The one production caller in this port is AnalyticHestonEngine::
//       AP_Helper::controlVariateValue with ComplexLogFormula::AsymptoticChF
//       (analytichestonengine.cpp:550-576), which evaluates
//           Ci(-0.5*(phi + i*freq)) and Si(0.5*(phi + i*freq))
//       with phi = -(v0 + T*kappa*theta)/sigma * (sqrt(1-rho^2) + i*rho)
//       (.cpp:478-480) and freq = log(fwd/strike). The parameter sets below are
//       chosen so |0.5*(phi+i*freq)| lands in B13, B8, B7 and B6 respectively —
//       i.e. the caller really does reach every branch.
//
// Value encoding
//   Reals: std::setprecision(17), with the non-finite sentinels "nan"/"inf"/
//   "-inf" already used by v143_math_distributions and v143_math_tail.
//   Complex numbers: a two-element JSON array [re, im].
//   A guard that throws is recorded as the string "raises".
//
// Emits ONE JSON object on stdout and nothing else; generate-references.sh
// redirects it to references/v143/math/expint.json.

#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/mathconstants.hpp>
#include <ql/math/integrals/exponentialintegrals.hpp>
#include <ql/types.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// ---------------------------------------------------------------------------
std::string num(Real v) {
    if (std::isnan(v))
        return "\"nan\"";
    if (std::isinf(v))
        return v > 0 ? "\"inf\"" : "\"-inf\"";
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

std::string cnum(const std::complex<Real>& v) {
    return "[" + num(v.real()) + ", " + num(v.imag()) + "]";
}

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& c(const std::string& k, const std::complex<Real>& v) { return put(k, cnum(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, "\"" + v + "\""); }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
    Obj& raw(const std::string& k, const std::string& v) { return put(k, v); }
    std::string str() const { return "{" + body_ + "}"; }

  private:
    Obj& put(const std::string& k, const std::string& v) {
        if (!body_.empty())
            body_ += ", ";
        body_ += "\"" + k + "\": " + v;
        return *this;
    }
    std::string body_;
};

std::vector<std::pair<std::string, std::string>> gCases;

void addCase(const std::string& name, const Obj& inputs, const Obj& expected) {
    gCases.emplace_back(name, "{\"inputs\": " + inputs.str() +
                                  ", \"expected\": " + expected.str() + "}");
}

void emitDocument() {
    std::cout << "{\n";
    for (std::size_t i = 0; i < gCases.size(); ++i)
        std::cout << "  \"" << gCases[i].first << "\": " << gCases[i].second
                  << (i + 1 < gCases.size() ? "," : "") << "\n";
    std::cout << "}\n";
}

// A guard that throws is pinned as the sentinel "raises" rather than a number,
// so a port that silently returns something is caught.
template <class F>
void realOrRaise(Obj& o, const std::string& key, const F& f) {
    Real v;
    try {
        v = f();
    } catch (const std::exception&) {
        o.s(key, "raises");
        return;
    }
    o.n(key, v);
}

// ---------------------------------------------------------------------------
// Argument tables. Each entry carries its own name so the JSON keys are stable
// under reordering, and the numeric argument is echoed into "inputs" so the
// Python test never has to restate a literal.
// ---------------------------------------------------------------------------
struct RealArg {
    const char* name;
    Real x;
};

// Straddles of the single real threshold x == 4.0, computed rather than typed
// so the neighbours really are the adjacent doubles.
const Real kBelow4 = std::nextafter(Real(4.0), Real(0.0));
const Real kAbove4 = std::nextafter(Real(4.0), Real(100.0));

const std::vector<RealArg> kSiRealArgs = {
    {"zero", 0.0},
    {"subnormal", 5e-324},
    {"e300", 1e-300},
    {"e08", 1e-8},
    {"e04", 1e-4},
    {"0p1", 0.1},
    {"0p5", 0.5},
    {"1p0", 1.0},
    {"2p0", 2.0},
    {"3p0", 3.0},
    {"below4", kBelow4},
    {"at4", 4.0},
    {"above4", kAbove4},
    {"4p5", 4.5},
    {"5p0", 5.0},
    {"two_pi", 2.0 * M_PI},
    {"8p0", 8.0},
    {"10p0", 10.0},
    {"20p0", 20.0},
    {"50p0", 50.0},
    {"100p0", 100.0},
    {"1000p0", 1000.0},
    {"1e6", 1e6},
    {"neg_e08", -1e-8},
    {"neg_0p5", -0.5},
    {"neg_1p0", -1.0},
    {"neg_below4", -kBelow4},
    {"neg_at4", -4.0},
    {"neg_above4", -kAbove4},
    {"neg_10p0", -10.0},
    {"neg_100p0", -100.0},
    {"neg_1e6", -1e6},
};

// Ci has the same threshold but a hard guard below zero, so the negative
// entries pin the throw instead of a mirrored value.
const std::vector<RealArg> kCiRealArgs = {
    {"zero", 0.0},
    {"subnormal", 5e-324},
    {"e300", 1e-300},
    {"e08", 1e-8},
    {"e04", 1e-4},
    {"0p1", 0.1},
    {"0p5", 0.5},
    {"0p616", 0.6165054856207162},  // near the Ci zero, where cancellation bites
    {"1p0", 1.0},
    {"2p0", 2.0},
    {"3p0", 3.0},
    {"below4", kBelow4},
    {"at4", 4.0},
    {"above4", kAbove4},
    {"4p5", 4.5},
    {"5p0", 5.0},
    {"two_pi", 2.0 * M_PI},
    {"8p0", 8.0},
    {"10p0", 10.0},
    {"20p0", 20.0},
    {"50p0", 50.0},
    {"100p0", 100.0},
    {"1000p0", 1000.0},
    {"1e6", 1e6},
    {"neg_tiny", -1e-300},
    {"neg_1p0", -1.0},
    {"neg_10p0", -10.0},
};

struct Magnitude {
    const char* name;
    Real r;
};

// 0.2 is the Si(complex) series cut-off, 4.5 is Ei's DIST, 41.4046 is
// 1.1*z_asym. Each is straddled from both sides.
const std::vector<Magnitude> kMagnitudes = {
    {"r0p05", 0.05},  {"r0p19", 0.19}, {"r0p2", 0.2},   {"r0p21", 0.21},
    {"r1p0", 1.0},    {"r3p0", 3.0},   {"r4p4", 4.4},   {"r4p5", 4.5},
    {"r4p6", 4.6},    {"r12p0", 12.0}, {"r41p0", 41.0}, {"r41p5", 41.5},
    {"r50p0", 50.0},
};

// Fewer magnitudes for the E1/Ei grid — those two are pinned to lock Ei's own
// branch selection, which Si/Ci already cover once per magnitude.
const std::vector<Magnitude> kEiMagnitudes = {
    {"r0p05", 0.05}, {"r1p0", 1.0},   {"r4p4", 4.4},   {"r4p6", 4.6},
    {"r12p0", 12.0}, {"r41p0", 41.0}, {"r41p5", 41.5}, {"r50p0", 50.0},
};

struct Direction {
    const char* name;
    Real cx;  // multiplied by the magnitude to get Re
    Real cy;  // multiplied by the magnitude to get Im
};

// The four half-axes are given as EXACT unit vectors (not cos/sin of an angle)
// because every sign predicate in this file distinguishes `>= 0` from `> 0`,
// and sin(M_PI) is 1.2e-16, not zero — a polar grid would silently miss the
// on-axis branches.
const Real kC45 = M_SQRT1_2;
const Real kC30 = 0.86602540378443864676;  // cos(30 deg)
const Real kS30 = 0.5;

const std::vector<Direction> kDirections = {
    {"pos_real", 1.0, 0.0},   {"pos_imag", 0.0, 1.0},   {"neg_real", -1.0, 0.0},
    {"neg_imag", 0.0, -1.0},  {"q1", kC45, kC45},       {"q2", -kC45, kC45},
    {"q3", -kC45, -kC45},     {"q4", kC45, -kC45},      {"q1_shallow", kC30, kS30},
    {"q4_shallow", kC30, -kS30},
};

std::complex<Real> point(const Magnitude& m, const Direction& d) {
    return {m.r * d.cx, m.r * d.cy};
}

void putComplexInput(Obj& in, const std::complex<Real>& z) {
    in.n("re", z.real());
    in.n("im", z.imag());
    in.n("abs", std::abs(z));
}

// ---------------------------------------------------------------------------
// Real Si / Ci — branches B1..B5 plus guard G1.
// ---------------------------------------------------------------------------
void emitRealSi() {
    for (const RealArg& a : kSiRealArgs) {
        Obj in;
        in.n("x", a.x);
        in.s("branch", a.x < 0 ? "negate" : (a.x <= 4.0 ? "series_le_4" : "asymptotic_gt_4"));
        Obj ex;
        realOrRaise(ex, "si", [&] { return ExponentialIntegral::Si(a.x); });
        addCase(std::string("si_real_") + a.name, in, ex);
    }
}

void emitRealCi() {
    for (const RealArg& a : kCiRealArgs) {
        Obj in;
        in.n("x", a.x);
        in.s("branch", a.x < 0 ? "guard_raises"
                               : (a.x <= 4.0 ? "series_le_4" : "asymptotic_gt_4"));
        Obj ex;
        realOrRaise(ex, "ci", [&] { return ExponentialIntegral::Ci(a.x); });
        addCase(std::string("ci_real_") + a.name, in, ex);
    }
}

// ---------------------------------------------------------------------------
// Complex Si / Ci / E1 / Ei — branches B6..B17.
// ---------------------------------------------------------------------------
void emitComplexSi() {
    for (const Magnitude& m : kMagnitudes) {
        for (const Direction& d : kDirections) {
            const std::complex<Real> z = point(m, d);
            Obj in;
            putComplexInput(in, z);
            in.s("branch", std::abs(z) <= 0.2 ? "taylor_le_0p2" : "via_E1");
            in.b("plus_pi", (z.real() >= 0 && z.imag() >= 0) || (z.real() > 0 && z.imag() < 0));
            Obj ex;
            ex.c("si", ExponentialIntegral::Si(z));
            addCase(std::string("si_cplx_") + m.name + "_" + d.name, in, ex);
        }
    }
}

void emitComplexCi() {
    for (const Magnitude& m : kMagnitudes) {
        for (const Direction& d : kDirections) {
            const std::complex<Real> z = point(m, d);
            Obj in;
            putComplexInput(in, z);
            const char* acc = (z.real() < 0.0 && z.imag() >= 0.0)
                                  ? "plus_i_pi"
                                  : ((z.real() <= 0.0 && z.imag() <= 0.0) ? "minus_i_pi" : "zero");
            in.s("acc", acc);
            Obj ex;
            ex.c("ci", ExponentialIntegral::Ci(z));
            addCase(std::string("ci_cplx_") + m.name + "_" + d.name, in, ex);
        }
    }
}

void emitE1AndEi() {
    for (const Magnitude& m : kEiMagnitudes) {
        for (const Direction& d : kDirections) {
            const std::complex<Real> z = point(m, d);
            Obj in;
            putComplexInput(in, z);
            in.s("e1_branch", z.imag() < 0.0 ? "acc_minus_i_pi"
                                             : ((z.imag() > 0.0 || z.real() < 0.0)
                                                    ? "acc_plus_i_pi"
                                                    : "acc_zero"));
            Obj ex;
            ex.c("e1", ExponentialIntegral::E1(z));
            ex.c("ei", ExponentialIntegral::Ei(z));
            addCase(std::string("expint_") + m.name + "_" + d.name, in, ex);
        }
    }

    // Ei's own z == 0 short circuit (G2), and E1 on top of it.
    {
        const std::complex<Real> z(0.0, 0.0);
        Obj in;
        putComplexInput(in, z);
        in.s("guard", "Ei(0) == -inf (has_infinity short circuit)");
        Obj ex;
        ex.c("e1", ExponentialIntegral::E1(z));
        ex.c("ei", ExponentialIntegral::Ei(z));
        addCase("expint_origin", in, ex);
    }
    // Si is a well-behaved odd function at the origin (B13 with |z| == 0);
    // Ci has a log pole there, matching the real Ci(0).
    {
        const std::complex<Real> z(0.0, 0.0);
        Obj in;
        putComplexInput(in, z);
        Obj ex;
        ex.c("si", ExponentialIntegral::Si(z));
        addCase("si_cplx_origin", in, ex);
    }
}

// ---------------------------------------------------------------------------
// The arguments the production Heston caller actually produces.
//
// Mirrors AnalyticHestonEngine::AP_Helper's constructor (.cpp:478-480) and
// controlVariateValue (.cpp:565-573) for ComplexLogFormula::AsymptoticChF:
//
//     phi     = -(v0 + T*kappa*theta)/sigma * (sqrt(1-rho^2) + i*rho)
//     freq    = log(fwd/strike)
//     phiFreq = phi + i*freq
//     ... uses Ci(-0.5*phiFreq) and Si(0.5*phiFreq)
//
// Written out here rather than driven through the engine because phi_ is a
// private member; the point is only that the magnitudes are realistic.
// ---------------------------------------------------------------------------
struct HestonSet {
    const char* name;
    Real v0, kappa, theta, sigma, rho, term, fwd, strike;
};

const std::vector<HestonSet> kHestonSets = {
    // |0.5*phiFreq| ~ 0.18 -> Si takes the Taylor branch B13.
    {"short_dated", 0.04, 1.0, 0.04, 0.5, -0.75, 1.0, 100.0, 100.0},
    // ~ 0.53 -> B14 with Ei on the power series B8.
    {"ten_year", 0.04, 1.0, 0.04, 0.5, -0.75, 10.0, 100.0, 80.0},
    // ~ 18 -> Ei continued fraction B7.
    {"low_vol_of_vol", 0.09, 2.0, 0.09, 0.1, -0.5, 20.0, 100.0, 120.0},
    // ~ 91 -> Ei asymptotic series B6.
    {"very_low_vol_of_vol", 0.1, 3.0, 0.1, 0.05, -0.3, 30.0, 100.0, 95.0},
    // positive correlation, deep ITM strike -> the other sign combination.
    {"positive_rho", 0.06, 1.5, 0.05, 0.3, 0.6, 5.0, 100.0, 150.0},
};

void emitHeston() {
    for (const HestonSet& h : kHestonSets) {
        const Real freq = std::log(h.fwd / h.strike);
        const std::complex<Real> phi =
            -(h.v0 + h.term * h.kappa * h.theta) / h.sigma *
            std::complex<Real>(std::sqrt(1 - h.rho * h.rho), h.rho);
        const std::complex<Real> phiFreq(phi.real(), phi.imag() + freq);

        const std::complex<Real> zSi = 0.5 * phiFreq;
        const std::complex<Real> zCi = -0.5 * phiFreq;

        Obj in;
        in.n("v0", h.v0);
        in.n("kappa", h.kappa);
        in.n("theta", h.theta);
        in.n("sigma", h.sigma);
        in.n("rho", h.rho);
        in.n("term", h.term);
        in.n("fwd", h.fwd);
        in.n("strike", h.strike);
        in.c("phi", phi);
        in.c("phi_freq", phiFreq);
        in.c("z_si", zSi);
        in.c("z_ci", zCi);
        in.n("abs_z_si", std::abs(zSi));

        Obj ex;
        ex.c("si", ExponentialIntegral::Si(zSi));
        ex.c("ci", ExponentialIntegral::Ci(zCi));
        addCase(std::string("heston_") + h.name, in, ex);
    }
}

// ---------------------------------------------------------------------------
// Identity self-checks. These are NOT a substitute for the pinned values; they
// exist so a probe that is self-consistent but wrong (uninitialised memory,
// a mis-linked symbol) cannot pass silently. Each is an independently known
// mathematical fact about Si/Ci.
// ---------------------------------------------------------------------------
void emitIdentities() {
    Obj in;
    in.s("note", "closed-form identities, independent of the pinned tables");
    Obj ex;
    // Si(0) == 0 exactly (odd function, series starts at x).
    ex.n("si_0", ExponentialIntegral::Si(Real(0.0)));
    // Si(-x) == -Si(x) exactly (B1 is literal negation).
    ex.n("si_symmetry_residual_3", ExponentialIntegral::Si(Real(-3.0)) +
                                       ExponentialIntegral::Si(Real(3.0)));
    ex.n("si_symmetry_residual_9", ExponentialIntegral::Si(Real(-9.0)) +
                                       ExponentialIntegral::Si(Real(9.0)));
    // Si(x) -> pi/2 as x -> inf.
    ex.n("si_1e6_minus_half_pi", ExponentialIntegral::Si(Real(1e6)) - M_PI_2);
    ex.n("half_pi", M_PI_2);
    // Ci(0) == -inf (log pole).
    ex.n("ci_0", ExponentialIntegral::Ci(Real(0.0)));
    // Ci has its first positive zero at 0.6165054856207162...
    ex.n("ci_first_zero", ExponentialIntegral::Ci(Real(0.6165054856207162)));
    // On the positive real axis the complex overloads must agree with the real
    // ones: Si(x + 0i) == Si(x) and Re Ci(x + 0i) == Ci(x), Im == 0.
    ex.c("si_cplx_on_axis_3", ExponentialIntegral::Si(std::complex<Real>(3.0, 0.0)));
    ex.n("si_real_3", ExponentialIntegral::Si(Real(3.0)));
    ex.c("si_cplx_on_axis_9", ExponentialIntegral::Si(std::complex<Real>(9.0, 0.0)));
    ex.n("si_real_9", ExponentialIntegral::Si(Real(9.0)));
    ex.c("ci_cplx_on_axis_3", ExponentialIntegral::Ci(std::complex<Real>(3.0, 0.0)));
    ex.n("ci_real_3", ExponentialIntegral::Ci(Real(3.0)));
    ex.c("ci_cplx_on_axis_9", ExponentialIntegral::Ci(std::complex<Real>(9.0, 0.0)));
    ex.n("ci_real_9", ExponentialIntegral::Ci(Real(9.0)));
    // E1 and Ei are reflections of each other on the positive real axis:
    // E1(x) == -Ei(-x) for x > 0.
    ex.c("e1_2", ExponentialIntegral::E1(std::complex<Real>(2.0, 0.0)));
    ex.c("ei_minus_2", ExponentialIntegral::Ei(std::complex<Real>(-2.0, 0.0)));
    addCase("_identities", in, ex);
}

void emitMeta() {
    // The literal thresholds the branches key on, recorded so a port can assert
    // it derived the same numbers rather than typing them in.
    constexpr double MAX_ERROR = 5.0 * QL_EPSILON;
    Obj in;
    in.s("source", "ql/math/integrals/exponentialintegrals.cpp (v1.43)");
    Obj ex;
    ex.n("QL_EPSILON", QL_EPSILON);
    ex.n("MAX_ERROR", MAX_ERROR);
    ex.n("DIST", 4.5);
    ex.n("z_inf", std::log(0.01 * QL_MAX_REAL) + std::log(100.0));
    ex.n("z_asym", 2.0 - 1.035 * std::log(MAX_ERROR));
    ex.n("asymptotic_threshold", 1.1 * (2.0 - 1.035 * std::log(MAX_ERROR)));
    ex.n("euler_mascheroni", 0.5772156649015328606065121);
    ex.n("real_threshold", 4.0);
    ex.n("si_series_threshold", 0.2);
    addCase("_meta", in, ex);
}

}  // namespace

int main() {
    emitMeta();
    emitIdentities();
    emitRealSi();
    emitRealCi();
    emitComplexSi();
    emitComplexCi();
    emitE1AndEi();
    emitHeston();

    emitDocument();
    return 0;
}
