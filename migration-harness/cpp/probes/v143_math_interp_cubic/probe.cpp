// migration-harness/cpp/probes/v143_math_interp_cubic/probe.cpp
//
// Reference values for the cubic / log / mixed / convex-monotone block of
// ql/math/interpolations at C++ QuantLib v1.43:
//
//   cubicinterpolation.hpp          CubicInterpolation + every convenience
//                                   preset (Cubic factory traits included)
//   loginterpolation.hpp            LogLinear / LogCubic / LogMixedLinearCubic
//                                   families
//   mixedinterpolation.hpp          MixedLinearCubic family (ShareRanges,
//                                   SplitRanges, derivative matching)
//   convexmonotoneinterpolation.hpp ConvexMonotone + the SectionHelper state
//                                   machine
//   linearinterpolation.hpp         Linear
//   backwardflatinterpolation.hpp   BackwardFlat
//   forwardflatinterpolation.hpp    ForwardFlat
//
// What is pinned, per the harness rule for interpolations: value, first AND
// second derivative, and the primitive, at nodes, at midpoints, and outside
// the range on both sides. For the cubic family the per-interval coefficient
// arrays (a, b, c), the primitive constants and the monotonicity-adjustment
// flags are pinned too: those are the actual output of the derivative
// approximation + boundary condition + Hyman filter pipeline, so a wrong
// DerivativeApprox preset shows up there directly rather than being smeared
// into a value mismatch.
//
// The factory/traits constants (`global`, `requiredPoints`) are pinned as
// well — the bootstrap machinery branches on them, so they are semantics,
// not decoration.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/interp/cubic.json.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/interpolations/backwardflatinterpolation.hpp>
#include <ql/math/interpolations/convexmonotoneinterpolation.hpp>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/interpolations/forwardflatinterpolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/interpolations/loginterpolation.hpp>
#include <ql/math/interpolations/mixedinterpolation.hpp>
#include <ql/utilities/null.hpp>

using namespace QuantLib;

namespace {

typedef std::vector<Real>::const_iterator It;
typedef CubicInterpolation CI;

// ---------------------------------------------------------------------------
// minimal JSON emission
// ---------------------------------------------------------------------------

// JSON has no literal for the non-finite doubles, and QuantLib genuinely
// produces them here: the FritschButland arm assigns QL_MIN_REAL / QL_MAX_REAL
// when `Smax + 2*Smin` cancels exactly, which then overflows the cubic
// coefficients. Emitting them as the strings "nan" / "inf" / "-inf" keeps the
// document parseable while still pinning the degenerate branch.
void emitOne(Real x) {
    if (std::isnan(x))
        std::cout << "\"nan\"";
    else if (std::isinf(x))
        std::cout << (x > 0 ? "\"inf\"" : "\"-inf\"");
    else
        std::cout << x;
}

void emitReals(const std::string& key, const std::vector<Real>& v, bool comma) {
    std::cout << "      \"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i)
            std::cout << ", ";
        emitOne(v[i]);
    }
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitFlags(const std::string& key, const std::vector<bool>& v, bool comma) {
    std::cout << "      \"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << (v[i] ? "true" : "false");
    std::cout << "]" << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------------------------
// test curves
// ---------------------------------------------------------------------------

struct Curve {
    std::string id;
    std::vector<Real> x, y;
};

// Monotone increasing, uniform grid. Same knots as the pre-existing Hyman
// probe so the two references stay comparable.
const Curve kMono = {"mono",
                     {0.0, 1.0, 2.0, 3.0, 4.0},
                     {0.0, 0.5, 1.5, 3.0, 3.2}};

// Non-monotone, non-uniform grid: sign changes in the chord slopes drive the
// Kruger / Harmonic / FritschButland zero branches and the Hyman filter's
// "correction = 0" arm.
const Curve kWiggly = {"wiggly",
                       {0.0, 1.0, 2.5, 3.0, 4.5, 6.0},
                       {5.0, 3.0, 4.0, 2.0, 1.0, 3.0}};

// Three points: exercises the local schemes' end-point formulas where
// n-3 == 0, i.e. the "both end formulas read the same two slopes" corner.
const Curve kTriple = {"triple", {0.0, 1.5, 4.0}, {1.0, 2.5, 2.0}};

// Two points: the local schemes' `n_ == 2` short-circuit (tmp_[0] = tmp_[1]).
const Curve kPair = {"pair", {0.0, 2.0}, {1.0, 4.0}};

// Strictly positive, discount-factor shaped — the log family needs y > 0.
const Curve kPositive = {"positive",
                         {0.0, 0.5, 1.0, 2.0, 5.0, 10.0},
                         {1.0, 0.99, 0.97, 0.94, 0.85, 0.70}};

// Evaluation grid: below the range, every node, every midpoint, above the
// range. Extrapolation is requested explicitly on every call.
std::vector<Real> evalPoints(const Curve& c) {
    std::vector<Real> p;
    p.push_back(c.x.front() - 0.5);
    for (Size i = 0; i < c.x.size(); ++i) {
        p.push_back(c.x[i]);
        if (i + 1 < c.x.size())
            p.push_back(0.5 * (c.x[i] + c.x[i + 1]));
    }
    p.push_back(c.x.back() + 0.7);
    return p;
}

// Which of the four evaluators the interpolation actually implements.
struct Emit {
    bool value = true;
    bool derivative = true;
    bool secondDerivative = true;
    bool primitive = true;
};

bool g_firstCase = true;

void openCase(const std::string& name, const Curve& c) {
    if (!g_firstCase)
        std::cout << ",\n";
    g_firstCase = false;
    std::cout << "    {\n";
    std::cout << "      \"name\": \"" << name << "\",\n";
    std::cout << "      \"curve\": \"" << c.id << "\",\n";
    emitReals("x", c.x, true);
    emitReals("y", c.y, true);
}

void emitEvaluations(const Curve& c, const Interpolation& f, const Emit& e,
                     bool comma) {
    const std::vector<Real> pts = evalPoints(c);
    emitReals("eval_x", pts, true);
    if (e.value) {
        std::vector<Real> v;
        for (Real x : pts)
            v.push_back(f(x, true));
        emitReals("value", v, e.derivative || e.secondDerivative || e.primitive || comma);
    }
    if (e.derivative) {
        std::vector<Real> v;
        for (Real x : pts)
            v.push_back(f.derivative(x, true));
        emitReals("derivative", v, e.secondDerivative || e.primitive || comma);
    }
    if (e.secondDerivative) {
        std::vector<Real> v;
        for (Real x : pts)
            v.push_back(f.secondDerivative(x, true));
        emitReals("second_derivative", v, e.primitive || comma);
    }
    if (e.primitive) {
        std::vector<Real> v;
        for (Real x : pts)
            v.push_back(f.primitive(x, true));
        emitReals("primitive", v, comma);
    }
}

void closeCase() { std::cout << "    }"; }

// ---------------------------------------------------------------------------
// cubic cases
// ---------------------------------------------------------------------------

void emitCubic(const std::string& name, const Curve& c, const CI& f) {
    openCase(name, c);
    emitEvaluations(c, f, Emit(), true);
    emitReals("a", f.aCoefficients(), true);
    emitReals("b", f.bCoefficients(), true);
    emitReals("c", f.cCoefficients(), true);
    emitReals("primitive_const", f.primitiveConstants(), true);
    emitFlags("monotonicity_adjustments", f.monotonicityAdjustments(), false);
    closeCase();
}

void emitCubicConfig(const std::string& name, const Curve& c,
                     CI::DerivativeApprox da, bool monotonic,
                     CI::BoundaryCondition leftC, Real leftV,
                     CI::BoundaryCondition rightC, Real rightV) {
    CI f(c.x.begin(), c.x.end(), c.y.begin(), da, monotonic, leftC, leftV,
         rightC, rightV);
    emitCubic(name, c, f);
}

void cubicSection() {
    const Curve curves[] = {kMono, kWiggly};

    for (const Curve& c : curves) {
        const std::string s = "/" + c.id;

        // --- Spline, every boundary condition -----------------------------
        emitCubicConfig("Spline.Natural" + s, c, CI::Spline, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Spline.Natural.monotonic" + s, c, CI::Spline, true,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Spline.SecondDerivative.nonzero" + s, c, CI::Spline,
                        false, CI::SecondDerivative, 0.3, CI::SecondDerivative,
                        -0.2);
        emitCubicConfig("Spline.FirstDerivative" + s, c, CI::Spline, false,
                        CI::FirstDerivative, 0.5, CI::FirstDerivative, -0.25);
        emitCubicConfig("Spline.NotAKnot" + s, c, CI::Spline, false,
                        CI::NotAKnot, 123.0 /*ignored*/, CI::NotAKnot,
                        -7.0 /*ignored*/);
        emitCubicConfig("Spline.Lagrange" + s, c, CI::Spline, false,
                        CI::Lagrange, 0.0, CI::Lagrange, 0.0);
        // Deliberately mixed ends — C++ allows any pairing.
        emitCubicConfig("Spline.NotAKnot.left.FirstDerivative.right" + s, c,
                        CI::Spline, false, CI::NotAKnot, 0.0,
                        CI::FirstDerivative, -0.4);
        emitCubicConfig("Spline.Lagrange.left.NotAKnot.right" + s, c, CI::Spline,
                        false, CI::Lagrange, 0.0, CI::NotAKnot, 0.0);

        // --- overshooting-minimisation splines ----------------------------
        emitCubicConfig("SplineOM1" + s, c, CI::SplineOM1, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("SplineOM2" + s, c, CI::SplineOM2, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);

        // --- local schemes ------------------------------------------------
        emitCubicConfig("Parabolic" + s, c, CI::Parabolic, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Parabolic.monotonic" + s, c, CI::Parabolic, true,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("FritschButland" + s, c, CI::FritschButland, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("FritschButland.monotonic" + s, c, CI::FritschButland,
                        true, CI::SecondDerivative, 0.0, CI::SecondDerivative,
                        0.0);
        emitCubicConfig("Akima" + s, c, CI::Akima, false, CI::SecondDerivative,
                        0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Akima.monotonic" + s, c, CI::Akima, true,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Kruger" + s, c, CI::Kruger, false, CI::SecondDerivative,
                        0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Kruger.monotonic" + s, c, CI::Kruger, true,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Harmonic" + s, c, CI::Harmonic, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Harmonic.monotonic" + s, c, CI::Harmonic, true,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);

        // --- convenience presets (these ARE the semantics) ----------------
        emitCubic("CubicNaturalSpline" + s, c,
                  CubicNaturalSpline(c.x.begin(), c.x.end(), c.y.begin()));
        emitCubic("MonotonicCubicNaturalSpline" + s, c,
                  MonotonicCubicNaturalSpline(c.x.begin(), c.x.end(),
                                              c.y.begin()));
        emitCubic("CubicSplineOvershootingMinimization1" + s, c,
                  CubicSplineOvershootingMinimization1(c.x.begin(), c.x.end(),
                                                       c.y.begin()));
        emitCubic("CubicSplineOvershootingMinimization2" + s, c,
                  CubicSplineOvershootingMinimization2(c.x.begin(), c.x.end(),
                                                       c.y.begin()));
        emitCubic("AkimaCubicInterpolation" + s, c,
                  AkimaCubicInterpolation(c.x.begin(), c.x.end(), c.y.begin()));
        emitCubic("KrugerCubic" + s, c,
                  KrugerCubic(c.x.begin(), c.x.end(), c.y.begin()));
        emitCubic("HarmonicCubic" + s, c,
                  HarmonicCubic(c.x.begin(), c.x.end(), c.y.begin()));
        emitCubic("FritschButlandCubic" + s, c,
                  FritschButlandCubic(c.x.begin(), c.x.end(), c.y.begin()));
        emitCubic("Parabolic" + s + "/preset", c,
                  QuantLib::Parabolic(c.x.begin(), c.x.end(), c.y.begin()));
        emitCubic("MonotonicParabolic" + s, c,
                  MonotonicParabolic(c.x.begin(), c.x.end(), c.y.begin()));

        // --- the Cubic factory (traits object) round-trips to the same -----
        {
            Cubic factory(CI::Kruger, false, CI::SecondDerivative, 0.0,
                          CI::SecondDerivative, 0.0);
            Interpolation f =
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin());
            openCase("Cubic.factory.Kruger" + s, c);
            emitEvaluations(c, f, Emit(), false);
            closeCase();
        }
        {
            // Cubic's own defaults: Kruger, non-monotonic, natural at both ends.
            Cubic factory;
            Interpolation f =
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin());
            openCase("Cubic.factory.defaults" + s, c);
            emitEvaluations(c, f, Emit(), false);
            closeCase();
        }
    }

    // Three-point curve: the local end-point formulas where n-3 == 0.
    {
        const Curve& c = kTriple;
        emitCubicConfig("Parabolic/triple", c, CI::Parabolic, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("FritschButland/triple", c, CI::FritschButland, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Kruger/triple", c, CI::Kruger, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Harmonic/triple", c, CI::Harmonic, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Spline.Natural/triple", c, CI::Spline, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        // NOTE: Spline + NotAKnot at BOTH ends with exactly three points is
        // singular in C++ — the elimination hits bet == 0 on the last row and
        // TridiagonalOperator::solveFor throws "division by zero". Not emitted
        // here; the Python test asserts the same failure instead.
    }

    // Two-point curve: the `n_ == 2` short-circuit of the local schemes and
    // the two-row tridiagonal solve of the spline schemes.
    {
        const Curve& c = kPair;
        emitCubicConfig("Kruger/pair", c, CI::Kruger, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Kruger.monotonic/pair", c, CI::Kruger, true,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Parabolic/pair", c, CI::Parabolic, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Harmonic/pair", c, CI::Harmonic, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("FritschButland/pair", c, CI::FritschButland, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Spline.Natural/pair", c, CI::Spline, false,
                        CI::SecondDerivative, 0.0, CI::SecondDerivative, 0.0);
        emitCubicConfig("Spline.FirstDerivative/pair", c, CI::Spline, false,
                        CI::FirstDerivative, 1.0, CI::FirstDerivative, 2.0);
    }
}

// ---------------------------------------------------------------------------
// linear / backward-flat / forward-flat
// ---------------------------------------------------------------------------

void flatSection() {
    const Curve curves[] = {kMono, kWiggly};
    for (const Curve& c : curves) {
        const std::string s = "/" + c.id;
        {
            Linear factory;
            Interpolation f =
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin());
            openCase("Linear" + s, c);
            emitEvaluations(c, f, Emit(), false);
            closeCase();
        }
        {
            BackwardFlat factory;
            Interpolation f =
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin());
            openCase("BackwardFlat" + s, c);
            emitEvaluations(c, f, Emit(), false);
            closeCase();
        }
        {
            ForwardFlat factory;
            Interpolation f =
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin());
            openCase("ForwardFlat" + s, c);
            emitEvaluations(c, f, Emit(), false);
            closeCase();
        }
    }
    // BackwardFlat accepts a single point (requiredPoints == 1); both value
    // and primitive have a dedicated branch for it.
    {
        Curve one = {"single", {2.0}, {0.75}};
        BackwardFlatInterpolation f(one.x.begin(), one.x.end(), one.y.begin());
        openCase("BackwardFlat/single", one);
        Emit e;
        // evalPoints() needs at least the node itself; build it by hand.
        std::vector<Real> pts = {1.0, 2.0, 3.5};
        emitReals("eval_x", pts, true);
        std::vector<Real> v, d, dd, p;
        for (Real x : pts) {
            v.push_back(f(x, true));
            d.push_back(f.derivative(x, true));
            dd.push_back(f.secondDerivative(x, true));
            p.push_back(f.primitive(x, true));
        }
        emitReals("value", v, true);
        emitReals("derivative", d, true);
        emitReals("second_derivative", dd, true);
        emitReals("primitive", p, false);
        (void)e;
        closeCase();
    }
}

// ---------------------------------------------------------------------------
// log family
// ---------------------------------------------------------------------------

void emitLog(const std::string& name, const Curve& c, const Interpolation& f) {
    openCase(name, c);
    Emit e;
    e.primitive = false; // LogInterpolationImpl::primitive QL_FAILs
    emitEvaluations(c, f, e, false);
    closeCase();
}

void logSection() {
    const Curve& c = kPositive;

    emitLog("LogLinearInterpolation",
            c, LogLinearInterpolation(c.x.begin(), c.x.end(), c.y.begin()));
    {
        LogLinear factory;
        emitLog("LogLinear.factory", c,
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin()));
    }

    // LogCubicInterpolation, spelled out across the derivative approximations.
    struct DaCase {
        const char* name;
        CI::DerivativeApprox da;
        bool monotonic;
    };
    const DaCase daCases[] = {
        {"Spline", CI::Spline, false},
        {"Spline.monotonic", CI::Spline, true},
        {"Parabolic", CI::Parabolic, false},
        {"Parabolic.monotonic", CI::Parabolic, true},
        {"FritschButland", CI::FritschButland, false},
        {"Kruger", CI::Kruger, false},
        {"Harmonic", CI::Harmonic, false},
        {"Akima", CI::Akima, false},
    };
    for (const DaCase& d : daCases) {
        emitLog(std::string("LogCubicInterpolation.") + d.name, c,
                LogCubicInterpolation(c.x.begin(), c.x.end(), c.y.begin(), d.da,
                                      d.monotonic, CI::SecondDerivative, 0.0,
                                      CI::SecondDerivative, 0.0));
    }

    // Convenience presets.
    emitLog("LogCubicNaturalSpline", c,
            LogCubicNaturalSpline(c.x.begin(), c.x.end(), c.y.begin()));
    emitLog("MonotonicLogCubicNaturalSpline", c,
            MonotonicLogCubicNaturalSpline(c.x.begin(), c.x.end(), c.y.begin()));
    emitLog("KrugerLogCubic", c,
            KrugerLogCubic(c.x.begin(), c.x.end(), c.y.begin()));
    emitLog("HarmonicLogCubic", c,
            HarmonicLogCubic(c.x.begin(), c.x.end(), c.y.begin()));
    emitLog("FritschButlandLogCubic", c,
            FritschButlandLogCubic(c.x.begin(), c.x.end(), c.y.begin()));
    emitLog("LogParabolic", c,
            LogParabolic(c.x.begin(), c.x.end(), c.y.begin()));
    emitLog("MonotonicLogParabolic", c,
            MonotonicLogParabolic(c.x.begin(), c.x.end(), c.y.begin()));

    // LogCubic factory presets.
    {
        LogCubic factory(CI::Kruger);
        emitLog("LogCubic.factory.Kruger.defaultMonotonicTrue", c,
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin()));
    }
    {
        MonotonicLogCubic factory;
        emitLog("MonotonicLogCubic.factory", c,
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin()));
    }
    {
        KrugerLog factory;
        emitLog("KrugerLog.factory", c,
                factory.interpolate(c.x.begin(), c.x.end(), c.y.begin()));
    }

    // Log-mixed-linear-cubic.
    for (Size n = 1; n <= 3; ++n) {
        for (int b = 0; b < 2; ++b) {
            MixedInterpolation::Behavior behavior =
                b == 0 ? MixedInterpolation::ShareRanges
                       : MixedInterpolation::SplitRanges;
            const std::string bn = b == 0 ? "ShareRanges" : "SplitRanges";
            emitLog("LogMixedLinearCubicInterpolation.Spline." + bn + ".n" +
                        std::to_string(n),
                    c,
                    LogMixedLinearCubicInterpolation(
                        c.x.begin(), c.x.end(), c.y.begin(), n, behavior,
                        CI::Spline, false, CI::SecondDerivative, 0.0,
                        CI::SecondDerivative, 0.0));
            emitLog("LogMixedLinearCubicNaturalSpline." + bn + ".n" +
                        std::to_string(n),
                    c,
                    LogMixedLinearCubicNaturalSpline(c.x.begin(), c.x.end(),
                                                     c.y.begin(), n, behavior));
            {
                MonotonicLogMixedLinearCubic factory(n, behavior);
                emitLog("MonotonicLogMixedLinearCubic.factory." + bn + ".n" +
                            std::to_string(n),
                        c,
                        factory.interpolate(c.x.begin(), c.x.end(),
                                            c.y.begin()));
            }
            {
                KrugerLogMixedLinearCubic factory(n, behavior);
                emitLog("KrugerLogMixedLinearCubic.factory." + bn + ".n" +
                            std::to_string(n),
                        c,
                        factory.interpolate(c.x.begin(), c.x.end(),
                                            c.y.begin()));
            }
            {
                LogMixedLinearCubic factory(n, behavior, CI::Parabolic, true,
                                            CI::SecondDerivative, 0.0,
                                            CI::SecondDerivative, 0.0);
                emitLog("LogMixedLinearCubic.factory.Parabolic.monotonic." + bn +
                            ".n" + std::to_string(n),
                        c,
                        factory.interpolate(c.x.begin(), c.x.end(),
                                            c.y.begin()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mixed linear/cubic
// ---------------------------------------------------------------------------

void mixedSection() {
    const Curve curves[] = {kMono, kWiggly};
    for (const Curve& c : curves) {
        const std::string s = "/" + c.id;
        for (Size n = 1; n + 1 < c.x.size(); ++n) {
            for (int b = 0; b < 2; ++b) {
                MixedInterpolation::Behavior behavior =
                    b == 0 ? MixedInterpolation::ShareRanges
                           : MixedInterpolation::SplitRanges;
                const std::string bn =
                    (b == 0 ? "ShareRanges" : "SplitRanges");
                const std::string tag = "." + bn + ".n" + std::to_string(n) + s;

                {
                    MixedLinearCubicInterpolation f(
                        c.x.begin(), c.x.end(), c.y.begin(), n, behavior,
                        CI::Spline, false, CI::SecondDerivative, 0.0,
                        CI::SecondDerivative, 0.0);
                    openCase("MixedLinearCubicInterpolation.Spline" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
                {
                    MixedLinearCubicNaturalSpline f(c.x.begin(), c.x.end(),
                                                    c.y.begin(), n, behavior);
                    openCase("MixedLinearCubicNaturalSpline" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
                {
                    MixedLinearMonotonicCubicNaturalSpline f(
                        c.x.begin(), c.x.end(), c.y.begin(), n, behavior);
                    openCase("MixedLinearMonotonicCubicNaturalSpline" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
                {
                    MixedLinearKrugerCubic f(c.x.begin(), c.x.end(),
                                             c.y.begin(), n, behavior);
                    openCase("MixedLinearKrugerCubic" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
                {
                    MixedLinearFritschButlandCubic f(
                        c.x.begin(), c.x.end(), c.y.begin(), n, behavior);
                    openCase("MixedLinearFritschButlandCubic" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
                {
                    MixedLinearParabolic f(c.x.begin(), c.x.end(), c.y.begin(),
                                           n, behavior);
                    openCase("MixedLinearParabolic" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
                {
                    MixedLinearMonotonicParabolic f(c.x.begin(), c.x.end(),
                                                    c.y.begin(), n, behavior);
                    openCase("MixedLinearMonotonicParabolic" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
                {
                    MixedLinearCubic factory(n, behavior, CI::Kruger, false,
                                             CI::SecondDerivative, 0.0,
                                             CI::SecondDerivative, 0.0);
                    Interpolation f = factory.interpolate(
                        c.x.begin(), c.x.end(), c.y.begin());
                    openCase("MixedLinearCubic.factory.Kruger" + tag, c);
                    emitEvaluations(c, f, Emit(), false);
                    closeCase();
                }
            }

            // Derivative matching: SplitRanges only, leftC = FirstDerivative
            // with leftConditionValue = Null<Real>(). The switch function
            // overwrites the cubic's left condition with the linear segment's
            // slope at the switch point before update().
            {
                MixedLinearCubicInterpolation f(
                    c.x.begin(), c.x.end(), c.y.begin(), n,
                    MixedInterpolation::SplitRanges, CI::Spline, false,
                    CI::FirstDerivative, Null<Real>(), CI::SecondDerivative,
                    0.0);
                openCase("MixedLinearCubicInterpolation.matchDerivatives.n" +
                             std::to_string(n) + s,
                         c);
                emitEvaluations(c, f, Emit(), false);
                closeCase();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// convex monotone
// ---------------------------------------------------------------------------

void emitConvex(const std::string& name, const Curve& c, Real quadraticity,
                Real monotonicity, bool forcePositive, bool flatFinalPeriod) {
    ConvexMonotoneInterpolation<It, It> f(c.x.begin(), c.x.end(), c.y.begin(),
                                          quadraticity, monotonicity,
                                          forcePositive, flatFinalPeriod);
    openCase(name, c);
    std::cout << "      \"quadraticity\": " << quadraticity << ",\n";
    std::cout << "      \"monotonicity\": " << monotonicity << ",\n";
    std::cout << "      \"force_positive\": "
              << (forcePositive ? "true" : "false") << ",\n";
    std::cout << "      \"flat_final_period\": "
              << (flatFinalPeriod ? "true" : "false") << ",\n";
    Emit e;
    e.derivative = false;      // QL_FAIL
    e.secondDerivative = false; // QL_FAIL
    emitEvaluations(c, f, e, false);
    closeCase();
}

void convexSection() {
    // Rate-shaped: strictly increasing discrete forwards.
    const Curve rates = {"rates",
                         {0.0, 0.5, 1.0, 2.0, 5.0, 10.0},
                         {0.0, 0.018, 0.020, 0.025, 0.028, 0.030}};
    // Sign-changing g's: drives ConvexMonotone2/3 and the eta clamps.
    const Curve humped = {"humped",
                          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          {0.0, 0.02, 0.05, 0.01, 0.04, 0.005, 0.03}};
    // Near-zero / negative averages: drives ConvexMonotone4MinHelper's
    // splitRegion_ branch and QuadraticMinHelper's dAv >= 0 branch.
    const Curve nearZero = {"near_zero",
                            {0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                            {0.0, 0.001, 0.03, 0.0005, 0.02, 0.0002}};
    // Flat: the |gPrev| < 1e-14 && |gNext| < 1e-14 ConstantGradHelper arm.
    const Curve flat = {"flat",
                        {0.0, 1.0, 2.0, 5.0, 10.0},
                        {0.03, 0.03, 0.03, 0.03, 0.03}};
    // Two points: the length_ == 2 EverywhereConstantHelper short-circuit.
    const Curve twoPoint = {"two_point", {0.0, 1.0}, {0.0, 0.02}};

    const Curve curves[] = {rates, humped, nearZero, flat, twoPoint};

    struct Params {
        Real quadraticity, monotonicity;
        bool forcePositive;
    };
    const Params params[] = {
        {0.0, 1.0, false}, {0.0, 1.0, true},  {0.0, 0.0, true},
        {0.3, 0.7, true},  {0.3, 0.7, false}, {1.0, 0.7, true},
        {1.0, 0.7, false}, {0.5, 0.5, true},
    };

    for (const Curve& c : curves) {
        for (const Params& p : params) {
            for (int flat_ = 0; flat_ < 2; ++flat_) {
                std::string name = "ConvexMonotone/" + c.id + "/q" +
                                   std::to_string(p.quadraticity) + "/m" +
                                   std::to_string(p.monotonicity) + "/fp" +
                                   (p.forcePositive ? "1" : "0") + "/flat" +
                                   (flat_ ? "1" : "0");
                emitConvex(name, c, p.quadraticity, p.monotonicity,
                           p.forcePositive, flat_ != 0);
            }
        }
    }

    // The ConvexMonotone factory's own defaults (0.3, 0.7, forcePositive).
    {
        ConvexMonotone factory;
        Interpolation f = factory.interpolate(rates.x.begin(), rates.x.end(),
                                              rates.y.begin());
        openCase("ConvexMonotone.factory.defaults", rates);
        Emit e;
        e.derivative = false;
        e.secondDerivative = false;
        emitEvaluations(rates, f, e, false);
        closeCase();
    }

    // localInterpolate: the LocalBootstrap path. Grow the curve one pillar at
    // a time, feeding the previous interpolation's helpers forward. The
    // final call (length == finalSize) turns the flat-final-period flag off.
    {
        ConvexMonotone factory(0.3, 0.7, true);
        const Size finalSize = rates.x.size();
        // The interpolations hold ITERATORS into these vectors, and the
        // chained call reads the previous interpolation's data, so every
        // prefix must outlive the whole loop.
        std::vector<std::vector<Real> > xStore, yStore;
        for (Size len = 2; len <= finalSize; ++len) {
            xStore.push_back(
                std::vector<Real>(rates.x.begin(), rates.x.begin() + len));
            yStore.push_back(
                std::vector<Real>(rates.y.begin(), rates.y.begin() + len));
        }
        Interpolation prev;
        Interpolation cur;
        for (Size len = 2; len <= finalSize; ++len) {
            const std::vector<Real>& xs = xStore[len - 2];
            const std::vector<Real>& ys = yStore[len - 2];
            // localisation == len - 1 on the first call, then 1.
            Size localisation = (len == 2) ? (len - 1) : 1;
            cur = factory.localInterpolate(xs.begin(), xs.end(), ys.begin(),
                                           localisation, prev, finalSize);
            Curve step = {"rates_prefix" + std::to_string(len), xs, ys};
            openCase("ConvexMonotone.localInterpolate.len" +
                         std::to_string(len),
                     step);
            std::cout << "      \"final_size\": " << finalSize << ",\n";
            std::cout << "      \"localisation\": " << localisation << ",\n";
            Emit e;
            e.derivative = false;
            e.secondDerivative = false;
            emitEvaluations(step, cur, e, false);
            closeCase();
            prev = cur;
        }
    }
}

// ---------------------------------------------------------------------------
// factory traits
// ---------------------------------------------------------------------------

void traitsSection() {
    struct T {
        const char* name;
        bool global;
        Size requiredPoints;
    };
    const T traits[] = {
        {"Linear", Linear::global, Linear::requiredPoints},
        {"BackwardFlat", BackwardFlat::global, BackwardFlat::requiredPoints},
        {"ForwardFlat", ForwardFlat::global, ForwardFlat::requiredPoints},
        {"Cubic", Cubic::global, Cubic::requiredPoints},
        {"LogLinear", LogLinear::global, LogLinear::requiredPoints},
        {"LogCubic", LogCubic::global, LogCubic::requiredPoints},
        {"MixedLinearCubic", MixedLinearCubic::global,
         MixedLinearCubic::requiredPoints},
        {"LogMixedLinearCubic", LogMixedLinearCubic::global,
         LogMixedLinearCubic::requiredPoints},
        {"ConvexMonotone", ConvexMonotone::global,
         ConvexMonotone::requiredPoints},
    };
    std::cout << "  \"traits\": {\n";
    for (Size i = 0; i < sizeof(traits) / sizeof(traits[0]); ++i) {
        std::cout << "    \"" << traits[i].name << "\": {\"global\": "
                  << (traits[i].global ? "true" : "false")
                  << ", \"required_points\": " << traits[i].requiredPoints
                  << "}" << (i + 1 < sizeof(traits) / sizeof(traits[0]) ? "," : "")
                  << "\n";
    }
    std::cout << "    },\n";
    std::cout << "  \"convex_monotone_data_size_adjustment\": "
              << ConvexMonotone::dataSizeAdjustment << ",\n";
}

void section(const char* key, void (*body)()) {
    g_firstCase = true;
    std::cout << "  \"" << key << "\": [\n";
    body();
    std::cout << "\n  ],\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    traitsSection();
    section("cubic", cubicSection);
    section("flat", flatSection);
    section("log", logSection);
    section("mixed", mixedSection);
    g_firstCase = true;
    std::cout << "  \"convex_monotone\": [\n";
    convexSection();
    std::cout << "\n  ]\n";
    std::cout << "}\n";
    return 0;
}
