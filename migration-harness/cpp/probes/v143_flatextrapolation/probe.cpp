// migration-harness/cpp/probes/v143_flatextrapolation/probe.cpp
//
// Reference values for FlatExtrapolator — the header-only 1-D interpolation
// decorator introduced in C++ QuantLib v1.43
// (ql/math/interpolations/flatextrapolation.hpp).
//
// The decorator clamps the abscissa into [xMin, xMax] before delegating, and
// overrides the calculus accordingly:
//
//   value(x)            = decorated(clamp(x, xMin, xMax), allowExtrapolation=true)
//   derivative(x)       = 0                       if x < xMin || x > xMax
//                       = decorated.derivative(x) otherwise
//   secondDerivative(x) = 0                       if x < xMin || x > xMax
//                       = decorated.secondDerivative(x) otherwise
//   primitive(x)        = decorated.primitive(xMin) + decorated(xMin)*(x - xMin)   [x < xMin]
//                       = decorated.primitive(xMax) + decorated(xMax)*(x - xMax)   [x > xMax]
//                       = decorated.primitive(x)                                   [otherwise]
//
// WHAT NEEDS PINNING, AND WHY
// ---------------------------
//   * The endpoint boundary is STRICT (`x < xMin || x > xMax`). At exactly
//     xMin and xMax the derivative and second derivative come from the
//     underlying interpolation, NOT from the flat branch. A port that writes
//     `<=` / `>=` there silently zeroes the endpoint slopes, and a natural
//     spline (second derivative 0 at both ends by construction) would hide the
//     bug — so a NOT-A-KNOT spline, whose endpoint second derivative is
//     nonzero, is probed as well.
//   * The primitive must EXTEND LINEARLY outside the range (constant
//     integrand => affine antiderivative), not stay flat and not clamp.
//   * The decorator has its OWN Extrapolator flag. Enabling extrapolation on
//     the decorated interpolation does not enable it on the decorator, so
//     FlatExtrapolator still throws out of range until its own
//     enableExtrapolation() is called. (Internally it always passes
//     allowExtrapolation = true down to the decorated object.)
//   * Three different underlyings are decorated (natural cubic spline,
//     not-a-knot cubic spline, linear) plus raw-underlying reference blocks,
//     so a failure can be localised to the decorator vs the interpolator.
//
// Mirrors test-suite/interpolations.cpp @ v1.43 (testFlatExtrapolation) for
// the natural-spline data, and extends it.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/flatextrapolation.json.

#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/interpolations/flatextrapolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/utilities/null.hpp>

#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
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

void emit_arr(const char* name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    emit_arr(name.c_str(), v);
}

// --------------------------------------------------------------------------
// Data. Namespace scope so it outlives every interpolation built on it
// (Interpolation stores iterators, not copies).
// --------------------------------------------------------------------------

// Upstream testFlatExtrapolation data.
const Real kX[] = {0.0, 1.0, 2.0, 3.0, 4.0};
const Real kY[] = {5.0, 3.0, 4.0, 2.0, 1.0};

// Non-collinear data for the linear-interpolation underlying.
const Real kXL[] = {0.0, 1.0, 2.0, 3.0, 4.0};
const Real kYL[] = {1.0, 2.5, 2.0, 4.0, 3.5};

// Probe abscissae.
const std::vector<Real> kNodes = {0.0, 1.0, 2.0, 3.0, 4.0};
const std::vector<Real> kMidpoints = {0.5, 1.5, 2.5, 3.5};
const std::vector<Real> kBelow = {-100.0, -10.0, -2.0, -1.0, -0.01};
const std::vector<Real> kAbove = {4.01, 5.0, 10.0, 100.0};
const std::vector<Real> kEndpoints = {0.0, 4.0};

std::vector<Real> concat(const std::vector<Real>& a, const std::vector<Real>& b) {
    std::vector<Real> r = a;
    r.insert(r.end(), b.begin(), b.end());
    return r;
}

// --------------------------------------------------------------------------
// Sampling helpers
// --------------------------------------------------------------------------

using Sampler = Real (*)(const Interpolation&, Real);

Real sampleValue(const Interpolation& i, Real x) { return i(x); }
Real sampleDerivative(const Interpolation& i, Real x) { return i.derivative(x); }
Real sampleSecondDerivative(const Interpolation& i, Real x) { return i.secondDerivative(x); }
Real samplePrimitive(const Interpolation& i, Real x) { return i.primitive(x); }

std::vector<Real> sampleAt(const Interpolation& interp,
                           const std::vector<Real>& xs,
                           Sampler s) {
    std::vector<Real> r;
    r.reserve(xs.size());
    for (Real x : xs)
        r.push_back(s(interp, x));
    return r;
}

void emitSampled(const std::string& prefix,
                 const Interpolation& interp,
                 const std::vector<Real>& xs,
                 Sampler s) {
    emit_arr(prefix + "_x", xs);
    emit_arr(prefix + "_values", sampleAt(interp, xs, s));
}

void emitAccessors(const std::string& prefix, const Interpolation& interp) {
    const std::vector<Real> xv = interp.xValues();
    emit((prefix + "_x_min").c_str(), interp.xMin());
    emit((prefix + "_x_max").c_str(), interp.xMax());
    emit_arr(prefix + "_x_values", xv);
    emit_arr(prefix + "_y_values", interp.yValues());
    emit_int((prefix + "_size").c_str(), static_cast<long>(xv.size()));
    emit_bool((prefix + "_empty").c_str(), interp.empty());
    emit_bool((prefix + "_allows_extrapolation").c_str(), interp.allowsExtrapolation());
    emit_bool((prefix + "_is_in_range_x_min").c_str(), interp.isInRange(interp.xMin()));
    emit_bool((prefix + "_is_in_range_x_max").c_str(), interp.isInRange(interp.xMax()));
    emit_bool((prefix + "_is_in_range_2").c_str(), interp.isInRange(2.0));
    emit_bool((prefix + "_is_in_range_minus1").c_str(), interp.isInRange(-1.0));
    emit_bool((prefix + "_is_in_range_5").c_str(), interp.isInRange(5.0));
}

void emitUnderlyingReference(const std::string& prefix, const Interpolation& raw) {
    const std::vector<Real> inRange = concat(kNodes, kMidpoints);
    emit_arr(prefix + "_x", inRange);
    emit_arr(prefix + "_value", sampleAt(raw, inRange, sampleValue));
    emit_arr(prefix + "_derivative", sampleAt(raw, inRange, sampleDerivative));
    emit_arr(prefix + "_second_derivative", sampleAt(raw, inRange, sampleSecondDerivative));
    emit_arr(prefix + "_primitive", sampleAt(raw, inRange, samplePrimitive));
}

// --------------------------------------------------------------------------
// Blocks
// --------------------------------------------------------------------------

void block_natural_cubic() {
    // Natural cubic spline (SecondDerivative 0 at both ends), exactly as in
    // upstream testFlatExtrapolation.
    auto naturalCubic = ext::make_shared<CubicInterpolation>(
        std::begin(kX), std::end(kX), std::begin(kY),
        CubicInterpolation::Spline, false,
        CubicInterpolation::SecondDerivative, 0.0,
        CubicInterpolation::SecondDerivative, 0.0);
    naturalCubic->enableExtrapolation();

    FlatExtrapolator flat(naturalCubic);

    // Extrapolation is NOT allowed by default. The decorator carries its own
    // Extrapolator flag; the decorated object already has extrapolation
    // enabled and that does not propagate.
    {
        bool valueBelowThrows = false, valueAboveThrows = false;
        bool derivativeBelowThrows = false, secondDerivativeAboveThrows = false;
        bool primitiveBelowThrows = false, valueInRangeThrows = false;

        try { flat(-1.0); } catch (const std::exception&) { valueBelowThrows = true; }
        try { flat(5.0); } catch (const std::exception&) { valueAboveThrows = true; }
        try { flat.derivative(-1.0); } catch (const std::exception&) { derivativeBelowThrows = true; }
        try { flat.secondDerivative(5.0); } catch (const std::exception&) { secondDerivativeAboveThrows = true; }
        try { flat.primitive(-1.0); } catch (const std::exception&) { primitiveBelowThrows = true; }
        try { flat(2.0); } catch (const std::exception&) { valueInRangeThrows = true; }

        emit_bool("cubic_default_decorator_allows_extrapolation", flat.allowsExtrapolation());
        emit_bool("cubic_default_decorated_allows_extrapolation",
                  naturalCubic->allowsExtrapolation());
        emit_bool("cubic_default_value_below_throws", valueBelowThrows);
        emit_bool("cubic_default_value_above_throws", valueAboveThrows);
        emit_bool("cubic_default_derivative_below_throws", derivativeBelowThrows);
        emit_bool("cubic_default_second_derivative_above_throws", secondDerivativeAboveThrows);
        emit_bool("cubic_default_primitive_below_throws", primitiveBelowThrows);
        emit_bool("cubic_default_value_in_range_throws", valueInRangeThrows);
    }

    flat.enableExtrapolation();

    // value
    emitSampled("cubic_value_at_nodes", flat, kNodes, sampleValue);
    emitSampled("cubic_value_in_range_midpoints", flat, kMidpoints, sampleValue);
    emitSampled("cubic_value_below_range", flat, kBelow, sampleValue);
    emitSampled("cubic_value_above_range", flat, kAbove, sampleValue);
    emitSampled("cubic_value_at_endpoints", flat, kEndpoints, sampleValue);

    // derivative — strict boundary: at exactly xMin/xMax the underlying slope
    // is used, NOT the flat 0.
    emitSampled("cubic_derivative_in_range", flat, kMidpoints, sampleDerivative);
    emitSampled("cubic_derivative_at_endpoints", flat, kEndpoints, sampleDerivative);
    emitSampled("cubic_derivative_outside_range", flat, concat(kBelow, kAbove),
                sampleDerivative);

    // second derivative — for a NATURAL spline the endpoint value is 0 by
    // construction, so it does not discriminate the boundary branch; the
    // not-a-knot block below does.
    emitSampled("cubic_second_derivative_in_range", flat, kMidpoints, sampleSecondDerivative);
    emitSampled("cubic_second_derivative_at_endpoints", flat, kEndpoints,
                sampleSecondDerivative);
    emitSampled("cubic_second_derivative_outside_range", flat, concat(kBelow, kAbove),
                sampleSecondDerivative);

    // primitive — linear extension with slope y[0] below and y[N-1] above.
    emitSampled("cubic_primitive_in_range", flat, kMidpoints, samplePrimitive);
    emitSampled("cubic_primitive_at_endpoints", flat, kEndpoints, samplePrimitive);
    emitSampled("cubic_primitive_below_range", flat, kBelow, samplePrimitive);
    emitSampled("cubic_primitive_above_range", flat, kAbove, samplePrimitive);

    // accessors + update
    {
        const Real before = flat(2.0);
        emitAccessors("cubic_accessors", flat);
        flat.update(); // delegates to the decorated interpolation
        emit("cubic_value_at_2_before_update", before);
        emit("cubic_value_at_2_after_update", flat(2.0));
    }

    emitUnderlyingReference("cubic_underlying_spline_reference", *naturalCubic);
}

void block_notaknot_cubic() {
    // NOT-A-KNOT cubic spline. Its second derivative at the endpoints is
    // nonzero, so it discriminates the strict `x < xMin` / `x > xMax`
    // boundary that the natural spline cannot.
    auto notAKnot = ext::make_shared<CubicInterpolation>(
        std::begin(kX), std::end(kX), std::begin(kY),
        CubicInterpolation::Spline, false,
        CubicInterpolation::NotAKnot, Null<Real>(),
        CubicInterpolation::NotAKnot, Null<Real>());
    notAKnot->enableExtrapolation();

    FlatExtrapolator flat(notAKnot);
    flat.enableExtrapolation();

    const std::vector<Real> outside = {-5.0, -1.0, 4.5, 9.0};

    emitSampled("notaknot_value_outside_range", flat, outside, sampleValue);
    emitSampled("notaknot_derivative_at_endpoints", flat, kEndpoints, sampleDerivative);
    // THE discriminating case: nonzero at exactly xMin/xMax, zero just outside.
    emitSampled("notaknot_second_derivative_at_endpoints", flat, kEndpoints,
                sampleSecondDerivative);
    emitSampled("notaknot_second_derivative_outside_range", flat,
                {-1.0, -0.01, 4.01, 5.0}, sampleSecondDerivative);
    emitSampled("notaknot_primitive_outside_range", flat, outside, samplePrimitive);

    emitUnderlyingReference("notaknot_underlying_spline_reference", *notAKnot);
}

void block_linear() {
    // Linear interpolation underlying. Independent of the spline solver, so
    // the decorator's own algebra is verifiable by hand:
    //   value below     = y[0] = 1.0,   value above = y[4] = 3.5
    //   primitive below = P(x0) + y[0]*(x - x0) = 0 + 1.0*x
    //   primitive above = P(x4) + y[4]*(x - x4)
    auto linear = ext::make_shared<LinearInterpolation>(
        std::begin(kXL), std::end(kXL), std::begin(kYL));
    linear->enableExtrapolation();

    FlatExtrapolator flat(linear);
    flat.enableExtrapolation();

    const std::vector<Real> xs =
        concat(concat(concat(kBelow, kNodes), kMidpoints), kAbove);

    emitSampled("linear_value_below_in_above_range", flat, xs, sampleValue);
    emitSampled("linear_derivative", flat, xs, sampleDerivative);
    emitSampled("linear_second_derivative", flat, xs, sampleSecondDerivative);
    emitSampled("linear_primitive", flat, xs, samplePrimitive);

    emitAccessors("linear_accessors", flat);
    emitUnderlyingReference("linear_underlying_reference", *linear);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    block_natural_cubic();
    block_notaknot_cubic();
    block_linear();
    std::cout << "\n}" << std::endl;
    return 0;
}
