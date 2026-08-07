// migration-harness/cpp/probes/v143_experimental_interp/probe.cpp
//
// Reference values for the two ql/experimental interpolation pimpls @ v1.43:
//
//   detail::LinearFlatInterpolationImpl   experimental/shortrate/generalizedhullwhite.hpp:341
//     ...reached only through the LinearFlatInterpolation facade (line 313)
//     and the LinearFlat factory/traits class (line 328).
//
//   detail::VannaVolgaInterpolationImpl   experimental/barrieroption/vannavolgainterpolation.hpp:82
//     ...reached only through the VannaVolgaInterpolation facade (line 39)
//     and the VannaVolga factory (line 58).
//
// WHY THIS PROBE EXISTS
// ---------------------
// Both Impl classes were "ported by name" but their expected values had never
// been produced by running C++ — the LinearFlat test carried hand-computed
// constants and VannaVolga was only exercised transitively through the two
// barrier engines. A hand-computed constant proves the porter's arithmetic,
// not QuantLib's. This probe pins both directly.
//
// WHAT IS PINNED, AND WHY EACH CASE IS THERE
// ------------------------------------------
// LinearFlatInterpolationImpl:
//   * value() on both flat branches (x <= xMin, x >= xMax) AND exactly at the
//     boundary nodes, because the C++ comparisons are `<=` / `>=`, so the node
//     itself takes the flat branch, not the locate() branch. A port using
//     `<` / `>` gives the same answer at the nodes only because the flat value
//     equals the node value there — but not for primitive/derivative.
//   * derivative() outside the range returns 0 (isInRange guard) while
//     primitive() has NO guard at all: it calls locate() directly, so for
//     x outside the range C++ returns the clamped-interval extrapolated
//     primitive rather than a clamped constant. That asymmetry is pinned.
//   * a non-uniform x grid with a sign change in the slope, so a port cannot
//     pass with a wrong segment index.
//   * the degenerate single-node case (requiredPoints == 1).
//   * an unsorted-y / steep grid so primitiveConst_ accumulation is exercised
//     over 5 segments.
//
// VannaVolgaInterpolationImpl:
//   * value() at the three pillars themselves. At a pillar k == x_i the
//     three weights x1/x2/x3 do NOT reduce to (1,0,0)/(0,1,0)/(0,0,1) by
//     inspection — log(x_i/k) == 0 makes two of the three numerators vanish,
//     and the surviving one gives exactly vega(k)/vegas[i], so the reproduced
//     price is exactly premiaMKT[i] and the implied vol round-trips to
//     y[i] up to the Brent solver accuracy of blackFormulaImpliedStdDev
//     (1e-6 on the std dev, i.e. ~1e-6/sqrt(T) on the vol). Pinning the
//     pillars catches a port that mis-orders premiaMKT/premiaBS.
//   * strikes between and outside the pillars, including deep wings where
//     the reproduced call price c can approach the no-arbitrage bounds.
//   * a second market with a pronounced skew and a different maturity so
//     the sqrt(T) scaling and the fwd = spot*fDiscount/dDiscount convention
//     are both exercised (dDiscount != fDiscount).
//   * the intermediate quantities (fwd, atmVol, premiaBS, premiaMKT, vegas)
//     so a failing port can be bisected instead of merely reported red.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/interp.json.

#include <ql/experimental/barrieroption/vannavolgainterpolation.hpp>
#include <ql/experimental/shortrate/generalizedhullwhite.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/pricingengines/blackformula.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

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

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

// --------------------------------------------------------------------------
// A. LinearFlatInterpolation
// --------------------------------------------------------------------------

void emitLinearFlat(const std::string& tag,
                    const std::vector<Real>& x,
                    const std::vector<Real>& y,
                    const std::vector<Real>& queries) {
    LinearFlatInterpolation f(x.begin(), x.end(), y.begin());
    f.enableExtrapolation();

    emit_arr(tag + "_x", x);
    emit_arr(tag + "_y", y);
    emit_arr(tag + "_q", queries);

    std::vector<Real> values, primitives, derivatives, second;
    for (Real q : queries) {
        values.push_back(f(q, true));
        primitives.push_back(f.primitive(q, true));
        derivatives.push_back(f.derivative(q, true));
        second.push_back(f.secondDerivative(q, true));
    }
    emit_arr(tag + "_value", values);
    emit_arr(tag + "_primitive", primitives);
    emit_arr(tag + "_derivative", derivatives);
    emit_arr(tag + "_second_derivative", second);

    emit(tag + "_xMin", f.xMin());
    emit(tag + "_xMax", f.xMax());
}

void sectionLinearFlat() {
    // A1: the grid the existing (hand-computed) Python test used, so the
    // migration can prove the old constants were right as well as replacing
    // their provenance.
    {
        const std::vector<Real> x = {0.0, 1.0, 3.0};
        const std::vector<Real> y = {10.0, 12.0, 8.0};
        const std::vector<Real> q = {-5.0, -1e-12, 0.0, 1e-12, 0.25, 0.5,
                                     0.999999, 1.0, 1.5, 2.0, 2.75, 3.0,
                                     3.0 + 1e-12, 7.5, 100.0};
        emitLinearFlat("lf_a", x, y, q);
    }
    // A2: non-uniform grid, five segments, slope sign flips twice, negative
    // y values so the primitive accumulation can go down as well as up.
    {
        const std::vector<Real> x = {-2.0, -0.5, 0.25, 1.0, 4.0, 9.5};
        const std::vector<Real> y = {3.5, -1.25, -1.25, 6.0, 0.5, -4.75};
        const std::vector<Real> q = {-10.0, -2.0, -1.75, -0.5, -0.125, 0.25,
                                     0.6, 1.0, 2.5, 4.0, 7.0, 9.5, 20.0};
        emitLinearFlat("lf_b", x, y, q);
    }
    // A3: the degenerate two-node constant grid (requiredPoints == 1 means
    // C++ accepts even a single node; Interpolation::templateImpl asserts
    // xEnd-xBegin >= requiredPoints).
    {
        const std::vector<Real> x = {2.0, 3.0};
        const std::vector<Real> y = {5.0, 5.0};
        const std::vector<Real> q = {0.0, 2.0, 2.5, 3.0, 10.0};
        emitLinearFlat("lf_c", x, y, q);
    }
    // A4: two-node grid with distinct y, i.e. pure linear inside plus flat
    // outside. The smallest grid for which every accessor is well defined.
    //
    // NOT PINNED, deliberately: the genuinely single-node grid (n == 1),
    // which LinearFlat::requiredPoints == 1 nominally admits. C++
    // Interpolation::templateImpl::locate (ql/math/interpolation.hpp:104-114)
    // returns `xEnd_-xBegin_-2`, i.e. Size(-1), for any x >= xMin when
    // n == 1, so LinearFlatInterpolationImpl::primitive and ::derivative
    // index primitiveConst_[Size(-1)] / s_[Size(-1)] — an out-of-bounds
    // read. A first draft of this probe pinned that case and the values it
    // printed changed between consecutive runs of the same binary
    // (2.1627069385209719e-314 / 2.1442184645101697e-314 / 0.0). Only
    // ::value is well defined for n == 1 (it short-circuits on the two flat
    // branches before ever calling locate). No reference value can be
    // derived from undefined behaviour, so n == 1 is out of scope for both
    // the probe and the port's tests.
    {
        const std::vector<Real> x = {2.0, 3.0};
        const std::vector<Real> y = {5.0, -1.0};
        const std::vector<Real> q = {0.0, 2.0, 2.25, 2.5, 3.0, 10.0};
        emitLinearFlat("lf_d", x, y, q);
    }

    // A5: traits constants on the factory.
    emit_int("lf_required_points", (long long)LinearFlat::requiredPoints);
    emit_int("lf_global", LinearFlat::global ? 1 : 0);

    // A6: the factory really does build the same interpolation.
    {
        const std::vector<Real> x = {0.0, 1.0, 3.0};
        const std::vector<Real> y = {10.0, 12.0, 8.0};
        LinearFlat factory;
        Interpolation f = factory.interpolate(x.begin(), x.end(), y.begin());
        f.enableExtrapolation();
        std::vector<Real> v;
        for (Real q : {-1.0, 0.5, 2.0, 5.0})
            v.push_back(f(q, true));
        emit_arr("lf_factory_value", v);
    }
}

// --------------------------------------------------------------------------
// B. VannaVolgaInterpolation
// --------------------------------------------------------------------------

// Mirrors the private VannaVolgaInterpolationImpl::vega so the intermediate
// quantities can be pinned without touching the private member.
Real vvVega(Real k, Real spot, Real fwd, Real atmVol, Real dDiscount, Real T) {
    Real d1 = (std::log(fwd / k) + 0.5 * std::pow(atmVol, 2.0) * T) / (atmVol * std::sqrt(T));
    NormalDistribution norm;
    return spot * dDiscount * std::sqrt(T) * norm(d1);
}

void emitVannaVolga(const std::string& tag,
                    const std::vector<Real>& strikes,
                    const std::vector<Real>& vols,
                    Real spot,
                    DiscountFactor dDiscount,
                    DiscountFactor fDiscount,
                    Time T,
                    const std::vector<Real>& queries) {
    VannaVolgaInterpolation f(strikes.begin(), strikes.end(), vols.begin(),
                              spot, dDiscount, fDiscount, T);
    f.enableExtrapolation();

    emit_arr(tag + "_strikes", strikes);
    emit_arr(tag + "_vols", vols);
    emit(tag + "_spot", spot);
    emit(tag + "_dDiscount", dDiscount);
    emit(tag + "_fDiscount", fDiscount);
    emit(tag + "_T", T);
    emit_arr(tag + "_q", queries);

    const Real fwd = spot * fDiscount / dDiscount;
    const Real atmVol = vols[1];
    emit(tag + "_fwd", fwd);
    emit(tag + "_atmVol", atmVol);

    std::vector<Real> premiaBS, premiaMKT, vegas;
    for (Size i = 0; i < 3; ++i) {
        premiaBS.push_back(blackFormula(Option::Call, strikes[i], fwd,
                                        atmVol * std::sqrt(T), dDiscount));
        premiaMKT.push_back(blackFormula(Option::Call, strikes[i], fwd,
                                         vols[i] * std::sqrt(T), dDiscount));
        vegas.push_back(vvVega(strikes[i], spot, fwd, atmVol, dDiscount, T));
    }
    emit_arr(tag + "_premiaBS", premiaBS);
    emit_arr(tag + "_premiaMKT", premiaMKT);
    emit_arr(tag + "_vegas", vegas);

    std::vector<Real> values;
    for (Real q : queries)
        values.push_back(f(q, true));
    emit_arr(tag + "_value", values);
}

void sectionVannaVolga() {
    // B1: EURUSD-shaped smile, 1y, mild skew. Domestic and foreign discount
    // factors deliberately differ so fwd != spot.
    {
        const std::vector<Real> k = {1.28, 1.35, 1.42};
        const std::vector<Real> v = {0.1250, 0.1100, 0.1180};
        emitVannaVolga("vv_a", k, v,
                       /*spot*/ 1.35, /*dDiscount*/ 0.98019867330675525,
                       /*fDiscount*/ 0.99004983374916811, /*T*/ 1.0,
                       {1.20, 1.25, 1.28, 1.30, 1.32, 1.35, 1.38, 1.40, 1.42,
                        1.46, 1.52});
    }
    // B2: pronounced skew, short maturity (3m), tighter strike spacing.
    // Short T stresses the sqrt(T) division on the returned vol.
    {
        const std::vector<Real> k = {0.94, 1.00, 1.06};
        const std::vector<Real> v = {0.2400, 0.1800, 0.1950};
        emitVannaVolga("vv_b", k, v,
                       /*spot*/ 1.00, /*dDiscount*/ 0.99501247919268232,
                       /*fDiscount*/ 0.99750312239746478, /*T*/ 0.25,
                       {0.88, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06,
                        1.10});
    }
    // B3: long-dated (5y) low-vol market, dDiscount << fDiscount so the
    // forward sits well above spot and the smile is queried far ITM/OTM.
    {
        const std::vector<Real> k = {80.0, 100.0, 130.0};
        const std::vector<Real> v = {0.2200, 0.2000, 0.2150};
        emitVannaVolga("vv_c", k, v,
                       /*spot*/ 100.0, /*dDiscount*/ 0.81873075307798182,
                       /*fDiscount*/ 0.95122942450071402, /*T*/ 5.0,
                       {70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 145.0});
    }

    // B4: traits constant + factory round-trip.
    emit_int("vv_required_points", (long long)VannaVolga::requiredPoints);
    {
        const std::vector<Real> k = {1.28, 1.35, 1.42};
        const std::vector<Real> v = {0.1250, 0.1100, 0.1180};
        VannaVolga factory(1.35, 0.98019867330675525, 0.99004983374916811, 1.0);
        Interpolation f = factory.interpolate(k.begin(), k.end(), v.begin());
        f.enableExtrapolation();
        std::vector<Real> out;
        for (Real q : {1.30, 1.35, 1.40})
            out.push_back(f(q, true));
        emit_arr("vv_factory_value", out);
    }
}

}  // namespace

int main() {
    std::cout << "{\n";
    sectionLinearFlat();
    sectionVannaVolga();
    std::cout << "\n}\n";
    return 0;
}
