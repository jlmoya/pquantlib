// migration-harness/cpp/probes/v143_zabr_mesher/probe.cpp
//
// Pins QuantLib::Concentrating1dMesher
// (ql/methods/finitedifferences/meshers/concentrating1dmesher.cpp), which
// ZabrModel::fdPrice and ::fullFdPrice both build their grids from, plus the
// Glued1dMesher composition fullFdPrice uses to splice two of them.
//
// The class has two constructors that share nothing but a name:
//
//  * Single critical point — nodes at
//    cPoint + density*(end-start)*sinh(c1*(1-l) + c2*l). Three things are easy
//    to get wrong and all three are covered:
//      - the density is scaled by (end - start) INSIDE the constructor, so
//        passing an already-scaled density silently changes the clustering;
//      - with requireCPoint the uniform parameter l is first pushed through a
//        piecewise-LINEAR reparameterisation whose knot is
//        u0 = clamp(lround(z0*(size-1)), 1, size-2)/(size-1) — an integer
//        rounding, so a port using round-half-to-even instead of C's
//        round-half-away-from-zero picks a different knot whenever
//        z0*(size-1) lands on a half integer;
//      - the reparameterisation is SKIPPED when the critical point coincides
//        with either endpoint (`c_at_start` / `c_at_end` below), and then
//        the grid is the plain sinh one.
//    `no_cpoint` checks that a Null critical point degenerates to a uniform
//    grid rather than to the sinh map.
//
//  * Several critical points — the node positions solve
//    y'(x) = a/sqrt(sum_i 1/(beta_i + (y - c_i)^2)) with y(0) = start, the
//    scale a being Brent-solved so y(1) = end, integrated with
//    AdaptiveRungeKutta. Two details are pinned by `multi_two_points` /
//    `multi_required`: beta_i is the SQUARE of density_i*(end-start) yet is
//    fed unsquared into asinh when seeding Brent (a quirk of the C++ source
//    that changes the initial guess and therefore the converged a), and the
//    "required" flag triggers a second Brent solve per point plus a
//    de-duplication of the reparameterisation knots with close_enough at 1000
//    epsilons.
//
// The `zabr_fd_grid` and `zabr_fullfd_*` blocks reproduce the exact
// constructor arguments ZabrModel uses, so a mismatch there localises a
// ZabrModel price discrepancy to the mesher rather than to the PDE.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/zabr/mesher.json.

#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include <ql/experimental/finitedifferences/glued1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/concentrating1dmesher.hpp>

using namespace QuantLib;

namespace {

void emitLocations(const char* key, const Fdm1dMesher& m, bool trailingComma) {
    std::cout << "  \"" << key << "\": {\n    \"size\": " << m.size() << ",\n";
    std::cout << "    \"locations\": [";
    for (Size i = 0; i < m.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << m.location(i);
    std::cout << "],\n";
    // dplus/dminus are what the FD operators actually read; emit the interior
    // ones (the two boundary sentinels are Null<Real>, not a number).
    std::cout << "    \"dplus\": [";
    for (Size i = 0; i + 1 < m.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << m.dplus(i);
    std::cout << "],\n";
    std::cout << "    \"dminus\": [";
    for (Size i = 1; i < m.size(); ++i)
        std::cout << (i != 1U ? ", " : "") << m.dminus(i);
    std::cout << "]\n  }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // Plain sinh clustering, no required point.
    emitLocations("cpoint_not_required",
                  Concentrating1dMesher(0.0, 1.0, 21, std::pair<Real, Real>(0.3, 0.1), false),
                  true);
    // Same arguments, requireCPoint on: the linear reparameterisation kicks in
    // and 0.3 becomes an exact node.
    emitLocations("cpoint_required",
                  Concentrating1dMesher(0.0, 1.0, 21, std::pair<Real, Real>(0.3, 0.1), true),
                  true);
    // Critical point ON an endpoint: the reparameterisation is skipped.
    emitLocations("c_at_start",
                  Concentrating1dMesher(0.0, 1.0, 15, std::pair<Real, Real>(0.0, 0.05), true),
                  true);
    emitLocations("c_at_end",
                  Concentrating1dMesher(0.0, 1.0, 15, std::pair<Real, Real>(1.0, 0.05), true),
                  true);
    // Null critical point: uniform grid.
    emitLocations(
        "no_cpoint",
        Concentrating1dMesher(0.0, 1.0, 11, std::pair<Real, Real>(Null<Real>(), Null<Real>()),
                              false),
        true);
    // An even size, so z0*(size-1) can land on a half integer and the lround
    // tie-breaking matters.
    emitLocations("even_size",
                  Concentrating1dMesher(0.0, 1.0, 20, std::pair<Real, Real>(0.5, 0.25), true),
                  true);

    // The exact grid ZabrModel::fdPrice builds for strikes {0.01,...,0.12}:
    // start = min(1e-5, 0.01*0.5), end = max(0.10, 0.12*1.5), 500 points
    // concentrated at forward = 0.03 with density 0.1.
    emitLocations("zabr_fd_grid",
                  Concentrating1dMesher(0.00001, 0.18, 500, std::pair<Real, Real>(0.03, 0.1),
                                        true),
                  true);

    // The two halves ZabrModel::fullFdPrice glues for forward = 0.03,
    // strike = 0.05 (so x0 = 0.03, x1 = 0.05, midpoint 0.04), together with
    // the glued result — the glue must collapse the shared midpoint, otherwise
    // FdmMesherComposite rejects the layout.
    const Concentrating1dMesher mfa(0.00502, 0.04, 55, std::pair<Real, Real>(0.03, 0.1), true);
    const Concentrating1dMesher mfb(0.04, 0.1793, 46, std::pair<Real, Real>(0.05, 0.1), true);
    emitLocations("zabr_fullfd_left", mfa, true);
    emitLocations("zabr_fullfd_right", mfb, true);
    emitLocations("zabr_fullfd_glued", Glued1dMesher(mfa, mfb), true);

    // Multi-critical-point constructor: ODE + Brent, no required points.
    std::vector<std::tuple<Real, Real, bool> > cp2;
    cp2.emplace_back(0.25, 0.05, false);
    cp2.emplace_back(0.75, 0.10, false);
    emitLocations("multi_two_points", Concentrating1dMesher(0.0, 1.0, 21, cp2), true);

    // Same points, but both required: two extra Brent solves plus knot
    // de-duplication.
    std::vector<std::tuple<Real, Real, bool> > cp3;
    cp3.emplace_back(0.25, 0.05, true);
    cp3.emplace_back(0.75, 0.10, true);
    emitLocations("multi_required", Concentrating1dMesher(0.0, 1.0, 21, cp3), false);

    std::cout << "}\n";
    return 0;
}
