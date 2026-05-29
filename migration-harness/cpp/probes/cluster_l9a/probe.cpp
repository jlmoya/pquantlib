// L9-A cluster probe: cubic + bicubic spline interpolators.
//
// Captures reference values for:
//
//   * CubicNaturalSpline(x, y) — Spline + Natural BC + non-monotonic
//     - value at each pillar node (must equal y[i] exactly)
//     - value at four intermediate x's
//     - derivative + second_derivative at one interior point
//     - primitive at one interior point
//
//   * MonotonicCubicNaturalSpline(x, y) — PCHIP-equivalent
//     - value at each pillar node (must equal y[i])
//     - value at four intermediate x's
//
//   * BicubicSpline(x, y, z) — 2D bicubic (RectBivariateSpline kx=ky=3)
//     - value at each (xi, yj) grid pillar
//     - value at four intermediate grid points
//
// C++ parity:
//   ql/math/interpolations/cubicinterpolation.hpp
//     ``CubicInterpolation`` (Spline + Natural BC + monotonic toggle).
//   ql/math/interpolations/bicubicsplineinterpolation.hpp
//     ``BicubicSpline``.
//   @ v1.42.1 (099987f0).

#include <ql/math/interpolations/bicubicsplineinterpolation.hpp>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/matrix.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) CubicNaturalSpline (Spline + Natural BC + non-monotonic).
    // ============================================================
    {
        std::vector<Real> xs = {0.0, 1.0, 2.0, 3.0, 4.0};
        std::vector<Real> ys = {0.0, 1.0, 0.5, 1.5, 1.0};
        CubicNaturalSpline interp(xs.begin(), xs.end(), ys.begin());
        interp.update();

        std::cout << "  \"cubic_natural_spline\": {\n";
        std::cout << "    \"xs\": [0.0, 1.0, 2.0, 3.0, 4.0],\n";
        std::cout << "    \"ys\": [0.0, 1.0, 0.5, 1.5, 1.0],\n";

        // Pillar values — interpolation must be EXACT at nodes.
        std::cout << "    \"pillars\": [\n";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::cout << "      " << interp(xs[i]);
            if (i + 1 < xs.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        // Intermediate values.
        std::vector<Real> mids = {0.5, 1.25, 2.7, 3.4};
        std::cout << "    \"mids_x\": [0.5, 1.25, 2.7, 3.4],\n";
        std::cout << "    \"mids_y\": [\n";
        for (std::size_t i = 0; i < mids.size(); ++i) {
            std::cout << "      " << interp(mids[i]);
            if (i + 1 < mids.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        std::cout << "    \"derivative_at_1_5\": " << interp.derivative(1.5) << ",\n";
        std::cout << "    \"second_derivative_at_1_5\": " << interp.secondDerivative(1.5) << ",\n";
        std::cout << "    \"primitive_at_2_5\": " << interp.primitive(2.5) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) MonotonicCubicNaturalSpline (PCHIP-equivalent).
    // ============================================================
    {
        std::vector<Real> xs = {0.0, 1.0, 2.0, 3.0, 4.0};
        // Strictly monotone-increasing ys → monotonic spline should preserve.
        std::vector<Real> ys = {0.0, 0.5, 1.5, 3.0, 3.2};
        MonotonicCubicNaturalSpline interp(xs.begin(), xs.end(), ys.begin());
        interp.update();

        std::cout << "  \"monotonic_cubic_natural_spline\": {\n";
        std::cout << "    \"xs\": [0.0, 1.0, 2.0, 3.0, 4.0],\n";
        std::cout << "    \"ys\": [0.0, 0.5, 1.5, 3.0, 3.2],\n";

        std::cout << "    \"pillars\": [\n";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::cout << "      " << interp(xs[i]);
            if (i + 1 < xs.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        std::vector<Real> mids = {0.5, 1.25, 2.7, 3.4};
        std::cout << "    \"mids_x\": [0.5, 1.25, 2.7, 3.4],\n";
        std::cout << "    \"mids_y\": [\n";
        for (std::size_t i = 0; i < mids.size(); ++i) {
            std::cout << "      " << interp(mids[i]);
            if (i + 1 < mids.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) BicubicSpline on a 4x4 grid.
    // ============================================================
    {
        std::vector<Real> xs = {0.0, 1.0, 2.0, 3.0};
        std::vector<Real> ys = {0.0, 1.0, 2.0, 3.0};
        // z[y_idx][x_idx]; choose a smooth f(x, y) = sin(x) + cos(y) so the
        // spline can roundtrip. We probe at exact grid points and four
        // off-grid points.
        Matrix z(4, 4);
        // Manually-precomputed sin(x) + cos(y) values
        // (we let the probe execute std::sin / std::cos to match the
        // exact double bit pattern emitted by glibc-equivalents).
        for (std::size_t j = 0; j < ys.size(); ++j) {
            for (std::size_t i = 0; i < xs.size(); ++i) {
                z[j][i] = std::sin(xs[i]) + std::cos(ys[j]);
            }
        }
        BicubicSpline interp(xs.begin(), xs.end(), ys.begin(), ys.end(), z);
        interp.update();

        std::cout << "  \"bicubic_spline\": {\n";
        std::cout << "    \"xs\": [0.0, 1.0, 2.0, 3.0],\n";
        std::cout << "    \"ys\": [0.0, 1.0, 2.0, 3.0],\n";
        std::cout << "    \"z\": [\n";
        for (std::size_t j = 0; j < ys.size(); ++j) {
            std::cout << "      [";
            for (std::size_t i = 0; i < xs.size(); ++i) {
                std::cout << z[j][i];
                if (i + 1 < xs.size()) std::cout << ", ";
            }
            std::cout << "]";
            if (j + 1 < ys.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        // Pillars — must roundtrip.
        std::cout << "    \"pillars\": [\n";
        for (std::size_t j = 0; j < ys.size(); ++j) {
            std::cout << "      [";
            for (std::size_t i = 0; i < xs.size(); ++i) {
                std::cout << interp(xs[i], ys[j]);
                if (i + 1 < xs.size()) std::cout << ", ";
            }
            std::cout << "]";
            if (j + 1 < ys.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        // Intermediate points: (x, y) ∈ off-grid.
        std::vector<std::pair<Real, Real>> mids = {
            {0.5, 0.5}, {1.5, 2.25}, {2.7, 1.1}, {0.25, 2.75}};
        std::cout << "    \"mids_xy\": [[0.5, 0.5], [1.5, 2.25], [2.7, 1.1], [0.25, 2.75]],\n";
        std::cout << "    \"mids_z\": [\n";
        for (std::size_t i = 0; i < mids.size(); ++i) {
            std::cout << "      " << interp(mids[i].first, mids[i].second);
            if (i + 1 < mids.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
