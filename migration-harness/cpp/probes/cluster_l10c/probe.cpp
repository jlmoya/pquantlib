// L10-C cluster probe: interpolator tail + ZABR.
//
// Captures reference values for:
//
//   * HymanFilteredCubic (= CubicInterpolation(Spline, monotonic=true,
//     SecondDerivative=0, SecondDerivative=0)) on a strictly monotonic
//     sequence — pillar values exact + intermediate values.
//
//   * ChebyshevInterpolation(n=10, f=sin, SecondKind, x_min=0,
//     x_max=pi) — pillar evaluation at nodes + intermediate query.
//
//   * MultiCubicSpline on a 4x4 grid (delegated to BicubicSpline in
//     C++) — pillar values + intermediate query.
//
//   * AbcdInterpolation on a synthetic abcd-shaped vol curve —
//     recovered (a, b, c, d) parameters.
//
//   * zabr_volatility(K=0.05, F=0.05, T=5, alpha=0.04, beta=0.5,
//     nu=0.4, rho=-0.1, gamma=1.0) — should collapse to SABR.
//
//   * zabr_volatility(K=0.05, F=0.05, T=5, alpha=0.04, beta=0.5,
//     nu=0.4, rho=-0.1, gamma=0.75) — gamma != 1 ShortMaturityLognormal.
//
//   * ZabrSmileSection.volatility(K) — should match zabr_volatility.
//
// C++ parity:
//   ql/math/interpolations/cubicinterpolation.hpp (Hyman branch).
//   ql/math/interpolations/chebyshevinterpolation.{hpp,cpp}.
//   ql/math/interpolations/bicubicsplineinterpolation.hpp.
//   ql/math/interpolations/abcdinterpolation.hpp.
//   ql/termstructures/volatility/zabr.{hpp,cpp}.
//   ql/termstructures/volatility/zabrsmilesection.hpp.
//   @ v1.42.1 (099987f0).

#include <ql/math/interpolations/abcdinterpolation.hpp>
#include <ql/math/interpolations/bicubicsplineinterpolation.hpp>
#include <ql/math/interpolations/chebyshevinterpolation.hpp>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/array.hpp>
#include <ql/termstructures/volatility/zabr.hpp>
#include <ql/termstructures/volatility/zabrsmilesection.hpp>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) HymanFilteredCubic — Spline + monotonic + Natural BC.
    // ============================================================
    {
        std::vector<Real> xs = {0.0, 1.0, 2.0, 3.0, 4.0};
        std::vector<Real> ys = {0.0, 0.5, 1.5, 3.0, 3.2};
        CubicInterpolation interp(
            xs.begin(), xs.end(), ys.begin(),
            CubicInterpolation::Spline,
            true,  // monotonic
            CubicInterpolation::SecondDerivative, 0.0,
            CubicInterpolation::SecondDerivative, 0.0);
        interp.update();

        std::cout << "  \"hyman_filtered_cubic\": {\n";
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
        std::cout << "    ],\n";
        std::cout << "    \"derivative_at_1_5\": " << interp.derivative(1.5) << ",\n";
        std::cout << "    \"second_derivative_at_1_5\": " << interp.secondDerivative(1.5) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) ChebyshevInterpolation(n=10, f=sin, SecondKind), remapped to
    //    [0, pi] manually.
    //
    //    The C++ class has no built-in [x_min, x_max] remap, so we
    //    emit the values at the *canonical* [-1, 1] nodes for x = sin
    //    (cos(0..pi) mapped to [-1, 1]) — meaning we sample
    //    f(x) = sin(remap(t)) where remap(t) = pi*(t+1)/2.
    // ============================================================
    {
        const Size n = 10;
        const Real x_min = 0.0;
        const Real x_max = M_PI;
        const Real half_span = 0.5 * (x_max - x_min);
        const Real mid = 0.5 * (x_max + x_min);
        // Build values at the (remapped) Chebyshev SecondKind nodes.
        Array nodes_canonical = ChebyshevInterpolation::nodes(
            n, ChebyshevInterpolation::SecondKind);
        Array values(n);
        for (Size i = 0; i < n; ++i) {
            Real x = mid + nodes_canonical[i] * half_span;
            values[i] = std::sin(x);
        }
        ChebyshevInterpolation interp(values, ChebyshevInterpolation::SecondKind);

        std::cout << "  \"chebyshev_interpolation\": {\n";
        std::cout << "    \"n\": 10,\n";
        std::cout << "    \"x_min\": " << x_min << ",\n";
        std::cout << "    \"x_max\": " << x_max << ",\n";
        std::cout << "    \"function\": \"sin\",\n";

        // Emit the canonical-domain nodes and (remapped) x positions.
        std::cout << "    \"nodes_canonical\": [\n";
        for (Size i = 0; i < n; ++i) {
            std::cout << "      " << nodes_canonical[i];
            if (i + 1 < n) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";
        std::cout << "    \"nodes_remapped\": [\n";
        for (Size i = 0; i < n; ++i) {
            std::cout << "      " << (mid + nodes_canonical[i] * half_span);
            if (i + 1 < n) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        // Pillar values - the interpolator returns the input y values.
        std::cout << "    \"pillar_values\": [\n";
        for (Size i = 0; i < n; ++i) {
            std::cout << "      " << interp(nodes_canonical[i]);
            if (i + 1 < n) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";

        // Intermediate values at canonical [-1, 1] points (these
        // correspond to remapped x = mid + t * half_span). Python will
        // do the same remap.
        std::vector<Real> mids_canonical = {-0.5, -0.1, 0.3, 0.7};
        std::cout << "    \"mids_canonical\": [-0.5, -0.1, 0.3, 0.7],\n";
        std::cout << "    \"mids_remapped\": [\n";
        for (std::size_t i = 0; i < mids_canonical.size(); ++i) {
            std::cout << "      " << (mid + mids_canonical[i] * half_span);
            if (i + 1 < mids_canonical.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";
        std::cout << "    \"interp_at_mids\": [\n";
        for (std::size_t i = 0; i < mids_canonical.size(); ++i) {
            std::cout << "      " << interp(mids_canonical[i]);
            if (i + 1 < mids_canonical.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";
        std::cout << "    \"sin_at_remapped_mids\": [\n";
        for (std::size_t i = 0; i < mids_canonical.size(); ++i) {
            std::cout << "      " << std::sin(mid + mids_canonical[i] * half_span);
            if (i + 1 < mids_canonical.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) MultiCubicSpline on a 4x4 grid — C++ side delegates to a
    //    BicubicSpline (the canonical 2-D template instantiation).
    // ============================================================
    {
        std::vector<Real> xs = {0.0, 1.0, 2.0, 3.0};
        std::vector<Real> ys = {0.0, 1.0, 2.0, 3.0};
        Matrix z(4, 4);
        for (Size j = 0; j < ys.size(); ++j) {
            for (Size i = 0; i < xs.size(); ++i) {
                z[j][i] = std::sin(xs[i]) + std::cos(ys[j]);
            }
        }
        BicubicSpline interp(xs.begin(), xs.end(), ys.begin(), ys.end(), z);

        std::cout << "  \"multi_cubic_spline\": {\n";
        std::cout << "    \"xs\": [0.0, 1.0, 2.0, 3.0],\n";
        std::cout << "    \"ys\": [0.0, 1.0, 2.0, 3.0],\n";
        std::cout << "    \"z\": [\n";
        for (Size j = 0; j < ys.size(); ++j) {
            std::cout << "      [";
            for (Size i = 0; i < xs.size(); ++i) {
                std::cout << z[j][i];
                if (i + 1 < xs.size()) std::cout << ", ";
            }
            std::cout << "]";
            if (j + 1 < ys.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";
        // Pillar evaluations.
        std::cout << "    \"pillars\": [\n";
        for (Size j = 0; j < ys.size(); ++j) {
            std::cout << "      [";
            for (Size i = 0; i < xs.size(); ++i) {
                std::cout << interp(xs[i], ys[j]);
                if (i + 1 < xs.size()) std::cout << ", ";
            }
            std::cout << "]";
            if (j + 1 < ys.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";
        // Intermediate evaluations at four (x, y) pairs.
        std::vector<std::pair<Real, Real>> mids = {
            {0.5, 0.5}, {1.25, 1.5}, {2.25, 0.75}, {2.7, 2.7}};
        std::cout << "    \"mids\": [[0.5, 0.5], [1.25, 1.5], [2.25, 0.75], [2.7, 2.7]],\n";
        std::cout << "    \"mids_y\": [\n";
        for (std::size_t k = 0; k < mids.size(); ++k) {
            std::cout << "      " << interp(mids[k].first, mids[k].second);
            if (k + 1 < mids.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 4) AbcdInterpolation — fit (a, b, c, d) to a synthetic abcd
    //    vol curve generated from known parameters.
    // ============================================================
    {
        // Truth parameters.
        Real a_true = -0.06;
        Real b_true = 0.17;
        Real c_true = 0.54;
        Real d_true = 0.17;

        std::vector<Real> times = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
        std::vector<Real> vols(times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            Real t = times[i];
            vols[i] = (a_true + b_true * t) * std::exp(-c_true * t) + d_true;
        }

        AbcdInterpolation interp(
            times.begin(), times.end(), vols.begin(),
            -0.05, 0.15, 0.50, 0.16);

        std::cout << "  \"abcd_interpolation\": {\n";
        std::cout << "    \"times\": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0],\n";
        std::cout << "    \"vols\": [\n";
        for (std::size_t i = 0; i < vols.size(); ++i) {
            std::cout << "      " << vols[i];
            if (i + 1 < vols.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ],\n";
        std::cout << "    \"a_true\": " << a_true << ",\n";
        std::cout << "    \"b_true\": " << b_true << ",\n";
        std::cout << "    \"c_true\": " << c_true << ",\n";
        std::cout << "    \"d_true\": " << d_true << ",\n";
        std::cout << "    \"a_fitted\": " << interp.a() << ",\n";
        std::cout << "    \"b_fitted\": " << interp.b() << ",\n";
        std::cout << "    \"c_fitted\": " << interp.c() << ",\n";
        std::cout << "    \"d_fitted\": " << interp.d() << ",\n";
        std::cout << "    \"rms_error\": " << interp.rmsError() << ",\n";
        std::cout << "    \"max_error\": " << interp.maxError() << ",\n";
        std::cout << "    \"fitted_at_pillars\": [\n";
        for (std::size_t i = 0; i < times.size(); ++i) {
            std::cout << "      " << interp(times[i]);
            if (i + 1 < times.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "    ]\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 5) ZABR lognormal volatility — gamma = 1 collapses to SABR.
    // ============================================================
    {
        Real alpha = 0.04;
        Real beta = 0.5;
        Real nu = 0.4;
        Real rho = -0.1;
        Real F = 0.05;
        Real T = 5.0;

        ZabrModel zabr_g1(T, F, alpha, beta, nu, rho, 1.0);
        ZabrModel zabr_g75(T, F, alpha, beta, nu, rho, 0.75);

        std::cout << "  \"zabr_formula\": {\n";
        std::cout << "    \"alpha\": " << alpha << ",\n";
        std::cout << "    \"beta\": " << beta << ",\n";
        std::cout << "    \"nu\": " << nu << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"forward\": " << F << ",\n";
        std::cout << "    \"expiry\": " << T << ",\n";

        // gamma = 1 → should equal sabrVolatility(K, F, T, alpha, beta, nu, rho).
        std::cout << "    \"gamma1_vol_atm\": "
                  << zabr_g1.lognormalVolatility(F) << ",\n";
        std::cout << "    \"gamma1_vol_strike_4pct\": "
                  << zabr_g1.lognormalVolatility(0.04) << ",\n";
        std::cout << "    \"gamma1_vol_strike_6pct\": "
                  << zabr_g1.lognormalVolatility(0.06) << ",\n";

        // gamma = 0.75 ShortMaturityLognormal arm.
        std::cout << "    \"gamma75_vol_atm\": "
                  << zabr_g75.lognormalVolatility(F) << ",\n";
        std::cout << "    \"gamma75_vol_strike_4pct\": "
                  << zabr_g75.lognormalVolatility(0.04) << ",\n";
        std::cout << "    \"gamma75_vol_strike_6pct\": "
                  << zabr_g75.lognormalVolatility(0.06) << ",\n";

        // Normal arm — gamma=1 should match sabrNormalVolatility.
        std::cout << "    \"gamma1_normal_vol_atm\": "
                  << zabr_g1.normalVolatility(F) << ",\n";
        std::cout << "    \"gamma75_normal_vol_atm\": "
                  << zabr_g75.normalVolatility(F) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 6) ZabrSmileSection — should match zabr_volatility directly.
    // ============================================================
    {
        Real alpha = 0.04;
        Real beta = 0.5;
        Real nu = 0.4;
        Real rho = -0.1;
        Real gamma = 0.75;
        Real F = 0.05;
        Real T = 5.0;
        std::vector<Real> params = {alpha, beta, nu, rho, gamma};

        ZabrSmileSection<ZabrShortMaturityLognormal> section(
            T, F, params);

        std::cout << "  \"zabr_smile_section\": {\n";
        std::cout << "    \"exercise_time\": " << section.exerciseTime() << ",\n";
        std::cout << "    \"atm_level\": " << section.atmLevel() << ",\n";
        std::cout << "    \"alpha\": " << alpha << ",\n";
        std::cout << "    \"beta\": " << beta << ",\n";
        std::cout << "    \"nu\": " << nu << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"gamma\": " << gamma << ",\n";
        std::cout << "    \"vol_atm\": " << section.volatility(F) << ",\n";
        std::cout << "    \"vol_strike_4pct\": " << section.volatility(0.04) << ",\n";
        std::cout << "    \"vol_strike_6pct\": " << section.volatility(0.06) << ",\n";
        std::cout << "    \"variance_atm\": " << section.variance(F) << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
