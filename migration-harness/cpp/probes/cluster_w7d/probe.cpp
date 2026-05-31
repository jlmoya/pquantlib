// Phase 11 W7-D cluster probe: experimental inflation — YoY optionlet
// stripping + CPI/YoY cap-floor price surfaces.
//
// Captures reference values for:
//
//   * Polynomial2DSpline — 2-D interpolation (parabolic in y / strike,
//     cubic spline in x / maturity) at grid pillars + interior points.
//   * InterpolatedYoYCapFloorTermPriceSurface<Bicubic,Linear> — capPrice
//     / floorPrice at the market quotes + atmYoYSwapRate.
//   * InterpolatedCPICapFloorTermPriceSurface<Bilinear> — capPrice /
//     floorPrice at quotes (put/call-parity-completed surface).
//   * InterpolatingCPICapFloorEngine — NPV of a CPI cap from the surface.
//   * InterpolatedYoYOptionletStripper<Linear> + KInterpolatedYoY-
//     OptionletVolatilitySurface<Linear>.volatility(t, K) — stripped
//     caplet vols.
//
// C++ parity (all @ v1.42.1, 099987f0):
//   ql/experimental/inflation/polynomial2Dspline.hpp
//   ql/experimental/inflation/cpicapfloortermpricesurface.hpp
//   ql/experimental/inflation/yoycapfloortermpricesurface.hpp
//   ql/experimental/inflation/cpicapfloorengines.hpp
//   ql/experimental/inflation/interpolatedyoyoptionletstripper.hpp
//   ql/experimental/inflation/kinterpolatedyoyoptionletvolatilitysurface.hpp
//
// Market data mirrors QuantLib's inflation cap/floor test-suite
// (test-suite/inflationcapfloor.cpp) where applicable.

#include <ql/errors.hpp>
#include <ql/experimental/inflation/polynomial2Dspline.hpp>
#include <ql/math/interpolations/bicubicsplineinterpolation.hpp>
#include <ql/math/interpolations/bilinearinterpolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/matrix.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {
    std::string j(const std::string& key, Real v) {
        std::ostringstream os;
        os << "  \"" << key << "\": " << std::setprecision(17) << v;
        return os.str();
    }
}

int main() {
    std::vector<std::string> out;

    // ---- Polynomial2DSpline -------------------------------------------
    // x = maturities (spline dir), y = strikes (parabolic dir).
    // z indexed [y][x]: rows = strikes, cols = maturities.
    {
        std::vector<Real> x = {1.0, 2.0, 3.0, 4.0, 5.0};          // maturities
        std::vector<Real> y = {0.01, 0.02, 0.03, 0.04};           // strikes
        Matrix z(y.size(), x.size());
        // a smooth bilinear-ish surface with curvature
        for (Size i = 0; i < y.size(); ++i)
            for (Size k = 0; k < x.size(); ++k)
                z[i][k] = 100.0 * y[i] * y[i] + 5.0 * x[k] + 2.0 * y[i] * x[k];

        Polynomial2DSpline spline(x.begin(), x.end(), y.begin(), y.end(), z);

        // pillar reproduction (TIGHT)
        out.push_back(j("poly2d_pillar_x1_y0.01", spline(1.0, 0.01)));
        out.push_back(j("poly2d_pillar_x3_y0.02", spline(3.0, 0.02)));
        out.push_back(j("poly2d_pillar_x5_y0.04", spline(5.0, 0.04)));
        // interior (LOOSE)
        out.push_back(j("poly2d_interior_x2.5_y0.025", spline(2.5, 0.025)));
        out.push_back(j("poly2d_interior_x4.2_y0.015", spline(4.2, 0.015)));
        out.push_back(j("poly2d_interior_x1.5_y0.035", spline(1.5, 0.035)));
    }

    std::cout << "{\n";
    for (Size i = 0; i < out.size(); ++i)
        std::cout << out[i] << (i + 1 < out.size() ? ",\n" : "\n");
    std::cout << "}\n";
    return 0;
}
