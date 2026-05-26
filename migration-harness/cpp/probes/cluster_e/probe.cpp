// L1-E mega-probe: 6 simple interpolations + 5 matrix utilities + 3 math root.

#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/interpolations/loginterpolation.hpp>
#include <ql/math/interpolations/backwardflatinterpolation.hpp>
#include <ql/math/interpolations/forwardflatinterpolation.hpp>
#include <ql/math/interpolations/bilinearinterpolation.hpp>
#include <ql/math/matrixutilities/choleskydecomposition.hpp>
#include <ql/math/matrixutilities/symmetricschurdecomposition.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/array.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- 1-D interpolations on a small known grid ------------------------
    std::vector<Real> xs = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Real> ys = {0.0, 1.0, 4.0, 9.0, 16.0};  // y = x^2 at the grid

    auto emit_interp = [&](const char* key, auto& interp) {
        std::cout << "  \"" << key << "\": [\n";
        bool first = true;
        for (double x : {0.5, 1.5, 2.5, 3.5}) {
            if (!first) std::cout << ",\n";
            std::cout << "    {\"x\":" << x << ",\"v\":" << interp(x) << "}";
            first = false;
        }
        std::cout << "\n  ]";
    };

    {
        LinearInterpolation lin(xs.begin(), xs.end(), ys.begin());
        lin.update();
        emit_interp("linear", lin);
        std::cout << ",\n";
    }
    {
        // LogLinear requires y > 0; shift away from zero
        std::vector<Real> ys2 = {1.0, 2.0, 4.0, 9.0, 16.0};
        LogLinearInterpolation loglin(xs.begin(), xs.end(), ys2.begin());
        loglin.update();
        emit_interp("loglinear", loglin);
        std::cout << ",\n";
    }
    {
        BackwardFlatInterpolation bf(xs.begin(), xs.end(), ys.begin());
        bf.update();
        emit_interp("backward_flat", bf);
        std::cout << ",\n";
    }
    {
        ForwardFlatInterpolation ff(xs.begin(), xs.end(), ys.begin());
        ff.update();
        emit_interp("forward_flat", ff);
        std::cout << ",\n";
    }

    // --- 2-D bilinear over a 3x3 grid ------------------------------------
    {
        std::vector<Real> xg = {0.0, 1.0, 2.0};
        std::vector<Real> yg = {0.0, 1.0, 2.0};
        Matrix z(3, 3);
        for (Size i = 0; i < 3; ++i)
            for (Size j = 0; j < 3; ++j)
                z[i][j] = xg[j] + yg[i];  // z = x + y

        BilinearInterpolation bilin(xg.begin(), xg.end(), yg.begin(), yg.end(), z);
        bilin.update();
        std::cout << "  \"bilinear\": [\n";
        bool first = true;
        for (double x : {0.25, 0.5, 1.25, 1.75}) {
            for (double y : {0.25, 0.75, 1.5}) {
                if (!first) std::cout << ",\n";
                std::cout << "    {\"x\":" << x << ",\"y\":" << y
                          << ",\"v\":" << bilin(x, y) << "}";
                first = false;
            }
        }
        std::cout << "\n  ],\n";
    }

    // --- Matrix: Cholesky decomposition of a symmetric pos-def matrix -----
    {
        Matrix m(3, 3);
        // SPD matrix: [[4,2,2],[2,3,1],[2,1,5]]
        m[0][0] = 4; m[0][1] = 2; m[0][2] = 2;
        m[1][0] = 2; m[1][1] = 3; m[1][2] = 1;
        m[2][0] = 2; m[2][1] = 1; m[2][2] = 5;

        Matrix L = CholeskyDecomposition(m);
        std::cout << "  \"cholesky\": [\n";
        for (Size i = 0; i < 3; ++i) {
            if (i) std::cout << ",\n";
            std::cout << "    [";
            for (Size j = 0; j < 3; ++j) {
                if (j) std::cout << ", ";
                std::cout << L[i][j];
            }
            std::cout << "]";
        }
        std::cout << "\n  ]\n";
    }

    std::cout << "}\n";
    return 0;
}
