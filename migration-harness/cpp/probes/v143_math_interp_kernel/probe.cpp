// migration-harness/cpp/probes/v143_math_interp_kernel/probe.cpp
//
// Reference values for the kernel / 2-D / SABR-plumbing block of
// ql/math/interpolations plus ql/math/kernelfunctions.hpp (C++ v1.43).
//
// What is pinned and why:
//
//  * GaussianKernel        — value, derivative and primitive.  The class is a
//                            NormalDistribution scaled by sqrt(2*pi), so the
//                            normalisation factor is the thing that a port
//                            silently gets wrong; probed on both tails.
//  * KernelInterpolation   — the interpolant itself AND the intermediate
//                            gamma(x) / M matrix, reconstructed here from the
//                            public GaussianKernel with the same arithmetic
//                            the (private) Impl uses.  Pinning M lets the
//                            Python test derive its tolerance from cond(M)
//                            rather than guessing, because the alpha vector
//                            comes out of a linear solve.
//  * KernelInterpolation2D — note the layout: this is the ONE 2-D
//                            interpolation in QuantLib whose zData is indexed
//                            [x][y] (rows == xSize).  Every other 2-D
//                            interpolation uses [y][x].  Both a Gaussian and
//                            the test-suite's Epanechnikov kernel are probed,
//                            the latter because it is compactly supported and
//                            therefore makes M sparse/near-singular.
//  * BilinearInterpolation / Bilinear factory.
//  * BicubicSpline         — value plus derivativeX / derivativeY /
//                            derivativeXY / secondDerivativeX /
//                            secondDerivativeY (the BicubicSplineDerivatives
//                            interface), on a NON-square grid with
//                            NON-uniform spacing so a transposed or
//                            uniform-spacing shortcut cannot pass.
//  * BackwardflatLinearInterpolation — every branch of its x selection:
//                            x <= x0, x exactly on a node, x between nodes,
//                            x past the last node.
//  * FlatExtrapolator2D    — the 2-D clamp decorator, on all eight
//                            out-of-range directions plus the corners.
//  * LagrangeInterpolation::value(Array, Real) — the UpdatedYInterpolation
//                            cheap-y-update path, against a fresh y vector.
//  * SABRSpecs / SABRWrapper — the XABR plumbing: dimension, eps1, eps2,
//                            dilationFactor, defaultValues, guess, the
//                            direct/inverse reparameterisation (both the
//                            in-branch and the saturated out-of-branch arms)
//                            and weight().
//  * AbcdCoeffHolder       — the Null-argument defaulting rule, including the
//                            quirk that a Null argument leaves the matching
//                            *IsFixed flag at false regardless of what the
//                            caller passed.
//  * MultiCubicSpline<2>/<3> — the n-D tensor product of NATURAL cubic
//                            splines, on non-uniform grids (a uniform grid
//                            lets too many schemes agree).  Only strictly
//                            interior nodes are pinned: the header carries a
//                            standing \bug note that it "cannot interpolate
//                            at the grid points on the boundary surface".
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/interp/kernel.json.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ql/math/array.hpp>
#include <ql/math/kernelfunctions.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/interpolations/abcdinterpolation.hpp>
#include <ql/math/interpolations/backwardflatlinearinterpolation.hpp>
#include <ql/math/interpolations/bicubicsplineinterpolation.hpp>
#include <ql/math/interpolations/bilinearinterpolation.hpp>
#include <ql/math/interpolations/flatextrapolation2d.hpp>
#include <ql/math/interpolations/kernelinterpolation.hpp>
#include <ql/math/interpolations/kernelinterpolation2d.hpp>
#include <ql/math/interpolations/lagrangeinterpolation.hpp>
#include <ql/math/interpolations/multicubicspline.hpp>
#include <ql/math/interpolations/sabrinterpolation.hpp>
#include <ql/utilities/null.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------- JSON utils

// JSON has no literal for nan/inf.  Non-finite results are real outputs here
// (a compactly-supported kernel evaluated outside every node's support divides
// 0 by 0), so they are emitted as the QUOTED strings "nan"/"inf"/"-inf" and
// the Python test decodes them back.
std::string R(Real v) {
    if (std::isnan(v))
        return "\"nan\"";
    if (std::isinf(v))
        return v > 0 ? "\"inf\"" : "\"-inf\"";
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

void openObj(const std::string& key, int indent) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": {\n";
}

void num(const std::string& key, Real v, int indent, bool comma = true) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": " << R(v)
              << (comma ? "," : "") << "\n";
}

void integer(const std::string& key, long v, int indent, bool comma = true) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": " << v
              << (comma ? "," : "") << "\n";
}

void boolean(const std::string& key, bool v, int indent, bool comma = true) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": "
              << (v ? "true" : "false") << (comma ? "," : "") << "\n";
}

void str(const std::string& key, const std::string& v, int indent, bool comma = true) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": \"" << v << "\""
              << (comma ? "," : "") << "\n";
}

void vec(const std::string& key, const std::vector<Real>& v, int indent, bool comma = true) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << R(v[i]);
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void vec(const std::string& key, const Array& a, int indent, bool comma = true) {
    vec(key, std::vector<Real>(a.begin(), a.end()), indent, comma);
}

// Emits a matrix as a list of rows, in the SOURCE object's own row order.
void mat(const std::string& key, const Matrix& m, int indent, bool comma = true) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": [\n";
    for (Size i = 0; i < m.rows(); ++i) {
        std::cout << std::string(indent + 2, ' ') << "[";
        for (Size j = 0; j < m.columns(); ++j)
            std::cout << (j ? ", " : "") << R(m[i][j]);
        std::cout << "]" << (i + 1 < m.rows() ? "," : "") << "\n";
    }
    std::cout << std::string(indent, ' ') << "]" << (comma ? "," : "") << "\n";
}

void mat(const std::string& key,
         const std::vector<std::vector<Real> >& m,
         int indent,
         bool comma = true) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": [\n";
    for (Size i = 0; i < m.size(); ++i) {
        std::cout << std::string(indent + 2, ' ') << "[";
        for (Size j = 0; j < m[i].size(); ++j)
            std::cout << (j ? ", " : "") << R(m[i][j]);
        std::cout << "]" << (i + 1 < m.size() ? "," : "") << "\n";
    }
    std::cout << std::string(indent, ' ') << "]" << (comma ? "," : "") << "\n";
}

void closeObj(int indent, bool comma = true) {
    std::cout << std::string(indent, ' ') << "}" << (comma ? "," : "") << "\n";
}

// The test-suite's compactly-supported kernel (test-suite/interpolations.cpp).
Real epanechnikovKernel(Real u) {
    if (std::fabs(u) <= 1)
        return (3.0 / 4.0) * (1 - u * u);
    return 0.0;
}

// ------------------------------------------------------------ GaussianKernel

void emitGaussianKernel() {
    openObj("gaussian_kernel", 2);
    const Real avgs[] = {0.0, 0.0, 0.25};
    const Real sigmas[] = {1.0, 2.05, 0.18};
    // Both tails, the centre, and a point far enough out that a wrong
    // normalisation shows up as a relative error rather than an absolute one.
    const Real xs[] = {-6.0, -2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5, 6.0};

    std::cout << "    \"cases\": [\n";
    bool first = true;
    for (Real average : avgs) {
        for (Real sigma : sigmas) {
            GaussianKernel k(average, sigma);
            for (Real x : xs) {
                if (!first)
                    std::cout << ",\n";
                first = false;
                std::cout << "      {\"average\": " << average
                          << ", \"sigma\": " << sigma
                          << ", \"x\": " << x
                          << ", \"value\": " << R(k(x))
                          << ", \"derivative\": " << R(k.derivative(x))
                          << ", \"primitive\": " << R(k.primitive(x)) << "}";
            }
        }
    }
    std::cout << "\n    ]\n";
    closeObj(2);
}

// ------------------------------------------------------- KernelInterpolation

// Reconstructs the (private) Impl intermediates with the same arithmetic:
//   gamma(x)  = sum_i K(|x - x_i|)
//   M[r][c]   = K(|x_r - x_c|) / gamma(x_r)
void kernelIntermediates(const std::vector<Real>& xs,
                         const GaussianKernel& k,
                         std::vector<Real>& gamma,
                         Matrix& m) {
    const Size n = xs.size();
    gamma.assign(n, 0.0);
    m = Matrix(n, n);
    for (Size r = 0; r < n; ++r) {
        Real g = 0.0;
        for (Size i = 0; i < n; ++i)
            g += k(std::fabs(xs[r] - xs[i]));
        gamma[r] = g;
        const Real tmp = 1.0 / g;
        for (Size c = 0; c < n; ++c)
            m[r][c] = k(std::fabs(xs[r] - xs[c])) * tmp;
    }
}

void emitKernelInterpolation() {
    // Grid and y-sets are the ones the C++ test-suite itself uses
    // (test-suite/interpolations.cpp testKernelInterpolation).
    const std::vector<Real> xs = {0.10, 0.25, 0.50, 0.75, 0.90};
    const std::vector<std::vector<Real> > ys = {
        {11.275, 11.125, 11.250, 11.825, 12.625},
        {16.025, 13.450, 11.350, 10.150, 10.075},
        {10.300, 9.6375, 9.2000, 9.1125, 9.4000}};
    // 0.05 makes the kernel matrix near-diagonal, 2.55 makes it near-singular:
    // the two ends of the conditioning range the linear solve has to survive.
    const std::vector<Real> sigmas = {0.05, 0.50, 0.75, 1.65, 2.55};
    // Nodes, midpoints, and outside the range on both sides.
    const std::vector<Real> evalXs = {-0.20, 0.0,   0.10,  0.121, 0.175, 0.25,
                                      0.279, 0.375, 0.50,  0.625, 0.678, 0.75,
                                      0.790, 0.825, 0.90,  0.980, 1.20};

    openObj("kernel_interpolation", 2);
    vec("xs", xs, 4);
    vec("eval_xs", evalXs, 4);
    std::cout << "    \"cases\": [\n";
    bool first = true;
    for (Size s = 0; s < sigmas.size(); ++s) {
        GaussianKernel kern(0.0, sigmas[s]);
        std::vector<Real> gamma;
        Matrix m;
        kernelIntermediates(xs, kern, gamma, m);
        for (Size yi = 0; yi < ys.size(); ++yi) {
            std::vector<Real> currY = ys[yi];
            KernelInterpolation f(xs.begin(), xs.end(), currY.begin(), kern);
            f.enableExtrapolation();
            std::vector<Real> vals;
            vals.reserve(evalXs.size());
            for (Real x : evalXs)
                vals.push_back(f(x));

            if (!first)
                std::cout << ",\n";
            first = false;
            std::cout << "      {\n";
            num("sigma", sigmas[s], 8);
            integer("y_index", static_cast<long>(yi), 8);
            vec("ys", currY, 8);
            vec("gamma", gamma, 8);
            mat("m", m, 8);
            vec("values", vals, 8, false);
            std::cout << "      }";
        }
    }
    std::cout << "\n    ]\n";
    closeObj(2);
}

// ----------------------------------------------------- KernelInterpolation2D

// NOTE the layout: zData is [x][y] here (rows == xSize), unlike every other
// 2-D interpolation in QuantLib.  The matrices below are written exactly as
// the C++ test-suite writes them, i.e. M[i][j] == z(x_i, y_j).
void emitKernelInterpolation2D() {
    openObj("kernel_interpolation_2d", 2);
    str("z_layout", "rows_are_x", 4);

    // --- case 1: Gaussian kernel, 10 x-points x 3 y-points -----------------
    {
        const std::vector<Real> xVec = {0.10, 0.20, 0.30, 0.40, 0.50,
                                        0.60, 0.70, 0.80, 0.90, 1.00};
        const std::vector<Real> yVec = {1.0, 2.0, 3.5};
        Matrix M(xVec.size(), yVec.size());
        const Real zdata[10][3] = {{0.25, 0.27, 0.21}, {0.24, 0.26, 0.22},
                                   {0.23, 0.25, 0.27}, {0.20, 0.22, 0.29},
                                   {0.19, 0.21, 0.24}, {0.20, 0.22, 0.28},
                                   {0.21, 0.23, 0.25}, {0.22, 0.24, 0.22},
                                   {0.26, 0.28, 0.29}, {0.29, 0.31, 0.30}};
        for (Size i = 0; i < xVec.size(); ++i)
            for (Size j = 0; j < yVec.size(); ++j)
                M[i][j] = zdata[i][j];

        GaussianKernel kern(0.0, 0.18);
        KernelInterpolation2D k2d(xVec.begin(), xVec.end(), yVec.begin(), yVec.end(), M, kern);
        k2d.enableExtrapolation();

        // Nodes, midpoints, and off-grid / out-of-range points.
        std::vector<std::vector<Real> > evals;
        for (Size i = 0; i < xVec.size(); ++i)
            for (Size j = 0; j < yVec.size(); ++j)
                evals.push_back({xVec[i], yVec[j], k2d(xVec[i], yVec[j])});
        const Real probeX[] = {0.05, 0.15, 0.45, 0.55, 0.95, 1.10};
        const Real probeY[] = {0.5, 1.5, 2.75, 3.0, 4.0};
        for (Real x : probeX)
            for (Real y : probeY)
                evals.push_back({x, y, k2d(x, y)});

        openObj("gaussian", 4);
        num("average", 0.0, 6);
        num("sigma", 0.18, 6);
        vec("xs", xVec, 6);
        vec("ys", yVec, 6);
        mat("z", M, 6);
        mat("evals", evals, 6, false);
        closeObj(4);
    }

    // --- case 2: Epanechnikov kernel, 4 x-points x 8 y-points --------------
    {
        const std::vector<Real> xVec = {80.0, 90.0, 100.0, 110.0};
        const std::vector<Real> yVec = {0.5, 0.7, 1.0, 2.0, 3.5, 4.5, 5.5, 6.5};
        Matrix M(xVec.size(), yVec.size());
        const Real zdata[4][8] = {
            {10.25, 12.25, 12.25, 13.25, 14.25, 15.25, 16.25, 14.25},
            {12.24, 15.24, 13.24, 15.24, 16.24, 17.24, 13.24, 14.24},
            {14.23, 16.23, 13.23, 12.23, 13.23, 14.23, 15.23, 16.23},
            {17.20, 16.20, 17.20, 19.20, 12.20, 12.20, 10.20, 19.20}};
        for (Size i = 0; i < xVec.size(); ++i)
            for (Size j = 0; j < yVec.size(); ++j)
                M[i][j] = zdata[i][j];

        KernelInterpolation2D k2d(xVec.begin(), xVec.end(), yVec.begin(), yVec.end(), M,
                                  &epanechnikovKernel);
        k2d.enableExtrapolation();

        std::vector<std::vector<Real> > evals;
        for (Size i = 0; i < xVec.size(); ++i)
            for (Size j = 0; j < yVec.size(); ++j)
                evals.push_back({xVec[i], yVec[j], k2d(xVec[i], yVec[j])});
        // The Epanechnikov kernel has support |u| <= 1 and the x-grid is spaced
        // by 10, so only points within distance 1 of a node see any mass at
        // all.  Both regimes are pinned: in-support points give a finite
        // weighted average, out-of-support points give 0/0 == nan.
        const Real probeX[] = {80.0, 80.4, 90.0, 90.6, 110.0, 85.0, 95.0, 105.0};
        const Real probeY[] = {0.55, 0.6, 0.9, 2.2, 4.0, 6.0};
        for (Real x : probeX)
            for (Real y : probeY)
                evals.push_back({x, y, k2d(x, y)});

        openObj("epanechnikov", 4);
        vec("xs", xVec, 6);
        vec("ys", yVec, 6);
        mat("z", M, 6);
        mat("evals", evals, 6, false);
        closeObj(4, false);
    }

    closeObj(2);
}

// --------------------------------------------------- 2-D grid shared by the
// bilinear / bicubic / backward-flat-linear / flat-extrapolator sections.
// 5 x-points x 4 y-points, non-uniform on both axes: rows != columns catches a
// transposed z, non-uniform spacing catches a uniform-spacing shortcut.
const std::vector<Real> kXs = {0.5, 1.0, 2.0, 3.5, 4.0};
const std::vector<Real> kYs = {-1.0, 0.0, 1.5, 3.0};

Matrix makeZ() {
    // zData is [y][x] for these three: rows == ySize, columns == xSize.
    Matrix z(kYs.size(), kXs.size());
    for (Size j = 0; j < kYs.size(); ++j)
        for (Size i = 0; i < kXs.size(); ++i)
            z[j][i] = std::sin(kXs[i]) + std::cos(kYs[j]);
    return z;
}

void emitBilinear() {
    Matrix z = makeZ();
    BilinearInterpolation f(kXs.begin(), kXs.end(), kYs.begin(), kYs.end(), z);
    f.enableExtrapolation();
    // Factory path must agree with direct construction.
    Interpolation2D viaFactory =
        Bilinear().interpolate(kXs.begin(), kXs.end(), kYs.begin(), kYs.end(), z);
    viaFactory.enableExtrapolation();

    std::vector<std::vector<Real> > evals;
    for (Size j = 0; j < kYs.size(); ++j)
        for (Size i = 0; i < kXs.size(); ++i)
            evals.push_back({kXs[i], kYs[j], f(kXs[i], kYs[j])});
    const Real px[] = {0.0, 0.75, 1.5, 2.75, 3.75, 4.5};
    const Real py[] = {-2.0, -0.5, 0.75, 2.25, 3.5};
    for (Real x : px)
        for (Real y : py)
            evals.push_back({x, y, f(x, y)});

    std::vector<std::vector<Real> > factoryEvals;
    for (Real x : px)
        for (Real y : py)
            factoryEvals.push_back({x, y, viaFactory(x, y)});

    openObj("bilinear", 2);
    vec("xs", kXs, 4);
    vec("ys", kYs, 4);
    mat("z", z, 4);
    mat("evals", evals, 4);
    mat("factory_evals", factoryEvals, 4, false);
    closeObj(2);
}

void emitBicubic() {
    Matrix z = makeZ();
    BicubicSpline f(kXs.begin(), kXs.end(), kYs.begin(), kYs.end(), z);
    f.enableExtrapolation();
    Interpolation2D viaFactory =
        Bicubic().interpolate(kXs.begin(), kXs.end(), kYs.begin(), kYs.end(), z);
    viaFactory.enableExtrapolation();

    // value(): nodes, midpoints and out-of-range on both axes.
    std::vector<std::vector<Real> > evals;
    for (Size j = 0; j < kYs.size(); ++j)
        for (Size i = 0; i < kXs.size(); ++i)
            evals.push_back({kXs[i], kYs[j], f(kXs[i], kYs[j])});
    const Real px[] = {0.0, 0.75, 1.5, 2.75, 3.75, 4.5};
    const Real py[] = {-2.0, -0.5, 0.75, 2.25, 3.5};
    for (Real x : px)
        for (Real y : py)
            evals.push_back({x, y, f(x, y)});

    std::vector<std::vector<Real> > factoryEvals;
    for (Real x : px)
        for (Real y : py)
            factoryEvals.push_back({x, y, viaFactory(x, y)});

    // The derivative API builds a fresh CubicInterpolation and calls
    // derivative()/secondDerivative() with allowExtrapolation defaulted to
    // FALSE, so it only accepts in-range arguments.
    const Real dx[] = {0.5, 0.75, 1.0, 1.5, 2.0, 2.75, 3.5, 4.0};
    const Real dy[] = {-1.0, -0.5, 0.0, 0.75, 1.5, 2.25, 3.0};
    std::vector<std::vector<Real> > derivs;
    for (Real x : dx)
        for (Real y : dy)
            derivs.push_back({x, y, f.derivativeX(x, y), f.derivativeY(x, y),
                              f.derivativeXY(x, y), f.secondDerivativeX(x, y),
                              f.secondDerivativeY(x, y)});

    openObj("bicubic", 2);
    vec("xs", kXs, 4);
    vec("ys", kYs, 4);
    mat("z", z, 4);
    mat("evals", evals, 4);
    mat("factory_evals", factoryEvals, 4);
    str("derivs_columns", "x,y,dX,dY,dXY,d2X,d2Y", 4);
    mat("derivs", derivs, 4, false);
    closeObj(2);
}

void emitBackwardflatLinear() {
    Matrix z = makeZ();
    BackwardflatLinearInterpolation f(kXs.begin(), kXs.end(), kYs.begin(), kYs.end(), z);
    f.enableExtrapolation();
    Interpolation2D viaFactory =
        BackwardflatLinear().interpolate(kXs.begin(), kXs.end(), kYs.begin(), kYs.end(), z);
    viaFactory.enableExtrapolation();

    // Every branch of the x selection: below x0, exactly on each node, just
    // above / just below each node, and beyond the last node.
    std::vector<Real> px;
    px.push_back(0.0);
    px.push_back(0.4999);
    for (Real x : kXs) {
        px.push_back(x - 1e-9);
        px.push_back(x);
        px.push_back(x + 1e-9);
    }
    px.push_back(0.75);
    px.push_back(1.5);
    px.push_back(2.75);
    px.push_back(3.75);
    px.push_back(4.5);
    const Real py[] = {-2.0, -1.0, -0.5, 0.0, 0.75, 1.5, 2.25, 3.0, 3.5};

    std::vector<std::vector<Real> > evals;
    for (Real x : px)
        for (Real y : py)
            evals.push_back({x, y, f(x, y)});
    std::vector<std::vector<Real> > factoryEvals;
    for (Real x : px)
        for (Real y : py)
            factoryEvals.push_back({x, y, viaFactory(x, y)});

    openObj("backwardflat_linear", 2);
    vec("xs", kXs, 4);
    vec("ys", kYs, 4);
    mat("z", z, 4);
    mat("evals", evals, 4);
    mat("factory_evals", factoryEvals, 4, false);
    closeObj(2);
}

void emitFlatExtrapolator2D() {
    // The decorated interpolation must outlive the decorator; keep both the
    // matrix and the shared_ptr alive for the whole emission.
    static Matrix z = makeZ();
    auto decorated = ext::make_shared<BicubicSpline>(kXs.begin(), kXs.end(), kYs.begin(),
                                                     kYs.end(), z);
    FlatExtrapolator2D f(decorated);
    f.enableExtrapolation();

    // In range, then each of the eight out-of-range directions, then the
    // corners: this is what distinguishes a clamp from a true extrapolation.
    const Real px[] = {-1.0, 0.0, 0.5, 1.5, 2.75, 4.0, 5.0, 7.5};
    const Real py[] = {-4.0, -2.0, -1.0, 0.75, 2.25, 3.0, 5.0};
    std::vector<std::vector<Real> > evals;
    for (Real x : px)
        for (Real y : py)
            evals.push_back({x, y, f(x, y)});

    // The decorated surface, evaluated at the CLAMPED point, must equal the
    // decorator's answer — pinned separately so a test can prove the clamp
    // rather than merely reproduce a number.
    std::vector<std::vector<Real> > clamped;
    for (Real x : px) {
        for (Real y : py) {
            const Real bx = std::min(std::max(x, f.xMin()), f.xMax());
            const Real by = std::min(std::max(y, f.yMin()), f.yMax());
            clamped.push_back({x, y, bx, by, (*decorated)(bx, by)});
        }
    }

    openObj("flat_extrapolator_2d", 2);
    str("decorated", "BicubicSpline", 4);
    vec("xs", kXs, 4);
    vec("ys", kYs, 4);
    mat("z", z, 4);
    num("x_min", f.xMin(), 4);
    num("x_max", f.xMax(), 4);
    num("y_min", f.yMin(), 4);
    num("y_max", f.yMax(), 4);
    mat("evals", evals, 4);
    str("clamped_columns", "x,y,bound_x,bound_y,decorated_at_bound", 4);
    mat("clamped", clamped, 4, false);
    closeObj(2);
}

// ------------------------------------------------- UpdatedYInterpolation path

void emitLagrangeUpdatedY() {
    // Non-uniform nodes so the barycentric weights are not symmetric.
    const std::vector<Real> xs = {-1.0, -0.4, 0.1, 0.7, 1.3, 2.2};
    const std::vector<Real> ys = {0.3, -0.2, 1.1, 0.55, -0.9, 2.4};
    LagrangeInterpolation f(xs.begin(), xs.end(), ys.begin());
    f.enableExtrapolation();

    // The fresh y vector the updatedValue path is for.
    Array y2(xs.size());
    for (Size i = 0; i < xs.size(); ++i)
        y2[i] = std::sin(2.0 * xs[i]) + 0.5 * xs[i];

    const std::vector<Real> evalXs = {-1.5, -1.0, -0.7, -0.4, 0.0,  0.1,
                                      0.4,  0.7,  1.0,  1.3,  1.75, 2.2, 3.0};
    std::vector<Real> baseVals, updatedVals, baseWithOwnY, derivatives;
    for (Real x : evalXs) {
        baseVals.push_back(f(x));
        updatedVals.push_back(f.value(y2, x));
        // Feeding back the original y must reproduce the plain value.
        Array y0(ys.begin(), ys.end());
        baseWithOwnY.push_back(f.value(y0, x));
        derivatives.push_back(f.derivative(x, true));
    }

    openObj("lagrange_updated_y", 2);
    vec("xs", xs, 4);
    vec("ys", ys, 4);
    vec("y2", y2, 4);
    vec("eval_xs", evalXs, 4);
    vec("values", baseVals, 4);
    vec("updated_values", updatedVals, 4);
    vec("values_with_own_y", baseWithOwnY, 4);
    vec("derivatives", derivatives, 4, false);
    closeObj(2);
}

// ------------------------------------------------------ SABR XABR plumbing

void emitSabrSpecs() {
    detail::SABRSpecs specs;

    openObj("sabr_specs", 2);
    integer("dimension", static_cast<long>(specs.dimension()), 4);
    num("eps1", specs.eps1(), 4);
    num("eps2", specs.eps2(), 4);
    num("dilation_factor", specs.dilationFactor(), 4);

    // --- defaultValues: every Null / non-Null combination that matters -----
    {
        const Real nullR = Null<Real>();
        num("null_real", nullR, 4);
        std::cout << "    \"default_values\": [\n";
        struct Case {
            Real a, b, nu, rho, forward, expiry, shift;
        };
        const Case cases[] = {
            {nullR, nullR, nullR, nullR, 0.03, 1.0, 0.0},
            {nullR, nullR, nullR, nullR, 0.03, 1.0, 0.02},
            {nullR, 0.99995, nullR, nullR, 0.03, 1.0, 0.0},  // beta >= 0.9999 arm
            {nullR, 0.7, nullR, nullR, 120.0, 5.0, 0.0},
            {0.25, nullR, 0.4, 0.1, 0.03, 1.0, 0.0},
        };
        bool first = true;
        for (const Case& c : cases) {
            std::vector<Real> params = {c.a, c.b, c.nu, c.rho};
            std::vector<bool> isFixed(4, false);
            std::vector<Real> addParams;
            if (c.shift != 0.0)
                addParams.push_back(c.shift);
            specs.defaultValues(params, isFixed, c.forward, c.expiry, addParams);
            if (!first)
                std::cout << ",\n";
            first = false;
            std::cout << "      {\"in\": [" << c.a << ", " << c.b << ", " << c.nu << ", "
                      << c.rho << "], \"forward\": " << c.forward
                      << ", \"expiry\": " << c.expiry << ", \"shift\": " << c.shift
                      << ", \"has_add_params\": " << (addParams.empty() ? "false" : "true")
                      << ", \"out\": [" << params[0] << ", " << params[1] << ", " << params[2]
                      << ", " << params[3] << "]}";
        }
        std::cout << "\n    ],\n";
    }

    // --- guess -------------------------------------------------------------
    {
        std::cout << "    \"guess\": [\n";
        struct GCase {
            bool fixed[4];
            Real r[4];
            Real forward, shift;
        };
        const GCase cases[] = {
            {{false, false, false, false}, {0.1, 0.2, 0.3, 0.4}, 0.03, 0.0},
            {{false, false, false, false}, {0.9, 0.95, 0.5, 0.05}, 120.0, 0.0},
            {{false, false, false, false}, {0.1, 0.9995, 0.3, 0.4}, 0.03, 0.02},
            {{true, false, true, false}, {0.25, 0.75, 0.5, 0.5}, 0.03, 0.0},
            {{true, true, true, true}, {0.5, 0.5, 0.5, 0.5}, 0.03, 0.0},
        };
        bool first = true;
        for (const GCase& c : cases) {
            Array values(4, 0.0);
            // Seed with the same starting values in every case so the untouched
            // slots are visible in the output.
            values[0] = -1.0;
            values[1] = -2.0;
            values[2] = -3.0;
            values[3] = -4.0;
            std::vector<bool> isFixed(c.fixed, c.fixed + 4);
            std::vector<Real> r(c.r, c.r + 4);
            std::vector<Real> addParams;
            if (c.shift != 0.0)
                addParams.push_back(c.shift);
            specs.guess(values, isFixed, c.forward, 1.0, r, addParams);
            if (!first)
                std::cout << ",\n";
            first = false;
            std::cout << "      {\"is_fixed\": [" << (c.fixed[0] ? "true" : "false") << ", "
                      << (c.fixed[1] ? "true" : "false") << ", "
                      << (c.fixed[2] ? "true" : "false") << ", "
                      << (c.fixed[3] ? "true" : "false") << "], \"r\": [" << c.r[0] << ", "
                      << c.r[1] << ", " << c.r[2] << ", " << c.r[3]
                      << "], \"forward\": " << c.forward << ", \"shift\": " << c.shift
                      << ", \"out\": [" << values[0] << ", " << values[1] << ", " << values[2]
                      << ", " << values[3] << "]}";
        }
        std::cout << "\n    ],\n";
    }

    // --- direct / inverse --------------------------------------------------
    // Both arms of every branch: |x0|<5 vs >=5, |x1| below/above
    // sqrt(-log(eps1)), |x3| below/above 2.5*pi (and both signs).
    {
        const Real xcases[][4] = {
            {0.3, 0.5, 0.8, 0.4},
            {-0.3, -0.5, -0.8, -0.4},
            {4.999, 1.0, 2.0, 2.0},
            {5.0, 1.0, 2.0, 2.0},
            {7.5, 4.0, 6.0, 8.0},
            {-7.5, -4.0, -6.0, -8.0},
            {1.0, 4.01, 1.0, 7.853981633974483},   // x1 just above sqrt(-log(eps1))
            {1.0, 4.0, 1.0, 7.9},                  // |x3| > 2.5*pi, positive arm
            {1.0, 4.0, 1.0, -7.9},                 // |x3| > 2.5*pi, negative arm
            {0.0, 0.0, 0.0, 0.0},
        };
        std::vector<bool> dummyFixed(4, false);
        std::vector<Real> dummyParams(4, 0.0);
        std::cout << "    \"direct\": [\n";
        bool first = true;
        for (const auto& xc : xcases) {
            Array x(4);
            for (Size i = 0; i < 4; ++i)
                x[i] = xc[i];
            const Array y = specs.direct(x, dummyFixed, dummyParams, 0.03);
            if (!first)
                std::cout << ",\n";
            first = false;
            std::cout << "      {\"x\": [" << x[0] << ", " << x[1] << ", " << x[2] << ", "
                      << x[3] << "], \"y\": [" << y[0] << ", " << y[1] << ", " << y[2] << ", "
                      << y[3] << "]}";
        }
        std::cout << "\n    ],\n";

        // inverse: y[1] must be in (0, 1] for the sqrt(-log(y1)) branch, and
        // y[3]/eps2 must be in [-1, 1] for asin.
        const Real ycases[][4] = {
            {0.09, 0.5, 0.64, 0.4},
            {0.09, 0.9999999, 0.64, -0.4},
            {25.0, 0.5, 25.0, 0.0},
            {25.0000002, 0.5, 25.0000002, 0.9999},
            {100.0, 0.25, 300.0, -0.9999},
            {1.0e-7, 0.1, 1.0e-7, 0.5},
        };
        std::cout << "    \"inverse\": [\n";
        first = true;
        for (const auto& yc : ycases) {
            Array y(4);
            for (Size i = 0; i < 4; ++i)
                y[i] = yc[i];
            const Array x = specs.inverse(y, dummyFixed, dummyParams, 0.03);
            if (!first)
                std::cout << ",\n";
            first = false;
            std::cout << "      {\"y\": [" << y[0] << ", " << y[1] << ", " << y[2] << ", "
                      << y[3] << "], \"x\": [" << x[0] << ", " << x[1] << ", " << x[2] << ", "
                      << x[3] << "]}";
        }
        std::cout << "\n    ],\n";
    }

    // --- weight ------------------------------------------------------------
    {
        std::cout << "    \"weight\": [\n";
        const Real strikes[] = {0.01, 0.02, 0.03, 0.05, 0.08};
        const Real stdDevs[] = {0.05, 0.20, 0.60};
        const Real shifts[] = {0.0, 0.01};
        bool first = true;
        for (Real shift : shifts) {
            std::vector<Real> addParams(1, shift);
            for (Real k : strikes) {
                for (Real sd : stdDevs) {
                    const Real w = specs.weight(k, 0.03, sd, addParams);
                    if (!first)
                        std::cout << ",\n";
                    first = false;
                    std::cout << "      {\"strike\": " << k << ", \"forward\": " << 0.03
                              << ", \"std_dev\": " << sd << ", \"shift\": " << shift
                              << ", \"weight\": " << w << "}";
                }
            }
        }
        std::cout << "\n    ]\n";
    }
    closeObj(2);
}

void emitSabrWrapper() {
    openObj("sabr_wrapper", 2);
    std::cout << "    \"cases\": [\n";
    struct WCase {
        Real t, forward, alpha, beta, nu, rho, shift;
        bool useAddParams;
    };
    const WCase cases[] = {
        {1.0, 0.03, 0.25, 0.5, 0.4, -0.2, 0.0, false},
        {1.0, 0.03, 0.25, 0.5, 0.4, -0.2, 0.0, true},
        {5.0, 0.03, 0.15, 0.7, 0.6, 0.3, 0.01, true},
        {0.25, 120.0, 2.5, 0.9, 0.35, -0.45, 0.0, false},
    };
    const Real strikes[] = {0.015, 0.02, 0.03, 0.045, 0.07};
    const Real bigStrikes[] = {80.0, 100.0, 120.0, 140.0, 170.0};
    bool first = true;
    for (const WCase& c : cases) {
        std::vector<Real> params = {c.alpha, c.beta, c.nu, c.rho};
        std::vector<Real> addParams;
        if (c.useAddParams)
            addParams.push_back(c.shift);
        detail::SABRWrapper w(c.t, c.forward, params, addParams);
        const bool big = c.forward > 1.0;
        std::vector<Real> ks;
        if (big)
            ks.assign(bigStrikes, bigStrikes + 5);
        else
            ks.assign(strikes, strikes + 5);
        std::vector<Real> lognormal, normal;
        for (Real k : ks) {
            lognormal.push_back(w.volatility(k, VolatilityType::ShiftedLognormal));
            normal.push_back(w.volatility(k, VolatilityType::Normal));
        }
        if (!first)
            std::cout << ",\n";
        first = false;
        std::cout << "      {\n";
        num("t", c.t, 8);
        num("forward", c.forward, 8);
        vec("params", params, 8);
        num("shift", c.useAddParams ? c.shift : 0.0, 8);
        boolean("has_add_params", c.useAddParams, 8);
        vec("strikes", ks, 8);
        vec("shifted_lognormal", lognormal, 8);
        vec("normal", normal, 8, false);
        std::cout << "      }";
    }
    std::cout << "\n    ]\n";
    closeObj(2);
}

// ------------------------------------------------------------ AbcdCoeffHolder

// AbcdCoeffHolder derives from Interpolation::Impl and is therefore abstract;
// only its constructor's Null-defaulting rule is under test, so the pure
// virtuals get minimal stubs.
struct ProbeAbcdCoeffHolder : detail::AbcdCoeffHolder {
    ProbeAbcdCoeffHolder(Real a, Real b, Real c, Real d,
                         bool af, bool bf, bool cf, bool df)
    : detail::AbcdCoeffHolder(a, b, c, d, af, bf, cf, df) {}
    void update() override {}
    Real xMin() const override { return 0.0; }
    Real xMax() const override { return 0.0; }
    std::vector<Real> xValues() const override { return {}; }
    std::vector<Real> yValues() const override { return {}; }
    bool isInRange(Real) const override { return true; }
    Real value(Real) const override { return 0.0; }
    Real primitive(Real) const override { return 0.0; }
    Real derivative(Real) const override { return 0.0; }
    Real secondDerivative(Real) const override { return 0.0; }
};

void emitAbcdCoeffHolder() {
    const Real nullR = Null<Real>();
    openObj("abcd_coeff_holder", 2);
    num("null_real", nullR, 4);
    boolean("abcd_global", Abcd::global, 4);
    integer("abcd_required_points", static_cast<long>(Abcd::requiredPoints), 4);
    std::cout << "    \"cases\": [\n";
    struct HCase {
        Real a, b, c, d;
        bool af, bf, cf, df;
    };
    const HCase cases[] = {
        {-0.06, 0.17, 0.54, 0.17, false, false, false, false},
        {-0.06, 0.17, 0.54, 0.17, true, true, true, true},
        // A Null argument takes the hard-coded default AND leaves the matching
        // *IsFixed flag at false, whatever the caller passed.
        {nullR, 0.17, 0.54, 0.17, true, true, true, true},
        {-0.06, nullR, 0.54, 0.17, true, true, true, true},
        {-0.06, 0.17, nullR, 0.17, true, true, true, true},
        {-0.06, 0.17, 0.54, nullR, true, true, true, true},
        {nullR, nullR, nullR, nullR, true, true, true, true},
        {-0.02, 0.3, 0.8, 0.05, true, false, true, false},
    };
    bool first = true;
    for (const HCase& hc : cases) {
        ProbeAbcdCoeffHolder h(hc.a, hc.b, hc.c, hc.d, hc.af, hc.bf, hc.cf, hc.df);
        if (!first)
            std::cout << ",\n";
        first = false;
        std::cout << "      {\"in\": [" << hc.a << ", " << hc.b << ", " << hc.c << ", " << hc.d
                  << "], \"in_is_fixed\": [" << (hc.af ? "true" : "false") << ", "
                  << (hc.bf ? "true" : "false") << ", " << (hc.cf ? "true" : "false") << ", "
                  << (hc.df ? "true" : "false") << "], \"out\": [" << h.a_ << ", " << h.b_ << ", "
                  << h.c_ << ", " << h.d_ << "], \"out_is_fixed\": ["
                  << (h.aIsFixed_ ? "true" : "false") << ", " << (h.bIsFixed_ ? "true" : "false")
                  << ", " << (h.cIsFixed_ ? "true" : "false") << ", "
                  << (h.dIsFixed_ ? "true" : "false") << "], \"end_criteria\": "
                  << static_cast<int>(h.abcdEndCriteria_) << "}";
    }
    std::cout << "\n    ]\n";
    closeObj(2, false);
}

// --------------------------------------------------------- MultiCubicSpline
//
// The n-dimensional tensor-product spline.  Its 1-D building block
// (detail::base_cubic_spline) sets y2[0] = y2[dim] = 0, i.e. NATURAL boundary
// conditions, and the recursion collapses one axis at a time — so this is the
// n-D generalisation of BicubicSpline, not a local Hermite interpolant.
//
// Two constraints from the header: each axis needs at least 4 points
// (set_shared_increments requires size()-1 > 2), and there is a documented
// bug — "cannot interpolate at the grid points on the boundary surface of the
// N-dimensional region" — so only strictly interior grid nodes are pinned as
// node round-trips.  Off-node interior points are pinned regardless.

void emitMultiCubicSpline() {
    openObj("multi_cubic_spline", 2);

    // --- 2-D: 5 x 4, non-uniform on both axes ------------------------------
    {
        SplineGrid grid(2);
        grid[0] = {0.5, 1.0, 2.0, 3.5, 4.0};
        grid[1] = {-1.0, 0.0, 1.5, 3.0};
        std::vector<Size> dim = {grid[0].size(), grid[1].size()};

        MultiCubicSpline<2>::data_table y(dim);
        std::vector<std::vector<Real> > values;
        for (Size i = 0; i < dim[0]; ++i) {
            std::vector<Real> row;
            for (Size j = 0; j < dim[1]; ++j) {
                const Real v = std::sin(grid[0][i]) + std::cos(grid[1][j]);
                y[i][j] = v;
                row.push_back(v);
            }
            values.push_back(row);
        }

        MultiCubicSpline<2> cs(grid, y);

        std::vector<std::vector<Real> > evals;
        // Strictly interior grid nodes (the header's documented bug excludes
        // the boundary surface).
        for (Size i = 1; i + 1 < dim[0]; ++i)
            for (Size j = 1; j + 1 < dim[1]; ++j) {
                std::vector<Real> args = {grid[0][i], grid[1][j]};
                evals.push_back({args[0], args[1], cs(args)});
            }
        // Interior off-node points.
        const Real px[] = {0.75, 1.5, 2.75, 3.75};
        const Real py[] = {-0.5, 0.75, 2.25};
        for (Real x : px)
            for (Real yv : py) {
                std::vector<Real> args = {x, yv};
                evals.push_back({x, yv, cs(args)});
            }

        openObj("dim2", 4);
        vec("axis0", grid[0], 6);
        vec("axis1", grid[1], 6);
        mat("values", values, 6);
        mat("evals", evals, 6, false);
        closeObj(4);
    }

    // --- 3-D: 5 x 4 x 4, non-uniform on all axes ---------------------------
    {
        SplineGrid grid(3);
        grid[0] = {0.5, 1.0, 2.0, 3.5, 4.0};
        grid[1] = {-1.0, 0.0, 1.5, 3.0};
        grid[2] = {0.0, 0.25, 0.75, 1.5};
        std::vector<Size> dim = {grid[0].size(), grid[1].size(), grid[2].size()};

        MultiCubicSpline<3>::data_table y(dim);
        // Flattened in C (row-major) order so the Python side can reshape it.
        std::vector<Real> flat;
        for (Size i = 0; i < dim[0]; ++i)
            for (Size j = 0; j < dim[1]; ++j)
                for (Size k = 0; k < dim[2]; ++k) {
                    const Real v = std::sin(grid[0][i]) + std::cos(grid[1][j])
                                   + 0.5 * grid[2][k] * grid[2][k];
                    y[i][j][k] = v;
                    flat.push_back(v);
                }

        MultiCubicSpline<3> cs(grid, y);

        std::vector<std::vector<Real> > evals;
        for (Size i = 1; i + 1 < dim[0]; ++i)
            for (Size j = 1; j + 1 < dim[1]; ++j)
                for (Size k = 1; k + 1 < dim[2]; ++k) {
                    std::vector<Real> args = {grid[0][i], grid[1][j], grid[2][k]};
                    evals.push_back({args[0], args[1], args[2], cs(args)});
                }
        const Real px[] = {0.75, 1.5, 2.75};
        const Real py[] = {-0.5, 0.75, 2.25};
        const Real pz[] = {0.1, 0.5, 1.1};
        for (Real x : px)
            for (Real yv : py)
                for (Real zv : pz) {
                    std::vector<Real> args = {x, yv, zv};
                    evals.push_back({x, yv, zv, cs(args)});
                }

        openObj("dim3", 4);
        vec("axis0", grid[0], 6);
        vec("axis1", grid[1], 6);
        vec("axis2", grid[2], 6);
        vec("values_flat", flat, 6);
        mat("evals", evals, 6, false);
        closeObj(4, false);
    }

    closeObj(2);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    emitGaussianKernel();
    emitKernelInterpolation();
    emitKernelInterpolation2D();
    emitBilinear();
    emitBicubic();
    emitBackwardflatLinear();
    emitFlatExtrapolator2D();
    emitLagrangeUpdatedY();
    emitSabrSpecs();
    emitSabrWrapper();
    emitMultiCubicSpline();
    emitAbcdCoeffHolder();
    std::cout << "}\n";
    return 0;
}
