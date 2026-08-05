// migration-harness/cpp/probes/v143_math_matrixutilities/probe.cpp
//
// Reference values for ql/math/matrixutilities/** (QuantLib v1.43).
//
// Every class here has a numpy/scipy routine with the same *name* and a
// different *algorithm*.  What is pinned is therefore the full factorisation —
// eigenvectors including sign, singular vectors including sign, the ILU L and U
// factors entry by entry, and the per-iteration error histories of the Krylov
// solvers — never a reconstruction residual, which would pass for a
// differently-signed or differently-ordered factorisation.
//
// Sections:
//   symmetric_schur            cyclic-Jacobi sweeps + the descending
//                              (value, vector) lexicographic sort
//   tqr_eigen                  all 3 EigenVectorCalculation x 3 ShiftStrategy
//                              combinations, plus the iteration count
//   svd                        Golub-Reinsch (JAMA/TNT); U, V, s, S, norm2,
//                              cond, rank, solveFor
//   bicgstab / gmres           iterations / error histories, the argument-norm
//                              trace of every A_ and M_ call (which pins the
//                              *shape* of the iteration, not just its answer)
//   sparse_ilu                 dense dumps of the L and U factors, plus apply()
//   householder_*              all three reflectionVector branches
//   get_covariance             getCovariance + CovarianceDecomposition
//   tap_correlations           the four angle parametrisations
//   frobenius_cost_function    value() (== DotProduct, NOT the CostFunction
//                              default rms) and values()
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/matrixutilities.json.

#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include <ql/math/array.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/matrixutilities/bicgstab.hpp>
#include <ql/math/matrixutilities/choleskydecomposition.hpp>
#include <ql/math/matrixutilities/getcovariance.hpp>
#include <ql/math/matrixutilities/gmres.hpp>
#include <ql/math/matrixutilities/householder.hpp>
#include <ql/math/matrixutilities/sparseilupreconditioner.hpp>
#include <ql/math/matrixutilities/svd.hpp>
#include <ql/math/matrixutilities/symmetricschurdecomposition.hpp>
#include <ql/math/matrixutilities/tapcorrelations.hpp>
#include <ql/math/matrixutilities/tqreigendecomposition.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------- JSON ----

// A bare `1` would come back from Python's json module as `int`, and `-0` would
// lose its sign; force every double to carry a '.' or an exponent so the reader
// always sees a float.
std::string num(Real x) {
    if (std::isnan(x) || std::isinf(x))
        return "null";
    std::ostringstream o;
    o << std::setprecision(17) << x;
    std::string s = o.str();
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos)
        s += ".0";
    return s;
}

std::string jarr(const Array& a) {
    std::string s = "[";
    for (Size i = 0; i < a.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += num(a[i]);
    }
    return s + "]";
}

std::string jvec(const std::vector<Real>& v) {
    std::string s = "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += num(v[i]);
    }
    return s + "]";
}

std::string jlist(const std::list<Real>& v) {
    std::string s = "[";
    bool first = true;
    for (Real e : v) {
        if (!first)
            s += ", ";
        first = false;
        s += num(e);
    }
    return s + "]";
}

std::string jmat(const Matrix& m) {
    std::string s = "[";
    for (Size i = 0; i < m.rows(); ++i) {
        if (i != 0)
            s += ", ";
        s += "[";
        for (Size j = 0; j < m.columns(); ++j) {
            if (j != 0)
                s += ", ";
            s += num(m[i][j]);
        }
        s += "]";
    }
    return s + "]";
}

// Dense dump of a boost::ublas compressed_matrix (read through the *const*
// operator(), so structural zeros read back as 0.0 without being inserted).
std::string jsparse(const SparseMatrix& m) {
    std::string s = "[";
    for (Size i = 0; i < m.size1(); ++i) {
        if (i != 0)
            s += ", ";
        s += "[";
        for (Size j = 0; j < m.size2(); ++j) {
            if (j != 0)
                s += ", ";
            s += num(m(i, j));
        }
        s += "]";
    }
    return s + "]";
}

std::string jstr(const std::string& s) { return "\"" + s + "\""; }

std::string jsize(Size n) { return std::to_string(n); }

using KV = std::vector<std::pair<std::string, std::string> >;

std::string jobj(const KV& kv) {
    std::string s = "{";
    for (Size i = 0; i < kv.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += jstr(kv[i].first) + ": " + kv[i].second;
    }
    return s + "}";
}

std::string joinSection(const std::vector<std::string>& items) {
    std::string s = "[\n";
    for (Size i = 0; i < items.size(); ++i) {
        s += "    " + items[i];
        if (i + 1 != items.size())
            s += ",";
        s += "\n";
    }
    return s + "  ]";
}

// --------------------------------------------------------------- inputs ----

Matrix mk(const std::vector<std::vector<Real> >& v) {
    Matrix m(v.size(), v[0].size());
    for (Size i = 0; i < v.size(); ++i)
        for (Size j = 0; j < v[0].size(); ++j)
            m[i][j] = v[i][j];
    return m;
}

Array ak(const std::vector<Real>& v) {
    Array a(v.size());
    for (Size i = 0; i < v.size(); ++i)
        a[i] = v[i];
    return a;
}

SparseMatrix toSparse(const Matrix& m) {
    SparseMatrix s(m.rows(), m.columns());
    for (Size i = 0; i < m.rows(); ++i)
        for (Size j = 0; j < m.columns(); ++j)
            if (m[i][j] != 0.0)
                s(i, j) = m[i][j];
    return s;
}

// Tridiagonal (sub, diag, super) of order n.
Matrix tridiag(Size n, Real sub, Real diag, Real super) {
    Matrix m(n, n, 0.0);
    for (Size i = 0; i < n; ++i) {
        m[i][i] = diag;
        if (i > 0)
            m[i][i - 1] = sub;
        if (i + 1 < n)
            m[i][i + 1] = super;
    }
    return m;
}

// 5-point Laplacian on an nx by ny grid (the shape SparseILUPreconditioner is
// actually used on inside the FDM stack).
Matrix laplacian2d(Size nx, Size ny) {
    const Size n = nx * ny;
    Matrix m(n, n, 0.0);
    for (Size iy = 0; iy < ny; ++iy) {
        for (Size ix = 0; ix < nx; ++ix) {
            const Size i = iy * nx + ix;
            m[i][i] = 4.0;
            if (ix > 0)
                m[i][i - 1] = -1.0;
            if (ix + 1 < nx)
                m[i][i + 1] = -1.0;
            if (iy > 0)
                m[i][i - nx] = -1.0;
            if (iy + 1 < ny)
                m[i][i + nx] = -1.0;
        }
    }
    return m;
}

// A matrix-vector product recorded call by call: the log of Norm2 of every
// argument handed to the operator pins the *trajectory* of the Krylov
// iteration, not merely the point it lands on.
struct RecordingMult {
    Matrix a;
    std::vector<Real>* log;
    Array operator()(const Array& x) const {
        log->push_back(Norm2(x));
        return a * x;
    }
};

struct RecordingIlu {
    const SparseILUPreconditioner* ilu;
    std::vector<Real>* log;
    Array operator()(const Array& x) const {
        log->push_back(Norm2(x));
        return ilu->apply(x);
    }
};

struct RecordingDiag {
    Array d;
    std::vector<Real>* log;
    Array operator()(const Array& x) const {
        log->push_back(Norm2(x));
        Array y(x.size());
        for (Size i = 0; i < x.size(); ++i)
            y[i] = x[i] / d[i];
        return y;
    }
};

// ------------------------------------------------- SymmetricSchur cases ----

std::string schurCase(const std::string& name, const Matrix& s) {
    SymmetricSchurDecomposition d(s);
    KV kv;
    kv.emplace_back("name", jstr(name));
    kv.emplace_back("matrix", jmat(s));
    kv.emplace_back("eigenvalues", jarr(d.eigenvalues()));
    kv.emplace_back("eigenvectors", jmat(d.eigenvectors()));
    return jobj(kv);
}

std::vector<std::string> symmetricSchur() {
    std::vector<std::string> out;
    out.push_back(schurCase("sym_1x1", mk({{3.5}})));
    out.push_back(schurCase("sym_2x2", mk({{2.0, 1.0}, {1.0, 2.0}})));
    // Already diagonal: sum == 0 on the very first pass, so the do/while exits
    // without a single rotation.
    out.push_back(schurCase("diagonal_3x3",
                            mk({{3.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 7.0}})));
    out.push_back(schurCase("sym_3x3_generic",
                            mk({{4.0, 1.0, -2.0}, {1.0, 3.0, 0.5}, {-2.0, 0.5, 6.0}})));
    // Repeated eigenvalue 2 with a structurally exact tie: exercises the
    // std::greater<> tie-break on the *eigenvector*, not just on the value.
    out.push_back(schurCase("sym_4x4_degenerate", mk({{2.0, 0.0, 0.0, 0.0},
                                                      {0.0, 2.0, 0.0, 0.0},
                                                      {0.0, 0.0, 5.0, 1.0},
                                                      {0.0, 0.0, 1.0, 5.0}})));
    // Rank one: the two zero eigenvalues are pushed to exactly 0.0 by the
    // |lambda/maxEv| < 1e-16 round-off guard. Their eigenvectors span a
    // two-dimensional null space, so the *basis* the sweep lands on is decided
    // by rounding, not by the algorithm — the test asserts the eigenvalues
    // only for this case.
    out.push_back(schurCase("rank_one_3x3",
                            mk({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}})));
    // Rank two with a *simple* zero eigenvalue: same round-off guard, but the
    // null direction is unique, so the eigenvectors are fully comparable.
    out.push_back(schurCase("rank_deficient_3x3",
                            mk({{5.0, 2.0, 0.0}, {2.0, 5.0, 0.0}, {0.0, 0.0, 0.0}})));
    out.push_back(schurCase("sym_4x4_indefinite", mk({{1.0, 2.0, 3.0, 4.0},
                                                      {2.0, -3.0, 0.5, 1.5},
                                                      {3.0, 0.5, 2.0, -1.0},
                                                      {4.0, 1.5, -1.0, 5.0}})));
    // Hilbert(5): eigenvalues spread over 6 decades; drives the sweeps past
    // ite > 5, where the threshold drops to 0 and the epsPrec skip kicks in.
    {
        Matrix h(5, 5);
        for (Size i = 0; i < 5; ++i)
            for (Size j = 0; j < 5; ++j)
                h[i][j] = 1.0 / Real(i + j + 1);
        out.push_back(schurCase("hilbert_5x5", h));
    }
    // A covariance-shaped matrix, the way market models feed it in.
    out.push_back(schurCase("covariance_4x4", mk({{0.04, 0.012, 0.008, 0.004},
                                                  {0.012, 0.09, 0.018, 0.009},
                                                  {0.008, 0.018, 0.16, 0.024},
                                                  {0.004, 0.009, 0.024, 0.25}})));
    return out;
}

// ------------------------------------------------------ Cholesky cases ----

std::string choleskyCase(const std::string& name,
                         const Matrix& s,
                         bool flexible,
                         const Array& b) {
    const Matrix l = CholeskyDecomposition(s, flexible);
    // CholeskySolveFor divides by L[i][i]; a semi-definite factor has a zero
    // there and the solve is not defined, so it is recorded as null.
    bool solvable = true;
    for (Size i = 0; i < l.rows(); ++i)
        if (l[i][i] == 0.0)
            solvable = false;

    KV kv;
    kv.emplace_back("name", jstr(name));
    kv.emplace_back("matrix", jmat(s));
    kv.emplace_back("flexible", flexible ? "true" : "false");
    kv.emplace_back("L", jmat(l));
    kv.emplace_back("b", jarr(b));
    kv.emplace_back("solve_for", solvable ? jarr(CholeskySolveFor(l, b)) : "null");
    return jobj(kv);
}

std::vector<std::string> choleskyCases() {
    std::vector<std::string> out;
    const Matrix spd3 = mk({{4.0, 2.0, 2.0}, {2.0, 3.0, 1.0}, {2.0, 1.0, 5.0}});
    out.push_back(choleskyCase("spd_3x3", spd3, false, ak({1.0, 2.0, 3.0})));
    out.push_back(choleskyCase("spd_3x3_flexible", spd3, true, ak({1.0, 2.0, 3.0})));
    out.push_back(choleskyCase("covariance_4x4",
                               mk({{0.04, 0.012, 0.008, 0.004},
                                   {0.012, 0.09, 0.018, 0.009},
                                   {0.008, 0.018, 0.16, 0.024},
                                   {0.004, 0.009, 0.024, 0.25}}),
                               false, ak({1.0, -1.0, 0.5, 2.0})));
    // Positive *semi*-definite (row 3 == row 1 + row 2): flexible=true takes
    // sqrt(max(sum, 0)) and then the close_enough(L[i][i], 0) guard writes an
    // exact 0.0 instead of dividing.
    out.push_back(choleskyCase("psd_rank2_3x3",
                               mk({{1.0, 1.0, 2.0}, {1.0, 2.0, 3.0}, {2.0, 3.0, 5.0}}),
                               true, ak({1.0, 2.0, 3.0})));
    // Indefinite: only reachable with flexible=true, where the negative pivot
    // is clamped to zero.
    out.push_back(choleskyCase("indefinite_2x2", mk({{0.0, 1.0}, {1.0, 0.0}}), true,
                               ak({1.0, 2.0})));
    // Only the *upper* triangle of the input is read: this asymmetric matrix
    // must give the same factor as its symmetrised-from-above counterpart,
    // which is where scipy's lower=True convention would diverge.
    out.push_back(choleskyCase("upper_triangle_only",
                               mk({{4.0, 2.0, 2.0}, {-7.0, 3.0, 1.0}, {13.0, 99.0, 5.0}}),
                               false, ak({1.0, 2.0, 3.0})));
    return out;
}

// ------------------------------------------------------ TqrEigen cases ----

const char* calcName(TqrEigenDecomposition::EigenVectorCalculation c) {
    switch (c) {
      case TqrEigenDecomposition::WithEigenVector:
        return "WithEigenVector";
      case TqrEigenDecomposition::WithoutEigenVector:
        return "WithoutEigenVector";
      default:
        return "OnlyFirstRowEigenVector";
    }
}

const char* shiftName(TqrEigenDecomposition::ShiftStrategy s) {
    switch (s) {
      case TqrEigenDecomposition::NoShift:
        return "NoShift";
      case TqrEigenDecomposition::Overrelaxation:
        return "Overrelaxation";
      default:
        return "CloseEigenValue";
    }
}

std::string tqrCase(const std::string& name,
                    const Array& diag,
                    const Array& sub,
                    TqrEigenDecomposition::EigenVectorCalculation calc,
                    TqrEigenDecomposition::ShiftStrategy strategy) {
    TqrEigenDecomposition d(diag, sub, calc, strategy);
    KV kv;
    kv.emplace_back("name", jstr(name));
    kv.emplace_back("diag", jarr(diag));
    kv.emplace_back("sub", jarr(sub));
    kv.emplace_back("calc", jstr(calcName(calc)));
    kv.emplace_back("strategy", jstr(shiftName(strategy)));
    kv.emplace_back("eigenvalues", jarr(d.eigenvalues()));
    kv.emplace_back("eigenvectors", jmat(d.eigenvectors()));
    kv.emplace_back("iterations", jsize(d.iterations()));
    return jobj(kv);
}

std::vector<std::string> tqrEigen() {
    const TqrEigenDecomposition::EigenVectorCalculation calcs[3] = {
        TqrEigenDecomposition::WithEigenVector,
        TqrEigenDecomposition::WithoutEigenVector,
        TqrEigenDecomposition::OnlyFirstRowEigenVector};
    const TqrEigenDecomposition::ShiftStrategy shifts[3] = {
        TqrEigenDecomposition::NoShift, TqrEigenDecomposition::Overrelaxation,
        TqrEigenDecomposition::CloseEigenValue};

    struct System {
        std::string name;
        Array diag;
        Array sub;
        // NoShift is the unshifted QR iteration, and offDiagIsZero() is an
        // exact floating-point equality test. On a spectrum containing a
        // +-lambda pair the off-diagonal ratio is exactly 1 and the while loop
        // never terminates. gauss_hermite_5 is precisely that case (alpha == 0
        // everywhere => spectrum symmetric about the origin), so NoShift is
        // skipped for it: the C++ reference does not terminate either.
        bool allowNoShift;
    };

    std::vector<System> systems;
    systems.push_back({"t_1x1", ak({2.75}), ak({}), true});
    systems.push_back(
        {"t_4_uniform_sub", ak({1.0, 2.0, 3.0, 4.0}), ak({0.5, 0.5, 0.5}), true});
    // Already diagonal: offDiagIsZero is true straight away, iterations == 0.
    systems.push_back({"t_3_zero_sub", ak({2.0, 2.0, 2.0}), ak({0.0, 0.0}), true});
    // A decoupling zero in the middle of the sub-diagonal.
    systems.push_back(
        {"t_4_split", ak({1.0, 1.0, 1.0, 1.0}), ak({1.0, 0.0, 1.0}), true});
    // The Golub-Welsch Jacobi matrix for 5-point Gauss-Hermite: alpha == 0,
    // beta(i) == i/2. This is exactly what GaussianQuadrature feeds in, with
    // OnlyFirstRowEigenVector + Overrelaxation.
    systems.push_back(
        {"gauss_hermite_5", ak({0.0, 0.0, 0.0, 0.0, 0.0}),
         ak({std::sqrt(0.5), std::sqrt(1.0), std::sqrt(1.5), std::sqrt(2.0)}), false});
    // Gauss-Laguerre(6): alpha(i) = 2i+1, beta(i) = i*i.
    {
        std::vector<Real> d(6), s(5);
        for (Size i = 0; i < 6; ++i)
            d[i] = 2.0 * Real(i) + 1.0;
        for (Size i = 1; i < 6; ++i)
            s[i - 1] = std::sqrt(Real(i) * Real(i));
        systems.push_back({"gauss_laguerre_6", ak(d), ak(s), true});
    }
    systems.push_back({"t_8_graded", ak({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}),
                       ak({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}), true});
    systems.push_back({"t_5_negative", ak({-2.0, 1.0, -0.5, 3.0, 0.25}),
                       ak({0.75, -0.5, 0.25, 1.5}), true});

    std::vector<std::string> out;
    for (const auto& sys : systems)
        for (auto calc : calcs)
            for (auto shift : shifts) {
                if (shift == TqrEigenDecomposition::NoShift && !sys.allowNoShift)
                    continue;
                out.push_back(tqrCase(sys.name, sys.diag, sys.sub, calc, shift));
            }
    return out;
}

// ----------------------------------------------------------- SVD cases ----

std::string svdCase(const std::string& name, const Matrix& m, const Array& b) {
    SVD svd(m);
    const Real cond = svd.cond();
    KV kv;
    kv.emplace_back("name", jstr(name));
    kv.emplace_back("matrix", jmat(m));
    kv.emplace_back("U", jmat(svd.U()));
    kv.emplace_back("V", jmat(svd.V()));
    kv.emplace_back("singular_values", jarr(svd.singularValues()));
    kv.emplace_back("S", jmat(svd.S()));
    kv.emplace_back("norm2", num(svd.norm2()));
    kv.emplace_back("cond", num(cond));
    kv.emplace_back("rank", jsize(svd.rank()));
    kv.emplace_back("b", jarr(b));
    kv.emplace_back("solve_for", jarr(svd.solveFor(b)));
    return jobj(kv);
}

std::vector<std::string> svdCases() {
    std::vector<std::string> out;
    out.push_back(svdCase("tall_4x3",
                          mk({{1.0, 2.0, 3.0},
                              {4.0, 5.0, 6.0},
                              {7.0, 8.0, 10.0},
                              {2.0, -1.0, 0.5}}),
                          ak({1.0, 2.0, 3.0, 4.0})));
    // rows < columns -> the transpose_ path; U() and V() come back swapped.
    out.push_back(svdCase("wide_3x5",
                          mk({{1.0, 2.0, 3.0, 4.0, 5.0},
                              {5.0, 4.0, 3.0, 2.0, 1.0},
                              {1.0, 0.0, -1.0, 2.0, 0.5}}),
                          ak({1.0, -2.0, 0.5})));
    out.push_back(svdCase("square_4x4",
                          mk({{4.0, 1.0, -2.0, 0.5},
                              {1.0, 3.0, 0.5, -1.0},
                              {-2.0, 0.5, 6.0, 2.0},
                              {0.5, -1.0, 2.0, 1.5}}),
                          ak({1.0, 2.0, -1.0, 0.5})));
    // Column 3 == column 1 + column 2: an exactly singular direction, which
    // drives the kase-2 "split at negligible s(k)" deflation.
    out.push_back(svdCase("rank_deficient_4x3",
                          mk({{1.0, 2.0, 3.0},
                              {4.0, 5.0, 9.0},
                              {7.0, 8.0, 15.0},
                              {2.0, -1.0, 1.0}}),
                          ak({1.0, 2.0, 3.0, 4.0})));
    out.push_back(svdCase("column_4x1", mk({{3.0}, {-4.0}, {1.0}, {0.5}}),
                          ak({1.0, 2.0, 3.0, 4.0})));
    out.push_back(svdCase("row_1x4", mk({{3.0, -4.0, 1.0, 0.5}}), ak({2.0})));
    out.push_back(svdCase("symmetric_3x3",
                          mk({{4.0, 1.0, -2.0}, {1.0, 3.0, 0.5}, {-2.0, 0.5, 6.0}}),
                          ak({1.0, 1.0, 1.0})));
    out.push_back(svdCase("zero_row_3x3",
                          mk({{1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, {4.0, 5.0, 7.0}}),
                          ak({1.0, 2.0, 3.0})));
    // Hilbert(4): singular values spanning 5 decades.
    {
        Matrix h(4, 4);
        for (Size i = 0; i < 4; ++i)
            for (Size j = 0; j < 4; ++j)
                h[i][j] = 1.0 / Real(i + j + 1);
        out.push_back(svdCase("hilbert_4x4", h, ak({1.0, 0.0, 0.0, 0.0})));
    }
    return out;
}

// ------------------------------------------------------- BiCGstab cases ----

std::string bicgCase(const std::string& name,
                     const Matrix& a,
                     const Array& b,
                     const Array& x0,
                     Size maxIter,
                     Real relTol,
                     const std::string& precond) {
    std::vector<Real> aLog, mLog;
    SparseILUPreconditioner ilu(toSparse(a));
    Array diag(a.rows());
    for (Size i = 0; i < a.rows(); ++i)
        diag[i] = a[i][i];

    BiCGstab::MatrixMult mm = RecordingMult{a, &aLog};
    BiCGstab::MatrixMult pre;
    if (precond == "ilu")
        pre = RecordingIlu{&ilu, &mLog};
    else if (precond == "jacobi")
        pre = RecordingDiag{diag, &mLog};

    BiCGstab solver(mm, maxIter, relTol, pre);
    const BiCGStabResult r = solver.solve(b, x0);

    KV kv;
    kv.emplace_back("name", jstr(name));
    kv.emplace_back("matrix", jmat(a));
    kv.emplace_back("b", jarr(b));
    kv.emplace_back("x0", jarr(x0));
    kv.emplace_back("max_iter", jsize(maxIter));
    kv.emplace_back("rel_tol", num(relTol));
    kv.emplace_back("preconditioner", jstr(precond));
    kv.emplace_back("iterations", jsize(r.iterations));
    kv.emplace_back("error", num(r.error));
    kv.emplace_back("x", jarr(r.x));
    kv.emplace_back("a_call_norms", jvec(aLog));
    kv.emplace_back("m_call_norms", jvec(mLog));
    return jobj(kv);
}

std::vector<std::string> bicgstabCases() {
    std::vector<std::string> out;
    const Matrix t8 = tridiag(8, -1.0, 2.5, -1.0);
    const Array b8 = ak({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    const Array none = Array();

    out.push_back(bicgCase("tridiag_8", t8, b8, none, 100, 1e-10, "none"));
    out.push_back(bicgCase("tridiag_8_jacobi", t8, b8, none, 100, 1e-10, "jacobi"));
    out.push_back(bicgCase("tridiag_8_ilu", t8, b8, none, 100, 1e-10, "ilu"));
    out.push_back(bicgCase("tridiag_8_loose", t8, b8, none, 100, 1e-4, "none"));
    out.push_back(bicgCase("tridiag_8_x0", t8, b8,
                           ak({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}), 100, 1e-10,
                           "none"));
    // b == 0 -> the {0, 0.0, b} short circuit, no A_ call at all.
    out.push_back(bicgCase("zero_rhs", t8, Array(8, 0.0), none, 100, 1e-10, "none"));
    // Non-symmetric: BiCG-Stab's actual reason for existing.
    out.push_back(bicgCase("nonsym_6",
                           mk({{4.0, 1.0, 0.0, 0.0, 0.5, 0.0},
                               {-2.0, 5.0, 1.0, 0.0, 0.0, 0.0},
                               {0.0, -1.0, 6.0, 2.0, 0.0, 0.0},
                               {0.0, 0.0, -1.5, 4.0, 1.0, 0.0},
                               {0.3, 0.0, 0.0, -1.0, 5.0, 2.0},
                               {0.0, 0.0, 0.0, 0.0, -2.0, 3.0}}),
                           ak({1.0, -1.0, 2.0, 0.5, 3.0, -2.0}), none, 100, 1e-10,
                           "none"));
    out.push_back(bicgCase("laplacian_3x3_ilu", laplacian2d(3, 3),
                           ak({1.0, 0.0, -1.0, 2.0, 0.5, 0.0, -0.5, 1.0, 3.0}), none, 100,
                           1e-12, "ilu"));
    return out;
}

// ---------------------------------------------------------- GMRES cases ----

std::string gmresCase(const std::string& name,
                      const Matrix& a,
                      const Array& b,
                      const Array& x0,
                      Size maxIter,
                      Real relTol,
                      const std::string& precond,
                      Size restart) {
    std::vector<Real> aLog, mLog;
    SparseILUPreconditioner ilu(toSparse(a));
    Array diag(a.rows());
    for (Size i = 0; i < a.rows(); ++i)
        diag[i] = a[i][i];

    GMRES::MatrixMult mm = RecordingMult{a, &aLog};
    GMRES::MatrixMult pre;
    if (precond == "ilu")
        pre = RecordingIlu{&ilu, &mLog};
    else if (precond == "jacobi")
        pre = RecordingDiag{diag, &mLog};

    GMRES solver(mm, maxIter, relTol, pre);
    const GMRESResult r =
        (restart == 0) ? solver.solve(b, x0) : solver.solveWithRestart(restart, b, x0);

    KV kv;
    kv.emplace_back("name", jstr(name));
    kv.emplace_back("matrix", jmat(a));
    kv.emplace_back("b", jarr(b));
    kv.emplace_back("x0", jarr(x0));
    kv.emplace_back("max_iter", jsize(maxIter));
    kv.emplace_back("rel_tol", num(relTol));
    kv.emplace_back("preconditioner", jstr(precond));
    kv.emplace_back("restart", jsize(restart));
    kv.emplace_back("errors", jlist(r.errors));
    kv.emplace_back("x", jarr(r.x));
    kv.emplace_back("a_call_norms", jvec(aLog));
    kv.emplace_back("m_call_norms", jvec(mLog));
    return jobj(kv);
}

std::vector<std::string> gmresCases() {
    std::vector<std::string> out;
    const Matrix t8 = tridiag(8, -1.0, 2.5, -1.0);
    const Array b8 = ak({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    const Array none = Array();

    out.push_back(gmresCase("tridiag_8", t8, b8, none, 10, 1e-10, "none", 0));
    out.push_back(gmresCase("tridiag_8_jacobi", t8, b8, none, 10, 1e-10, "jacobi", 0));
    out.push_back(gmresCase("tridiag_8_ilu", t8, b8, none, 10, 1e-10, "ilu", 0));
    // maxIter 3 on a 6x6 system forces solveWithRestart to actually restart:
    // six cycles, and the concatenated error list is 23 entries long.
    out.push_back(gmresCase("tridiag_6_restart", tridiag(6, -1.0, 4.0, -1.0),
                            ak({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), none, 3, 1e-10, "none",
                            6));
    out.push_back(gmresCase("tridiag_8_x0", t8, b8,
                            ak({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}), 10, 1e-10,
                            "none", 0));
    out.push_back(gmresCase("zero_rhs", t8, Array(8, 0.0), none, 10, 1e-10, "none", 0));
    out.push_back(gmresCase("nonsym_6",
                            mk({{4.0, 1.0, 0.0, 0.0, 0.5, 0.0},
                                {-2.0, 5.0, 1.0, 0.0, 0.0, 0.0},
                                {0.0, -1.0, 6.0, 2.0, 0.0, 0.0},
                                {0.0, 0.0, -1.5, 4.0, 1.0, 0.0},
                                {0.3, 0.0, 0.0, -1.0, 5.0, 2.0},
                                {0.0, 0.0, 0.0, 0.0, -2.0, 3.0}}),
                            ak({1.0, -1.0, 2.0, 0.5, 3.0, -2.0}), none, 10, 1e-10, "none",
                            0));
    out.push_back(gmresCase("laplacian_3x3_ilu", laplacian2d(3, 3),
                            ak({1.0, 0.0, -1.0, 2.0, 0.5, 0.0, -0.5, 1.0, 3.0}), none, 12,
                            1e-12, "ilu", 0));
    return out;
}

// ------------------------------------------------------ SparseILU cases ----

std::string iluCase(const std::string& name,
                    const Matrix& a,
                    Integer lfil,
                    const std::vector<Array>& rhs) {
    SparseILUPreconditioner ilu(toSparse(a), lfil);
    std::string applied = "[";
    std::string rhsJson = "[";
    for (Size i = 0; i < rhs.size(); ++i) {
        if (i != 0) {
            applied += ", ";
            rhsJson += ", ";
        }
        applied += jarr(ilu.apply(rhs[i]));
        rhsJson += jarr(rhs[i]);
    }
    applied += "]";
    rhsJson += "]";

    KV kv;
    kv.emplace_back("name", jstr(name));
    kv.emplace_back("matrix", jmat(a));
    kv.emplace_back("lfil", std::to_string(lfil));
    kv.emplace_back("L", jsparse(ilu.L()));
    kv.emplace_back("U", jsparse(ilu.U()));
    kv.emplace_back("rhs", rhsJson);
    kv.emplace_back("applied", applied);
    return jobj(kv);
}

std::vector<std::string> sparseIluCases() {
    std::vector<std::string> out;
    const Array b6 = ak({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const Array c6 = ak({-1.0, 0.5, 0.0, 2.0, -3.0, 1.0});
    out.push_back(iluCase("tridiag_6", tridiag(6, -1.0, 4.0, -1.0), 1, {b6, c6}));
    out.push_back(iluCase("tridiag_6_lfil0", tridiag(6, -1.0, 4.0, -1.0), 0, {b6}));
    out.push_back(iluCase("tridiag_6_lfil3", tridiag(6, -1.0, 4.0, -1.0), 3, {b6}));
    out.push_back(iluCase("asym_tridiag_6", tridiag(6, -2.0, 5.0, 1.0), 1, {b6, c6}));
    {
        const Array b9 = ak({1.0, 0.0, -1.0, 2.0, 0.5, 0.0, -0.5, 1.0, 3.0});
        out.push_back(iluCase("laplacian_3x3", laplacian2d(3, 3), 1, {b9}));
        out.push_back(iluCase("laplacian_3x3_lfil2", laplacian2d(3, 3), 2, {b9}));
    }
    out.push_back(iluCase("dense_5",
                          mk({{10.0, 1.0, 0.5, 0.0, 0.25},
                              {2.0, 9.0, 1.0, 0.5, 0.0},
                              {0.0, 3.0, 8.0, 1.5, 0.5},
                              {0.5, 0.0, 2.0, 7.0, 1.0},
                              {0.25, 0.5, 0.0, 1.5, 6.0}}),
                          1, {ak({1.0, 2.0, 3.0, 4.0, 5.0})}));
    out.push_back(iluCase("diagonal_4",
                          mk({{2.0, 0.0, 0.0, 0.0},
                              {0.0, 3.0, 0.0, 0.0},
                              {0.0, 0.0, 4.0, 0.0},
                              {0.0, 0.0, 0.0, 5.0}}),
                          1, {ak({1.0, 1.0, 1.0, 1.0})}));
    return out;
}

// ----------------------------------------------------- Householder cases ----

std::vector<std::string> householderTransformationCases() {
    std::vector<std::string> out;
    const std::vector<std::pair<std::string, std::pair<Array, Array> > > cases = {
        {"axis_3", {ak({1.0, 0.0, 0.0}), ak({1.0, 2.0, 3.0})}},
        {"unit_3", {ak({0.6, 0.8, 0.0}), ak({1.0, 2.0, 3.0})}},
        {"nonunit_4", {ak({1.0, -2.0, 0.5, 3.0}), ak({2.0, -1.0, 0.0, 4.0})}},
        {"scaled_2", {ak({3.0, 4.0}), ak({-1.0, 7.0})}},
    };
    for (const auto& c : cases) {
        HouseholderTransformation h(c.second.first);
        KV kv;
        kv.emplace_back("name", jstr(c.first));
        kv.emplace_back("v", jarr(c.second.first));
        kv.emplace_back("x", jarr(c.second.second));
        kv.emplace_back("matrix", jmat(h.getMatrix()));
        kv.emplace_back("applied", jarr(h(c.second.second)));
        out.push_back(jobj(kv));
    }
    return out;
}

std::vector<std::string> householderReflectionCases() {
    std::vector<std::string> out;
    const Real s = 1.0 / std::sqrt(3.0);
    const std::vector<std::pair<std::string, std::pair<Array, Array> > > cases = {
        // eps == 0 exactly -> the "return zero vector" branch.
        {"parallel_3", {ak({1.0, 0.0, 0.0}), ak({3.0, 0.0, 0.0})}},
        // eps ~ 2e-6 -> the fourth-order series branch.
        {"near_parallel_3", {ak({1.0, 0.0, 0.0}), ak({1.0, 1e-3, 1e-3})}},
        // eps ~ 2e-8, still the series branch but a decade closer to the cut.
        {"very_near_parallel_3", {ak({1.0, 0.0, 0.0}), ak({1.0, 1e-4, 1e-4})}},
        // eps just under 1e-4 -> the series branch at its least accurate.
        {"series_edge_3", {ak({1.0, 0.0, 0.0}), ak({1.0, 0.006, 0.006})}},
        // eps > 1e-4 -> the exact a - |a| e branch.
        {"general_3", {ak({1.0, 0.0, 0.0}), ak({1.0, 2.0, 3.0})}},
        {"general_tilted_3", {ak({s, s, s}), ak({1.0, 2.0, 3.0})}},
        {"general_4", {ak({0.0, 1.0, 0.0, 0.0}), ak({2.0, -1.0, 0.5, 3.0})}},
        // Negative projection: aDotE < 0, so a - na*e is the *far* reflection.
        {"antiparallel_ish_3", {ak({1.0, 0.0, 0.0}), ak({-2.0, 0.5, 0.25})}},
    };
    for (const auto& c : cases) {
        HouseholderReflection h(c.second.first);
        KV kv;
        kv.emplace_back("name", jstr(c.first));
        kv.emplace_back("e", jarr(c.second.first));
        kv.emplace_back("a", jarr(c.second.second));
        kv.emplace_back("reflection_vector", jarr(h.reflectionVector(c.second.second)));
        kv.emplace_back("applied", jarr(h(c.second.second)));
        out.push_back(jobj(kv));
    }
    return out;
}

// ---------------------------------------------------- getCovariance cases ----

std::vector<std::string> covarianceCases() {
    std::vector<std::string> out;

    std::vector<std::pair<std::string, std::pair<std::vector<Real>, Matrix> > > cases;
    cases.emplace_back("three_asset",
                       std::make_pair(std::vector<Real>{0.2, 0.3, 0.5},
                                      mk({{1.0, 0.35, -0.2},
                                          {0.35, 1.0, 0.6},
                                          {-0.2, 0.6, 1.0}})));
    cases.emplace_back("two_asset", std::make_pair(std::vector<Real>{0.1, 0.4},
                                                   mk({{1.0, 0.9}, {0.9, 1.0}})));
    cases.emplace_back("four_asset_identity",
                       std::make_pair(std::vector<Real>{0.15, 0.25, 0.35, 0.45},
                                      mk({{1.0, 0.0, 0.0, 0.0},
                                          {0.0, 1.0, 0.0, 0.0},
                                          {0.0, 0.0, 1.0, 0.0},
                                          {0.0, 0.0, 0.0, 1.0}})));
    // Slightly asymmetric within the default 1e-12 tolerance: getCovariance
    // symmetrises with 0.5*(c[i][j] + c[j][i]).
    cases.emplace_back("nearly_symmetric",
                       std::make_pair(std::vector<Real>{0.2, 0.3},
                                      mk({{1.0, 0.5}, {0.5 + 4e-13, 1.0}})));

    for (const auto& c : cases) {
        const Matrix cov = getCovariance(c.second.first.begin(), c.second.first.end(),
                                         c.second.second);
        const CovarianceDecomposition dec(cov);
        KV kv;
        kv.emplace_back("name", jstr(c.first));
        kv.emplace_back("std_devs", jvec(c.second.first));
        kv.emplace_back("correlation", jmat(c.second.second));
        kv.emplace_back("covariance", jmat(cov));
        kv.emplace_back("variances", jarr(dec.variances()));
        kv.emplace_back("standard_deviations", jarr(dec.standardDeviations()));
        kv.emplace_back("correlation_matrix", jmat(dec.correlationMatrix()));
        out.push_back(jobj(kv));
    }

    // A covariance matrix that is *not* a rescaled correlation matrix, fed
    // straight to CovarianceDecomposition.
    {
        const Matrix cov = mk({{0.04, 0.012, 0.008}, {0.012, 0.09, 0.018},
                               {0.008, 0.018, 0.16}});
        const CovarianceDecomposition dec(cov);
        KV kv;
        kv.emplace_back("name", jstr("decomposition_only"));
        kv.emplace_back("std_devs", jvec(std::vector<Real>()));
        kv.emplace_back("correlation", jmat(Matrix(0, 0)));
        kv.emplace_back("covariance", jmat(cov));
        kv.emplace_back("variances", jarr(dec.variances()));
        kv.emplace_back("standard_deviations", jarr(dec.standardDeviations()));
        kv.emplace_back("correlation_matrix", jmat(dec.correlationMatrix()));
        out.push_back(jobj(kv));
    }
    return out;
}

// --------------------------------------------------- tapcorrelations cases ----

std::string tapCorrelations() {
    // matrixSize 5, rank 3 -> (rank-1)*(2*matrixSize-rank) == 14 == 2*7 angles.
    const Array angles7 = ak({0.3, 0.7, 1.1, 0.45, 1.35, 0.9, 0.15});
    const Array x7 = ak({0.4, -1.2, 0.8, 2.5, -0.3, 1.7, 0.05});
    const Array angles4 = ak({0.25, 0.6, 1.0, 1.4});
    const Array x4 = ak({0.5, -0.75, 1.25, 0.1});
    const Array rank3Params = ak({0.6, 2.5, -0.35});

    KV kv;
    kv.emplace_back("angles7", jarr(angles7));
    kv.emplace_back("x7", jarr(x7));
    kv.emplace_back("angles4", jarr(angles4));
    kv.emplace_back("x4", jarr(x4));
    kv.emplace_back("rank3_params", jarr(rank3Params));
    kv.emplace_back("triangular_5_3", jmat(triangularAnglesParametrization(angles7, 5, 3)));
    kv.emplace_back("triangular_unconstrained_5_3",
                    jmat(triangularAnglesParametrizationUnconstrained(x7, 5, 3)));
    kv.emplace_back("lmm_triangular_5",
                    jmat(lmmTriangularAnglesParametrization(angles4, 5, 5)));
    kv.emplace_back("lmm_triangular_unconstrained_5",
                    jmat(lmmTriangularAnglesParametrizationUnconstrained(x4, 5, 5)));
    kv.emplace_back("rank_three_6",
                    jmat(triangularAnglesParametrizationRankThree(0.6, 2.5, -0.35, 6)));
    kv.emplace_back("rank_three_vectorial_6",
                    jmat(triangularAnglesParametrizationRankThreeVectorial(rank3Params, 6)));
    // rank == matrixSize: (n-1)*(2n-n) == n*(n-1) == 2 * (n*(n-1)/2) angles.
    {
        const Array angles10 =
            ak({0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 0.35, 0.55, 0.75});
        kv.emplace_back("angles10", jarr(angles10));
        kv.emplace_back("triangular_5_5",
                        jmat(triangularAnglesParametrization(angles10, 5, 5)));
    }
    return jobj(kv);
}

std::vector<std::string> frobeniusCases() {
    std::vector<std::string> out;
    const Matrix target = mk({{1.0, 0.85, 0.72, 0.61, 0.55},
                              {0.85, 1.0, 0.88, 0.75, 0.66},
                              {0.72, 0.88, 1.0, 0.9, 0.8},
                              {0.61, 0.75, 0.9, 1.0, 0.92},
                              {0.55, 0.66, 0.8, 0.92, 1.0}});

    struct Probe {
        std::string name;
        std::string parametrisation;
        Array x;
    };

    const std::vector<Probe> probes = {
        {"unconstrained_a", "triangular_unconstrained",
         ak({0.4, -1.2, 0.8, 2.5, -0.3, 1.7, 0.05})},
        {"unconstrained_b", "triangular_unconstrained",
         ak({1.5, 0.25, -0.6, 0.9, 1.1, -2.0, 0.75})},
        {"lmm_unconstrained_a", "lmm_triangular_unconstrained",
         ak({0.5, -0.75, 1.25, 0.1})},
        {"rank_three_a", "rank_three_vectorial", ak({0.6, 2.5, -0.35})},
        {"rank_three_b", "rank_three_vectorial", ak({1.2, 0.8, -1.5})},
    };

    for (const auto& p : probes) {
        std::function<Matrix(const Array&, Size, Size)> f;
        Size rank = 3;
        if (p.parametrisation == "triangular_unconstrained") {
            f = [](const Array& x, Size n, Size r) {
                return triangularAnglesParametrizationUnconstrained(x, n, r);
            };
        } else if (p.parametrisation == "lmm_triangular_unconstrained") {
            f = [](const Array& x, Size n, Size r) {
                return lmmTriangularAnglesParametrizationUnconstrained(x, n, r);
            };
            rank = 5;
        } else {
            f = [](const Array& x, Size n, Size) {
                return triangularAnglesParametrizationRankThreeVectorial(x, n);
            };
        }

        FrobeniusCostFunction cf(target, f, 5, rank);
        KV kv;
        kv.emplace_back("name", jstr(p.name));
        kv.emplace_back("parametrisation", jstr(p.parametrisation));
        kv.emplace_back("rank", jsize(rank));
        kv.emplace_back("matrix_size", jsize(5));
        kv.emplace_back("x", jarr(p.x));
        kv.emplace_back("values", jarr(cf.values(p.x)));
        kv.emplace_back("value", num(cf.value(p.x)));
        out.push_back(jobj(kv));
    }

    KV wrap;
    wrap.emplace_back("target", jmat(target));
    out.insert(out.begin(), jobj(wrap));
    return out;
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"cholesky\": " << joinSection(choleskyCases()) << ",\n";
    std::cout << "  \"symmetric_schur\": " << joinSection(symmetricSchur()) << ",\n";
    std::cout << "  \"tqr_eigen\": " << joinSection(tqrEigen()) << ",\n";
    std::cout << "  \"svd\": " << joinSection(svdCases()) << ",\n";
    std::cout << "  \"bicgstab\": " << joinSection(bicgstabCases()) << ",\n";
    std::cout << "  \"gmres\": " << joinSection(gmresCases()) << ",\n";
    std::cout << "  \"sparse_ilu\": " << joinSection(sparseIluCases()) << ",\n";
    std::cout << "  \"householder_transformation\": "
              << joinSection(householderTransformationCases()) << ",\n";
    std::cout << "  \"householder_reflection\": "
              << joinSection(householderReflectionCases()) << ",\n";
    std::cout << "  \"covariance\": " << joinSection(covarianceCases()) << ",\n";
    std::cout << "  \"tap_correlations\": " << tapCorrelations() << ",\n";
    std::cout << "  \"frobenius_cost_function\": " << joinSection(frobeniusCases())
              << "\n";
    std::cout << "}\n";
    return 0;
}
