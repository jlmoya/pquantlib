// v1.43 reference for QuantLib::Array (ql/math/array.hpp:52) and
// QuantLib::Matrix (ql/math/matrix.hpp:41).
//
// WHY THIS PROBE EXISTS
// ---------------------
// pquantlib does not port Array/Matrix as classes: `Array` and `Matrix` are
// type ALIASES for numpy.typing.NDArray[float64] (pquantlib/math/array.py,
// pquantlib/math/matrix.py), rank-1 and rank-2 respectively. numpy already
// provides every operation QuantLib's hand-rolled wrappers provide, BLAS
// backed.
//
// Allowlisting the two names in migration-harness/check_coverage.py is only
// honest if the linear algebra numpy stands in for actually agrees with C++.
// This probe pins the whole observable surface of both headers:
//
//   Array : element access, unary -, +,-,*,/ (array-array and array-scalar),
//           DotProduct, Norm2, Abs, Sqrt, Log, Exp, Pow
//   Matrix: +,-,*,/ (matrix-matrix, matrix-scalar), Matrix*Matrix,
//           Matrix*Array, Array*Matrix, transpose, outerProduct, inverse,
//           determinant, row/column extraction, diagonal
//
// All reals at 17 significant digits.

#include <ql/math/array.hpp>
#include <ql/math/matrix.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

Array mkArray(std::initializer_list<Real> values) {
    Array a(values.size());
    Size i = 0;
    for (const Real v : values)
        a[i++] = v;
    return a;
}

Matrix mkMatrix(Size rows, Size cols, std::initializer_list<Real> values) {
    Matrix m(rows, cols);
    Size k = 0;
    for (const Real v : values) {
        m[k / cols][k % cols] = v;
        ++k;
    }
    return m;
}

void emitArray(const char* key, const Array& a, bool comma = true) {
    std::cout << "  \"" << key << "\": [";
    for (Size i = 0; i < a.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << a[i];
    }
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitMatrix(const char* key, const Matrix& m, bool comma = true) {
    std::cout << "  \"" << key << "\": [";
    for (Size i = 0; i < m.rows(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << "[";
        for (Size j = 0; j < m.columns(); ++j) {
            if (j != 0) std::cout << ", ";
            std::cout << m[i][j];
        }
        std::cout << "]";
    }
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitReal(const char* key, Real v, bool comma = true) {
    std::cout << "  \"" << key << "\": " << v << (comma ? "," : "") << "\n";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ---------------- Array -------------------------------------------------
    const Array u = mkArray({1.5, -2.25, 3.0, 0.5});
    const Array v = mkArray({-0.5, 4.0, 2.5, -1.25});

    emitArray("array_u", u);
    emitArray("array_v", v);
    emitArray("array_neg_u", -u);
    emitArray("array_u_plus_v", u + v);
    emitArray("array_u_minus_v", u - v);
    emitArray("array_u_times_v", u * v);
    emitArray("array_u_div_v", u / v);
    emitArray("array_u_plus_scalar", u + 2.5);
    emitArray("array_u_minus_scalar", u - 2.5);
    emitArray("array_u_times_scalar", u * 2.5);
    emitArray("array_u_div_scalar", u / 2.5);
    emitArray("array_scalar_minus_u", 2.5 - u);
    emitArray("array_scalar_div_u", 2.5 / u);
    emitReal("array_dot_product", DotProduct(u, v));
    emitReal("array_norm2_u", Norm2(u));
    emitArray("array_abs_u", Abs(u));
    // Sqrt/Log need a positive argument; use Abs(u).
    emitArray("array_sqrt_abs_u", Sqrt(Abs(u)));
    emitArray("array_log_abs_u", Log(Abs(u)));
    emitArray("array_exp_u", Exp(u));
    emitArray("array_pow_abs_u_1_5", Pow(Abs(u), 1.5));

    // ---------------- Matrix ------------------------------------------------
    const Matrix A = mkMatrix(3, 3, {4.0, 1.0, -2.0,
                                     1.0, 5.0, 0.5,
                                     -2.0, 0.5, 3.0});
    const Matrix B = mkMatrix(3, 3, {0.5, -1.0, 2.0,
                                     1.5, 0.25, -0.5,
                                     -3.0, 2.0, 1.0});
    // Non-square, to pin transpose / multiply shape handling.
    const Matrix C = mkMatrix(2, 3, {1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0});

    emitMatrix("matrix_a", A);
    emitMatrix("matrix_b", B);
    emitMatrix("matrix_c", C);
    emitMatrix("matrix_neg_a", -A);
    emitMatrix("matrix_a_plus_b", A + B);
    emitMatrix("matrix_a_minus_b", A - B);
    emitMatrix("matrix_a_times_scalar", A * 2.5);
    emitMatrix("matrix_a_div_scalar", A / 2.5);
    emitMatrix("matrix_a_mul_b", A * B);
    emitMatrix("matrix_c_mul_a", C * A);
    emitMatrix("matrix_transpose_c", transpose(C));
    emitMatrix("matrix_outer_product_u_v", outerProduct(u, v));
    emitMatrix("matrix_inverse_a", inverse(A));
    emitReal("matrix_determinant_a", determinant(A));
    emitReal("matrix_determinant_b", determinant(B));

    const Array w = mkArray({1.0, -2.0, 0.5});
    emitArray("matrix_a_mul_array", A * w);
    emitArray("array_mul_matrix_a", w * A);

    // Row / column / diagonal views.
    {
        Array row1(A.columns());
        for (Size j = 0; j < A.columns(); ++j)
            row1[j] = A[1][j];
        emitArray("matrix_a_row1", row1);

        Array col2(A.rows());
        for (Size i = 0; i < A.rows(); ++i)
            col2[i] = A[i][2];
        emitArray("matrix_a_col2", col2);

        emitArray("matrix_a_diagonal", A.diagonal());
    }

    emitReal("matrix_a_rows", static_cast<Real>(A.rows()));
    emitReal("matrix_c_rows", static_cast<Real>(C.rows()));
    emitReal("matrix_c_columns", static_cast<Real>(C.columns()), /*comma=*/false);

    std::cout << "}\n";
    return 0;
}
