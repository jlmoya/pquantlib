"""Cross-validate the numpy stand-ins for C++ ``Array`` and ``Matrix``.

Probe source: migration-harness/cpp/probes/v143_math_arraymatrix/probe.cpp
Reference:    migration-harness/references/v143/math/arraymatrix.json

pquantlib does not port ``QuantLib::Array`` (ql/math/array.hpp:52) or
``QuantLib::Matrix`` (ql/math/matrix.hpp:41) as classes: they are type
ALIASES for ``numpy.typing.NDArray[numpy.float64]``, rank-1 and rank-2
respectively (``pquantlib/math/array.py``, ``pquantlib/math/matrix.py``).
Both C++ classes are hand-rolled containers whose entire job — element
access, elementwise arithmetic, dot products, matrix products, transpose,
outer product, inverse, determinant — numpy already does, BLAS/LAPACK
backed.

Those two names are therefore allowlisted in
``migration-harness/check_coverage.py``. This module is what makes that
honest: it re-runs the whole observable surface of both headers through
numpy and diffs against C++ v1.43.

Tolerance tier: TIGHT (1e-14 abs / 1e-12 rel). Elementwise arithmetic is
bit-exact; ``inverse`` goes through LAPACK on one side and QuantLib's own
LU on the other, so it is the one operation where last-place differences
are expected — and 1e-12 relative is still far tighter than any use of a
covariance inverse in the library needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/arraymatrix")


def _arr(cpp: dict[str, Any], key: str) -> Array:
    return np.asarray(cpp[key], dtype=np.float64)


def _mat(cpp: dict[str, Any], key: str) -> Matrix:
    return np.asarray(cpp[key], dtype=np.float64)


def _check_arr(got: Array, want: Array) -> None:
    assert got.shape == want.shape
    for g, w in zip(got.tolist(), want.tolist(), strict=True):
        tolerance.tight(float(g), float(w))


def _check_mat(got: Matrix, want: Matrix) -> None:
    assert got.shape == want.shape
    for grow, wrow in zip(got.tolist(), want.tolist(), strict=True):
        for g, w in zip(grow, wrow, strict=True):
            tolerance.tight(float(g), float(w))


# --------------------------------------------------------------------------
# Array — ql/math/array.hpp
# --------------------------------------------------------------------------


@pytest.mark.tight
def test_array_arithmetic_matches_cpp(cpp: dict[str, Any]) -> None:
    u = _arr(cpp, "array_u")
    v = _arr(cpp, "array_v")
    _check_arr(-u, _arr(cpp, "array_neg_u"))
    _check_arr(u + v, _arr(cpp, "array_u_plus_v"))
    _check_arr(u - v, _arr(cpp, "array_u_minus_v"))
    _check_arr(u * v, _arr(cpp, "array_u_times_v"))
    _check_arr(u / v, _arr(cpp, "array_u_div_v"))
    _check_arr(u + 2.5, _arr(cpp, "array_u_plus_scalar"))
    _check_arr(u - 2.5, _arr(cpp, "array_u_minus_scalar"))
    _check_arr(u * 2.5, _arr(cpp, "array_u_times_scalar"))
    _check_arr(u / 2.5, _arr(cpp, "array_u_div_scalar"))
    _check_arr(2.5 - u, _arr(cpp, "array_scalar_minus_u"))
    _check_arr(2.5 / u, _arr(cpp, "array_scalar_div_u"))


@pytest.mark.tight
def test_array_reductions_match_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``DotProduct`` / ``Norm2`` (array.hpp:153, 156)."""
    u = _arr(cpp, "array_u")
    v = _arr(cpp, "array_v")
    tolerance.tight(float(np.dot(u, v)), float(cpp["array_dot_product"]))
    tolerance.tight(float(np.sqrt(np.dot(u, u))), float(cpp["array_norm2_u"]))


@pytest.mark.tight
def test_array_transcendentals_match_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``Abs`` / ``Sqrt`` / ``Log`` / ``Exp`` / ``Pow`` (array.hpp:236-254)."""
    u = _arr(cpp, "array_u")
    _check_arr(np.abs(u), _arr(cpp, "array_abs_u"))
    _check_arr(np.sqrt(np.abs(u)), _arr(cpp, "array_sqrt_abs_u"))
    _check_arr(np.log(np.abs(u)), _arr(cpp, "array_log_abs_u"))
    _check_arr(np.exp(u), _arr(cpp, "array_exp_u"))
    _check_arr(np.power(np.abs(u), 1.5), _arr(cpp, "array_pow_abs_u_1_5"))


# --------------------------------------------------------------------------
# Matrix — ql/math/matrix.hpp
# --------------------------------------------------------------------------


@pytest.mark.tight
def test_matrix_arithmetic_matches_cpp(cpp: dict[str, Any]) -> None:
    a = _mat(cpp, "matrix_a")
    b = _mat(cpp, "matrix_b")
    _check_mat(-a, _mat(cpp, "matrix_neg_a"))
    _check_mat(a + b, _mat(cpp, "matrix_a_plus_b"))
    _check_mat(a - b, _mat(cpp, "matrix_a_minus_b"))
    _check_mat(a * 2.5, _mat(cpp, "matrix_a_times_scalar"))
    _check_mat(a / 2.5, _mat(cpp, "matrix_a_div_scalar"))


@pytest.mark.tight
def test_matrix_products_match_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``operator*`` overloads (matrix.hpp:191-195) and ``outerProduct``."""
    a = _mat(cpp, "matrix_a")
    b = _mat(cpp, "matrix_b")
    c = _mat(cpp, "matrix_c")
    _check_mat(a @ b, _mat(cpp, "matrix_a_mul_b"))
    _check_mat(c @ a, _mat(cpp, "matrix_c_mul_a"))
    _check_mat(c.T, _mat(cpp, "matrix_transpose_c"))
    u = _arr(cpp, "array_u")
    v = _arr(cpp, "array_v")
    _check_mat(np.outer(u, v), _mat(cpp, "matrix_outer_product_u_v"))


@pytest.mark.tight
def test_matrix_vector_products_match_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``Matrix * Array`` and ``Array * Matrix`` (matrix.hpp:663, 676)."""
    a = _mat(cpp, "matrix_a")
    w = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    _check_arr(a @ w, _arr(cpp, "matrix_a_mul_array"))
    _check_arr(w @ a, _arr(cpp, "array_mul_matrix_a"))


@pytest.mark.tight
def test_matrix_inverse_and_determinant_match_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``inverse`` (matrix.hpp:216) and ``determinant`` (matrix.hpp:219)."""
    a = _mat(cpp, "matrix_a")
    b = _mat(cpp, "matrix_b")
    _check_mat(np.asarray(np.linalg.inv(a), dtype=np.float64), _mat(cpp, "matrix_inverse_a"))
    tolerance.tight(float(np.linalg.det(a)), float(cpp["matrix_determinant_a"]))
    tolerance.tight(float(np.linalg.det(b)), float(cpp["matrix_determinant_b"]))


@pytest.mark.tight
def test_matrix_views_match_cpp(cpp: dict[str, Any]) -> None:
    """Row / column / diagonal access, and the C++ ``rows()`` / ``columns()``."""
    a = _mat(cpp, "matrix_a")
    c = _mat(cpp, "matrix_c")
    _check_arr(a[1, :], _arr(cpp, "matrix_a_row1"))
    _check_arr(a[:, 2], _arr(cpp, "matrix_a_col2"))
    _check_arr(np.diag(a), _arr(cpp, "matrix_a_diagonal"))
    assert a.shape[0] == int(cpp["matrix_a_rows"])
    assert c.shape[0] == int(cpp["matrix_c_rows"])
    assert c.shape[1] == int(cpp["matrix_c_columns"])


def test_array_and_matrix_are_the_numpy_aliases() -> None:
    """The two names pquantlib exports really are numpy arrays, not classes.

    This is the fact the ``Array`` / ``Matrix`` allowlist entries assert.
    """
    a: Array = np.zeros(3, dtype=np.float64)
    m: Matrix = np.zeros((2, 3), dtype=np.float64)
    assert isinstance(a, np.ndarray)
    assert isinstance(m, np.ndarray)
    assert a.dtype == np.float64
    assert m.dtype == np.float64
    assert a.ndim == 1
    assert m.ndim == 2
