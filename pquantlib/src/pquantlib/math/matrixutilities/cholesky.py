"""Cholesky decomposition and the matching triangular solve.

# C++ parity: ql/math/matrixutilities/choleskydecomposition.{hpp,cpp} (v1.43) —
# ``Matrix CholeskyDecomposition(const Matrix& S, bool flexible)`` and
# ``Array CholeskySolveFor(const Matrix& L, const Array& b)``.

For a symmetric positive-definite ``S`` this returns the lower-triangular ``L``
with ``S == L @ L.T``.

**This used to delegate to ``scipy.linalg.cholesky(m, lower=True)``, and the
delegation did not hold.** Three ways it diverged from the C++:

* ``lower=True`` makes LAPACK ``dpotrf`` read the **lower** triangle of the
  input. The C++ loop reads ``S[i][j]`` for ``j >= i`` — the **upper** triangle.
  QuantLib only checks symmetry under ``QL_EXTRA_SAFETY_CHECKS``, which is off
  in a normal build, so for a matrix that is not exactly symmetric the two
  produce different factors. The ``upper_triangle_only`` probe case pins this.
* C++ handles the positive *semi*-definite case: a non-positive pivot becomes
  ``sqrt(max(sum, 0.0))`` and, once ``L[i][i]`` is ``close_enough`` to zero,
  the off-diagonal entries in that column are written as an exact ``0.0``
  instead of being divided. ``dpotrf`` raises on the first non-positive pivot.
* The old docstring justified deferring ``flexible=True`` by claiming the C++
  falls back to an eigenvalue-based pseudo-Cholesky via
  ``SymmetricSchurDecomposition``. It does not, in v1.43 or in v1.42.1 — the
  ``flexible`` flag only relaxes the ``QL_REQUIRE`` and lets the
  ``sqrt(max(...))`` clamp run. The deferral had no basis.

So the ``choleskydecomposition.cpp`` loops are transcribed here.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.math.matrix import Matrix


def cholesky_decomposition(m: Matrix, flexible: bool = False) -> Matrix:
    """Return lower-triangular ``L`` such that ``m == L @ L.T``.

    # C++ parity: ``CholeskyDecomposition`` (choleskydecomposition.cpp:27-64).

    Only the upper triangle of ``m`` is read, matching C++.

    Args:
        m: a symmetric ``(n, n)`` matrix.
        flexible: if True, accept a positive *semi*-definite (or indefinite)
            input: a non-positive pivot is clamped to zero rather than raising.

    Raises:
        LibraryException: if ``m`` is not a square 2-D matrix, or if a pivot is
            non-positive and ``flexible`` is False.
    """
    s = np.asarray(m, dtype=np.float64)
    qassert.require(s.ndim == 2, f"cholesky_decomposition requires a 2-D matrix, got ndim={s.ndim}")
    size = int(s.shape[0])
    qassert.require(size == int(s.shape[1]), "input matrix is not a square matrix")

    result: Matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i, size):
            total = float(s[i, j])
            for k in range(i):
                total -= float(result[i, k]) * float(result[j, k])
            if i == j:
                qassert.require(flexible or total > 0.0, "input matrix is not positive definite")
                # To handle positive semi-definite matrices take the square
                # root of sum if positive, else zero.
                result[i, i] = math.sqrt(max(total, 0.0))
            else:
                # With positive semi-definite matrices it is possible to have
                # result[i][i] == 0.0; in that case sum happens to be zero too.
                result[j, i] = (
                    0.0
                    if close_enough(float(result[i, i]), 0.0)
                    else total / float(result[i, i])
                )
    return result


def cholesky_solve_for(cholesky_factor: Matrix, b: Array) -> Array:
    """Solve ``L L.T x = b`` by forward then backward substitution.

    # C++ parity: ``CholeskySolveFor`` (choleskydecomposition.cpp:66-84). The
    # C++ name is kept here for grep; the accumulations mirror its
    # ``std::inner_product`` calls, including their ``-b[i]`` / ``-x[i]``
    # initialisers and the trailing negation.

    Args:
        cholesky_factor: the ``(n, n)`` lower-triangular factor from
            :func:`cholesky_decomposition`.
        b: the ``n``-vector right-hand side.

    Raises:
        LibraryException: on a size mismatch.
    """
    lower = np.asarray(cholesky_factor, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n = int(b_arr.shape[0])

    qassert.require(
        int(lower.shape[0]) == n and int(lower.shape[1]) == n,
        "Size of input matrix and vector does not match.",
    )

    x: Array = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = -float(b_arr[i])
        for k in range(i):
            acc += float(lower[i, k]) * float(x[k])
        x[i] = -acc
        x[i] /= float(lower[i, i])

    for i in range(n - 1, -1, -1):
        acc = -float(x[i])
        for k in range(i + 1, n):
            acc += float(lower[k, i]) * float(x[k])
        x[i] = -acc
        x[i] /= float(lower[i, i])

    return x
