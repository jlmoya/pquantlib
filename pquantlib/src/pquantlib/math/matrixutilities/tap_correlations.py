"""Triangular-Angles-Parametrized correlation matrices and their Frobenius cost.

# C++ parity: ql/math/matrixutilities/tapcorrelations.{hpp,cpp} (v1.43).

The angle parametrisations of Rapisarda, Brigo & Mercurio, "Parameterizing
correlations: a geometric interpretation" — equation (24) for the triangular
form and equation (32) for the rank-three spherical spiral. Each returns a
pseudo-root ``B``; the correlation matrix is ``B @ B.T``, which is positive
semi-definite and unit-diagonal by construction whatever the angles.

:class:`FrobeniusCostFunction` wraps one of them for calibration. Note that it
overrides ``value`` with ``DotProduct(values, values)`` — the plain **sum of
squares**, not the ``sqrt(mean(...))`` of the ``CostFunction`` default — so a
generic optimizer sees a different objective scale from the base class.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.math.optimization.cost_function import CostFunction

#: ``(x, matrix_size, rank) -> pseudo-root``; C++
#: ``std::function<Matrix(const Array&, Size, Size)>``.
type Parametrisation = Callable[[Array, int, int], Matrix]


def triangular_angles_parametrization(angles: Array, matrix_size: int, rank: int) -> Matrix:
    """Triangular-Angles-Parametrized pseudo-root.

    # C++ parity: ``triangularAnglesParametrization``
    # (tapcorrelations.cpp:25-55). C++ name kept in this docstring for grep.

    Args:
        angles: ``(rank - 1) * (2 * matrix_size - rank) / 2`` angles.
        matrix_size: the order of the returned square matrix.
        rank: the target rank.
    """
    a = np.asarray(angles, dtype=np.float64)
    qassert.require(
        (rank - 1) * (2 * matrix_size - rank) == 2 * int(a.shape[0]),
        "rank-1) * (matrixSize - rank/2) == angles.size()",
    )
    m: Matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)

    # first row filling
    m[0, 0] = 1.0
    for j in range(1, matrix_size):
        m[0, j] = 0.0

    # next ones...
    k = 0  # angles index
    for i in range(1, matrix_size):
        sin_product = 1.0
        bound = min(i, rank - 1)
        for j in range(bound):
            m[i, j] = math.cos(float(a[k]))
            m[i, j] *= sin_product
            sin_product *= math.sin(float(a[k]))
            k += 1
        m[i, bound] = sin_product
        for j in range(bound + 1, matrix_size):
            m[i, j] = 0
    return m


def lmm_triangular_angles_parametrization(angles: Array, matrix_size: int, rank: int) -> Matrix:
    """LMM variant: one angle per row, applied cumulatively.

    # C++ parity: ``lmmTriangularAnglesParametrization``
    # (tapcorrelations.cpp:57-80). ``rank`` is accepted and ignored, as in C++.
    """
    del rank
    a = np.asarray(angles, dtype=np.float64)
    m: Matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    for i in range(matrix_size):
        if i > 0:
            cos_phi = math.cos(float(a[i - 1]))
            sin_phi = math.sin(float(a[i - 1]))
        else:
            cos_phi = 1.0
            sin_phi = 0.0

        for j in range(i):
            m[i, j] = sin_phi * float(m[i - 1, j])

        m[i, i] = cos_phi

        for j in range(i + 1, matrix_size):
            m[i, j] = 0.0
    return m


def triangular_angles_parametrization_unconstrained(
    x: Array, matrix_size: int, rank: int
) -> Matrix:
    """As :func:`triangular_angles_parametrization`, over unconstrained ``x``.

    # C++ parity: ``triangularAnglesParametrizationUnconstrained``
    # (tapcorrelations.cpp:82-90) — ``theta_i = pi/2 - arctan(x_i)``.
    """
    xa = np.asarray(x, dtype=np.float64)
    angles: Array = np.zeros(int(xa.shape[0]), dtype=np.float64)
    for i in range(int(xa.shape[0])):
        angles[i] = math.pi * 0.5 - math.atan(float(xa[i]))
    return triangular_angles_parametrization(angles, matrix_size, rank)


def lmm_triangular_angles_parametrization_unconstrained(
    x: Array, matrix_size: int, rank: int
) -> Matrix:
    """As :func:`lmm_triangular_angles_parametrization`, over unconstrained ``x``.

    # C++ parity: ``lmmTriangularAnglesParametrizationUnconstrained``
    # (tapcorrelations.cpp:92-100).
    """
    xa = np.asarray(x, dtype=np.float64)
    angles: Array = np.zeros(int(xa.shape[0]), dtype=np.float64)
    for i in range(int(xa.shape[0])):
        angles[i] = math.pi * 0.5 - math.atan(float(xa[i]))
    return lmm_triangular_angles_parametrization(angles, matrix_size, rank)


def triangular_angles_parametrization_rank_three(
    alpha: float, t0: float, epsilon: float, matrix_size: int
) -> Matrix:
    """Rank-three pseudo-root from a 3-D spherical spiral.

    # C++ parity: ``triangularAnglesParametrizationRankThree``
    # (tapcorrelations.cpp:102-113) — equation (32) of the reference.

    Returns:
        A ``(matrix_size, 3)`` matrix.
    """
    m: Matrix = np.zeros((matrix_size, 3), dtype=np.float64)
    for i in range(matrix_size):
        t = t0 * (1 - math.exp(epsilon * float(i)))
        phi = math.atan(alpha * t)
        m[i, 0] = math.cos(t) * math.cos(phi)
        m[i, 1] = math.sin(t) * math.cos(phi)
        m[i, 2] = -math.sin(phi)
    return m


def triangular_angles_parametrization_rank_three_vectorial(
    parameters: Array, nb_rows: int
) -> Matrix:
    """:func:`triangular_angles_parametrization_rank_three` with packed parameters.

    # C++ parity: ``triangularAnglesParametrizationRankThreeVectorial``
    # (tapcorrelations.cpp:115-124).
    """
    p = np.asarray(parameters, dtype=np.float64)
    qassert.require(int(p.shape[0]) == 3, "the parameter array must contain exactly 3 values")
    return triangular_angles_parametrization_rank_three(
        float(p[0]), float(p[1]), float(p[2]), nb_rows
    )


class FrobeniusCostFunction(CostFunction):
    """Frobenius distance between a parametrised pseudo-root and a target.

    # C++ parity: ``class FrobeniusCostFunction`` (tapcorrelations.hpp:88-103,
    # tapcorrelations.cpp:126-144).

    Args:
        target: the correlation matrix to fit.
        f: the parametrisation, ``(x, matrix_size, rank) -> pseudo-root``.
        matrix_size: passed through to ``f``.
        rank: passed through to ``f``.
    """

    __slots__ = ("_f", "_matrix_size", "_rank", "_target")

    def __init__(
        self, target: Matrix, f: Parametrisation, matrix_size: int, rank: int
    ) -> None:
        self._target: Matrix = np.asarray(target, dtype=np.float64)
        self._f: Parametrisation = f
        self._matrix_size: int = matrix_size
        self._rank: int = rank

    def value(self, x: Array) -> float:
        """Sum of squares of :meth:`values`.

        # C++ parity: ``FrobeniusCostFunction::value``
        # (tapcorrelations.cpp:126-129) — ``DotProduct(temp, temp)``. This
        # deliberately overrides the ``CostFunction`` default
        # ``sqrt(mean(values**2))``.
        """
        temp = self.values(x)
        return float(np.dot(temp, temp))

    def values(self, x: Array) -> Array:
        """Strictly-lower-triangular entries of ``B B^T - target``.

        # C++ parity: ``FrobeniusCostFunction::values``
        # (tapcorrelations.cpp:131-144).
        """
        rows = int(self._target.shape[0])
        cols = int(self._target.shape[1])
        result: Array = np.zeros((rows * (cols - 1)) // 2, dtype=np.float64)
        pseudo_root = self._f(np.asarray(x, dtype=np.float64), self._matrix_size, self._rank)
        differences = pseudo_root @ pseudo_root.T - self._target
        k = 0
        # then we store the elementwise differences in a vector.
        for i in range(rows):
            for j in range(i):
                result[k] = differences[i, j]
                k += 1
        return result
