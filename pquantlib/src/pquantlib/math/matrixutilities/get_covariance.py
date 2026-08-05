"""Covariance <-> (correlation, volatilities) conversion.

# C++ parity: ql/math/matrixutilities/getcovariance.{hpp,cpp} (v1.43).

:func:`get_covariance` combines a correlation matrix with a vector of standard
deviations; :class:`CovarianceDecomposition` goes the other way. Both use only
the lower triangle and symmetrise explicitly, and both validate their input
against a tolerance (default ``1e-12``) rather than assuming symmetry.

``get_covariance`` writes ``0.5 * (corr[i][j] + corr[j][i])`` — the *average* of
the two off-diagonal entries, not ``corr[i][j]`` — so a correlation matrix that
is only symmetric to within the tolerance still yields an exactly symmetric
covariance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix


def get_covariance(std_devs: Sequence[float], corr: Matrix, tolerance: float = 1.0e-12) -> Matrix:
    """Covariance matrix from standard deviations and a correlation matrix.

    # C++ parity: ``getCovariance`` (getcovariance.hpp:48-82). The C++ template
    # takes an iterator pair; the Python port takes the sequence.

    Args:
        std_devs: the ``n`` standard deviations.
        corr: the ``(n, n)`` correlation matrix; must be symmetric to within
            ``tolerance`` with unit diagonal.
        tolerance: symmetry / unit-diagonal tolerance.

    Raises:
        LibraryException: on a dimension mismatch, an asymmetric entry, or a
            diagonal entry that is not 1.
    """
    c = np.asarray(corr, dtype=np.float64)
    size = len(std_devs)
    qassert.require(
        int(c.shape[0]) == size,
        f"dimension mismatch between volatilities ({size}) and correlation rows ({c.shape[0]})",
    )
    qassert.require(
        int(c.shape[1]) == size,
        f"correlation matrix is not square: {size} rows and {c.shape[1]} columns",
    )

    covariance: Matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i):
            qassert.require(
                abs(float(c[i, j]) - float(c[j, i])) <= tolerance,
                f"correlation matrix not symmetric:\nc[{i},{j}] = {float(c[i, j])}"
                f"\nc[{j},{i}] = {float(c[j, i])}",
            )
            covariance[i, i] = std_devs[i] * std_devs[i]
            covariance[i, j] = std_devs[i] * std_devs[j] * 0.5 * (float(c[i, j]) + float(c[j, i]))
            covariance[j, i] = covariance[i, j]
        qassert.require(
            abs(float(c[i, i]) - 1.0) <= tolerance,
            f"invalid correlation matrix, diagonal element of row {i + 1} "
            f"is {float(c[i, i])} instead of 1.0",
        )
        covariance[i, i] = std_devs[i] * std_devs[i]
    return covariance


class CovarianceDecomposition:
    """Split a covariance matrix into variances and a correlation matrix.

    # C++ parity: ``class CovarianceDecomposition`` (getcovariance.hpp:95-110,
    # getcovariance.cpp:24-49).

    Args:
        covariance_matrix: a symmetric ``(n, n)`` covariance matrix. Only the
            lower triangle is used.
        tolerance: symmetry tolerance.
    """

    __slots__ = ("_correlation_matrix", "_std_devs", "_variances")

    def __init__(self, covariance_matrix: Matrix, tolerance: float = 1.0e-12) -> None:
        # C++ parity: getcovariance.cpp:24-49.
        cov = np.asarray(covariance_matrix, dtype=np.float64)
        size = int(cov.shape[0])
        qassert.require(
            size == int(cov.shape[1]),
            f"input covariance matrix must be square, it is [{size}x{cov.shape[0]}]",
        )

        variances: Array = np.array([cov[i, i] for i in range(size)], dtype=np.float64)
        std_devs: Array = np.zeros(size, dtype=np.float64)
        correlation: Matrix = np.zeros((size, size), dtype=np.float64)

        for i in range(size):
            std_devs[i] = math.sqrt(float(variances[i]))
            correlation[i, i] = 1.0
            for j in range(i):
                qassert.require(
                    abs(float(cov[i, j]) - float(cov[j, i])) <= tolerance,
                    f"invalid covariance matrix:\nc[{i}, {j}] = {float(cov[i, j])}"
                    f"\nc[{j}, {i}] = {float(cov[j, i])}",
                )
                value = float(cov[i, j]) / (float(std_devs[i]) * float(std_devs[j]))
                correlation[i, j] = value
                correlation[j, i] = value

        self._variances: Array = variances
        self._std_devs: Array = std_devs
        self._correlation_matrix: Matrix = correlation

    def variances(self) -> Array:
        """The diagonal of the covariance matrix.

        # C++ parity: ``variances()`` (getcovariance.hpp:102).
        """
        return self._variances

    def standard_deviations(self) -> Array:
        """Square roots of :meth:`variances`.

        # C++ parity: ``standardDeviations()`` (getcovariance.hpp:104).
        """
        return self._std_devs

    def correlation_matrix(self) -> Matrix:
        """The implied correlation matrix.

        # C++ parity: ``correlationMatrix()`` (getcovariance.hpp:106).
        """
        return self._correlation_matrix
