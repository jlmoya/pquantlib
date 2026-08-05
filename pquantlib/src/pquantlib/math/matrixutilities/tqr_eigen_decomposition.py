"""TqrEigenDecomposition — tridiagonal QR eigen-decomposition with explicit shift.

# C++ parity: ql/math/matrixutilities/tqreigendecomposition.{hpp,cpp} (v1.43).

Eigenvalues (and optionally eigenvectors) of a real symmetric **tridiagonal**
matrix given by its diagonal and sub-diagonal.

**This is not ``scipy.linalg.eigh_tridiagonal``.** The C++ routine is the
Wilkinson/Reinsch implicit-shift QR sweep with QuantLib's own three-way
:class:`EigenVectorCalculation` and three-way :class:`ShiftStrategy`, and its
observable behaviour differs from LAPACK's in ways callers depend on:

* eigenvalues come back **descending** (LAPACK returns them ascending);
* the ``(eigenvalue, eigenvector)`` pairs are sorted lexicographically, so the
  eigenvector breaks ties;
* the first component of every eigenvector is pinned non-negative;
* :attr:`EigenVectorCalculation.ONLY_FIRST_ROW_EIGEN_VECTOR` accumulates a
  single row of the eigenvector matrix — which is all Golub-Welsch needs, and
  is why ``GaussianQuadrature`` asks for it together with
  :attr:`ShiftStrategy.OVERRELAXATION`;
* :meth:`iterations` is part of the public API.

Convergence is decided by ``offDiagIsZero``, an *exact* floating-point equality
test (``|d[k-1]| + |d[k]| == |d[k-1]| + |d[k]| + |e[k]|``). With
:attr:`ShiftStrategy.NO_SHIFT` and a spectrum containing a ``+-lambda`` pair
that test can never become true and the loop does not terminate — the C++ has
the same behaviour, so it is transcribed rather than papered over.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix


class EigenVectorCalculation(Enum):
    """How much of the eigenvector matrix to accumulate.

    # C++ parity: ``TqrEigenDecomposition::EigenVectorCalculation``
    # (tqreigendecomposition.hpp:46-48).
    """

    WITH_EIGEN_VECTOR = "WithEigenVector"
    WITHOUT_EIGEN_VECTOR = "WithoutEigenVector"
    ONLY_FIRST_ROW_EIGEN_VECTOR = "OnlyFirstRowEigenVector"


class ShiftStrategy(Enum):
    """Which shift to subtract before each QR sweep.

    # C++ parity: ``TqrEigenDecomposition::ShiftStrategy``
    # (tqreigendecomposition.hpp:50-52).
    """

    NO_SHIFT = "NoShift"
    OVERRELAXATION = "Overrelaxation"
    CLOSE_EIGEN_VALUE = "CloseEigenValue"


class TqrEigenDecomposition:
    """Eigen-decomposition of a symmetric tridiagonal matrix.

    # C++ parity: ``class TqrEigenDecomposition``
    # (tqreigendecomposition.hpp:44-70, tqreigendecomposition.cpp:29-140).

    Args:
        diag: the ``n`` diagonal entries.
        sub: the ``n - 1`` sub-diagonal entries.
        calc: how much of the eigenvector matrix to build.
        strategy: which shift to apply.
    """

    __slots__ = ("_d", "_ev", "_iter")

    def __init__(  # noqa: PLR0915
        self,
        diag: Array,
        sub: Array,
        calc: EigenVectorCalculation = EigenVectorCalculation.WITH_EIGEN_VECTOR,
        strategy: ShiftStrategy = ShiftStrategy.CLOSE_EIGEN_VALUE,
    ) -> None:
        # C++ parity: tqreigendecomposition.cpp:29-140.
        d: Array = np.asarray(diag, dtype=np.float64).astype(np.float64, copy=True)
        sub_arr: Array = np.asarray(sub, dtype=np.float64)
        n = int(d.shape[0])

        qassert.require(n == int(sub_arr.shape[0]) + 1, "Wrong dimensions")

        if calc == EigenVectorCalculation.WITH_EIGEN_VECTOR:
            ev_rows = n
        elif calc == EigenVectorCalculation.WITHOUT_EIGEN_VECTOR:
            ev_rows = 0
        else:
            ev_rows = 1
        ev: Matrix = np.zeros((ev_rows, n), dtype=np.float64)

        self._iter: int = 0
        self._d: Array = d
        self._ev: Matrix = ev

        e: Array = np.zeros(n, dtype=np.float64)
        e[1:] = sub_arr
        for i in range(ev_rows):
            ev[i, i] = 1.0

        for k in range(n - 1, 0, -1):
            while not self._off_diag_is_zero(k, e):
                u_index = k
                while True:
                    u_index -= 1
                    if u_index <= 0 or self._off_diag_is_zero(u_index, e):
                        break
                lo = u_index
                self._iter += 1

                q = float(d[lo])
                if strategy != ShiftStrategy.NO_SHIFT:
                    # calculated eigenvalue of the 2x2 sub matrix of
                    # [ d[k-1] e[k] ]
                    # [  e[k]  d[k] ]
                    # which is closer to d[k+1].
                    t1 = math.sqrt(
                        0.25 * (float(d[k]) * float(d[k]) + float(d[k - 1]) * float(d[k - 1]))
                        - 0.5 * float(d[k - 1]) * float(d[k])
                        + float(e[k]) * float(e[k])
                    )
                    t2 = 0.5 * (float(d[k]) + float(d[k - 1]))

                    lambda_ = (
                        t2 + t1 if abs(t2 + t1 - float(d[k])) < abs(t2 - t1 - float(d[k])) else t2 - t1
                    )

                    if strategy == ShiftStrategy.CLOSE_EIGEN_VALUE:
                        q -= lambda_
                    else:
                        q -= (1.25 if k == n - 1 else 1.0) * lambda_

                # the QR transformation
                sine = 1.0
                cosine = 1.0
                u = 0.0

                recover_underflow = False
                i = lo + 1
                while i <= k and not recover_underflow:
                    h = cosine * float(e[i])
                    p = sine * float(e[i])

                    e[i - 1] = math.sqrt(p * p + q * q)
                    if float(e[i - 1]) != 0.0:
                        sine = p / float(e[i - 1])
                        cosine = q / float(e[i - 1])

                        g = float(d[i - 1]) - u
                        t = (float(d[i]) - g) * sine + 2.0 * cosine * h

                        u = sine * t
                        d[i - 1] = g + u
                        q = cosine * t - h

                        for j in range(ev_rows):
                            tmp = float(ev[j, i - 1])
                            ev[j, i - 1] = sine * float(ev[j, i]) + cosine * tmp
                            ev[j, i] = cosine * float(ev[j, i]) - sine * tmp
                    else:
                        # recover from underflow
                        d[i - 1] -= u
                        e[lo] = 0.0
                        recover_underflow = True
                    i += 1

                if not recover_underflow:
                    d[k] -= u
                    e[k] = q
                    e[lo] = 0.0

        # sort (eigenvalues, eigenvectors), code taken from
        # symmetricSchureDecomposition.cpp. ``std::sort`` with
        # ``std::greater<>`` over ``pair<Real, vector<Real>>`` is the same
        # lexicographic descending order as Python's ``sort(reverse=True)``
        # over ``(float, tuple[float, ...])``.
        # C++ parity: tqreigendecomposition.cpp:119-139.
        temp: list[tuple[float, tuple[float, ...]]] = [
            (float(d[i]), tuple(float(v) for v in ev[:, i])) for i in range(n)
        ]
        temp.sort(reverse=True)
        # first element is positive
        for i in range(n):
            d[i] = temp[i][0]
            sign = -1.0 if ev_rows > 0 and temp[i][1][0] < 0.0 else 1.0
            for j in range(ev_rows):
                ev[j, i] = sign * temp[i][1][j]

    def _off_diag_is_zero(self, k: int, e: Array) -> bool:
        """``True`` once ``e[k]`` is negligible against ``d[k-1]`` and ``d[k]``.

        # C++ parity: ``TqrEigenDecomposition::offDiagIsZero``
        # (tqreigendecomposition.cpp:144-147). See NR for the abort assumption:
        # it is not part of the original Wilkinson algorithm, and it is an
        # *exact* equality test, deliberately.
        """
        d = self._d
        # ``fabs(d[k-1]) + fabs(d[k])`` is the same rounded value on both sides
        # of the C++ comparison, so binding it once is an exact transcription.
        base = abs(float(d[k - 1])) + abs(float(d[k]))
        return base == base + abs(float(e[k]))

    def eigenvalues(self) -> Array:
        """Eigenvalues in decreasing order.

        # C++ parity: ``eigenvalues()`` (tqreigendecomposition.hpp:59).
        """
        return self._d

    def eigenvectors(self) -> Matrix:
        """Eigenvectors as columns; ``0``, ``1`` or ``n`` rows per ``calc``.

        # C++ parity: ``eigenvectors()`` (tqreigendecomposition.hpp:60).
        """
        return self._ev

    def iterations(self) -> int:
        """Number of QR sweeps performed.

        # C++ parity: ``iterations()`` (tqreigendecomposition.hpp:62).
        """
        return self._iter
