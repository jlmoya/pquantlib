"""SymmetricSchurDecomposition — symmetric threshold Jacobi eigen-decomposition.

# C++ parity: ql/math/matrixutilities/symmetricschurdecomposition.{hpp,cpp} (v1.43).

For a real symmetric ``S`` this finds ``U`` and ``D`` with ``S == U @ D @ U.T``.

**This is not ``numpy.linalg.eigh``.** C++ runs the *threshold cyclic Jacobi*
sweep of Golub & Van Loan: at most 100 sweeps of plane rotations in the fixed
order ``(j, k)`` for ``j < k``, with

* a rotation threshold ``0.2 * sum|off-diagonal| / size**2`` for the first four
  sweeps and ``0.0`` afterwards,
* an ``epsPrec = 1e-15`` relative skip that zeroes an off-diagonal entry
  outright once it is negligible against both diagonal entries it couples,
* the eigenvalue update accumulated through ``tmpAccumulate`` / ``tmpDiag``
  once per sweep rather than in place,
* a final descending sort of the ``(eigenvalue, eigenvector)`` pairs compared
  **lexicographically** — the eigenvector breaks ties between equal
  eigenvalues,
* a sign convention pinning the first component of every eigenvector to be
  non-negative, and
* a round-off guard that snaps any eigenvalue with
  ``|lambda / lambda_max| < 1e-16`` to exactly ``0.0``.

LAPACK reproduces neither the ordering rule, nor the sign rule, nor the
snap-to-zero rule, and the eigenvectors it returns for a degenerate eigenvalue
span the same subspace with a different basis. Callers such as
``rankReducedSqrt`` consume the eigenvector *signs* directly, so the algorithm
is transcribed rather than delegated.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix

_EPS_PREC: float = 1e-15
_MAX_ITERATIONS: int = 100


class SymmetricSchurDecomposition:
    """Eigenvalues and eigenvectors of a real symmetric matrix.

    # C++ parity: ``class SymmetricSchurDecomposition``
    # (symmetricschurdecomposition.hpp:50-61).

    Args:
        s: A symmetric ``(n, n)`` matrix. Only the upper triangle is touched
           by the sweep, exactly as in C++ — the lower triangle is read once
           for the initial diagonal and then ignored.
    """

    __slots__ = ("_diagonal", "_eigen_vectors")

    def __init__(self, s: Matrix) -> None:  # noqa: PLR0915
        # C++ parity: symmetricschurdecomposition.cpp:26-138.
        arr = np.asarray(s, dtype=np.float64)
        qassert.require(arr.ndim == 2, "input matrix must be 2-D")
        qassert.require(arr.shape[0] > 0 and arr.shape[1] > 0, "null matrix given")
        qassert.require(arr.shape[0] == arr.shape[1], "input matrix must be square")

        size = int(arr.shape[0])
        diagonal: Array = np.zeros(size, dtype=np.float64)
        eigen_vectors: Matrix = np.zeros((size, size), dtype=np.float64)
        for q in range(size):
            diagonal[q] = arr[q, q]
            eigen_vectors[q, q] = 1.0
        ss: Matrix = arr.astype(np.float64, copy=True)

        tmp_diag: list[float] = [float(v) for v in diagonal]
        tmp_accumulate: list[float] = [0.0] * size
        keeplooping = True
        ite = 1
        while True:
            # main loop
            total = 0.0
            for a in range(size - 1):
                for b in range(a + 1, size):
                    total += abs(float(ss[a, b]))

            if total == 0.0:
                keeplooping = False
            else:
                # To speed up computation a threshold is introduced to make
                # sure it is worthy to perform the Jacobi rotation.
                threshold = 0.2 * total / (size * size) if ite < 5 else 0.0

                for j in range(size - 1):
                    for k in range(j + 1, size):
                        smll = abs(float(ss[j, k]))
                        if (
                            ite > 5
                            and smll < _EPS_PREC * abs(float(diagonal[j]))
                            and smll < _EPS_PREC * abs(float(diagonal[k]))
                        ):
                            ss[j, k] = 0.0
                        elif abs(float(ss[j, k])) > threshold:
                            heig = float(diagonal[k]) - float(diagonal[j])
                            if smll < _EPS_PREC * abs(heig):
                                tang = float(ss[j, k]) / heig
                            else:
                                beta = 0.5 * heig / float(ss[j, k])
                                tang = 1.0 / (abs(beta) + math.sqrt(1.0 + beta * beta))
                                if beta < 0.0:
                                    tang = -tang
                            cosin = 1.0 / math.sqrt(1.0 + tang * tang)
                            sine = tang * cosin
                            rho = sine / (1.0 + cosin)
                            heig = tang * float(ss[j, k])
                            tmp_accumulate[j] -= heig
                            tmp_accumulate[k] += heig
                            diagonal[j] -= heig
                            diagonal[k] += heig
                            ss[j, k] = 0.0
                            for u in range(j):
                                _jacobi_rotate(ss, rho, sine, u, j, u, k)
                            for u in range(j + 1, k):
                                _jacobi_rotate(ss, rho, sine, j, u, u, k)
                            for u in range(k + 1, size):
                                _jacobi_rotate(ss, rho, sine, j, u, k, u)
                            for u in range(size):
                                _jacobi_rotate(eigen_vectors, rho, sine, u, j, u, k)
                for k in range(size):
                    tmp_diag[k] += tmp_accumulate[k]
                    diagonal[k] = tmp_diag[k]
                    tmp_accumulate[k] = 0.0

            ite += 1
            if not (ite <= _MAX_ITERATIONS and keeplooping):
                break

        qassert.require(ite <= _MAX_ITERATIONS, f"Too many iterations ({_MAX_ITERATIONS}) reached")

        # sort (eigenvalues, eigenvectors)
        # C++ parity: symmetricschurdecomposition.cpp:115-137. ``std::sort``
        # with ``std::greater<>`` over ``pair<Real, vector<Real>>`` compares
        # the eigenvalue first and the whole eigenvector second, descending;
        # Python's ``sorted(..., reverse=True)`` over ``(float, tuple[float,
        # ...])`` is the same lexicographic order.
        temp: list[tuple[float, tuple[float, ...]]] = [
            (float(diagonal[col]), tuple(float(v) for v in eigen_vectors[:, col]))
            for col in range(size)
        ]
        temp.sort(reverse=True)
        max_ev = temp[0][0]
        for col in range(size):
            # check for round-off errors
            diagonal[col] = 0.0 if abs(temp[col][0] / max_ev) < 1e-16 else temp[col][0]
            sign = -1.0 if temp[col][1][0] < 0.0 else 1.0
            for row in range(size):
                eigen_vectors[row, col] = sign * temp[col][1][row]

        self._diagonal: Array = diagonal
        self._eigen_vectors: Matrix = eigen_vectors

    def eigenvalues(self) -> Array:
        """Eigenvalues in decreasing order.

        # C++ parity: ``eigenvalues()`` (symmetricschurdecomposition.hpp:54).
        """
        return self._diagonal

    def eigenvectors(self) -> Matrix:
        """Eigenvectors as **columns**, aligned with :meth:`eigenvalues`.

        # C++ parity: ``eigenvectors()`` (symmetricschurdecomposition.hpp:55).
        """
        return self._eigen_vectors


def _jacobi_rotate(m: Matrix, rot: float, dil: float, j1: int, k1: int, j2: int, k2: int) -> None:
    """Jacobi, a.k.a. Givens, rotation applied in place to two entries of ``m``.

    # C++ parity: ``SymmetricSchurDecomposition::jacobiRotate_``
    # (symmetricschurdecomposition.hpp:67-75).
    """
    x1 = float(m[j1, k1])
    x2 = float(m[j2, k2])
    m[j1, k1] = x1 - dil * (x2 + x1 * rot)
    m[j2, k2] = x2 + dil * (x1 - x2 * rot)
