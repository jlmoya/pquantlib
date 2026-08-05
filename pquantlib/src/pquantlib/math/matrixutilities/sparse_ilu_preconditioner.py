"""SparseILUPreconditioner — ILU(p) preconditioner with level-of-fill control.

# C++ parity: ql/math/matrixutilities/sparseilupreconditioner.{hpp,cpp} (v1.43).

Incomplete LU factorisation in the level-of-fill formulation of Saad (1996):
an entry is allowed to fill in only while its *level* stays at or below
``lfil + 1``. ``apply(b)`` is one forward and one backward band-limited
substitution through the resulting ``L`` and ``U``.

**This is not ``scipy.sparse.linalg.spilu``.** SuperLU's ILU is a
threshold/drop-tolerance factorisation with column pivoting and equilibration;
this one is symbolic, level-based, un-pivoted, and its triangular solves visit
only the *band offsets* that were populated during the factorisation, in the
specific order ``lBands_`` descending / ``uBands_`` ascending. The ``L`` and
``U`` factors are therefore not the same matrices, and neither are the
preconditioned Krylov iterates that come out of them.

Two C++ quirks are transcribed rather than corrected, because the factors
depend on them:

* ``leviiNonZeroEntries`` is filtered from ``levii`` (all non-zero levels)
  but indexed by the position within ``wNonZeros`` (all non-zero *values*).
  When a level survives at a position whose value has cancelled to zero, the
  two lists slip and ``levs`` picks up a neighbour's level.
* The elimination loop advances by scanning for the next non-zero level; if a
  row has no non-zero level at or before its diagonal, the ``while`` never
  terminates. In practice ``A`` always has a structurally non-zero diagonal,
  which an ILU needs anyway.

Input is a dense ``Matrix``: PQuantLib has no ``SparseMatrix`` container, and
the C++ algorithm reads whole rows of ``A`` and writes structurally into ``L``
and ``U``, so a dense buffer reproduces its arithmetic exactly. Callers holding
a ``scipy.sparse`` operator pass ``op.toarray()``.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix

_QL_EPSILON: float = float(np.finfo(np.float64).eps)


class SparseILUPreconditioner:
    """Level-of-fill incomplete LU preconditioner.

    # C++ parity: ``class SparseILUPreconditioner``
    # (sparseilupreconditioner.hpp:36-51, sparseilupreconditioner.cpp:29-187).

    Args:
        a: a square matrix with a structurally non-zero diagonal.
        lfil: level of fill; entries with level above ``lfil + 1`` are dropped.
    """

    __slots__ = ("_l", "_l_bands", "_u", "_u_bands")

    def __init__(self, a: Matrix, lfil: int = 1) -> None:  # noqa: PLR0915
        # C++ parity: sparseilupreconditioner.cpp:29-146.
        arr = np.asarray(a, dtype=np.float64)
        qassert.require(
            arr.ndim == 2 and arr.shape[0] == arr.shape[1],
            "sparse ILU preconditioner works only with square matrices",
        )

        n = int(arr.shape[0])
        lower: Matrix = np.zeros((n, n), dtype=np.float64)
        upper: Matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            lower[i, i] = 1.0

        l_band_set: set[int] = set()
        u_band_set: set[int] = set()

        levs = np.zeros((n, n), dtype=np.int64)
        lfilp = lfil + 1

        for ii in range(n):
            w: Array = np.zeros(n, dtype=np.float64)
            for k in range(n):
                w[k] = arr[ii, k]

            levii = [0] * n
            for i in range(n):
                if float(w[i]) > _QL_EPSILON or float(w[i]) < -1.0 * _QL_EPSILON:
                    levii[i] = 1
            jj = -1
            while jj < ii:
                for k in range(jj + 1, n):
                    if levii[k] != 0:
                        jj = k
                        break
                if jj >= ii:
                    break
                jlev = levii[jj]
                if jlev <= lfilp:
                    non_zeros: list[int] = []
                    non_zero_entries: list[float] = []
                    entry = float(upper[jj, jj])
                    if entry > _QL_EPSILON or entry < -1.0 * _QL_EPSILON:
                        non_zeros.append(jj)
                        non_zero_entries.append(entry)
                    for band in sorted(u_band_set):
                        # C++ reads U_(jj, jj+band) unguarded; jj+band can run
                        # past the last column, where the entry is structurally
                        # absent and therefore zero.
                        if jj + band >= n:
                            continue
                        entry = float(upper[jj, jj + band])
                        if entry > _QL_EPSILON or entry < -1.0 * _QL_EPSILON:
                            non_zeros.append(jj + band)
                            non_zero_entries.append(entry)
                    fact = float(w[jj])
                    if non_zero_entries:
                        fact /= non_zero_entries[0]
                    for k in range(len(non_zeros)):
                        j = non_zeros[k]
                        temp = int(levs[jj, j]) + jlev
                        if levii[j] == 0:
                            if temp <= lfilp:
                                w[j] = -fact * non_zero_entries[k]
                                levii[j] = temp
                        else:
                            w[j] -= fact * non_zero_entries[k]
                            levii[j] = min(levii[j], temp)
                    w[jj] = fact

            w_non_zeros: list[int] = []
            w_non_zero_entries: list[float] = []
            for i in range(n):
                entry = float(w[i])
                if entry > _QL_EPSILON or entry < -1.0 * _QL_EPSILON:
                    w_non_zeros.append(i)
                    w_non_zero_entries.append(entry)
            levii_non_zero_entries: list[int] = [
                entry for entry in levii if entry > _QL_EPSILON or entry < -1.0 * _QL_EPSILON
            ]
            for k in range(len(w_non_zeros)):
                j = w_non_zeros[k]
                if j < ii:
                    lower[ii, j] = w_non_zero_entries[k]
                    l_band_set.add(ii - j)
                else:
                    upper[ii, j] = w_non_zero_entries[k]
                    # C++ indexes leviiNonZeroEntries by the *value* position;
                    # see the module docstring.
                    levs[ii, j] = levii_non_zero_entries[k]
                    if j - ii > 0:
                        u_band_set.add(j - ii)

        self._l: Matrix = lower
        self._u: Matrix = upper
        self._l_bands: list[int] = sorted(l_band_set)
        self._u_bands: list[int] = sorted(u_band_set)

    def l(self) -> Matrix:  # noqa: E743
        """Unit-lower-triangular factor.

        # C++ parity: ``SparseILUPreconditioner::L()``
        # (sparseilupreconditioner.cpp:148-150).
        """
        return self._l

    def u(self) -> Matrix:
        """Upper-triangular factor.

        # C++ parity: ``SparseILUPreconditioner::U()``
        # (sparseilupreconditioner.cpp:152-154).
        """
        return self._u

    def apply(self, b: Array) -> Array:
        """Apply ``M^-1``, i.e. solve ``L U x = b``.

        # C++ parity: ``SparseILUPreconditioner::apply``
        # (sparseilupreconditioner.cpp:156-158).
        """
        return self._backward_solve(self._forward_solve(b))

    def _forward_solve(self, b: Array) -> Array:
        # C++ parity: sparseilupreconditioner.cpp:160-174. The loop walks
        # lBands_ from the widest offset inward; the `i - band <= i - 1` guard
        # in C++ is vacuous (every band is >= 1), so only `k >= 0` filters.
        b_arr = np.asarray(b, dtype=np.float64)
        n = int(b_arr.shape[0])
        y: Array = np.zeros(n, dtype=np.float64)
        y[0] = float(b_arr[0]) / float(self._l[0, 0])
        for i in range(1, n):
            y[i] = float(b_arr[i]) / float(self._l[i, i])
            for j in range(len(self._l_bands) - 1, -1, -1):
                k = i - self._l_bands[j]
                if k >= 0:
                    y[i] -= float(self._l[i, k]) * float(y[k]) / float(self._l[i, i])
        return y

    def _backward_solve(self, y: Array) -> Array:
        # C++ parity: sparseilupreconditioner.cpp:176-187. uBands_ is ascending
        # and the loop stops at the first offset that leaves the matrix.
        n = int(y.shape[0])
        x: Array = np.zeros(n, dtype=np.float64)
        x[n - 1] = float(y[n - 1]) / float(self._u[n - 1, n - 1])
        for i in range(n - 2, -1, -1):
            x[i] = float(y[i]) / float(self._u[i, i])
            for band in self._u_bands:
                if i + band > n - 1:
                    break
                x[i] -= float(self._u[i, i + band]) * float(x[i + band]) / float(self._u[i, i])
        return x
