"""TripleBandLinearOp — general triple-band linear operator on a FD grid.

# C++ parity: ql/methods/finitedifferences/operators/triplebandlinearop.{hpp,cpp}
# (v1.42.1).

A triple-band operator stores three per-node coefficients
(``lower``, ``diag``, ``upper``) plus per-node neighbor indices
(``i0`` = the "lower" neighbour, ``i2`` = the "upper" neighbour).
``apply`` does ``out[i] = lower[i]*r[i0[i]] + diag[i]*r[i] +
upper[i]*r[i2[i]]``.

For the **1-D** case (the only case exercised in L5-D), neighbour
indices are ``i0[i] = max(0, i-1)`` and ``i2[i] = min(N-1, i+1)``;
the reverse-index permutation is the identity.

The Python implementation is built directly on numpy / scipy.sparse:
``apply`` uses numpy fancy-indexing for speed, and ``solve_splitting``
uses the classic Thomas tridiagonal algorithm (1-D only). Multi-D
support is deferred to Phase 6 along with the rest of the multi-asset
FD scaffolding.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op import FdmLinearOp

# Integer index array (separate from the float-Array alias).
_IntArray = npt.NDArray[np.int64]


class TripleBandLinearOp(FdmLinearOp):
    """Triple-band linear operator.

    # C++ parity: ``class TripleBandLinearOp : public FdmLinearOp``.
    """

    def __init__(self, direction: int, mesher: FdmMesher) -> None:
        self._direction: int = direction
        self._mesher: FdmMesher = mesher
        n = mesher.layout().size()
        self._i0: _IntArray = np.zeros(n, dtype=np.int64)
        self._i2: _IntArray = np.zeros(n, dtype=np.int64)
        # 1-D specialisation: reverseIndex is the identity for any
        # 1-D direction (multi-D requires the iter_swap permutation —
        # deferred to Phase 6).
        self._reverse_index: _IntArray = np.arange(n, dtype=np.int64)
        self._lower: Array = np.zeros(n, dtype=np.float64)
        self._diag: Array = np.zeros(n, dtype=np.float64)
        self._upper: Array = np.zeros(n, dtype=np.float64)

        for iter_ in mesher.layout().iter():
            i = iter_.index
            self._i0[i] = mesher.layout().neighbourhood(iter_, direction, -1)
            self._i2[i] = mesher.layout().neighbourhood(iter_, direction, +1)

    # --- mutating arithmetic builders ----------------------------------

    def axpyb(
        self,
        a: Array | None,
        x: TripleBandLinearOp | None,
        y: TripleBandLinearOp,
        b: Array | None,
    ) -> None:
        """In-place set ``self <- a*x + y + b`` on each band.

        # C++ parity: ``TripleBandLinearOp::axpyb`` —
        # ``diag = y.diag + a*x.diag + b``, ``lower = y.lower + a*x.lower``,
        # ``upper = y.upper + a*x.upper``. Empty ``a`` means skip the
        # ``a*x`` term; empty ``b`` means skip the bias.
        """
        y_diag = y._diag
        y_lower = y._lower
        y_upper = y._upper

        # Start from y.
        diag = y_diag.copy()
        lower = y_lower.copy()
        upper = y_upper.copy()

        if a is not None and x is not None:
            a_arr = np.asarray(a, dtype=np.float64)
            if a_arr.size == 1:
                s = float(a_arr.flat[0])
                diag += s * x._diag
                lower += s * x._lower
                upper += s * x._upper
            else:
                diag += a_arr * x._diag
                lower += a_arr * x._lower
                upper += a_arr * x._upper

        if b is not None:
            b_arr = np.asarray(b, dtype=np.float64)
            if b_arr.size == 1:
                diag += float(b_arr.flat[0])
            else:
                diag += b_arr

        self._diag = diag
        self._lower = lower
        self._upper = upper

    def mult(self, u: Array) -> TripleBandLinearOp:
        """Return ``diag(u) @ self`` — pointwise multiplication of each band.

        # C++ parity: ``TripleBandLinearOp::mult(const Array&)`` —
        # interprets ``u`` as a diagonal matrix multiplied on the LEFT
        # (each row of the operator scales by ``u[i]``).
        """
        u_arr = np.asarray(u, dtype=np.float64)
        result = TripleBandLinearOp(self._direction, self._mesher)
        result._lower = self._lower * u_arr
        result._diag = self._diag * u_arr
        result._upper = self._upper * u_arr
        return result

    def add(self, other: TripleBandLinearOp | Array) -> TripleBandLinearOp:
        """Return ``self + other`` (band-wise sum, or self with diag offset).

        # C++ parity: two overloads — ``add(TripleBandLinearOp)`` and
        # ``add(Array)`` (the latter adds to the diagonal only).
        """
        result = TripleBandLinearOp(self._direction, self._mesher)
        if isinstance(other, TripleBandLinearOp):
            result._lower = self._lower + other._lower
            result._diag = self._diag + other._diag
            result._upper = self._upper + other._upper
        else:
            arr = np.asarray(other, dtype=np.float64)
            result._lower = self._lower.copy()
            result._upper = self._upper.copy()
            result._diag = self._diag + arr
        return result

    # --- FdmLinearOp overrides -----------------------------------------

    def apply(self, r: Array) -> Array:
        """Triple-band matrix-vector product.

        # C++ parity: ``apply`` —
        # ``out[i] = r[i0[i]]*lower[i] + r[i]*diag[i] + r[i2[i]]*upper[i]``.
        """
        qassert.require(
            r.size == self._mesher.layout().size(),
            f"inconsistent length of r (got {r.size}, expected {self._mesher.layout().size()})",
        )
        return r[self._i0] * self._lower + r * self._diag + r[self._i2] * self._upper

    def to_matrix(self) -> csr_matrix:
        """Return the operator as a sparse CSR matrix.

        # C++ parity: ``TripleBandLinearOp::toMatrix`` —
        # ``M[i, i0[i]] += lower[i]; M[i, i] += diag[i];
        # M[i, i2[i]] += upper[i]``. Note ``+=`` because boundary
        # nodes can map both neighbours back to themselves.
        """
        n = self._mesher.layout().size()
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(n):
            rows.append(i)
            cols.append(int(self._i0[i]))
            vals.append(float(self._lower[i]))
            rows.append(i)
            cols.append(i)
            vals.append(float(self._diag[i]))
            rows.append(i)
            cols.append(int(self._i2[i]))
            vals.append(float(self._upper[i]))
        # Use sum_duplicates semantics so boundary self-loops accumulate.
        return csr_matrix((vals, (rows, cols)), shape=(n, n))

    # --- splitting solve ------------------------------------------------

    def solve_splitting(self, r: Array, a: float, b: float = 1.0) -> Array:
        """Solve ``(b * I + a * L) x = r`` via the Thomas algorithm (1-D).

        # C++ parity: ``TripleBandLinearOp::solve_splitting(r, a, b)``.

        The Thomas algorithm is a classical tridiagonal direct solver
        in O(N) time. We use the C++ in-place variant adapted to
        Python — ``reverse_index`` is the identity in 1-D, so the
        sweep is straightforward.
        """
        qassert.require(
            r.size == self._mesher.layout().size(),
            f"inconsistent size of rhs (got {r.size}, expected {self._mesher.layout().size()})",
        )

        n = r.size
        ret_val: Array = np.zeros(n, dtype=np.float64)
        tmp: Array = np.zeros(n, dtype=np.float64)

        rim1 = int(self._reverse_index[0])
        bet = 1.0 / (a * self._diag[rim1] + b)
        qassert.require(bet != 0.0, "division by zero")
        ret_val[self._reverse_index[0]] = r[rim1] * bet

        for j in range(1, n):
            ri = int(self._reverse_index[j])
            tmp[j] = a * self._upper[rim1] * bet
            bet = b + a * (self._diag[ri] - tmp[j] * self._lower[ri])
            qassert.require(bet != 0.0, "division by zero")
            bet = 1.0 / bet
            ret_val[ri] = (r[ri] - a * self._lower[ri] * ret_val[rim1]) * bet
            rim1 = ri

        # Back-substitution: indices n-2 down to 1, then 0.
        for j in range(n - 2, 0, -1):
            ret_val[self._reverse_index[j]] -= tmp[j + 1] * ret_val[self._reverse_index[j + 1]]
        ret_val[self._reverse_index[0]] -= tmp[1] * ret_val[self._reverse_index[1]]

        return ret_val


__all__ = ["TripleBandLinearOp"]
