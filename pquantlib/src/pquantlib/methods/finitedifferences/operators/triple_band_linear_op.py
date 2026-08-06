"""TripleBandLinearOp — general triple-band linear operator on a FD grid.

# C++ parity: ql/methods/finitedifferences/operators/triplebandlinearop.{hpp,cpp}
# (v1.42.1).

A triple-band operator stores three per-node coefficients
(``lower``, ``diag``, ``upper``) plus per-node neighbor indices
(``i0`` = the "lower" neighbour, ``i2`` = the "upper" neighbour).
``apply`` does ``out[i] = lower[i]*r[i0[i]] + diag[i]*r[i] +
upper[i]*r[i2[i]]``.

For the **1-D** case, neighbour indices are ``i0[i] = max(0, i-1)`` and
``i2[i] = min(N-1, i+1)``; the reverse-index permutation is the identity.

For **multi-D**, ``reverse_index`` is the permutation that walks the grid
with ``direction`` varying fastest, so that the Thomas sweep runs along
the operator's own direction rather than along the layout's first axis.
C++ builds it by re-deriving the spacings of a layout whose first and
``direction``-th dimensions are swapped
(``triplebandlinearop.cpp`` constructor).

The Python implementation is built directly on numpy / scipy.sparse:
``apply`` uses numpy fancy-indexing for speed, and ``solve_splitting``
uses the classic Thomas tridiagonal algorithm. Because a valid triple-band
operator has ``lower == 0`` at each line's lower boundary and
``upper == 0`` at its upper boundary, C++'s single sweep over the whole
reordered array decouples exactly into independent per-line sweeps
(the coupling terms multiply by exact zeros), so the sweep is run
batched across lines — same arithmetic, vectorised.
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
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)

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
        self._reverse_index: _IntArray = np.zeros(n, dtype=np.int64)
        self._lower: Array = np.zeros(n, dtype=np.float64)
        self._diag: Array = np.zeros(n, dtype=np.float64)
        self._upper: Array = np.zeros(n, dtype=np.float64)

        # C++ parity: the constructor swaps dim[0] with dim[direction],
        # takes the spacings of that permuted layout, swaps element 0 with
        # element `direction` back, and uses the result to renumber each grid
        # point so that `direction` is the fastest-varying axis. For
        # direction == 0 this collapses to the identity.
        dim = list(mesher.layout().dim())
        dim[0], dim[direction] = dim[direction], dim[0]
        new_spacing = list(FdmLinearOpLayout(tuple(dim)).spacing())
        new_spacing[0], new_spacing[direction] = new_spacing[direction], new_spacing[0]

        for iter_ in mesher.layout().iter():
            i = iter_.index
            self._i0[i] = mesher.layout().neighbourhood(iter_, direction, -1)
            self._i2[i] = mesher.layout().neighbourhood(iter_, direction, +1)
            new_index = 0
            for c, s in zip(iter_.coordinates, new_spacing, strict=True):
                new_index += c * s
            self._reverse_index[new_index] = i

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
        """Solve ``(b * I + a * L) x = r`` via the Thomas algorithm.

        # C++ parity: ``TripleBandLinearOp::solve_splitting(r, a, b)``.

        C++ runs one Thomas sweep over the whole array in
        ``reverse_index`` order. A valid triple-band operator has
        ``lower == 0`` at each line's first node and ``upper == 0`` at its
        last (C++ asserts exactly this under ``QL_EXTRA_SAFETY_CHECKS``),
        so the two coupling terms across a line boundary multiply by exact
        zeros and the long chain decouples into independent per-line
        systems. This implementation therefore reshapes into
        ``(lines, line_length)`` and runs the identical recurrence batched
        across lines — the per-line arithmetic, and hence the result, is
        unchanged.
        """
        qassert.require(
            r.size == self._mesher.layout().size(),
            f"inconsistent size of rhs (got {r.size}, expected {self._mesher.layout().size()})",
        )

        line_len = self._mesher.layout().dim()[self._direction]
        idx = self._reverse_index.reshape(-1, line_len)

        diag = self._diag[idx]
        lower = self._lower[idx]
        upper = self._upper[idx]
        rhs = r[idx]

        x: Array = np.zeros_like(rhs)
        tmp: Array = np.zeros_like(rhs)

        bet = a * diag[:, 0] + b
        qassert.require(bool(np.all(bet != 0.0)), "division by zero")
        bet = 1.0 / bet
        x[:, 0] = rhs[:, 0] * bet

        for j in range(1, line_len):
            tmp[:, j] = a * upper[:, j - 1] * bet
            bet = b + a * (diag[:, j] - tmp[:, j] * lower[:, j])
            qassert.require(bool(np.all(bet != 0.0)), "division by zero")
            bet = 1.0 / bet
            x[:, j] = (rhs[:, j] - a * lower[:, j] * x[:, j - 1]) * bet

        for j in range(line_len - 2, -1, -1):
            x[:, j] -= tmp[:, j + 1] * x[:, j + 1]

        ret_val: Array = np.zeros(r.size, dtype=np.float64)
        ret_val[idx.reshape(-1)] = x.reshape(-1)
        return ret_val


__all__ = ["TripleBandLinearOp"]
