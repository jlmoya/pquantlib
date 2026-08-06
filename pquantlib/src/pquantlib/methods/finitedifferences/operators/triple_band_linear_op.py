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
``apply`` uses numpy fancy-indexing, and ``solve_splitting`` uses the
classic Thomas tridiagonal algorithm driven along ``reverse_index``, so
it solves along any direction of a multi-D layout exactly as C++ does.
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
        layout = mesher.layout()
        n = layout.size()
        self._i0: _IntArray = np.zeros(n, dtype=np.int64)
        self._i2: _IntArray = np.zeros(n, dtype=np.int64)
        self._reverse_index: _IntArray = np.zeros(n, dtype=np.int64)
        self._lower: Array = np.zeros(n, dtype=np.float64)
        self._diag: Array = np.zeros(n, dtype=np.float64)
        self._upper: Array = np.zeros(n, dtype=np.float64)

        # reverseIndex_ orders the flat grid so that walking it visits
        # every line along ``direction`` contiguously — that is what makes
        # the Thomas sweep in ``solve_splitting`` a per-line tridiagonal
        # solve. C++ builds it by swapping axis 0 with ``direction`` in
        # the dim vector, taking the resulting spacing, swapping the same
        # two entries back, and using that as an alternative stride set.
        #
        # # C++ parity: TripleBandLinearOp::TripleBandLinearOp
        # (triplebandlinearop.cpp) — std::iter_swap on dim then on spacing.
        new_dim = list(layout.dim())
        new_dim[0], new_dim[direction] = new_dim[direction], new_dim[0]
        new_spacing = list(FdmLinearOpLayout(tuple(new_dim)).spacing())
        new_spacing[0], new_spacing[direction] = new_spacing[direction], new_spacing[0]

        for iter_ in layout.iter():
            i = iter_.index
            self._i0[i] = layout.neighbourhood(iter_, direction, -1)
            self._i2[i] = layout.neighbourhood(iter_, direction, +1)
            new_index = sum(c * s for c, s in zip(iter_.coordinates, new_spacing, strict=True))
            self._reverse_index[new_index] = i

    def _like(self) -> TripleBandLinearOp:
        """A zero-banded operator sharing this one's structural arrays.

        ``i0`` / ``i2`` / ``reverseIndex`` depend only on ``(direction,
        mesher)``, so the derived operators C++ builds by value
        (``mult`` / ``multR`` / ``add``) can reuse them instead of
        re-running the O(N) layout walk. The arrays are never mutated
        after construction, so sharing them is safe.
        """
        other = object.__new__(type(self))
        other._direction = self._direction
        other._mesher = self._mesher
        other._i0 = self._i0
        other._i2 = self._i2
        other._reverse_index = self._reverse_index
        n = self._i0.shape[0]
        other._lower = np.zeros(n, dtype=np.float64)
        other._diag = np.zeros(n, dtype=np.float64)
        other._upper = np.zeros(n, dtype=np.float64)
        return other

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
        result = self._like()
        result._lower = self._lower * u_arr
        result._diag = self._diag * u_arr
        result._upper = self._upper * u_arr
        return result

    def mult_r(self, u: Array) -> TripleBandLinearOp:
        """Return ``self @ diag(u)`` — ``u`` as a diagonal matrix on the RIGHT.

        # C++ parity: ``TripleBandLinearOp::multR(const Array&)``. Note
        # C++ scales the bands with the *flat* neighbours ``u[i-1]`` /
        # ``u[i+1]`` (falling back to 1.0 at the two flat ends), **not**
        # with ``u[i0[i]]`` / ``u[i2[i]]``. Ported verbatim.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        size = self._mesher.layout().size()
        qassert.require(u_arr.size == size, "inconsistent size of rhs")
        result = self._like()
        sm1 = np.empty(size, dtype=np.float64)
        sm1[0] = 1.0
        sm1[1:] = u_arr[:-1]
        sp1 = np.empty(size, dtype=np.float64)
        sp1[-1] = 1.0
        sp1[:-1] = u_arr[1:]
        result._lower = self._lower * sm1
        result._diag = self._diag * u_arr
        result._upper = self._upper * sp1
        return result

    def add(self, other: TripleBandLinearOp | Array) -> TripleBandLinearOp:
        """Return ``self + other`` (band-wise sum, or self with diag offset).

        # C++ parity: two overloads — ``add(TripleBandLinearOp)`` and
        # ``add(Array)`` (the latter adds to the diagonal only).
        """
        result = self._like()
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

        The Thomas algorithm is a classical tridiagonal direct solver
        in O(N) time. This is the C++ in-place variant: the sweep runs
        along ``reverse_index``, which is the identity in 1-D and the
        direction-ordered permutation of the layout otherwise, so the
        same code solves along any direction.
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
