"""NinePointLinearOp — 9-stencil 2-D linear operator.

# C++ parity: ql/methods/finitedifferences/operators/ninepointlinearop.{hpp,cpp}
# (v1.42.1).

A nine-point operator stores, per flat-index ``i``, nine coefficients
``a_{ab}`` and eight neighbour indices ``i_{ab}`` (where ``ab`` are
the row / column offsets along two directions ``d0`` and ``d1``,
each in ``{-1, 0, +1}`` — the diagonal ``i_{11}`` is implicit, equal
to ``i``).

``apply`` returns the matrix-vector product::

    out[i] = sum_{ab} a_{ab}[i] * r[i_{ab}[i]]

with ``a_{11}[i] * r[i]`` for the diagonal entry.

Used by ``SecondOrderMixedDerivativeOp`` and ``FdmZabrOp``.
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


class NinePointLinearOp(FdmLinearOp):
    """9-stencil 2-D linear operator.

    # C++ parity: ``class NinePointLinearOp : public FdmLinearOp``.

    Concrete subclasses (``SecondOrderMixedDerivativeOp``) populate
    the nine coefficient arrays in their constructor; this base
    populates only the neighbour-index arrays.
    """

    def __init__(self, d0: int, d1: int, mesher: FdmMesher) -> None:
        layout = mesher.layout()
        qassert.require(
            d0 != d1 and d0 < len(layout.dim()) and d1 < len(layout.dim()),
            "inconsistent derivative directions",
        )
        n = layout.size()
        self._d0: int = d0
        self._d1: int = d1
        self._mesher: FdmMesher = mesher

        # Neighbour indices: i_ab where (a, b) is (column-offset along d0,
        # row-offset along d1). a, b in {-1, 0, +1}. We index by
        # (a + 1, b + 1) → name pattern i00 / i01 / i02 / i10 / i12 /
        # i20 / i21 / i22 (i11 is the iter's own index).
        self._i00: _IntArray = np.zeros(n, dtype=np.int64)
        self._i01: _IntArray = np.zeros(n, dtype=np.int64)
        self._i02: _IntArray = np.zeros(n, dtype=np.int64)
        self._i10: _IntArray = np.zeros(n, dtype=np.int64)
        self._i12: _IntArray = np.zeros(n, dtype=np.int64)
        self._i20: _IntArray = np.zeros(n, dtype=np.int64)
        self._i21: _IntArray = np.zeros(n, dtype=np.int64)
        self._i22: _IntArray = np.zeros(n, dtype=np.int64)

        # Coefficient arrays.
        self._a00: Array = np.zeros(n, dtype=np.float64)
        self._a01: Array = np.zeros(n, dtype=np.float64)
        self._a02: Array = np.zeros(n, dtype=np.float64)
        self._a10: Array = np.zeros(n, dtype=np.float64)
        self._a11: Array = np.zeros(n, dtype=np.float64)
        self._a12: Array = np.zeros(n, dtype=np.float64)
        self._a20: Array = np.zeros(n, dtype=np.float64)
        self._a21: Array = np.zeros(n, dtype=np.float64)
        self._a22: Array = np.zeros(n, dtype=np.float64)

        for iter_ in layout.iter():
            i = iter_.index
            # Single-direction neighbours along d0 and d1.
            self._i10[i] = layout.neighbourhood(iter_, d1, -1)
            self._i01[i] = layout.neighbourhood(iter_, d0, -1)
            self._i21[i] = layout.neighbourhood(iter_, d0, +1)
            self._i12[i] = layout.neighbourhood(iter_, d1, +1)
            # Corner neighbours (both directions).
            self._i00[i] = layout.neighbourhood2(iter_, d0, -1, d1, -1)
            self._i20[i] = layout.neighbourhood2(iter_, d0, +1, d1, -1)
            self._i02[i] = layout.neighbourhood2(iter_, d0, -1, d1, +1)
            self._i22[i] = layout.neighbourhood2(iter_, d0, +1, d1, +1)

    @property
    def d0(self) -> int:
        return self._d0

    @property
    def d1(self) -> int:
        return self._d1

    def apply(self, r: Array) -> Array:
        """9-stencil apply.

        # C++ parity: ``NinePointLinearOp::apply``.
        """
        qassert.require(
            r.size == self._mesher.layout().size(),
            f"inconsistent length of r (got {r.size}, expected {self._mesher.layout().size()})",
        )
        return (
            self._a00 * r[self._i00]
            + self._a01 * r[self._i01]
            + self._a02 * r[self._i02]
            + self._a10 * r[self._i10]
            + self._a11 * r
            + self._a12 * r[self._i12]
            + self._a20 * r[self._i20]
            + self._a21 * r[self._i21]
            + self._a22 * r[self._i22]
        )

    def mult(self, u: Array) -> NinePointLinearOp:
        """Return a new ``NinePointLinearOp`` with each coefficient row scaled by ``u``.

        # C++ parity: ``NinePointLinearOp::mult``.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        result = NinePointLinearOp(self._d0, self._d1, self._mesher)
        result._a00 = self._a00 * u_arr
        result._a01 = self._a01 * u_arr
        result._a02 = self._a02 * u_arr
        result._a10 = self._a10 * u_arr
        result._a11 = self._a11 * u_arr
        result._a12 = self._a12 * u_arr
        result._a20 = self._a20 * u_arr
        result._a21 = self._a21 * u_arr
        result._a22 = self._a22 * u_arr
        return result

    def to_matrix(self) -> csr_matrix:
        """Return the operator as a sparse CSR matrix.

        # C++ parity: ``NinePointLinearOp::toMatrix``.
        """
        n = self._mesher.layout().size()
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(n):
            rows.append(i)
            cols.append(int(self._i00[i]))
            vals.append(float(self._a00[i]))
            rows.append(i)
            cols.append(int(self._i01[i]))
            vals.append(float(self._a01[i]))
            rows.append(i)
            cols.append(int(self._i02[i]))
            vals.append(float(self._a02[i]))
            rows.append(i)
            cols.append(int(self._i10[i]))
            vals.append(float(self._a10[i]))
            rows.append(i)
            cols.append(i)
            vals.append(float(self._a11[i]))
            rows.append(i)
            cols.append(int(self._i12[i]))
            vals.append(float(self._a12[i]))
            rows.append(i)
            cols.append(int(self._i20[i]))
            vals.append(float(self._a20[i]))
            rows.append(i)
            cols.append(int(self._i21[i]))
            vals.append(float(self._a21[i]))
            rows.append(i)
            cols.append(int(self._i22[i]))
            vals.append(float(self._a22[i]))
        return csr_matrix((vals, (rows, cols)), shape=(n, n))


__all__ = ["NinePointLinearOp"]
