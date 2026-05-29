"""NinePointLinearOp — nine-point stencil linear operator.

# C++ parity: ql/methods/finitedifferences/operators/ninepointlinearop.{hpp,cpp}
# (v1.42.1).

Multi-D linear operator with a 3x3 stencil in two given directions
``d0`` and ``d1``. Stores per-node coefficients ``a_{ij}`` for
``i, j in {0, 1, 2}`` plus the corresponding neighbour indices
``i_{ij}``. ``apply(r)`` evaluates the matrix-vector product.

The pattern of indices ``i_{ij}`` is::

    i00 = (-1, -1)   i10 = ( 0, -1)   i20 = (+1, -1)
    i01 = (-1,  0)   i11 = ( 0,  0)   i21 = (+1,  0)
    i02 = (-1, +1)   i12 = ( 0, +1)   i22 = (+1, +1)

where the offsets are along ``d0`` (first index) and ``d1`` (second).

Subclasses populate the coefficient arrays in the constructor.
``SecondOrderMixedDerivativeOp`` is the only concrete user in W5-A
(via ``FdmKlugeExtOUOp``).
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
    FdmLinearOpIterator,
)

_IntArray = npt.NDArray[np.int64]


def _neighbourhood_2d(
    layout_dim: tuple[int, ...],
    layout_spacing: tuple[int, ...],
    iter_: FdmLinearOpIterator,
    d0: int,
    off0: int,
    d1: int,
    off1: int,
) -> int:
    """Flat index of the neighbour with offsets ``off0`` along ``d0`` and
    ``off1`` along ``d1`` (other axes unchanged).

    # C++ parity: ``FdmLinearOpLayout::neighbourhood(iter, i1, off1, i2, off2)``
    # — boundary clamping is *not* applied (the C++ overload does NOT clamp
    # like the single-axis version does; instead it computes
    # ``iter.index + off1 * spacing[i1] + off2 * spacing[i2]`` after
    # clamping each coordinate to ``[0, dim[i] - 1]``).
    """
    coords = list(iter_.coordinates)
    new0 = coords[d0] + off0
    new1 = coords[d1] + off1
    if new0 < 0:
        new0 = 0
    elif new0 >= layout_dim[d0]:
        new0 = layout_dim[d0] - 1
    if new1 < 0:
        new1 = 0
    elif new1 >= layout_dim[d1]:
        new1 = layout_dim[d1] - 1
    coords[d0] = new0
    coords[d1] = new1
    return sum(c * s for c, s in zip(coords, layout_spacing, strict=True))


class NinePointLinearOp(FdmLinearOp):
    """Nine-point stencil linear operator in two given directions.

    # C++ parity: ``class NinePointLinearOp : public FdmLinearOp``.
    """

    def __init__(self, d0: int, d1: int, mesher: FdmMesher) -> None:
        qassert.require(
            d0 != d1 and d0 < len(mesher.layout().dim()) and d1 < len(mesher.layout().dim()),
            "inconsistent derivative directions",
        )
        self._d0: int = d0
        self._d1: int = d1
        self._mesher: FdmMesher = mesher
        n = mesher.layout().size()

        # Index arrays.
        self._i00: _IntArray = np.zeros(n, dtype=np.int64)
        self._i10: _IntArray = np.zeros(n, dtype=np.int64)
        self._i20: _IntArray = np.zeros(n, dtype=np.int64)
        self._i01: _IntArray = np.zeros(n, dtype=np.int64)
        self._i21: _IntArray = np.zeros(n, dtype=np.int64)
        self._i02: _IntArray = np.zeros(n, dtype=np.int64)
        self._i12: _IntArray = np.zeros(n, dtype=np.int64)
        self._i22: _IntArray = np.zeros(n, dtype=np.int64)

        # Coefficient arrays.
        self._a00: Array = np.zeros(n, dtype=np.float64)
        self._a10: Array = np.zeros(n, dtype=np.float64)
        self._a20: Array = np.zeros(n, dtype=np.float64)
        self._a01: Array = np.zeros(n, dtype=np.float64)
        self._a11: Array = np.zeros(n, dtype=np.float64)
        self._a21: Array = np.zeros(n, dtype=np.float64)
        self._a02: Array = np.zeros(n, dtype=np.float64)
        self._a12: Array = np.zeros(n, dtype=np.float64)
        self._a22: Array = np.zeros(n, dtype=np.float64)

        layout = mesher.layout()
        layout_dim = layout.dim()
        layout_spacing = layout.spacing()
        for iter_ in layout.iter():
            i = iter_.index
            self._i10[i] = layout.neighbourhood(iter_, d1, -1)
            self._i01[i] = layout.neighbourhood(iter_, d0, -1)
            self._i21[i] = layout.neighbourhood(iter_, d0, 1)
            self._i12[i] = layout.neighbourhood(iter_, d1, 1)
            self._i00[i] = _neighbourhood_2d(layout_dim, layout_spacing, iter_, d0, -1, d1, -1)
            self._i20[i] = _neighbourhood_2d(layout_dim, layout_spacing, iter_, d0, 1, d1, -1)
            self._i02[i] = _neighbourhood_2d(layout_dim, layout_spacing, iter_, d0, -1, d1, 1)
            self._i22[i] = _neighbourhood_2d(layout_dim, layout_spacing, iter_, d0, 1, d1, 1)

    def apply(self, r: Array) -> Array:
        """Nine-point matrix-vector product.

        # C++ parity: ``NinePointLinearOp::apply`` — vectorised in
        # numpy via fancy-indexing on the 8 neighbour-index arrays
        # plus the diagonal.
        """
        qassert.require(
            r.size == self._mesher.layout().size(),
            f"inconsistent length of r: {r.size} vs {self._mesher.layout().size()}",
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
        """Return a new NinePointLinearOp with each coefficient scaled by ``u[i]``.

        # C++ parity: ``NinePointLinearOp::mult(const Array&)``.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        result = NinePointLinearOp(self._d0, self._d1, self._mesher)
        result._a00 = self._a00 * u_arr
        result._a10 = self._a10 * u_arr
        result._a20 = self._a20 * u_arr
        result._a01 = self._a01 * u_arr
        result._a11 = self._a11 * u_arr
        result._a21 = self._a21 * u_arr
        result._a02 = self._a02 * u_arr
        result._a12 = self._a12 * u_arr
        result._a22 = self._a22 * u_arr
        return result

    def to_matrix(self) -> csr_matrix:
        """Return the operator as a sparse CSR matrix.

        # C++ parity: ``NinePointLinearOp::toMatrix`` — note the
        # ``+=`` accumulation in C++ for handling boundary self-loops.
        """
        n = self._mesher.layout().size()
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(n):
            for col_arr, val_arr in (
                (self._i00, self._a00),
                (self._i01, self._a01),
                (self._i02, self._a02),
                (self._i10, self._a10),
                (None, self._a11),
                (self._i12, self._a12),
                (self._i20, self._a20),
                (self._i21, self._a21),
                (self._i22, self._a22),
            ):
                rows.append(i)
                cols.append(i if col_arr is None else int(col_arr[i]))
                vals.append(float(val_arr[i]))
        return csr_matrix((vals, (rows, cols)), shape=(n, n))


__all__ = ["NinePointLinearOp", "_neighbourhood_2d"]
