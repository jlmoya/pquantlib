"""NthOrderDerivativeOp — n-th order derivative operator of arbitrary width.

# C++ parity: ql/methods/finitedifferences/operators/nthorderderivativeop.{hpp,cpp}
# (v1.43).

Builds a sparse matrix whose row ``i`` holds the ``nPoints``-wide
Fornberg stencil for the ``order``-th derivative along ``direction``,
evaluated on the (possibly non-uniform) grid of that direction. Near
the two ends of the direction the stencil is shifted inwards so it
always fits, which is what the ``offset`` term computes.
"""

from __future__ import annotations

from typing import cast, final

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op import FdmLinearOp
from pquantlib.methods.finitedifferences.operators.numerical_differentiation import (
    NumericalDifferentiation,
)


@final
class NthOrderDerivativeOp(FdmLinearOp):
    """Sparse ``order``-th derivative operator with an ``n_points`` stencil.

    # C++ parity: ``class NthOrderDerivativeOp : public FdmLinearOp``.
    """

    def __init__(
        self, direction: int, order: int, n_points: int, mesher: FdmMesher
    ) -> None:
        layout = mesher.layout()
        size = layout.size()

        h_points = n_points // 2
        is_even = n_points == 2 * h_points

        # C++ funnels locations(direction) through a std::set to get the
        # unique, ascending grid values along that direction.
        x_values = sorted({float(v) for v in mesher.locations(direction)})

        nx = layout.dim()[direction]
        qassert.require(
            len(x_values) == nx,
            f"inconsistent set of grid values in direction {direction}",
        )
        qassert.require(n_points > 1 and n_points <= nx, "inconsistent number of points")

        # The stencil (hence the weights) depends only on the coordinate
        # along ``direction``; memoise per coordinate. Purely a speed
        # win — the values are identical to recomputing per node.
        weights_by_coord: dict[int, tuple[int, Array]] = {}

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        # C++ assigns (``m_(i, k) = weights[j]``) rather than accumulating,
        # so a repeated (i, k) must keep the LAST value written.
        entries: dict[tuple[int, int], float] = {}

        for iter_ in layout.iter():
            ix = iter_.coordinates[direction]
            cached = weights_by_coord.get(ix)
            if cached is None:
                offset = max(0, h_points - ix) - max(
                    0, h_points - (nx - (0 if is_even else 1) - ix)
                )
                ilx = ix - h_points + offset
                x_offsets = np.array(
                    [x_values[ilx + j] - x_values[ix] for j in range(n_points)],
                    dtype=np.float64,
                )
                weights = NumericalDifferentiation(None, order, x_offsets).weights()
                cached = (ilx, weights)
                weights_by_coord[ix] = cached
            ilx, weights = cached

            i = iter_.index
            for j in range(n_points):
                k = layout.neighbourhood(iter_, direction, ilx - ix + j)
                entries[(i, k)] = float(weights[j])

        for (i, k), v in entries.items():
            rows.append(i)
            cols.append(k)
            vals.append(v)

        self._mesher: FdmMesher = mesher
        self._m: csr_matrix = csr_matrix((vals, (rows, cols)), shape=(size, size))

    def apply(self, r: Array) -> Array:
        """Sparse matrix-vector product.

        # C++ parity: ``NthOrderDerivativeOp::apply`` — ``prod(m_, r)``.
        """
        qassert.require(
            r.size == self._mesher.layout().size(),
            f"inconsistent length of r (got {r.size}, "
            f"expected {self._mesher.layout().size()})",
        )
        # scipy has no type stubs for the sparse matmul; the product of a
        # CSR matrix with a 1-D float64 vector is a 1-D float64 vector.
        product = cast("Array", self._m @ r)
        return np.asarray(product, dtype=np.float64)

    def to_matrix(self) -> csr_matrix:
        """# C++ parity: ``NthOrderDerivativeOp::toMatrix``."""
        return self._m


__all__ = ["NthOrderDerivativeOp"]
