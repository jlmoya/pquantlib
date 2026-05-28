"""UniformGridMesher — multi-D uniform mesher.

# C++ parity: ql/methods/finitedifferences/meshers/uniformgridmesher.{hpp,cpp}
# (v1.42.1).

Takes a ``FdmLinearOpLayout`` (multi-D index) plus a list of
``(start, end)`` boundary pairs (one per direction) and builds a
uniform mesh. ``dplus`` / ``dminus`` are constant per direction
(``(end - start) / (dim - 1)``); ``location`` looks up the per-axis
position by the iterator's coordinate.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
    FdmLinearOpLayout,
)


@final
class UniformGridMesher(FdmMesher):
    """Multi-D uniform mesher.

    # C++ parity: ``class UniformGridMesher : public FdmMesher``.
    """

    def __init__(
        self,
        layout: FdmLinearOpLayout,
        boundaries: list[tuple[float, float]],
    ) -> None:
        super().__init__(layout)
        dim = layout.dim()
        qassert.require(
            len(boundaries) == len(dim),
            f"inconsistent boundaries given (got {len(boundaries)}, expected {len(dim)})",
        )

        self._dx: list[float] = []
        self._locations_per_dir: list[Array] = []
        for d in range(len(dim)):
            start, end = boundaries[d]
            n = dim[d]
            dx = (end - start) / (n - 1)
            self._dx.append(dx)
            self._locations_per_dir.append(np.array([start + j * dx for j in range(n)], dtype=np.float64))

    def dplus(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        # C++ parity: constant dx[direction] regardless of position.
        return self._dx[direction]

    def dminus(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        return self._dx[direction]

    def location(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        return float(self._locations_per_dir[direction][iterator.coordinates[direction]])

    def locations(self, direction: int) -> Array:
        """Per-flat-index locations along ``direction``.

        # C++ parity: ``UniformGridMesher::locations(d)`` builds an
        # ``Array`` of length ``layout.size()`` and fills it by
        # iterating the layout.
        """
        ret = np.empty(self._layout.size(), dtype=np.float64)
        positions = self._locations_per_dir[direction]
        for iter_ in self._layout.iter():
            ret[iter_.index] = positions[iter_.coordinates[direction]]
        return ret


__all__ = ["UniformGridMesher"]
