"""FdmMesherComposite — multi-D mesher built from per-axis ``Fdm1dMesher`` instances.

# C++ parity: ql/methods/finitedifferences/meshers/fdmmeshercomposite.{hpp,cpp}
# (v1.42.1).

For the L5-D scope only the single-direction composite is used
(wrapping a 1-D log-spot mesher into the ``FdmMesher`` shape that
the operators and engine expect). Multi-D composites are kept on
the API surface for forward compatibility with Phase 6.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
    FdmLinearOpLayout,
)


@final
class FdmMesherComposite(FdmMesher):
    """Multi-D mesher made of per-direction 1-D meshers.

    # C++ parity: ``class FdmMesherComposite : public FdmMesher``.
    """

    def __init__(self, *meshers: Fdm1dMesher) -> None:
        qassert.require(len(meshers) > 0, "at least one 1d mesher required")
        dim = tuple(m.size() for m in meshers)
        layout = FdmLinearOpLayout(dim)
        super().__init__(layout)
        self._meshers: tuple[Fdm1dMesher, ...] = meshers

    def get_fdm_1d_meshers(self) -> tuple[Fdm1dMesher, ...]:
        return self._meshers

    def dplus(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        return self._meshers[direction].dplus(iterator.coordinates[direction])

    def dminus(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        return self._meshers[direction].dminus(iterator.coordinates[direction])

    def location(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        return self._meshers[direction].location(iterator.coordinates[direction])

    def locations(self, direction: int) -> Array:
        ret = np.empty(self._layout.size(), dtype=np.float64)
        loc_1d = self._meshers[direction].locations()
        for iter_ in self._layout.iter():
            ret[iter_.index] = loc_1d[iter_.coordinates[direction]]
        return ret


__all__ = ["FdmMesherComposite"]
