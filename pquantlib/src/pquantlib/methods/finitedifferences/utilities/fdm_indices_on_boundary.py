"""FdmIndicesOnBoundary — flat indices of a boundary hypersurface.

# C++ parity: ql/methods/finitedifferences/utilities/fdmindicesonboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).

Given a layout, a direction and a side, collect the flat indices of every
grid node whose coordinate along ``direction`` sits at the extreme
(``0`` for ``LOWER``, ``dim[direction] - 1`` for ``UPPER``). The result
has ``prod(dim) / dim[direction]`` entries and is ordered by the layout's
row-major traversal.

The C++ header includes ``fdmdirichletboundary.hpp`` purely to name the
``Side`` enum; the Python port takes ``BoundaryConditionSide`` from
``pquantlib.methods.finitedifferences.fdm_boundary_condition`` instead,
which breaks the import cycle that the C++ header would otherwise create.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)


@final
class FdmIndicesOnBoundary:
    """Flat indices on the ``side`` boundary along ``direction``.

    # C++ parity: ``class FdmIndicesOnBoundary``.
    """

    __slots__ = ("_indices",)

    def __init__(
        self,
        layout: FdmLinearOpLayout,
        direction: int,
        side: BoundaryConditionSide,
    ) -> None:
        dim = layout.dim()
        new_dim = list(dim)
        new_dim[direction] = 1
        # C++ parity: ``accumulate(newDim, Size(1), multiplies<>())``.
        hyper_size = math.prod(new_dim)

        indices: list[int] = []
        extreme = dim[direction] - 1
        for iterator in layout.iter():
            coord = iterator.coordinates[direction]
            if (side == BoundaryConditionSide.LOWER and coord == 0) or (
                side == BoundaryConditionSide.UPPER and coord == extreme
            ):
                qassert.require(hyper_size > len(indices), "index missmatch")
                indices.append(iterator.index)

        # C++ resizes ``indices_`` to ``hyperSize`` up front and fills the
        # first ``i`` slots; for a well-formed layout every slot is filled,
        # so the Python list has exactly ``hyperSize`` entries. Any other
        # outcome would mean the layout and the requested side disagree.
        self._indices: tuple[int, ...] = tuple(indices)

    def get_indices(self) -> tuple[int, ...]:
        """The boundary indices in layout order.

        # C++ parity: ``FdmIndicesOnBoundary::getIndices``.
        """
        return self._indices


__all__ = ["FdmIndicesOnBoundary"]
