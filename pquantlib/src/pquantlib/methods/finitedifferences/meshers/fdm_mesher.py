"""FdmMesher — abstract multi-D FD grid.

# C++ parity: ql/methods/finitedifferences/meshers/fdmmesher.hpp
# (v1.42.1).

A mesher associates a ``FdmLinearOpLayout`` (multi-D index) with
the physical grid: per-direction node positions plus per-node
``dplus`` / ``dminus`` step sizes used by the finite-difference
operators.

Concrete meshers (``UniformGridMesher``, ``FdmBlackScholesMesher``)
override the abstract methods. Multi-D composite meshers
(``FdmMesherComposite``) wrap multiple 1-D meshers; the 1-D case
plus the trivial composite is enough for the L5-D scope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
    FdmLinearOpLayout,
)


class FdmMesher(ABC):
    """Abstract multi-D mesher.

    # C++ parity: ``class FdmMesher``.
    """

    def __init__(self, layout: FdmLinearOpLayout) -> None:
        self._layout: FdmLinearOpLayout = layout

    def layout(self) -> FdmLinearOpLayout:
        """The index-layout of the grid."""
        return self._layout

    @abstractmethod
    def dplus(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        """Forward step size at ``iterator`` along ``direction``."""

    @abstractmethod
    def dminus(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        """Backward step size at ``iterator`` along ``direction``."""

    @abstractmethod
    def location(self, iterator: FdmLinearOpIterator, direction: int) -> float:
        """Physical location at ``iterator`` along ``direction``."""

    @abstractmethod
    def locations(self, direction: int) -> Array:
        """All physical locations along ``direction`` (length = layout.size())."""


__all__ = ["FdmMesher"]
