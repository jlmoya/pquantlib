"""FdmTimeDepDirichletBoundary — time-dependent value on one grid face.

# C++ parity: ql/methods/finitedifferences/utilities/fdmtimedepdirichletboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).

Two C++ constructors differ only in the callable's return type:

* ``std::function<Real(Real)>`` — one scalar for the whole hypersurface;
* ``std::function<Array(Real)>`` — one value per boundary node.

Python cannot overload a constructor on argument type alone, so the port
takes a single ``value_on_boundary`` parameter and dispatches on the shape
of what the callable returns at ``set_time`` — matching the C++
``if (valueOnBoundary_) ... else if (valuesOnBoundary_)`` dispatch. The
scalar/array distinction is declared explicitly by the caller through the
``value_on_boundary`` / ``values_on_boundary`` keyword pair so that a
callable returning a 1-element array is never mistaken for a scalar.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
    FdmBoundaryCondition,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.utilities.fdm_indices_on_boundary import (
    FdmIndicesOnBoundary,
)


@final
class FdmTimeDepDirichletBoundary(FdmBoundaryCondition):
    """Dirichlet condition whose boundary value is a function of time.

    # C++ parity: ``class FdmTimeDepDirichletBoundary : public
    # BoundaryCondition<FdmLinearOp>``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        direction: int,
        side: BoundaryConditionSide,
        *,
        value_on_boundary: Callable[[float], float] | None = None,
        values_on_boundary: Callable[[float], Array] | None = None,
    ) -> None:
        qassert.require(
            (value_on_boundary is None) != (values_on_boundary is None),
            "exactly one of value_on_boundary / values_on_boundary must be given",
        )
        self._indices: tuple[int, ...] = FdmIndicesOnBoundary(
            mesher.layout(), direction, side
        ).get_indices()
        self._value_on_boundary: Callable[[float], float] | None = value_on_boundary
        self._values_on_boundary: Callable[[float], Array] | None = values_on_boundary
        # C++ constructs ``values_(indices_.size())`` — an uninitialised
        # Array of the hypersurface size; Python zero-fills.
        self._values: Array = np.zeros(len(self._indices), dtype=np.float64)

    def set_time(self, t: float) -> None:
        """Evaluate the boundary function at ``t``.

        # C++ parity: ``setTime(Time)``.
        """
        if self._value_on_boundary is not None:
            self._values = np.full(
                len(self._indices), self._value_on_boundary(t), dtype=np.float64
            )
        elif self._values_on_boundary is not None:
            self._values = np.asarray(self._values_on_boundary(t), dtype=np.float64)
        else:  # pragma: no cover — constructor guards against this
            qassert.fail("no boundary values defined")

    def apply_before_applying(self, op: object) -> None:
        """No-op. # C++ parity: ``applyBeforeApplying(operator_type&) {}``."""
        del op

    def apply_before_solving(self, op: object, rhs: Array) -> None:
        """No-op. # C++ parity: ``applyBeforeSolving(...) {}``."""
        del op, rhs

    def apply_after_applying(self, u: Array) -> None:
        """Write the cached boundary values into ``u``.

        # C++ parity: ``applyAfterApplying(array_type&)``.
        """
        qassert.require(
            len(self._indices) == self._values.shape[0],
            f"values on boundary size ({self._values.shape[0]}) does not match "
            f"hypersurface size ({len(self._indices)})",
        )
        for offset, index in enumerate(self._indices):
            u[index] = self._values[offset]

    def apply_after_solving(self, u: Array) -> None:
        """Same as :meth:`apply_after_applying`.

        # C++ parity: ``applyAfterSolving`` forwards to ``applyAfterApplying``.
        """
        self.apply_after_applying(u)


__all__ = ["FdmTimeDepDirichletBoundary"]
