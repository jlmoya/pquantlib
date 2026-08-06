"""BoundaryConditionSchemeHelper — fan-out of a boundary-condition set.

# C++ parity: ql/methods/finitedifferences/schemes/boundaryconditionschemehelper.hpp
# (v1.43) — header-only.

Every operator-splitting scheme in this package holds one of these and calls
it at the four points a boundary condition can intervene (before/after an
``apply``, before/after a ``solve``), plus ``set_time``. The class itself is
nothing but the loop over the set — it exists so that the five call sites do
not have to be written out in each of the seven schemes.

The C++ class stores ``OperatorTraits<FdmLinearOp>::bc_set``, i.e.
``std::vector<shared_ptr<BoundaryCondition<FdmLinearOp>>>``; the Python
equivalent is :data:`~pquantlib.methods.finitedifferences.fdm_boundary_condition.FdmBoundaryConditionSet`.

The C++ private default constructor (used by nothing) is spelled here as a
defaulted empty ``bc_set`` argument, which is how every caller reaches the
"no boundary conditions" state anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    FdmBoundaryCondition,
    FdmBoundaryConditionSet,
)

if TYPE_CHECKING:
    from pquantlib.math.array import Array


@final
class BoundaryConditionSchemeHelper:
    """Applies a whole ``bc_set`` at each of the scheme intervention points.

    # C++ parity: ``class BoundaryConditionSchemeHelper``.
    """

    __slots__ = ("_bc_set",)

    def __init__(self, bc_set: FdmBoundaryConditionSet = ()) -> None:
        # C++ parity: the explicit constructor takes the set by value and moves.
        self._bc_set: tuple[FdmBoundaryCondition, ...] = tuple(bc_set)

    def apply_before_applying(self, op: object) -> None:
        """# C++ parity: ``applyBeforeApplying(operator_type&)``."""
        for bc in self._bc_set:
            bc.apply_before_applying(op)

    def apply_before_solving(self, op: object, a: Array) -> None:
        """# C++ parity: ``applyBeforeSolving(operator_type&, array_type&)``."""
        for bc in self._bc_set:
            bc.apply_before_solving(op, a)

    def apply_after_applying(self, a: Array) -> None:
        """# C++ parity: ``applyAfterApplying(array_type&)``."""
        for bc in self._bc_set:
            bc.apply_after_applying(a)

    def apply_after_solving(self, a: Array) -> None:
        """# C++ parity: ``applyAfterSolving(array_type&)``."""
        for bc in self._bc_set:
            bc.apply_after_solving(a)

    def set_time(self, t: float) -> None:
        """# C++ parity: ``setTime(Time)``."""
        for bc in self._bc_set:
            bc.set_time(t)


__all__ = ["BoundaryConditionSchemeHelper"]
