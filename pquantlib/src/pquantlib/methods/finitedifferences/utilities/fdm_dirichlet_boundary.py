"""FdmDirichletBoundary — constant value on one face of the grid.

# C++ parity: ql/methods/finitedifferences/utilities/fdmdirichletboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).

Pins every node on the ``side`` face along ``direction`` to
``value_on_boundary``. ``apply_before_applying`` / ``apply_before_solving``
are no-ops in C++ too; only the after-hooks write.

The extra scalar helper ``apply_after_applying_value(x, value)`` is the
C++ overload ``Real applyAfterApplying(Real x, Real value) const`` — Python
cannot overload on signature, so it gets its own name.
"""

from __future__ import annotations

from typing import final

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
class FdmDirichletBoundary(FdmBoundaryCondition):
    """Constant Dirichlet condition on one grid face.

    # C++ parity: ``class FdmDirichletBoundary : public
    # BoundaryCondition<FdmLinearOp>``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        value_on_boundary: float,
        direction: int,
        side: BoundaryConditionSide,
    ) -> None:
        self._side: BoundaryConditionSide = side
        self._value_on_boundary: float = value_on_boundary
        self._indices: tuple[int, ...] = FdmIndicesOnBoundary(
            mesher.layout(), direction, side
        ).get_indices()

        locations = mesher.locations(direction)
        if side == BoundaryConditionSide.LOWER:
            self._x_extreme: float = float(locations[0])
        elif side == BoundaryConditionSide.UPPER:
            # C++ parity — reproduced verbatim, quirk included:
            #   xExtreme_ = mesher->locations(direction)[dim[direction]-1];
            # ``locations(direction)`` is a *full-layout* array indexed by
            # flat index, so this only picks the last node along ``direction``
            # when ``direction == 0`` (stride 1). For direction > 0 the flat
            # index ``dim[direction]-1`` still has coordinate 0 along
            # ``direction``, so ``xExtreme_`` ends up at the *first* node.
            # This only affects the scalar
            # :meth:`apply_after_applying_value` helper (the array hooks use
            # ``indices_``), and the Python port matches C++ bit for bit.
            self._x_extreme = float(locations[mesher.layout().dim()[direction] - 1])
        else:
            qassert.fail("internal error")

    def apply_before_applying(self, op: object) -> None:
        """No-op. # C++ parity: empty ``applyBeforeApplying``."""
        del op

    def apply_before_solving(self, op: object, rhs: Array) -> None:
        """No-op. # C++ parity: empty ``applyBeforeSolving``."""
        del op, rhs

    def apply_after_applying(self, u: Array) -> None:
        """Pin the boundary nodes of ``u``.

        # C++ parity: ``applyAfterApplying(Array&)``.
        """
        for index in self._indices:
            u[index] = self._value_on_boundary

    def apply_after_solving(self, u: Array) -> None:
        """Same as :meth:`apply_after_applying`.

        # C++ parity: ``applyAfterSolving`` forwards to ``applyAfterApplying``.
        """
        self.apply_after_applying(u)

    def set_time(self, t: float) -> None:
        """No-op — the boundary value is time-independent.

        # C++ parity: ``void setTime(Time) override {}``.
        """
        del t

    def apply_after_applying_value(self, x: float, value: float) -> float:
        """Scalar form: replace ``value`` when ``x`` lies outside the face.

        # C++ parity: ``Real applyAfterApplying(Real x, Real value) const``
        # — the value is overridden only *strictly* beyond the extreme
        # location, so a point sitting exactly on the boundary keeps its
        # own value.
        """
        outside = (self._side == BoundaryConditionSide.LOWER and x < self._x_extreme) or (
            self._side == BoundaryConditionSide.UPPER and x > self._x_extreme
        )
        return self._value_on_boundary if outside else value


__all__ = ["FdmDirichletBoundary"]
