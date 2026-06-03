"""Boundary conditions for the legacy finite-difference framework.

# Retired-API compat layer — see package docstring.

C++ parity: ``ql/methods/finitedifferences/boundarycondition.hpp`` (v1.42.1).
That header ships the abstract ``BoundaryCondition<Operator>`` interface (the
four ``apply*`` hooks plus ``setTime``) together with the ``Side`` enum and the
two concrete subclasses ``NeumannBC`` and ``DirichletBC``.

Java parity: ``org.jquantlib.methods.finitedifferences`` —
``BoundaryCondition`` (interface + ``Side`` enum), ``DirichletBC``,
``NeumannBC``, ``BoundaryConditionSet``.

FD-alpha1 already declared a structural :class:`BoundaryCondition` *Protocol*
(the hook surface consumed by :class:`MixedScheme`); the concrete classes here
implement exactly that surface, so they are accepted anywhere the Protocol is
expected.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib.exceptions import LibraryException

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
        TridiagonalOperator,
    )


class Side(IntEnum):
    """Which boundary the condition is imposed on.

    C++ parity: ``BoundaryCondition::Side`` (``None``, ``Upper``, ``Lower``).
    Java parity: ``BoundaryCondition.Side``. We rename the ``None`` member to
    ``Nil`` because ``None`` is a Python keyword; the integer value is unchanged.
    """

    Nil = 0
    Upper = 1
    Lower = 2


class DirichletBC:
    """Dirichlet boundary condition — pins a constant *value* at the boundary.

    C++ parity: ``class DirichletBC : public BoundaryCondition<TridiagonalOperator>``.
    Java parity: ``DirichletBC implements BoundaryCondition<TridiagonalOperator>``.

    The boundary node is forced to hold ``value``:

    - ``apply_before_applying`` rewrites the boundary operator row to the
      identity row (so ``applyTo`` leaves the boundary entry unchanged), and
    - ``apply_after_applying`` overwrites the boundary entry with ``value``;
    - on the implicit half ``apply_before_solving`` writes the identity row and
      sets the RHS boundary entry to ``value`` so the solve returns it exactly.
    """

    def __init__(self, value: float, side: Side) -> None:
        """Pin ``value`` on the given ``side`` of the grid."""
        self._value = value
        self._side = side

    def set_time(self, t: float) -> None:
        """No-op (this condition is time-independent). C++ parity: empty body."""

    def apply_before_applying(self, op: TridiagonalOperator) -> None:
        """Rewrite the boundary operator row to the identity row."""
        if self._side == Side.Lower:
            op.set_first_row(1.0, 0.0)
        elif self._side == Side.Upper:
            op.set_last_row(0.0, 1.0)
        else:
            raise LibraryException("unknown side for Dirichlet boundary condition")

    def apply_after_applying(self, a: Array) -> None:
        """Overwrite the boundary array entry with ``value``."""
        if self._side == Side.Lower:
            a[0] = self._value
        elif self._side == Side.Upper:
            a[a.shape[0] - 1] = self._value
        else:
            raise LibraryException("unknown side for Dirichlet boundary condition")

    def apply_before_solving(self, op: TridiagonalOperator, a: Array) -> None:
        """Identity-ify the boundary row and set the RHS boundary entry to ``value``."""
        if self._side == Side.Lower:
            op.set_first_row(1.0, 0.0)
            a[0] = self._value
        elif self._side == Side.Upper:
            op.set_last_row(0.0, 1.0)
            a[a.shape[0] - 1] = self._value
        else:
            raise LibraryException("unknown side for Dirichlet boundary condition")

    def apply_after_solving(self, a: Array) -> None:
        """No-op. C++ parity: empty body."""


class NeumannBC:
    """Neumann boundary condition — pins a constant first *derivative* at the boundary.

    C++ parity: ``class NeumannBC : public BoundaryCondition<TridiagonalOperator>``.
    Java parity: ``NeumannBC implements BoundaryCondition<TridiagonalOperator>``.

    ``value`` is the imposed finite-difference of the boundary node against its
    interior neighbour. On the explicit half the boundary operator row becomes
    ``(-1, 1)`` and ``apply_after_applying`` enforces
    ``u[boundary] = u[neighbour] ∓ value``; on the implicit half the RHS
    boundary entry is set to ``value``.
    """

    def __init__(self, value: float, side: Side) -> None:
        """Impose first-derivative ``value`` on the given ``side``."""
        self._value = value
        self._side = side

    def set_time(self, t: float) -> None:
        """No-op (time-independent). C++ parity: empty body."""

    def apply_before_applying(self, op: TridiagonalOperator) -> None:
        """Rewrite the boundary operator row to the difference row ``(-1, 1)``."""
        if self._side == Side.Lower:
            op.set_first_row(-1.0, 1.0)
        elif self._side == Side.Upper:
            op.set_last_row(-1.0, 1.0)
        else:
            raise LibraryException("unknown side for Neumann boundary condition")

    def apply_after_applying(self, a: Array) -> None:
        """Enforce the boundary derivative against the interior neighbour."""
        n = a.shape[0]
        if self._side == Side.Lower:
            a[0] = a[1] - self._value
        elif self._side == Side.Upper:
            a[n - 1] = a[n - 2] + self._value
        else:
            raise LibraryException("unknown side for Neumann boundary condition")

    def apply_before_solving(self, op: TridiagonalOperator, a: Array) -> None:
        """Set the boundary difference row and the RHS boundary entry to ``value``."""
        n = a.shape[0]
        if self._side == Side.Lower:
            op.set_first_row(-1.0, 1.0)
            a[0] = self._value
        elif self._side == Side.Upper:
            op.set_last_row(-1.0, 1.0)
            a[n - 1] = self._value
        else:
            raise LibraryException("unknown side for Neumann boundary condition")

    def apply_after_solving(self, a: Array) -> None:
        """No-op. C++ parity: empty body."""


class BoundaryConditionSet:
    """Ordered set of boundary-condition lists, one list per system component.

    C++ parity: ``BoundaryConditionSet`` (a ``std::vector`` of bc-lists used by
    the system FD model). Java parity: ``BoundaryConditionSet<T>``.

    Only consumed by the system/parallel FD path (deferred to FD-beta); kept
    here for completeness of the conditions layer.
    """

    def __init__(self) -> None:
        """Create an empty set."""
        self._bc_set: list[list[object]] = []

    def push_back(self, a: list[object]) -> None:
        """Append a boundary-condition list (one system component)."""
        self._bc_set.append(a)

    def get(self, i: int) -> list[object]:
        """Return the boundary-condition list for component ``i``."""
        return self._bc_set[i]


__all__ = [
    "BoundaryConditionSet",
    "DirichletBC",
    "NeumannBC",
    "Side",
]
