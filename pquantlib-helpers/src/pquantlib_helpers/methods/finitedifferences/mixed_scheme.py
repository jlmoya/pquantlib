"""MixedScheme — mixed explicit/implicit finite-difference evolver.

# Retired-API compat layer — see package docstring.

Java parity: ``org.jquantlib.methods.finitedifferences.MixedScheme<T>``.
C++ parity: ``ql/methods/finitedifferences/mixedscheme.hpp`` (v1.42.1, marked
``[[deprecated]]`` — "Part of the old FD framework").

One ``theta``-weighted step combines an explicit Euler half (``I - (1-theta)*dt*L``,
applied) with an implicit Euler half (``I + theta*dt*L``, solved):

- ``theta = 0``   -> fully explicit (ExplicitEuler).
- ``theta = 0.5`` -> Crank-Nicolson.
- ``theta = 1``   -> fully implicit (ImplicitEuler).

The differential operator must be linear for this evolver to work.

# Boundary conditions: the C++/Java ``step`` consults a ``bc_set`` and calls
# ``applyBeforeApplying`` / ``applyAfterApplying`` / ``applyBeforeSolving`` /
# ``applyAfterSolving`` hooks. Concrete boundary conditions (Dirichlet/Neumann)
# are part of the next cluster (FD-alpha2). We model the hook surface as a
# :class:`BoundaryCondition` Protocol so the full step algorithm is faithful,
# and default ``bcs`` to an empty list (the dividend FD path applies none).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)

if TYPE_CHECKING:
    from pquantlib.math.array import Array


class BoundaryCondition(Protocol):
    """Hook surface a finite-difference boundary condition must satisfy.

    C++ parity: ``BoundaryCondition<Operator>`` (the four ``apply*`` hooks plus
    ``setTime``). Concrete implementations land in FD-alpha2.
    """

    def set_time(self, t: float) -> None:
        """Update the boundary condition for time ``t``."""
        ...

    def apply_before_applying(self, op: TridiagonalOperator) -> None:
        """Adjust the explicit operator before ``applyTo``."""
        ...

    def apply_after_applying(self, a: Array) -> None:
        """Adjust the array after ``applyTo``."""
        ...

    def apply_before_solving(self, op: TridiagonalOperator, a: Array) -> None:
        """Adjust the implicit operator/array before ``solveFor``."""
        ...

    def apply_after_solving(self, a: Array) -> None:
        """Adjust the array after ``solveFor``."""
        ...


class MixedScheme:
    """Mixed explicit/implicit (theta-weighted) finite-difference scheme.

    Java parity: ``MixedScheme<T extends Operator>``. Parametrised here over
    :class:`TridiagonalOperator` (the only concrete operator in this cluster).
    """

    def __init__(
        self,
        op: TridiagonalOperator,
        theta: float,
        bcs: list[BoundaryCondition] | None = None,
    ) -> None:
        """Build an evolver around operator ``op`` with weight ``theta``."""
        self._l: TridiagonalOperator = op
        self._i: TridiagonalOperator = op.identity(op.size())
        self._theta: float = theta
        self._bcs: list[BoundaryCondition] = list(bcs) if bcs is not None else []
        self._dt: float = 0.0
        self._explicit_part: TridiagonalOperator | None = None
        self._implicit_part: TridiagonalOperator | None = None

    def set_step(self, dt: float) -> None:
        """Cache the explicit/implicit operators for the time step ``dt``.

        C++ parity: ``void MixedScheme::setStep(Time dt)``.
        """
        self._dt = dt
        if self._theta != 1.0:  # there is an explicit part
            # I - ((1 - theta) * dt) * L
            self._explicit_part = self._i.subtract(
                self._l.multiply((1.0 - self._theta) * dt)
            )
        if self._theta != 0.0:  # there is an implicit part
            # I + (theta * dt) * L
            self._implicit_part = self._i.add(self._l.multiply(self._theta * dt))

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` by one step ending at time ``t``; return the new array.

        C++ parity: ``void MixedScheme::step(array_type& a, Time t)`` (C++
        mutates in place; we return the resulting array for ergonomics, and the
        :class:`FiniteDifferenceModel` rebinds ``a = evolver.step(a, now)``).
        """
        for bc in self._bcs:
            bc.set_time(t)

        if self._theta != 1.0:  # explicit part
            if self._l.is_time_dependent():
                self._l.set_time(t)
                self._explicit_part = self._i.subtract(
                    self._l.multiply((1.0 - self._theta) * self._dt)
                )
            assert self._explicit_part is not None
            for bc in self._bcs:
                bc.apply_before_applying(self._explicit_part)
            a = self._explicit_part.apply_to(a)
            for bc in self._bcs:
                bc.apply_after_applying(a)

        if self._theta != 0.0:  # implicit part
            if self._l.is_time_dependent():
                self._l.set_time(t - self._dt)
                self._implicit_part = self._i.add(
                    self._l.multiply(self._theta * self._dt)
                )
            assert self._implicit_part is not None
            for bc in self._bcs:
                bc.apply_before_solving(self._implicit_part, a)
            a = self._implicit_part.solve_for(a)
            for bc in self._bcs:
                bc.apply_after_solving(a)

        return a


__all__ = ["BoundaryCondition", "MixedScheme"]
