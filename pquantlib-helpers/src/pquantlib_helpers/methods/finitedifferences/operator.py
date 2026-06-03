"""Operator — structural interface for finite-difference differential operators.

# Retired-API compat layer — see package docstring.

Java parity: ``org.jquantlib.methods.finitedifferences.Operator`` (interface).
C++ parity: there is no standalone ``Operator`` interface in C++; the modern
v1.42.1 ``MixedScheme`` is a template over a duck-typed ``Operator`` concept
(see ``mixedscheme.hpp`` doc comment listing the required member functions).
We model that concept as a :class:`typing.Protocol` so :class:`MixedScheme`
can be parametrized over any conforming operator (the only concrete one in
this cluster being :class:`TridiagonalOperator`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pquantlib.math.array import Array


@runtime_checkable
class Operator(Protocol):
    """Structural interface a finite-difference evolver's operator must satisfy.

    Java parity: ``interface Operator``. The C++ ``MixedScheme`` template
    requires exactly this member surface (``size``, ``isTimeDependent``,
    ``setTime``, ``identity``, ``applyTo``, ``solveFor``, and the operator
    algebra ``+ - *``).
    """

    def size(self) -> int:
        """Number of grid points (the operator's square dimension)."""
        ...

    def is_time_dependent(self) -> bool:
        """``True`` iff a time-setter hook is installed."""
        ...

    def set_time(self, t: float) -> None:
        """Update time-dependent coefficients to time ``t`` (no-op if constant)."""
        ...

    def identity(self, size: int) -> Operator:
        """Return the identity operator of the given size."""
        ...

    def apply_to(self, v: Array) -> Array:
        """Apply the operator to ``v`` (matrix-vector product)."""
        ...

    def solve_for(self, rhs: Array) -> Array:
        """Solve the linear system ``self @ x = rhs`` for ``x``."""
        ...

    def add(self, other: Operator) -> Operator:
        """Operator sum ``self + other``."""
        ...

    def subtract(self, other: Operator) -> Operator:
        """Operator difference ``self - other``."""
        ...

    def multiply(self, a: float) -> Operator:
        """Scalar multiple ``a * self``."""
        ...


__all__ = ["Operator"]
