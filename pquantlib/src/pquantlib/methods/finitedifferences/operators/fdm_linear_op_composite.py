"""FdmLinearOpComposite — Protocol for composable FD linear operators.

# C++ parity: ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp
# (v1.42.1).

A composite FD linear operator exposes the following surface used by
the time-stepping schemes (Crank-Nicolson / implicit-Euler / explicit-
Euler) and the backward solver:

* ``size()`` — number of directions.
* ``set_time(t1, t2)`` — recompute time-dependent coefficients over
  ``[t1, t2]``.
* ``apply(r)`` — full ``L @ r``.
* ``apply_mixed(r)`` — only the mixed-derivative contribution.
* ``apply_direction(direction, r)`` — only the directional component.
* ``solve_splitting(direction, r, dt)`` — ``(I - dt * L_dir)^{-1} r``.
* ``preconditioner(r, dt)`` — preconditioner (default = solve along 0).

For 1-D operators (BSM, OU, Dupire) ``size() == 1``, ``apply_mixed``
returns zero/identity, ``apply_direction(0, r) == apply(r)``.

This Protocol replaces the previous tight coupling of the schemes
to ``FdmBlackScholesOp`` — any concrete op satisfying this surface
can now be plugged into ``CrankNicolsonScheme`` / ``ImplicitEuler``
/ ``ExplicitEuler`` / ``FdmBackwardSolver``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pquantlib.math.array import Array


@runtime_checkable
class FdmLinearOpComposite(Protocol):
    """Protocol for composite FD linear operators.

    # C++ parity: ``class FdmLinearOpComposite : public FdmLinearOp``
    # — abstract virtual methods that 1-D / 2-D / 3-D operators
    # override. Python uses a structural Protocol instead.
    """

    def size(self) -> int:
        """Number of directions (1 for 1-D ops; 2 for 2-D; etc.)."""
        ...

    def set_time(self, t1: float, t2: float) -> None:
        """Update time-dependent coefficients over ``[t1, t2]``."""
        ...

    def apply(self, r: Array) -> Array:
        """Apply the operator: ``L @ r``."""
        ...

    def apply_mixed(self, r: Array) -> Array:
        """Apply only the mixed-derivative contribution."""
        ...

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Apply only the contribution along ``direction``."""
        ...

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """Solve ``(I - dt * L_dir) x = r`` along ``direction``."""
        ...

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner application (default: solve along direction 0)."""
        ...


__all__ = ["FdmLinearOpComposite"]
