"""DouglasScheme — Douglas operator splitting.

# C++ parity: ql/methods/finitedifferences/schemes/douglasscheme.{hpp,cpp}
# (v1.43).

One step is an explicit Euler predictor followed by one implicit correction
per direction::

    y  = a + dt * L a
    y <- (I - theta*dt*L_i)^-1 (y - theta*dt*L_i a)   for i = 0 .. size-1

In one dimension this is algebraically the Crank-Nicolson scheme; the
splitting loop is what makes it a distinct scheme in two or more.

Naming: the C++ constructor parameter is ``map``; the Python port calls it
``op``, matching the sibling schemes already in this package (``map`` is a
Python builtin).
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    FdmBoundaryConditionSet,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.methods.finitedifferences.schemes.boundary_condition_scheme_helper import (
    BoundaryConditionSchemeHelper,
)


@final
class DouglasScheme:
    """Douglas ADI one-step evolver.

    # C++ parity: ``class DouglasScheme``.
    """

    __slots__ = ("_bc_set", "_dt", "_op", "_theta")

    def __init__(
        self,
        theta: float,
        op: FdmLinearOpComposite,
        bc_set: FdmBoundaryConditionSet = (),
    ) -> None:
        # C++ parity: dt_ starts at Null<Real>(); NaN is the Python analogue.
        self._dt: float = float("nan")
        self._theta: float = theta
        self._op: FdmLinearOpComposite = op
        self._bc_set: BoundaryConditionSchemeHelper = BoundaryConditionSchemeHelper(bc_set)

    def set_step(self, dt: float) -> None:
        """# C++ parity: ``DouglasScheme::setStep``."""
        self._dt = dt

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt`` and return the new array.

        # C++ parity: ``DouglasScheme::step(array_type& a, Time t)``. C++
        # mutates its reference argument; the Python FD package returns the new
        # array instead (see ``ExplicitEulerScheme``), so ``a`` is left intact.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        t1 = max(0.0, t - self._dt)
        self._op.set_time(t1, t)
        self._bc_set.set_time(t1)

        self._bc_set.apply_before_applying(self._op)
        y = a + self._dt * self._op.apply(a)
        self._bc_set.apply_after_applying(y)

        for i in range(self._op.size()):
            rhs = y - (self._theta * self._dt) * self._op.apply_direction(i, a)
            y = self._op.solve_splitting(i, rhs, -self._theta * self._dt)
        self._bc_set.apply_after_solving(y)

        return y


__all__ = ["DouglasScheme"]
