"""ExplicitEulerScheme — explicit-Euler time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/expliciteulerscheme.{hpp,cpp}
# (v1.43).

One step advances ``a <- a + theta*dt * L(t) a``, with the boundary-condition
set applied before the ``apply`` and after it.
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
class ExplicitEulerScheme:
    """Explicit-Euler one-step evolver.

    # C++ parity: ``class ExplicitEulerScheme``.
    """

    __slots__ = ("_bc_set", "_dt", "_op")

    def __init__(
        self,
        op: FdmLinearOpComposite,
        bc_set: FdmBoundaryConditionSet = (),
    ) -> None:
        self._op: FdmLinearOpComposite = op
        self._dt: float = float("nan")
        self._bc_set: BoundaryConditionSchemeHelper = BoundaryConditionSchemeHelper(bc_set)

    def set_step(self, dt: float) -> None:
        """# C++ parity: ``ExplicitEulerScheme::setStep``."""
        self._dt = dt

    def step(self, a: Array, t: float, theta: float = 1.0) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt`` (backward in time).

        # C++ parity: ``ExplicitEulerScheme::step(a, t, theta)``. C++ mutates
        # its reference argument; the Python FD package returns the new array.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        t1 = max(0.0, t - self._dt)
        self._op.set_time(t1, t)
        self._bc_set.set_time(t1)

        self._bc_set.apply_before_applying(self._op)
        result = a + (theta * self._dt) * self._op.apply(a)
        self._bc_set.apply_after_applying(result)
        return result


__all__ = ["ExplicitEulerScheme"]
