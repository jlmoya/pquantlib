"""CrankNicolsonScheme — theta-weighted explicit + implicit stepping.

# C++ parity: ql/methods/finitedifferences/schemes/cranknicolsonscheme.{hpp,cpp}
# (v1.43).

    (I - theta*dt*L) a_new = (I + (1-theta)*dt*L) a_old

implemented, exactly as C++ does, as one ``ExplicitEulerScheme`` step with
weight ``1-theta`` followed by one ``ImplicitEulerScheme`` step with weight
``theta``. In one dimension this is the Douglas scheme; in higher dimensions
it is usually inferior to the operator-splitting schemes.
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
from pquantlib.methods.finitedifferences.schemes.explicit_euler_scheme import (
    ExplicitEulerScheme,
)
from pquantlib.methods.finitedifferences.schemes.implicit_euler_scheme import (
    ImplicitEulerScheme,
    ImplicitEulerSchemeSolverType,
)


@final
class CrankNicolsonScheme:
    """Theta-weighted explicit + implicit one-step evolver.

    # C++ parity: ``class CrankNicolsonScheme`` — same internal composition
    # (an ``ExplicitEulerScheme`` + an ``ImplicitEulerScheme`` over the same
    # operator and the same boundary-condition set).
    """

    __slots__ = ("_dt", "_explicit", "_implicit", "_theta")

    def __init__(
        self,
        theta: float,
        op: FdmLinearOpComposite,
        bc_set: FdmBoundaryConditionSet = (),
        rel_tol: float = 1e-8,
        solver_type: ImplicitEulerSchemeSolverType = ImplicitEulerSchemeSolverType.BiCGstab,
    ) -> None:
        self._theta: float = theta
        self._dt: float = float("nan")
        self._explicit: ExplicitEulerScheme = ExplicitEulerScheme(op, bc_set)
        self._implicit: ImplicitEulerScheme = ImplicitEulerScheme(
            op, bc_set, rel_tol, solver_type
        )

    def set_step(self, dt: float) -> None:
        """# C++ parity: ``CrankNicolsonScheme::setStep``."""
        self._dt = dt
        self._explicit.set_step(dt)
        self._implicit.set_step(dt)

    def number_of_iterations(self) -> int:
        """# C++ parity: ``CrankNicolsonScheme::numberOfIterations``."""
        return self._implicit.number_of_iterations()

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt`` via the theta rule.

        # C++ parity: ``CrankNicolsonScheme::step(a, t)``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        if self._theta != 1.0:
            a = self._explicit.step(a, t, 1.0 - self._theta)
        if self._theta != 0.0:
            a = self._implicit.step(a, t, self._theta)
        return a


__all__ = ["CrankNicolsonScheme"]
