"""ImplicitEulerScheme — implicit-Euler time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/impliciteulerscheme.{hpp,cpp}
# (v1.42.1).

One step of implicit Euler solves

    (I - dt * L(t)) a_new = a_old

via the operator's ``solve_splitting`` (tridiagonal Thomas algorithm)
when the operator has a single direction (the 1-D BSM case). For
multi-direction operators the system is NOT split — C++ solves the full
operator iteratively with :class:`BiCGstab`, preconditioned by a
splitting solve along direction 0. The GMRES ``SolverType`` alternative
is not ported; nothing in the library selects it.

Because the multi-D branch stops at a relative residual of ``rel_tol``
(C++ default 1e-8), results that pass through it agree with an exact
solve only to that order.

Boundary conditions are deferred (see ``ExplicitEulerScheme`` docstring).
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.matrixutilities.bicgstab import BiCGstab
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)


@final
class ImplicitEulerScheme:
    """Implicit-Euler one-step evolver.

    # C++ parity: ``class ImplicitEulerScheme``.

    # Phase 11 W5-C: generalized ``op`` argument from concrete
    # ``FdmBlackScholesOp`` to ``FdmLinearOpComposite`` Protocol so OU
    # / Dupire / ZABR ops can be plugged in too.
    """

    def __init__(self, op: FdmLinearOpComposite, rel_tol: float = 1e-8) -> None:
        self._op: FdmLinearOpComposite = op
        self._dt: float = float("nan")
        self._rel_tol: float = rel_tol
        self._iterations: int = 0

    def set_step(self, dt: float) -> None:
        self._dt = dt

    def number_of_iterations(self) -> int:
        """Cumulative BiCGstab iterations across all multi-D steps taken.

        # C++ parity: ``ImplicitEulerScheme::numberOfIterations``.
        """
        return self._iterations

    def step(self, a: Array, t: float, theta: float = 1.0) -> Array:
        """Solve ``(I - theta * dt * L(t)) a_new = a`` and return ``a_new``.

        # C++ parity: ``ImplicitEulerScheme::step(a, t, theta)``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        t1 = max(0.0, t - self._dt)
        self._op.set_time(t1, t)
        if self._op.size() == 1:
            # solve_splitting solves (b*I + a*L) x = r  with a = -theta*dt, b = 1.
            # That gives x = (I - theta*dt*L)^{-1} r — the implicit-Euler step.
            return self._op.solve_splitting(0, a, -theta * self._dt)

        def apply_f(r: Array) -> Array:
            return r - (theta * self._dt) * self._op.apply(r)

        def preconditioner(r: Array) -> Array:
            return self._op.preconditioner(r, -theta * self._dt)

        result = BiCGstab(apply_f, max(10, a.size), self._rel_tol, preconditioner).solve(a, a)
        self._iterations += result.iterations
        return result.x


__all__ = ["ImplicitEulerScheme"]
