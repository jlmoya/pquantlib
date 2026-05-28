"""ImplicitEulerScheme — implicit-Euler time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/impliciteulerscheme.{hpp,cpp}
# (v1.42.1).

One step of implicit Euler solves

    (I - dt * L(t)) a_new = a_old

via the operator's ``solve_splitting`` (tridiagonal Thomas algorithm)
when the operator has a single direction (the 1-D BSM case). The
multi-direction iterative path (BiCGstab / GMRES) is deferred to
Phase 6 — the L5-D scope is purely 1-D.

Boundary conditions are deferred (see ``ExplicitEulerScheme`` docstring).
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import (
    FdmBlackScholesOp,
)


@final
class ImplicitEulerScheme:
    """Implicit-Euler one-step evolver.

    # C++ parity: ``class ImplicitEulerScheme``.
    """

    def __init__(self, op: FdmBlackScholesOp) -> None:
        self._op: FdmBlackScholesOp = op
        self._dt: float = float("nan")

    def set_step(self, dt: float) -> None:
        self._dt = dt

    def step(self, a: Array, t: float, theta: float = 1.0) -> Array:
        """Solve ``(I - theta * dt * L(t)) a_new = a`` and return ``a_new``.

        # C++ parity: ``ImplicitEulerScheme::step(a, t, theta)``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        t1 = max(0.0, t - self._dt)
        self._op.set_time(t1, t)
        # solve_splitting solves (b*I + a*L) x = r  with a = -theta*dt, b = 1.
        # That gives x = (I - theta*dt*L)^{-1} r — exactly the implicit-Euler step.
        return self._op.solve_splitting(0, a, -theta * self._dt)


__all__ = ["ImplicitEulerScheme"]
