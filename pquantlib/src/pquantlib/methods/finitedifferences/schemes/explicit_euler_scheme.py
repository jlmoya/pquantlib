"""ExplicitEulerScheme — explicit-Euler time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/expliciteulerscheme.{hpp,cpp}
# (v1.42.1).

One step of explicit Euler advances the unknown ``a`` by

    a <- a + dt * L(t) @ a

The operator's ``set_time(t1, t2)`` is invoked before each step.

Boundary conditions are deferred (the L5-D scope uses zero/no-flow
boundaries inherent in the operator stencil — see
``second_derivative_op.SecondDerivativeOp`` which zeros the boundary
rows).
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import (
    FdmBlackScholesOp,
)


@final
class ExplicitEulerScheme:
    """Explicit-Euler one-step evolver.

    # C++ parity: ``class ExplicitEulerScheme``.
    """

    def __init__(self, op: FdmBlackScholesOp) -> None:
        self._op: FdmBlackScholesOp = op
        self._dt: float = float("nan")

    def set_step(self, dt: float) -> None:
        self._dt = dt

    def step(self, a: Array, t: float, theta: float = 1.0) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt`` (backward in time).

        # C++ parity: ``ExplicitEulerScheme::step(a, t, theta)``.
        Returns the new array (Python uses an immutable contract on
        arrays — caller assigns).
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        t1 = max(0.0, t - self._dt)
        self._op.set_time(t1, t)
        return a + (theta * self._dt) * self._op.apply(a)


__all__ = ["ExplicitEulerScheme"]
