"""CrankNicolsonScheme — Crank-Nicolson (theta=0.5) time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/cranknicolsonscheme.{hpp,cpp}
# (v1.42.1).

Crank-Nicolson is the midpoint rule

    (I - theta*dt*L) a_new = (I + (1-theta)*dt*L) a_old

with ``theta = 0.5``. In 1-D this is equivalent to the Douglas scheme.
Implementation = one explicit step (``1-theta``) followed by one
implicit step (``theta``).
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.methods.finitedifferences.schemes.explicit_euler_scheme import (
    ExplicitEulerScheme,
)
from pquantlib.methods.finitedifferences.schemes.implicit_euler_scheme import (
    ImplicitEulerScheme,
)


@final
class CrankNicolsonScheme:
    """Theta-weighted explicit + implicit one-step evolver.

    # C++ parity: ``class CrankNicolsonScheme`` — same internal
    # composition (an ``ExplicitEulerScheme`` + ``ImplicitEulerScheme``
    # tied to the same operator).

    # Phase 11 W5-C: generalized ``op`` argument from concrete
    # ``FdmBlackScholesOp`` to ``FdmLinearOpComposite`` Protocol so OU
    # / Dupire / ZABR ops can be plugged in too.
    """

    def __init__(self, theta: float, op: FdmLinearOpComposite) -> None:
        self._theta: float = theta
        self._dt: float = float("nan")
        self._explicit: ExplicitEulerScheme = ExplicitEulerScheme(op)
        self._implicit: ImplicitEulerScheme = ImplicitEulerScheme(op)

    def set_step(self, dt: float) -> None:
        self._dt = dt
        self._explicit.set_step(dt)
        self._implicit.set_step(dt)

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt`` via theta-weighted CN.

        # C++ parity: ``CrankNicolsonScheme::step(a, t)``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        if self._theta != 1.0:
            a = self._explicit.step(a, t, 1.0 - self._theta)
        if self._theta != 0.0:
            a = self._implicit.step(a, t, self._theta)
        return a


__all__ = ["CrankNicolsonScheme"]
