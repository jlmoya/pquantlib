"""HundsdorferScheme — Hundsdorfer-Verwer operator splitting.

# C++ parity: ql/methods/finitedifferences/schemes/hundsdorferscheme.{hpp,cpp}
# (v1.43).

A two-stage ADI predictor/corrector for multi-dimensional operators: an
explicit full-operator step, a per-direction implicit correction, then a
``mu``-weighted second pass over both. Unlike Douglas it stays
second-order accurate in the presence of a mixed-derivative term, which
is why the ZABR 2-D PDE uses it.

``FdmSchemeDesc.hundsdorfer()`` supplies ``theta = 0.5 + sqrt(3)/6`` and
``mu = 0.5``.

Boundary conditions are not modelled — the ported operators are used
with an empty ``FdmBoundaryConditionSet``, as ZABR's ``fdPrice`` /
``fullFdPrice`` do.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)


@final
class HundsdorferScheme:
    """Hundsdorfer-Verwer ADI one-step evolver.

    # C++ parity: ``class HundsdorferScheme``.
    """

    def __init__(self, theta: float, mu: float, op: FdmLinearOpComposite) -> None:
        self._theta: float = theta
        self._mu: float = mu
        self._op: FdmLinearOpComposite = op
        self._dt: float = float("nan")

    def set_step(self, dt: float) -> None:
        self._dt = dt

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt``.

        # C++ parity: ``HundsdorferScheme::step``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        dt = self._dt
        self._op.set_time(max(0.0, t - dt), t)

        y = a + dt * self._op.apply(a)
        y0 = y

        for i in range(self._op.size()):
            rhs = y - self._theta * dt * self._op.apply_direction(i, a)
            y = self._op.solve_splitting(i, rhs, -self._theta * dt)

        yt = y0 + self._mu * dt * self._op.apply(y - a)

        for i in range(self._op.size()):
            rhs = yt - self._theta * dt * self._op.apply_direction(i, y)
            yt = self._op.solve_splitting(i, rhs, -self._theta * dt)

        return yt


__all__ = ["HundsdorferScheme"]
