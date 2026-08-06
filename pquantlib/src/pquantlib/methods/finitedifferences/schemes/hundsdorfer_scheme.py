"""HundsdorferScheme — Hundsdorfer-Verwer operator splitting.

# C++ parity: ql/methods/finitedifferences/schemes/hundsdorferscheme.{hpp,cpp}
# (v1.43).

Same skeleton as :class:`CraigSneydScheme`, with two deliberate differences in
the corrector half — both of which this port keeps verbatim, since they are
the whole content of the scheme::

    yt = y0 + mu*dt*L (y - a)                            (full L, not L_mixed)
    yt <- (I - theta*dt*L_i)^-1 (yt - theta*dt*L_i y)    (linearised at y, not a)

The canonical parameter pair is ``theta = 1/2 + sqrt(3)/6``, ``mu = 1/2``.

Naming: the C++ constructor parameter is ``map``; the Python port calls it
``op`` for consistency with the sibling schemes in this package.
"""

from __future__ import annotations

from typing import final

import numpy as np

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
class HundsdorferScheme:
    """Hundsdorfer-Verwer ADI one-step evolver.

    # C++ parity: ``class HundsdorferScheme``.
    """

    __slots__ = ("_bc_set", "_dt", "_mu", "_op", "_theta")

    def __init__(
        self,
        theta: float,
        mu: float,
        op: FdmLinearOpComposite,
        bc_set: FdmBoundaryConditionSet = (),
    ) -> None:
        # C++ parity: dt_ starts at Null<Real>(); NaN is the Python analogue.
        self._dt: float = float("nan")
        self._theta: float = theta
        self._mu: float = mu
        self._op: FdmLinearOpComposite = op
        self._bc_set: BoundaryConditionSchemeHelper = BoundaryConditionSchemeHelper(bc_set)

    def set_step(self, dt: float) -> None:
        """# C++ parity: ``HundsdorferScheme::setStep``."""
        self._dt = dt

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt`` and return the new array.

        # C++ parity: ``HundsdorferScheme::step(array_type& a, Time t)``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        t1 = max(0.0, t - self._dt)
        self._op.set_time(t1, t)
        self._bc_set.set_time(t1)

        self._bc_set.apply_before_applying(self._op)
        y = a + self._dt * self._op.apply(a)
        self._bc_set.apply_after_applying(y)

        # C++ ``Array y0 = y;`` is a value copy; ``y`` is rebound below.
        y0 = np.array(y, dtype=np.float64, copy=True)

        for i in range(self._op.size()):
            rhs = y - (self._theta * self._dt) * self._op.apply_direction(i, a)
            y = self._op.solve_splitting(i, rhs, -self._theta * self._dt)

        self._bc_set.apply_before_applying(self._op)
        yt = y0 + (self._mu * self._dt) * self._op.apply(y - a)
        self._bc_set.apply_after_applying(yt)

        for i in range(self._op.size()):
            rhs = yt - (self._theta * self._dt) * self._op.apply_direction(i, y)
            yt = self._op.solve_splitting(i, rhs, -self._theta * self._dt)
        self._bc_set.apply_after_solving(yt)

        return yt


__all__ = ["HundsdorferScheme"]
