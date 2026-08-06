"""MethodOfLinesScheme — method-of-lines time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/methodoflinesscheme.{hpp,cpp}
# (v1.43).

Instead of discretising in time, the method of lines hands the semi-discrete
ODE system ``du/dt = -L(t) u`` to an adaptive ODE integrator and lets it pick
its own internal steps between ``t`` and ``t - dt``.

The integrator is QuantLib's own :class:`AdaptiveRungeKutta` (Cash-Karp 4/5),
constructed as ``AdaptiveRungeKutta<Real>(eps, relInitStepSize*dt)`` — i.e.
``hmin`` left at its 0.0 default. It is emphatically **not**
``scipy.integrate.solve_ivp``: the step controller differs in its error norm,
its shrink/grow clamps and its failure behaviour, so the two do not produce
the same trajectory. See the ``adaptive_runge_kutta`` module docstring.

The operator is re-timed at every stage evaluation with the C++ hard-coded
lookahead ``setTime(t, t + 0.0001)``, and the boundary conditions'
``applyBeforeApplying`` hook runs inside the RHS. Note that
``MethodOfLinesScheme::step`` never calls ``bcSet_.setTime`` — only
``applyAfterSolving`` at the end — so a time-dependent boundary condition
keeps whatever value it was last set to. That is C++ behaviour and is
reproduced here rather than "fixed".

Naming: the C++ constructor parameter is ``map``; the Python port calls it
``op`` for consistency with the sibling schemes in this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.ode.adaptive_runge_kutta import AdaptiveRungeKutta
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
class MethodOfLinesScheme:
    """Method-of-lines one-step evolver driven by an adaptive Runge-Kutta.

    # C++ parity: ``class MethodOfLinesScheme``.
    """

    __slots__ = ("_bc_set", "_dt", "_eps", "_op", "_rel_init_step_size")

    def __init__(
        self,
        eps: float,
        rel_init_step_size: float,
        op: FdmLinearOpComposite,
        bc_set: FdmBoundaryConditionSet = (),
    ) -> None:
        # C++ parity: dt_ starts at Null<Real>(); NaN is the Python analogue.
        self._dt: float = float("nan")
        self._eps: float = eps
        self._rel_init_step_size: float = rel_init_step_size
        self._op: FdmLinearOpComposite = op
        self._bc_set: BoundaryConditionSchemeHelper = BoundaryConditionSchemeHelper(bc_set)

    def set_step(self, dt: float) -> None:
        """# C++ parity: ``MethodOfLinesScheme::setStep``."""
        self._dt = dt

    def _apply(self, t: float, u: Sequence[float]) -> list[float]:
        """ODE right-hand side handed to the Runge-Kutta integrator.

        # C++ parity: ``std::vector<Real> MethodOfLinesScheme::apply(Time,
        # const std::vector<Real>&) const``.
        """
        self._op.set_time(t, t + 0.0001)
        self._bc_set.apply_before_applying(self._op)

        dxdt = -self._op.apply(np.asarray(u, dtype=np.float64))

        return [float(v) for v in dxdt]

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` from ``t`` to ``t - dt`` and return the new array.

        # C++ parity: ``MethodOfLinesScheme::step(array_type& a, Time t)``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")

        v = AdaptiveRungeKutta(self._eps, self._rel_init_step_size * self._dt).solve(
            self._apply,
            [float(x) for x in a],
            t,
            max(0.0, t - self._dt),
        )

        y: Array = np.array(v, dtype=np.float64)
        self._bc_set.apply_after_solving(y)

        return y


__all__ = ["MethodOfLinesScheme"]
