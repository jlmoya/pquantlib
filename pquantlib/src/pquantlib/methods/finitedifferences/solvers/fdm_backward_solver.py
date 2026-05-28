"""FdmBackwardSolver — back-propagate maturity payoff to t=0.

# C++ parity: ql/methods/finitedifferences/solvers/fdmbackwardsolver.{hpp,cpp}
# (v1.42.1).

The C++ implementation is a thin dispatcher over the
``FiniteDifferenceModel<Evolver>`` template: for each scheme tag in
``FdmSchemeDesc`` it instantiates the matching evolver and calls
``rollback(rhs, from, to, steps, *condition)``.

The Python port inlines the rollback logic since only four schemes
are needed in L5-D:

* ``CrankNicolsonType`` / ``DouglasType`` — equivalent in 1-D,
  routed via ``CrankNicolsonScheme(theta=0.5)``.
* ``ImplicitEulerType`` — ``CrankNicolsonScheme(theta=1.0)`` is
  equivalent to one implicit-Euler step (the implicit branch is
  exercised, the explicit branch contributes nothing).
* ``ExplicitEulerType`` — ``CrankNicolsonScheme(theta=0.0)``.

The damping-steps branch (``dampingSteps != 0`` and scheme !=
``ImplicitEulerType``) runs ``dampingSteps`` implicit-Euler steps
at the start to smooth the kink in the payoff before switching to
the requested scheme.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.constants import QL_EPSILON
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import (
    FdmBlackScholesOp,
)
from pquantlib.methods.finitedifferences.schemes.crank_nicolson_scheme import (
    CrankNicolsonScheme,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import (
    FdmSchemeDesc,
    FdmSchemeType,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)


def _rollback_with_scheme(
    scheme: CrankNicolsonScheme,
    a: Array,
    from_t: float,
    to_t: float,
    steps: int,
    condition: FdmStepConditionComposite,
) -> Array:
    """Generic backward rollback for a Crank-Nicolson-shaped evolver.

    # C++ parity: ``FiniteDifferenceModel::rollback``.
    """
    qassert.require(from_t >= to_t, f"trying to roll back from {from_t} to {to_t}")
    dt = (from_t - to_t) / steps
    t = from_t
    scheme.set_step(dt)
    stopping_times = condition.stopping_times()
    # Match C++: if there's a stopping time exactly at ``from``, apply.
    if stopping_times and stopping_times[-1] == from_t:
        condition.apply_to(a, from_t)
    for i in range(steps):
        now = t
        next_t = (t - dt) if (i < steps - 1) else to_t
        if abs(to_t - next_t) < math.sqrt(QL_EPSILON):
            next_t = to_t
        hit = False
        for j in range(len(stopping_times) - 1, -1, -1):
            stj = stopping_times[j]
            if next_t <= stj < now:
                hit = True
                scheme.set_step(now - stj)
                a = scheme.step(a, now)
                condition.apply_to(a, stj)
                now = stj
        if hit:
            if now > next_t:
                scheme.set_step(now - next_t)
                a = scheme.step(a, now)
                condition.apply_to(a, next_t)
            scheme.set_step(dt)
        else:
            a = scheme.step(a, now)
            condition.apply_to(a, next_t)
        t -= dt
    return a


@final
class FdmBackwardSolver:
    """Backward FD solver — rolls payoff from maturity back to t=0.

    # C++ parity: ``class FdmBackwardSolver``.
    """

    def __init__(
        self,
        op: FdmBlackScholesOp,
        condition: FdmStepConditionComposite | None,
        scheme_desc: FdmSchemeDesc,
    ) -> None:
        self._op: FdmBlackScholesOp = op
        self._condition: FdmStepConditionComposite = (
            condition if condition is not None else FdmStepConditionComposite([], [])
        )
        self._scheme_desc: FdmSchemeDesc = scheme_desc

    def rollback(
        self,
        rhs: Array,
        from_t: float,
        to_t: float,
        steps: int,
        damping_steps: int,
    ) -> Array:
        """Back-propagate ``rhs`` from ``from_t`` to ``to_t`` in ``steps``+``damping_steps``.

        # C++ parity: ``FdmBackwardSolver::rollback``.
        """
        delta_t = from_t - to_t
        all_steps = steps + damping_steps
        damping_to = from_t - (delta_t * damping_steps) / all_steps

        # Damping branch: implicit-Euler smoothing at the start (only
        # if the requested scheme is not already implicit-Euler).
        if damping_steps > 0 and self._scheme_desc.type != FdmSchemeType.ImplicitEulerType:
            implicit = CrankNicolsonScheme(theta=1.0, op=self._op)
            rhs = _rollback_with_scheme(implicit, rhs, from_t, damping_to, damping_steps, self._condition)

        # Main scheme.
        if self._scheme_desc.type in (
            FdmSchemeType.CrankNicolsonType,
            FdmSchemeType.DouglasType,
        ):
            scheme = CrankNicolsonScheme(theta=self._scheme_desc.theta, op=self._op)
            rhs = _rollback_with_scheme(scheme, rhs, damping_to, to_t, steps, self._condition)
        elif self._scheme_desc.type == FdmSchemeType.ImplicitEulerType:
            scheme = CrankNicolsonScheme(theta=1.0, op=self._op)
            rhs = _rollback_with_scheme(scheme, rhs, from_t, to_t, all_steps, self._condition)
        elif self._scheme_desc.type == FdmSchemeType.ExplicitEulerType:
            scheme = CrankNicolsonScheme(theta=0.0, op=self._op)
            rhs = _rollback_with_scheme(scheme, rhs, damping_to, to_t, steps, self._condition)
        else:
            raise NotImplementedError(
                f"FdmBackwardSolver: scheme {self._scheme_desc.type.name} not yet implemented "
                "(deferred to Phase 6 — multi-asset operator splittings)"
            )
        return rhs


__all__ = ["FdmBackwardSolver"]
