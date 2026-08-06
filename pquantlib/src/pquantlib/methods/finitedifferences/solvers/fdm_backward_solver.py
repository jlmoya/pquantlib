"""FdmBackwardSolver — back-propagate a maturity payoff to t=0.

# C++ parity: ql/methods/finitedifferences/solvers/fdmbackwardsolver.{hpp,cpp}
# (v1.43).

C++ is a dispatcher over ``FiniteDifferenceModel<Evolver>``: it picks the
evolver named by the ``FdmSchemeDesc`` tag and rolls the array back. The
``FiniteDifferenceModel`` template itself belongs to the retired pre-1.0
framework (hosted in ``pquantlib-helpers``), which core must not depend on, so
its ``rollback`` loop — including the stopping-time bisection — is inlined
here as ``_rollback_with_scheme``.

All nine scheme tags are dispatched. Earlier this file handled only the three
Euler/Crank-Nicolson tags and raised ``NotImplementedError`` for the rest.

Damping: when ``damping_steps != 0`` and the requested scheme is not already
implicit Euler, C++ runs ``damping_steps`` implicit-Euler steps first to smooth
the payoff kink, then switches. Reproduced, including the fact that the
implicit-Euler branch instead rolls the *whole* ``steps + damping_steps`` in
one go from ``from`` rather than from ``damping_to``.
"""

from __future__ import annotations

import math
from typing import Protocol, final

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.constants import QL_EPSILON
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    FdmBoundaryConditionSet,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.methods.finitedifferences.schemes.craig_sneyd_scheme import (
    CraigSneydScheme,
)
from pquantlib.methods.finitedifferences.schemes.crank_nicolson_scheme import (
    CrankNicolsonScheme,
)
from pquantlib.methods.finitedifferences.schemes.douglas_scheme import DouglasScheme
from pquantlib.methods.finitedifferences.schemes.explicit_euler_scheme import (
    ExplicitEulerScheme,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import (
    FdmSchemeDesc,
    FdmSchemeType,
)
from pquantlib.methods.finitedifferences.schemes.hundsdorfer_scheme import (
    HundsdorferScheme,
)
from pquantlib.methods.finitedifferences.schemes.implicit_euler_scheme import (
    ImplicitEulerScheme,
)
from pquantlib.methods.finitedifferences.schemes.method_of_lines_scheme import (
    MethodOfLinesScheme,
)
from pquantlib.methods.finitedifferences.schemes.modified_craig_sneyd_scheme import (
    ModifiedCraigSneydScheme,
)
from pquantlib.methods.finitedifferences.schemes.tr_bdf2_scheme import TrBDF2Scheme
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)


class _Evolver(Protocol):
    """The one-step surface every FD scheme in this package exposes.

    Both arguments are positional-only: the schemes spell the array parameter
    differently (``a`` in the Euler family, ``fn`` in ``TrBDF2Scheme``, as C++
    does) and the protocol must not force them to agree on the name.
    """

    def set_step(self, dt: float, /) -> None: ...

    def step(self, a: Array, t: float, /) -> Array: ...


def _rollback_with_scheme(
    scheme: _Evolver,
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
        op: FdmLinearOpComposite,
        condition: FdmStepConditionComposite | None = None,
        scheme_desc: FdmSchemeDesc | None = None,
        bc_set: FdmBoundaryConditionSet = (),
    ) -> None:
        """# C++ parity: ``FdmBackwardSolver(map, bcSet, condition, schemeDesc)``.

        The argument order differs from C++: ``bc_set`` is last and defaults to
        empty, so the pre-existing three-argument call sites keep working.
        """
        self._op: FdmLinearOpComposite = op
        self._condition: FdmStepConditionComposite = (
            condition if condition is not None else FdmStepConditionComposite([], [])
        )
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )
        self._bc_set: FdmBoundaryConditionSet = bc_set

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

        # Damping branch: implicit-Euler smoothing at the start (only if the
        # requested scheme is not already implicit Euler).
        if damping_steps > 0 and self._scheme_desc.type != FdmSchemeType.ImplicitEulerType:
            damper = ImplicitEulerScheme(self._op, self._bc_set)
            rhs = _rollback_with_scheme(
                damper, rhs, from_t, damping_to, damping_steps, self._condition
            )

        theta = self._scheme_desc.theta
        mu = self._scheme_desc.mu
        kind = self._scheme_desc.type
        evolver: _Evolver

        # C++ parity: the switch in FdmBackwardSolver::rollback. Note the
        # ImplicitEulerType arm rolls all_steps from `from`, not steps from
        # damping_to — every other arm rolls steps from damping_to.
        if kind == FdmSchemeType.ImplicitEulerType:
            evolver = ImplicitEulerScheme(self._op, self._bc_set)
            return _rollback_with_scheme(
                evolver, rhs, from_t, to_t, all_steps, self._condition
            )

        if kind == FdmSchemeType.HundsdorferType:
            evolver = HundsdorferScheme(theta, mu, self._op, self._bc_set)
        elif kind == FdmSchemeType.DouglasType:
            evolver = DouglasScheme(theta, self._op, self._bc_set)
        elif kind == FdmSchemeType.CrankNicolsonType:
            evolver = CrankNicolsonScheme(theta, self._op, self._bc_set)
        elif kind == FdmSchemeType.CraigSneydType:
            evolver = CraigSneydScheme(theta, mu, self._op, self._bc_set)
        elif kind == FdmSchemeType.ModifiedCraigSneydType:
            evolver = ModifiedCraigSneydScheme(theta, mu, self._op, self._bc_set)
        elif kind == FdmSchemeType.ExplicitEulerType:
            evolver = ExplicitEulerScheme(self._op, self._bc_set)
        elif kind == FdmSchemeType.MethodOfLinesType:
            evolver = MethodOfLinesScheme(theta, mu, self._op, self._bc_set)
        elif kind == FdmSchemeType.TrBDF2Type:
            # C++ hard-codes the trapezoidal stage to CraigSneyd()'s (theta, mu).
            tr_desc = FdmSchemeDesc.craig_sneyd()
            trapezoidal = CraigSneydScheme(tr_desc.theta, tr_desc.mu, self._op, self._bc_set)
            evolver = TrBDF2Scheme(theta, self._op, trapezoidal, self._bc_set, mu)
        else:
            raise LibraryException("Unknown scheme type")

        rhs = _rollback_with_scheme(evolver, rhs, damping_to, to_t, steps, self._condition)
        return rhs


__all__ = ["FdmBackwardSolver"]
