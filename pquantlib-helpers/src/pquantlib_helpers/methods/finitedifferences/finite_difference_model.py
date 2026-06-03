"""FiniteDifferenceModel — generic finite-difference rollback driver.

# Retired-API compat layer — see package docstring.

Java parity: ``org.jquantlib.methods.finitedifferences.FiniteDifferenceModel``
and ``StandardFiniteDifferenceModel``.
C++ parity: ``ql/methods/finitedifferences/finitedifferencemodel.hpp``
(v1.42.1). The rollback loop follows the **C++** algorithm (source of truth),
including the ``next = (i < steps-1) ? t-dt : to`` last-step refinement and the
leading ``stoppingTimes.back() == from`` condition check, which the older Java
port lacks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from pquantlib.exceptions import LibraryException
from pquantlib_helpers.methods.finitedifferences.crank_nicolson import CrankNicolson

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib_helpers.methods.finitedifferences.mixed_scheme import (
        BoundaryCondition,
        MixedScheme,
    )
    from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
        TridiagonalOperator,
    )

# sqrt(machine epsilon) — C++ std::sqrt(QL_EPSILON); used to snap the final
# step exactly onto `to`.
_SQRT_EPSILON = math.sqrt(2.220446049250313e-16)


class StepCondition(Protocol):
    """A condition applied to the grid array at every rollback step.

    C++ parity: ``StepCondition<array_type>`` (``applyTo(a, t)``).
    """

    def apply_to(self, a: Array, t: float) -> None:
        """Apply the condition to ``a`` at time ``t`` (in place)."""
        ...


class FiniteDifferenceModel:
    """Generic finite-difference model rolling a grid back over an evolver.

    Java parity: ``FiniteDifferenceModel<S extends Operator, T extends MixedScheme<S>>``.
    """

    def __init__(
        self,
        evolver: MixedScheme,
        stopping_times: list[float] | None = None,
    ) -> None:
        """Build a model around a pre-constructed ``evolver``.

        ``stopping_times`` are de-duplicated and sorted (C++ ``std::sort`` +
        ``std::unique``).
        """
        self._evolver: MixedScheme = evolver
        times = sorted(set(stopping_times)) if stopping_times else []
        self._stopping_times: list[float] = times

    def evolver(self) -> MixedScheme:
        """The wrapped evolver."""
        return self._evolver

    def stopping_times(self) -> list[float]:
        """The de-duplicated, sorted stopping times."""
        return self._stopping_times

    def rollback(
        self,
        a: Array,
        from_time: float,
        to_time: float,
        steps: int,
        condition: StepCondition | None = None,
    ) -> Array:
        """Roll ``a`` back from ``from_time`` to ``to_time`` in ``steps`` steps.

        C++ parity: ``FiniteDifferenceModel::rollback`` (both overloads — the
        optional ``condition`` argument unifies them). ``from_time`` must be a
        later time than ``to_time``. Returns the rolled-back array (the evolver
        rebinds ``a`` at each step).
        """
        return self._rollback_impl(a, from_time, to_time, steps, condition)

    def _rollback_impl(
        self,
        a: Array,
        from_time: float,
        to_time: float,
        steps: int,
        condition: StepCondition | None,
    ) -> Array:
        if from_time < to_time:
            raise LibraryException(
                f"trying to roll back from {from_time} to {to_time}"
            )

        dt = (from_time - to_time) / steps
        t = from_time
        self._evolver.set_step(dt)

        # C++ parity: if the latest stopping time equals `from`, apply the
        # condition once up front.
        if (
            self._stopping_times
            and self._stopping_times[-1] == from_time
            and condition is not None
        ):
            condition.apply_to(a, from_time)

        for i in range(steps):
            now = t
            # make the last step land exactly on `to` (numerical safety).
            next_t = (t - dt) if i < steps - 1 else to_time
            if math.fabs(to_time - next_t) < _SQRT_EPSILON:
                next_t = to_time

            hit = False
            for j in range(len(self._stopping_times) - 1, -1, -1):
                stop = self._stopping_times[j]
                if next_t <= stop < now:
                    # a stopping time was hit
                    hit = True
                    # perform a small step to stop...
                    self._evolver.set_step(now - stop)
                    a = self._evolver.step(a, now)
                    if condition is not None:
                        condition.apply_to(a, stop)
                    # ...and continue the cycle
                    now = stop

            if hit:
                # ...complete the big step if a fragment remains...
                if now > next_t:
                    self._evolver.set_step(now - next_t)
                    a = self._evolver.step(a, now)
                    if condition is not None:
                        condition.apply_to(a, next_t)
                # ...and reset the evolver to the default step.
                self._evolver.set_step(dt)
            else:
                # evolver already at the default step.
                a = self._evolver.step(a, now)
                if condition is not None:
                    condition.apply_to(a, next_t)

            t -= dt
        return a


class StandardFiniteDifferenceModel(FiniteDifferenceModel):
    """FiniteDifferenceModel wired to a Crank-Nicolson TridiagonalOperator evolver.

    Java parity: ``StandardFiniteDifferenceModel`` (extends
    ``FiniteDifferenceModel<TridiagonalOperator, CrankNicolson<TridiagonalOperator>>``).
    """

    def __init__(
        self,
        op: TridiagonalOperator,
        bcs: list[BoundaryCondition] | None = None,
        stopping_times: list[float] | None = None,
    ) -> None:
        """Build a Crank-Nicolson model directly from operator ``op``."""
        super().__init__(CrankNicolson(op, bcs), stopping_times)


__all__ = [
    "FiniteDifferenceModel",
    "StandardFiniteDifferenceModel",
    "StepCondition",
]
