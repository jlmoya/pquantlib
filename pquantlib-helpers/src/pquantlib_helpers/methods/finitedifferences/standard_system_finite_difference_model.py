"""StandardSystemFiniteDifferenceModel — system (vector-of-arrays) FD rollback.

# Retired-API compat layer — see package docstring.

Java parity: ``org.jquantlib.methods.finitedifferences.StandardSystemFiniteDifferenceModel``
(+ its inner ``ParallelEvolver``).
C++ parity: ``ql/methods/finitedifferences/finitedifferencemodel.hpp`` driving a
``ParallelEvolver`` (``ql/methods/finitedifferences/parallelevolver.hpp``,
v1.42.1, ``[[deprecated]]``).

This is the 2-component variant of
:class:`~pquantlib_helpers.methods.finitedifferences.finite_difference_model.FiniteDifferenceModel`
that the control-variate
:class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_step_condition_engine.FDStepConditionEngine`
needs: it evolves a *list* of arrays in lock-step, one Crank-Nicolson evolver
per (operator, boundary-condition-list) pair, under a
:class:`~pquantlib_helpers.methods.finitedifferences.boundary_condition.BoundaryConditionSet`
and an optional
:class:`~pquantlib_helpers.methods.finitedifferences.step_condition.StepConditionSet`.

The rollback loop is the Java/old-QuantLib system rollback — it has NO C++
last-step ``next = (i < steps-1) ? t-dt : to`` snap (that refinement lives only
in the scalar ``FiniteDifferenceModel``); the system model always uses
``next = t - dt``. This is faithful to the Java ``rollbackImpl`` the FD American
engine was cross-validated against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pquantlib.exceptions import LibraryException
from pquantlib_helpers.methods.finitedifferences.crank_nicolson import CrankNicolson

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib_helpers.methods.finitedifferences.boundary_condition import (
        BoundaryConditionSet,
    )
    from pquantlib_helpers.methods.finitedifferences.mixed_scheme import (
        BoundaryCondition,
    )
    from pquantlib_helpers.methods.finitedifferences.step_condition import (
        StepConditionSet,
    )
    from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
        TridiagonalOperator,
    )


class _ParallelEvolver:
    """One Crank-Nicolson evolver per system component.

    Java parity: ``ParallelEvolver<TridiagonalOperator, CrankNicolson<...>>``
    (specialised to the FD system path). The reflective ``classS`` / ``classT``
    type-token indirection the Java class carries to dynamically instantiate the
    scheme is unnecessary in Python — we build :class:`CrankNicolson` evolvers
    directly.
    """

    def __init__(
        self,
        operators: list[TridiagonalOperator],
        bcs: BoundaryConditionSet,
    ) -> None:
        """Build a Crank-Nicolson evolver for each ``(operator, bc-list)`` pair."""
        # ``BoundaryConditionSet.get`` is typed over the structurally-identical
        # ``BoundaryConditionLike`` protocol; ``CrankNicolson`` expects the
        # ``BoundaryCondition`` protocol from ``mixed_scheme`` (same hook surface).
        self._evolvers: list[CrankNicolson] = [
            CrankNicolson(operators[i], cast("list[BoundaryCondition]", bcs.get(i)))
            for i in range(len(operators))
        ]

    def set_step(self, dt: float) -> None:
        """Cache the step on every component evolver."""
        for evolver in self._evolvers:
            evolver.set_step(dt)

    def step(self, a: list[Array], t: float) -> list[Array]:
        """Advance every component array by one step ending at time ``t``."""
        for i, evolver in enumerate(self._evolvers):
            a[i] = evolver.step(a[i], t)
        return a


class StandardSystemFiniteDifferenceModel:
    """System finite-difference model rolling a list of arrays back in lock-step.

    Java parity: ``StandardSystemFiniteDifferenceModel``.
    """

    def __init__(
        self,
        operators: list[TridiagonalOperator],
        bcs: BoundaryConditionSet,
        stopping_times: list[float] | None = None,
    ) -> None:
        """Build the model from per-component operators + boundary-condition set.

        ``stopping_times`` are de-duplicated and sorted (Java ``HashSet`` +
        ``Collections.sort``).
        """
        self._evolver = _ParallelEvolver(operators, bcs)
        times = sorted(set(stopping_times)) if stopping_times else []
        self._stopping_times: list[float] = times

    def evolver(self) -> _ParallelEvolver:
        """The wrapped parallel evolver."""
        return self._evolver

    def rollback(
        self,
        a: list[Array],
        from_time: float,
        to_time: float,
        steps: int,
        condition: StepConditionSet | None = None,
    ) -> list[Array]:
        """Roll the system ``a`` back from ``from_time`` to ``to_time``.

        Java parity: ``StandardSystemFiniteDifferenceModel.rollbackImpl``.
        ``from_time`` must be a later time than ``to_time``.
        """
        if from_time <= to_time:
            raise LibraryException(
                f"trying to roll back from {from_time} to {to_time}"
            )

        dt = (from_time - to_time) / steps
        t = from_time
        self._evolver.set_step(dt)

        for _ in range(steps):
            now = t
            next_t = t - dt
            hit = False
            for j in range(len(self._stopping_times) - 1, -1, -1):
                stop = self._stopping_times[j]
                if next_t <= stop < now:
                    hit = True
                    self._evolver.set_step(now - stop)
                    a = self._evolver.step(a, now)
                    if condition is not None:
                        condition.apply_to(a, stop)
                    now = stop

            if hit:
                if now > next_t:
                    self._evolver.set_step(now - next_t)
                    a = self._evolver.step(a, now)
                    if condition is not None:
                        condition.apply_to(a, next_t)
                self._evolver.set_step(dt)
            else:
                a = self._evolver.step(a, now)
                if condition is not None:
                    condition.apply_to(a, next_t)

            t -= dt
        return a


__all__ = ["StandardSystemFiniteDifferenceModel"]
