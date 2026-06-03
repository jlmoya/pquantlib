"""Step conditions for the legacy finite-difference framework.

# Retired-API compat layer — see package docstring.

C++ parity: ``ql/methods/finitedifferences/stepcondition.hpp`` (the abstract
``StepCondition<array_type>`` with ``applyTo(a, t)``) plus
``ql/methods/finitedifferences/americancondition.hpp`` and
``ql/methods/finitedifferences/nullcondition.hpp``.

Java parity: ``org.jquantlib.methods.finitedifferences`` —
``StepCondition`` (interface), ``NullCondition``,
``CurveDependentStepCondition`` (abstract), ``AmericanCondition``,
``StepConditionSet``.

FD-alpha1 already declared a structural :class:`StepCondition` *Protocol*
(``apply_to(a, t)``); the concrete classes here implement that surface and are
accepted anywhere the Protocol is expected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pquantlib.payoffs import OptionType, PlainVanillaPayoff

if TYPE_CHECKING:
    from collections.abc import Callable

    from pquantlib.math.array import Array


class NullCondition:
    """Step condition that does nothing.

    C++ parity: ``NullCondition<array_type>``. Java parity: ``NullCondition``.
    """

    def apply_to(self, a: Array, t: float) -> None:
        """No-op. C++ parity: empty body."""


class CurveDependentStepCondition(ABC):
    """Abstract step condition driven by a per-node payoff/curve value.

    C++ parity: ``CurveDependentStepCondition`` (no modern Fdm* counterpart —
    the modern framework has a different layout; this remains the old-framework
    base for :class:`AmericanCondition`). Java parity:
    ``CurveDependentStepCondition``.

    Subclasses implement :meth:`apply_to_value` (transform a node's current
    value given its intrinsic/curve value). The curve value comes from either a
    payoff (evaluated at the node's array value) or a fixed array of values.
    """

    def __init__(self, curve_value: Callable[[Array, int], float]) -> None:
        """Bind the per-node curve-value accessor."""
        self._curve_value = curve_value

    @classmethod
    def from_payoff(cls, payoff: Callable[[float], float]) -> Callable[[Array, int], float]:
        """Build a curve-value accessor that evaluates ``payoff`` at ``a[i]``.

        C++ parity: ``CurveDependentStepCondition::PayoffWrapper``.
        """
        return lambda a, i: payoff(float(a[i]))

    @classmethod
    def from_type_strike(
        cls, option_type: OptionType, strike: float
    ) -> Callable[[Array, int], float]:
        """Build a curve-value accessor from a plain-vanilla payoff.

        C++ parity: ``PayoffWrapper(Option::Type, Real strike)``.
        """
        payoff = PlainVanillaPayoff(option_type, strike)
        return lambda a, i: payoff(float(a[i]))

    @classmethod
    def from_array(cls, values: Array) -> Callable[[Array, int], float]:
        """Build a curve-value accessor that reads a fixed ``values`` array.

        C++ parity: ``CurveDependentStepCondition::ArrayWrapper``.
        """
        snapshot = np.array(values, dtype=np.float64)
        return lambda a, i: float(snapshot[i])

    @abstractmethod
    def apply_to_value(self, current: float, intrinsic: float) -> float:
        """Transform a node's ``current`` value given its ``intrinsic`` value."""

    def get_value(self, a: Array, index: int) -> float:
        """Return the curve value for node ``index``."""
        return self._curve_value(a, index)

    def apply_to(self, a: Array, t: float) -> None:
        """Apply the scalar rule node-by-node in place.

        C++ parity: ``CurveDependentStepCondition::applyTo``.
        """
        for i in range(a.shape[0]):
            a[i] = self.apply_to_value(float(a[i]), self.get_value(a, i))


class AmericanCondition(CurveDependentStepCondition):
    """Early-exercise step condition: ``max(value, intrinsic)`` per node.

    C++ parity: ``AmericanCondition``. Java parity: ``AmericanCondition``.
    """

    def __init__(
        self,
        *,
        option_type: OptionType | None = None,
        strike: float | None = None,
        values: Array | None = None,
    ) -> None:
        """Construct from a ``(type, strike)`` payoff or a fixed ``values`` array.

        C++ parity: ``AmericanCondition(Option::Type, Real)`` and
        ``AmericanCondition(const Array&)`` (the two Java constructors).
        """
        if values is not None:
            super().__init__(self.from_array(values))
        elif option_type is not None and strike is not None:
            super().__init__(self.from_type_strike(option_type, strike))
        else:
            raise ValueError(
                "AmericanCondition requires either values=... or "
                "(option_type=..., strike=...)"
            )

    def apply_to_value(self, current: float, intrinsic: float) -> float:
        """Return ``max(current, intrinsic)``."""
        return max(current, intrinsic)


class StepConditionSet:
    """Ordered set of step conditions, one per system component.

    C++ parity: ``StepConditionSet`` (used by the system FD model). Java
    parity: ``StepConditionSet<T>``. Only consumed by the system/parallel FD
    path (deferred to FD-beta); kept here for completeness.
    """

    def __init__(self) -> None:
        """Create an empty set."""
        self._step_conditions: list[StepConditionLike] = []

    def push_back(self, a: StepConditionLike) -> None:
        """Append a step condition (one system component)."""
        self._step_conditions.append(a)

    def apply_to(self, a: list[Array], t: float) -> None:
        """Apply each member condition to the matching component array."""
        for i, condition in enumerate(self._step_conditions):
            condition.apply_to(a[i], t)


if TYPE_CHECKING:
    from typing import Protocol

    class StepConditionLike(Protocol):
        """Structural type accepted by :class:`StepConditionSet`."""

        def apply_to(self, a: Array, t: float) -> None: ...
else:
    StepConditionLike = object


__all__ = [
    "AmericanCondition",
    "CurveDependentStepCondition",
    "NullCondition",
    "StepConditionSet",
]
