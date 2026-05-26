"""Abstract integrator base class.

# C++ parity: ql/math/integrals/integral.hpp + ql/math/integrals/integral.cpp
# (v1.42.1) class ``Integrator``.

Concrete integrators (Simpson, Trapezoid, etc.) override ``_integrate(f, a, b)``
to implement the per-rule quadrature. The base class's :meth:`__call__`
handles the orientation of integration limits (``a == b``, ``a > b``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON


class RealFunction(Protocol):
    """Structural type for an integrand: ``f(x) -> float``."""

    def __call__(self, x: float, /) -> float: ...


class Integrator(ABC):
    """Abstract 1-D integrator. Subclasses override :meth:`_integrate`."""

    def __init__(self, absolute_accuracy: float, max_evaluations: int) -> None:
        qassert.require(
            absolute_accuracy > QL_EPSILON,
            f"required tolerance ({absolute_accuracy:e}) not allowed. It must be > {QL_EPSILON:e}",
        )
        self._absolute_accuracy: float = absolute_accuracy
        self._max_evaluations: int = max_evaluations
        self._absolute_error: float = 0.0
        self._evaluations: int = 0

    # --- modifiers -----------------------------------------------------

    def set_absolute_accuracy(self, accuracy: float) -> None:
        self._absolute_accuracy = accuracy

    def set_max_evaluations(self, max_evaluations: int) -> None:
        self._max_evaluations = max_evaluations

    # --- inspectors ----------------------------------------------------

    def absolute_accuracy(self) -> float:
        return self._absolute_accuracy

    def max_evaluations(self) -> int:
        return self._max_evaluations

    def absolute_error(self) -> float:
        return self._absolute_error

    def number_of_evaluations(self) -> int:
        return self._evaluations

    def integration_success(self) -> bool:
        return self._evaluations <= self._max_evaluations and self._absolute_error <= self._absolute_accuracy

    # --- callable ------------------------------------------------------

    def __call__(self, f: RealFunction, a: float, b: float) -> float:
        self._evaluations = 0
        if a == b:
            return 0.0
        if b > a:
            return self._integrate(f, a, b)
        return -self._integrate(f, b, a)

    # --- subclass hook -------------------------------------------------

    @abstractmethod
    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        """Subclass-specific quadrature rule."""

    # --- protected helpers (mutable accounting) ------------------------

    def _set_absolute_error(self, error: float) -> None:
        self._absolute_error = error

    def _set_number_of_evaluations(self, evaluations: int) -> None:
        self._evaluations = evaluations

    def _increase_number_of_evaluations(self, increase: int) -> None:
        self._evaluations += increase
