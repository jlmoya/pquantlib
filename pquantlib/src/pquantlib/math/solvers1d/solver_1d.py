"""Abstract base class for 1-D root solvers.

# C++ parity: ql/math/solver1d.hpp (v1.42.1) class ``Solver1D<Impl>``.

The C++ implementation uses the Barton-Nackman / curiously-recurring template
pattern (CRTP) to dispatch the per-algorithm step ``solveImpl`` without
runtime polymorphism. The Python port replaces CRTP with a normal abstract
method ``_solve_impl(f, x_accuracy)`` — Python doesn't gain anything from
template-style devirtualization here, and direct inheritance gives clearer
introspection.

Two ``solve`` entry points exist (matching the two C++ overloads):

* ``solve_unbracketed(f, accuracy, guess, step)`` — auto-brackets by
  expanding around ``guess`` with growth factor 1.6 (cf. Press et al.,
  Numerical Recipes 2e).
* ``solve_bracketed(f, accuracy, guess, x_min, x_max)`` — caller provides
  a valid bracket.

The base class sets the protected attributes (``_root``, ``_x_min``,
``_x_max``, ``_fx_min``, ``_fx_max``, ``_evaluation_number``) before
delegating to ``_solve_impl``. Subclasses may safely assume the bracket
is valid and ``_root`` holds a starting guess.

Subclasses that require derivatives (Newton, NewtonSafe, Halley) call
``f.derivative(x)`` (and ``f.second_derivative(x)`` for Halley) directly.
The Python translation expects the integrand to expose those as methods;
no ``Null<Real>``-style sentinel is needed because Python raises
``AttributeError`` naturally if the method is missing — which the
subclass catches and converts to a ``LibraryException`` if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.constants import QL_EPSILON

# C++ parity: solver1d.hpp #define MAX_FUNCTION_EVALUATIONS 100.
_MAX_FUNCTION_EVALUATIONS: int = 100


class RealFunction(Protocol):
    """Structural type for an integrand: ``f(x) -> float``."""

    def __call__(self, x: float, /) -> float: ...


class Solver1D(ABC):
    """Abstract 1-D solver. Concrete solvers override :meth:`_solve_impl`."""

    def __init__(self) -> None:
        # Mutable bracket / evaluation state set up by ``solve_*`` before
        # dispatching to ``_solve_impl``. Subclasses read and may mutate
        # these (matching the C++ ``mutable`` protected data members).
        self._root: float = 0.0
        self._x_min: float = 0.0
        self._x_max: float = 0.0
        self._fx_min: float = 0.0
        self._fx_max: float = 0.0
        self._evaluation_number: int = 0
        self._max_evaluations: int = _MAX_FUNCTION_EVALUATIONS
        self._lower_bound: float = 0.0
        self._upper_bound: float = 0.0
        self._lower_bound_enforced: bool = False
        self._upper_bound_enforced: bool = False

    # --- public API ----------------------------------------------------

    def set_max_evaluations(self, evaluations: int) -> None:
        """Set the maximum number of function evaluations."""
        self._max_evaluations = evaluations

    def set_lower_bound(self, lower_bound: float) -> None:
        """Enforce a lower bound on the function domain."""
        self._lower_bound = lower_bound
        self._lower_bound_enforced = True

    def set_upper_bound(self, upper_bound: float) -> None:
        """Enforce an upper bound on the function domain."""
        self._upper_bound = upper_bound
        self._upper_bound_enforced = True

    def solve(
        self,
        f: RealFunction,
        accuracy: float,
        guess: float,
        step_or_x_min: float,
        x_max: float | None = None,
    ) -> float:
        """Dispatch to bracketed or unbracketed solve.

        If ``x_max`` is omitted the call is interpreted as the unbracketed
        4-arg form ``solve(f, accuracy, guess, step)``; otherwise the
        5-arg bracketed form ``solve(f, accuracy, guess, x_min, x_max)``
        is invoked.
        """
        if x_max is None:
            return self._solve_unbracketed(f, accuracy, guess, step_or_x_min)
        return self._solve_bracketed(f, accuracy, guess, step_or_x_min, x_max)

    # --- abstract hook --------------------------------------------------

    @abstractmethod
    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        """Per-algorithm step. Bracket + initial guess are set by the base."""

    # --- internal helpers ----------------------------------------------

    def _enforce_bounds(self, x: float) -> float:
        if self._lower_bound_enforced and x < self._lower_bound:
            return self._lower_bound
        if self._upper_bound_enforced and x > self._upper_bound:
            return self._upper_bound
        return x

    def _solve_unbracketed(
        self,
        f: RealFunction,
        accuracy: float,
        guess: float,
        step: float,
    ) -> float:
        qassert.require(accuracy > 0.0, f"accuracy ({accuracy}) must be positive")
        # Floor the accuracy to QL_EPSILON to match C++ behavior.
        accuracy = max(accuracy, QL_EPSILON)

        growth_factor = 1.6
        flipflop = -1

        self._root = guess
        self._fx_max = f(self._root)

        # monotonically increasing bias, as in optionValue(volatility)
        if close(self._fx_max, 0.0):
            return self._root
        if self._fx_max > 0.0:
            self._x_min = self._enforce_bounds(self._root - step)
            self._fx_min = f(self._x_min)
            self._x_max = self._root
        else:
            self._x_min = self._root
            self._fx_min = self._fx_max
            self._x_max = self._enforce_bounds(self._root + step)
            self._fx_max = f(self._x_max)

        self._evaluation_number = 2
        while self._evaluation_number <= self._max_evaluations:
            if self._fx_min * self._fx_max <= 0.0:
                if close(self._fx_min, 0.0):
                    return self._x_min
                if close(self._fx_max, 0.0):
                    return self._x_max
                self._root = (self._x_max + self._x_min) / 2.0
                return self._solve_impl(f, accuracy)
            if abs(self._fx_min) < abs(self._fx_max):
                self._x_min = self._enforce_bounds(self._x_min + growth_factor * (self._x_min - self._x_max))
                self._fx_min = f(self._x_min)
            elif abs(self._fx_min) > abs(self._fx_max):
                self._x_max = self._enforce_bounds(self._x_max + growth_factor * (self._x_max - self._x_min))
                self._fx_max = f(self._x_max)
            elif flipflop == -1:
                self._x_min = self._enforce_bounds(self._x_min + growth_factor * (self._x_min - self._x_max))
                self._fx_min = f(self._x_min)
                self._evaluation_number += 1
                flipflop = 1
            elif flipflop == 1:
                self._x_max = self._enforce_bounds(self._x_max + growth_factor * (self._x_max - self._x_min))
                self._fx_max = f(self._x_max)
                flipflop = -1
            self._evaluation_number += 1

        qassert.fail(
            f"unable to bracket root in {self._max_evaluations} function evaluations "
            f"(last bracket attempt: f[{self._x_min},{self._x_max}] -> "
            f"[{self._fx_min},{self._fx_max}])"
        )

    def _solve_bracketed(
        self,
        f: RealFunction,
        accuracy: float,
        guess: float,
        x_min: float,
        x_max: float,
    ) -> float:
        qassert.require(accuracy > 0.0, f"accuracy ({accuracy}) must be positive")
        accuracy = max(accuracy, QL_EPSILON)

        self._x_min = x_min
        self._x_max = x_max

        qassert.require(
            self._x_min < self._x_max,
            f"invalid range: xMin_ ({self._x_min}) >= xMax_ ({self._x_max})",
        )
        qassert.require(
            not self._lower_bound_enforced or self._x_min >= self._lower_bound,
            f"xMin_ ({self._x_min}) < enforced low bound ({self._lower_bound})",
        )
        qassert.require(
            not self._upper_bound_enforced or self._x_max <= self._upper_bound,
            f"xMax_ ({self._x_max}) > enforced hi bound ({self._upper_bound})",
        )

        self._fx_min = f(self._x_min)
        if close(self._fx_min, 0.0):
            return self._x_min

        self._fx_max = f(self._x_max)
        if close(self._fx_max, 0.0):
            return self._x_max

        self._evaluation_number = 2

        qassert.require(
            self._fx_min * self._fx_max < 0.0,
            f"root not bracketed: f[{self._x_min},{self._x_max}] -> [{self._fx_min},{self._fx_max}]",
        )

        qassert.require(guess > self._x_min, f"guess ({guess}) < xMin_ ({self._x_min})")
        qassert.require(guess < self._x_max, f"guess ({guess}) > xMax_ ({self._x_max})")

        self._root = guess

        return self._solve_impl(f, accuracy)
