"""Trapezoid composite quadrature.

# C++ parity: ql/math/integrals/trapezoidintegral.hpp (v1.43) class
# ``TrapezoidIntegral<IntegrationPolicy>`` plus the two policy tag structs
# ``Default`` and ``MidPoint``.

The C++ template is parameterized by an integration policy selecting the
refinement stencil. The Python port keeps both policies as classes with
``integrate`` / ``nb_evaluations`` static methods and takes the policy as a
constructor argument defaulting to :class:`Default` (the C++ default in
practice — ``SimpsonIntegral`` and every QuantLib caller instantiate
``TrapezoidIntegral<Default>``).

* :class:`Default` doubles ``N`` per refinement and samples the midpoints of
  the existing panels: ``I <- (I + dx * sum) / 2``.
* :class:`MidPoint` triples ``N`` and samples at ``(k + 1/6) dx`` and
  ``(k + 5/6) dx``: ``I <- (I + dx * sum) / 3``.

Note that ``MidPoint`` is seeded with the *trapezoid* value
``(f(a)+f(b))(b-a)/2`` and then driven by the midpoint trisection recursion,
so the seed error decays only by a factor 3 per iteration while ``N`` grows by
3 — a tight ``absolute_accuracy`` costs O(3^k) evaluations. That is C++
behaviour, reproduced here.
"""

from __future__ import annotations

from typing import Protocol

from pquantlib import qassert
from pquantlib.math.integrals.integrator import Integrator, RealFunction


class IntegrationPolicy(Protocol):
    """Structural interface of the C++ ``IntegrationPolicy`` template parameter."""

    @staticmethod
    def integrate(f: RealFunction, a: float, b: float, i: float, n: int) -> float: ...

    @staticmethod
    def nb_evaluations() -> int: ...


class Default:
    """Trapezoid refinement policy — bisection.

    # C++ parity: ``struct Default`` (trapezoidintegral.hpp:86-101).
    """

    __slots__ = ()

    @staticmethod
    def integrate(f: RealFunction, a: float, b: float, i: float, n: int) -> float:
        # C++ parity: trapezoidintegral.hpp:87-99.
        total = 0.0
        dx = (b - a) / n
        x = a + dx / 2.0
        for _ in range(n):
            total += f(x)
            x += dx
        return (i + dx * total) / 2.0

    @staticmethod
    def nb_evaluations() -> int:
        # C++ parity: trapezoidintegral.hpp:100.
        return 2


class MidPoint:
    """Trapezoid refinement policy — trisection at the 1/6 and 5/6 points.

    # C++ parity: ``struct MidPoint`` (trapezoidintegral.hpp:103-119).
    """

    __slots__ = ()

    @staticmethod
    def integrate(f: RealFunction, a: float, b: float, i: float, n: int) -> float:
        # C++ parity: trapezoidintegral.hpp:104-117.
        total = 0.0
        dx = (b - a) / n
        x = a + dx / 6.0
        d = 2.0 * dx / 3.0
        for _ in range(n):
            total += f(x) + f(x + d)
            x += dx
        return (i + dx * total) / 3.0

    @staticmethod
    def nb_evaluations() -> int:
        # C++ parity: trapezoidintegral.hpp:118.
        return 3


def default_refine(f: RealFunction, a: float, b: float, prev: float, n: int) -> float:
    """One :class:`Default` refinement step.

    Kept as a free function because ``SimpsonIntegral`` drives the same
    trapezoid sequence; it is exactly ``Default.integrate``.
    """
    return Default.integrate(f, a, b, prev, n)


class TrapezoidIntegral(Integrator):
    """Composite trapezoid rule, refined until ``absolute_accuracy``.

    # C++ parity: ``TrapezoidIntegral<IntegrationPolicy>``
    # (trapezoidintegral.hpp:53-83).
    """

    __slots__ = ("_policy",)

    def __init__(
        self,
        absolute_accuracy: float,
        max_evaluations: int,
        policy: type[IntegrationPolicy] = Default,
    ) -> None:
        super().__init__(absolute_accuracy, max_evaluations)
        self._policy: type[IntegrationPolicy] = policy

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: trapezoidintegral.hpp:60-82 — a do/while, so the first
        # refinement always happens even when max_evaluations <= 1.
        policy = self._policy
        # start from the coarsest trapezoid...
        n = 1
        prev = (f(a) + f(b)) * (b - a) / 2.0
        self._increase_number_of_evaluations(2)
        # ...and refine it
        i = 1
        while True:
            new = policy.integrate(f, a, b, prev, n)
            self._increase_number_of_evaluations(n * (policy.nb_evaluations() - 1))
            n *= policy.nb_evaluations()
            # good enough? Also, don't run away immediately
            if abs(prev - new) <= self._absolute_accuracy and i > 5:
                return new
            # oh well. Another step.
            prev = new
            i += 1
            if i >= self._max_evaluations:
                break
        qassert.fail("max number of iterations reached")


__all__ = [
    "Default",
    "IntegrationPolicy",
    "MidPoint",
    "TrapezoidIntegral",
    "default_refine",
]
