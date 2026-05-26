"""Trapezoid composite quadrature.

# C++ parity: ql/math/integrals/trapezoidintegral.hpp (v1.42.1) class
# ``TrapezoidIntegral<IntegrationPolicy>``.

The C++ template is parameterized by an integration policy (``Default``
or ``MidPoint``) selecting the refinement strategy. The probe uses the
``Default`` policy; the Python port collapses to that single specialization
here and exposes :class:`TrapezoidIntegral`.

The ``Default`` policy doubles N each refinement (``nb_evaluations() == 2``)
and refines via midpoint sampling: each new iteration evaluates ``f`` at
the midpoints of the previous grid. The composite trapezoid sum is
``(I + dx * sum) / 2``, where ``I`` is the previous estimate.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.integrals.integrator import Integrator, RealFunction


def default_refine(f: RealFunction, a: float, b: float, prev: float, n: int) -> float:
    """One ``Default`` refinement step: insert ``n`` new midpoints into the existing grid."""
    total = 0.0
    dx = (b - a) / n
    x = a + dx / 2.0
    for _ in range(n):
        total += f(x)
        x += dx
    return (prev + dx * total) / 2.0


class TrapezoidIntegral(Integrator):
    """Composite trapezoid rule, refined adaptively until ``absolute_accuracy``."""

    def __init__(self, absolute_accuracy: float, max_evaluations: int) -> None:
        super().__init__(absolute_accuracy, max_evaluations)

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # start from the coarsest trapezoid
        n = 1
        prev = (f(a) + f(b)) * (b - a) / 2.0
        self._increase_number_of_evaluations(2)
        # ...and refine it
        i = 1
        while i < self._max_evaluations:
            new = default_refine(f, a, b, prev, n)
            # Default::nbEvalutions() == 2, so we add N*(2-1) == N new points.
            self._increase_number_of_evaluations(n)
            n *= 2
            # good enough? Also, don't run away immediately
            if abs(prev - new) <= self._absolute_accuracy and i > 5:
                return new
            prev = new
            i += 1
        qassert.fail("max number of iterations reached")
