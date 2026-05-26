"""Simpson composite quadrature.

# C++ parity: ql/math/integrals/simpsonintegral.hpp (v1.42.1) class
# ``SimpsonIntegral``.

Simpson's rule via Romberg-style refinement: at each iteration the
trapezoid estimate is improved by the standard ``adj = (4*new - prev) / 3``
combination (degree-3 polynomial cancellation).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.integrals.integrator import Integrator, RealFunction
from pquantlib.math.integrals.trapezoid import default_refine


class SimpsonIntegral(Integrator):
    """Romberg-improved Simpson's rule (adapted from trapezoid sequence)."""

    def __init__(self, absolute_accuracy: float, max_evaluations: int) -> None:
        super().__init__(absolute_accuracy, max_evaluations)

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # start from the coarsest trapezoid
        n = 1
        prev = (f(a) + f(b)) * (b - a) / 2.0
        self._increase_number_of_evaluations(2)
        adj_prev = prev
        i = 1
        while i < self._max_evaluations:
            new = default_refine(f, a, b, prev, n)
            self._increase_number_of_evaluations(n)
            n *= 2
            new_adj = (4.0 * new - prev) / 3.0
            # good enough? Also, don't run away immediately
            if abs(adj_prev - new_adj) <= self._absolute_accuracy and i > 5:
                return new_adj
            prev = new
            adj_prev = new_adj
            i += 1
        qassert.fail("max number of iterations reached")
