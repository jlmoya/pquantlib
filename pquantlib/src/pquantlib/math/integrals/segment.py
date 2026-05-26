"""Segment (uniform-interval trapezoid) integral.

# C++ parity: ql/math/integrals/segmentintegral.hpp +
# ql/math/integrals/segmentintegral.cpp (v1.42.1) class ``SegmentIntegral``.

Fixed N-segment trapezoid: no adaptivity. The C++ constructor calls the
base with ``(absoluteAccuracy=1, maxEvaluations=1)`` — the base ctor
only enforces ``absoluteAccuracy > QL_EPSILON``, which ``1`` satisfies.
The constructor here mirrors that exact invocation.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.closeness import close_enough
from pquantlib.math.integrals.integrator import Integrator, RealFunction


class SegmentIntegral(Integrator):
    """Uniform-segment trapezoid rule with fixed N intervals."""

    def __init__(self, intervals: int) -> None:
        # C++ parity: SegmentIntegral::SegmentIntegral calls
        # Integrator(1, 1), then requires intervals > 0.
        super().__init__(1.0, 1)
        qassert.require(intervals > 0, "at least 1 interval needed, 0 given")
        self._intervals: int = intervals

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        if close_enough(a, b):
            return 0.0
        dx = (b - a) / self._intervals
        total = 0.5 * (f(a) + f(b))
        end = b - 0.5 * dx
        x = a + dx
        while x < end:
            total += f(x)
            x += dx
        return total * dx
