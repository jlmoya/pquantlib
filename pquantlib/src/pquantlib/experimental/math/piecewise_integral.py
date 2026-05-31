"""PiecewiseIntegral — integrator that splits at critical points.

# C++ parity: ql/experimental/math/piecewiseintegral.hpp + ...integral.cpp
# @ v1.42.1 (099987f0).

Integrates a piecewise-well-behaved function by delegating each sub-interval to
a supplied 1-D ``integrator``, splitting the integration domain at a set of
critical points. When ``avoid_critical_points`` is set (the default) each
sub-interval is nudged inward by a factor ``1 + eps`` / ``1 / (1 + eps)`` so the
critical points themselves are excluded from the open integration intervals.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.integrals.integrator import Integrator, RealFunction


class PiecewiseIntegral(Integrator):
    """Integrator splitting the domain at a set of critical points."""

    __slots__ = ("_critical_points", "_eps", "_integrator")

    def __init__(
        self,
        integrator: Integrator,
        critical_points: Sequence[float],
        avoid_critical_points: bool = True,
    ) -> None:
        # C++ parity: base Integrator(1.0, 1) — accuracy/evals are the inner
        # integrator's responsibility.
        super().__init__(1.0, 1)
        self._integrator = integrator
        self._eps = (1.0 + QL_EPSILON) if avoid_critical_points else 1.0
        # sort + unique (close_enough collapse), as in the C++ ctor.
        pts = sorted(critical_points)
        unique: list[float] = []
        for p in pts:
            if not unique or not close_enough(unique[-1], p):
                unique.append(p)
        self._critical_points = unique

    def _integrate_h(self, f: RealFunction, a: float, b: float) -> float:
        if not close_enough(a, b):
            return self._integrator(f, a, b)
        return 0.0

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        cp = self._critical_points
        eps = self._eps

        # a0 = first critical point >= a (lower_bound)
        a0 = bisect_left(cp, a)

        if a0 == len(cp):
            tmp = 1.0
            if cp and close_enough(a, cp[-1]):
                tmp = eps
            return self._integrate_h(f, a * tmp, b)

        res = 0.0

        if not close_enough(a, cp[a0]):
            res += self._integrate_h(f, a, min(cp[a0] / eps, b))

        # b0 = first critical point >= b (lower_bound)
        b0 = bisect_left(cp, b)
        if b0 == len(cp):
            b0 -= 1
            if not close_enough(cp[b0], b):
                res += self._integrate_h(f, cp[b0] * eps, b)

        x = a0
        while x < b0:
            res += self._integrate_h(f, cp[x] * eps, min(cp[x + 1] / eps, b))
            x += 1

        return res
