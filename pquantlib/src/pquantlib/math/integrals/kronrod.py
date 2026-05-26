"""Gauss-Kronrod adaptive 1-D integration.

# C++ parity: ql/math/integrals/kronrodintegral.hpp +
# ql/math/integrals/kronrodintegral.cpp (v1.42.1) class
# ``GaussKronrodAdaptive``.

15-point Gauss-Kronrod rule with G7 nested inside K15. The interval is
recursively halved whenever the K15/G7 error estimate exceeds the
tolerance.

Only :class:`GaussKronrodAdaptive` is ported here (the cluster C scope).
The non-adaptive 10/21/43/87-point version (``GaussKronrodNonAdaptive``)
is deferred to a follow-up cluster.
"""

from __future__ import annotations

from typing import Final

from pquantlib import qassert
from pquantlib.math.integrals.integrator import Integrator, RealFunction

# weights for 7-point Gauss-Legendre integration (4 unique values, symmetric).
# C++ parity: kronrodintegral.cpp static const Real g7w[].
_G7W: Final[tuple[float, ...]] = (
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870,
)
# weights for 15-point Gauss-Kronrod integration.
# C++ parity: kronrodintegral.cpp static const Real k15w[].
_K15W: Final[tuple[float, ...]] = (
    0.209482141084728,
    0.204432940075298,
    0.190350578064785,
    0.169004726639267,
    0.140653259715525,
    0.104790010322250,
    0.063092092629979,
    0.022935322010529,
)
# abscissae for 15-point Gauss-Kronrod integration.
# C++ parity: kronrodintegral.cpp static const Real k15t[].
_K15T: Final[tuple[float, ...]] = (
    0.000000000000000,
    0.207784955007898,
    0.405845151377397,
    0.586087235467691,
    0.741531185599394,
    0.864864423359769,
    0.949107912342758,
    0.991455371120813,
)


class GaussKronrodAdaptive(Integrator):
    """Adaptive Gauss-Kronrod (G7 / K15 nested rule, recursive interval bisection)."""

    def __init__(self, tolerance: float, max_evaluations: int) -> None:
        super().__init__(tolerance, max_evaluations)
        qassert.require(
            max_evaluations >= 15,
            f"required maxEvaluations ({max_evaluations}) not allowed. It must be >= 15",
        )

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        return self._integrate_recursively(f, a, b, self._absolute_accuracy)

    def _integrate_recursively(self, f: RealFunction, a: float, b: float, tolerance: float) -> float:
        half_length = (b - a) / 2
        center = (a + b) / 2

        fc = f(center)
        g7 = fc * _G7W[0]
        k15 = fc * _K15W[0]

        # calculate g7 and half of k15
        # j runs 1..3, j2 runs 2,4,6
        for j, j2 in ((1, 2), (2, 4), (3, 6)):
            t = half_length * _K15T[j2]
            fsum = f(center - t) + f(center + t)
            g7 += fsum * _G7W[j]
            k15 += fsum * _K15W[j2]

        # calculate other half of k15 (odd j2 indices: 1,3,5,7)
        for j2 in (1, 3, 5, 7):
            t = half_length * _K15T[j2]
            fsum = f(center - t) + f(center + t)
            k15 += fsum * _K15W[j2]

        # multiply by (b - a) / 2
        g7 *= half_length
        k15 *= half_length

        # 15 more function evaluations have been used
        self._increase_number_of_evaluations(15)

        # error is <= |k15 - g7|; if larger than tolerance, split & recurse
        if abs(k15 - g7) < tolerance:
            return k15
        qassert.require(
            self._evaluations + 30 <= self._max_evaluations,
            "maximum number of function evaluations exceeded",
        )
        return self._integrate_recursively(f, a, center, tolerance / 2) + self._integrate_recursively(
            f, center, b, tolerance / 2
        )
