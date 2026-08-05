"""Cumulative Gamma distribution.

# C++ parity: ql/math/distributions/gammadistribution.{hpp,cpp} (v1.43).

``GammaFunction`` — the other class declared in the same C++ header — already
lives in :mod:`pquantlib.math.distributions.gamma_function`; it is re-exported
here so the C++ header maps onto one Python module.

``CumulativeGammaDistribution`` is *not* ``scipy.special.gammainc``. It is a
distinct Numerical-Recipes evaluation with its own convergence test
(``|del| < |sum| * 3.0e-7`` in the series branch, ``|del - 1| < QL_EPSILON``
in the continued fraction) and its own 100-iteration cap, and it is the exact
routine ``CumulativeChiSquareDistribution`` is defined in terms of. Note in
particular that the series branch's tolerance is ``3.0e-7`` *relative to the
running sum*, which is looser than the 1e-13 used by
:func:`~pquantlib.math.incomplete_gamma.incomplete_gamma_function`; the two
C++ routines therefore do not agree to machine precision with each other, and
the port must keep them separate to reproduce either.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON, QL_MAX_REAL
from pquantlib.math.distributions.gamma_function import GammaFunction

_MAX_ITERATIONS: Final[int] = 100
# C++ gammadistribution.cpp:44 — the series-branch relative convergence test.
_SERIES_ACCURACY: Final[float] = 3.0e-7


class CumulativeGammaDistribution:
    """Regularized lower incomplete Gamma ``P(a, x)`` as a distribution.

    # C++ parity: ``class CumulativeGammaDistribution`` —
    # gammadistribution.hpp:33-40, gammadistribution.cpp:24-60.
    """

    __slots__ = ("_a",)

    def __init__(self, a: float) -> None:
        qassert.require(a > 0.0, "invalid parameter for gamma distribution")
        self._a: float = float(a)

    def __call__(self, x: float) -> float:
        # C++ parity: gammadistribution.cpp:24-60, transcribed branch for branch.
        if x <= 0.0:
            return 0.0

        a = self._a
        gln = GammaFunction().log_value(a)

        if x < (a + 1.0):
            ap = a
            delta = 1.0 / a
            total = delta
            for _ in range(1, _MAX_ITERATIONS + 1):
                ap += 1.0
                delta *= x / ap
                total += delta
                if math.fabs(delta) < math.fabs(total) * _SERIES_ACCURACY:
                    return total * math.exp(-x + a * math.log(x) - gln)
        else:
            b = x + 1.0 - a
            c = QL_MAX_REAL
            d = 1.0 / b
            h = d
            for n in range(1, _MAX_ITERATIONS + 1):
                an = -1.0 * n * (n - a)
                b += 2.0
                d = an * d + b
                if math.fabs(d) < QL_EPSILON:
                    d = QL_EPSILON
                c = b + an / c
                if math.fabs(c) < QL_EPSILON:
                    c = QL_EPSILON
                d = 1.0 / d
                delta = d * c
                h *= delta
                if math.fabs(delta - 1.0) < QL_EPSILON:
                    return 1.0 - h * math.exp(-x + a * math.log(x) - gln)
        qassert.fail("too few iterations")


__all__ = ["CumulativeGammaDistribution", "GammaFunction"]
