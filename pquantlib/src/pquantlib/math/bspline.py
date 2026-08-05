"""B-spline basis functions.

# C++ parity: ql/math/bspline.{hpp,cpp} (v1.43).

The ``(p+1)``-th order basis functions ``N_{i,p}(x)``, defined by the Cox-de
Boor recursion, over ``n+1`` control points and a knot vector of length
``p + n + 2``.

Not ``scipy.interpolate.BSpline``: that class evaluates a *spline* (a
coefficient-weighted sum) with de Boor's algorithm and its own handling of
repeated knots and of the right end point. This is the individual basis
function, evaluated by the naive recursion, with a half-open support test
(``knots[i] <= x < knots[i+1]``) at degree 0 — so the value exactly at the
last knot is 0, not 1, and repeated knots divide by zero rather than being
special-cased. Both of those are observable and both are C++'s: with a clamped
knot vector the C++ recursion returns NaN, and the probe pins that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert


def _ieee_div(num: float, den: float) -> float:
    """C++ ``/`` on doubles, which does not raise on a zero denominator.

    Python's ``/`` raises ``ZeroDivisionError`` where IEEE-754 (and therefore
    C++) yields NaN for 0/0 and a signed infinity otherwise. The B-spline
    recursion divides by knot *differences*, which are zero exactly when the
    knot vector has repeated entries — the clamped case — so this is reached
    in ordinary use and the C++ answer there is NaN, not an exception.
    """
    if den != 0.0:
        return num / den
    if num == 0.0 or math.isnan(num):
        return math.nan
    return math.copysign(math.inf, num) * math.copysign(1.0, den)


class BSpline:
    """B-spline basis functions of degree ``p`` over ``n+1`` control points.

    # C++ parity: ``class BSpline`` — bspline.hpp:57-72, bspline.cpp:24-58.
    """

    __slots__ = ("_knots", "_n", "_p")

    def __init__(self, p: int, n: int, knots: Sequence[float]) -> None:
        # C++ parity: bspline.cpp:24-42.
        qassert.require(p >= 1, "lowest degree B-spline has p = 1")
        qassert.require(n >= 1, "number of control points n+1 >= 2")
        qassert.require(p <= n, "must have p <= n")
        qassert.require(len(knots) == p + n + 2, "number of knots must equal p+n+2")
        for i in range(len(knots) - 1):
            qassert.require(knots[i] <= knots[i + 1], "knots points must be nondecreasing")

        self._p: int = p
        self._n: int = n
        self._knots: tuple[float, ...] = tuple(float(k) for k in knots)

    def __call__(self, i: int, x: float) -> float:
        # C++ parity: bspline.cpp:45-48.
        qassert.require(i <= self._n, "i must not be greater than n")
        return self._basis(i, self._p, x)

    def _basis(self, i: int, p: int, x: float) -> float:
        """Cox-de Boor recursion.

        # C++ parity: ``BSpline::N`` — bspline.cpp:51-58. The divisions go
        # through :func:`_ieee_div` so a repeated knot yields NaN as it does
        # in C++ rather than raising.
        """
        knots = self._knots
        if p == 0:
            return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
        return _ieee_div(x - knots[i], knots[i + p] - knots[i]) * self._basis(i, p - 1, x) + _ieee_div(
            knots[i + p + 1] - x, knots[i + p + 1] - knots[i + 1]
        ) * self._basis(i + 1, p - 1, x)


__all__ = ["BSpline"]
