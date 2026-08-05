"""Filon's formulae for sine and cosine integrals.

# C++ parity: ql/math/integrals/filonintegral.{hpp,cpp} (v1.43).

Given an even number of intervals ``N = 2n`` over ``[a, b]``, computes

    int_a^b f(x) cos(t x) dx      (Type.Cosine)
    int_a^b f(x) sin(t x) dx      (Type.Sine)

The oscillatory factor is *not* part of the integrand handed to
:meth:`FilonIntegral.__call__` — it is built into the rule, which is the point
of Filon's method: the quadrature stays accurate when ``t`` is large enough
that a polynomial rule would need an impractical number of panels.

Reference: Abramowitz & Stegun, "Handbook of Mathematical Functions", 9th
printing, pp. 890-891, 1972.
"""

from __future__ import annotations

import math
from enum import IntEnum

from pquantlib import qassert
from pquantlib.math.integrals.integrator import Integrator, RealFunction

# C++ ``Null<Real>()`` — ql/utilities/null.hpp yields
# ``std::numeric_limits<float>::max()``.
_NULL_REAL: float = 3.4028234663852886e38


class FilonIntegral(Integrator):
    """Filon's sine / cosine quadrature on ``intervals`` (even) panels.

    # C++ parity: filonintegral.hpp:47-61, filonintegral.cpp:33-88.
    """

    class Type(IntEnum):
        """# C++ parity: ``enum Type { Sine, Cosine }`` (filonintegral.hpp:49)."""

        Sine = 0
        Cosine = 1

    __slots__ = ("_intervals", "_n", "_t", "_type")

    def __init__(self, type_: Type, t: float, intervals: int) -> None:
        # C++ parity: Integrator(Null<Real>(), intervals+1).
        super().__init__(_NULL_REAL, intervals + 1)
        self._type: FilonIntegral.Type = type_
        self._t: float = t
        self._intervals: int = intervals
        self._n: int = intervals // 2
        qassert.require(not (intervals & 1), "number of intervals must be even")

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: filonintegral.cpp:42-88.
        n = self._n
        t = self._t
        h = (b - a) / (2 * n)

        # C++ Array(size, value, increment) accumulates (value += increment)
        # rather than computing a + i*h, so the abscissae round identically
        # only if the accumulation is reproduced.
        x: list[float] = []
        value = a
        for _ in range(2 * n + 1):
            x.append(value)
            value += h

        theta = t * h
        theta2 = theta * theta
        theta3 = theta2 * theta

        alpha = (
            1 / theta
            + math.sin(2 * theta) / (2 * theta2)
            - 2 * (math.sin(theta) * math.sin(theta)) / theta3
        )
        beta = 2 * (
            (1 + math.cos(theta) * math.cos(theta)) / theta2 - math.sin(2 * theta) / theta3
        )
        gamma = 4 * (math.sin(theta) / theta3 - math.cos(theta) / theta2)

        v = [f(e) for e in x]

        if self._type == FilonIntegral.Type.Cosine:
            f1 = math.sin
            f2 = math.cos
        else:
            f1 = math.cos
            f2 = math.sin

        c_2n_1 = 0.0
        c_2n = v[0] * f2(t * a) - 0.5 * (v[2 * n] * f2(t * b) + v[0] * f2(t * a))

        for i in range(1, n + 1):
            c_2n += v[2 * i] * f2(t * x[2 * i])
            c_2n_1 += v[2 * i - 1] * f2(t * x[2 * i - 1])

        return h * (
            alpha
            * (v[2 * n] * f1(t * x[2 * n]) - v[0] * f1(t * x[0]))
            * (1.0 if self._type == FilonIntegral.Type.Cosine else -1.0)
            + beta * c_2n
            + gamma * c_2n_1
        )


__all__ = ["FilonIntegral"]
