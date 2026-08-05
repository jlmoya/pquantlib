"""Abcd functional form ``f(t) = (a + b t) e^{-c t} + d``.

# C++ parity: ql/math/abcdmathfunction.{hpp,cpp} (v1.43).

Rebonato's parameterisation of an instantaneous volatility. PQuantLib already
had ``abcd_value`` (the bare ``operator()``) in
:mod:`pquantlib.math.interpolations.abcd_interpolation` and ``AbcdFunction``
(the covariance integrals) in the market-models package; neither is this
class, which owns the derivative and primitive coefficient vectors and the
rolling-window coefficient transforms.

Two behaviours are easy to lose in translation and are transcribed
deliberately: ``operator()``, ``derivative`` and ``primitive`` all clamp to
exactly 0 for ``t < 0`` rather than evaluating the formula, and
``maximumLocation`` returns ``QL_MAX_REAL`` (not infinity, not a raise) when
``b == 0`` and ``a < 0``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.math.constants import QL_MAX_REAL


class AbcdMathFunction:
    """``f(t) = (a + b t) e^{-c t} + d``, with derivative and primitive.

    # C++ parity: ``class AbcdMathFunction`` — abcdmathfunction.hpp:35-102,
    # abcdmathfunction.cpp:28-135.
    """

    __slots__ = (
        "_a",
        "_b",
        "_c",
        "_d",
        "_da",
        "_db",
        "_diacplusbcc",
        "_dibc",
        "_k",
        "_pa",
        "_pb",
    )

    def __init__(
        self,
        a: float = 0.002,
        b: float = 0.001,
        c: float = 0.16,
        d: float = 0.0005,
    ) -> None:
        self._a: float = a
        self._b: float = b
        self._c: float = c
        self._d: float = d
        # C++ parity: abcdmathfunction.cpp:54-70 — ``initialize_``.
        self.validate(a, b, c, d)
        self._da: float = b - c * a
        self._db: float = -c * b
        self._pa: float = -(a + b / c) / c
        self._pb: float = -b / c
        self._k: float = 0.0
        self._dibc: float = b / c
        self._diacplusbcc: float = a / c + self._dibc / c

    @classmethod
    def from_coefficients(cls, abcd: Sequence[float]) -> AbcdMathFunction:
        """C++ parity: the ``AbcdMathFunction(std::vector<Real>)`` overload."""
        return cls(abcd[0], abcd[1], abcd[2], abcd[3])

    @staticmethod
    def validate(a: float, b: float, c: float, d: float) -> None:
        """C++ parity: ``AbcdMathFunction::validate`` — abcdmathfunction.cpp:28-52."""
        qassert.require(c > 0, f"c ({c}) must be positive")
        qassert.require(d >= 0, f"d ({d}) must be non negative")
        qassert.require(a + d >= 0, f"a+d ({a}+{d}) must be non negative")

        if b >= 0.0:
            return

        # the one and only stationary point...
        zero_first_derivative = 1.0 / c - a / b
        if zero_first_derivative >= 0.0:
            # ... is a minimum: must have f(zeroFirstDerivative) >= 0
            bound = -(d * c) / math.exp(c * a / b - 1.0)
            qassert.require(
                b >= bound,
                f"b ({b}) less than {bound}: negative function value at "
                f"stationary point {zero_first_derivative}",
            )

    def __call__(self, t: float) -> float:
        # C++ parity: abcdmathfunction.hpp:105-108 — clamped at t < 0.
        return 0.0 if t < 0 else (self._a + self._b * t) * math.exp(-self._c * t) + self._d

    def derivative(self, t: float) -> float:
        # C++ parity: abcdmathfunction.hpp:110-113.
        return 0.0 if t < 0 else (self._da + self._db * t) * math.exp(-self._c * t)

    def primitive(self, t: float) -> float:
        # C++ parity: abcdmathfunction.hpp:115-118.
        if t < 0:
            return 0.0
        return (self._pa + self._pb * t) * math.exp(-self._c * t) + self._d * t + self._k

    def definite_integral(self, t1: float, t2: float) -> float:
        # C++ parity: abcdmathfunction.cpp:118-121.
        return self.primitive(t2) - self.primitive(t1)

    def maximum_location(self) -> float:
        # C++ parity: abcdmathfunction.cpp:104-116.
        if self._b == 0.0:
            return 0.0 if self._a >= 0.0 else QL_MAX_REAL
        zero_first_derivative = 1.0 / self._c - self._a / self._b
        return zero_first_derivative if zero_first_derivative > 0.0 else 0.0

    def maximum_value(self) -> float:
        # C++ parity: abcdmathfunction.hpp:120-124.
        if self._b == 0.0 or self._a <= 0.0:
            return self._d
        return self(self.maximum_location())

    def long_term_value(self) -> float:
        # C++ parity: abcdmathfunction.hpp:54.
        return self._d

    def a(self) -> float:
        return self._a

    def b(self) -> float:
        return self._b

    def c(self) -> float:
        return self._c

    def d(self) -> float:
        return self._d

    def coefficients(self) -> list[float]:
        # C++ parity: abcdmathfunction.hpp:74.
        return [self._a, self._b, self._c, self._d]

    def derivative_coefficients(self) -> list[float]:
        # C++ parity: abcdmathfunction.cpp:60-63 — note the fourth entry is 0,
        # not d: the derivative has no constant term.
        return [self._da, self._db, self._c, 0.0]

    def definite_integral_coefficients(self, t: float, t2: float) -> list[float]:
        """Abcd coefficients of the integral over a rolling window ``t2 - t``.

        # C++ parity: abcdmathfunction.cpp:123-135.
        """
        dt = t2 - t
        expcdt = math.exp(-self._c * dt)
        return [
            self._diacplusbcc - (self._diacplusbcc + self._dibc * dt) * expcdt,
            self._dibc * (1.0 - expcdt),
            self._c,
            self._d * dt,
        ]

    def definite_derivative_coefficients(self, t: float, t2: float) -> list[float]:
        """Abcd coefficients of the derivative over a rolling window ``t2 - t``.

        # C++ parity: abcdmathfunction.cpp:121-133.
        """
        dt = t2 - t
        expcdt = math.exp(-self._c * dt)
        result = [0.0, 0.0, 0.0, 0.0]
        result[1] = self._b * self._c / (1.0 - expcdt)
        result[0] = self._a * self._c - self._b + result[1] * dt * expcdt
        result[0] /= 1.0 - expcdt
        result[2] = self._c
        result[3] = self._d / dt
        return result


__all__ = ["AbcdMathFunction"]
