"""Quadratic formula helper.

# C++ parity: ql/math/quadratic.{hpp,cpp} (v1.42.1).

A tiny value object wrapping the coefficients ``(a, b, c)`` of
``a*x^2 + b*x + c``. Used by the MarketModels caplet-coterminal calibration
machinery (AlphaFinder, the max-homogeneity single-rate closest-point finder)
to solve for swap-rate vol multipliers.
"""

from __future__ import annotations

import math


class Quadratic:
    """``a*x^2 + b*x + c`` with root / turning-point helpers.

    # C++ parity: ``QuantLib::quadratic`` (lower-case in C++).
    """

    def __init__(self, a: float, b: float, c: float) -> None:
        # C++ parity: quadratic::quadratic(Real a, Real b, Real c).
        self._a = a
        self._b = b
        self._c = c

    def turning_point(self) -> float:
        """The x-coordinate of the vertex: ``-b / (2a)``.

        # C++ parity: quadratic::turningPoint.
        """
        return -self._b / (2.0 * self._a)

    def value_at_turning_point(self) -> float:
        """The value of the quadratic at its vertex.

        # C++ parity: quadratic::valueAtTurningPoint.
        """
        return self(self.turning_point())

    def __call__(self, x: float) -> float:
        # C++ parity: quadratic::operator()(Real x) == x*(x*a+b)+c.
        return x * (x * self._a + self._b) + self._c

    def discriminant(self) -> float:
        """``b^2 - 4ac``.

        # C++ parity: quadratic::discriminant.
        """
        return self._b * self._b - 4 * self._a * self._c

    def roots(self) -> tuple[float, float, bool]:
        """Return ``(x, y, real)``.

        If the roots are real, ``x <= y`` are the two roots and ``real`` is
        ``True``. Otherwise both equal the turning point and ``real`` is
        ``False``.

        # C++ parity: quadratic::roots(Real& x, Real& y) — the C++ signature
        returns the real/complex flag and writes the roots through reference
        parameters; Python returns the triple instead.
        """
        d = self.discriminant()
        if d < 0:
            tp = self.turning_point()
            return tp, tp, False
        d = math.sqrt(d)
        x = (-self._b - d) / (2 * self._a)
        y = (-self._b + d) / (2 * self._a)
        return x, y, True
