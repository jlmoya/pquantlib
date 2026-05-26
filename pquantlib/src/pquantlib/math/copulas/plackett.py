"""Plackett copula.

# C++ parity: ql/math/copulas/plackettcopula.hpp + plackettcopula.cpp (v1.42.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class PlackettCopula:
    """Plackett bivariate copula.

    ``C(x, y) = ((1 + (theta-1)(x+y)) - sqrt((1+(theta-1)(x+y))^2 - 4*x*y*theta*(theta-1)))
               / (2*(theta-1))``
    for ``theta >= 0`` and ``theta != 1``.
    """

    theta: float

    def __post_init__(self) -> None:
        qassert.require(self.theta >= 0.0, f"theta ({self.theta}) must be greater or equal to 0")
        qassert.require(self.theta != 1.0, f"theta ({self.theta}) must be different from 1")

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        t = self.theta
        a = 1.0 + (t - 1.0) * (x + y)
        return (a - math.sqrt(a * a - 4.0 * x * y * t * (t - 1.0))) / (2.0 * (t - 1.0))
