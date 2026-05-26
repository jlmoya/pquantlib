"""Galambos copula.

# C++ parity: ql/math/copulas/galamboscopula.hpp + galamboscopula.cpp (v1.42.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class GalambosCopula:
    """Galambos bivariate copula.

    ``C(x, y) = x * y * exp(((-log x)^-theta + (-log y)^-theta)^(-1/theta))``
    for ``theta >= 0``.
    """

    theta: float

    def __post_init__(self) -> None:
        qassert.require(self.theta >= 0.0, f"theta ({self.theta}) must be greater or equal to 0")

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return (
            x
            * y
            * math.exp(((-math.log(x)) ** -self.theta + (-math.log(y)) ** -self.theta) ** (-1.0 / self.theta))
        )
