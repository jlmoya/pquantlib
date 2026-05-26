"""Gumbel copula.

# C++ parity: ql/math/copulas/gumbelcopula.hpp + gumbelcopula.cpp (v1.42.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class GumbelCopula:
    """Gumbel bivariate copula.

    ``C(x, y) = exp(-((-log x)^theta + (-log y)^theta)^(1/theta))`` for ``theta >= 1``.
    """

    theta: float

    def __post_init__(self) -> None:
        qassert.require(self.theta >= 1.0, f"theta ({self.theta}) must be greater or equal to 1")

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return math.exp(
            -(((-math.log(x)) ** self.theta + (-math.log(y)) ** self.theta) ** (1.0 / self.theta))
        )
