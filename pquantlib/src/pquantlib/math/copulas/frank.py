"""Frank copula.

# C++ parity: ql/math/copulas/frankcopula.hpp + frankcopula.cpp (v1.42.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class FrankCopula:
    """Frank bivariate copula.

    ``C(x, y) = -(1/theta) * log(1 + (exp(-theta * x) - 1)(exp(-theta * y) - 1) / (exp(-theta) - 1))``
    for ``theta != 0``.
    """

    theta: float

    def __post_init__(self) -> None:
        qassert.require(self.theta != 0.0, f"theta ({self.theta}) must be different from 0")

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return (
            -1.0
            / self.theta
            * math.log(
                1.0
                + (math.exp(-self.theta * x) - 1.0)
                * (math.exp(-self.theta * y) - 1.0)
                / (math.exp(-self.theta) - 1.0)
            )
        )
