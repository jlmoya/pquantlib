"""Clayton copula.

# C++ parity: ql/math/copulas/claytoncopula.hpp + claytoncopula.cpp (v1.42.1).
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class ClaytonCopula:
    """Clayton bivariate copula.

    ``C(x, y) = max((x^-theta + y^-theta - 1)^(-1/theta), 0)``
    for ``theta >= -1`` and ``theta != 0``.
    """

    theta: float

    def __post_init__(self) -> None:
        qassert.require(self.theta >= -1.0, f"theta ({self.theta}) must be greater or equal to -1")
        qassert.require(self.theta != 0.0, f"theta ({self.theta}) must be different from 0")

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        inner = x**-self.theta + y**-self.theta - 1.0
        return max(inner ** (-1.0 / self.theta), 0.0)
