"""Husler-Reiss copula.

# C++ parity: ql/math/copulas/huslerreisscopula.hpp + huslerreisscopula.cpp (v1.42.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)


@dataclass(frozen=True, slots=True)
class HuslerReissCopula:
    """Husler-Reiss bivariate copula.

    ``C(x, y) = x^Phi(1/theta + 0.5 * theta * log(-log x / -log y)) *
                 y^Phi(1/theta + 0.5 * theta * log(-log y / -log x))``
    for ``theta >= 0``, with ``Phi`` the standard normal CDF.
    """

    theta: float
    # NOTE: ``field(repr=False, compare=False)`` keeps equality/hashing keyed on
    # ``theta`` alone — semantic parity with the C++ class which only stores theta.
    _cum_normal: CumulativeNormalDistribution = field(
        default_factory=CumulativeNormalDistribution, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        qassert.require(self.theta >= 0.0, f"theta ({self.theta}) must be greater or equal to 0")

    def __call__(self, x: float, y: float) -> float:
        qassert.require(0.0 <= x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(0.0 <= y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        # Mirrors C++ literally: -log(x) / -log(y) (the two minus signs cancel
        # for log purposes but match the source verbatim).
        log_ratio_xy = math.log(-math.log(x) / -math.log(y))
        log_ratio_yx = math.log(-math.log(y) / -math.log(x))
        exp_x = self._cum_normal(1.0 / self.theta + 0.5 * self.theta * log_ratio_xy)
        exp_y = self._cum_normal(1.0 / self.theta + 0.5 * self.theta * log_ratio_yx)
        return x**exp_x * y**exp_y
