"""PiecewiseConstantVariance — abstract piecewise-constant variance.

# C++ parity: ql/models/marketmodels/models/piecewiseconstantvariance.{hpp,cpp}
# (v1.42.1).

Abstract base for a per-rate variance structure that is constant on each
rate-time interval. Subclasses supply ``variances()`` / ``volatilities()`` /
``rate_times()``; the ``variance(i)`` / ``volatility(i)`` / ``total_variance(i)``
/ ``total_volatility(i)`` accessors are concrete.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import reduce

from pquantlib import qassert


class PiecewiseConstantVariance(ABC):
    """Abstract piecewise-constant variance structure.

    # C++ parity: piecewiseconstantvariance.hpp PiecewiseConstantVariance.
    """

    @abstractmethod
    def variances(self) -> list[float]:
        """The per-step variances."""

    @abstractmethod
    def volatilities(self) -> list[float]:
        """The per-step volatilities."""

    @abstractmethod
    def rate_times(self) -> list[float]:
        """The rate times the variance is defined against."""

    def variance(self, i: int) -> float:
        """Variance of step ``i``.

        # C++ parity: piecewiseconstantvariance.cpp variance.
        """
        qassert.require(i < len(self.variances()), "invalid step index")
        return self.variances()[i]

    def volatility(self, i: int) -> float:
        """Volatility of step ``i``.

        # C++ parity: piecewiseconstantvariance.cpp volatility.
        """
        qassert.require(i < len(self.volatilities()), "invalid step index")
        return self.volatilities()[i]

    def total_variance(self, i: int) -> float:
        """Cumulative variance through step ``i`` (inclusive).

        # C++ parity: piecewiseconstantvariance.cpp totalVariance.
        """
        qassert.require(i < len(self.variances()), "invalid step index")
        # C++ parity: std::accumulate(begin, begin+i+1, 0.0) — sequential
        # left-to-right summation (not pairwise / Kahan).
        return reduce(lambda acc, v: acc + v, self.variances()[: i + 1], 0.0)

    def total_volatility(self, i: int) -> float:
        """Cumulative volatility ``sqrt(totalVariance(i) / rateTimes()[i])``.

        # C++ parity: piecewiseconstantvariance.cpp totalVolatility.
        """
        return math.sqrt(self.total_variance(i) / self.rate_times()[i])
