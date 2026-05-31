"""PiecewiseConstantCorrelation — abstract piecewise-constant correlation.

# C++ parity: ql/models/marketmodels/piecewiseconstantcorrelation.hpp (v1.42.1).

Abstract base for an instantaneous-correlation structure that is constant on
each ``[times[i-1], times[i]]`` interval. ``corr_times`` must include all
rate times but the last. The ``correlation(i)`` accessor is concrete (it
indexes into ``correlations()``); the rest is abstract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix


class PiecewiseConstantCorrelation(ABC):
    """Abstract piecewise-constant instantaneous-correlation structure.

    # C++ parity: piecewiseconstantcorrelation.hpp PiecewiseConstantCorrelation.
    """

    @abstractmethod
    def times(self) -> list[float]:
        """The interval boundary times."""

    @abstractmethod
    def rate_times(self) -> list[float]:
        """The rate times the correlation is defined against."""

    @abstractmethod
    def correlations(self) -> list[Matrix]:
        """The per-interval correlation matrices."""

    @abstractmethod
    def number_of_rates(self) -> int:
        """Number of forward rates."""

    def correlation(self, i: int) -> Matrix:
        """The correlation matrix for interval ``i``.

        # C++ parity: piecewiseconstantcorrelation.hpp
        PiecewiseConstantCorrelation::correlation.
        """
        results = self.correlations()
        qassert.require(
            i < len(results),
            f"index ({i}) must be less than correlations vector size ({len(results)})",
        )
        return results[i]
