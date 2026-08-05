"""Gaussian-assumption risk measures layered over a statistics tool.

# C++ parity: ql/math/statistics/gaussianstatistics.hpp (v1.43).

C++ spells the layering as a template that inherits from its parameter::

    template<class Stat> class GenericGaussianStatistics : public Stat;
    typedef GenericGaussianStatistics<GeneralStatistics> GaussianStatistics;

Python has no "inherit from a template parameter", so the analogue is a
**mixin**: :class:`GenericGaussianStatistics` carries only the Gaussian
closed forms and requires the class it is mixed into to supply ``mean()``
and ``standard_deviation()``. The C++ instantiations become concrete
classes built by multiple inheritance, mixin first::

    class GaussianStatistics(GenericGaussianStatistics, GeneralStatistics): ...

The same recipe reproduces the other two instantiations the C++
test-suite exercises — ``GenericGaussianStatistics<IncrementalStatistics>``
and ``GenericGaussianStatistics<StatsHolder>`` — by swapping the second
base. Exactly as in C++, the unparameterised
``GenericGaussianStatistics`` is not usable on its own.

Every measure is a closed form in the mean and standard deviation only;
none of them looks at the samples.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import InverseCumulativeNormal
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.statistics.general_statistics import GeneralStatistics


class GenericGaussianStatistics:
    """Gaussian risk measures over any statistics tool.

    # C++ parity: ``template<class Stat> class GenericGaussianStatistics``
    # (gaussianstatistics.hpp:39-107).
    """

    __slots__ = ()

    if TYPE_CHECKING:
        # Supplied by the class this mixin is combined with — the C++ ``Stat``
        # template parameter. Declared for the type checker only, so the
        # concrete class's own implementations are what run.
        def mean(self) -> float: ...
        def standard_deviation(self) -> float: ...

    # --- Gaussian risk measures -----------------------------------------

    def gaussian_downside_variance(self) -> float:
        """Downside variance under the Gaussian assumption.

        # C++ parity: gaussianstatistics.hpp:53-55 — ``gaussianRegret(0.0)``.
        """
        return self.gaussian_regret(0.0)

    def gaussian_downside_deviation(self) -> float:
        """Square root of :meth:`gaussian_downside_variance`.

        # C++ parity: gaussianstatistics.hpp:60-62.
        """
        return math.sqrt(self.gaussian_downside_variance())

    def gaussian_regret(self, target: float) -> float:
        """Variance of the observations below ``target``, Gaussian closed form.

        # C++ parity: gaussianstatistics.hpp:130-144.
        """
        m = self.mean()
        std = self.standard_deviation()
        variance = std * std
        g_integral = CumulativeNormalDistribution(m, std)
        g = NormalDistribution(m, std)
        first_term = variance + m * m - 2.0 * target * m + target * target
        alfa = g_integral(target)
        second_term = m - target
        beta = variance * g(target)
        result = alfa * first_term - beta * second_term
        return result / alfa

    def gaussian_percentile(self, percentile: float) -> float:
        """Gaussian-assumption ``percentile``-th percentile.

        # C++ parity: gaussianstatistics.hpp:147-159. Both extremes are
        # excluded.
        """
        qassert.require(percentile > 0.0, f"percentile ({percentile}) must be > 0.0")
        qassert.require(percentile < 1.0, f"percentile ({percentile}) must be < 1.0")
        g_inverse = InverseCumulativeNormal(self.mean(), self.standard_deviation())
        return g_inverse(percentile)

    def gaussian_top_percentile(self, percentile: float) -> float:
        """Gaussian-assumption top percentile.

        # C++ parity: gaussianstatistics.hpp:162-167.
        """
        return self.gaussian_percentile(1.0 - percentile)

    def gaussian_potential_upside(self, percentile: float) -> float:
        """Gaussian-assumption potential upside, floored at zero.

        # C++ parity: gaussianstatistics.hpp:170-180 — ``percentile`` must be
        # in [0.9, 1.0).
        """
        qassert.require(
            percentile < 1.0 and percentile >= 0.9,
            f"percentile ({percentile}) out of range [0.9, 1)",
        )
        result = self.gaussian_percentile(percentile)
        # potential upside must be a gain, i.e., floored at 0.0
        return max(result, 0.0)

    def gaussian_value_at_risk(self, percentile: float) -> float:
        """Gaussian-assumption value-at-risk, reported as a positive loss.

        # C++ parity: gaussianstatistics.hpp:184-196.
        """
        qassert.require(
            percentile < 1.0 and percentile >= 0.9,
            f"percentile ({percentile}) out of range [0.9, 1)",
        )
        result = self.gaussian_percentile(1.0 - percentile)
        # VAR must be a loss: MIN(dist(1-percentile), 0.0), negated so it is
        # reported as a positive quantity.
        return -min(result, 0.0)

    def gaussian_expected_shortfall(self, percentile: float) -> float:
        """Gaussian-assumption expected shortfall (conditional VaR).

        # C++ parity: gaussianstatistics.hpp:200-216.
        """
        qassert.require(
            percentile < 1.0 and percentile >= 0.9,
            f"percentile ({percentile}) out of range [0.9, 1)",
        )
        m = self.mean()
        std = self.standard_deviation()
        g_inverse = InverseCumulativeNormal(m, std)
        var = g_inverse(1.0 - percentile)
        g = NormalDistribution(m, std)
        result = m - std * std * g(var) / (1.0 - percentile)
        # expectedShortfall must be a loss, capped at 0.0 and negated
        return -min(result, 0.0)

    def gaussian_shortfall(self, target: float) -> float:
        """Gaussian-assumption probability of ending below ``target``.

        # C++ parity: gaussianstatistics.hpp:219-225.
        """
        g_integral = CumulativeNormalDistribution(self.mean(), self.standard_deviation())
        return g_integral(target)

    def gaussian_average_shortfall(self, target: float) -> float:
        """Gaussian-assumption averaged shortfallness below ``target``.

        # C++ parity: gaussianstatistics.hpp:228-236.
        """
        m = self.mean()
        std = self.standard_deviation()
        g_integral = CumulativeNormalDistribution(m, std)
        g = NormalDistribution(m, std)
        return (target - m) + std * std * g(target) / g_integral(target)


class StatsHolder:
    """Precomputed (mean, standard deviation) pair usable as a statistics tool.

    # C++ parity: ``class StatsHolder`` (gaussianstatistics.hpp:114-125).

    Exists so ``GenericGaussianStatistics`` can be layered over moments that
    were computed elsewhere, without carrying a sample set.
    """

    __slots__ = ("_mean", "_standard_deviation")

    def __init__(self, mean: float, standard_deviation: float) -> None:
        self._mean = mean
        self._standard_deviation = standard_deviation

    def mean(self) -> float:
        """The stored mean."""
        return self._mean

    def standard_deviation(self) -> float:
        """The stored standard deviation."""
        return self._standard_deviation


class GaussianStatistics(GenericGaussianStatistics, GeneralStatistics):
    """Default Gaussian statistics tool.

    # C++ parity: ``typedef GenericGaussianStatistics<GeneralStatistics>
    # GaussianStatistics`` (gaussianstatistics.hpp:110).
    """

    __slots__ = ()
