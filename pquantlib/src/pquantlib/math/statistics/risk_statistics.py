"""Empirical-distribution risk measures layered over a statistics tool.

# C++ parity: ql/math/statistics/riskstatistics.hpp (v1.43).

C++::

    template <class S> class GenericRiskStatistics : public S;
    typedef GenericRiskStatistics<GaussianStatistics> RiskStatistics;

Same mixin recipe as :mod:`pquantlib.math.statistics.gaussian_statistics`:
:class:`GenericRiskStatistics` holds only the empirical risk measures and
requires the class it is mixed into to supply ``mean()``, ``percentile()``,
``samples()`` and ``expectation_value()``. The C++ instantiation becomes::

    class RiskStatistics(GenericRiskStatistics, GaussianStatistics): ...

Conventions worth spelling out, because they are QuantLib's and not the
textbook's:

* every below-target measure uses the **strict** predicate ``x < target``,
  so a sample sitting exactly on the target is excluded;
* ``regret`` requires **more than one** sample below the target and raises
  otherwise; ``expectedShortfall`` and ``averageShortfall`` require at
  least one and raise otherwise — a data set entirely above the target is
  an error, not a zero;
* ``potentialUpside``/``valueAtRisk``/``expectedShortfall`` accept a
  percentile in ``[0.9, 1.0)`` only;
* ``valueAtRisk`` and ``expectedShortfall`` are reported as *positive*
  losses (``-min(x, 0)``), and ``potentialUpside`` as a *non-negative*
  gain (``max(x, 0)``);
* ``expectedShortfall`` averages over ``x < -valueAtRisk(centile)``, i.e.
  over the re-negated VaR, which is **not** the same as the raw
  ``percentile(1 - centile)`` whenever the VaR floor at zero bites.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.math.statistics.gaussian_statistics import GaussianStatistics


class GenericRiskStatistics:
    """Empirical risk measures over any statistics tool.

    # C++ parity: ``template <class S> class GenericRiskStatistics``
    # (riskstatistics.hpp:40-116).
    """

    __slots__ = ()

    if TYPE_CHECKING:
        # Supplied by the class this mixin is combined with — the C++ ``S``
        # template parameter.
        def mean(self) -> float: ...
        def samples(self) -> int: ...
        def percentile(self, percent: float) -> float: ...
        def expectation_value(
            self,
            f: Callable[[float], float],
            in_range: Callable[[float], bool] | None = None,
        ) -> tuple[float, int]: ...

    # --- inline definitions ---------------------------------------------

    def semi_variance(self) -> float:
        """Variance of the observations below the mean.

        # C++ parity: riskstatistics.hpp:129-132 — ``regret(mean())``.
        """
        return self.regret(self.mean())

    def semi_deviation(self) -> float:
        """Square root of :meth:`semi_variance`.

        # C++ parity: riskstatistics.hpp:134-137.
        """
        return math.sqrt(self.semi_variance())

    def downside_variance(self) -> float:
        """Variance of the observations below zero.

        # C++ parity: riskstatistics.hpp:139-142 — ``regret(0.0)``.
        """
        return self.regret(0.0)

    def downside_deviation(self) -> float:
        """Square root of :meth:`downside_variance`.

        # C++ parity: riskstatistics.hpp:144-147.
        """
        return math.sqrt(self.downside_variance())

    # --- template definitions -------------------------------------------

    def regret(self, target: float) -> float:
        """Variance of the observations strictly below ``target``.

        # C++ parity: riskstatistics.hpp:151-165. Requires more than one
        # sample below the target.
        """
        # average over the range below the target
        x, n = self.expectation_value(
            lambda xi: (xi - target) * (xi - target),
            lambda xi: xi < target,
        )
        qassert.require(n > 1, "samples under target <= 1, unsufficient")
        return (n / (n - 1.0)) * x

    def potential_upside(self, centile: float) -> float:
        """Potential upside at ``centile``, floored at zero.

        # C++ parity: riskstatistics.hpp:167-176.
        """
        qassert.require(
            centile >= 0.9 and centile < 1.0,
            f"percentile ({centile}) out of range [0.9, 1.0)",
        )
        # potential upside must be a gain, i.e., floored at 0.0
        return max(self.percentile(centile), 0.0)

    def value_at_risk(self, centile: float) -> float:
        """Value-at-risk at ``centile``, reported as a positive loss.

        # C++ parity: riskstatistics.hpp:178-187.
        """
        qassert.require(
            centile >= 0.9 and centile < 1.0,
            f"percentile ({centile}) out of range [0.9, 1.0)",
        )
        # must be a loss, i.e., capped at 0.0 and negated
        return -min(self.percentile(1.0 - centile), 0.0)

    def expected_shortfall(self, centile: float) -> float:
        """Expected shortfall (conditional VaR) at ``centile``.

        # C++ parity: riskstatistics.hpp:189-205. Averages the samples
        # strictly below ``-value_at_risk(centile)`` and raises when there
        # are none.
        """
        qassert.require(
            centile >= 0.9 and centile < 1.0,
            f"percentile ({centile}) out of range [0.9, 1.0)",
        )
        qassert.require(self.samples() != 0, "empty sample set")
        target = -self.value_at_risk(centile)
        x, n = self.expectation_value(lambda xi: xi, lambda xi: xi < target)
        qassert.require(n != 0, "no data below the target")
        # must be a loss, i.e., capped at 0.0 and negated
        return -min(x, 0.0)

    def shortfall(self, target: float) -> float:
        """Probability of missing ``target``.

        # C++ parity: riskstatistics.hpp:207-211 — averaged over *all*
        # samples, so this is the fraction of samples below the target.
        """
        qassert.require(self.samples() != 0, "empty sample set")
        return self.expectation_value(lambda x: 1.0 if x < target else 0.0)[0]

    def average_shortfall(self, target: float) -> float:
        """Averaged shortfallness, ``E[target - x | x < target]``.

        # C++ parity: riskstatistics.hpp:213-223.
        """
        x, n = self.expectation_value(lambda xi: target - xi, lambda xi: xi < target)
        qassert.require(n != 0, "no data below the target")
        return x


class RiskStatistics(GenericRiskStatistics, GaussianStatistics):
    """Default risk-measures tool.

    # C++ parity: ``typedef GenericRiskStatistics<GaussianStatistics>
    # RiskStatistics`` (riskstatistics.hpp:123).
    """

    __slots__ = ()
