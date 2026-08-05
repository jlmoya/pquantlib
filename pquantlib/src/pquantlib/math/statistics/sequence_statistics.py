"""N-dimensional (sequence) statistics aggregator.

# C++ parity: ql/math/statistics/sequencestatistics.hpp (v1.43) —
# ``template <class StatisticsType> class GenericSequenceStatistics`` plus
# the ``SequenceStatistics`` / ``SequenceStatisticsInc`` typedefs.

One underlying 1-D statistics object per dimension; its inspectors are
lifted to return per-component lists. The cross-dimensional covariance and
correlation come from a quadratic sum accumulated as an outer product on
each ``add``.

Two things the C++ template does implicitly and Python must do explicitly:

* **the element type.** C++ writes ``GenericSequenceStatistics<S>``; the
  port carries the element class in the ``statistics_type`` class attribute
  — the analogue of C++'s public ``typedef StatisticsType
  statistics_type`` — which the concrete subclasses set. As in C++, the
  unparameterised ``GenericSequenceStatistics`` is not usable on its own.

* **which lifted methods exist.** C++ *declares* every lifted method on the
  template but only instantiates the ones you call, so
  ``SequenceStatisticsInc::percentile`` simply never compiles. Python has
  no lazy instantiation, so the methods that need the risk surface
  (``semiVariance``, ``percentile``, ``valueAtRisk``, the ``gaussian*``
  family, …) live on :class:`SequenceStatistics`, whose element type is
  ``Statistics``; :class:`GenericSequenceStatistics` carries only the
  surface both element types share.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.math.statistics.statistics import Statistics


class SequenceElementStatistics(Protocol):
    """The 1-D surface ``GenericSequenceStatistics`` lifts unconditionally.

    Every method here is present on both C++ element types
    (``Statistics`` and ``IncrementalStatistics``).
    """

    def add(self, value: float, weight: float = ...) -> None: ...
    def reset(self) -> None: ...
    def samples(self) -> int: ...
    def weight_sum(self) -> float: ...
    def mean(self) -> float: ...
    def variance(self) -> float: ...
    def standard_deviation(self) -> float: ...
    def error_estimate(self) -> float: ...
    def skewness(self) -> float: ...
    def kurtosis(self) -> float: ...
    def min(self) -> float: ...
    def max(self) -> float: ...
    def downside_variance(self) -> float: ...
    def downside_deviation(self) -> float: ...


class GenericSequenceStatistics[S: SequenceElementStatistics]:
    """Statistics of N-dimensional (sequence) data.

    # C++ parity: ``GenericSequenceStatistics``
    # (sequencestatistics.hpp:49-148).
    """

    # C++ parity: ``typedef StatisticsType statistics_type``
    # (sequencestatistics.hpp:53), here also the factory for the components.
    statistics_type: type[S]

    def __init__(self, dimension: int = 0) -> None:
        self._dimension: int = 0
        self._stats: list[S] = []
        self._quadratic_sum: Matrix = np.zeros((0, 0), dtype=np.float64)
        self.reset(dimension)

    # --- inspectors -----------------------------------------------------

    def size(self) -> int:
        """Number of dimensions.

        # C++ parity: sequencestatistics.hpp:59.
        """
        return self._dimension

    def samples(self) -> int:
        """Number of samples, taken from the first component.

        # C++ parity: sequencestatistics.hpp:164-167.
        """
        return 0 if not self._stats else self._stats[0].samples()

    def weight_sum(self) -> float:
        """Sum of the sample weights, taken from the first component.

        # C++ parity: sequencestatistics.hpp:169-172.
        """
        return 0.0 if not self._stats else self._stats[0].weight_sum()

    # --- N-D inspectors lifted from the underlying statistics ------------
    #
    # C++ parity: the DEFINE_SEQUENCE_STAT_CONST_METHOD_VOID macro block
    # (sequencestatistics.hpp:178-198).

    def mean(self) -> list[float]:
        """Per-component means."""
        return [self._stats[i].mean() for i in range(self._dimension)]

    def variance(self) -> list[float]:
        """Per-component variances."""
        return [self._stats[i].variance() for i in range(self._dimension)]

    def standard_deviation(self) -> list[float]:
        """Per-component standard deviations."""
        return [self._stats[i].standard_deviation() for i in range(self._dimension)]

    def downside_variance(self) -> list[float]:
        """Per-component downside variances."""
        return [self._stats[i].downside_variance() for i in range(self._dimension)]

    def downside_deviation(self) -> list[float]:
        """Per-component downside deviations."""
        return [self._stats[i].downside_deviation() for i in range(self._dimension)]

    def error_estimate(self) -> list[float]:
        """Per-component error estimates."""
        return [self._stats[i].error_estimate() for i in range(self._dimension)]

    def skewness(self) -> list[float]:
        """Per-component skewness."""
        return [self._stats[i].skewness() for i in range(self._dimension)]

    def kurtosis(self) -> list[float]:
        """Per-component kurtosis."""
        return [self._stats[i].kurtosis() for i in range(self._dimension)]

    def min(self) -> list[float]:
        """Per-component minima."""
        return [self._stats[i].min() for i in range(self._dimension)]

    def max(self) -> list[float]:
        """Per-component maxima."""
        return [self._stats[i].max() for i in range(self._dimension)]

    # --- covariance and correlation -------------------------------------

    def covariance(self) -> Matrix:
        """The covariance matrix.

        # C++ parity: sequencestatistics.hpp:246-265.
        """
        sample_weight = self.weight_sum()
        qassert.require(sample_weight > 0.0, "sampleWeight=0, unsufficient")
        sample_number = float(self.samples())
        qassert.require(sample_number > 1.0, "sample number <=1, unsufficient")

        m = np.asarray(self.mean(), dtype=np.float64)
        inv = 1.0 / sample_weight
        result = inv * self._quadratic_sum
        result = result - np.outer(m, m)
        result = result * (sample_number / (sample_number - 1.0))
        return result

    def correlation(self) -> Matrix:
        """The correlation matrix.

        # C++ parity: sequencestatistics.hpp:268-295.
        """
        correlation = self.covariance()
        # C++ ``correlation.diagonal()`` returns an Array (a copy); numpy's
        # ``np.diagonal`` returns a read-only *view* aliasing ``correlation``,
        # so it must be copied before the in-place rescale below mutates the
        # diagonal (otherwise variance reads get corrupted).
        variances = np.diagonal(correlation).copy()
        for i in range(self._dimension):
            for j in range(self._dimension):
                if i == j:
                    if variances[i] == 0.0:
                        correlation[i, j] = 1.0
                    else:
                        correlation[i, j] *= 1.0 / math.sqrt(variances[i] * variances[j])
                elif variances[i] == 0.0 and variances[j] == 0.0:
                    correlation[i, j] = 1.0
                elif variances[i] == 0.0 or variances[j] == 0.0:
                    correlation[i, j] = 0.0
                else:
                    correlation[i, j] *= 1.0 / math.sqrt(variances[i] * variances[j])
        return correlation

    # --- modifiers ------------------------------------------------------

    def reset(self, dimension: int = 0) -> None:
        """(Re-)initialize to ``dimension`` empty components.

        # C++ parity: sequencestatistics.hpp:228-244.
        """
        if dimension > 0:
            if dimension == self._dimension:
                for s in self._stats:
                    s.reset()
            else:
                if not hasattr(self, "statistics_type"):
                    qassert.fail(
                        "GenericSequenceStatistics has no element type: subclass it and set "
                        "statistics_type, as SequenceStatistics / SequenceStatisticsInc do "
                        "(C++ cannot name the template without its argument either)"
                    )
                self._dimension = dimension
                self._stats = [self.statistics_type() for _ in range(dimension)]
            self._quadratic_sum = np.zeros((self._dimension, self._dimension), dtype=np.float64)
        else:
            self._dimension = dimension

    def add(self, sample: Sequence[float], weight: float = 1.0) -> None:
        """Add a sequence sample, auto-sizing on the first add.

        # C++ parity: sequencestatistics.hpp:114-141.
        """
        if self._dimension == 0:
            # stat wasn't initialized yet
            qassert.require(len(sample) > 0, "sample error: end<=begin")
            self.reset(len(sample))

        qassert.require(
            len(sample) == self._dimension,
            f"sample size mismatch: {self._dimension} required, {len(sample)} provided",
        )

        arr = np.asarray(sample, dtype=np.float64)
        self._quadratic_sum = self._quadratic_sum + weight * np.outer(arr, arr)
        for i in range(self._dimension):
            self._stats[i].add(sample[i], weight)


class SequenceStatistics(GenericSequenceStatistics[Statistics]):
    """Default multi-dimensional statistics tool.

    # C++ parity: ``typedef GenericSequenceStatistics<Statistics>
    # SequenceStatistics`` (sequencestatistics.hpp:154).

    Adds the lifted methods that need the ``Statistics`` risk surface —
    the ones C++ declares on the template but can only instantiate for an
    element type that has them.
    """

    statistics_type = Statistics

    # --- lifted, void argument list --------------------------------------

    def semi_variance(self) -> list[float]:
        """Per-component semi-variances."""
        return [self._stats[i].semi_variance() for i in range(self._dimension)]

    def semi_deviation(self) -> list[float]:
        """Per-component semi-deviations."""
        return [self._stats[i].semi_deviation() for i in range(self._dimension)]

    # --- lifted, single argument -----------------------------------------
    #
    # C++ parity: the DEFINE_SEQUENCE_STAT_CONST_METHOD_DOUBLE macro block
    # (sequencestatistics.hpp:202-225).

    def percentile(self, y: float) -> list[float]:
        """Per-component percentiles."""
        return [self._stats[i].percentile(y) for i in range(self._dimension)]

    def gaussian_percentile(self, y: float) -> list[float]:
        """Per-component Gaussian percentiles."""
        return [self._stats[i].gaussian_percentile(y) for i in range(self._dimension)]

    def potential_upside(self, percentile: float) -> list[float]:
        """Per-component potential upside."""
        return [self._stats[i].potential_upside(percentile) for i in range(self._dimension)]

    def gaussian_potential_upside(self, percentile: float) -> list[float]:
        """Per-component Gaussian potential upside."""
        return [
            self._stats[i].gaussian_potential_upside(percentile) for i in range(self._dimension)
        ]

    def value_at_risk(self, percentile: float) -> list[float]:
        """Per-component value-at-risk."""
        return [self._stats[i].value_at_risk(percentile) for i in range(self._dimension)]

    def gaussian_value_at_risk(self, percentile: float) -> list[float]:
        """Per-component Gaussian value-at-risk."""
        return [self._stats[i].gaussian_value_at_risk(percentile) for i in range(self._dimension)]

    def expected_shortfall(self, percentile: float) -> list[float]:
        """Per-component expected shortfall."""
        return [self._stats[i].expected_shortfall(percentile) for i in range(self._dimension)]

    def gaussian_expected_shortfall(self, percentile: float) -> list[float]:
        """Per-component Gaussian expected shortfall."""
        return [
            self._stats[i].gaussian_expected_shortfall(percentile)
            for i in range(self._dimension)
        ]

    def regret(self, target: float) -> list[float]:
        """Per-component regret."""
        return [self._stats[i].regret(target) for i in range(self._dimension)]

    def shortfall(self, target: float) -> list[float]:
        """Per-component shortfall."""
        return [self._stats[i].shortfall(target) for i in range(self._dimension)]

    def gaussian_shortfall(self, target: float) -> list[float]:
        """Per-component Gaussian shortfall."""
        return [self._stats[i].gaussian_shortfall(target) for i in range(self._dimension)]

    def average_shortfall(self, target: float) -> list[float]:
        """Per-component average shortfall."""
        return [self._stats[i].average_shortfall(target) for i in range(self._dimension)]

    def gaussian_average_shortfall(self, target: float) -> list[float]:
        """Per-component Gaussian average shortfall."""
        return [
            self._stats[i].gaussian_average_shortfall(target) for i in range(self._dimension)
        ]


class SequenceStatisticsInc(GenericSequenceStatistics[IncrementalStatistics]):
    """Multi-dimensional statistics over ``IncrementalStatistics``.

    # C++ parity: ``typedef GenericSequenceStatistics<IncrementalStatistics>
    # SequenceStatisticsInc`` (sequencestatistics.hpp:155).
    """

    statistics_type = IncrementalStatistics
