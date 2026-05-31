"""SequenceStatistics — N-dimensional (sequence) statistics aggregator.

# C++ parity: ql/math/statistics/sequencestatistics.hpp (v1.42.1) —
# ``GenericSequenceStatistics<StatisticsType>`` + the
# ``SequenceStatistics = GenericSequenceStatistics<Statistics>`` typedef.

Wraps one underlying 1-D statistics object per dimension and lifts its
inspectors to return per-component vectors, plus the cross-dimensional
covariance / correlation matrices (accumulated via an outer-product
quadratic-sum on each ``add``).

PQuantLib divergence: the C++ ``Statistics`` typedef is
``RiskStatistics`` (GenericRiskStatistics over GeneralStatistics) which adds
Gaussian / risk inspectors (VaR, expected shortfall, …). PQuantLib's
``GeneralStatistics`` covers the moment + percentile surface used by the
market-model historical-analysis drivers (mean / variance / standard
deviation / skewness / kurtosis / min / max / percentile / error estimate);
the risk inspectors are out of scope here and are not lifted. The
covariance / correlation matrices — the only part the W9-B
``HistoricalForwardRatesAnalysis`` / ``historical_rates_analysis`` actually
consume — match C++ exactly.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.math.statistics.general_statistics import GeneralStatistics


class SequenceStatistics:
    """N-dimensional statistics over ``GeneralStatistics`` components.

    # C++ parity: GenericSequenceStatistics<Statistics> (the
    SequenceStatistics typedef).
    """

    def __init__(self, dimension: int = 0) -> None:
        self._dimension = 0
        self._stats: list[GeneralStatistics] = []
        self._quadratic_sum: Matrix = np.zeros((0, 0), dtype=np.float64)
        self.reset(dimension)

    # --- inspectors -----------------------------------------------------

    def size(self) -> int:
        """Number of dimensions.

        # C++ parity: GenericSequenceStatistics::size.
        """
        return self._dimension

    def samples(self) -> int:
        """Number of samples (from the first component).

        # C++ parity: GenericSequenceStatistics::samples.
        """
        return 0 if not self._stats else self._stats[0].samples()

    def weight_sum(self) -> float:
        """Sum of sample weights (from the first component).

        # C++ parity: GenericSequenceStatistics::weightSum.
        """
        return 0.0 if not self._stats else self._stats[0].weight_sum()

    # --- N-D inspectors lifted from the underlying statistics ------------

    def mean(self) -> list[float]:
        """Per-component means.

        # C++ parity: GenericSequenceStatistics::mean.
        """
        return [self._stats[i].mean() for i in range(self._dimension)]

    def variance(self) -> list[float]:
        """Per-component variances.

        # C++ parity: GenericSequenceStatistics::variance.
        """
        return [self._stats[i].variance() for i in range(self._dimension)]

    def standard_deviation(self) -> list[float]:
        """Per-component standard deviations.

        # C++ parity: GenericSequenceStatistics::standardDeviation.
        """
        return [self._stats[i].standard_deviation() for i in range(self._dimension)]

    def error_estimate(self) -> list[float]:
        """Per-component error estimates.

        # C++ parity: GenericSequenceStatistics::errorEstimate.
        """
        return [self._stats[i].error_estimate() for i in range(self._dimension)]

    def skewness(self) -> list[float]:
        """Per-component skewness.

        # C++ parity: GenericSequenceStatistics::skewness.
        """
        return [self._stats[i].skewness() for i in range(self._dimension)]

    def kurtosis(self) -> list[float]:
        """Per-component kurtosis.

        # C++ parity: GenericSequenceStatistics::kurtosis.
        """
        return [self._stats[i].kurtosis() for i in range(self._dimension)]

    def min(self) -> list[float]:
        """Per-component minima.

        # C++ parity: GenericSequenceStatistics::min.
        """
        return [self._stats[i].min() for i in range(self._dimension)]

    def max(self) -> list[float]:
        """Per-component maxima.

        # C++ parity: GenericSequenceStatistics::max.
        """
        return [self._stats[i].max() for i in range(self._dimension)]

    def percentile(self, y: float) -> list[float]:
        """Per-component percentiles.

        # C++ parity: GenericSequenceStatistics::percentile.
        """
        return [self._stats[i].percentile(y) for i in range(self._dimension)]

    # --- covariance and correlation -------------------------------------

    def covariance(self) -> Matrix:
        """The covariance matrix.

        # C++ parity: GenericSequenceStatistics::covariance.
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

        # C++ parity: GenericSequenceStatistics::correlation.
        """
        correlation = self.covariance()
        # C++ parity: ``correlation.diagonal()`` returns an Array (a copy);
        # numpy's ``np.diagonal`` returns a read-only *view* aliasing
        # ``correlation``, so it must be copied before the in-place rescale
        # below mutates the diagonal (otherwise variance reads get corrupted).
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

        # C++ parity: GenericSequenceStatistics::reset.
        """
        if dimension > 0:
            if dimension == self._dimension:
                for s in self._stats:
                    s.reset()
            else:
                self._dimension = dimension
                self._stats = [GeneralStatistics() for _ in range(dimension)]
            self._quadratic_sum = np.zeros((self._dimension, self._dimension), dtype=np.float64)
        else:
            self._dimension = dimension

    def add(self, sample: list[float], weight: float = 1.0) -> None:
        """Add a sequence ``sample`` (auto-sizing on the first add).

        # C++ parity: GenericSequenceStatistics::add (Sequence overload,
        which forwards to the iterator overload).
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
