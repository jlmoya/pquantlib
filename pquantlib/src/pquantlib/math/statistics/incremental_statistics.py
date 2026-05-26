"""Incremental statistics aggregator — Welford-style online algorithm.

# C++ parity: ql/math/statistics/incrementalstatistics.hpp +
#             ql/math/statistics/incrementalstatistics.cpp (v1.42.1).

The C++ class is a thin wrapper over boost::accumulators with weighted
mean/variance/skewness/kurtosis tags. PQuantLib replaces the
boost-accumulator backbone with a Welford-style online aggregator
(Knuth, TAOCP vol. 2 §4.2.2) for mean and variance, keeping memory O(1)
and numerically stable for large sample counts.

Per L1-B design (phase1-l1-B-design.md), this initial port covers the
core accessors used in tests:
``samples()``, ``weight_sum()``, ``mean()``, ``variance()``,
``standard_deviation()``, ``error_estimate()``, ``min()``, ``max()``,
plus the downside-statistics accessors. ``skewness()`` and ``kurtosis()``
are deferred to a follow-up cluster — the boost weighted-skewness /
weighted-kurtosis math has subtle numerical quirks that warrant a
separate cross-validated batch.
"""

from __future__ import annotations

import math

from pquantlib import qassert


class IncrementalStatistics:
    """Online aggregator for weighted samples (mean / variance / min / max).

    Uses Welford's algorithm extended to weights:
        M_n     = M_{n-1} + (w / W_n) * (x - M_{n-1})
        S_n     = S_{n-1} + w * (x - M_{n-1}) * (x - M_n)
    where ``W_n`` is the running weight sum. The unbiased sample variance
    matches the C++ definition: ``s2 * N / (N - 1)``.
    """

    __slots__ = (
        "_count",
        "_down_count",
        "_down_moment2",
        "_down_weight_sum",
        "_m2",
        "_max",
        "_mean",
        "_min",
        "_weight_sum",
    )

    def __init__(self) -> None:
        self._count: int = 0
        self._weight_sum: float = 0.0
        self._mean: float = 0.0
        self._m2: float = 0.0  # running sum of squared deviations (weighted)
        self._min: float = math.inf
        self._max: float = -math.inf
        # Downside (value < 0) running stats.
        self._down_count: int = 0
        self._down_weight_sum: float = 0.0
        self._down_moment2: float = 0.0  # weighted second moment about 0

    # --- Modifiers ------------------------------------------------------

    def add(self, value: float, weight: float = 1.0) -> None:
        qassert.require(weight >= 0.0, f"negative weight ({weight}) not allowed")
        if weight == 0.0:
            return
        self._count += 1
        new_weight_sum = self._weight_sum + weight
        delta = value - self._mean
        # Weighted Welford update.
        self._mean += (weight / new_weight_sum) * delta
        self._m2 += weight * delta * (value - self._mean)
        self._weight_sum = new_weight_sum

        self._min = min(self._min, value)
        self._max = max(self._max, value)

        if value < 0.0:
            self._down_count += 1
            self._down_weight_sum += weight
            self._down_moment2 += weight * value * value

    def reset(self) -> None:
        self._count = 0
        self._weight_sum = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = math.inf
        self._max = -math.inf
        self._down_count = 0
        self._down_weight_sum = 0.0
        self._down_moment2 = 0.0

    # --- Inspectors -----------------------------------------------------

    def samples(self) -> int:
        return self._count

    def weight_sum(self) -> float:
        return self._weight_sum

    def mean(self) -> float:
        qassert.require(self._weight_sum > 0.0, "sampleWeight_= 0, unsufficient")
        return self._mean

    def variance(self) -> float:
        qassert.require(self._weight_sum > 0.0, "sampleWeight_= 0, unsufficient")
        qassert.require(self._count > 1, "sample number <= 1, unsufficient")
        # Weighted-variance estimator: second-central moment normalized by
        # weight_sum, then debiased by N/(N-1) to mirror the C++ output of
        # ``weighted_variance`` * ``N/(N-1)``.
        weighted_var = self._m2 / self._weight_sum
        n = float(self._count)
        return weighted_var * n / (n - 1.0)

    def standard_deviation(self) -> float:
        return math.sqrt(self.variance())

    def error_estimate(self) -> float:
        return math.sqrt(self.variance() / self._count)

    def min(self) -> float:
        qassert.require(self._count > 0, "empty sample set")
        return self._min

    def max(self) -> float:
        qassert.require(self._count > 0, "empty sample set")
        return self._max

    def downside_samples(self) -> int:
        return self._down_count

    def downside_weight_sum(self) -> float:
        return self._down_weight_sum

    def downside_variance(self) -> float:
        qassert.require(self._down_weight_sum > 0.0, "sampleWeight_= 0, unsufficient")
        qassert.require(self._down_count > 1, "sample number <= 1, unsufficient")
        # Mirrors the C++ downsideVariance: r1 * E[x^2 | x<0] = (N/(N-1)) *
        # (sum_{x<0} w*x^2 / sum_{x<0} w).
        n = float(self._down_count)
        return (n / (n - 1.0)) * (self._down_moment2 / self._down_weight_sum)

    def downside_deviation(self) -> float:
        return math.sqrt(self.downside_variance())
