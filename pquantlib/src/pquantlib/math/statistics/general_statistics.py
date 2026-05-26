"""General statistics aggregator (stores all samples).

# C++ parity: ql/math/statistics/generalstatistics.hpp +
#             ql/math/statistics/generalstatistics.cpp (v1.42.1).

Accumulates (value, weight) pairs and exposes empirical moments
(mean / variance / skewness / kurtosis), min / max, and quantile lookups.
It stores all samples, so it does not suffer the numerical instability
of an online Welford aggregator, at the cost of O(N) memory.
"""

from __future__ import annotations

import builtins
import math

from pquantlib import qassert


class GeneralStatistics:
    """Mirrors C++ ``GeneralStatistics`` — stores all (value, weight) pairs."""

    __slots__ = ("_samples", "_sorted")

    def __init__(self) -> None:
        self._samples: list[tuple[float, float]] = []
        self._sorted: bool = True

    # --- Modifiers ------------------------------------------------------

    def add(self, value: float, weight: float = 1.0) -> None:
        qassert.require(weight >= 0.0, "negative weight not allowed")
        self._samples.append((value, weight))
        self._sorted = False

    def add_sequence(self, values: object) -> None:
        """Convenience for adding an iterable of values with default weight 1.

        Mirrors the C++ ``addSequence(begin, end)`` template overload.
        """
        for v in values:  # type: ignore[attr-defined]
            self.add(float(v))  # type: ignore[arg-type]

    def reset(self) -> None:
        self._samples = []
        self._sorted = True

    def sort(self) -> None:
        if not self._sorted:
            self._samples.sort(key=lambda pair: pair[0])
            self._sorted = True

    # --- Inspectors -----------------------------------------------------

    def samples(self) -> int:
        return len(self._samples)

    def data(self) -> list[tuple[float, float]]:
        return self._samples

    def weight_sum(self) -> float:
        return sum(w for _, w in self._samples)

    def mean(self) -> float:
        n = self.samples()
        qassert.require(n != 0, "empty sample set")
        num = 0.0
        den = 0.0
        for x, w in self._samples:
            num += x * w
            den += w
        return num / den

    def variance(self) -> float:
        n = self.samples()
        qassert.require(n > 1, "sample number <=1, unsufficient")
        m = self.mean()
        num = 0.0
        den = 0.0
        for x, w in self._samples:
            d = x - m
            num += d * d * w
            den += w
        s2 = num / den
        return s2 * n / (n - 1.0)

    def standard_deviation(self) -> float:
        return math.sqrt(self.variance())

    def error_estimate(self) -> float:
        return math.sqrt(self.variance() / self.samples())

    def skewness(self) -> float:
        n = self.samples()
        qassert.require(n > 2, "sample number <=2, unsufficient")
        m = self.mean()
        num = 0.0
        den = 0.0
        for x, w in self._samples:
            d = x - m
            num += d * d * d * w
            den += w
        x_moment = num / den
        sigma = self.standard_deviation()
        return (x_moment / (sigma * sigma * sigma)) * (n / (n - 1.0)) * (n / (n - 2.0))

    def kurtosis(self) -> float:
        n = self.samples()
        qassert.require(n > 3, "sample number <=3, unsufficient")
        m = self.mean()
        num = 0.0
        den = 0.0
        for x, w in self._samples:
            d = x - m
            d2 = d * d
            num += d2 * d2 * w
            den += w
        x_moment = num / den
        sigma2 = self.variance()
        c1 = (n / (n - 1.0)) * (n / (n - 2.0)) * ((n + 1.0) / (n - 3.0))
        c2 = 3.0 * ((n - 1.0) / (n - 2.0)) * ((n - 1.0) / (n - 3.0))
        return c1 * (x_moment / (sigma2 * sigma2)) - c2

    def min(self) -> float:
        qassert.require(self.samples() > 0, "empty sample set")
        return builtins.min(x for x, _ in self._samples)

    def max(self) -> float:
        qassert.require(self.samples() > 0, "empty sample set")
        return builtins.max(x for x, _ in self._samples)

    def percentile(self, percent: float) -> float:
        qassert.require(0.0 < percent <= 1.0, f"percentile ({percent}) must be in (0.0, 1.0]")
        sample_weight = self.weight_sum()
        qassert.require(sample_weight > 0.0, "empty sample set")
        self.sort()
        target = percent * sample_weight
        integral = 0.0
        last = self._samples[-1][0]
        for x, w in self._samples:
            integral += w
            if integral >= target:
                return x
        return last

    def top_percentile(self, percent: float) -> float:
        qassert.require(0.0 < percent <= 1.0, f"percentile ({percent}) must be in (0.0, 1.0]")
        sample_weight = self.weight_sum()
        qassert.require(sample_weight > 0.0, "empty sample set")
        self.sort()
        target = percent * sample_weight
        integral = 0.0
        first = self._samples[0][0]
        for x, w in reversed(self._samples):
            integral += w
            if integral >= target:
                return x
        return first
