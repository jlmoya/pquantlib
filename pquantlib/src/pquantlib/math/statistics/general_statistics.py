"""General statistics aggregator (stores all samples).

# C++ parity: ql/math/statistics/generalstatistics.hpp +
#             ql/math/statistics/generalstatistics.cpp (v1.43).

Accumulates (value, weight) pairs and exposes empirical moments
(mean / variance / skewness / kurtosis), min / max, and quantile lookups.
It stores all samples, so it does not suffer the numerical instability
of an online aggregator, at the cost of O(N) memory.

``expectation_value`` is the primitive every risk measure in
``GenericRiskStatistics`` is expressed through: it averages ``f`` over the
subset of samples selected by a range predicate, and reports how many
samples fell in the range.

``GeneralStatistics::reserve(Size)`` has no Python counterpart (``list``
grows amortised-O(1) and exposes no capacity hint) and is not ported.
"""

from __future__ import annotations

import builtins
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Final

from pquantlib import qassert

# C++ parity: ql/utilities/null.hpp — ``Null<Real>()`` is
# ``std::numeric_limits<float>::max()``, the sentinel ``expectationValue``
# returns when no sample falls in the requested range.
NULL_REAL: Final[float] = 3.4028234663852886e38


class GeneralStatistics:
    """Mirrors C++ ``GeneralStatistics`` — stores all (value, weight) pairs."""

    __slots__ = ("_samples", "_sorted")

    def __init__(self) -> None:
        self._samples: list[tuple[float, float]] = []
        self._sorted: bool = True

    # --- Modifiers ------------------------------------------------------

    def add(self, value: float, weight: float = 1.0) -> None:
        """Add a datum, optionally weighted.

        # C++ parity: generalstatistics.hpp:232-236.
        """
        qassert.require(weight >= 0.0, "negative weight not allowed")
        self._samples.append((value, weight))
        self._sorted = False

    def add_sequence(
        self, values: Iterable[float], weights: Iterable[float] | None = None
    ) -> None:
        """Add a sequence of data, with default or per-sample weights.

        # C++ parity: ``addSequence(begin, end)`` and
        # ``addSequence(begin, end, wbegin)`` (generalstatistics.hpp:168-179).
        """
        if weights is None:
            for v in values:
                self.add(v)
        else:
            for v, w in zip(values, weights, strict=False):
                self.add(v, w)

    def reset(self) -> None:
        """Drop every sample.

        # C++ parity: generalstatistics.hpp:238-241.
        """
        self._samples = []
        self._sorted = True

    def sort(self) -> None:
        """Sort the sample set in increasing order.

        # C++ parity: generalstatistics.hpp:247-252 — ``std::sort`` over
        # ``std::pair<Real,Real>``, i.e. lexicographic on (value, weight),
        # not on the value alone.
        """
        if not self._sorted:
            self._samples.sort()
            self._sorted = True

    # --- Inspectors -----------------------------------------------------

    def samples(self) -> int:
        """Number of samples collected."""
        return len(self._samples)

    def data(self) -> list[tuple[float, float]]:
        """The collected (value, weight) pairs, in storage order."""
        return self._samples

    def weight_sum(self) -> float:
        """Sum of the sample weights.

        # C++ parity: generalstatistics.cpp:25-32 — a plain running sum in
        # insertion order.
        """
        result = 0.0
        for _, w in self._samples:
            result += w
        return result

    def expectation_value(
        self,
        f: Callable[[float], float],
        in_range: Callable[[float], bool] | None = None,
    ) -> tuple[float, int]:
        """Weighted average of ``f`` over the samples selected by ``in_range``.

        # C++ parity: generalstatistics.hpp:115-142 — both
        # ``expectationValue`` overloads. ``in_range=None`` is the overload
        # that passes a predicate always returning ``true``.

        Returns:
            ``(value, count)``. When no sample falls in the range, C++
            returns ``(Null<Real>(), 0)``; the port returns
            ``(NULL_REAL, 0)`` with the same sentinel value.
        """
        num = 0.0
        den = 0.0
        n = 0
        for x, w in self._samples:
            if in_range is None or in_range(x):
                num += f(x) * w
                den += w
                n += 1
        if n == 0:
            return (NULL_REAL, 0)
        return (num / den, n)

    def mean(self) -> float:
        """Weighted mean.

        # C++ parity: generalstatistics.cpp:34-38.
        """
        n = self.samples()
        qassert.require(n != 0, "empty sample set")
        return self.expectation_value(lambda x: x)[0]

    def variance(self) -> float:
        """Unbiased weighted variance.

        # C++ parity: generalstatistics.cpp:40-52.
        """
        n = self.samples()
        qassert.require(n > 1, "sample number <=1, unsufficient")
        m = self.mean()
        s2 = self.expectation_value(lambda x: (x - m) * (x - m))[0]
        return s2 * n / (n - 1.0)

    def standard_deviation(self) -> float:
        """Square root of :meth:`variance`."""
        return math.sqrt(self.variance())

    def error_estimate(self) -> float:
        """Standard error of the mean, ``sigma / sqrt(N)``."""
        return math.sqrt(self.variance() / self.samples())

    def skewness(self) -> float:
        """Unbiased weighted skewness.

        # C++ parity: generalstatistics.cpp:54-67.
        """
        n = self.samples()
        qassert.require(n > 2, "sample number <=2, unsufficient")
        m = self.mean()
        x_moment = self.expectation_value(lambda x: (x - m) * (x - m) * (x - m))[0]
        sigma = self.standard_deviation()
        return (x_moment / (sigma * sigma * sigma)) * (n / (n - 1.0)) * (n / (n - 2.0))

    def kurtosis(self) -> float:
        """Unbiased weighted excess kurtosis.

        # C++ parity: generalstatistics.cpp:69-86.
        """
        n = self.samples()
        qassert.require(n > 3, "sample number <=3, unsufficient")
        m = self.mean()

        def _fourth(x: float) -> float:
            d = x - m
            d2 = d * d
            return d2 * d2

        x_moment = self.expectation_value(_fourth)[0]
        sigma2 = self.variance()
        c1 = (n / (n - 1.0)) * (n / (n - 2.0)) * ((n + 1.0) / (n - 3.0))
        c2 = 3.0 * ((n - 1.0) / (n - 2.0)) * ((n - 1.0) / (n - 3.0))
        return c1 * (x_moment / (sigma2 * sigma2)) - c2

    def min(self) -> float:
        """Smallest sample value.

        # C++ parity: generalstatistics.hpp:219-223 — ``std::min_element``
        # over the (value, weight) pairs, whose ordering is decided by the
        # value first.
        """
        qassert.require(self.samples() > 0, "empty sample set")
        return builtins.min(self._samples)[0]

    def max(self) -> float:
        """Largest sample value.

        # C++ parity: generalstatistics.hpp:225-229.
        """
        qassert.require(self.samples() > 0, "empty sample set")
        return builtins.max(self._samples)[0]

    def percentile(self, percent: float) -> float:
        """``percent``-th percentile of the empirical distribution.

        # C++ parity: generalstatistics.cpp:88-110 — walks the sorted
        # samples accumulating weight and stops at the *first* sample whose
        # cumulative weight reaches ``percent * weightSum()``; the walk is
        # additionally clamped at the last sample.
        """
        qassert.require(percent > 0.0 and percent <= 1.0, f"percentile ({percent}) must be in (0.0, 1.0]")
        sample_weight = self.weight_sum()
        qassert.require(sample_weight > 0.0, "empty sample set")

        self.sort()

        samples: Sequence[tuple[float, float]] = self._samples
        k = 0
        last = len(samples) - 1
        # the sum of weight is non null, therefore there's at least one sample
        integral = samples[k][1]
        target = percent * sample_weight
        while integral < target and k != last:
            k += 1
            integral += samples[k][1]
        return samples[k][0]

    def top_percentile(self, percent: float) -> float:
        """``percent``-th top percentile of the empirical distribution.

        # C++ parity: generalstatistics.cpp:112-134 — the same walk run over
        # a reverse iterator, clamped at the first sample.
        """
        qassert.require(percent > 0.0 and percent <= 1.0, f"percentile ({percent}) must be in (0.0, 1.0]")
        sample_weight = self.weight_sum()
        qassert.require(sample_weight > 0.0, "empty sample set")

        self.sort()

        samples: Sequence[tuple[float, float]] = self._samples
        k = len(samples) - 1
        last = 0
        integral = samples[k][1]
        target = percent * sample_weight
        while integral < target and k != last:
            k -= 1
            integral += samples[k][1]
        return samples[k][0]
