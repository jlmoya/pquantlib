"""Incremental statistics aggregator.

# C++ parity: ql/math/statistics/incrementalstatistics.hpp +
#             ql/math/statistics/incrementalstatistics.cpp (v1.43).

The C++ class is a thin wrapper over ``boost::accumulators`` with the
``weighted_mean`` / ``weighted_variance`` / ``weighted_skewness`` /
``weighted_kurtosis`` / ``weighted_moment<2>`` tags. Because "the boost
accumulator computes the mean" is not one algorithm but four different
ones, the port transcribes each accumulator's own recurrence rather than
picking a single online scheme:

* ``tag::weighted_mean`` is boost's **lazy** mean —
  ``weighted_sum / sum_of_weights`` with ``weighted_sum`` a plain running
  ``sum += w*x`` (boost/accumulators/statistics/weighted_mean.hpp,
  ``weighted_mean_impl::result``). This is what ``mean()`` returns.

* ``tag::weighted_variance`` is boost's **immediate** (iterative) variance,
  which depends on a *second*, separately maintained mean accumulator,
  ``immediate_weighted_mean``
  (boost/accumulators/statistics/weighted_variance.hpp:70-115 +
  weighted_mean.hpp ``immediate_weighted_mean_impl::operator()``)::

      mean_n = (mean_{n-1}*(W_n - w_n) + x_n*w_n) / W_n
      var_n  = var_{n-1}*(W_n - w_n)/W_n + (x_n - mean_n)^2 * w_n/(W_n - w_n)

  with ``var`` left untouched for the first sample. So the port keeps both
  means: the lazy one is what ``mean()`` reports, the immediate one is what
  the variance recurrence consumes. Algebraically the two agree; in
  floating point they do not, and the C++ test-suite's numerical-stability
  case (samples ~1e8 with variance 1e-2) depends on the recurrence being
  the immediate one.

* ``tag::weighted_skewness`` / ``tag::weighted_kurtosis`` are boost's
  **lazy** estimators, written in raw weighted moments
  ``m_k = sum(w*x^k)/sum(w)`` and the *lazy* mean
  (weighted_skewness.hpp:50-68, weighted_kurtosis.hpp:52-72). Note that
  boost's ``numeric::pow`` is binary exponentiation, so ``x^3`` is
  ``(x*x)*x`` and ``x^4`` is ``(x*x)*(x*x)`` — transcribed as such.

* the downside accumulator carries ``weighted_moment<2>`` only, i.e.
  ``sum(w*(x*x))/sum(w)`` over the samples with ``x < 0``.

A zero-weight sample still increments ``count`` (and updates min/max) on
the boost side, so it does here too.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from pquantlib import qassert


def _fdiv(numerator: float, denominator: float) -> float:
    """IEEE-754 division, matching boost's ``numeric::fdiv`` on doubles.

    The accumulator recurrences divide by a running weight sum that is
    legitimately zero while every weight seen so far is zero. C++ yields
    NaN or an infinity there; Python raises ``ZeroDivisionError``, so the
    C++ result is reproduced explicitly.
    """
    if denominator == 0.0:
        return math.nan if numerator == 0.0 else math.copysign(math.inf, numerator)
    return numerator / denominator


class IncrementalStatistics:
    """Online aggregator for weighted samples.

    # C++ parity: ``class IncrementalStatistics``
    # (incrementalstatistics.hpp:53-171).
    """

    _count: int
    _weight_sum: float
    _weighted_sum: float
    _immediate_mean: float
    _variance: float
    _moment2_sum: float
    _moment3_sum: float
    _moment4_sum: float
    _min: float
    _max: float
    _down_count: int
    _down_weight_sum: float
    _down_moment2_sum: float

    __slots__ = (
        "_count",
        "_down_count",
        "_down_moment2_sum",
        "_down_weight_sum",
        "_immediate_mean",
        "_max",
        "_min",
        "_moment2_sum",
        "_moment3_sum",
        "_moment4_sum",
        "_variance",
        "_weight_sum",
        "_weighted_sum",
    )

    def __init__(self) -> None:
        self.reset()

    # --- Modifiers ------------------------------------------------------

    def add(self, value: float, weight: float = 1.0) -> None:
        """Add a datum, optionally weighted.

        # C++ parity: incrementalstatistics.cpp:126-132, expanded into the
        # individual boost accumulator updates (which run in dependency
        # order: count, sum_of_weights, weighted_sum, immediate mean, raw
        # moments, then the variance recurrence).
        """
        qassert.require(weight >= 0.0, f"negative weight ({weight}) not allowed")

        self._count += 1
        w_sum_prev = self._weight_sum
        self._weight_sum = w_sum_prev + weight

        # tag::weighted_sum — plain running sum.
        self._weighted_sum += weight * value

        # tag::weighted_moment<N> — sum += w * pow(x, N), binary exponentiation.
        x2 = value * value
        x3 = x2 * value
        x4 = x2 * x2
        self._moment2_sum += weight * x2
        self._moment3_sum += weight * x3
        self._moment4_sum += weight * x4

        # tag::immediate_weighted_mean
        self._immediate_mean = _fdiv(
            self._immediate_mean * w_sum_prev + value * weight, self._weight_sum
        )

        # tag::weighted_variance — only from the second sample on.
        if self._count > 1:
            tmp = value - self._immediate_mean
            self._variance = _fdiv(self._variance * w_sum_prev, self._weight_sum) + _fdiv(
                tmp * tmp * weight, w_sum_prev
            )

        self._min = min(self._min, value)
        self._max = max(self._max, value)

        if value < 0.0:
            self._down_count += 1
            self._down_weight_sum += weight
            self._down_moment2_sum += weight * x2

    def reset(self) -> None:
        """Drop every sample.

        # C++ parity: incrementalstatistics.cpp:134-137.
        """
        self._count = 0
        self._weight_sum = 0.0
        self._weighted_sum = 0.0
        self._immediate_mean = 0.0
        self._variance = 0.0
        self._moment2_sum = 0.0
        self._moment3_sum = 0.0
        self._moment4_sum = 0.0
        self._min = math.inf
        self._max = -math.inf
        self._down_count = 0
        self._down_weight_sum = 0.0
        self._down_moment2_sum = 0.0

    def add_sequence(
        self, values: Iterable[float], weights: Iterable[float] | None = None
    ) -> None:
        """Add a sequence of data, with default or per-sample weights.

        # C++ parity: ``addSequence`` overloads
        # (incrementalstatistics.hpp:135-147).
        """
        if weights is None:
            for v in values:
                self.add(v)
        else:
            for v, w in zip(values, weights, strict=False):
                self.add(v, w)

    # --- Inspectors -----------------------------------------------------

    def samples(self) -> int:
        """Number of samples collected (zero-weight ones included)."""
        return self._count

    def weight_sum(self) -> float:
        """Sum of the sample weights."""
        return self._weight_sum

    def mean(self) -> float:
        """Weighted mean, boost's lazy ``weighted_sum / sum_of_weights``.

        # C++ parity: incrementalstatistics.cpp:41-45.
        """
        qassert.require(self._weight_sum > 0.0, "sampleWeight_= 0, unsufficient")
        return self._weighted_sum / self._weight_sum

    def variance(self) -> float:
        """Unbiased weighted variance.

        # C++ parity: incrementalstatistics.cpp:47-54 —
        # ``N/(N-1) * weighted_variance``.
        """
        qassert.require(self._weight_sum > 0.0, "sampleWeight_= 0, unsufficient")
        qassert.require(self._count > 1, "sample number <= 1, unsufficient")
        n = float(self._count)
        return n / (n - 1.0) * self._variance

    def standard_deviation(self) -> float:
        """Square root of :meth:`variance`."""
        return math.sqrt(self.variance())

    def error_estimate(self) -> float:
        """Standard error of the mean.

        # C++ parity: incrementalstatistics.cpp:60-62.
        """
        return math.sqrt(self.variance() / self._count)

    def _lazy_mean(self) -> float:
        """boost ``tag::weighted_mean``, used by skewness and kurtosis."""
        return self._weighted_sum / self._weight_sum

    def _weighted_moment(self, order: int) -> float:
        """boost ``tag::weighted_moment<order>`` = ``sum(w*x^order)/sum(w)``."""
        if order == 2:
            return self._moment2_sum / self._weight_sum
        if order == 3:
            return self._moment3_sum / self._weight_sum
        return self._moment4_sum / self._weight_sum

    def skewness(self) -> float:
        """Unbiased weighted skewness.

        # C++ parity: incrementalstatistics.cpp:64-72, over boost's lazy
        # ``weighted_skewness`` (weighted_skewness.hpp:60-68).
        """
        qassert.require(self._count > 2, "sample number <= 2, unsufficient")
        m1 = self._lazy_mean()
        m2 = self._weighted_moment(2)
        m3 = self._weighted_moment(3)
        central2 = m2 - m1 * m1
        weighted_skewness = (m3 - 3.0 * m2 * m1 + 2.0 * m1 * m1 * m1) / (
            central2 * math.sqrt(central2)
        )
        n = float(self._count)
        r1 = n / (n - 2.0)
        r2 = (n - 1.0) / (n - 2.0)
        return math.sqrt(r1 * r2) * weighted_skewness

    def kurtosis(self) -> float:
        """Unbiased weighted excess kurtosis.

        # C++ parity: incrementalstatistics.cpp:74-88, over boost's lazy
        # ``weighted_kurtosis`` (weighted_kurtosis.hpp:64-72).
        """
        qassert.require(self._count > 3, "sample number <= 3, unsufficient")
        m1 = self._lazy_mean()
        m2 = self._weighted_moment(2)
        m3 = self._weighted_moment(3)
        m4 = self._weighted_moment(4)
        central2 = m2 - m1 * m1
        weighted_kurtosis = (
            m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 * m1 - 3.0 * m1 * m1 * m1 * m1
        ) / (central2 * central2) - 3.0
        n = float(self._count)
        r1 = (n - 1.0) / (n - 2.0)
        r2 = (n + 1.0) / (n - 3.0)
        r3 = (n - 1.0) / (n - 3.0)
        return ((3.0 + weighted_kurtosis) * r2 - 3.0 * r3) * r1

    def min(self) -> float:
        """Smallest sample value."""
        qassert.require(self._count > 0, "empty sample set")
        return self._min

    def max(self) -> float:
        """Largest sample value."""
        qassert.require(self._count > 0, "empty sample set")
        return self._max

    def downside_samples(self) -> int:
        """Number of samples strictly below zero."""
        return self._down_count

    def downside_weight_sum(self) -> float:
        """Sum of the weights of the samples strictly below zero."""
        return self._down_weight_sum

    def downside_variance(self) -> float:
        """Downside variance.

        # C++ parity: incrementalstatistics.cpp:112-120 —
        # ``N/(N-1) * weighted_moment<2>`` over the downside accumulator.
        """
        qassert.require(self._down_weight_sum > 0.0, "sampleWeight_= 0, unsufficient")
        qassert.require(self._down_count > 1, "sample number <= 1, unsufficient")
        n = float(self._down_count)
        r1 = n / (n - 1.0)
        return r1 * (self._down_moment2_sum / self._down_weight_sum)

    def downside_deviation(self) -> float:
        """Square root of :meth:`downside_variance`."""
        return math.sqrt(self.downside_variance())
