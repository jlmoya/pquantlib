"""Rolling-window "constant" volatility estimator.

# C++ parity: ql/models/volatility/constantestimator.{hpp,cpp} @ v1.43.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.time.time_series import TimeSeries
from pquantlib.volatility_model import VolatilityCompositor


class ConstantEstimator(VolatilityCompositor):
    """Volatility over a trailing window of ``size`` observations.

    # C++ parity: ``class ConstantEstimator``
    # (ql/models/volatility/constantestimator.hpp:35-43,
    # constantestimator.cpp:23-44).

    Volatilities are expressed on an annual basis.

    The window statistic is NEITHER the biased nor the unbiased sample
    variance. C++ computes (constantestimator.cpp:38-39)::

        s = sqrt(sumu2/size - sumu*sumu/size/(size + 1))

    — note the ``size + 1`` in the second denominator, where a sample
    variance would have ``size`` (biased) or would divide the whole
    expression by ``size - 1`` (unbiased). That is reproduced verbatim,
    including the left-to-right division order.
    """

    __slots__ = ("_size",)

    def __init__(self, size: int) -> None:
        # C++ parity: constantestimator.hpp:39-40.
        self._size: int = size

    def size(self) -> int:
        """Window width. # C++ parity: the private ``size_`` member has no C++ inspector."""
        return self._size

    def calculate(self, volatility_series: TimeSeries[float]) -> TimeSeries[float]:
        """Rolling-window volatility, filed under the date AFTER the window.

        # C++ parity: ``ConstantEstimator::calculate``
        # (constantestimator.cpp:23-44).

        The output cursor is advanced by ``size`` before the loop
        (constantestimator.cpp:28-29) while the summation for output ``i``
        runs over inputs ``[i - size, i)``. So the estimate for date ``i``
        uses only observations STRICTLY BEFORE it, and the first ``size``
        dates carry no output at all.
        """
        u = volatility_series.values()
        dates = volatility_series.dates()
        n = len(u)
        # C++ parity divergence — a guard C++ does not have. C++ does
        # ``std::advance(cur, size_)`` (constantestimator.cpp:29) before
        # testing anything, which is undefined behaviour once ``size_``
        # exceeds the series length, and divides by ``size_`` unguarded,
        # which is 0/0 for a zero window. Both are diagnosed here instead.
        qassert.require(self._size > 0, "ConstantEstimator window size must be positive")
        qassert.require(
            self._size <= n,
            f"ConstantEstimator window size ({self._size}) exceeds series size ({n})",
        )

        retval: TimeSeries[float] = TimeSeries()
        size = self._size
        for i in range(size, n):
            sumu = 0.0
            sumu2 = 0.0
            for j in range(i - size, i):
                sumu += u[j]
                sumu2 += u[j] * u[j]
            s = math.sqrt(sumu2 / size - sumu * sumu / size / (size + 1))
            retval[dates[i]] = s
        return retval

    def calibrate(self, volatility_series: TimeSeries[float]) -> None:
        """No-op — there is nothing to fit.

        # C++ parity: ``void calibrate(const TimeSeries<Volatility>&) override {}``
        # (constantestimator.hpp:42). The override exists only because
        # ``VolatilityCompositor::calibrate`` is pure-virtual.
        """


__all__ = ["ConstantEstimator"]
