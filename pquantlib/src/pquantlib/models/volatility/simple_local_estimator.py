"""Close-to-close local volatility estimator.

# C++ parity: ql/models/volatility/simplelocalestimator.hpp @ v1.43
# (header-only — there is no .cpp).
"""

from __future__ import annotations

import math

from pquantlib.time.time_series import TimeSeries
from pquantlib.volatility_model import LocalVolatilityEstimator


class SimpleLocalEstimator(LocalVolatilityEstimator[float]):
    """One-step log-return magnitude, scaled by ``1/sqrt(year_fraction)``.

    # C++ parity: ``class SimpleLocalEstimator :
    # public LocalVolatilityEstimator<Real>``
    # (ql/models/volatility/simplelocalestimator.hpp:35-55).

    Volatilities are expressed on an annual basis: ``year_fraction`` is the
    length of ONE interval in years (e.g. ``1/252`` for daily data), and the
    estimate is ``|log(q_i / q_{i-1})| / sqrt(year_fraction)``.

    Because each estimate needs the previous quote, the output series starts
    at the SECOND date of the input (simplelocalestimator.hpp:45-46).
    """

    __slots__ = ("_year_fraction",)

    def __init__(self, y: float) -> None:
        # C++ parity: simplelocalestimator.hpp:40-41.
        self._year_fraction: float = y

    def year_fraction(self) -> float:
        """Interval length in years. # C++ parity: the protected ``yearFraction_`` member."""
        return self._year_fraction

    def calculate(self, quote_series: TimeSeries[float]) -> TimeSeries[float]:
        """Absolute log-return per step, annualised.

        # C++ parity: ``SimpleLocalEstimator::calculate``
        # (simplelocalestimator.hpp:42-54).

        C++ increments an iterator past ``begin()`` unconditionally, which
        is undefined behaviour on an empty series; the Python loop simply
        yields an empty result there.
        """
        retval: TimeSeries[float] = TimeSeries()
        items = quote_series.items()
        for i in range(1, len(items)):
            date, cur = items[i]
            prev = items[i - 1][1]
            retval[date] = abs(math.log(cur / prev)) / math.sqrt(self._year_fraction)
        return retval


__all__ = ["SimpleLocalEstimator"]
