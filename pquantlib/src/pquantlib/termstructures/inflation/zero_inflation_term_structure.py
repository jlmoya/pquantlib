"""ZeroInflationTermStructure — abstract base for zero-coupon inflation curves.

# C++ parity: ql/termstructures/inflationtermstructure.{hpp,cpp} (v1.42.1) —
   ``ZeroInflationTermStructure`` class.

Concrete subclasses (L7-B InterpolatedZeroInflationCurve /
PiecewiseZeroInflationCurve) implement ``_zero_rate_impl(t)`` and
``max_date()``. The public ``zero_rate(d, extrapolate)`` enforces the
inflation-period bucketing rule (the zero rate is constant within a
period because zero fixings are non-interpolated), then applies any
configured seasonality correction.

Divergence: C++ exposes ``zeroRate(d, lag, forceLinear, extrapolate)``
as a deprecated overload, plus a ``zeroRate(t, extrapolate)`` overload.
We expose only the modern signature pair (date / time), without the
deprecated forceLinear path.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.termstructures.inflation.inflation_term_structure import (
    InflationTermStructure,
)
from pquantlib.termstructures.inflation.seasonality import Seasonality
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class ZeroInflationTermStructure(InflationTermStructure):
    """Abstract base for zero-coupon inflation rate curves.

    Concrete subclasses must implement ``_zero_rate_impl(t)`` and
    ``max_date()``.
    """

    def __init__(
        self,
        *,
        base_date: Date,
        frequency: Frequency,
        day_counter: DayCounter,
        observation_lag: Period | None = None,
        nominal_term_structure: YieldTermStructureProtocol | None = None,
        seasonality: Seasonality | None = None,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            base_date=base_date,
            frequency=frequency,
            day_counter=day_counter,
            observation_lag=observation_lag,
            nominal_term_structure=nominal_term_structure,
            seasonality=seasonality,
            reference_date=reference_date,
            calendar=calendar,
            settlement_days=settlement_days,
        )

    @abstractmethod
    def _zero_rate_impl(self, t: float) -> float:
        """Raw zero-inflation rate at time ``t``. Concrete classes override."""

    def zero_rate(self, d: Date, extrapolate: bool = False) -> float:
        """Zero-coupon inflation rate at the start of the period containing ``d``.

        # C++ parity: ``ZeroInflationTermStructure::zeroRate(const Date&,
        # bool)``. Buckets the date to its inflation period (because zero
        # fixings are non-interpolated), checks range, then optionally
        # applies the seasonality correction at ``d``.
        """
        period_start, _ = inflation_period(d, self.frequency())
        self.check_range(period_start, extrapolate)
        t = self.day_counter().year_fraction(self.reference_date(), period_start)
        rate = self._zero_rate_impl(t)
        if self.has_seasonality():
            season = self.seasonality()
            assert season is not None
            rate = season.correct_zero_rate(d, rate, self)
        return rate

    def zero_rate_at_time(self, t: float, extrapolate: bool = False) -> float:
        """Raw zero-rate at time ``t`` — no seasonality, no lag, no bucketing.

        # C++ parity: ``ZeroInflationTermStructure::zeroRate(Time, bool)``.
        # Power-user override; clients that need correct lag/seasonality
        # handling should call ``zero_rate(d, ...)`` instead.
        """
        self.check_time_range(t, extrapolate)
        return self._zero_rate_impl(t)
