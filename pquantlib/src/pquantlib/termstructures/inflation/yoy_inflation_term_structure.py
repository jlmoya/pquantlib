"""YoYInflationTermStructure — abstract base for year-on-year inflation curves.

# C++ parity: ql/termstructures/inflationtermstructure.{hpp,cpp} (v1.42.1) —
   ``YoYInflationTermStructure`` class.

Concrete subclasses (L7-B InterpolatedYoYInflationCurve /
PiecewiseYoYInflationCurve) implement ``_yoy_rate_impl(t)`` and
``max_date()``.
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


class YoYInflationTermStructure(InflationTermStructure):
    """Abstract base for year-on-year inflation rate curves."""

    def __init__(
        self,
        *,
        base_date: Date,
        base_yoy_rate: float,
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
            base_rate=base_yoy_rate,
            seasonality=seasonality,
            reference_date=reference_date,
            calendar=calendar,
            settlement_days=settlement_days,
        )

    @abstractmethod
    def _yoy_rate_impl(self, t: float) -> float:
        """Raw year-on-year rate at time ``t``. Concrete classes override."""

    def yoy_rate(self, d: Date, extrapolate: bool = False) -> float:
        """Year-on-year inflation rate at the start of the period containing ``d``.

        # C++ parity: ``YoYInflationTermStructure::yoyRate(const Date&, bool)``.
        # Bucket the date to its inflation period, check range, apply
        # seasonality correction (if any) via correctYoYRate.
        """
        period_start, _ = inflation_period(d, self.frequency())
        self.check_range(period_start, extrapolate)
        t = self.day_counter().year_fraction(self.reference_date(), period_start)
        rate = self._yoy_rate_impl(t)
        if self.has_seasonality():
            season = self.seasonality()
            assert season is not None
            rate = season.correct_yoy_rate(d, rate, self)
        return rate

    def yoy_rate_at_time(self, t: float, extrapolate: bool = False) -> float:
        """Raw YoY rate at time ``t``. # C++ parity: ``yoyRate(Time, bool)``."""
        self.check_time_range(t, extrapolate)
        return self._yoy_rate_impl(t)
