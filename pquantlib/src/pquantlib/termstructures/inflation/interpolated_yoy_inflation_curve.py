"""InterpolatedYoYInflationCurve — YoY curve from known (date, rate) nodes.

# C++ parity: ql/termstructures/inflation/interpolatedyoyinflationcurve.hpp
   (v1.42.1) — ``InterpolatedYoYInflationCurve<Interpolator>`` template.

The YoY analogue of ``InterpolatedZeroInflationCurve``: nodes are YoY
inflation rates (which may be negative) and the interpolation runs in
time space (year fractions from the reference date).

C++ sets the base YoY rate from ``rates[0]`` in the YoY-abstract
constructor; we forward the same value through the
``base_yoy_rate`` slot on ``YoYInflationTermStructure``.

Note: the C++ docstring explicitly notes "The provided rates are not
YY inflation-swap quotes" — they are raw YoY rates from a calibrated
curve, not market quotes that still need bootstrapping.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.inflation.seasonality import Seasonality
from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
    YoYInflationTermStructure,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedYoYInflationCurve(YoYInflationTermStructure):
    """YoY inflation curve from linear (or other) interpolation of YoY rates."""

    def __init__(
        self,
        reference_date: Date,
        dates: Sequence[Date],
        rates: Sequence[float],
        frequency: Frequency,
        day_counter: DayCounter,
        seasonality: Seasonality | None = None,
        interpolator: InterpolationFactory = LinearInterpolation,
        calendar: Calendar | None = None,
        observation_lag: Period | None = None,
        nominal_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        qassert.require(len(dates) > 1, f"too few dates: {len(dates)}")
        qassert.require(
            len(rates) == len(dates),
            f"indices/dates count mismatch: {len(rates)} vs {len(dates)}",
        )
        for i in range(1, len(rates)):
            # C++ parity: YoY rates may be < 0 but must be > -1 (>-100%).
            qassert.require(rates[i] > -1.0, "year-on-year inflation data < -100 %")

        super().__init__(
            base_date=dates[0],
            base_yoy_rate=rates[0],
            frequency=frequency,
            day_counter=day_counter,
            observation_lag=observation_lag,
            nominal_term_structure=nominal_term_structure,
            seasonality=seasonality,
            reference_date=reference_date,
            calendar=calendar,
        )
        self._dates: list[Date] = list(dates)
        self._data: list[float] = list(rates)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in self._dates
        ]
        self._interpolation: Interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )

    # ---- InflationTermStructure interface ----------------------------

    def max_date(self) -> Date:
        return self._dates[-1]

    # ---- YoYInflationTermStructure implementation --------------------

    def _yoy_rate_impl(self, t: float) -> float:
        """C++ parity: ``yoyRateImpl(Time)`` = ``interpolation_(t, true)``."""
        return self._interpolation(t, allow_extrapolation=True)

    # ---- inspectors --------------------------------------------------

    def dates(self) -> list[Date]:
        return list(self._dates)

    def times(self) -> list[float]:
        return list(self._times)

    def data(self) -> list[float]:
        return list(self._data)

    def rates(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))
