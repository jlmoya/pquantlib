"""InterpolatedYoYOptionletVolatilityCurve — experimental (v2) YoY vol curve.

# C++ parity: ql/experimental/inflation/yoyinflationoptionletvolatilitystructure2.hpp
   (v1.42.1) — ``InterpolatedYoYOptionletVolatilityCurve<Interpolator1D>``.

An interpolated-in-T, flat-in-K YoY optionlet volatility surface. The
volatility is the same for every strike (the smile is, by construction,
flat) and is interpolated in the time direction over ``(time, vol)``
pillars derived from the supplied ``(date, vol)`` data. The base vol
level (at ``base_date()``) is set to the interpolated value, which the
bootstrap uses as its anchor.

A second, data-less construction path (:meth:`for_bootstrap`) seeds only
the base vol level; the piecewise stripper fills the pillars later.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.volatility.inflation.yoy_optionlet_volatility_surface import (
    YoYOptionletVolatilitySurface,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

Interpolation1DFactory = Callable[[Array, Array], Interpolation]


class InterpolatedYoYOptionletVolatilityCurve(YoYOptionletVolatilitySurface):
    """Interpolated flat-smile YoY optionlet volatility surface."""

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        bdc: BusinessDayConvention,
        day_counter: DayCounter,
        lag: Period,
        frequency: Frequency,
        index_is_interpolated: bool,
        dates: Sequence[Date],
        volatilities: Sequence[float],
        min_strike: float,
        max_strike: float,
        interpolator: Interpolation1DFactory = LinearInterpolation,
    ) -> None:
        super().__init__(
            settlement_days=settlement_days,
            calendar=calendar,
            business_day_convention=bdc,
            day_counter=day_counter,
            observation_lag=lag,
            frequency=frequency,
            index_is_interpolated=index_is_interpolated,
        )
        qassert.require(
            len(dates) == len(volatilities),
            f"must have same number of dates and vols: {len(dates)} vs {len(volatilities)}",
        )
        qassert.require(len(dates) > 1, f"must have at least two dates: {len(dates)}")

        self._interpolator: Interpolation1DFactory = interpolator
        self._dates: list[Date] = list(dates)
        self._min_strike: float = min_strike
        self._max_strike: float = max_strike
        self._times: list[float] = []
        self._data: list[float] = []
        self._nodes: list[tuple[Date, float]] = []
        for j in range(len(self._dates)):
            self._times.append(self.time_from_reference(self._dates[j]))
            self._data.append(volatilities[j])
            self._nodes.append((self._dates[j], self._data[j]))

        self._interpolation: Interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )
        # set the base vol to the interpolated value (extrapolation allowed)
        base_time = self.time_from_reference(self.base_date())
        self._set_base_level(self._interpolation(base_time, allow_extrapolation=True))

    @classmethod
    def for_bootstrap(
        cls,
        settlement_days: int,
        calendar: Calendar,
        bdc: BusinessDayConvention,
        day_counter: DayCounter,
        lag: Period,
        frequency: Frequency,
        index_is_interpolated: bool,
        min_strike: float,
        max_strike: float,
        base_yoy_volatility: float,
        interpolator: Interpolation1DFactory = LinearInterpolation,
    ) -> InterpolatedYoYOptionletVolatilityCurve:
        """Data-less constructor used by the piecewise bootstrapper.

        # C++ parity: the protected second constructor
        # (yoyinflationoptionletvolatilitystructure2.hpp:91-102) that only
        # sets the base vol level so the bootstrap has its anchor.
        """
        instance = cls.__new__(cls)
        YoYOptionletVolatilitySurface.__init__(
            instance,
            settlement_days=settlement_days,
            calendar=calendar,
            business_day_convention=bdc,
            day_counter=day_counter,
            observation_lag=lag,
            frequency=frequency,
            index_is_interpolated=index_is_interpolated,
        )
        instance._interpolator = interpolator
        instance._dates = []
        instance._min_strike = min_strike
        instance._max_strike = max_strike
        instance._times = []
        instance._data = []
        instance._nodes = []
        instance._interpolation = None  # type: ignore[assignment]
        instance._set_base_level(base_yoy_volatility)
        return instance

    # ---- limits ------------------------------------------------------

    def min_strike(self) -> float:
        return self._min_strike

    def max_strike(self) -> float:
        return self._max_strike

    def max_date(self) -> Date:
        # # C++ parity: optionDateFromTenor(Period(ceil(interpolation.xMax()), Years)).
        assert self._interpolation is not None
        x_max = self._interpolation.x_max
        return self.option_date_from_tenor(Period(ceil(x_max), TimeUnit.Years))

    # ---- bootstrap interface ----------------------------------------

    def times(self) -> list[float]:
        return self._times

    def dates(self) -> list[Date]:
        return self._dates

    def data(self) -> list[float]:
        return self._data

    def nodes(self) -> list[tuple[Date, float]]:
        return list(self._nodes)

    # ---- volatility (flat in strike) --------------------------------

    def _volatility_impl(self, t: float, strike: float) -> float:
        # # C++ parity: returns interpolation_(t) — strike ignored (flat smile).
        del strike
        assert self._interpolation is not None
        return self._interpolation(t, allow_extrapolation=True)
