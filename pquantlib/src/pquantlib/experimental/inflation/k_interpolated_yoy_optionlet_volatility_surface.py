"""KInterpolatedYoYOptionletVolatilitySurface — strike-interpolated YoY vol.

# C++ parity: ql/experimental/inflation/kinterpolatedyoyoptionletvolatilitysurface.hpp
   (v1.42.1) — ``KInterpolatedYoYOptionletVolatilitySurface<Interpolator1D>``.

Wraps a :class:`YoYOptionletStripper`: the stripper provides curves in the
T direction along each K; this class interpolates *in the K direction* a
slice of those curves at a query date. The stripping is performed once at
construction (``performCalculations``).

.. note::
   Upstream C++ carries a ``\\bug Tests currently fail`` annotation.
"""

from __future__ import annotations

from collections.abc import Callable
from math import floor

import numpy as np

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.inflation.yoy_cap_floor_term_price_surface import (
    YoYCapFloorTermPriceSurface,
)
from pquantlib.experimental.inflation.yoy_optionlet_stripper import YoYOptionletStripper
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.pricingengines.inflation.yoy_inflation_capfloor_engine import (
    YoYInflationCapFloorEngine,
)
from pquantlib.termstructures.volatility.inflation.yoy_optionlet_volatility_surface import (
    YoYOptionletVolatilitySurface,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

Interpolation1DFactory = Callable[[Array, Array], Interpolation]


class KInterpolatedYoYOptionletVolatilitySurface(YoYOptionletVolatilitySurface):
    """K-interpolated YoY optionlet volatility surface."""

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        bdc: BusinessDayConvention,
        day_counter: DayCounter,
        lag: Period,
        cap_floor_prices: YoYCapFloorTermPriceSurface,
        pricer: YoYInflationCapFloorEngine,
        yoy_optionlet_stripper: YoYOptionletStripper,
        slope: float,
        interpolator: Interpolation1DFactory = LinearInterpolation,
        vol_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
    ) -> None:
        super().__init__(
            settlement_days=settlement_days,
            calendar=calendar,
            business_day_convention=bdc,
            day_counter=day_counter,
            observation_lag=lag,
            frequency=cap_floor_prices.yoy_index().frequency(),
            index_is_interpolated=cap_floor_prices.yoy_index().interpolated(),
            volatility_type=vol_type,
            displacement=displacement,
        )
        self._cap_floor_prices: YoYCapFloorTermPriceSurface = cap_floor_prices
        self._pricer: YoYInflationCapFloorEngine = pricer
        self._stripper: YoYOptionletStripper = yoy_optionlet_stripper
        self._factory1d: Interpolation1DFactory = interpolator
        self._slope: float = slope
        self._last_date_is_set: bool = False
        self._last_date: Date | None = None
        self._temp_k_interpolation: Interpolation | None = None
        self._slice: tuple[list[float], list[float]] = ([], [])
        self._perform_calculations()

    # ---- limits ------------------------------------------------------

    def min_strike(self) -> float:
        return self._cap_floor_prices.strikes()[0]

    def max_strike(self) -> float:
        return self._cap_floor_prices.strikes()[-1]

    def max_date(self) -> Date:
        mats = self._cap_floor_prices.maturities()
        return self.reference_date() + mats[-1]

    # ---- slice access ------------------------------------------------

    def d_slice(self, d: Date) -> tuple[list[float], list[float]]:
        self._update_slice(d)
        return self._slice

    # ---- internals ---------------------------------------------------

    def _perform_calculations(self) -> None:
        # slope is the assumption on the initial caplet vol change.
        self._stripper.initialize(self._cap_floor_prices, self._pricer, self._slope)

    def _volatility_impl(self, t: float, strike: float) -> float:
        # # C++ parity: the Time overload converts t -> a date then defers.
        years = floor(t)
        days = floor((t - years) * 365.0)
        d = self.reference_date() + Period(years, TimeUnit.Years) + Period(days, TimeUnit.Days)
        return self._volatility_impl_date(d, strike)

    def _volatility_impl_date(self, d: Date, strike: float) -> float:
        self._update_slice(d)
        assert self._temp_k_interpolation is not None
        return self._temp_k_interpolation(strike, allow_extrapolation=self.allows_extrapolation())

    def _update_slice(self, d: Date) -> None:
        if not self._last_date_is_set or d != self._last_date:
            self._slice = self._stripper.slice(d)
            self._temp_k_interpolation = self._factory1d(
                np.asarray(self._slice[0], dtype=np.float64),
                np.asarray(self._slice[1], dtype=np.float64),
            )
            self._last_date_is_set = True
            self._last_date = d
