"""YoYOptionletVolatilitySurface — abstract base for YoY-rate vol surfaces.

# C++ parity: ql/termstructures/volatility/inflation/yoyinflationoptionletvolatilitystructure.{hpp,cpp}
   (v1.42.1).

Concrete subclasses (``ConstantYoYOptionletVolatility`` here; SABR / bilinear
in carve-out) implement ``_volatility_impl(t, strike)``. The public
``volatility(maturity_date, strike, obs_lag, extrapolate)`` enforces the
lag-relative time-from-reference computation and applies inflation-period
bucketing if the underlying YoY index is non-interpolated.

Volatility types: surface can be either ShiftedLognormal (the default in
C++) or Normal — selected via the ``volatility_type`` argument. Displacement
applies only to ShiftedLognormal and must be 0 or 1 (C++ asserts the same).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Final

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.volatility_term_structure import (
    VolatilityTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# Sentinel: the C++ default observation-lag is ``Period(-1, Days)`` meaning
# "use the structure's lag". PQuantLib mirrors this with ``None``.
_NEGATIVE_ONE_DAY: Final[Period] = Period(-1, TimeUnit.Days)


class YoYOptionletVolatilitySurface(VolatilityTermStructure):
    """Abstract YoY-rate optionlet volatility surface.

    Concrete subclasses MUST override:
        * ``_volatility_impl(t, strike)`` — raw vol calculation,
        * ``min_strike()`` / ``max_strike()`` (inherited),
        * ``max_date()`` (inherited).
    """

    def __init__(
        self,
        *,
        settlement_days: int,
        calendar: Calendar,
        business_day_convention: BusinessDayConvention,
        day_counter: DayCounter,
        observation_lag: Period,
        frequency: Frequency,
        index_is_interpolated: bool,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        # # C++ parity: yoyinflationoptionletvolatilitystructure.cpp:45-47 —
        # # displacement must be 0 or 1.
        qassert.require(
            math.isclose(displacement, 0.0, abs_tol=1e-12)
            or math.isclose(displacement, 1.0, abs_tol=1e-12),
            f"YoYOptionletVolatilitySurface: displacement ({displacement}) must be 0 or 1",
        )
        self._observation_lag: Period = observation_lag
        self._frequency: Frequency = frequency
        self._index_is_interpolated: bool = index_is_interpolated
        self._volatility_type: VolatilityType = volatility_type
        self._displacement: float = displacement
        self._base_level: float | None = None

    # ---- inspectors ---------------------------------------------------

    def observation_lag(self) -> Period:
        return self._observation_lag

    def frequency(self) -> Frequency:
        return self._frequency

    def index_is_interpolated(self) -> bool:
        return self._index_is_interpolated

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def displacement(self) -> float:
        return self._displacement

    def base_date(self) -> Date:
        """Surface base date — in the past because of observation lag.

        # C++ parity: ``YoYOptionletVolatilitySurface::baseDate``.
        """
        if self._index_is_interpolated:
            return self.reference_date() - self._observation_lag
        start, _ = inflation_period(
            self.reference_date() - self._observation_lag, self._frequency
        )
        return start

    def time_from_base(self, date: Date, obs_lag: Period | None = None) -> float:
        """Time from this surface's ``base_date()`` to ``date``.

        # C++ parity: ``YoYOptionletVolatilitySurface::timeFromBase``.
        """
        use_lag = obs_lag if (obs_lag is not None and obs_lag != _NEGATIVE_ONE_DAY) else self._observation_lag
        if self._index_is_interpolated:
            use_date = date - use_lag
        else:
            start, _ = inflation_period(date - use_lag, self._frequency)
            use_date = start
        return self.day_counter().year_fraction(self.base_date(), use_date)

    def base_level(self) -> float:
        """Vol at base date (used as the bootstrap anchor).

        # C++ parity: ``YoYOptionletVolatilitySurface::baseLevel()``. Raises if unset.
        """
        qassert.require(
            self._base_level is not None,
            "Base volatility, for baseDate(), not set.",
        )
        assert self._base_level is not None
        return self._base_level

    def _set_base_level(self, vol: float) -> None:
        self._base_level = vol

    # ---- public volatility / variance overloads -----------------------

    def volatility(
        self,
        date: Date,
        strike: float,
        obs_lag: Period | None = None,
        extrapolate: bool = False,
    ) -> float:
        """Vol at ``(maturity_date, strike)`` with optional lag override.

        # C++ parity: ``YoYOptionletVolatilitySurface::volatility(const Date&, ...)``.
        """
        use_lag = obs_lag if (obs_lag is not None and obs_lag != _NEGATIVE_ONE_DAY) else self._observation_lag
        if self._index_is_interpolated:
            check_date = date - use_lag
        else:
            start, _ = inflation_period(date - use_lag, self._frequency)
            check_date = start
        self._check_range(check_date, strike, extrapolate)
        t = self.day_counter().year_fraction(self.reference_date(), check_date)
        return self._volatility_impl(t, strike)

    def volatility_at_time(self, t: float, strike: float) -> float:
        """Time-based overload — no lag, no bucketing.

        # C++ parity: ``YoYOptionletVolatilitySurface::volatility(Time, Rate)``.
        """
        return self._volatility_impl(t, strike)

    def total_variance(
        self,
        date: Date,
        strike: float,
        obs_lag: Period | None = None,
        extrapolate: bool = False,
    ) -> float:
        """Total integrated variance ``vol^2 * time_from_base``.

        # C++ parity: ``YoYOptionletVolatilitySurface::totalVariance(const Date&, ...)``.
        """
        vol = self.volatility(date, strike, obs_lag, extrapolate)
        t = self.time_from_base(date, obs_lag)
        return vol * vol * t

    # ---- range checks (date + strike) ---------------------------------

    def _check_range(self, date: Date, strike: float, extrapolate: bool) -> None:
        # # C++ parity: ``YoYOptionletVolatilitySurface::checkRange(const Date&, Rate, bool)``.
        qassert.require(
            date >= self.base_date(),
            f"date ({date}) is before base date ({self.base_date()})",
        )
        qassert.require(
            extrapolate or self.allows_extrapolation() or date <= self.max_date(),
            f"date ({date}) is past max curve date ({self.max_date()})",
        )
        qassert.require(
            extrapolate
            or self.allows_extrapolation()
            or (self.min_strike() <= strike <= self.max_strike()),
            f"strike ({strike}) is outside the curve domain "
            f"[{self.min_strike()},{self.max_strike()}] at date = {date}",
        )

    # ---- subclass hook ------------------------------------------------

    @abstractmethod
    def _volatility_impl(self, t: float, strike: float) -> float:
        """Raw volatility computation at ``(time, strike)``. Subclasses override."""


__all__ = ["YoYOptionletVolatilitySurface"]
