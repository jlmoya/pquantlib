"""CPIVolatilitySurface — abstract base for zero-CPI (lognormal-style) vol.

# C++ parity: ql/termstructures/volatility/inflation/cpivolatilitystructure.{hpp,cpp}
   (v1.42.1).

Concrete subclasses (``ConstantCPIVolatility`` here; SABR / surface-bilinear
in carve-out) implement ``_volatility_impl(t, strike)``. The public
``volatility(date, strike, ...)`` method handles inflation-period bucketing
(zero CPI fixings are non-interpolated by default) and lag-relative time
calculations.

Python divergence: C++ exposes a tenor + a date overload plus a Time
overload. The Python port collapses to ``volatility(date, strike, ...)``
and ``volatility_at_time(t, strike)``; tenor → date conversion is the
caller's job (use ``option_date_from_tenor``).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Final

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.inflation_index import inflation_period
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
# "use the structure's lag". PQuantLib mirrors this by passing ``None``
# from public callers and resolving to ``observation_lag()`` internally.
_NEGATIVE_ONE_DAY: Final[Period] = Period(-1, TimeUnit.Days)


class CPIVolatilitySurface(VolatilityTermStructure):
    """Abstract zero-CPI volatility surface.

    Concrete subclasses MUST override:
        * ``_volatility_impl(t, strike)`` — raw vol calculation,
        * ``min_strike()`` / ``max_strike()`` (inherited contract),
        * ``max_date()`` (inherited from TermStructure).
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
    ) -> None:
        # # C++ parity: CPIVolatilitySurface delegates to the settlement-days
        # # constructor of VolatilityTermStructure. PQuantLib's TermStructure
        # # supports settlement_days only via the moving-reference-date mode
        # # (mode 3 in C++ parlance). For Constant variants we resolve the
        # # reference date inline.
        super().__init__(
            business_day_convention=business_day_convention,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._observation_lag: Period = observation_lag
        self._frequency: Frequency = frequency
        self._index_is_interpolated: bool = index_is_interpolated
        self._base_level: float | None = None

    # ---- inspectors ---------------------------------------------------

    def observation_lag(self) -> Period:
        return self._observation_lag

    def frequency(self) -> Frequency:
        return self._frequency

    def index_is_interpolated(self) -> bool:
        return self._index_is_interpolated

    def base_date(self) -> Date:
        """Surface base date — accounts for observation lag.

        # C++ parity: ``CPIVolatilitySurface::baseDate`` (cpivolatilitystructure.cpp).
        # The base date is in the past because of the observation lag.
        """
        if self._index_is_interpolated:
            return self.reference_date() - self._observation_lag
        start, _ = inflation_period(
            self.reference_date() - self._observation_lag, self._frequency
        )
        return start

    def time_from_base(self, date: Date, obs_lag: Period | None = None) -> float:
        """Time from this surface's ``base_date()`` to ``date``.

        # C++ parity: ``CPIVolatilitySurface::timeFromBase``. The ``obs_lag``
        # override is for callers that want to evaluate against a non-default
        # observation lag.
        """
        use_lag = obs_lag if (obs_lag is not None and obs_lag != _NEGATIVE_ONE_DAY) else self._observation_lag
        use_date: Date
        if self._index_is_interpolated:
            use_date = date - use_lag
        else:
            start, _ = inflation_period(date - use_lag, self._frequency)
            use_date = start
        return self.day_counter().year_fraction(self.base_date(), use_date)

    def base_level(self) -> float:
        """Vol at base date (used as the bootstrap anchor).

        # C++ parity: ``CPIVolatilitySurface::baseLevel()``. Raises if unset.
        """
        qassert.require(
            self._base_level is not None,
            "Base volatility, for baseDate(), not set.",
        )
        assert self._base_level is not None
        return self._base_level

    def _set_base_level(self, vol: float) -> None:
        """Concrete bootstrappers call this; constant surfaces don't need it."""
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

        # C++ parity: ``CPIVolatilitySurface::volatility(const Date&, ...)``.
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

        # C++ parity: ``CPIVolatilitySurface::volatility(Time, Rate)``.
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

        # C++ parity: ``CPIVolatilitySurface::totalVariance(const Date&, ...)``.
        """
        vol = self.volatility(date, strike, obs_lag, extrapolate)
        t = self.time_from_base(date, obs_lag)
        return vol * vol * t

    # ---- range checks (date + strike) ---------------------------------

    def _check_range(self, date: Date, strike: float, extrapolate: bool) -> None:
        # # C++ parity: ``CPIVolatilitySurface::checkRange(const Date&, Rate, bool)``.
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


__all__ = ["CPIVolatilitySurface"]
