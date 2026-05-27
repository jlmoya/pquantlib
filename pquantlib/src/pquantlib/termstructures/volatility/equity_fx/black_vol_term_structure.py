"""Black volatility term-structure abstract bases (equity / FX).

# C++ parity: ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp +
#             blackvoltermstructure.cpp (v1.42.1).

Three abstract classes in this module:

- ``BlackVolTermStructure``: the root. Subclasses must provide both
  ``_black_vol_impl(t, strike)`` and ``_black_variance_impl(t, strike)``.
  Provides ``black_vol`` / ``black_variance`` (by date or by time),
  ``black_forward_vol`` / ``black_forward_variance`` (by date or by time).

- ``BlackVolatilityTermStructure``: adapter that derives variance from
  volatility (``variance = vol^2 * t``). Subclasses implement only
  ``_black_vol_impl``.

- ``BlackVarianceTermStructure``: adapter that derives volatility from
  variance (``vol = sqrt(variance / max(t, 1e-5))``). Subclasses
  implement only ``_black_variance_impl``.

All three classes default the business-day convention to ``Following``
(the C++ default).

Construction modes 1 (delegated) and 2 (fixed reference date) are
ported; mode 3 (moving via settlement days) is deferred — see the
note in VolatilityTermStructure.
"""

from __future__ import annotations

import math
from abc import abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility_term_structure import VolatilityTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class BlackVolTermStructure(VolatilityTermStructure):
    """Abstract Black volatility term structure.

    Subclasses MUST override ``max_date``, ``min_strike``, ``max_strike``,
    AND both ``_black_vol_impl(t, strike)`` and
    ``_black_variance_impl(t, strike)``. The two impls must agree:
    ``variance(t, K) = vol(t, K)^2 * t``.

    Use ``BlackVolatilityTermStructure`` if you only want to define
    ``_black_vol_impl``; ``BlackVarianceTermStructure`` if you only want
    to define ``_black_variance_impl``.
    """

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.Following,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )

    # --- subclass-implemented hooks ----------------------------------------

    @abstractmethod
    def _black_vol_impl(self, t: float, strike: float) -> float:
        """Subclass: return the Black volatility at ``(t, strike)``.

        Range and strike-range checks have already been performed; treat
        the call as if extrapolation is required.
        """

    @abstractmethod
    def _black_variance_impl(self, t: float, strike: float) -> float:
        """Subclass: return the Black variance at ``(t, strike)``."""

    # --- public Black-vol API: by Date -------------------------------------

    def black_vol(self, maturity: Date, strike: float, extrapolate: bool = False) -> float:
        self.check_range(maturity, extrapolate)
        self.check_strike(strike, extrapolate)
        t = self.time_from_reference(maturity)
        return self._black_vol_impl(t, strike)

    def black_vol_at_time(self, t: float, strike: float, extrapolate: bool = False) -> float:
        """Spot Black volatility, time-anchored variant.

        Mirrors C++ ``blackVol(Time, Real, bool)`` overload.
        """
        self.check_time_range(t, extrapolate)
        self.check_strike(strike, extrapolate)
        return self._black_vol_impl(t, strike)

    # --- public Black-variance API: by Date --------------------------------

    def black_variance(self, maturity: Date, strike: float, extrapolate: bool = False) -> float:
        self.check_range(maturity, extrapolate)
        self.check_strike(strike, extrapolate)
        t = self.time_from_reference(maturity)
        return self._black_variance_impl(t, strike)

    def black_variance_at_time(self, t: float, strike: float, extrapolate: bool = False) -> float:
        """Spot Black variance, time-anchored variant."""
        self.check_time_range(t, extrapolate)
        self.check_strike(strike, extrapolate)
        return self._black_variance_impl(t, strike)

    # --- forward vol/variance ---------------------------------------------

    def black_forward_vol(
        self, date1: Date, date2: Date, strike: float, extrapolate: bool = False
    ) -> float:
        """Forward (ATM) Black volatility between two dates."""
        qassert.require(date1 <= date2, f"{date1} later than {date2}")
        self.check_range(date2, extrapolate)
        t1 = self.time_from_reference(date1)
        t2 = self.time_from_reference(date2)
        return self.black_forward_vol_at_time(t1, t2, strike, extrapolate)

    def black_forward_vol_at_time(
        self, time1: float, time2: float, strike: float, extrapolate: bool = False
    ) -> float:
        """Forward (ATM) Black volatility between two times.

        Mirrors C++ ``blackForwardVol(Time, Time, Real, bool)``. When
        ``time1 == time2``, falls back to a central / forward finite-
        difference of variance (epsilon = 1e-5).
        """
        qassert.require(time1 <= time2, f"{time1} later than {time2}")
        self.check_time_range(time2, extrapolate)
        self.check_strike(strike, extrapolate)
        if time2 == time1:
            if time1 == 0.0:
                epsilon = 1.0e-5
                var = self._black_variance_impl(epsilon, strike)
                return math.sqrt(var / epsilon)
            epsilon = min(1.0e-5, time1)
            var1 = self._black_variance_impl(time1 - epsilon, strike)
            var2 = self._black_variance_impl(time1 + epsilon, strike)
            qassert.require(var2 >= var1, "variances must be non-decreasing")
            return math.sqrt((var2 - var1) / (2 * epsilon))
        var1 = self._black_variance_impl(time1, strike)
        var2 = self._black_variance_impl(time2, strike)
        qassert.require(var2 >= var1, "variances must be non-decreasing")
        return math.sqrt((var2 - var1) / (time2 - time1))

    def black_forward_variance(
        self, date1: Date, date2: Date, strike: float, extrapolate: bool = False
    ) -> float:
        qassert.require(date1 <= date2, f"{date1} later than {date2}")
        self.check_range(date2, extrapolate)
        t1 = self.time_from_reference(date1)
        t2 = self.time_from_reference(date2)
        return self.black_forward_variance_at_time(t1, t2, strike, extrapolate)

    def black_forward_variance_at_time(
        self, time1: float, time2: float, strike: float, extrapolate: bool = False
    ) -> float:
        qassert.require(time1 <= time2, f"{time1} later than {time2}")
        self.check_time_range(time2, extrapolate)
        self.check_strike(strike, extrapolate)
        v1 = self._black_variance_impl(time1, strike)
        v2 = self._black_variance_impl(time2, strike)
        qassert.require(v2 >= v1, "variances must be non-decreasing")
        return v2 - v1


class BlackVolatilityTermStructure(BlackVolTermStructure):
    """Adapter: subclasses implement only ``_black_vol_impl``.

    Variance is derived as ``vol^2 * t``.
    """

    def _black_variance_impl(self, t: float, strike: float) -> float:
        v = self._black_vol_impl(t, strike)
        return v * v * t

    # _black_vol_impl remains abstract — re-declared to flag the contract.
    @abstractmethod
    def _black_vol_impl(self, t: float, strike: float) -> float:
        """Subclass: Black volatility at ``(t, strike)``."""


class BlackVarianceTermStructure(BlackVolTermStructure):
    """Adapter: subclasses implement only ``_black_variance_impl``.

    Volatility is derived as ``sqrt(variance / max(t, 1e-5))``. The
    1e-5 floor mirrors C++ ``BlackVarianceTermStructure::blackVolImpl``.
    """

    def _black_vol_impl(self, t: float, strike: float) -> float:
        non_zero_t = 1.0e-5 if t == 0.0 else t
        var = self._black_variance_impl(non_zero_t, strike)
        return math.sqrt(var / non_zero_t)

    # _black_variance_impl remains abstract — re-declared to flag the contract.
    @abstractmethod
    def _black_variance_impl(self, t: float, strike: float) -> float:
        """Subclass: Black variance at ``(t, strike)``."""
