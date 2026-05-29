"""OptionletVolatilityStructure — abstract caplet/floorlet vol surface.

# C++ parity: ql/termstructures/volatility/optionlet/optionletvolatilitystructure.{hpp,cpp}
# (v1.42.1).

Defines the interface for caplet/floorlet vol surfaces — vol or
Black-variance as a function of (option_date_or_time, strike). The
C++ class also exposes smile-section accessors; PQuantLib defers
``smile_section`` to subclasses that need it (mostly carve-outs such
as SABR cubes).

Subclasses MUST override ``max_date``, ``min_strike``, ``max_strike``,
AND ``_volatility_impl(t, strike)``. They MAY override
``volatility_type()`` and ``displacement()``; defaults are
``ShiftedLognormal`` and ``0``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.volatility_term_structure import VolatilityTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class OptionletVolatilityStructure(VolatilityTermStructure):
    """Abstract optionlet (caplet/floorlet) volatility structure."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.Following,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )

    # --- subclass hooks --------------------------------------------------

    @abstractmethod
    def _volatility_impl(self, t: float, strike: float) -> float:
        """Subclass: return the optionlet vol at (time, strike)."""

    def volatility_type(self) -> VolatilityType:
        """# C++ parity: OptionletVolatilityStructure::volatilityType (default)."""
        return VolatilityType.ShiftedLognormal

    def displacement(self) -> float:
        """# C++ parity: OptionletVolatilityStructure::displacement (default)."""
        return 0.0

    # --- public API ------------------------------------------------------

    def volatility(
        self,
        option_date: Period | Date | float,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Return the optionlet volatility at (option_date_or_time, strike).

        # C++ parity: OptionletVolatilityStructure::volatility — three
        # overloads (Period / Date / Time) collapse onto the Time
        # implementation here.
        """
        if isinstance(option_date, Period):
            d = self.option_date_from_tenor(option_date)
            return self.volatility(d, strike, extrapolate)
        if isinstance(option_date, Date):
            self.check_range(option_date, extrapolate)
            self.check_strike(strike, extrapolate)
            t = self.time_from_reference(option_date)
            return self._volatility_impl(t, strike)
        # Time-based.
        self.check_time_range(option_date, extrapolate)
        self.check_strike(strike, extrapolate)
        return self._volatility_impl(option_date, strike)

    def black_variance(
        self,
        option_date: Period | Date | float,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Black variance at (option_date_or_time, strike) = vol^2 * t.

        # C++ parity: OptionletVolatilityStructure::blackVariance.
        """
        if isinstance(option_date, Period):
            d = self.option_date_from_tenor(option_date)
            return self.black_variance(d, strike, extrapolate)
        if isinstance(option_date, Date):
            v = self.volatility(option_date, strike, extrapolate)
            t = self.time_from_reference(option_date)
            return v * v * t
        v = self.volatility(option_date, strike, extrapolate)
        return v * v * option_date
