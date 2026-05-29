"""SwaptionVolatilityStructure — abstract swaption vol surface.

# C++ parity: ql/termstructures/volatility/swaption/swaptionvolstructure.{hpp,cpp}
# (v1.42.1).

Vol / variance / shift as a function of (option_expiry, swap_tenor,
strike). The C++ class supports both ``Period`` swap-tenor and raw
swap-length-in-years ``Time`` overloads; PQuantLib collapses on
``Period`` (the ``Time`` overload is mostly an internal optimization
used by SwapVolMatrix's locate-cell helper).

Subclasses MUST override ``max_date``, ``min_strike``, ``max_strike``,
``max_swap_tenor``, and ``_volatility_impl(option_time, swap_length,
strike)``. They MAY override ``volatility_type``, ``_shift_impl``.

C++ helper ``swapLength(Period)`` converts (Months / Years) → Time
(float years). We port it as a static-ish method.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.volatility_term_structure import VolatilityTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.termstructures.volatility.smile_section import SmileSection

_MONTHS_PER_YEAR: int = 12


class SwaptionVolatilityStructure(VolatilityTermStructure):
    """Abstract swaption-volatility structure."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention,
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

    # --- swap-length helpers --------------------------------------------

    def swap_length(self, swap_tenor: Period) -> float:
        """Convert a swap-tenor Period to a float year-fraction.

        # C++ parity: SwaptionVolatilityStructure::swapLength(Period).
        Only ``Months`` and ``Years`` units are accepted.
        """
        qassert.require(
            swap_tenor.length > 0,
            f"non-positive swap tenor ({swap_tenor}) given",
        )
        if swap_tenor.units == TimeUnit.Months:
            return swap_tenor.length / float(_MONTHS_PER_YEAR)
        if swap_tenor.units == TimeUnit.Years:
            return float(swap_tenor.length)
        qassert.fail(f"invalid time unit ({swap_tenor.units}) for swap length")
        # Unreachable, satisfies type checker.
        return 0.0

    # --- subclass hooks --------------------------------------------------

    @abstractmethod
    def max_swap_tenor(self) -> Period:
        """Largest swap tenor for which vols are defined."""

    def max_swap_length(self) -> float:
        return self.swap_length(self.max_swap_tenor())

    @abstractmethod
    def _volatility_impl(
        self,
        option_time: float,
        swap_length: float,
        strike: float,
    ) -> float:
        """Subclass: return vol at (option_time, swap_length, strike).

        Range / strike checks have already been performed.
        """

    def volatility_type(self) -> VolatilityType:
        return VolatilityType.ShiftedLognormal

    def _shift_impl(self, option_time: float, swap_length: float) -> float:
        """Subclass: return the shift parameter; default 0 for non-shifted."""
        _ = option_time, swap_length
        qassert.require(
            self.volatility_type() == VolatilityType.ShiftedLognormal,
            "shift parameter only makes sense for shifted lognormal vols",
        )
        return 0.0

    # --- range checks ----------------------------------------------------

    def check_swap_tenor(self, swap_tenor: Period | float, extrapolate: bool) -> None:
        if isinstance(swap_tenor, Period):
            qassert.require(
                swap_tenor.length > 0,
                f"non-positive swap tenor ({swap_tenor}) given",
            )
            qassert.require(
                extrapolate
                or self.allows_extrapolation()
                or swap_tenor <= self.max_swap_tenor(),
                f"swap tenor ({swap_tenor}) is past max tenor "
                f"({self.max_swap_tenor()})",
            )
        else:
            qassert.require(
                swap_tenor > 0.0,
                f"non-positive swap length ({swap_tenor}) given",
            )
            qassert.require(
                extrapolate
                or self.allows_extrapolation()
                or swap_tenor <= self.max_swap_length(),
                f"swap tenor ({swap_tenor}) is past max tenor "
                f"({self.max_swap_length()})",
            )

    # --- public API ------------------------------------------------------

    def volatility(
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Volatility at ``(option_expiry, swap_tenor, strike)``.

        Three overloads collapse onto ``_volatility_impl(option_time,
        swap_length, strike)`` — Period → Date → Time for expiry,
        Period → Time for swap tenor.
        """
        # Convert expiry to Date.
        if isinstance(option_expiry, Period):
            option_date = self.option_date_from_tenor(option_expiry)
        elif isinstance(option_expiry, Date):
            option_date = option_expiry
        else:
            # Already a float Time.
            self.check_swap_tenor(swap_tenor, extrapolate)
            self.check_time_range(option_expiry, extrapolate)
            self.check_strike(strike, extrapolate)
            length = (
                self.swap_length(swap_tenor) if isinstance(swap_tenor, Period) else swap_tenor
            )
            return self._volatility_impl(option_expiry, length, strike)

        self.check_swap_tenor(swap_tenor, extrapolate)
        self.check_range(option_date, extrapolate)
        self.check_strike(strike, extrapolate)
        t = self.time_from_reference(option_date)
        length = (
            self.swap_length(swap_tenor) if isinstance(swap_tenor, Period) else swap_tenor
        )
        return self._volatility_impl(t, length, strike)

    def black_variance(
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Black variance ``= vol^2 * t``.

        # C++ parity: SwaptionVolatilityStructure::blackVariance.
        """
        v = self.volatility(option_expiry, swap_tenor, strike, extrapolate)
        if isinstance(option_expiry, Period):
            option_expiry = self.option_date_from_tenor(option_expiry)
        if isinstance(option_expiry, Date):
            t = self.time_from_reference(option_expiry)
        else:
            t = float(option_expiry)
        return v * v * t

    # --- smile-section default surface ---------------------------------

    def smile_section(
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
        extrapolate: bool = False,
    ) -> SmileSection:
        """Return a ``SmileSection`` at ``(option_expiry, swap_tenor)``.

        # C++ parity: ``SwaptionVolatilityStructure::smileSection``
        # (swaptionvolstructure.cpp). The base wraps the
        # vol-as-of-strike call in a ``FlatSmileSection`` for structures
        # that don't have a smile (e.g. ``SwaptionConstantVolatility``
        # or the matrix interpolators). Subclasses may override (e.g.
        # ``SwaptionVolatilityCube.smile_section_impl``).
        """
        # Convert expiry to (Date, time).
        from pquantlib.termstructures.volatility.flat_smile_section import (  # noqa: PLC0415
            FlatSmileSection,
        )

        if isinstance(option_expiry, Period):
            option_date = self.option_date_from_tenor(option_expiry)
            t = self.time_from_reference(option_date)
        elif isinstance(option_expiry, Date):
            option_date = option_expiry
            t = self.time_from_reference(option_date)
        else:
            option_date = None
            t = float(option_expiry)
        length = (
            self.swap_length(swap_tenor) if isinstance(swap_tenor, Period) else swap_tenor
        )
        # Flat smile evaluates ``_volatility_impl(t, length, atm)`` once
        # and treats it as strike-independent. This is correct for
        # ``SwaptionConstantVolatility`` and surface-level matrix
        # interpolators that lack a true smile.
        # We need an atm to pick a vol; use the ATM forward = 0.0 query
        # point (constant-vol surfaces don't depend on strike so the
        # value is the same).
        vol = self._volatility_impl(t, length, 0.0)
        shift = self._shift_impl(t, length) if (
            self.volatility_type() == VolatilityType.ShiftedLognormal
        ) else 0.0
        _ = extrapolate
        return FlatSmileSection(
            volatility=vol,
            exercise_date=option_date,
            exercise_time=t,
            day_counter=self.day_counter(),
            reference_date=self.reference_date(),
            volatility_type=self.volatility_type(),
            shift=shift,
        )

    def shift(
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
        extrapolate: bool = False,
    ) -> float:
        """Shift parameter at ``(option_expiry, swap_tenor)``.

        # C++ parity: SwaptionVolatilityStructure::shift.
        """
        if isinstance(option_expiry, Period):
            option_date = self.option_date_from_tenor(option_expiry)
            return self.shift(option_date, swap_tenor, extrapolate)
        if isinstance(option_expiry, Date):
            self.check_swap_tenor(swap_tenor, extrapolate)
            self.check_range(option_expiry, extrapolate)
            t = self.time_from_reference(option_expiry)
        else:
            self.check_swap_tenor(swap_tenor, extrapolate)
            self.check_time_range(option_expiry, extrapolate)
            t = float(option_expiry)
        length = (
            self.swap_length(swap_tenor) if isinstance(swap_tenor, Period) else swap_tenor
        )
        return self._shift_impl(t, length)
