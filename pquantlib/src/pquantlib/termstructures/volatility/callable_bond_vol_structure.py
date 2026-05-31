"""CallableBondVolatilityStructure - vol surface for callable-bond yield vol.

# C++ parity: ql/experimental/callablebonds/callablebondvolstructure.{hpp,cpp}
#             (v1.42.1).

Abstract base for the (option-time x bond-length x strike) volatility
surface consumed by ``BlackCallableFixedRateBondEngine``. The quoted
quantity is a *forward yield volatility* (Hull, Fourth Edition, Ch.20),
not a price volatility - the engine converts yield-vol → price-vol
internally.

Only the time-based ``volatility(option_time, bond_length, strike)`` and
``black_variance`` paths are needed by the Black engine; the date/tenor
overloads delegate to ``convert_dates``. The full smile-section machinery
is reproduced for parity, deferring nothing the engine relies on.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.volatility.smile_section import SmileSection
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date


class CallableBondVolatilityStructure(TermStructure):
    """Abstract callable-bond volatility surface.

    # C++ parity: ``class CallableBondVolatilityStructure : public
    # TermStructure`` (callablebondvolstructure.hpp:38).

    Three construction modes mirror :class:`TermStructure` (delegated,
    fixed-reference-date, moving). The business-day convention used to
    advance option tenors → option dates is stored here.
    """

    def __init__(
        self,
        *,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        settlement_days: int | None = None,
        bdc: BusinessDayConvention = BusinessDayConvention.Following,
    ) -> None:
        TermStructure.__init__(
            self,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._bdc: BusinessDayConvention = bdc

    # ------------------------------------------------------------------
    # Volatility / variance - time-based (what the Black engine uses)
    # ------------------------------------------------------------------

    def volatility(
        self,
        option_time: float,
        bond_length: float,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Vol for an option time + bond length + strike.

        # C++ parity: callablebondvolstructure.hpp:167-174 (inline).
        """
        self._check_range_time(option_time, bond_length, strike, extrapolate)
        return self._volatility_impl(option_time, bond_length, strike)

    def black_variance(
        self,
        option_time: float,
        bond_length: float,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Black variance ``vol^2 * option_time``.

        # C++ parity: callablebondvolstructure.hpp:177-185 (inline).
        """
        self._check_range_time(option_time, bond_length, strike, extrapolate)
        vol = self._volatility_impl(option_time, bond_length, strike)
        return vol * vol * option_time

    # ------------------------------------------------------------------
    # Volatility / variance - date + tenor
    # ------------------------------------------------------------------

    def volatility_date(
        self,
        option_date: Date,
        bond_tenor: Period,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Vol for an option date + bond tenor + strike.

        # C++ parity: callablebondvolstructure.hpp:188-195 (inline).
        """
        self._check_range_date(option_date, bond_tenor, strike, extrapolate)
        return self._volatility_impl_date(option_date, bond_tenor, strike)

    def black_variance_date(
        self,
        option_date: Date,
        bond_tenor: Period,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Black variance for an option date + bond tenor.

        # C++ parity: callablebondvolstructure.hpp:197-206 (inline).
        """
        vol = self.volatility_date(option_date, bond_tenor, strike, extrapolate)
        t, _ = self.convert_dates(option_date, bond_tenor)
        return vol * vol * t

    def volatility_tenor(
        self,
        option_tenor: Period,
        bond_tenor: Period,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Vol for an option tenor + bond tenor + strike.

        # C++ parity: callablebondvolstructure.hpp:208-215 (inline).
        """
        option_date = self.option_date_from_tenor(option_tenor)
        return self.volatility_date(option_date, bond_tenor, strike, extrapolate)

    def smile_section(self, option_date: Date, bond_tenor: Period) -> SmileSection:
        """Smile section for an option date + bond tenor.

        # C++ parity: callablebondvolstructure.hpp:87-92 (inline).
        """
        t1, t2 = self.convert_dates(option_date, bond_tenor)
        return self._smile_section_impl(t1, t2)

    # ------------------------------------------------------------------
    # Limits (subclasses implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def max_bond_tenor(self) -> Period:
        """Largest bond tenor the surface can return vols for.

        # C++ parity: callablebondvolstructure.hpp:111 (pure virtual).
        """

    def max_bond_length(self) -> float:
        """Largest bond length (in time) the surface can return vols for.

        # C++ parity: callablebondvolstructure.cpp ``maxBondLength`` -
        # ``timeFromReference(referenceDate() + maxBondTenor())``.
        """
        return self.time_from_reference(self.reference_date() + self.max_bond_tenor())

    @abstractmethod
    def min_strike(self) -> float:
        """Minimum strike the surface can return vols for.

        # C++ parity: callablebondvolstructure.hpp:115 (pure virtual).
        """

    @abstractmethod
    def max_strike(self) -> float:
        """Maximum strike the surface can return vols for.

        # C++ parity: callablebondvolstructure.hpp:117 (pure virtual).
        """

    # ------------------------------------------------------------------
    # Date/time conversion + helpers
    # ------------------------------------------------------------------

    def convert_dates(self, option_date: Date, bond_tenor: Period) -> tuple[float, float]:
        """Convert (option_date, bond_tenor) → (option_time, bond_length).

        # C++ parity: callablebondvolstructure.cpp ``convertDates``.
        """
        end = self.calendar().advance_period(option_date, bond_tenor, self._bdc)
        option_time = self.time_from_reference(option_date)
        time_length = self.day_counter().year_fraction(option_date, end)
        return option_time, time_length

    def business_day_convention(self) -> BusinessDayConvention:
        # C++ parity: callablebondvolstructure.hpp:155-158 (inline).
        return self._bdc

    def option_date_from_tenor(self, option_tenor: Period) -> Date:
        """Advance the reference date by an option tenor.

        # C++ parity: callablebondvolstructure.hpp:160-165 (inline).
        """
        return self.calendar().advance_period(
            self.reference_date(), option_tenor, self._bdc
        )

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    @abstractmethod
    def _smile_section_impl(self, option_time: float, bond_length: float) -> SmileSection:
        """Return the smile section at (option_time, bond_length).

        # C++ parity: callablebondvolstructure.hpp:130-132 (pure virtual).
        """

    @abstractmethod
    def _volatility_impl(self, option_time: float, bond_length: float, strike: float) -> float:
        """Actual vol calculation (time-based).

        # C++ parity: callablebondvolstructure.hpp:135-137 (pure virtual).
        """

    def _volatility_impl_date(
        self, option_date: Date, bond_tenor: Period, strike: float
    ) -> float:
        """Vol calculation (date-based) - default converts then delegates.

        # C++ parity: callablebondvolstructure.hpp:138-143 (virtual w/ body).
        """
        t1, t2 = self.convert_dates(option_date, bond_tenor)
        return self._volatility_impl(t1, t2, strike)

    # ------------------------------------------------------------------
    # Range checks
    # ------------------------------------------------------------------

    def _check_range_time(
        self, option_time: float, bond_length: float, strike: float, extrapolate: bool
    ) -> None:
        # C++ parity: callablebondvolstructure.hpp:239-252 (inline).
        self.check_time_range(option_time, extrapolate)
        qassert.require(bond_length >= 0.0, f"negative bondLength ({bond_length}) given")
        qassert.require(
            extrapolate or self.allows_extrapolation() or bond_length <= self.max_bond_length(),
            f"bondLength ({bond_length}) is past max curve bondLength ({self.max_bond_length()})",
        )
        qassert.require(
            extrapolate
            or self.allows_extrapolation()
            or (self.min_strike() <= strike <= self.max_strike()),
            f"strike ({strike}) is outside the curve domain "
            f"[{self.min_strike()},{self.max_strike()}]",
        )

    def _check_range_date(
        self, option_date: Date, bond_tenor: Period, strike: float, extrapolate: bool
    ) -> None:
        # C++ parity: callablebondvolstructure.cpp ``checkRange(Date, Period, ...)``.
        t1, t2 = self.convert_dates(option_date, bond_tenor)
        self._check_range_time(t1, t2, strike, extrapolate)


# Re-export TimeUnit-built 100Y constant convenience (used by the concrete vol).
_HUNDRED_YEARS: Period = Period(100, TimeUnit.Years)


__all__ = ["CallableBondVolatilityStructure"]
