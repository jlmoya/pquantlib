"""CapFloorTermVolatilityStructure — abstract cap/floor term-vol surface.

# C++ parity: ql/termstructures/volatility/capfloor/capfloortermvolatilitystructure.{hpp,cpp}
# (v1.42.1).

C++ defines the ATM cap/floor term volatility surface as a function of
``(maturity Period|Date|Time, strike Rate)`` with three constructors
(delegated / fixed reference date / moving via settlement days). Mode 3
is supported here via the parent ``VolatilityTermStructure``
``settlement_days`` ctor argument (inherited from ``TermStructure``).

Subclasses MUST override ``_volatility_impl(t, strike)``; the public
date- and Period-shaped overloads are dispatched on this single hook.
``min_strike`` / ``max_strike`` are inherited from
``VolatilityTermStructure``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility_term_structure import VolatilityTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class CapFloorTermVolatilityStructure(VolatilityTermStructure):
    """Abstract cap/floor term-volatility structure.

    Subclasses MUST override ``max_date``, ``min_strike``,
    ``max_strike``, AND ``_volatility_impl(t, strike)``.
    """

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

    # --- subclass-implemented hook ---------------------------------------

    @abstractmethod
    def _volatility_impl(self, t: float, strike: float) -> float:
        """Subclass: return the volatility at ``(time, strike)``.

        Range and strike-range checks have already been performed.
        """

    # --- public API ------------------------------------------------------

    def volatility(
        self,
        maturity: Period | Date | float,
        strike: float,
        extrapolate: bool = False,
    ) -> float:
        """Cap/floor term volatility at ``(maturity, strike)``.

        # C++ parity: CapFloorTermVolatilityStructure::volatility (three
        # overloads — Period / Date / Time — collapse onto the Time
        # implementation here).
        """
        if isinstance(maturity, Period):
            d = self.option_date_from_tenor(maturity)
            return self.volatility(d, strike, extrapolate)
        if isinstance(maturity, Date):
            self.check_range(maturity, extrapolate)
            t = self.time_from_reference(maturity)
            self.check_strike(strike, extrapolate)
            return self._volatility_impl(t, strike)
        # Time-based.
        self.check_time_range(maturity, extrapolate)
        self.check_strike(strike, extrapolate)
        return self._volatility_impl(maturity, strike)
