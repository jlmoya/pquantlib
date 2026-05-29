"""FlatHazardRate — constant hazard rate.

# C++ parity: ql/termstructures/credit/flathazardrate.{hpp,cpp} (v1.42.1).

S(t) = exp(-h * t); h(t) = h (constant); p(t) = h * S(t).

C++ has four constructors: (Date, Quote, dc), (Date, Rate, dc),
(settlement_days, Calendar, Quote, dc), (settlement_days, Calendar, Rate, dc).
The Python port collapses to a single keyword-driven ``__init__`` taking a
``Quote`` (and a classmethod ``from_rate`` wrapping a plain float in
SimpleQuote, plus optional jump support inherited from the abstract
``DefaultProbabilityTermStructure``).
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.hazard_rate_structure import HazardRateStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class FlatHazardRate(HazardRateStructure):
    """Constant hazard-rate term structure."""

    def __init__(
        self,
        reference_date: Date,
        hazard_rate: Quote,
        day_counter: DayCounter,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> None:
        """Construct from a date + Quote.

        # C++ parity: FlatHazardRate(Date, Handle<Quote>, dc).
        """
        super().__init__(
            reference_date=reference_date,
            calendar=None,
            day_counter=day_counter,
            jumps=jumps,
            jump_dates=jump_dates,
        )
        self._hazard_rate_quote: Quote = hazard_rate
        hazard_rate.register_with(self)

    @classmethod
    def from_rate(
        cls,
        reference_date: Date,
        hazard_rate: float,
        day_counter: DayCounter,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> FlatHazardRate:
        """Construct from a date + plain rate.

        # C++ parity: FlatHazardRate(Date, Rate, dc).
        """
        return cls(reference_date, SimpleQuote(hazard_rate), day_counter, jumps, jump_dates)

    @classmethod
    def with_settlement_days(
        cls,
        settlement_days: int,
        calendar: Calendar,
        hazard_rate: Quote,
        day_counter: DayCounter,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> FlatHazardRate:
        """Construct in moving (settlement-days) mode.

        # C++ parity: FlatHazardRate(Natural, Calendar, Handle<Quote>, dc).
        """
        instance = cls.__new__(cls)
        # Skip the default __init__ to set the moving-mode base directly.
        HazardRateStructure.__init__(
            instance,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
            jumps=jumps,
            jump_dates=jump_dates,
        )
        instance._hazard_rate_quote = hazard_rate
        hazard_rate.register_with(instance)
        return instance

    # ---- inspectors --------------------------------------------------------

    def max_date(self) -> Date:
        # C++ parity: ``Date::maxDate()``.
        return Date.max_date()

    # ---- HazardRateStructure overrides ------------------------------------

    def _hazard_rate_impl(self, t: float) -> float:
        del t  # constant in t
        return self._hazard_rate_quote.value()

    def _survival_probability_impl(self, t: float) -> float:
        """Closed-form S(t) = exp(-h * t).

        # C++ parity: flathazardrate.hpp:72-74 inline.
        """
        return math.exp(-self._hazard_rate_quote.value() * t)


__all__ = ["FlatHazardRate"]
