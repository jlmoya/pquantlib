"""VolatilityTermStructure — abstract base for vol-anchored term structures.

# C++ parity: ql/termstructures/voltermstructure.hpp + voltermstructure.cpp (v1.42.1).

C++ defines three construction modes (delegated, fixed reference date,
moving reference date with settlement-days offset from the global
evaluation date). PQuantLib ports modes 1 and 2 only — mode 3 needs an
``ObservableSettings.evaluation_date`` observer wiring that's deferred
along with the same mode in the L2-A base ``TermStructure``.

Subclasses MUST override:

- ``max_date()`` (inherited from ``TermStructure``)
- ``min_strike()`` / ``max_strike()`` (new in this layer)

``business_day_convention()`` is stored on the instance and used by
``option_date_from_tenor`` to advance the reference date by the requested
``Period`` (swaption-style).
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class VolatilityTermStructure(TermStructure):
    """Abstract volatility term structure.

    Adds ``business_day_convention()``, ``option_date_from_tenor()``, and
    ``min_strike()`` / ``max_strike()`` to the base ``TermStructure`` API.
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
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._business_day_convention: BusinessDayConvention = business_day_convention

    def business_day_convention(self) -> BusinessDayConvention:
        return self._business_day_convention

    def option_date_from_tenor(self, p: Period) -> Date:
        """Convert a tenor ``Period`` to a date by advancing the reference date.

        # C++ parity: VolatilityTermStructure::optionDateFromTenor
        """
        return self.calendar().advance_period(
            self.reference_date(), p, self._business_day_convention
        )

    @abstractmethod
    def min_strike(self) -> float:
        """Smallest strike value for which the structure can return vols."""

    @abstractmethod
    def max_strike(self) -> float:
        """Largest strike value for which the structure can return vols."""

    def check_strike(self, strike: float, extrapolate: bool) -> None:
        """Range-check ``strike`` against ``[min_strike, max_strike]``.

        # C++ parity: VolatilityTermStructure::checkStrike
        """
        qassert.require(
            extrapolate
            or self.allows_extrapolation()
            or (self.min_strike() <= strike <= self.max_strike()),
            f"strike ({strike}) is outside the curve domain "
            f"[{self.min_strike()},{self.max_strike()}]",
        )
