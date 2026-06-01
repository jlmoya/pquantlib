"""BMAIndex — Bond Market Association (SIFMA) short-term tax-exempt index.

# C++ parity: ql/indexes/bmaindex.{hpp,cpp} (v1.42.1, 099987f0).

The BMA index has tenor one week, is fixed weekly on Wednesdays, and is
applied with a one-day fixing gap from Thursdays on for one week. It is the
tax-exempt correspondent of the 1M USD-Libor.

Construction fixes the C++ defaults:
``InterestRateIndex("BMA", 1*Weeks, fixingDays=1, USDCurrency(),
UnitedStates(GovernmentBond), ActualActual(ISDA))``.

Python divergences from C++:

- ``forecast_fixing`` requires a forwarding term structure (as in C++). With
  a fully-known fixing history (the W12-C test surface) it is never reached.
- ``Handle`` indirection is replaced by a direct optional term-structure
  reference (consistent with the rest of this port's index hierarchy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.schedule import Schedule


def _previous_wednesday(date: Date) -> Date:
    """The Wednesday on/before ``date`` (rolling a full week back if needed).

    # C++ parity: anonymous ``previousWednesday`` (bmaindex.cpp:29-35).
    Weekday integer values match C++ (Sunday=1 .. Saturday=7, Wednesday=4).
    """
    w = int(date.weekday())
    if w >= 4:
        return date - (w - 4)
    # roll forward 4-w days and back one week
    return date + (4 - w - 7)


def _next_wednesday(date: Date) -> Date:
    """The Wednesday strictly after ``date``. C++ parity: bmaindex.cpp:37-39."""
    return _previous_wednesday(date + 7)


class BMAIndex(InterestRateIndex):
    """Bond Market Association weekly-Wednesday tax-exempt index.

    # C++ parity: ``BMAIndex`` in bmaindex.hpp:40-65.
    """

    def __init__(self, term_structure: YieldTermStructure | None = None) -> None:
        super().__init__(
            "BMA",
            Period(1, TimeUnit.Weeks),
            1,  # fixing_days
            USDCurrency(),
            UnitedStates(UnitedStates.Market.GovernmentBond),
            ActualActual(Convention.ISDA),
        )
        self._term_structure: YieldTermStructure | None = term_structure

    # --- Index interface -----------------------------------------------

    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        """BMA is fixed weekly on Wednesdays (or the next business day after a
        holiday-spanning Wednesday).

        # C++ parity: ``BMAIndex::isValidFixingDate`` (bmaindex.cpp:54-65).
        """
        cal = self.fixing_calendar()
        # either the fixing date is last Wednesday, or all days between last
        # Wednesday (included) and the fixing date are holidays.
        d = _previous_wednesday(fixing_date)
        while d < fixing_date:
            if cal.is_business_day(d):
                return False
            d = d + 1
        return cal.is_business_day(fixing_date)

    # --- inspectors ----------------------------------------------------

    def forwarding_term_structure(self) -> YieldTermStructure | None:
        """C++ parity: ``BMAIndex::forwardingTermStructure`` (bmaindex.cpp:67-69)."""
        return self._term_structure

    # --- date calculations ---------------------------------------------

    def maturity_date(self, value_date: Date) -> Date:
        """One-week maturity (next Wednesday + 1 day from the value date).

        # C++ parity: ``BMAIndex::maturityDate`` (bmaindex.cpp:71-76).
        """
        cal = self.fixing_calendar()
        fixing_date = cal.advance(value_date, -1, TimeUnit.Days)
        next_wed = _previous_wednesday(fixing_date + 7)
        return cal.advance(next_wed, 1, TimeUnit.Days)

    def fixing_schedule(self, start: Date, end: Date) -> Schedule:
        """Weekly schedule of fixing dates between ``start`` and ``end``.

        # C++ parity: ``BMAIndex::fixingSchedule`` (bmaindex.cpp:78-85).
        """
        from pquantlib.time.schedule import MakeSchedule  # noqa: PLC0415 (lazy: avoids cycle)

        return (
            MakeSchedule()
            .from_date(_previous_wednesday(start))
            .to(_next_wednesday(end))
            .with_frequency(Frequency.Weekly)
            .with_calendar(self.fixing_calendar())
            .with_convention(BusinessDayConvention.Following)
            .forwards()
            .build()
        )

    # --- fixing forecast -----------------------------------------------

    def forecast_fixing(self, fixing_date: Date) -> float:
        """Simple forward rate over the one-week period implied by the curve.

        # C++ parity: ``BMAIndex::forecastFixing`` (bmaindex.cpp:87-95).
        """
        ts = self._term_structure
        qassert.require(
            ts is not None,
            f"null term structure set to this instance of {self.name()}",
        )
        assert ts is not None
        cal = self.fixing_calendar()
        start = cal.advance(fixing_date, 1, TimeUnit.Days)
        end = self.maturity_date(start)
        return ts.forward_rate(
            start, end, Compounding.Simple, Frequency.Annual, False, self._day_counter
        ).rate()
