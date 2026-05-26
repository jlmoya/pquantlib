"""TermStructure — abstract base for all time-anchored curve types.

# C++ parity: ql/termstructure.hpp + ql/termstructure.cpp (v1.42.1)

C++ defines three construction modes:

1. **Delegated**: ``TermStructure(DayCounter)``. Subclass must override
   ``reference_date()``.
2. **Fixed**: ``TermStructure(Date, Calendar, DayCounter)``. Reference
   date is stored once and never recomputed.
3. **Moving**: ``TermStructure(settlementDays, Calendar, DayCounter)``.
   Reference date = ``calendar.advance(today, settlementDays, Days)``
   where ``today`` comes from ``Settings::instance().evaluationDate()``;
   the term structure registers as an observer so it invalidates when
   the evaluation date moves.

PQuantLib pilot ports modes 1 and 2 only. **Mode 3 is deferred** — it
needs an ``Settings.evaluation_date`` observable that isn't yet wired in
PQuantLib's ``ObservableSettings``. The first L2-B subclass that needs
moving mode (likely ``FlatForward``) will add the wiring at that time;
``TermStructure`` will then grow a ``settlement_days`` keyword constructor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.patterns.observer import Observable
from pquantlib.termstructures.extrapolator import Extrapolator
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class TermStructure(Observable, Extrapolator, ABC):
    """Basic term-structure functionality.

    Subclasses must override ``max_date()``. ``reference_date()`` has a
    default that returns the fixed reference date passed at construction;
    in the delegated mode (no reference date supplied), the subclass must
    also override ``reference_date()``.
    """

    def __init__(
        self,
        *,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        Observable.__init__(self)
        Extrapolator.__init__(self)
        self._reference_date: Date | None = reference_date
        self._calendar: Calendar | None = calendar
        self._day_counter: DayCounter | None = day_counter

    def day_counter(self) -> DayCounter:
        qassert.require(self._day_counter is not None, "day counter not provided")
        assert self._day_counter is not None
        return self._day_counter

    def calendar(self) -> Calendar:
        qassert.require(self._calendar is not None, "calendar not provided")
        assert self._calendar is not None
        return self._calendar

    def reference_date(self) -> Date:
        """Date at which discount = 1 / variance = 0.

        Default implementation returns the fixed reference date passed at
        construction. Subclasses using the delegated construction mode
        must override this.
        """
        qassert.require(
            self._reference_date is not None,
            "reference date not provided — subclass must override reference_date()",
        )
        assert self._reference_date is not None
        return self._reference_date

    @abstractmethod
    def max_date(self) -> Date:
        """Latest date for which the curve can return values."""

    def max_time(self) -> float:
        return self.day_counter().year_fraction(self.reference_date(), self.max_date())

    def time_from_reference(self, d: Date) -> float:
        return self.day_counter().year_fraction(self.reference_date(), d)

    def update(self) -> None:
        """Observer.update — propagates to own observers."""
        self.notify_observers()

    def check_range(self, d: Date, extrapolate: bool) -> None:
        """Raise unless ``d`` is within the curve's valid range."""
        qassert.require(
            d >= self.reference_date(),
            f"date ({d}) before reference date ({self.reference_date()})",
        )
        qassert.require(
            extrapolate or self.allows_extrapolation() or d <= self.max_date(),
            f"date ({d}) is past max curve date ({self.max_date()})",
        )

    def check_time_range(self, t: float, extrapolate: bool) -> None:
        """Raise unless ``t`` is within the curve's valid time range."""
        qassert.require(t >= 0.0, f"negative time ({t}) given")
        qassert.require(
            extrapolate or self.allows_extrapolation() or t <= self.max_time(),
            f"time ({t}) is past max curve time ({self.max_time()})",
        )
