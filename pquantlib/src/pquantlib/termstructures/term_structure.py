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

All three modes are now supported. Moving mode requires both
``settlement_days`` and ``calendar``. The TS registers with
``ObservableSettings()`` so it auto-invalidates on eval-date changes;
``reference_date()`` is then computed lazily on first call after each
notification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.termstructures.extrapolator import Extrapolator
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit


class TermStructure(Observable, Extrapolator, ABC):
    """Basic term-structure functionality.

    Subclasses must override ``max_date()``. ``reference_date()`` has a
    default that returns:

    * the fixed reference date passed at construction (fixed mode), OR
    * ``calendar.advance(evaluation_date_or_today, settlement_days, Days)``
      recomputed lazily on each ``update()`` (moving mode), OR
    * the value supplied by an override in the delegated mode.
    """

    def __init__(
        self,
        *,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        settlement_days: int | None = None,
    ) -> None:
        Observable.__init__(self)
        Extrapolator.__init__(self)
        self._reference_date: Date | None = reference_date
        self._calendar: Calendar | None = calendar
        self._day_counter: DayCounter | None = day_counter
        self._settlement_days: int | None = settlement_days
        # Moving mode flag + cache validity. In C++ these are ``moving_``
        # and ``updated_`` (mutable). The ``_moving`` flag is read-only;
        # ``_reference_date_cache_valid`` flips on ``update()`` and is
        # re-set in ``reference_date()``.
        self._moving: bool = settlement_days is not None
        self._reference_date_cache_valid: bool = not self._moving
        if self._moving:
            qassert.require(
                calendar is not None,
                "calendar is required in moving (settlement_days) mode",
            )
            ObservableSettings().register_with(self)

    def day_counter(self) -> DayCounter:
        qassert.require(self._day_counter is not None, "day counter not provided")
        assert self._day_counter is not None
        return self._day_counter

    def calendar(self) -> Calendar:
        qassert.require(self._calendar is not None, "calendar not provided")
        assert self._calendar is not None
        return self._calendar

    def settlement_days(self) -> int:
        """Settlement days used for moving-mode reference-date calculation.

        # C++ parity: ``TermStructure::settlementDays()``. Raises if the
        # term structure was constructed in fixed or delegated mode.
        """
        qassert.require(
            self._settlement_days is not None,
            "settlement days not provided for this instance",
        )
        assert self._settlement_days is not None
        return self._settlement_days

    def reference_date(self) -> Date:
        """Date at which discount = 1 / variance = 0.

        Resolution priority:
        - Moving mode (``settlement_days != None``): recomputed lazily
          as ``calendar.advance(evaluation_date_or_today, settlement_days,
          Days)`` each time the cache is invalidated by ``update()``.
        - Fixed mode (constructed with ``reference_date=Date``): returns
          the stored date.
        - Delegated mode: subclasses must override this method.
        """
        if self._moving:
            if not self._reference_date_cache_valid:
                assert self._calendar is not None
                assert self._settlement_days is not None
                today = ObservableSettings().evaluation_date_or_today()
                self._reference_date = self._calendar.advance(
                    today,
                    self._settlement_days,
                    TimeUnit.Days,
                    BusinessDayConvention.Following,
                )
                self._reference_date_cache_valid = True
            assert self._reference_date is not None
            return self._reference_date
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
        """Observer.update — invalidate moving cache + propagate.

        # C++ parity: ``TermStructure::update`` sets ``updated_=false``
        # when moving and then calls ``notifyObservers()``.
        """
        if self._moving:
            self._reference_date_cache_valid = False
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
