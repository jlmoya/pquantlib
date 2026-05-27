"""InterestRateIndex — base for IBOR / swap / overnight indexes.

# C++ parity: ql/indexes/interestrateindex.hpp + .cpp (v1.42.1)

Stores the common (familyName, tenor, fixingDays, currency, fixingCalendar,
dayCounter) tuple and builds the index ``name`` deterministically as

    f"{familyName}{short_period(tenor)} {dayCounter.name}"

with the special case ``tenor == 1*Days`` rendering as ``ON`` / ``TN`` / ``SN``
for fixingDays 0 / 1 / 2 (matching C++ ``InterestRateIndex::InterestRateIndex``
in interestrateindex.cpp).

Subclasses must override ``maturity_date`` and ``forecast_fixing``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.index import Index
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _short_period(p: Period) -> str:
    """C++ parity: ql/time/period.cpp ``io::short_period_holder`` operator<<."""
    n = p.length
    u = p.units
    if u == TimeUnit.Days:
        return f"{n}D"
    if u == TimeUnit.Weeks:
        return f"{n}W"
    if u == TimeUnit.Months:
        return f"{n}M"
    if u == TimeUnit.Years:
        return f"{n}Y"
    qassert.fail(f"unknown time unit ({int(u)})")


class InterestRateIndex(Index):
    """Abstract base for interest-rate indexes (IBOR / overnight / swap).

    # C++ parity: ql/indexes/interestrateindex.hpp
    """

    def __init__(
        self,
        family_name: str,
        tenor: Period,
        fixing_days: int,
        currency: Currency,
        fixing_calendar: Calendar,
        day_counter: DayCounter,
    ) -> None:
        super().__init__()
        # Normalize ``N*Months`` with N % 12 == 0 to ``(N/12)*Years``
        # (C++ ql/indexes/interestrateindex.cpp lines 39-40).
        if tenor.units == TimeUnit.Months and tenor.length % 12 == 0:
            tenor = Period(tenor.length // 12, TimeUnit.Years)
        self._family_name: str = family_name
        self._tenor: Period = tenor
        self._fixing_days: int = fixing_days
        self._currency: Currency = currency
        self._fixing_calendar: Calendar = fixing_calendar
        self._day_counter: DayCounter = day_counter

        # Build name. C++ idiom: if tenor is 1 day, special-case ``ON``/``TN``/``SN``.
        suffix = ""
        if tenor.length == 1 and tenor.units == TimeUnit.Days:
            if fixing_days == 0:
                suffix = "ON"
            elif fixing_days == 1:
                suffix = "TN"
            elif fixing_days == 2:
                suffix = "SN"
            else:
                suffix = _short_period(tenor)
        else:
            suffix = _short_period(tenor)
        self._name: str = f"{family_name}{suffix} {day_counter.name()}"

    # --- Index interface -------------------------------------------------------

    def name(self) -> str:
        return self._name

    def fixing_calendar(self) -> Calendar:
        return self._fixing_calendar

    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        return self._fixing_calendar.is_business_day(fixing_date)

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        """C++ parity: InterestRateIndex::fixing in interestrateindex.cpp.

        The C++ code threads ``Settings::instance().evaluationDate()`` and
        ``enforcesTodaysHistoricFixings`` flag; PQuantLib doesn't yet expose
        an observable Settings module, so we behave as if today == fixing_date
        when ``forecast_todays_fixing`` is True (callers can opt in).
        """
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"Fixing date {fixing_date} is not valid",
        )
        # C++ parity: with no Settings module, we default to "always forecast"
        # when caller asks for it. Past-fixing lookup is still available via
        # explicit IndexManager use.
        if self.has_historical_fixing(fixing_date) and not forecast_todays_fixing:
            return self.past_fixing(fixing_date)
        return self.forecast_fixing(fixing_date)

    # --- inspectors ------------------------------------------------------------

    def family_name(self) -> str:
        return self._family_name

    def tenor(self) -> Period:
        return self._tenor

    def fixing_days(self) -> int:
        return self._fixing_days

    def currency(self) -> Currency:
        return self._currency

    def day_counter(self) -> DayCounter:
        return self._day_counter

    # --- date calculations -----------------------------------------------------

    def fixing_date(self, value_date: Date) -> Date:
        """Mirror C++ inline ``InterestRateIndex::fixingDate`` — advance ``-fixingDays``."""
        return self._fixing_calendar.advance(value_date, -self._fixing_days, TimeUnit.Days)

    def value_date(self, fixing_date: Date) -> Date:
        """Mirror C++ inline ``InterestRateIndex::valueDate`` — advance ``+fixingDays``."""
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"{fixing_date} is not a valid fixing date",
        )
        return self._fixing_calendar.advance(fixing_date, self._fixing_days, TimeUnit.Days)

    @abstractmethod
    def maturity_date(self, value_date: Date) -> Date:
        """Concrete subclass must define maturity calculation."""

    # --- fixing forecast -------------------------------------------------------

    @abstractmethod
    def forecast_fixing(self, fixing_date: Date) -> float:
        """Concrete subclass forecasts the index fixing from a term structure."""
