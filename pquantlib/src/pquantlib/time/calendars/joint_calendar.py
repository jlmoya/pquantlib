"""JointCalendar — combine multiple calendars under JoinHolidays / JoinBusinessDays.

# C++ parity: ql/time/calendars/jointcalendar.hpp + .cpp (v1.42.1).

C++ exposes the join rule as an unscoped enum ``JointCalendarRule``;
Python uses an ``IntEnum`` for stronger typing and parity with the
other time-layer enums.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.weekday import Weekday


class JointCalendarRule(IntEnum):
    JoinHolidays = 0
    JoinBusinessDays = 1


class JointCalendar(Calendar):
    """Joint calendar formed from a list of underlying calendars.

    ``JoinHolidays``: a date is a holiday for the joint calendar iff it is
    a holiday for ANY of the underlying calendars.

    ``JoinBusinessDays``: a date is a business day iff it is a business day
    for ANY of the underlying calendars.
    """

    def __init__(
        self,
        calendars: Sequence[Calendar],
        rule: JointCalendarRule = JointCalendarRule.JoinHolidays,
    ) -> None:
        super().__init__()
        qassert.require(len(calendars) >= 1, "JointCalendar needs at least one calendar")
        self._calendars: tuple[Calendar, ...] = tuple(calendars)
        self._rule: JointCalendarRule = rule

    def name(self) -> str:
        prefix = "JoinHolidays" if self._rule == JointCalendarRule.JoinHolidays else "JoinBusinessDays"
        return f"{prefix}({', '.join(c.name() for c in self._calendars)})"

    def _is_weekend(self, w: Weekday) -> bool:
        if self._rule == JointCalendarRule.JoinHolidays:
            return any(c.is_weekend(w) for c in self._calendars)
        return all(c.is_weekend(w) for c in self._calendars)

    def _is_business_day(self, d: Date) -> bool:
        if self._rule == JointCalendarRule.JoinHolidays:
            return all(not c.is_holiday(d) for c in self._calendars)
        return any(c.is_business_day(d) for c in self._calendars)
