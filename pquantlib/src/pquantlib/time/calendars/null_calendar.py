"""NullCalendar — every day is a business day, no weekends.

# C++ parity: ql/time/calendars/nullcalendar.hpp (v1.42.1).
"""

from __future__ import annotations

from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.weekday import Weekday


class NullCalendar(Calendar):
    """Every day is a business day. Used for theoretical-calculation tests."""

    def name(self) -> str:
        return "Null"

    def _is_weekend(self, w: Weekday) -> bool:
        _ = w
        return False

    def _is_business_day(self, d: Date) -> bool:
        _ = d
        return True
