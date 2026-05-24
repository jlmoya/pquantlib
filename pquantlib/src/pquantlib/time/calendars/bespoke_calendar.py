"""BespokeCalendar — user-configured weekends + holidays.

# C++ parity: ql/time/calendars/bespokecalendar.hpp + .cpp (v1.42.1).

Caller adds weekends with ``add_weekend(weekday)`` and one-off holidays
with ``add_holiday(date)`` (inherited from Calendar). All non-weekend
non-holiday dates are business days.
"""

from __future__ import annotations

from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.weekday import Weekday


class BespokeCalendar(Calendar):
    def __init__(self, name: str = "") -> None:
        super().__init__()
        self._name: str = name
        self._weekend_mask: int = 0  # bit i set <=> Weekday(i) is a weekend day

    def name(self) -> str:
        return self._name

    def add_weekend(self, w: Weekday) -> None:
        self._weekend_mask |= 1 << int(w)

    def _is_weekend(self, w: Weekday) -> bool:
        return bool(self._weekend_mask & (1 << int(w)))

    def _is_business_day(self, d: Date) -> bool:
        return not self._is_weekend(d.weekday())
