"""WeekendsOnly — Sat+Sun are holidays, all other days are business days.

# C++ parity: ql/time/calendars/weekendsonly.hpp (v1.42.1).

Note: C++ ``WeekendsOnly().name()`` returns the lowercase string
``"weekends only"`` (with a space, not capitalized) — preserved here.
"""

from __future__ import annotations

from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.weekday import Weekday


class WeekendsOnly(Calendar):
    def name(self) -> str:
        return "weekends only"

    def _is_weekend(self, w: Weekday) -> bool:
        return w in (Weekday.Saturday, Weekday.Sunday)

    def _is_business_day(self, d: Date) -> bool:
        return not self._is_weekend(d.weekday())
