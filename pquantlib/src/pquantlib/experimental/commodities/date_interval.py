"""DateInterval — a [start, end] date interval helper.

# C++ parity: ql/experimental/commodities/dateinterval.hpp +
#             dateinterval.cpp (v1.42.1).

Supports membership tests (``is_date_between``), intersection, and equality.
``end >= start`` is enforced at construction.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.time.date import Date


class DateInterval:
    """A closed date interval ``[start_date, end_date]`` with ``end >= start``."""

    def __init__(
        self,
        start_date: Date | None = None,
        end_date: Date | None = None,
    ) -> None:
        if start_date is None and end_date is None:
            self._start_date: Date = Date()
            self._end_date: Date = Date()
            return
        assert start_date is not None
        assert end_date is not None
        qassert.require(end_date >= start_date, "end date must be >= start date")
        self._start_date = start_date
        self._end_date = end_date

    @property
    def start_date(self) -> Date:
        return self._start_date

    @property
    def end_date(self) -> Date:
        return self._end_date

    def is_date_between(
        self,
        date: Date,
        include_first: bool = True,
        include_last: bool = True,
    ) -> bool:
        """True if ``date`` lies in the interval (parity with ``isDateBetween``)."""
        # Faithful port of the C++ branch structure (note: the lower-bound
        # ``else if`` only fires when include_first is False, and likewise for
        # the upper bound).
        if include_first and not (date >= self._start_date):
            return False
        if not include_first and not (date > self._start_date):
            return False
        if include_last and not (date <= self._end_date):
            return False
        if not include_last and not (date < self._end_date):  # noqa: SIM103
            return False
        return True

    def intersection(self, di: DateInterval) -> DateInterval:
        """Intersection with ``di`` (empty interval if disjoint)."""
        if (
            self._start_date < di._start_date and self._end_date < di._start_date
        ) or (self._start_date > di._end_date and self._end_date > di._end_date):
            return DateInterval()
        return DateInterval(
            max(self._start_date, di._start_date),
            min(self._end_date, di._end_date),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DateInterval):
            return NotImplemented
        return (
            self._start_date == other._start_date
            and self._end_date == other._end_date
        )

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash((self._start_date, self._end_date))

    def __str__(self) -> str:
        if self._start_date == Date() or self._end_date == Date():
            return "Null<DateInterval>()"
        return f"{self._start_date} to {self._end_date}"

    def __repr__(self) -> str:
        return f"DateInterval({self.__str__()!r})"
