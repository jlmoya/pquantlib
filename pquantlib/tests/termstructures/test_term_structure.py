"""Tests for pquantlib.termstructures.term_structure (TermStructure abstract)."""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class _StubTS(TermStructure):
    """Minimal concrete TermStructure for behavior tests."""

    def __init__(
        self,
        *,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: object | None = None,
        max_date: Date | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,  # type: ignore[arg-type]
        )
        self._max_date_override: Date = (
            max_date if max_date is not None else Date.from_ymd(31, Month.December, 2050)
        )

    def max_date(self) -> Date:
        return self._max_date_override


def test_cannot_instantiate_abstract_term_structure() -> None:
    with pytest.raises(TypeError):
        TermStructure()  # type: ignore[abstract]


def test_fixed_reference_date_returned() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    ts = _StubTS(reference_date=ref, day_counter=Actual360())
    assert ts.reference_date() == ref


def test_reference_date_required_when_not_overridden() -> None:
    ts = _StubTS(day_counter=Actual360())
    with pytest.raises(LibraryException, match="reference date not provided"):
        ts.reference_date()


def test_day_counter_required() -> None:
    ts = _StubTS(reference_date=Date.from_ymd(15, Month.June, 2026))
    with pytest.raises(LibraryException, match="day counter not provided"):
        ts.day_counter()


def test_calendar_required() -> None:
    ts = _StubTS(reference_date=Date.from_ymd(15, Month.June, 2026), day_counter=Actual360())
    with pytest.raises(LibraryException, match="calendar not provided"):
        ts.calendar()


def test_time_from_reference_uses_day_counter() -> None:
    ref = Date.from_ymd(1, Month.January, 2026)
    target = Date.from_ymd(1, Month.July, 2026)
    ts = _StubTS(reference_date=ref, day_counter=Actual360())
    # Actual/360: (Jul 1 - Jan 1) days / 360 = 181 / 360
    assert ts.time_from_reference(target) == 181.0 / 360.0


def test_check_range_rejects_date_before_reference() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    ts = _StubTS(reference_date=ref, day_counter=Actual360())
    with pytest.raises(LibraryException, match="before reference date"):
        ts.check_range(Date.from_ymd(1, Month.January, 2026), extrapolate=False)


def test_check_range_rejects_date_past_max_when_no_extrapolation() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    ts = _StubTS(
        reference_date=ref,
        day_counter=Actual360(),
        max_date=Date.from_ymd(15, Month.June, 2027),
    )
    with pytest.raises(LibraryException, match="past max curve date"):
        ts.check_range(Date.from_ymd(15, Month.June, 2030), extrapolate=False)


def test_check_range_accepts_date_past_max_when_extrapolation_allowed() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    ts = _StubTS(
        reference_date=ref,
        day_counter=Actual360(),
        max_date=Date.from_ymd(15, Month.June, 2027),
    )
    ts.enable_extrapolation()
    # No exception.
    ts.check_range(Date.from_ymd(15, Month.June, 2030), extrapolate=False)


def test_update_notifies_observers() -> None:
    ts = _StubTS(reference_date=Date.from_ymd(15, Month.June, 2026), day_counter=Actual360())
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    ts.register_with(obs)
    ts.update()
    assert counts[0] == 1


def test_calendar_returned_when_provided() -> None:
    cal = WeekendsOnly()
    ts = _StubTS(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=cal,
        day_counter=Actual360(),
    )
    assert ts.calendar() is cal
