"""Tests for pquantlib.indexes.index (Index abstract base)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.indexes.index import Index
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class _StubIndex(Index):
    """Minimal concrete Index for behavior tests."""

    def __init__(self, name: str = "STUB", forecast: float = 0.0) -> None:
        super().__init__()
        self._name = name
        self._calendar = WeekendsOnly()
        self._forecast = forecast

    def name(self) -> str:
        return self._name

    def fixing_calendar(self) -> Calendar:
        return self._calendar

    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        return self._calendar.is_business_day(fixing_date)

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        del forecast_todays_fixing
        return self._forecast


@pytest.fixture(autouse=True)
def _clean_manager() -> None:  # pyright: ignore[reportUnusedFunction]  # pytest auto-uses
    IndexManager().clear_histories()


def test_cannot_instantiate_abstract_index() -> None:
    with pytest.raises(TypeError):
        Index()  # type: ignore[abstract]


def test_add_fixing_persists_via_manager() -> None:
    idx = _StubIndex()
    # 2026-06-15 is a Monday — business day in WeekendsOnly.
    d = Date.from_ymd(15, Month.June, 2026)
    idx.add_fixing(d, 0.035)
    assert idx.has_historical_fixing(d) is True
    assert IndexManager().has_historical_fixing("STUB", d) is True


def test_past_fixing_returns_stored_value() -> None:
    idx = _StubIndex()
    d = Date.from_ymd(15, Month.June, 2026)
    idx.add_fixing(d, 0.035)
    assert idx.past_fixing(d) == 0.035


def test_past_fixing_rejects_invalid_date() -> None:
    idx = _StubIndex()
    # Sunday — invalid in WeekendsOnly.
    d = Date.from_ymd(14, Month.June, 2026)
    with pytest.raises(LibraryException, match="not a valid fixing date"):
        idx.past_fixing(d)


def test_past_fixing_raises_when_no_history() -> None:
    idx = _StubIndex()
    d = Date.from_ymd(15, Month.June, 2026)
    with pytest.raises(LibraryException, match="no past fixing"):
        idx.past_fixing(d)


def test_clear_fixings_removes_index_history() -> None:
    idx = _StubIndex()
    d = Date.from_ymd(15, Month.June, 2026)
    idx.add_fixing(d, 0.035)
    idx.clear_fixings()
    assert idx.has_historical_fixing(d) is False


def test_update_notifies_observers() -> None:
    idx = _StubIndex()
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()  # keep a strong reference (Observable holds via WeakSet).
    idx.register_with(obs)
    idx.update()
    assert counts[0] == 1


def test_time_series_returns_index_manager_history() -> None:
    idx = _StubIndex()
    d = Date.from_ymd(15, Month.June, 2026)
    idx.add_fixing(d, 0.035)
    ts = idx.time_series()
    assert ts[d] == 0.035
