"""Tests for pquantlib.indexes.index_manager (singleton fixing repository)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_series import TimeSeries


@pytest.fixture(autouse=True)
def _clean_manager() -> None:  # pyright: ignore[reportUnusedFunction]  # pytest auto-uses
    """Reset the singleton between tests (otherwise state leaks)."""
    IndexManager().clear_histories()


def test_singleton_returns_same_instance() -> None:
    assert IndexManager() is IndexManager()


def test_has_history_false_before_any_fixing() -> None:
    assert IndexManager().has_history("EURIBOR3M") is False


def test_add_fixing_creates_history() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    assert IndexManager().has_history("EURIBOR3M") is True


def test_get_history_returns_stored_values() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    h = IndexManager().get_history("EURIBOR3M")
    assert h[d] == 0.035


def test_name_lookup_is_case_insensitive() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    h = IndexManager().get_history("euribor3m")
    assert h[d] == 0.035


def test_add_fixing_rejects_duplicate_with_different_value() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    with pytest.raises(LibraryException, match="duplicated fixing"):
        IndexManager().add_fixing("EURIBOR3M", d, 0.04)


def test_add_fixing_idempotent_with_same_value() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)  # no raise
    assert IndexManager().get_history("EURIBOR3M")[d] == 0.035


def test_force_overwrite_replaces_value() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    IndexManager().add_fixing("EURIBOR3M", d, 0.04, force_overwrite=True)
    assert IndexManager().get_history("EURIBOR3M")[d] == 0.04


def test_clear_history_removes_one_index() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    IndexManager().add_fixing("USDLIBOR3M", d, 0.05)
    IndexManager().clear_history("EURIBOR3M")
    assert IndexManager().has_history("EURIBOR3M") is False
    assert IndexManager().has_history("USDLIBOR3M") is True


def test_clear_histories_removes_all() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    IndexManager().add_fixing("USDLIBOR3M", d, 0.05)
    IndexManager().clear_histories()
    assert IndexManager().histories() == []


def test_set_history_replaces_existing() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    fresh = TimeSeries[float]()
    fresh[d] = 0.04
    IndexManager().set_history("EURIBOR3M", fresh)
    assert IndexManager().get_history("EURIBOR3M")[d] == 0.04


def test_has_historical_fixing_checks_date_presence() -> None:
    d = Date.from_ymd(15, Month.June, 2026)
    other = Date.from_ymd(16, Month.June, 2026)
    IndexManager().add_fixing("EURIBOR3M", d, 0.035)
    assert IndexManager().has_historical_fixing("EURIBOR3M", d) is True
    assert IndexManager().has_historical_fixing("EURIBOR3M", other) is False
