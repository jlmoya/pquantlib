"""Tests for pquantlib.patterns.observable_settings."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date, Month


def test_observable_settings_is_singleton() -> None:
    a = ObservableSettings()
    b = ObservableSettings()
    assert a is b


def test_observable_settings_has_default_flags() -> None:
    s = ObservableSettings()
    # Defaults at first import (other tests may have mutated; assert
    # presence of attributes, not specific values).
    assert hasattr(s, "enforces_business_day_convention")
    assert hasattr(s, "include_reference_date_events")
    assert hasattr(s, "include_todays_cash_flows")
    assert hasattr(s, "enforces_todays_historic_fixings")


def test_class_defaults_match_cpp() -> None:
    """C++ ``Settings`` field initialisers (ql/settings.hpp:112-116).

    ``includeReferenceDateEvents_ = false``, ``includeTodaysCashFlows_`` an UNSET
    ``ext::optional<bool>`` and ``enforcesTodaysHistoricFixings_ = false``. Read
    off the class, not an instance, so a mutating test cannot mask a regression.
    """
    assert ObservableSettings.include_reference_date_events is False
    assert ObservableSettings.include_todays_cash_flows is None
    assert ObservableSettings.enforces_todays_historic_fixings is False


def test_observable_settings_mutation_visible_across_calls() -> None:
    a = ObservableSettings()
    original = a.enforces_todays_historic_fixings
    try:
        a.enforces_todays_historic_fixings = not original
        b = ObservableSettings()
        assert b.enforces_todays_historic_fixings == (not original)
    finally:
        a.enforces_todays_historic_fixings = original


# --- evaluation_date observable wiring ---------------------------------


@pytest.fixture(autouse=True)
def _reset_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Ensure each test starts with evaluation_date cleared."""
    s = ObservableSettings()
    s.evaluation_date = None
    yield
    s.evaluation_date = None


class _Counter:
    """Tiny Observer that counts update() calls."""

    def __init__(self) -> None:
        self.count: int = 0

    def update(self) -> None:
        self.count += 1


def test_observable_settings_is_an_observable() -> None:
    s = ObservableSettings()
    assert isinstance(s, Observable)


def test_evaluation_date_default_is_none() -> None:
    s = ObservableSettings()
    assert s.evaluation_date is None


def test_evaluation_date_or_today_returns_today_when_unpinned() -> None:
    s = ObservableSettings()
    assert s.evaluation_date is None
    assert s.evaluation_date_or_today() == Date.todays_date()


def test_evaluation_date_or_today_returns_pinned_date_when_pinned() -> None:
    s = ObservableSettings()
    pinned = Date.from_ymd(15, Month.June, 2026)
    s.evaluation_date = pinned
    assert s.evaluation_date_or_today() == pinned


def test_setting_evaluation_date_notifies_observers() -> None:
    s = ObservableSettings()
    c = _Counter()
    s.register_with(c)
    s.evaluation_date = Date.from_ymd(15, Month.June, 2026)
    assert c.count == 1
    s.evaluation_date = Date.from_ymd(16, Month.June, 2026)
    assert c.count == 2


def test_setting_evaluation_date_to_same_value_is_no_op() -> None:
    s = ObservableSettings()
    c = _Counter()
    d = Date.from_ymd(15, Month.June, 2026)
    s.evaluation_date = d
    s.register_with(c)
    s.evaluation_date = d  # same value → no notification
    assert c.count == 0


def test_clearing_evaluation_date_notifies_observers() -> None:
    s = ObservableSettings()
    s.evaluation_date = Date.from_ymd(15, Month.June, 2026)
    c = _Counter()
    s.register_with(c)
    s.evaluation_date = None
    assert c.count == 1


def test_evaluation_date_setter_round_trip() -> None:
    s = ObservableSettings()
    d = Date.from_ymd(1, Month.January, 2026)
    s.evaluation_date = d
    assert s.evaluation_date == d
    s.evaluation_date = None
    assert s.evaluation_date is None
