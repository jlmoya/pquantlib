"""Tests for pquantlib.patterns.observable_settings."""

from __future__ import annotations

from pquantlib.patterns.observable_settings import ObservableSettings


def test_observable_settings_is_singleton() -> None:
    a = ObservableSettings()
    b = ObservableSettings()
    assert a is b


def test_observable_settings_has_default_flags() -> None:
    s = ObservableSettings()
    # Defaults at first import (other tests may have mutated; assert
    # presence of attributes, not specific values).
    assert hasattr(s, "enforces_business_day_convention")
    assert hasattr(s, "include_today_in_payments")
    assert hasattr(s, "include_reference_date_events")


def test_observable_settings_mutation_visible_across_calls() -> None:
    a = ObservableSettings()
    original = a.include_today_in_payments
    try:
        a.include_today_in_payments = not original
        b = ObservableSettings()
        assert b.include_today_in_payments == (not original)
    finally:
        a.include_today_in_payments = original
