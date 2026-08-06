"""Tests for ql/settings.{hpp,cpp} ``SavedSettings``.

Every expected value comes from ``migration-harness/cpp/probes/v143_root_tail/probe.cpp``
(``emitSavedSettings``) run against C++ QuantLib v1.43; the reference lives at
``migration-harness/references/v143/root/tail.json``.

The probe sets ``Settings::instance().evaluationDate()`` in ``emitSavedSettings``
(`Settings::instance().evaluationDate() = EVAL_DATE;`, first statement of the
"Case 1" block). Every global it touches is pinned here by the
``_pinned_settings`` autouse fixture and restored in teardown, so the module is
NOT wall-clock dependent — except for the one assertion that is explicitly ABOUT
today's date, which the probe itself reduced to a boolean for that reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.patterns.observable_settings import ObservableSettings, SavedSettings
from pquantlib.testing import reference_reader
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/root/tail")


@pytest.fixture(autouse=True)
def _pinned_settings(cpp: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved_date = settings.evaluation_date
    saved_irde = settings.include_reference_date_events
    saved_itcf = settings.include_todays_cash_flows
    saved_ethf = settings.enforces_todays_historic_fixings
    settings.evaluation_date = Date(int(cpp["has_occurred"]["eval_date_serial"]))
    try:
        yield
    finally:
        settings.evaluation_date = saved_date
        settings.include_reference_date_events = saved_irde
        settings.include_todays_cash_flows = saved_itcf
        settings.enforces_todays_historic_fixings = saved_ethf


def test_restores_every_field(cpp: dict[str, Any]) -> None:
    """C++ ``SavedSettings`` snapshots and restores exactly four fields."""
    exp = cpp["saved_settings"]["pinned_restore"]
    settings = ObservableSettings()
    settings.evaluation_date = Date(int(cpp["has_occurred"]["eval_date_serial"]))
    settings.include_reference_date_events = False
    settings.include_todays_cash_flows = None
    settings.enforces_todays_historic_fixings = False

    with SavedSettings():
        settings.evaluation_date = Date.from_ymd(1, Month.December, 2030)
        settings.include_reference_date_events = True
        settings.include_todays_cash_flows = True
        settings.enforces_todays_historic_fixings = True

    assert settings.evaluation_date_or_today().serial_number() == exp["evaluation_date"]
    assert settings.include_reference_date_events is exp["include_reference_date_events"]
    assert settings.include_todays_cash_flows is exp["include_todays_cash_flows"]
    assert settings.enforces_todays_historic_fixings is exp["enforces_todays_historic_fixings"]


def test_optional_false_is_restored_as_false(cpp: dict[str, Any]) -> None:
    """C++ ``includeTodaysCashFlows`` is ``ext::optional<bool>``; False != unset."""
    settings = ObservableSettings()
    settings.include_todays_cash_flows = False
    with SavedSettings():
        settings.include_todays_cash_flows = None
    assert settings.include_todays_cash_flows is cpp["saved_settings"]["optional_false_restore"]


def test_optional_unset_is_restored_as_unset(cpp: dict[str, Any]) -> None:
    settings = ObservableSettings()
    settings.include_todays_cash_flows = None
    with SavedSettings():
        settings.include_todays_cash_flows = True
    assert settings.include_todays_cash_flows is cpp["saved_settings"]["optional_unset_restore"]


def test_unset_evaluation_date_comes_back_anchored(cpp: dict[str, Any]) -> None:
    """C++ snapshots the RESOLVED date, so an unset date restores ANCHORED.

    The C++ ctor reads ``Settings::instance().evaluationDate()`` through
    ``DateProxy::operator Date()`` (ql/settings.hpp:135-140), which maps the null
    date to ``Date::todaysDate()``. The probe reduced this to a boolean because
    "today" is not reproducible in a reference file.
    """
    assert cpp["saved_settings"]["unset_entry_restores_to_today"] is True
    settings = ObservableSettings()
    settings.reset_evaluation_date()
    today_at_entry = Date.todays_date()
    with SavedSettings():
        settings.evaluation_date = Date.from_ymd(1, Month.December, 2030)
    assert settings.evaluation_date == today_at_entry


def test_restore_is_callable_without_the_context_manager(cpp: dict[str, Any]) -> None:
    """C++ shape is an object whose destructor restores; expose that too."""
    settings = ObservableSettings()
    settings.include_reference_date_events = False
    guard = SavedSettings()
    settings.include_reference_date_events = True
    guard.restore()
    assert settings.include_reference_date_events is False


def test_anchor_and_reset_evaluation_date() -> None:
    """C++ ``Settings::anchorEvaluationDate`` / ``resetEvaluationDate``."""
    settings = ObservableSettings()
    pinned = Date.from_ymd(3, Month.March, 2025)
    settings.evaluation_date = pinned
    settings.anchor_evaluation_date()  # no-op when already set (ql/settings.cpp:38-43)
    assert settings.evaluation_date == pinned

    settings.reset_evaluation_date()
    assert settings.evaluation_date is None
    settings.anchor_evaluation_date()
    assert settings.evaluation_date == Date.todays_date()
