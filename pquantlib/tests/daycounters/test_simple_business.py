"""Cross-validate SimpleDayCounter + Business252 against the C++ probe.

Probe source: migration-harness/cpp/probes/daycounters/simple_business_probe.cpp
Reference:    migration-harness/references/daycounters/simple/business.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.business_252 import Business252
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.simple_day_counter import SimpleDayCounter
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("daycounters/simple/business")


def _check_section(dc: DayCounter, section: dict[str, Any]) -> None:
    assert dc.name() == section["name"]
    for case in section["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        assert dc.day_count(a, b) == int(case["day_count"]), case
        tolerance.tight(dc.year_fraction(a, b), float(case["year_fraction"]))


def test_simple_day_counter_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(SimpleDayCounter(), cpp["simple"])


def test_business252_with_weekends_only_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Business252(WeekendsOnly()), cpp["business252_weekends"])


# --- Python-side coverage -----------------------------------------------


def test_simple_whole_month_clean() -> None:
    dc = SimpleDayCounter()
    # 6 months on the 15th → exactly 0.5
    a = Date.from_ymd(15, Month.March, 2024)
    b = Date.from_ymd(15, Month.September, 2024)
    tolerance.exact(dc.year_fraction(a, b), 0.5)


def test_simple_one_year_clean() -> None:
    dc = SimpleDayCounter()
    a = Date.from_ymd(15, Month.March, 2024)
    b = Date.from_ymd(15, Month.March, 2025)
    tolerance.exact(dc.year_fraction(a, b), 1.0)


def test_business252_name_includes_calendar_name() -> None:
    dc = Business252(WeekendsOnly())
    assert "weekends only" in dc.name()


def test_business252_equality_by_name() -> None:
    assert Business252(WeekendsOnly()) == Business252(WeekendsOnly())
    # Different calendar names → different Business252 names → not equal.
    assert Business252(WeekendsOnly()) != Business252(NullCalendar())
