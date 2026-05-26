"""Cross-validate SaudiArabia (Tadawul) holidays for 2020-2030 against the C++ probe.

Probe key: time/calendars/all -> "saudi_arabia"
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader
from pquantlib.time.calendars.saudi_arabia import SaudiArabia
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/calendars/all")


def test_name_matches_cpp(cpp: dict[str, Any]) -> None:
    assert SaudiArabia().name() == cpp["saudi_arabia"]["name"]


def test_holidays_match_cpp(cpp: dict[str, Any]) -> None:
    cal = SaudiArabia()
    expected = {
        Date.from_ymd(int(h["d"]), Month(int(h["m"])), int(h["y"])) for h in cpp["saudi_arabia"]["holidays"]
    }
    actual: set[Date] = set()
    for year in range(2020, 2031):
        for d in cal.holiday_list(
            Date.from_ymd(1, Month.January, year),
            Date.from_ymd(31, Month.December, year),
            include_weekends=False,
        ):
            actual.add(d)
    assert actual == expected, f"diff: missing={expected - actual}, extra={actual - expected}"
