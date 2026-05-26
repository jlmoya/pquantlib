"""Cross-validate South Korea holidays for 2020-2030 against the C++ probe.

Probe key: time/calendars/all -> "south_korea"
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader
from pquantlib.time.calendars.south_korea import SouthKorea
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/calendars/all")


def test_name_matches_cpp(cpp: dict[str, Any]) -> None:
    assert SouthKorea().name() == cpp["south_korea"]["name"]


def test_holidays_match_cpp(cpp: dict[str, Any]) -> None:
    cal = SouthKorea()
    expected = {
        Date.from_ymd(int(h["d"]), Month(int(h["m"])), int(h["y"])) for h in cpp["south_korea"]["holidays"]
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
