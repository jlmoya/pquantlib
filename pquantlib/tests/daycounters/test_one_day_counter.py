"""Cross-validate OneDayCounter against the C++ probe.

Probe source: migration-harness/cpp/probes/daycounters/one_day_counter_probe.cpp
Reference:    migration-harness/references/daycounters/one/day/counter.json
              (path determined by generate-references.sh underscore→slash
              convention applied to the executable name).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.one_day_counter import OneDayCounter
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("daycounters/one/day/counter")


def test_name_matches_cpp(cpp: dict[str, Any]) -> None:
    assert OneDayCounter().name() == cpp["name"]


def test_day_count_matches_cpp(cpp: dict[str, Any]) -> None:
    dc = OneDayCounter()
    for case in cpp["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        assert dc.day_count(a, b) == int(case["day_count"]), case


def test_year_fraction_matches_cpp(cpp: dict[str, Any]) -> None:
    dc = OneDayCounter()
    for case in cpp["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        tolerance.exact(dc.year_fraction(a, b), float(case["year_fraction"]))


# --- Python-side coverage (DayCounter base behaviors) ---------------------


def test_daycounter_is_abstract() -> None:
    with pytest.raises(TypeError):
        DayCounter()  # type: ignore[abstract]


def test_equality_by_name() -> None:
    assert OneDayCounter() == OneDayCounter()


def test_hashable() -> None:
    assert len({OneDayCounter(), OneDayCounter()}) == 1


def test_repr_and_str() -> None:
    dc = OneDayCounter()
    assert repr(dc) == "OneDayCounter('1/1')"
    assert str(dc) == "1/1"
