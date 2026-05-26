"""Cross-validate Actual360 / Actual364 / Actual36525 / Actual365Fixed /
Actual366 against the C++ probe.

Probe source: migration-harness/cpp/probes/daycounters/actual_family_probe.cpp
Reference:    migration-harness/references/daycounters/actual/family.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_364 import Actual364
from pquantlib.daycounters.actual_365_25 import Actual36525
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed, Convention
from pquantlib.daycounters.actual_366 import Actual366
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("daycounters/actual/family")


def _check_section(dc: DayCounter, section: dict[str, Any], *, check_day_count: bool = True) -> None:
    assert dc.name() == section["name"]
    for case in section["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        if check_day_count and "day_count" in case:
            assert dc.day_count(a, b) == int(case["day_count"]), case
        tolerance.tight(dc.year_fraction(a, b), float(case["year_fraction"]))


# --- Actual360 -----------------------------------------------------------


def test_actual360_default_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual360(), cpp["actual360"])


def test_actual360_include_last_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual360(include_last_day=True), cpp["actual360_inc"])


# --- Actual364 -----------------------------------------------------------


def test_actual364_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual364(), cpp["actual364"])


# --- Actual36525 ---------------------------------------------------------


def test_actual36525_default_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual36525(), cpp["actual36525"])


def test_actual36525_include_last_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual36525(include_last_day=True), cpp["actual36525_inc"])


# --- Actual365Fixed Standard --------------------------------------------


def test_actual365fixed_standard_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual365Fixed(Convention.Standard), cpp["actual365fixed_standard"])


# --- Actual365Fixed NoLeap ----------------------------------------------


def test_actual365fixed_no_leap_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual365Fixed(Convention.NoLeap), cpp["actual365fixed_no_leap"])


# --- Actual366 ----------------------------------------------------------


def test_actual366_default_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual366(), cpp["actual366"])


def test_actual366_include_last_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Actual366(include_last_day=True), cpp["actual366_inc"])


# --- Actual365Fixed Canadian (needs ref period) -------------------------


def test_actual365fixed_canadian_matches_cpp(cpp: dict[str, Any]) -> None:
    dc = Actual365Fixed(Convention.Canadian)
    assert dc.name() == cpp["actual365fixed_canadian"]["name"]
    for case in cpp["actual365fixed_canadian"]["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        rs = Date.from_ymd(int(case["rs_d"]), Month(int(case["rs_m"])), int(case["rs_y"]))
        re = Date.from_ymd(int(case["re_d"]), Month(int(case["re_m"])), int(case["re_y"]))
        tolerance.tight(dc.year_fraction(a, b, rs, re), float(case["year_fraction"]))


def test_actual365fixed_canadian_missing_ref_raises() -> None:
    dc = Actual365Fixed(Convention.Canadian)
    a = Date.from_ymd(15, Month.March, 2024)
    b = Date.from_ymd(15, Month.April, 2024)
    with pytest.raises(LibraryException, match="invalid refPeriodStart"):
        dc.year_fraction(a, b)


# --- Python-side equality + name distinctness ---------------------------


def test_include_last_changes_equality() -> None:
    assert Actual360() != Actual360(include_last_day=True)
    assert Actual360() == Actual360()


def test_convention_changes_actual365fixed_equality() -> None:
    assert Actual365Fixed(Convention.Standard) != Actual365Fixed(Convention.NoLeap)
    assert Actual365Fixed(Convention.Standard) != Actual365Fixed(Convention.Canadian)
    assert Actual365Fixed(Convention.NoLeap) == Actual365Fixed(Convention.NoLeap)
