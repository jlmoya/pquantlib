"""Cross-validate Date arithmetic against the C++ probe.

Probe source: migration-harness/cpp/probes/time/date_probe.cpp
Reference:    migration-harness/references/time/date.json

Tolerance: all values are integer-valued under the hood, so EXACT tier
across the board.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader
from pquantlib.time.date import Date, is_leap
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib.time.weekday import Weekday


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/date")


# --- constants --------------------------------------------------------------


def test_min_max_serial_match_cpp(cpp: dict[str, Any]) -> None:
    c = cpp["constants"]
    assert Date.min_date().serial == int(c["min_serial"])
    assert Date.max_date().serial == int(c["max_serial"])
    assert Date.min_date().year() == int(c["min_y"])
    assert int(Date.min_date().month()) == int(c["min_m"])
    assert Date.min_date().day_of_month() == int(c["min_d"])
    assert Date.max_date().year() == int(c["max_y"])
    assert int(Date.max_date().month()) == int(c["max_m"])
    assert Date.max_date().day_of_month() == int(c["max_d"])


# --- is_leap (301 years) ---------------------------------------------------


def test_is_leap_matches_cpp_full_table(cpp: dict[str, Any]) -> None:
    table: dict[str, bool] = cpp["leap_years"]
    for year_str, expected in table.items():
        y = int(year_str)
        assert is_leap(y) is bool(expected), f"year {y}"
        # And via the Date class's static accessor:
        assert Date.is_leap(y) is bool(expected), f"year {y}"


# --- from_ymd → serial + weekday + day_of_year -----------------------------


def test_from_ymd_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["from_ymd"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        assert d.serial == int(case["serial"]), case
        assert d.weekday().name == case["weekday"], case
        assert d.day_of_year() == int(case["day_of_year"]), case


# --- from_serial → (y, m, d, weekday) --------------------------------------


def test_from_serial_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["from_serial"]:
        d = Date(int(case["serial"]))
        assert d.year() == int(case["y"]), case
        assert int(d.month()) == int(case["m"]), case
        assert d.day_of_month() == int(case["d"]), case
        assert d.weekday().name == case["weekday"], case
        assert d.day_of_year() == int(case["day_of_year"]), case


# --- arithmetic: Date + int days -------------------------------------------


def test_add_days_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["add_days"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        r = d + int(case["n"])
        assert isinstance(r, Date)
        assert r.serial == int(case["out_serial"]), case
        assert r.year() == int(case["out_y"]), case
        assert int(r.month()) == int(case["out_m"]), case
        assert r.day_of_month() == int(case["out_d"]), case


# --- arithmetic: Date + Period (months/years with EOM clipping) ------------


def test_add_period_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["add_period"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        units = TimeUnit[case["units"]]
        r = d + Period(int(case["n"]), units)
        assert isinstance(r, Date)
        assert r.year() == int(case["out_y"]), case
        assert int(r.month()) == int(case["out_m"]), case
        assert r.day_of_month() == int(case["out_d"]), case


# --- Date - Date difference ------------------------------------------------


def test_diff_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["diff"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        diff = a - b
        assert isinstance(diff, int)
        assert diff == int(case["diff"]), case


# --- end_of_month / is_end_of_month ----------------------------------------


def test_end_of_month_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["end_of_month"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        e = Date.end_of_month(d)
        assert e.day_of_month() == int(case["eom_d"]), case
        assert Date.is_end_of_month(d) is bool(case["is_eom"]), case


# --- start_of_month / is_start_of_month ------------------------------------


def test_start_of_month_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["start_of_month"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        s = Date.start_of_month(d)
        assert s.day_of_month() == int(case["som_d"]), case
        assert Date.is_start_of_month(d) is bool(case["is_som"]), case


# --- next_weekday ---------------------------------------------------------


def test_next_weekday_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["next_weekday"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        r = Date.next_weekday(d, Weekday[case["target"]])
        assert r.year() == int(case["out_y"]), case
        assert int(r.month()) == int(case["out_m"]), case
        assert r.day_of_month() == int(case["out_d"]), case


# --- nth_weekday ----------------------------------------------------------


def test_nth_weekday_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["nth_weekday"]:
        r = Date.nth_weekday(
            int(case["n"]),
            Weekday[case["w"]],
            Month(int(case["m"])),
            int(case["y"]),
        )
        assert r.day_of_month() == int(case["out_d"]), case


# --- Python-side behaviors (no probe needed) ------------------------------


def test_default_date_is_null() -> None:
    d = Date()
    assert d.serial == 0
    assert str(d) == "Date(null)"


def test_serial_out_of_range_raises() -> None:
    with pytest.raises(LibraryException, match="outside allowed range"):
        Date(1)  # below min
    with pytest.raises(LibraryException, match="outside allowed range"):
        Date(109575)  # above max


def test_from_ymd_year_out_of_range_raises() -> None:
    with pytest.raises(LibraryException, match="out of bound"):
        Date.from_ymd(1, Month.January, 1900)
    with pytest.raises(LibraryException, match="out of bound"):
        Date.from_ymd(1, Month.January, 2200)


def test_from_ymd_day_out_of_range_raises() -> None:
    with pytest.raises(LibraryException, match="day outside month"):
        Date.from_ymd(30, Month.February, 2024)  # leap year only has 29
    with pytest.raises(LibraryException, match="day outside month"):
        Date.from_ymd(0, Month.January, 2024)


def test_nth_weekday_zero_or_six_raises() -> None:
    with pytest.raises(LibraryException, match="zeroth day of week"):
        Date.nth_weekday(0, Weekday.Monday, Month.January, 2024)
    with pytest.raises(LibraryException, match="no more than 5"):
        Date.nth_weekday(6, Weekday.Monday, Month.January, 2024)


def test_ordering() -> None:
    a = Date.from_ymd(1, Month.January, 2024)
    b = Date.from_ymd(2, Month.January, 2024)
    a_copy = Date.from_ymd(1, Month.January, 2024)
    assert a < b
    assert b > a
    assert a <= a_copy
    assert a >= a_copy


def test_hashable_and_equality() -> None:
    a = Date.from_ymd(15, Month.March, 2024)
    b = Date.from_ymd(15, Month.March, 2024)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_str_format() -> None:
    d = Date.from_ymd(15, Month.March, 2024)
    assert str(d) == "2024-03-15"


def test_subtract_int_returns_date() -> None:
    d = Date.from_ymd(15, Month.March, 2024)
    r = d - 14
    assert isinstance(r, Date)
    assert str(r) == "2024-03-01"


def test_radd_int() -> None:
    d = Date.from_ymd(15, Month.March, 2024)
    r = 5 + d
    assert isinstance(r, Date)
    assert str(r) == "2024-03-20"


def test_todays_date_is_in_valid_range() -> None:
    t = Date.todays_date()
    assert Date.min_date() <= t <= Date.max_date()


def test_excel_1900_leap_bug_is_preserved() -> None:
    # Year 1900 is invalid as a Date constructor input (year must be ≥ 1901),
    # but is_leap(1900) must still report True for serial-table compatibility.
    assert is_leap(1900) is True
    assert Date.is_leap(1900) is True
