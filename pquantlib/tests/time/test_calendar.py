"""Cross-validate Calendar abstract + 4 trivial concretes against the C++ probe.

Probe source: migration-harness/cpp/probes/time/calendar_probe.cpp
Reference:    migration-harness/references/time/calendar.json

All comparisons are integer/boolean — EXACT tier throughout.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.bespoke_calendar import BespokeCalendar
from pquantlib.time.calendars.joint_calendar import (
    JointCalendar,
    JointCalendarRule,
)
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit
from pquantlib.time.weekday import Weekday


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/calendar")


# --- names -----------------------------------------------------------------


def test_calendar_names_match_cpp(cpp: dict[str, Any]) -> None:
    n = cpp["names"]
    assert NullCalendar().name() == n["Null"]
    assert WeekendsOnly().name() == n["WeekendsOnly"]
    jh = JointCalendar([NullCalendar(), WeekendsOnly()], JointCalendarRule.JoinHolidays)
    jbd = JointCalendar([NullCalendar(), WeekendsOnly()], JointCalendarRule.JoinBusinessDays)
    assert jh.name() == n["JoinHolidays_Null_WeekendsOnly"]
    assert jbd.name() == n["JoinBusinessDays_Null_WeekendsOnly"]
    assert BespokeCalendar("MyBespoke").name() == n["Bespoke_named"]
    assert BespokeCalendar().name() == n["Bespoke_anon"]


# --- per-day classification ------------------------------------------------


def _check_week(cal: Calendar, table: list[dict[str, Any]]) -> None:
    for row in table:
        d = Date.from_ymd(int(row["d"]), Month.March, 2024)
        assert cal.is_business_day(d) is bool(row["is_bd"]), row
        assert cal.is_holiday(d) is bool(row["is_hol"]), row
        assert cal.is_weekend(d.weekday()) is bool(row["is_we"]), row


def test_null_week_classification(cpp: dict[str, Any]) -> None:
    _check_week(NullCalendar(), cpp["null_week"])


def test_weekends_only_week_classification(cpp: dict[str, Any]) -> None:
    _check_week(WeekendsOnly(), cpp["weekends_only_week"])


# --- JointCalendar --------------------------------------------------------


def test_joint_calendar_classification(cpp: dict[str, Any]) -> None:
    jc_h = JointCalendar([NullCalendar(), WeekendsOnly()], JointCalendarRule.JoinHolidays)
    jc_b = JointCalendar([NullCalendar(), WeekendsOnly()], JointCalendarRule.JoinBusinessDays)
    sat = Date.from_ymd(16, Month.March, 2024)
    mon = Date.from_ymd(11, Month.March, 2024)
    jc = cpp["joint_calendar"]
    assert jc_h.is_business_day(sat) is bool(jc["sat_join_holidays_is_bd"])
    assert jc_b.is_business_day(sat) is bool(jc["sat_join_business_days_is_bd"])
    assert jc_h.is_business_day(mon) is bool(jc["mon_join_holidays_is_bd"])
    assert jc_b.is_business_day(mon) is bool(jc["mon_join_business_days_is_bd"])


# --- BespokeCalendar ------------------------------------------------------


def test_bespoke_calendar_custom_weekends(cpp: dict[str, Any]) -> None:
    bc = BespokeCalendar("MidEast")
    bc.add_weekend(Weekday.Friday)
    bc.add_weekend(Weekday.Sunday)
    b = cpp["bespoke"]
    assert bc.is_business_day(Date.from_ymd(15, Month.March, 2024)) is bool(b["fri_is_bd"])
    assert bc.is_business_day(Date.from_ymd(16, Month.March, 2024)) is bool(b["sat_is_bd"])
    assert bc.is_business_day(Date.from_ymd(17, Month.March, 2024)) is bool(b["sun_is_bd"])


# --- adjust ---------------------------------------------------------------


def test_adjust_matches_cpp(cpp: dict[str, Any]) -> None:
    weo = WeekendsOnly()
    sat = Date.from_ymd(16, Month.March, 2024)
    for case in cpp["adjust"]:
        r = weo.adjust(sat, BusinessDayConvention[case["convention"]])
        assert r.year() == int(case["out_y"]), case
        assert int(r.month()) == int(case["out_m"]), case
        assert r.day_of_month() == int(case["out_d"]), case


def test_adjust_mod_following_month_cross_matches_cpp(cpp: dict[str, Any]) -> None:
    weo = WeekendsOnly()
    d = Date.from_ymd(30, Month.March, 2024)
    r = weo.adjust(d, BusinessDayConvention.ModifiedFollowing)
    expected = cpp["adjust_mod_following_month_cross"]
    assert r.year() == int(expected["out_y"])
    assert int(r.month()) == int(expected["out_m"])
    assert r.day_of_month() == int(expected["out_d"])


# --- advance --------------------------------------------------------------


def test_advance_matches_cpp(cpp: dict[str, Any]) -> None:
    weo = WeekendsOnly()
    for case in cpp["advance"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        r = weo.advance(
            d,
            int(case["n"]),
            TimeUnit[case["units"]],
            BusinessDayConvention[case["convention"]],
            bool(case["end_of_month"]),
        )
        assert r.year() == int(case["out_y"]), case
        assert int(r.month()) == int(case["out_m"]), case
        assert r.day_of_month() == int(case["out_d"]), case


# --- business_days_between ------------------------------------------------


def test_business_days_between_matches_cpp(cpp: dict[str, Any]) -> None:
    weo = WeekendsOnly()
    d1 = Date.from_ymd(11, Month.March, 2024)
    d2 = Date.from_ymd(18, Month.March, 2024)
    b = cpp["business_days_between"]
    assert weo.business_days_between(d1, d2, True, False) == b["mon_to_next_mon_incl_first_excl_last"]
    assert weo.business_days_between(d1, d2, True, True) == b["mon_to_next_mon_incl_first_incl_last"]
    assert weo.business_days_between(d1, d2, False, False) == b["mon_to_next_mon_excl_first_excl_last"]
    assert weo.business_days_between(d2, d1, True, False) == b["reversed_direction"]


# --- holiday_list ---------------------------------------------------------


def test_holiday_list_weekends_only_matches_cpp(cpp: dict[str, Any]) -> None:
    weo = WeekendsOnly()
    a = Date.from_ymd(11, Month.March, 2024)
    b = Date.from_ymd(24, Month.March, 2024)
    holidays = weo.holiday_list(a, b, include_weekends=True)
    days_of_month = [d.day_of_month() for d in holidays]
    assert days_of_month == list(cpp["holiday_list_weo"])


# --- Python-side behaviors (no probe) -------------------------------------


def test_add_and_remove_holiday() -> None:
    weo = WeekendsOnly()
    extra = Date.from_ymd(15, Month.March, 2024)  # a Friday
    assert weo.is_business_day(extra) is True
    weo.add_holiday(extra)
    assert weo.is_business_day(extra) is False
    weo.remove_holiday(extra)
    assert weo.is_business_day(extra) is True


def test_remove_holiday_force_business_day_on_weekend() -> None:
    weo = WeekendsOnly()
    sat = Date.from_ymd(16, Month.March, 2024)
    assert weo.is_business_day(sat) is False
    weo.remove_holiday(sat)
    assert weo.is_business_day(sat) is True
    weo.add_holiday(sat)  # revert
    # add_holiday on what is now a business-day-by-override removes the override + re-adds
    # the original weekend rule.
    assert weo.is_business_day(sat) is False


def test_reset_added_and_removed_holidays() -> None:
    weo = WeekendsOnly()
    sat = Date.from_ymd(16, Month.March, 2024)
    fri = Date.from_ymd(15, Month.March, 2024)
    weo.remove_holiday(sat)
    weo.add_holiday(fri)
    weo.reset_added_and_removed_holidays()
    assert weo.added_holidays == frozenset()
    assert weo.removed_holidays == frozenset()


def test_calendar_equality_by_name() -> None:
    assert NullCalendar() == NullCalendar()
    assert NullCalendar() != WeekendsOnly()


def test_calendar_hashable() -> None:
    s = {NullCalendar(), NullCalendar(), WeekendsOnly()}
    assert len(s) == 2


def test_calendar_repr() -> None:
    r = repr(NullCalendar())
    assert "NullCalendar" in r
    assert "Null" in r


def test_start_of_month_and_end_of_month_with_weekend_adjust() -> None:
    weo = WeekendsOnly()
    # March 2024: 1st is Friday (business day), 31st is Sunday (weekend).
    d = Date.from_ymd(15, Month.March, 2024)
    assert weo.start_of_month(d) == Date.from_ymd(1, Month.March, 2024)
    # End-of-month: March 31 (Sun) adjusted backwards via Preceding = Fri 29.
    assert weo.end_of_month(d) == Date.from_ymd(29, Month.March, 2024)
