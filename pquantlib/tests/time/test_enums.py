"""Cross-validate time-layer enum integer values against the C++ probe.

Probe source: migration-harness/cpp/probes/time/enums_probe.cpp
Reference:    migration-harness/references/time/enums.json
"""

from __future__ import annotations

import pytest

from pquantlib.testing import reference_reader
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit
from pquantlib.time.weekday import Weekday


@pytest.fixture(scope="module")
def cpp_enums() -> dict[str, dict[str, int]]:
    return reference_reader.load("time/enums")


# --- Weekday -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
)
def test_weekday_long_name_matches_cpp(name: str, cpp_enums: dict[str, dict[str, int]]) -> None:
    assert int(Weekday[name]) == cpp_enums["Weekday"][name]


@pytest.mark.parametrize(
    ("short", "long"),
    [
        ("Sun", "Sunday"),
        ("Mon", "Monday"),
        ("Tue", "Tuesday"),
        ("Wed", "Wednesday"),
        ("Thu", "Thursday"),
        ("Fri", "Friday"),
        ("Sat", "Saturday"),
    ],
)
def test_weekday_short_alias_resolves_to_long_member(short: str, long: str) -> None:
    # IntEnum aliases: Weekday.Sun and Weekday.Sunday are the same canonical member.
    assert Weekday[short] is Weekday[long]


# --- Month -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
)
def test_month_long_name_matches_cpp(name: str, cpp_enums: dict[str, dict[str, int]]) -> None:
    assert int(Month[name]) == cpp_enums["Month"][name]


@pytest.mark.parametrize(
    ("short", "long"),
    [
        ("Jan", "January"),
        ("Feb", "February"),
        ("Mar", "March"),
        ("Apr", "April"),
        ("Jun", "June"),
        ("Jul", "July"),
        ("Aug", "August"),
        ("Sep", "September"),
        ("Oct", "October"),
        ("Nov", "November"),
        ("Dec", "December"),
    ],
)
def test_month_short_alias_resolves_to_long_member(short: str, long: str) -> None:
    assert Month[short] is Month[long]


# --- TimeUnit ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Days",
        "Weeks",
        "Months",
        "Years",
        "Hours",
        "Minutes",
        "Seconds",
        "Milliseconds",
        "Microseconds",
    ],
)
def test_time_unit_matches_cpp(name: str, cpp_enums: dict[str, dict[str, int]]) -> None:
    assert int(TimeUnit[name]) == cpp_enums["TimeUnit"][name]


# --- Frequency ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "NoFrequency",
        "Once",
        "Annual",
        "Semiannual",
        "EveryFourthMonth",
        "Quarterly",
        "Bimonthly",
        "Monthly",
        "EveryFourthWeek",
        "Biweekly",
        "Weekly",
        "Daily",
        "OtherFrequency",
    ],
)
def test_frequency_matches_cpp(name: str, cpp_enums: dict[str, dict[str, int]]) -> None:
    assert int(Frequency[name]) == cpp_enums["Frequency"][name]


# --- BusinessDayConvention ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Following",
        "ModifiedFollowing",
        "Preceding",
        "ModifiedPreceding",
        "Unadjusted",
        "HalfMonthModifiedFollowing",
        "Nearest",
    ],
)
def test_business_day_convention_matches_cpp(name: str, cpp_enums: dict[str, dict[str, int]]) -> None:
    assert int(BusinessDayConvention[name]) == cpp_enums["BusinessDayConvention"][name]


# --- DateGeneration ----------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Backward",
        "Forward",
        "Zero",
        "ThirdWednesday",
        "ThirdWednesdayInclusive",
        "Twentieth",
        "TwentiethIMM",
        "OldCDS",
        "CDS",
        "CDS2015",
    ],
)
def test_date_generation_matches_cpp(name: str, cpp_enums: dict[str, dict[str, int]]) -> None:
    assert int(DateGeneration[name]) == cpp_enums["DateGenerationRule"][name]
