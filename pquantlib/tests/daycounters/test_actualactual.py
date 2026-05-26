"""Cross-validate ActualActual (7 conventions, 4 impls) against the C++ probe.

Probe source: migration-harness/cpp/probes/daycounters/actualactual_probe.cpp
Reference:    migration-harness/references/daycounters/actualactual.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("daycounters/actualactual")


def _check_section(dc: DayCounter, section: dict[str, Any]) -> None:
    assert dc.name() == section["name"]
    for case in section["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        tolerance.tight(dc.year_fraction(a, b), float(case["year_fraction"]))


# --- ISDA / Historical / Actual365 (share ISDA_Impl) ---------------------


_ISDA_FAMILY = [
    ("isda", Convention.ISDA),
    ("historical", Convention.Historical),
    ("actual365", Convention.Actual365),
]


@pytest.mark.parametrize(("key", "convention"), _ISDA_FAMILY)
def test_isda_family_matches_cpp(key: str, convention: Convention, cpp: dict[str, Any]) -> None:
    _check_section(ActualActual(convention), cpp[key])


# --- AFB / Euro (share AFB_Impl) -----------------------------------------


_AFB_FAMILY = [
    ("afb", Convention.AFB),
    ("euro", Convention.Euro),
]


@pytest.mark.parametrize(("key", "convention"), _AFB_FAMILY)
def test_afb_family_matches_cpp(key: str, convention: Convention, cpp: dict[str, Any]) -> None:
    _check_section(ActualActual(convention), cpp[key])


# --- ISMA / Bond without schedule (Old_ISMA_Impl) ------------------------


def test_isma_no_schedule_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(ActualActual(Convention.ISMA), cpp["isma_no_schedule"])


def test_bond_no_schedule_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(ActualActual(Convention.Bond), cpp["bond_no_schedule"])


# --- ISMA with schedule (ISMA_Impl) -------------------------------------


def test_isma_with_schedule_matches_cpp(cpp: dict[str, Any]) -> None:
    # Reconstruct the same schedule used in the probe: 2024-01-01 → 2025-01-01,
    # 6-month tenor, NullCalendar, Unadjusted, Forward rule.
    s = Schedule.from_rule(
        Date.from_ymd(1, Month.January, 2024),
        Date.from_ymd(1, Month.January, 2025),
        Period(6, TimeUnit.Months),
        NullCalendar(),
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Forward,
        end_of_month=False,
    )
    section = cpp["isma_with_schedule"]
    # Sanity-check schedule shape matches probe.
    assert [d.serial for d in s.dates] == list(section["schedule_dates_serials"])

    dc = ActualActual(Convention.ISMA, schedule=s)
    _check_section(dc, section)


# --- Python-side coverage -----------------------------------------------


def test_isda_family_aliases_share_name() -> None:
    n = ActualActual(Convention.ISDA).name()
    assert ActualActual(Convention.Historical).name() == n
    assert ActualActual(Convention.Actual365).name() == n


def test_afb_family_aliases_share_name() -> None:
    n = ActualActual(Convention.AFB).name()
    assert ActualActual(Convention.Euro).name() == n


def test_isma_and_bond_share_name() -> None:
    assert ActualActual(Convention.ISMA).name() == ActualActual(Convention.Bond).name()


def test_isma_backward_negates_year_fraction() -> None:
    dc = ActualActual(Convention.ISDA)
    a = Date.from_ymd(1, Month.January, 2024)
    b = Date.from_ymd(1, Month.July, 2024)
    fwd = dc.year_fraction(a, b)
    bwd = dc.year_fraction(b, a)
    tolerance.tight(fwd, -bwd)


def test_isma_zero_same_day() -> None:
    dc = ActualActual(Convention.ISMA)
    d = Date.from_ymd(1, Month.January, 2024)
    assert dc.year_fraction(d, d) == 0.0
