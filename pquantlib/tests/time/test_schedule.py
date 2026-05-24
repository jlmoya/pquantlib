"""Cross-validate Schedule generation against the C++ probe.

Probe source: migration-harness/cpp/probes/time/schedule_probe.cpp
Reference:    migration-harness/references/time/schedule.json

All output is integer/boolean — EXACT tier.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule, Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/schedule")


# --- Common inputs (mirrors probe) ----------------------------------------


def _eff() -> Date:
    return Date.from_ymd(15, Month.March, 2024)


def _term() -> Date:
    return Date.from_ymd(15, Month.March, 2026)


def _cal() -> WeekendsOnly:
    return WeekendsOnly()


# --- Each rule's dates + is_regular --------------------------------------


def _check_section(
    s: Schedule,
    section: dict[str, Any],
) -> None:
    serials = [d.serial for d in s.dates]
    assert serials == list(section["dates"]), section
    if "is_regular" in section:
        if section["is_regular"]:
            assert list(s.is_regular) == [bool(x) for x in section["is_regular"]], section
        else:
            assert not s.has_is_regular() or list(s.is_regular) == []


def test_zero_rule_matches_cpp(cpp: dict[str, Any]) -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(6, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Zero,
        end_of_month=False,
    )
    _check_section(s, cpp["zero"])


def test_backward_rule_matches_cpp(cpp: dict[str, Any]) -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(6, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        end_of_month=False,
    )
    _check_section(s, cpp["backward"])


def test_forward_rule_matches_cpp(cpp: dict[str, Any]) -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(6, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Forward,
        end_of_month=False,
    )
    _check_section(s, cpp["forward"])


def test_third_wednesday_rule_matches_cpp(cpp: dict[str, Any]) -> None:
    eff = Date.from_ymd(20, Month.March, 2024)
    term = Date.from_ymd(17, Month.December, 2025)
    s = Schedule.from_rule(
        eff,
        term,
        Period(3, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.ThirdWednesday,
        end_of_month=False,
    )
    _check_section(s, cpp["third_wednesday"])


def test_twentieth_rule_matches_cpp(cpp: dict[str, Any]) -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(3, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Twentieth,
        end_of_month=False,
    )
    _check_section(s, cpp["twentieth"])


def test_cds2015_rule_matches_cpp(cpp: dict[str, Any]) -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(3, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.CDS2015,
        end_of_month=False,
    )
    _check_section(s, cpp["cds2015"])


# --- Truncation ------------------------------------------------------------


def test_after_truncation_matches_cpp(cpp: dict[str, Any]) -> None:
    full = Schedule.from_rule(
        _eff(),
        _term(),
        Period(6, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        end_of_month=False,
    )
    trunc = Date.from_ymd(15, Month.September, 2024)
    serials = [d.serial for d in full.after(trunc).dates]
    assert serials == list(cpp["after"]["dates"])


def test_until_truncation_matches_cpp(cpp: dict[str, Any]) -> None:
    full = Schedule.from_rule(
        _eff(),
        _term(),
        Period(6, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        end_of_month=False,
    )
    trunc = Date.from_ymd(15, Month.September, 2024)
    serials = [d.serial for d in full.until(trunc).dates]
    assert serials == list(cpp["until"]["dates"])


# --- previous_date / next_date -------------------------------------------


def test_previous_next_date_matches_cpp(cpp: dict[str, Any]) -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(6, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        end_of_month=False,
    )
    probe = Date.from_ymd(1, Month.July, 2024)
    pn = cpp["prev_next"]
    assert probe.serial == int(pn["probe_serial"])
    assert s.previous_date(probe).serial == int(pn["prev_serial"])
    assert s.next_date(probe).serial == int(pn["next_serial"])


# --- Backward with first_date --------------------------------------------


def test_backward_with_first_date_matches_cpp(cpp: dict[str, Any]) -> None:
    first = Date.from_ymd(15, Month.April, 2024)
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(6, TimeUnit.Months),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        end_of_month=False,
        first_date=first,
    )
    _check_section(s, cpp["backward_with_first_date"])


# --- Date-list constructor -----------------------------------------------


def test_date_list_constructor_matches_cpp(cpp: dict[str, Any]) -> None:
    dates = [
        Date.from_ymd(15, Month.March, 2024),
        Date.from_ymd(17, Month.September, 2024),
        Date.from_ymd(17, Month.March, 2025),
        Date.from_ymd(15, Month.September, 2025),
        Date.from_ymd(16, Month.March, 2026),
    ]
    s = Schedule(dates)
    serials = [d.serial for d in s.dates]
    assert serials == list(cpp["date_list_ctor"]["dates"])


# --- MakeSchedule builder ------------------------------------------------


def test_make_schedule_basic() -> None:
    s = (
        MakeSchedule()
        .from_date(_eff())
        .to(_term())
        .with_tenor(Period(6, TimeUnit.Months))
        .with_calendar(_cal())
        .with_convention(BusinessDayConvention.Following)
        .backwards()
        .build()
    )
    assert len(s) == 5
    assert s.front() == _eff()
    # _term() is 2026-03-15 (Sun); Following adjusts it to Mon 2026-03-16.
    assert s.back() == Date.from_ymd(16, Month.March, 2026)


def test_make_schedule_missing_effective_date_raises() -> None:
    with pytest.raises(LibraryException, match="effective date not provided"):
        MakeSchedule().to(_term()).with_tenor(Period(1, TimeUnit.Years)).build()


def test_make_schedule_missing_tenor_raises() -> None:
    with pytest.raises(LibraryException, match="tenor/frequency not provided"):
        MakeSchedule().from_date(_eff()).to(_term()).build()


# --- Python-side behaviors -----------------------------------------------


def test_schedule_iterable_and_indexable() -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(1, TimeUnit.Years),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        end_of_month=False,
    )
    by_iter = list(s)
    by_index = [s[i] for i in range(len(s))]
    assert by_iter == by_index


def test_inspectors_round_trip() -> None:
    s = Schedule.from_rule(
        _eff(),
        _term(),
        Period(1, TimeUnit.Years),
        _cal(),
        BusinessDayConvention.Following,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        end_of_month=True,
    )
    assert s.has_tenor()
    assert s.tenor == Period(1, TimeUnit.Years)
    assert s.has_rule()
    assert s.rule == DateGeneration.Backward
    assert s.has_termination_date_business_day_convention()
    assert s.termination_date_business_day_convention == BusinessDayConvention.ModifiedFollowing
    assert s.business_day_convention == BusinessDayConvention.Following
    assert s.has_end_of_month()
    # 1Y tenor allows EOM.
    assert s.end_of_month is True


def test_null_effective_date_raises() -> None:
    with pytest.raises(LibraryException, match="null effective date"):
        Schedule.from_rule(
            Date(),
            _term(),
            Period(1, TimeUnit.Years),
            _cal(),
            BusinessDayConvention.Following,
            BusinessDayConvention.Following,
            DateGeneration.Backward,
            end_of_month=False,
        )


def test_termination_before_effective_raises() -> None:
    with pytest.raises(LibraryException, match="later than or equal"):
        Schedule.from_rule(
            _term(),
            _eff(),
            Period(1, TimeUnit.Years),
            _cal(),
            BusinessDayConvention.Following,
            BusinessDayConvention.Following,
            DateGeneration.Backward,
            end_of_month=False,
        )
