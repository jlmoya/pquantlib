"""Cross-validate parsers against the C++ probe.

Probe source: migration-harness/cpp/probes/time/parsers_probe.cpp
Reference:    migration-harness/references/time/parsers.json

The C++ ``DateParser.parseFormatted`` is not cross-validated — see the
date_parser module docstring for the documented divergence.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader
from pquantlib.time import date_parser, period_parser
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/parsers")


# --- PeriodParser.parse -----------------------------------------------------


def test_period_parse_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["period_parse"]:
        p = period_parser.parse(str(case["input"]))
        assert p.length == int(case["length"]), case
        assert p.units.name == case["units"], case


def test_period_parse_too_short_raises() -> None:
    with pytest.raises(LibraryException, match="at least 2"):
        period_parser.parse("")
    with pytest.raises(LibraryException, match="at least 2"):
        period_parser.parse("M")


def test_period_parse_no_unit_letter_raises() -> None:
    with pytest.raises(LibraryException, match="unknown"):
        period_parser.parse("123")


def test_period_parse_non_numeric_prefix_raises() -> None:
    with pytest.raises(LibraryException, match="no numbers"):
        period_parser.parse("xM")


def test_period_parse_unit_only_raises() -> None:
    # "1Mx" — invalid trailing unit "x"
    with pytest.raises(LibraryException, match="unknown"):
        period_parser.parse("1Mx")


# --- DateParser.parse_iso ---------------------------------------------------


def test_date_parse_iso_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["date_parse_iso"]:
        d = date_parser.parse_iso(str(case["input"]))
        assert d.serial == int(case["serial"]), case
        assert d.year() == int(case["y"]), case
        assert int(d.month()) == int(case["m"]), case
        assert d.day_of_month() == int(case["d"]), case


def test_date_parse_iso_wrong_length_raises() -> None:
    with pytest.raises(LibraryException, match="invalid format"):
        date_parser.parse_iso("2024-1-1")
    with pytest.raises(LibraryException, match="invalid format"):
        date_parser.parse_iso("2024-01-01 ")  # 11 chars


def test_date_parse_iso_wrong_separator_raises() -> None:
    with pytest.raises(LibraryException, match="invalid format"):
        date_parser.parse_iso("2024/01/01")


def test_date_parse_iso_non_numeric_raises() -> None:
    with pytest.raises(LibraryException, match="invalid format"):
        date_parser.parse_iso("xxxx-xx-xx")


def test_date_parse_iso_month_out_of_range_raises_library_exception() -> None:
    # Month(0) and Month(13) raise stdlib ValueError; parse_iso wraps to
    # LibraryException so callers can rely on the documented API contract.
    with pytest.raises(LibraryException, match="invalid format"):
        date_parser.parse_iso("2024-00-15")
    with pytest.raises(LibraryException, match="invalid format"):
        date_parser.parse_iso("2024-13-15")


# --- DateParser.parse_formatted (Python-native, no probe) -------------------


def test_date_parse_formatted_iso() -> None:
    d = date_parser.parse_formatted("2024-03-15", "%Y-%m-%d")
    assert d == Date.from_ymd(15, Month.March, 2024)


def test_date_parse_formatted_us_style() -> None:
    d = date_parser.parse_formatted("03/15/2024", "%m/%d/%Y")
    assert d == Date.from_ymd(15, Month.March, 2024)


def test_date_parse_formatted_european_style() -> None:
    d = date_parser.parse_formatted("15.03.2024", "%d.%m.%Y")
    assert d == Date.from_ymd(15, Month.March, 2024)


def test_date_parse_formatted_invalid_raises() -> None:
    with pytest.raises(LibraryException, match="unable to parse"):
        date_parser.parse_formatted("not-a-date", "%Y-%m-%d")


# --- Round-trip sanity ------------------------------------------------------


def test_period_parse_round_trip_via_period_arithmetic() -> None:
    # parser → Period arithmetic agreement
    p = period_parser.parse("1Y6M")
    assert p.length == 18
    assert p.units == TimeUnit.Months
    # via direct addition
    assert p == Period(1, TimeUnit.Years) + Period(6, TimeUnit.Months)
