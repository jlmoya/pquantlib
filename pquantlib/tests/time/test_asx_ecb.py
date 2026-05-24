"""Cross-validate ASX + ECB helpers against the C++ probe.

Probe source: migration-harness/cpp/probes/time/asx_ecb_probe.cpp
Reference:    migration-harness/references/time/asx/ecb.json  (path determined
              by the generate-references.sh underscore->slash convention)
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader
from pquantlib.time import asx, ecb
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/asx/ecb")


# --- ASX ----------------------------------------------------------------


def test_asx_is_asx_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["asx"]["is_asx_date"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        assert asx.is_asx_date(d, bool(case["main_cycle"])) is bool(case["result"]), case


def test_asx_is_asx_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["asx"]["is_asx_code"]:
        assert asx.is_asx_code(str(case["code"]), bool(case["main_cycle"])) is bool(case["result"]), case


def test_asx_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["asx"]["code"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        assert asx.code(d) == case["code"], case


def test_asx_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["asx"]["date"]:
        ref = Date.from_ymd(1, Month.January, int(case["ref_y"]))
        d = asx.date(str(case["code"]), ref)
        assert d.year() == int(case["out_y"]), case
        assert int(d.month()) == int(case["out_m"]), case
        assert d.day_of_month() == int(case["out_d"]), case


def test_asx_next_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["asx"]["next_date"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        nx = asx.next_date(d, bool(case["main_cycle"]))
        assert nx.year() == int(case["out_y"]), case
        assert int(nx.month()) == int(case["out_m"]), case
        assert nx.day_of_month() == int(case["out_d"]), case


def test_asx_code_raises_for_non_asx() -> None:
    not_asx = Date.from_ymd(15, Month.March, 2024)  # day > 14
    with pytest.raises(LibraryException, match="not an ASX"):
        asx.code(not_asx)


# --- ECB ----------------------------------------------------------------


def test_ecb_is_ecb_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["ecb"]["is_ecb_code"]:
        assert ecb.is_ecb_code(str(case["code"])) is bool(case["result"]), case


def test_ecb_known_dates_count_matches_cpp(cpp: dict[str, Any]) -> None:
    e = cpp["ecb"]
    assert len(ecb.known_dates()) == int(e["known_dates_count"])
    sorted_known = sorted(ecb.known_dates())
    assert sorted_known[0].serial == int(e["known_dates_first_serial"])
    assert sorted_known[-1].serial == int(e["known_dates_last_serial"])


def test_ecb_next_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["ecb"]["next_date"]:
        d = Date(int(case["input_serial"]))
        nx = ecb.next_date(d)
        assert nx.serial == int(case["out_serial"]), case


def test_ecb_first_known_code_matches_cpp(cpp: dict[str, Any]) -> None:
    e = cpp["ecb"]
    first_known = Date(int(e["first_known_serial"]))
    assert ecb.code(first_known) == e["first_known_code"]


def test_ecb_next_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["ecb"]["next_code"]:
        assert ecb.next_code(str(case["in"])) == case["out"], case


# --- Python-side coverage -----------------------------------------------


def test_ecb_invalid_code_raises() -> None:
    with pytest.raises(LibraryException, match="not a valid ECB"):
        ecb.next_code("XYZ10")


def test_ecb_add_remove_date_round_trip() -> None:
    extra = Date.from_ymd(1, Month.January, 2099)
    assert extra not in ecb.known_dates()
    ecb.add_date(extra)
    try:
        assert extra in ecb.known_dates()
    finally:
        ecb.remove_date(extra)
    assert extra not in ecb.known_dates()


def test_ecb_date_via_month_year_overload() -> None:
    # ECB has Jan 2010 serial 40198 — date(January, 2010) should return that
    # (since next_date(Jan 1, 2010 - 1) = first ECB date in or after Jan 2010).
    d = ecb.date(Month.January, 2010)
    assert d.serial == 40198


def test_ecb_next_code_december_overflow_into_next_decade() -> None:
    # DEC29 → JAN30 (the year overflow case).
    assert ecb.next_code("DEC29") == "JAN30"
