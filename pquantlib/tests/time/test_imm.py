"""Cross-validate IMM helpers against the C++ probe.

Probe source: migration-harness/cpp/probes/time/imm_probe.cpp
Reference:    migration-harness/references/time/imm.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader
from pquantlib.time import imm
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/imm")


def test_is_imm_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["is_imm_date"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        assert imm.is_imm_date(d, bool(case["main_cycle"])) is bool(case["result"]), case


def test_is_imm_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["is_imm_code"]:
        assert imm.is_imm_code(str(case["code"]), bool(case["main_cycle"])) is bool(case["result"]), case


def test_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["code"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        assert imm.code(d) == case["code"], case


def test_code_for_non_imm_date_raises() -> None:
    not_imm = Date.from_ymd(13, Month.March, 2013)  # Wed but day < 15
    with pytest.raises(LibraryException, match="not an IMM"):
        imm.code(not_imm)


def test_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["date"]:
        ref = Date.from_ymd(1, Month.January, int(case["ref_y"]))
        d = imm.date(str(case["code"]), ref)
        assert d.year() == int(case["out_y"]), case
        assert int(d.month()) == int(case["out_m"]), case
        assert d.day_of_month() == int(case["out_d"]), case


def test_date_for_invalid_code_raises() -> None:
    ref = Date.from_ymd(1, Month.January, 2024)
    with pytest.raises(LibraryException, match="not a valid IMM code"):
        imm.date("AB", ref)


def test_next_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["next_date"]:
        d = Date.from_ymd(int(case["d"]), Month(int(case["m"])), int(case["y"]))
        nx = imm.next_date(d, bool(case["main_cycle"]))
        assert nx.year() == int(case["out_y"]), case
        assert int(nx.month()) == int(case["out_m"]), case
        assert nx.day_of_month() == int(case["out_d"]), case


def test_next_code_roundtrip() -> None:
    d = Date.from_ymd(15, Month.January, 2024)
    # next main IMM after Jan 15, 2024 is Mar 20, 2024 → code H4
    assert imm.next_code(d, main_cycle=True) == "H4"


def test_main_cycle_versus_non_main_filtering() -> None:
    # 3rd Wed of January is a non-main IMM date.
    third_wed_jan = Date.from_ymd(17, Month.January, 2024)
    assert imm.is_imm_date(third_wed_jan, main_cycle=False) is True
    assert imm.is_imm_date(third_wed_jan, main_cycle=True) is False
