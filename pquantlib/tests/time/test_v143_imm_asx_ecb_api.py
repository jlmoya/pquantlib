"""Cross-validate the IMM / ASX / ECB entry points the earlier port omitted.

Probe source: migration-harness/cpp/probes/v143_time_immasxecb/probe.cpp
Reference:    migration-harness/references/v143/time/immasxecb.json

The pre-existing references (``time/imm.json``, ``time/asx/ecb.json``) pin the
Date-argument entry points. This module covers the rest of the C++ API:

* ``IMM::nextDate(const std::string&, bool, const Date&)``   imm.cpp:189
* ``IMM::nextCode(const std::string&, bool, const Date&)``   imm.cpp:202
* ``IMM::nextCode(const Date&, bool)``                       imm.cpp:196
* ``ASX::nextDate(const std::string&, bool, const Date&)``   asx.cpp:144
* ``ASX::nextCode(const std::string&, bool, const Date&)``   asx.cpp:157
* ``ASX::nextCode(const Date&, bool)``                       asx.cpp:151
* ``ECB::isECBdate(const Date&)``                            ecb.hpp:81
* ``ECB::nextDate(const std::string&, const Date&)``         ecb.hpp:65
* ``ECB::nextDates(const std::string&, const Date&)``        ecb.hpp:74
* ``ECB::nextCode(const Date&)``                             ecb.hpp:87
* ``IMM::Month`` / ``ASX::Month`` enumerators                imm.hpp:36, asx.hpp:37

Every assertion is exact: these are integer/date/string results with no
floating point anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader
from pquantlib.time.asx import ASX
from pquantlib.time.date import Date
from pquantlib.time.ecb import ECB
from pquantlib.time.imm import IMM


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/time/immasxecb")


# --------------------------------------------------------------------------
# IMM::Month / ASX::Month
# --------------------------------------------------------------------------


def test_imm_month_enum_matches_cpp(cpp: dict[str, Any]) -> None:
    assert {m.name: int(m) for m in IMM.Month} == {k: int(v) for k, v in cpp["imm_month_enum"].items()}


def test_asx_month_enum_matches_cpp(cpp: dict[str, Any]) -> None:
    assert {m.name: int(m) for m in ASX.Month} == {k: int(v) for k, v in cpp["asx_month_enum"].items()}


# --------------------------------------------------------------------------
# IMM
# --------------------------------------------------------------------------


@pytest.mark.exact
def test_imm_next_date_from_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["imm_next_date_from_code"]:
        got = IMM.next_date(row["code"], row["main_cycle"], Date(int(row["ref"])))
        assert got.serial == int(row["out"]), row


@pytest.mark.exact
def test_imm_next_code_from_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["imm_next_code_from_code"]:
        got = IMM.next_code(row["code"], row["main_cycle"], Date(int(row["ref"])))
        assert got == row["out"], row


@pytest.mark.exact
def test_imm_next_code_from_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["imm_next_code_from_date"]:
        assert IMM.next_code(Date(int(row["d"])), row["main_cycle"]) == row["out"], row


def test_imm_namespace_class_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="namespace"):
        IMM()


# --------------------------------------------------------------------------
# ASX
# --------------------------------------------------------------------------


@pytest.mark.exact
def test_asx_next_date_from_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["asx_next_date_from_code"]:
        got = ASX.next_date(row["code"], row["main_cycle"], Date(int(row["ref"])))
        assert got.serial == int(row["out"]), row


@pytest.mark.exact
def test_asx_next_code_from_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["asx_next_code_from_code"]:
        got = ASX.next_code(row["code"], row["main_cycle"], Date(int(row["ref"])))
        assert got == row["out"], row


@pytest.mark.exact
def test_asx_next_code_from_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["asx_next_code_from_date"]:
        assert ASX.next_code(Date(int(row["d"])), row["main_cycle"]) == row["out"], row


def test_asx_namespace_class_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="namespace"):
        ASX()


# --------------------------------------------------------------------------
# ECB
# --------------------------------------------------------------------------


@pytest.mark.exact
def test_ecb_is_ecb_date_matches_cpp(cpp: dict[str, Any]) -> None:
    lo, hi = (int(x) for x in cpp["ecb_is_ecb_date_range"])
    expected = {int(s) for s in cpp["ecb_is_ecb_date_true"]}
    actual = {s for s in range(lo, hi + 1) if ECB.is_ecb_date(Date(s))}
    assert actual == expected


@pytest.mark.exact
def test_ecb_next_date_from_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["ecb_next_date_from_code"]:
        got = ECB.next_date(row["code"], Date(int(row["ref"])))
        assert got.serial == int(row["out"]), row


@pytest.mark.exact
def test_ecb_next_dates_from_code_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["ecb_next_dates_from_code"]:
        got = ECB.next_dates(row["code"], Date(int(row["ref"])))
        assert [d.serial for d in got] == [int(s) for s in row["out"]], row["code"]


@pytest.mark.exact
def test_ecb_next_code_from_date_matches_cpp(cpp: dict[str, Any]) -> None:
    for row in cpp["ecb_next_code_from_date"]:
        assert ECB.next_code(Date(int(row["d"]))) == row["out"], row


def test_ecb_namespace_class_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="namespace"):
        ECB()


def test_ecb_namespace_class_exposes_the_whole_cpp_surface() -> None:
    """Every non-inherited static member of C++ ``struct ECB`` has a name here.

    C++ ql/time/ecb.hpp:34-97 declares exactly these ten entry points
    (counting the overload pairs once each).
    """
    for member in (
        "known_dates",
        "add_date",
        "remove_date",
        "date",
        "code",
        "next_date",
        "next_dates",
        "is_ecb_date",
        "is_ecb_code",
        "next_code",
    ):
        assert callable(getattr(ECB, member)), member
