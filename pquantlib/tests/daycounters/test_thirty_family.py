"""Cross-validate Thirty360 (9 conventions) + Thirty365 against the C++ probe.

Probe source: migration-harness/cpp/probes/daycounters/thirty_family_probe.cpp
Reference:    migration-harness/references/daycounters/thirty/family.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.daycounters.thirty_365 import Thirty365
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("daycounters/thirty/family")


def _check_section(dc: DayCounter, section: dict[str, Any]) -> None:
    assert dc.name() == section["name"]
    for case in section["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        assert dc.day_count(a, b) == int(case["day_count"]), case
        tolerance.tight(dc.year_fraction(a, b), float(case["year_fraction"]))


# --- Thirty360 — all 9 convention aliases ---------------------------------

_CONVENTION_KEYS = [
    ("usa", Convention.USA),
    ("bond_basis", Convention.BondBasis),
    ("european", Convention.European),
    ("eurobond_basis", Convention.EurobondBasis),
    ("italian", Convention.Italian),
    ("german", Convention.German),
    ("isma", Convention.ISMA),
    ("isda", Convention.ISDA),
    ("nasd", Convention.NASD),
]


@pytest.mark.parametrize(("key", "convention"), _CONVENTION_KEYS)
def test_thirty360_convention_matches_cpp(key: str, convention: Convention, cpp: dict[str, Any]) -> None:
    _check_section(Thirty360(convention), cpp[key])


# --- Thirty365 ----------------------------------------------------------


def test_thirty365_matches_cpp(cpp: dict[str, Any]) -> None:
    _check_section(Thirty365(), cpp["thirty365"])


# --- Thirty360 ISDA with termination date --------------------------------


def test_thirty360_isda_with_termination_date_matches_cpp(cpp: dict[str, Any]) -> None:
    section = cpp["isda_term_2024_02_29"]
    term = Date.from_ymd(
        int(section["termination"]["d"]),
        Month(int(section["termination"]["m"])),
        int(section["termination"]["y"]),
    )
    dc = Thirty360(Convention.ISDA, termination_date=term)
    assert dc.name() == section["name"]
    for case in section["cases"]:
        a = Date.from_ymd(int(case["d1"]), Month(int(case["m1"])), int(case["y1"]))
        b = Date.from_ymd(int(case["d2"]), Month(int(case["m2"])), int(case["y2"]))
        assert dc.day_count(a, b) == int(case["day_count"]), case
        tolerance.tight(dc.year_fraction(a, b), float(case["year_fraction"]))


# --- Python-side coverage -----------------------------------------------


def test_thirty360_aliases_share_name() -> None:
    # BondBasis ≡ ISMA, European ≡ EurobondBasis, ISDA ≡ German
    assert Thirty360(Convention.BondBasis).name() == Thirty360(Convention.ISMA).name()
    assert Thirty360(Convention.European).name() == Thirty360(Convention.EurobondBasis).name()
    assert Thirty360(Convention.ISDA).name() == Thirty360(Convention.German).name()


def test_thirty360_distinct_conventions_compare_distinct() -> None:
    assert Thirty360(Convention.USA) != Thirty360(Convention.NASD)
    assert Thirty360(Convention.USA) != Thirty365()


def test_thirty365_equality() -> None:
    assert Thirty365() == Thirty365()
