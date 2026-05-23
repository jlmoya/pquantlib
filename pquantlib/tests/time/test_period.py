"""Cross-validate Period algebra against the C++ probe.

Probe source: migration-harness/cpp/probes/time/period_probe.cpp
Reference:    migration-harness/references/time/period.json

Tolerance: arithmetic + comparison + frequency conversion are integer-valued
so default tier is EXACT. years/months/weeks/days extractors use TIGHT
because of the float division involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import (
    Period,
    days,
    months,
    weeks,
    years,
)
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/period")


def _u(name: str) -> TimeUnit:
    return TimeUnit[name]


def _p(pair: list[Any]) -> Period:
    return Period(int(pair[0]), _u(str(pair[1])))


# --- Frequency → Period -----------------------------------------------------


def test_from_frequency_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["frequency_to_period"]:
        f = Frequency[case["frequency"]]
        p = Period.from_frequency(f)
        assert p.length == case["length"], f"freq={f.name}"
        assert p.units.name == case["units"], f"freq={f.name}"


def test_from_frequency_otherfrequency_raises() -> None:
    with pytest.raises(LibraryException):
        Period.from_frequency(Frequency.OtherFrequency)


# --- Period → Frequency -----------------------------------------------------


def test_frequency_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["period_to_frequency"]:
        p = Period(int(case["length"]), _u(str(case["units"])))
        assert p.frequency().name == case["frequency"]


# --- normalize --------------------------------------------------------------


def test_normalized_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["normalize"]:
        src = _p(case["in"])
        expected = _p(case["out"])
        assert src.normalized() == expected, f"{src} normalized"


# --- addition ---------------------------------------------------------------


def test_addition_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["addition"]:
        a = _p(case["a"])
        b = _p(case["b"])
        expected = _p(case["result"])
        assert a + b == expected, f"{a} + {b}"


# --- multiplication ---------------------------------------------------------


def test_mul_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["mul"]:
        a = _p(case["a"])
        k = int(case["k"])
        expected = _p(case["result"])
        assert a * k == expected, f"{a} * {k}"
        # rmul commutativity
        assert k * a == expected, f"{k} * {a}"


# --- division ---------------------------------------------------------------


def test_floordiv_preserves_units_when_divisible() -> None:
    assert Period(24, TimeUnit.Months) // 2 == Period(12, TimeUnit.Months)
    assert Period(2, TimeUnit.Years) // 2 == Period(1, TimeUnit.Years)


def test_floordiv_converts_units_when_needed() -> None:
    # 1 year / 2 → 6 months (Years → Months)
    assert Period(1, TimeUnit.Years) // 2 == Period(6, TimeUnit.Months)
    # 1 week / 7 → 1 day (Weeks → Days)
    assert Period(1, TimeUnit.Weeks) // 7 == Period(1, TimeUnit.Days)


def test_floordiv_by_zero_raises() -> None:
    with pytest.raises(LibraryException, match="divided by zero"):
        _ = Period(1, TimeUnit.Years) // 0


def test_floordiv_indivisible_raises() -> None:
    with pytest.raises(LibraryException, match="cannot be divided"):
        _ = Period(5, TimeUnit.Months) // 3


# --- years / months / weeks / days extractors -------------------------------


def test_years_of_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["years_of"]:
        tolerance.tight(years(_p(case["in"])), float(case["out"]))


def test_months_of_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["months_of"]:
        tolerance.tight(months(_p(case["in"])), float(case["out"]))


def test_weeks_of_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["weeks_of"]:
        tolerance.tight(weeks(_p(case["in"])), float(case["out"]))


def test_days_of_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["days_of"]:
        tolerance.tight(days(_p(case["in"])), float(case["out"]))


def test_years_of_days_raises() -> None:
    with pytest.raises(LibraryException, match="Days into Years"):
        years(Period(5, TimeUnit.Days))


def test_weeks_of_months_raises() -> None:
    with pytest.raises(LibraryException, match="Months into Weeks"):
        weeks(Period(1, TimeUnit.Months))


# --- comparison -------------------------------------------------------------


def test_compare_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["compare"]:
        a = _p(case["a"])
        b = _p(case["b"])
        assert (a < b) is bool(case["a_lt_b"]), f"{a} < {b}"
        assert (b < a) is bool(case["b_lt_a"]), f"{b} < {a}"


# --- equality / hashing -----------------------------------------------------


def test_equality_is_field_by_field() -> None:
    assert Period(12, TimeUnit.Months) == Period(12, TimeUnit.Months)
    # Same duration, different units, NOT equal at the Period level — equality
    # only after normalize().
    assert Period(12, TimeUnit.Months) != Period(1, TimeUnit.Years)
    assert Period(12, TimeUnit.Months).normalized() == Period(1, TimeUnit.Years)


def test_hashable() -> None:
    s = {Period(1, TimeUnit.Years), Period(12, TimeUnit.Months), Period(1, TimeUnit.Years)}
    assert len(s) == 2


def test_negation() -> None:
    assert -Period(3, TimeUnit.Months) == Period(-3, TimeUnit.Months)


def test_subtraction() -> None:
    assert Period(1, TimeUnit.Years) - Period(6, TimeUnit.Months) == Period(6, TimeUnit.Months)


def test_default_period_is_zero_days() -> None:
    p = Period()
    assert p.length == 0
    assert p.units == TimeUnit.Days
