"""Tests for ql/rebatedexercise.{hpp,cpp}.

Every expected value comes from ``migration-harness/cpp/probes/v143_root_tail/probe.cpp``
(``emitRebatedExercise``) run against C++ QuantLib v1.43; the reference lives at
``migration-harness/references/v143/root/tail.json``.

``RebatedExercise`` reads no global state, and the probe's rebate section sets no
evaluation date, so this module needs no settings fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, BermudanExercise, EuropeanExercise, Exercise
from pquantlib.rebated_exercise import RebatedExercise
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/root/tail")["rebated_exercise"]


@pytest.fixture
def bermudan(cpp: dict[str, Any]) -> BermudanExercise:
    return BermudanExercise([Date(int(s)) for s in cpp["bermudan_date_serials"]])


def test_scalar_ctor_broadcasts_and_rolls(cpp: dict[str, Any], bermudan: BermudanExercise) -> None:
    """Scalar rebate + 3-day lag + TARGET + ModifiedPreceding (probe "scalar")."""
    exp = cpp["scalar"]
    re = RebatedExercise(bermudan, -12.5, 3, TARGET(), BusinessDayConvention.ModifiedPreceding)
    assert re.type() == exp["type"]
    for i, expected in enumerate(exp["rebates"]):
        tolerance.exact(re.rebate(i), expected)
    assert [re.rebate_payment_date(i).serial_number() for i in range(3)] == exp["payment_dates"]


def test_settlement_days_is_not_discarded(cpp: dict[str, Any], bermudan: BermudanExercise) -> None:
    """Zero lag must give different dates from the 3-day lag — else the arg is dead."""
    zero = RebatedExercise(bermudan, -12.5, 0, TARGET(), BusinessDayConvention.ModifiedPreceding)
    got = [zero.rebate_payment_date(i).serial_number() for i in range(3)]
    assert got == cpp["scalar_zero_lag_payment_dates"]
    assert got != cpp["scalar"]["payment_dates"]


def test_calendar_and_convention_are_not_discarded(
    cpp: dict[str, Any], bermudan: BermudanExercise
) -> None:
    """NullCalendar/Following must give different dates from TARGET/ModifiedPreceding."""
    null_cal = RebatedExercise(bermudan, -12.5, 3, NullCalendar(), BusinessDayConvention.Following)
    got = [null_cal.rebate_payment_date(i).serial_number() for i in range(3)]
    assert got == cpp["scalar_null_calendar_payment_dates"]
    assert got != cpp["scalar"]["payment_dates"]


def test_vector_ctor(cpp: dict[str, Any], bermudan: BermudanExercise) -> None:
    exp = cpp["vector"]
    re = RebatedExercise(
        bermudan, [1.5, -2.5, 3.5], 2, TARGET(), BusinessDayConvention.Preceding
    )
    for i, expected in enumerate(exp["rebates"]):
        tolerance.exact(re.rebate(i), expected)
    assert re.rebates() == exp["rebates"]
    assert [re.rebate_payment_date(i).serial_number() for i in range(3)] == exp["payment_dates"]


def test_european(cpp: dict[str, Any]) -> None:
    exp = cpp["european"]
    eur = EuropeanExercise(Date.from_ymd(30, Month.April, 2025))
    re = RebatedExercise(eur, 7.25, 1, TARGET(), BusinessDayConvention.Following)
    assert len(re.rebates()) == exp["n_rebates"]
    tolerance.exact(re.rebate(0), exp["rebate0"])
    assert re.rebate_payment_date(0).serial_number() == exp["payment_date0"]
    assert re.type() == exp["type"]


def test_vector_on_non_bermudan_raises(cpp: dict[str, Any]) -> None:
    assert cpp["vector_on_european_throws"] is True
    eur = EuropeanExercise(Date(int(cpp["european"]["payment_date0"])))
    with pytest.raises(LibraryException):
        RebatedExercise(eur, [1.0], 0, TARGET(), BusinessDayConvention.Following)


def test_vector_wrong_size_raises(cpp: dict[str, Any], bermudan: BermudanExercise) -> None:
    assert cpp["vector_wrong_size_throws"] is True
    with pytest.raises(LibraryException):
        RebatedExercise(bermudan, [1.0, 2.0], 0, TARGET(), BusinessDayConvention.Following)


def test_american_payment_date_raises(cpp: dict[str, Any]) -> None:
    assert cpp["american_payment_date_throws"] is True
    serials = cpp["bermudan_date_serials"]
    amer = AmericanExercise(Date(int(serials[0])), Date(int(serials[-1])))
    re = RebatedExercise(amer, 1.0, 0, TARGET(), BusinessDayConvention.Following)
    assert re.type() == Exercise.Type.American
    tolerance.exact(re.rebate(0), 1.0)
    with pytest.raises(LibraryException):
        re.rebate_payment_date(0)


def test_rebate_index_out_of_range_raises(
    cpp: dict[str, Any], bermudan: BermudanExercise
) -> None:
    assert cpp["rebate_out_of_range_throws"] is True
    re = RebatedExercise(bermudan, -12.5, 3, TARGET(), BusinessDayConvention.ModifiedPreceding)
    with pytest.raises(LibraryException):
        re.rebate(len(cpp["bermudan_date_serials"]))


def test_default_calendar_is_null_calendar(bermudan: BermudanExercise) -> None:
    """C++ default argument is ``NullCalendar()`` (ql/rebatedexercise.hpp:45)."""
    re = RebatedExercise(bermudan)
    assert isinstance(re.rebate_payment_calendar(), NullCalendar)
    assert re.rebate_settlement_days() == 0
    assert re.rebate_payment_convention() == BusinessDayConvention.Following
    assert re.rebates() == [0.0, 0.0, 0.0]
