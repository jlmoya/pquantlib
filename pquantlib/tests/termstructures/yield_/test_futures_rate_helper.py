"""Tests for FuturesRateHelper."""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.futures_rate_helper import FuturesRateHelper, FuturesType
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from tests.indexes._mock_curves import FlatForwardMock


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2c")


def test_futures_rate_helper_implied_quote(ref: dict[str, Any]) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    # IMM date: 2024-03-20 (third Wednesday of March).
    start = Date.from_ymd(20, Month.March, 2024)
    helper = FuturesRateHelper(
        SimpleQuote(95.0),
        ibor_start_date=start,
        length_in_months=3,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        convexity_adjustment=SimpleQuote(0.0),
    )
    helper.set_term_structure(ts)
    expected = ref["futures_rate_helper"]
    assert helper.earliest_date().serial == expected["earliest_serial"]
    assert helper.maturity_date().serial == expected["maturity_serial"]
    tight(helper.implied_quote(), float(expected["implied_quote"]))


def test_futures_rate_helper_rejects_non_imm_date() -> None:
    bad = Date.from_ymd(2, Month.January, 2024)  # not an IMM date
    with pytest.raises(LibraryException, match="IMM"):
        FuturesRateHelper(
            SimpleQuote(95.0),
            ibor_start_date=bad,
            length_in_months=3,
            calendar=TARGET(),
            convention=BusinessDayConvention.ModifiedFollowing,
            end_of_month=True,
            day_counter=Actual360(),
        )


def test_futures_rate_helper_custom_type_accepts_any_date() -> None:
    any_date = Date.from_ymd(17, Month.January, 2024)
    helper = FuturesRateHelper(
        SimpleQuote(95.0),
        ibor_start_date=any_date,
        length_in_months=3,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        futures_type=FuturesType.Custom,
    )
    assert helper.earliest_date() == any_date


def test_futures_rate_helper_convexity_adjustment_inspectable() -> None:
    start = Date.from_ymd(20, Month.March, 2024)
    helper = FuturesRateHelper(
        SimpleQuote(95.0),
        ibor_start_date=start,
        length_in_months=3,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        convexity_adjustment=0.001,
    )
    assert helper.convexity_adjustment() == 0.001
