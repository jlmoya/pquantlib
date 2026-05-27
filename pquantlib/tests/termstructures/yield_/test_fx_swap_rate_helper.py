"""Tests for FxSwapRateHelper."""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.fx_swap_rate_helper import FxSwapRateHelper
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from tests.indexes._mock_curves import FlatForwardMock


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2c")


def test_fx_swap_rate_helper_implied_quote(ref: dict[str, Any]) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    coll = FlatForwardMock(eval_date, 0.03, Actual365Fixed())
    ts = FlatForwardMock(eval_date, 0.05, Actual365Fixed())
    helper = FxSwapRateHelper(
        SimpleQuote(0.01),
        SimpleQuote(1.10),
        Period(3, TimeUnit.Months),
        2,
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        is_fx_base_currency_collateral_currency=True,
        collateral_curve=coll,
        evaluation_date=eval_date,
    )
    helper.set_term_structure(ts)
    expected = ref["fx_swap_rate_helper"]
    assert helper.earliest_date().serial == expected["earliest_serial"]
    assert helper.latest_date().serial == expected["latest_serial"]
    tight(helper.implied_quote(), float(expected["implied_quote"]))


def test_fx_swap_rate_helper_inspectors() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    coll = FlatForwardMock(eval_date, 0.03, Actual365Fixed())
    helper = FxSwapRateHelper(
        SimpleQuote(0.01),
        SimpleQuote(1.10),
        Period(3, TimeUnit.Months),
        2,
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        is_fx_base_currency_collateral_currency=True,
        collateral_curve=coll,
        evaluation_date=eval_date,
    )
    assert helper.spot() == 1.10
    assert helper.tenor() == Period(3, TimeUnit.Months)
    assert helper.fixing_days() == 2
    assert helper.business_day_convention() == BusinessDayConvention.ModifiedFollowing
    assert helper.end_of_month() is False
    assert helper.is_fx_base_currency_collateral_currency() is True
    assert helper.trading_calendar() is None
