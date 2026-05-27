"""Tests for DepositRateHelper."""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
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


def _flat_curve(rate: float, eval_date: Date) -> FlatForwardMock:
    return FlatForwardMock(eval_date, rate, Actual360())


def test_deposit_rate_helper_implied_quote_round_trips(ref: dict[str, Any]) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = _flat_curve(0.05, eval_date)
    helper = DepositRateHelper(
        SimpleQuote(0.05),
        tenor=Period(3, TimeUnit.Months),
        fixing_days=2,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        evaluation_date=eval_date,
    )
    helper.set_term_structure(ts)
    expected = ref["deposit_rate_helper"]
    assert helper.earliest_date().serial == expected["earliest_serial"]
    assert helper.maturity_date().serial == expected["maturity_serial"]
    tight(helper.implied_quote(), float(expected["implied_quote"]))


def test_deposit_rate_helper_with_explicit_ibor_index() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = _flat_curve(0.05, eval_date)
    idx = Euribor.three_months()
    helper = DepositRateHelper(SimpleQuote(0.05), ibor_index=idx, evaluation_date=eval_date)
    helper.set_term_structure(ts)
    # implied_quote should match the C++-probe value (continuous-comp vs
    # simple-rate divergence is ~3bp at 5% / 3M, matching the L2-C probe).
    tight(helper.implied_quote(), 0.050317307618303497)


def test_deposit_rate_helper_requires_ts_for_implied_quote() -> None:
    helper = DepositRateHelper(
        SimpleQuote(0.05),
        tenor=Period(3, TimeUnit.Months),
        fixing_days=2,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        evaluation_date=Date.from_ymd(17, Month.January, 2024),
    )
    with pytest.raises(LibraryException, match="term structure"):
        helper.implied_quote()


def test_deposit_rate_helper_requires_either_index_or_full_args() -> None:
    with pytest.raises(LibraryException, match="provide"):
        DepositRateHelper(SimpleQuote(0.05))
