"""Tests for FraRateHelper."""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.fra_rate_helper import FraRateHelper
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import loose, tight
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


def test_fra_rate_helper_implied_quote(ref: dict[str, Any]) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    helper = FraRateHelper(
        SimpleQuote(0.05),
        months_to_start=3,
        length_in_months=3,
        fixing_days=2,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        evaluation_date=eval_date,
    )
    helper.set_term_structure(ts)
    expected = ref["fra_rate_helper"]
    assert helper.earliest_date().serial == expected["earliest_serial"]
    assert helper.maturity_date().serial == expected["maturity_serial"]
    tight(helper.implied_quote(), float(expected["implied_quote"]))


def test_fra_rate_helper_indexed_coupon_implied_quote() -> None:
    """useIndexedCoupon=True branch — L2-C carry-over now closed.

    Probe ``cluster/l3e`` covers this case explicitly.
    """
    ref_l3e = reference_reader.load("cluster/l3e")["fra_rate_helper_indexed"]
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    helper = FraRateHelper(
        SimpleQuote(0.05),
        months_to_start=3,
        length_in_months=3,
        fixing_days=2,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        use_indexed_coupon=True,
        evaluation_date=eval_date,
    )
    helper.set_term_structure(ts)
    assert helper.earliest_date().serial == ref_l3e["earliest_serial"]
    assert helper.maturity_date().serial == ref_l3e["maturity_serial"]
    # LOOSE tier — the indexed branch routes through index.fixing which
    # exercises an extra day-counter / interest-rate roundtrip vs the
    # discount-factor formula; ULP error accumulates but stays well below
    # 1e-8 absolute.
    loose(helper.implied_quote(), float(ref_l3e["implied_quote"]))


def test_fra_rate_helper_period_form() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    helper = FraRateHelper(
        SimpleQuote(0.05),
        period_to_start=Period(3, TimeUnit.Months),
        length_in_months=3,
        fixing_days=2,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        evaluation_date=eval_date,
    )
    # Just check dates were initialized.
    assert helper.earliest_date().serial > 0
    assert helper.maturity_date().serial > helper.earliest_date().serial
