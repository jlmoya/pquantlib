"""Tests for SwapRateHelper. Full implied_quote test deferred to L3."""

from __future__ import annotations

import pytest

from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.swap_rate_helper import SwapRateHelper
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def test_swap_rate_helper_via_swap_index_inherits_meta() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    swap_idx = EuriborSwapIsdaFixA(Period(5, TimeUnit.Years))
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        swap_index=swap_idx,
        evaluation_date=eval_date,
    )
    assert helper.spread() == 0.0
    assert helper.forward_start() == Period(0, TimeUnit.Days)
    # Dates initialized:
    assert helper.earliest_date().serial > 0
    assert helper.maturity_date().serial > helper.earliest_date().serial


def test_swap_rate_helper_explicit_conventions() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    euribor3m = Euribor.three_months()
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        tenor=Period(5, TimeUnit.Years),
        calendar=TARGET(),
        fixed_frequency=Frequency.Annual,
        fixed_convention=BusinessDayConvention.ModifiedFollowing,
        fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
        ibor_index=euribor3m,
        evaluation_date=eval_date,
    )
    assert helper.earliest_date().serial > 0
    assert helper.maturity_date().serial > helper.earliest_date().serial


def test_swap_rate_helper_implied_quote_deferred() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    swap_idx = EuriborSwapIsdaFixA(Period(5, TimeUnit.Years))
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        swap_index=swap_idx,
        evaluation_date=eval_date,
    )
    with pytest.raises(LibraryException, match="L3"):
        helper.implied_quote()


def test_swap_rate_helper_with_spread() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    swap_idx = EuriborSwapIsdaFixA(Period(5, TimeUnit.Years))
    helper = SwapRateHelper(
        SimpleQuote(0.04),
        swap_index=swap_idx,
        spread=SimpleQuote(0.0010),
        evaluation_date=eval_date,
    )
    assert helper.spread() == 0.0010
