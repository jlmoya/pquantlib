"""Tests for OISRateHelper. Full implied_quote test deferred to L3."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.ois_rate_helper import OISRateHelper
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def test_ois_rate_helper_constructs_with_dates() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    helper = OISRateHelper(
        settlement_days=2,
        tenor=Period(5, TimeUnit.Years),
        fixed_rate=SimpleQuote(0.04),
        overnight_index=Sofr(),
        evaluation_date=eval_date,
    )
    assert helper.earliest_date().serial > 0
    assert helper.maturity_date().serial > helper.earliest_date().serial
    assert helper.telescopic_value_dates() is False


def test_ois_rate_helper_implied_quote_deferred() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    helper = OISRateHelper(
        settlement_days=2,
        tenor=Period(5, TimeUnit.Years),
        fixed_rate=SimpleQuote(0.04),
        overnight_index=Sofr(),
        evaluation_date=eval_date,
    )
    with pytest.raises(LibraryException, match="L3"):
        helper.implied_quote()
