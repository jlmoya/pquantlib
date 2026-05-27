"""Tests for SwapIndex abstract base.

Full ``forecast_fixing()`` depends on L3 ``VanillaSwap`` and is deliberately
deferred — we only test inspectors + ``maturity_date`` here.
"""

from __future__ import annotations

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _make_swap_index_5y() -> SwapIndex:
    euribor3m = IborIndex(
        "Euribor", Period(3, TimeUnit.Months), 2,
        EURCurrency(), TARGET(),
        BusinessDayConvention.ModifiedFollowing, True, Actual360(),
    )
    return SwapIndex(
        "EuriborSwapIsdaFixA",
        Period(5, TimeUnit.Years),
        2,
        EURCurrency(),
        TARGET(),
        Period(1, TimeUnit.Years),
        BusinessDayConvention.ModifiedFollowing,
        Thirty360(Thirty360Convention.BondBasis),
        euribor3m,
    )


def test_swap_index_inspectors_round_trip() -> None:
    idx = _make_swap_index_5y()
    assert idx.family_name() == "EuriborSwapIsdaFixA"
    assert idx.tenor() == Period(5, TimeUnit.Years)
    assert idx.fixing_days() == 2
    assert idx.fixed_leg_tenor() == Period(1, TimeUnit.Years)
    assert idx.fixed_leg_convention() == BusinessDayConvention.ModifiedFollowing
    assert idx.ibor_index().family_name() == "Euribor"
    assert idx.exogenous_discount() is False
    assert idx.discounting_term_structure() is None


def test_swap_index_name() -> None:
    idx = _make_swap_index_5y()
    # Day counter is the *fixed leg* day counter (Thirty360).
    assert idx.name() == "EuriborSwapIsdaFixA5Y 30/360 (Bond Basis)"


def test_swap_index_maturity_date_advances_by_tenor() -> None:
    idx = _make_swap_index_5y()
    value = Date.from_ymd(19, Month.January, 2024)
    maturity = idx.maturity_date(value)
    # 5Y MF on TARGET from 2024-01-19 → 2029-01-19 (Fri).
    assert maturity == Date.from_ymd(19, Month.January, 2029)


def test_swap_index_forecast_fixing_deferred() -> None:
    idx = _make_swap_index_5y()
    with pytest.raises(LibraryException, match="deferred to L3"):
        idx.forecast_fixing(Date.from_ymd(17, Month.January, 2024))


def test_swap_index_clone_overrides_curves() -> None:
    base = _make_swap_index_5y()
    cloned = base.clone()
    assert cloned.family_name() == base.family_name()
    assert cloned.tenor() == base.tenor()


def test_swap_index_uses_fixed_leg_day_counter() -> None:
    idx = _make_swap_index_5y()
    assert idx.day_counter() is idx.fixed_leg_day_counter()
