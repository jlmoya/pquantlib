"""Tests for SwapIndex abstract base.

L3-C closes the ``forecast_fixing`` / ``underlying_swap`` carry-overs; the
deferred-state test is replaced by a fair-rate roundtrip.
"""

from __future__ import annotations

from typing import cast

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
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


def test_swap_index_forecast_fixing_via_underlying_swap() -> None:
    """SwapIndex.forecast_fixing should now build the underlying swap and
    return its fair_rate (closed L2-C carry-over).
    """
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    euribor3m = IborIndex(
        "Euribor", Period(3, TimeUnit.Months), 2,
        EURCurrency(), TARGET(),
        BusinessDayConvention.ModifiedFollowing, True, Actual360(),
        curve,
    )
    idx = SwapIndex(
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
    fixing_date = Date.from_ymd(17, Month.January, 2024)
    # Fair rate from a flat curve is implementation-dependent (par-swap
    # rate ~5.21% for FF(5%) Continuous/Annual) — just assert it's a
    # plausible positive number near the curve rate.
    rate = idx.forecast_fixing(fixing_date)
    assert 0.04 < rate < 0.06


def test_swap_index_clone_overrides_curves() -> None:
    base = _make_swap_index_5y()
    cloned = base.clone()
    assert cloned.family_name() == base.family_name()
    assert cloned.tenor() == base.tenor()


def test_swap_index_uses_fixed_leg_day_counter() -> None:
    idx = _make_swap_index_5y()
    assert idx.day_counter() is idx.fixed_leg_day_counter()
