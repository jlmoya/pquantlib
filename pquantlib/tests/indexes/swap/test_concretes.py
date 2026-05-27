"""Tests for the 2 concrete swap indexes."""

from __future__ import annotations

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.indexes.swap.usd_libor_swap_isda_fix_am import UsdLiborSwapIsdaFixAm
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def test_euribor_swap_isda_fix_a_2y_uses_6m_euribor() -> None:
    idx = EuriborSwapIsdaFixA(Period(2, TimeUnit.Years))
    assert idx.family_name() == "EuriborSwapIsdaFixA"
    assert idx.tenor() == Period(2, TimeUnit.Years)
    assert idx.fixing_days() == 2
    assert isinstance(idx.currency(), EURCurrency)
    assert idx.fixed_leg_tenor() == Period(1, TimeUnit.Years)
    assert idx.fixed_leg_convention() == BusinessDayConvention.ModifiedFollowing
    assert isinstance(idx.fixed_leg_day_counter(), Thirty360)
    # For tenor > 1Y → 6M Euribor.
    assert idx.ibor_index().tenor() == Period(6, TimeUnit.Months)


def test_euribor_swap_isda_fix_a_1y_uses_3m_euribor() -> None:
    idx = EuriborSwapIsdaFixA(Period(1, TimeUnit.Years))
    # tenor == 1Y is NOT > 1Y → 3M Euribor.
    assert idx.ibor_index().tenor() == Period(3, TimeUnit.Months)


def test_euribor_swap_isda_fix_a_fixed_leg_uses_bond_basis() -> None:
    idx = EuriborSwapIsdaFixA(Period(5, TimeUnit.Years))
    dc = idx.fixed_leg_day_counter()
    assert isinstance(dc, Thirty360)
    # Indirect check — name encodes the convention.
    assert dc.name() == Thirty360(Thirty360Convention.BondBasis).name()


def test_usd_libor_swap_isda_fix_am_uses_6m_fixed_3m_libor() -> None:
    idx = UsdLiborSwapIsdaFixAm(Period(5, TimeUnit.Years))
    assert idx.family_name() == "UsdLiborSwapIsdaFixAm"
    assert idx.tenor() == Period(5, TimeUnit.Years)
    assert idx.fixing_days() == 2
    assert isinstance(idx.currency(), USDCurrency)
    assert idx.fixed_leg_tenor() == Period(6, TimeUnit.Months)
    assert idx.fixed_leg_convention() == BusinessDayConvention.ModifiedFollowing
    # Always 3M USDLibor.
    assert idx.ibor_index().tenor() == Period(3, TimeUnit.Months)
    assert idx.ibor_index().family_name() == "USDLibor"


def test_swap_index_exogenous_discount_false_by_default() -> None:
    a = EuriborSwapIsdaFixA(Period(2, TimeUnit.Years))
    b = UsdLiborSwapIsdaFixAm(Period(5, TimeUnit.Years))
    assert a.exogenous_discount() is False
    assert b.exogenous_discount() is False
