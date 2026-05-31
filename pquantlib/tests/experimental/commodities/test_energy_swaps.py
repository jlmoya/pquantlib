"""EnergySwap + EnergyVanillaSwap + EnergyBasisSwap tests (W7-C batch c).

Probe source: migration-harness/cpp/probes/cluster_w7c/probe.cpp
Reference:    migration-harness/references/cluster/w7c.json
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.money import Money
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.commodities.commodity_curve import CommodityCurve
from pquantlib.experimental.commodities.commodity_index import CommodityIndex
from pquantlib.experimental.commodities.commodity_settings import CommoditySettings
from pquantlib.experimental.commodities.commodity_type import NullCommodityType
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.energy_basis_swap import EnergyBasisSwap
from pquantlib.experimental.commodities.energy_swap import EnergySwap
from pquantlib.experimental.commodities.energy_vanilla_swap import EnergyVanillaSwap
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
)
from pquantlib.experimental.commodities.pricing_period import PricingPeriod
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.March, 2020)


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w7c")


@pytest.fixture
def base_settings() -> Iterator[None]:
    settings = ObservableSettings()
    prev_date = settings.evaluation_date
    cs = CommoditySettings.instance()
    prev_ccy = cs.currency
    prev_uom = cs.unit_of_measure
    settings.evaluation_date = _TODAY
    cs.currency = USDCurrency()
    cs.unit_of_measure = BarrelUnitOfMeasure()
    try:
        yield
    finally:
        settings.evaluation_date = prev_date
        cs.currency = prev_ccy
        cs.unit_of_measure = prev_uom


def _make_curve() -> CommodityCurve:
    return CommodityCurve(
        "WTI_FWD",
        NullCommodityType(),
        USDCurrency(),
        BarrelUnitOfMeasure(),
        NullCalendar(),
        [
            Date.from_ymd(15, Month.March, 2020),
            Date.from_ymd(15, Month.June, 2020),
            Date.from_ymd(15, Month.September, 2020),
            Date.from_ymd(15, Month.December, 2020),
        ],
        [30.0, 35.0, 40.0, 45.0],
        Actual365Fixed(),
    )


def _flat_ts() -> FlatForward:
    return FlatForward.from_rate(_TODAY, 0.01, Actual365Fixed())


def _swap_period() -> PricingPeriod:
    return PricingPeriod(
        Date.from_ymd(16, Month.March, 2020),
        Date.from_ymd(20, Month.March, 2020),
        Date.from_ymd(31, Month.March, 2020),
        Quantity(NullCommodityType(), BarrelUnitOfMeasure(), 1000.0),
    )


def _index_with_daily_fixings(name: str, start_value: float) -> CommodityIndex:
    cal = NullCalendar()
    index = CommodityIndex(
        name,
        NullCommodityType(),
        USDCurrency(),
        BarrelUnitOfMeasure(),
        cal,
        1.0,
        _make_curve(),
    )
    index.clear_fixings()
    v = start_value
    d = Date.from_ymd(16, Month.March, 2020)
    end = Date.from_ymd(20, Month.March, 2020)
    while d <= end:
        index.add_fixing(d, v)
        v += 1.0
        d = cal.advance(d, 1, TimeUnit.Days)
    return index


# ---- EnergyVanillaSwap ----


def test_vanilla_swap_npv(cpp_ref: dict[str, Any], base_settings: None) -> None:
    usd = USDCurrency()
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    index = _index_with_daily_fixings("WTI_SWAP", 30.0)
    ts = _flat_ts()
    swap = EnergyVanillaSwap(
        True,
        NullCalendar(),
        Money(usd, 30.0),
        bbl,
        index,
        usd,
        usd,
        [_swap_period()],
        nct,
        None,
        ts,
        ts,
        ts,
    )
    tolerance.loose(swap.npv(), cpp_ref["swap_npv"])
    assert len(swap.daily_positions) == int(cpp_ref["swap_daily_positions"])
    assert len(swap.payment_cash_flows) == int(cpp_ref["swap_payment_cashflows"])


def test_vanilla_swap_zero_when_fixed_equals_avg(base_settings: None) -> None:
    # floating avg (30..34) = 32; fixed price 32 -> NPV ~ 0.
    usd = USDCurrency()
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    index = _index_with_daily_fixings("WTI_SWAP_Z", 30.0)
    ts = _flat_ts()
    swap = EnergyVanillaSwap(
        True, NullCalendar(), Money(usd, 32.0), bbl, index, usd, usd,
        [_swap_period()], nct, None, ts, ts, ts,
    )
    assert abs(swap.npv()) < 1e-6


def test_vanilla_swap_payer_sign_flips(base_settings: None) -> None:
    # Receiver of fixed (payer=False) is the negative of payer=True.
    usd = USDCurrency()
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    ts = _flat_ts()
    payer = EnergyVanillaSwap(
        True, NullCalendar(), Money(usd, 30.0), bbl,
        _index_with_daily_fixings("WTI_P", 30.0), usd, usd,
        [_swap_period()], nct, None, ts, ts, ts,
    )
    receiver = EnergyVanillaSwap(
        False, NullCalendar(), Money(usd, 30.0), bbl,
        _index_with_daily_fixings("WTI_R", 30.0), usd, usd,
        [_swap_period()], nct, None, ts, ts, ts,
    )
    tolerance.loose(receiver.npv(), -payer.npv())


def test_vanilla_swap_inspectors(base_settings: None) -> None:
    usd = USDCurrency()
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    ts = _flat_ts()
    index = _index_with_daily_fixings("WTI_INS", 30.0)
    swap = EnergyVanillaSwap(
        True, NullCalendar(), Money(usd, 30.0), bbl, index, usd, usd,
        [_swap_period()], nct, None, ts, ts, ts,
    )
    assert swap.pay_receive == 1
    assert swap.fixed_price.value == 30.0
    assert swap.fixed_price_unit_of_measure == bbl
    assert swap.index is index
    assert swap.is_expired() is False
    # aggregate quantity over the single period.
    assert swap.quantity().amount == 1000.0


# ---- EnergyBasisSwap ----


def test_basis_swap_zero_basis_equal_indices(base_settings: None) -> None:
    # Two identical index price paths, zero basis -> NPV ~ 0.
    usd = USDCurrency()
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    ts = _flat_ts()
    pay = _index_with_daily_fixings("BASIS_PAY", 30.0)
    rec = _index_with_daily_fixings("BASIS_REC", 30.0)
    spread = _index_with_daily_fixings("BASIS_SPR", 30.0)
    basis = CommodityUnitCost(Money(usd, 0.0), bbl)
    swap = EnergyBasisSwap(
        NullCalendar(), spread, pay, rec, True, usd, usd,
        [_swap_period()], basis, nct, None, ts, ts, ts,
    )
    assert abs(swap.npv()) < 1e-6


def test_basis_swap_basis_spread_on_pay_leg(base_settings: None) -> None:
    # Equal index paths but +2 basis on the pay leg. payLeg = -(price+2),
    # receiveLeg = price; per day net = -2 * avgQty. With 5 days and qty
    # 1000 spread evenly (avgQty = 200): undiscounted dDelta = -2*1000 = -2000.
    usd = USDCurrency()
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    ts = _flat_ts()
    pay = _index_with_daily_fixings("BASIS_PAY2", 30.0)
    rec = _index_with_daily_fixings("BASIS_REC2", 30.0)
    spread = _index_with_daily_fixings("BASIS_SPR2", 30.0)
    basis = CommodityUnitCost(Money(usd, 2.0), bbl)
    swap = EnergyBasisSwap(
        NullCalendar(), spread, pay, rec, True, usd, usd,
        [_swap_period()], basis, nct, None, ts, ts, ts,
    )
    # discounted by flat 1% from 15-Mar to 31-Mar; magnitude just under 2000.
    npv = swap.npv()
    assert -2000.0 < npv < -1990.0


def test_basis_swap_inspectors(base_settings: None) -> None:
    usd = USDCurrency()
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    ts = _flat_ts()
    pay = _index_with_daily_fixings("BASIS_PAY3", 30.0)
    rec = _index_with_daily_fixings("BASIS_REC3", 30.0)
    spread = _index_with_daily_fixings("BASIS_SPR3", 30.0)
    basis = CommodityUnitCost(Money(usd, 1.0), bbl)
    swap = EnergyBasisSwap(
        NullCalendar(), spread, pay, rec, False, usd, usd,
        [_swap_period()], basis, nct, None, ts, ts, ts,
    )
    assert swap.pay_index is pay
    assert swap.receive_index is rec
    assert swap.basis is basis


# ---- EnergySwap base (via concrete) ----


def test_energy_swap_is_abstract_base() -> None:
    assert issubclass(EnergyVanillaSwap, EnergySwap)
    assert issubclass(EnergyBasisSwap, EnergySwap)
