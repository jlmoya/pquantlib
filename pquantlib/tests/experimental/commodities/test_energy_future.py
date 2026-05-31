"""EnergyCommodity + EnergyFuture tests (W7-C batch b).

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
from pquantlib.experimental.commodities.commodity import PricingErrorLevel
from pquantlib.experimental.commodities.commodity_curve import CommodityCurve
from pquantlib.experimental.commodities.commodity_index import CommodityIndex
from pquantlib.experimental.commodities.commodity_settings import CommoditySettings
from pquantlib.experimental.commodities.commodity_type import NullCommodityType
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.energy_commodity import (
    DeliverySchedule,
    EnergyCommodity,
    EnergyDailyPosition,
)
from pquantlib.experimental.commodities.energy_future import EnergyFuture
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
)
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_TODAY = Date.from_ymd(15, Month.March, 2020)


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w7c")


@pytest.fixture
def base_settings() -> Iterator[None]:
    """Pin the eval date + commodity base currency/UOM (USD / Barrel)."""
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


# ---- EnergyCommodity enums + EnergyDailyPosition ----


def test_energy_commodity_enums() -> None:
    assert EnergyCommodity.DeliverySchedule is DeliverySchedule
    assert int(DeliverySchedule.DAILY) == 3
    assert int(EnergyCommodity.QuantityPeriodicity.PER_MONTH) == 4
    assert int(EnergyCommodity.PaymentSchedule.YEARLY_SETTLEMENT) == 3


def test_energy_daily_position_defaults() -> None:
    dp = EnergyDailyPosition()
    assert dp.quantity_amount == 0.0
    assert dp.pay_leg_price == 0.0
    assert dp.receive_leg_price == 0.0
    assert dp.unrealized is False
    dp2 = EnergyDailyPosition(date=_TODAY, pay_leg_price=10.0, receive_leg_price=12.0)
    assert dp2.date == _TODAY
    assert dp2.pay_leg_price == 10.0


# ---- EnergyFuture ----


def test_energy_future_npv(cpp_ref: dict[str, Any], base_settings: None) -> None:
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    usd = USDCurrency()
    q = Quantity(nct, bbl, 1000.0)
    trade_price = CommodityUnitCost(Money(usd, 28.0), bbl)
    index = CommodityIndex(
        "WTI_FUT", nct, usd, bbl, NullCalendar(), 1.0, _make_curve()
    )
    index.clear_fixings()
    index.add_fixing(_TODAY, 31.0)

    fut = EnergyFuture(1, q, trade_price, index, nct)
    tolerance.loose(fut.npv(), cpp_ref["future_npv"])


def test_energy_future_buysell_sign(base_settings: None) -> None:
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    usd = USDCurrency()
    q = Quantity(nct, bbl, 1000.0)
    trade_price = CommodityUnitCost(Money(usd, 28.0), bbl)
    index = CommodityIndex(
        "WTI_FUT2", nct, usd, bbl, NullCalendar(), 1.0, _make_curve()
    )
    index.clear_fixings()
    index.add_fixing(_TODAY, 31.0)

    buy = EnergyFuture(1, q, trade_price, index, nct)
    sell = EnergyFuture(-1, q, trade_price, index, nct)
    # Sell side is the exact negative of the buy side.
    tolerance.loose(sell.npv(), -buy.npv())


def test_energy_future_accessors(base_settings: None) -> None:
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    usd = USDCurrency()
    q = Quantity(nct, bbl, 500.0)
    trade_price = CommodityUnitCost(Money(usd, 28.0), bbl)
    index = CommodityIndex(
        "WTI_FUT3", nct, usd, bbl, NullCalendar(), 1.0, _make_curve()
    )
    index.clear_fixings()
    index.add_fixing(_TODAY, 31.0)
    fut = EnergyFuture(1, q, trade_price, index, nct)
    assert fut.is_expired() is False
    assert fut.quantity().amount == 500.0
    assert fut.index is index
    assert fut.trade_price is trade_price
    assert fut.commodity_type == nct


def test_energy_future_uses_forward_when_no_recent_quote(base_settings: None) -> None:
    # No fixing near today -> forward price from the curve is used and a
    # pricing-error warning is recorded.
    nct = NullCommodityType()
    bbl = BarrelUnitOfMeasure()
    usd = USDCurrency()
    q = Quantity(nct, bbl, 1000.0)
    trade_price = CommodityUnitCost(Money(usd, 28.0), bbl)
    index = CommodityIndex(
        "WTI_FUT4", nct, usd, bbl, NullCalendar(), 1.0, _make_curve()
    )
    index.clear_fixings()
    # only an old fixing (well before eval - 1).
    index.add_fixing(Date.from_ymd(1, Month.January, 2020), 25.0)

    fut = EnergyFuture(1, q, trade_price, index, nct)
    # forward price at today (node 15-Mar -> 30). NPV = (30-28)*1000*1*1 = 2000.
    tolerance.loose(fut.npv(), 2000.0)
    assert any(
        e.error_level == PricingErrorLevel.WARNING for e in fut.pricing_errors
    )
