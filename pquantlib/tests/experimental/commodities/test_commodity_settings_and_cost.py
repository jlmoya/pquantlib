"""Money + CommodityUnitCost + CommoditySettings foundation tests.

Money same-currency arithmetic is exact; cross-currency paths are deferred
(ExchangeRateManager not yet ported) and raise. CommoditySettings defaults
match C++ (USD + Barrel).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.money import Money, MoneySettings, close, close_enough
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.commodities.commodity_settings import CommoditySettings
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
    LitreUnitOfMeasure,
)
from pquantlib.testing import tolerance


@pytest.fixture(autouse=True)
def _reset_money_settings() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    saved_type = MoneySettings.instance().conversion_type
    saved_base = MoneySettings.instance().base_currency
    yield
    MoneySettings.instance().conversion_type = saved_type
    MoneySettings.instance().base_currency = saved_base


# ---- Money ----


def test_money_ctor_both_orders() -> None:
    usd = USDCurrency()
    a = Money(usd, 100.0)
    b = Money(100.0, usd)
    assert a.value == 100.0
    assert b.value == 100.0
    assert a.currency == usd
    assert a == b


def test_money_default_is_empty_currency() -> None:
    m = Money()
    assert m.value == 0.0
    assert m.currency.empty()


def test_money_same_currency_arithmetic() -> None:
    usd = USDCurrency()
    tolerance.tight((Money(usd, 100.0) + Money(usd, 50.0)).value, 150.0)
    tolerance.tight((Money(usd, 100.0) - Money(usd, 50.0)).value, 50.0)
    tolerance.tight((Money(usd, 100.0) * 3.0).value, 300.0)
    tolerance.tight((3.0 * Money(usd, 100.0)).value, 300.0)
    half = Money(usd, 100.0) / 2.0
    assert isinstance(half, Money)
    tolerance.tight(half.value, 50.0)
    ratio = Money(usd, 100.0) / Money(usd, 25.0)
    assert isinstance(ratio, float)
    tolerance.tight(ratio, 4.0)
    tolerance.tight((-Money(usd, 7.0)).value, -7.0)


def test_money_comparisons_and_close() -> None:
    usd = USDCurrency()
    assert Money(usd, 1.0) < Money(usd, 2.0)
    assert Money(usd, 2.0) > Money(usd, 1.0)
    assert Money(usd, 1.0) <= Money(usd, 1.0)
    assert Money(usd, 2.0) >= Money(usd, 2.0)
    assert Money(usd, 1.0) == Money(usd, 1.0)
    assert Money(usd, 1.0) != Money(usd, 2.0)
    assert close(Money(usd, 1.0), Money(usd, 1.0))
    assert close_enough(Money(usd, 1.0), Money(usd, 1.0))


def test_money_cross_currency_no_conversion_raises() -> None:
    usd = USDCurrency()
    eur = Currency(name="European Euro", code="EUR", numeric_code=978)
    assert MoneySettings.instance().conversion_type == Money.ConversionType.NO_CONVERSION
    with pytest.raises(LibraryException):
        _ = Money(usd, 1.0) + Money(eur, 1.0)


def test_money_cross_currency_automated_deferred() -> None:
    # AutomatedConversion would need ExchangeRateManager (not ported) -> raises.
    usd = USDCurrency()
    eur = Currency(name="European Euro", code="EUR", numeric_code=978)
    MoneySettings.instance().conversion_type = (
        Money.ConversionType.AUTOMATED_CONVERSION
    )
    with pytest.raises(LibraryException):
        _ = Money(usd, 1.0) + Money(eur, 1.0)


def test_money_settings_singleton() -> None:
    assert MoneySettings.instance() is MoneySettings()


# ---- CommodityUnitCost ----


def test_commodity_unit_cost_accessors() -> None:
    usd = USDCurrency()
    cost = CommodityUnitCost(Money(usd, 75.5), BarrelUnitOfMeasure())
    tolerance.tight(cost.amount.value, 75.5)
    assert cost.amount.currency == usd
    assert cost.unit_of_measure == BarrelUnitOfMeasure()


def test_commodity_unit_cost_default() -> None:
    cost = CommodityUnitCost()
    assert cost.amount.value == 0.0
    assert cost.unit_of_measure.empty()


# ---- CommoditySettings ----


def test_commodity_settings_defaults() -> None:
    s = CommoditySettings.instance()
    assert s.currency == USDCurrency()
    assert s.unit_of_measure == BarrelUnitOfMeasure()


def test_commodity_settings_is_singleton() -> None:
    assert CommoditySettings.instance() is CommoditySettings()


def test_commodity_settings_mutable() -> None:
    s = CommoditySettings.instance()
    saved = s.unit_of_measure
    try:
        s.unit_of_measure = LitreUnitOfMeasure()
        assert CommoditySettings.instance().unit_of_measure == LitreUnitOfMeasure()
    finally:
        s.unit_of_measure = saved
