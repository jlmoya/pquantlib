"""ExchangeContract + PaymentTerm + DateInterval + PricingPeriod +
Commodity base + CommodityPricingHelper foundation tests.

Probe source: migration-harness/cpp/probes/cluster_w7b/probe.cpp
Reference:    migration-harness/references/cluster/w7b.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.money import Money
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.commodities.commodity import (
    Commodity,
    PricingError,
    PricingErrorLevel,
)
from pquantlib.experimental.commodities.commodity_pricing_helpers import (
    CommodityPricingHelper,
)
from pquantlib.experimental.commodities.commodity_type import NullCommodityType
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.date_interval import DateInterval
from pquantlib.experimental.commodities.exchange_contract import ExchangeContract
from pquantlib.experimental.commodities.payment_term import PaymentTerm
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
    LitreUnitOfMeasure,
)
from pquantlib.experimental.commodities.pricing_period import PricingPeriod
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w7b")


# ---- ExchangeContract ----


def test_exchange_contract_accessors() -> None:
    exp = Date.from_ymd(20, Month.December, 2020)
    start = Date.from_ymd(1, Month.January, 2021)
    end = Date.from_ymd(31, Month.January, 2021)
    ec = ExchangeContract("CLF21", exp, start, end)
    assert ec.code == "CLF21"
    assert ec.expiration_date == exp
    assert ec.underlying_start_date == start
    assert ec.underlying_end_date == end


def test_exchange_contract_default() -> None:
    ec = ExchangeContract()
    assert ec.code == ""
    assert ec.expiration_date == Date()


# ---- PaymentTerm ----


def test_payment_term_accessors_and_payment_date() -> None:
    cal = NullCalendar()
    pt = PaymentTerm(
        "Pricing end + 5 days", PaymentTerm.EventType.PRICING_DATE, 5, cal
    )
    assert pt.name == "Pricing end + 5 days"
    assert pt.event_type == PaymentTerm.EventType.PRICING_DATE
    assert pt.offset_days == 5
    assert pt.calendar == cal
    assert not pt.empty()
    # getPaymentDate: calendar.adjust(date + offset_days). NullCalendar never
    # shifts, so the result is exactly date + 5.
    ref = Date.from_ymd(10, Month.June, 2020)
    assert pt.get_payment_date(ref) == ref + 5


def test_payment_term_default_is_empty() -> None:
    pt = PaymentTerm()
    assert pt.empty()
    assert str(pt) == "null payment term type"


def test_payment_term_flyweight_and_equality() -> None:
    cal = NullCalendar()
    a = PaymentTerm("net 30", PaymentTerm.EventType.TRADE_DATE, 30, cal)
    b = PaymentTerm("net 30", PaymentTerm.EventType.TRADE_DATE, 30, cal)
    # Name-based equality + shared flyweight payload.
    assert a == b
    assert a._data is b._data  # pyright: ignore[reportPrivateUsage]
    assert a != PaymentTerm()


# ---- DateInterval ----


def test_date_interval_basics() -> None:
    start = Date.from_ymd(1, Month.March, 2020)
    end = Date.from_ymd(31, Month.March, 2020)
    di = DateInterval(start, end)
    assert di.start_date == start
    assert di.end_date == end
    mid = Date.from_ymd(15, Month.March, 2020)
    assert di.is_date_between(mid)
    assert di.is_date_between(start)  # include_first default True
    assert not di.is_date_between(start, include_first=False)
    assert di.is_date_between(end)
    assert not di.is_date_between(end, include_last=False)
    after = Date.from_ymd(1, Month.April, 2020)
    assert not di.is_date_between(after)


def test_date_interval_end_before_start_raises() -> None:
    start = Date.from_ymd(31, Month.March, 2020)
    end = Date.from_ymd(1, Month.March, 2020)
    with pytest.raises(LibraryException):
        DateInterval(start, end)


def test_date_interval_intersection() -> None:
    a = DateInterval(
        Date.from_ymd(1, Month.March, 2020), Date.from_ymd(20, Month.March, 2020)
    )
    b = DateInterval(
        Date.from_ymd(10, Month.March, 2020), Date.from_ymd(31, Month.March, 2020)
    )
    inter = a.intersection(b)
    assert inter.start_date == Date.from_ymd(10, Month.March, 2020)
    assert inter.end_date == Date.from_ymd(20, Month.March, 2020)


def test_date_interval_disjoint_intersection_is_empty() -> None:
    a = DateInterval(
        Date.from_ymd(1, Month.March, 2020), Date.from_ymd(10, Month.March, 2020)
    )
    b = DateInterval(
        Date.from_ymd(20, Month.March, 2020), Date.from_ymd(31, Month.March, 2020)
    )
    inter = a.intersection(b)
    assert inter == DateInterval()


def test_date_interval_default_str() -> None:
    assert str(DateInterval()) == "Null<DateInterval>()"


# ---- PricingPeriod (probe-validated accessor round-trips) ----


def test_pricing_period_accessors(cpp_ref: dict[str, Any]) -> None:
    start = Date.from_ymd(15, Month.March, 2020)
    end = Date.from_ymd(14, Month.April, 2020)
    pay = Date.from_ymd(20, Month.April, 2020)
    q = Quantity(NullCommodityType(), BarrelUnitOfMeasure(), 1000.0)
    pp = PricingPeriod(start, end, pay, q)
    tolerance.tight(
        float(pp.start_date.serial_number()),
        cpp_ref["pricing_period_start_serial"],
    )
    tolerance.tight(
        float(pp.end_date.serial_number()), cpp_ref["pricing_period_end_serial"]
    )
    tolerance.tight(
        float(pp.payment_date.serial_number()),
        cpp_ref["pricing_period_pay_serial"],
    )
    tolerance.tight(pp.quantity.amount, cpp_ref["pricing_period_qty_amount"])
    # PricingPeriod is a DateInterval.
    assert isinstance(pp, DateInterval)
    assert pp.is_date_between(Date.from_ymd(1, Month.April, 2020))


# ---- Commodity base ----


class _DummyCommodity(Commodity):
    """Minimal concrete Commodity to exercise the abstract base surface."""

    def is_expired(self) -> bool:
        return False

    def perform_calculations(self) -> None:
        self._npv = 0.0


def test_commodity_base_secondary_costs_and_errors() -> None:
    costs: dict[str, Any] = {"freight": "TBD"}
    c = _DummyCommodity(costs)
    assert c.secondary_costs == costs
    assert c.secondary_cost_amounts == {}
    assert c.pricing_errors == []
    c.add_pricing_error(PricingErrorLevel.WARNING, "low liquidity", "thin book")
    assert len(c.pricing_errors) == 1
    err = c.pricing_errors[0]
    assert err.error_level == PricingErrorLevel.WARNING
    assert err.error == "low liquidity"
    assert err.detail == "thin book"


def test_commodity_base_default_secondary_costs() -> None:
    c = _DummyCommodity()
    assert c.secondary_costs == {}


def test_pricing_error_str() -> None:
    assert str(PricingError(PricingError.Level.INFO, "note")) == "info: note"
    assert (
        str(PricingError(PricingError.Level.ERROR, "boom", "detail"))
        == "*** error: boom: detail"
    )


# ---- CommodityPricingHelper ----


def test_helper_uom_conversion_factor(cpp_ref: dict[str, Any]) -> None:
    nct = NullCommodityType()
    f = CommodityPricingHelper.calculate_uom_conversion_factor(
        nct, BarrelUnitOfMeasure(), LitreUnitOfMeasure()
    )
    tolerance.tight(f, cpp_ref["helper_uom_factor_bbl_litre"])
    same = CommodityPricingHelper.calculate_uom_conversion_factor(
        nct, BarrelUnitOfMeasure(), BarrelUnitOfMeasure()
    )
    tolerance.tight(same, cpp_ref["helper_uom_factor_same"])


def test_helper_fx_same_currency_is_one() -> None:
    usd = USDCurrency()
    f = CommodityPricingHelper.calculate_fx_conversion_factor(
        usd, usd, Date.from_ymd(1, Month.January, 2020)
    )
    tolerance.exact(f, 1.0)


def test_helper_fx_cross_currency_deferred() -> None:
    usd = USDCurrency()
    eur = Currency(name="European Euro", code="EUR", numeric_code=978)
    with pytest.raises(LibraryException):
        CommodityPricingHelper.calculate_fx_conversion_factor(
            usd, eur, Date.from_ymd(1, Month.January, 2020)
        )


def test_helper_calculate_unit_cost_same_currency_same_uom() -> None:
    usd = USDCurrency()
    nct = NullCommodityType()
    cost = CommodityUnitCost(Money(usd, 50.0), BarrelUnitOfMeasure())
    result = CommodityPricingHelper.calculate_unit_cost(
        nct, cost, usd, BarrelUnitOfMeasure(), Date.from_ymd(1, Month.January, 2020)
    )
    # Same currency (factor 1) + same UOM (factor 1) -> just the amount.
    tolerance.tight(result, 50.0)


def test_helper_calculate_unit_cost_cross_uom() -> None:
    usd = USDCurrency()
    nct = NullCommodityType()
    # 1 USD per litre, re-expressed per barrel: * (barrel->litre factor 158.987).
    cost = CommodityUnitCost(Money(usd, 1.0), LitreUnitOfMeasure())
    result = CommodityPricingHelper.calculate_unit_cost(
        nct, cost, usd, BarrelUnitOfMeasure(), Date.from_ymd(1, Month.January, 2020)
    )
    tolerance.loose(result, 158.987)


def test_helper_calculate_unit_cost_zero_amount() -> None:
    usd = USDCurrency()
    nct = NullCommodityType()
    cost = CommodityUnitCost(Money(usd, 0.0), BarrelUnitOfMeasure())
    result = CommodityPricingHelper.calculate_unit_cost(
        nct, cost, usd, LitreUnitOfMeasure(), Date.from_ymd(1, Month.January, 2020)
    )
    tolerance.exact(result, 0.0)
