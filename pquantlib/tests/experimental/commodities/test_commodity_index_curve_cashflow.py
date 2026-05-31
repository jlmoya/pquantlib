"""CommodityCurve + CommodityIndex + CommodityCashFlow tests (W7-C batch a).

Probe source: migration-harness/cpp/probes/cluster_w7c/probe.cpp
Reference:    migration-harness/references/cluster/w7c.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.money import Money
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.commodities.commodity_cash_flow import (
    CommodityCashFlow,
    CommodityCashFlows,
)
from pquantlib.experimental.commodities.commodity_curve import CommodityCurve
from pquantlib.experimental.commodities.commodity_index import CommodityIndex
from pquantlib.experimental.commodities.commodity_type import NullCommodityType
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w7c")


def _curve_dates() -> list[Date]:
    return [
        Date.from_ymd(15, Month.March, 2020),
        Date.from_ymd(15, Month.June, 2020),
        Date.from_ymd(15, Month.September, 2020),
        Date.from_ymd(15, Month.December, 2020),
    ]


def _make_curve() -> CommodityCurve:
    return CommodityCurve(
        "WTI_FWD",
        NullCommodityType(),
        USDCurrency(),
        BarrelUnitOfMeasure(),
        NullCalendar(),
        _curve_dates(),
        [30.0, 35.0, 40.0, 45.0],
        Actual365Fixed(),
    )


# ---- CommodityCurve ----


def test_curve_price_at_node(cpp_ref: dict[str, Any]) -> None:
    curve = _make_curve()
    p = curve.price(Date.from_ymd(15, Month.June, 2020))
    tolerance.loose(p, cpp_ref["curve_price_node1"])


def test_curve_price_forward_flat_mid(cpp_ref: dict[str, Any]) -> None:
    # Forward-flat holds the LEFT-hand knot value going forward: 15-Apr
    # (between 15-Mar=30 and 15-Jun=35) prices at 30.
    curve = _make_curve()
    p = curve.price(Date.from_ymd(15, Month.April, 2020))
    tolerance.loose(p, cpp_ref["curve_price_mid01"])


def test_curve_price_last_node(cpp_ref: dict[str, Any]) -> None:
    curve = _make_curve()
    p = curve.price(Date.from_ymd(15, Month.December, 2020))
    tolerance.loose(p, cpp_ref["curve_price_node3"])


def test_curve_max_date(cpp_ref: dict[str, Any]) -> None:
    curve = _make_curve()
    assert curve.max_date().serial_number() == int(cpp_ref["curve_max_date_serial"])


def test_curve_inspectors() -> None:
    curve = _make_curve()
    assert curve.name == "WTI_FWD"
    assert curve.currency == USDCurrency()
    assert curve.unit_of_measure == BarrelUnitOfMeasure()
    assert not curve.empty()
    assert len(curve.nodes()) == 4
    assert curve.dates() == _curve_dates()
    assert curve.prices() == [30.0, 35.0, 40.0, 45.0]


def test_curve_too_few_dates() -> None:
    with pytest.raises(LibraryException, match="too few dates"):
        CommodityCurve(
            "X",
            NullCommodityType(),
            USDCurrency(),
            BarrelUnitOfMeasure(),
            NullCalendar(),
            [Date.from_ymd(15, Month.March, 2020)],
            [30.0],
        )


def test_curve_set_prices() -> None:
    # The empty ctor creates a *moving* term structure (reference == eval date);
    # pin the eval date so the priced date is after the reference.
    settings = ObservableSettings()
    prev = settings.evaluation_date
    settings.evaluation_date = Date.from_ymd(15, Month.March, 2020)
    try:
        curve = CommodityCurve(
            "EMPTY",
            NullCommodityType(),
            USDCurrency(),
            BarrelUnitOfMeasure(),
            NullCalendar(),
        )
        assert curve.empty()
        curve.set_prices(
            {
                Date.from_ymd(15, Month.March, 2020): 30.0,
                Date.from_ymd(15, Month.June, 2020): 35.0,
            }
        )
        assert not curve.empty()
        # Forward-flat: 15-Jun is the right node -> 35.
        tolerance.loose(curve.price(Date.from_ymd(15, Month.June, 2020)), 35.0)
    finally:
        settings.evaluation_date = prev


def test_curve_basis_curve_adds_on_top() -> None:
    base = _make_curve()
    overlay = CommodityCurve(
        "OVERLAY",
        NullCommodityType(),
        USDCurrency(),
        BarrelUnitOfMeasure(),
        NullCalendar(),
        _curve_dates(),
        [1.0, 1.0, 1.0, 1.0],
        Actual365Fixed(),
    )
    overlay.set_basis_of_curve(base)
    # overlay(node1) = 1 (own) + 35 (basis) = 36.
    tolerance.loose(overlay.price(Date.from_ymd(15, Month.June, 2020)), 36.0)


# ---- CommodityIndex ----


def _make_index(curve: CommodityCurve | None = None) -> CommodityIndex:
    return CommodityIndex(
        "WTI",
        NullCommodityType(),
        USDCurrency(),
        BarrelUnitOfMeasure(),
        NullCalendar(),
        1000.0,
        curve,
    )


def test_index_empty_and_forward_curve_empty(cpp_ref: dict[str, Any]) -> None:
    curve = _make_curve()
    index = _make_index(curve)
    index.clear_fixings()
    assert index.empty() == bool(cpp_ref["index_empty_before"])
    assert index.forward_curve_empty() == bool(cpp_ref["index_fwd_curve_empty"])


def test_index_fixing_from_history(cpp_ref: dict[str, Any]) -> None:
    curve = _make_curve()
    index = _make_index(curve)
    index.clear_fixings()
    index.add_fixing(Date.from_ymd(13, Month.March, 2020), 29.5)
    index.add_fixing(Date.from_ymd(15, Month.March, 2020), 31.0)
    # History fixings must match to TIGHT (no interpolation, stored verbatim).
    tolerance.tight(
        index.fixing(Date.from_ymd(15, Month.March, 2020)),
        cpp_ref["index_fixing_hist"],
    )
    assert index.empty() == bool(cpp_ref["index_empty_after"])
    assert index.last_quote_date().serial_number() == int(
        cpp_ref["index_last_quote_serial"]
    )


def test_index_forward_price(cpp_ref: dict[str, Any]) -> None:
    curve = _make_curve()
    index = _make_index(curve)
    fwd = index.forward_price(Date.from_ymd(15, Month.September, 2020))
    tolerance.loose(fwd, cpp_ref["index_forward_price"])


def test_index_lot_quantity_and_accessors() -> None:
    curve = _make_curve()
    index = _make_index(curve)
    assert index.name() == "WTI"
    assert index.lot_quantity == 1000.0
    assert index.currency == USDCurrency()
    assert index.unit_of_measure == BarrelUnitOfMeasure()
    assert index.forward_curve is curve


def test_index_forward_price_requires_curve() -> None:
    index = _make_index(None)
    with pytest.raises(LibraryException, match="no forward curve"):
        index.forward_price(Date.from_ymd(15, Month.September, 2020))


# ---- CommodityCashFlow ----


def test_cashflow_amount(cpp_ref: dict[str, Any]) -> None:
    usd = USDCurrency()
    disc = Money(usd, 12345.67)
    undisc = Money(usd, 12500.00)
    ccf = CommodityCashFlow(
        Date.from_ymd(20, Month.December, 2020),
        disc,
        undisc,
        disc,
        undisc,
        0.987,
        0.987,
        finalized=False,
    )
    tolerance.tight(ccf.amount(), cpp_ref["cashflow_amount"])
    tolerance.tight(ccf.discount_factor, cpp_ref["cashflow_disc_factor"])


def test_cashflow_accessors() -> None:
    usd = USDCurrency()
    disc = Money(usd, 100.0)
    undisc = Money(usd, 110.0)
    pay_disc = Money(usd, 90.0)
    pay_undisc = Money(usd, 95.0)
    d = Date.from_ymd(20, Month.December, 2020)
    ccf = CommodityCashFlow(
        d, disc, undisc, pay_disc, pay_undisc, 0.9, 0.91, finalized=True
    )
    assert ccf.date() == d
    assert ccf.currency == usd
    assert ccf.discounted_amount.value == 100.0
    assert ccf.undiscounted_amount.value == 110.0
    assert ccf.discounted_payment_amount.value == 90.0
    assert ccf.undiscounted_payment_amount.value == 95.0
    assert ccf.payment_discount_factor == 0.91
    assert ccf.finalized is True


def test_cashflows_typedef_is_dict() -> None:
    usd = USDCurrency()
    d = Date.from_ymd(20, Month.December, 2020)
    ccf = CommodityCashFlow(
        d, Money(usd, 1.0), Money(usd, 1.0), Money(usd, 1.0), Money(usd, 1.0),
        1.0, 1.0, finalized=False,
    )
    flows: CommodityCashFlows = {d: ccf}
    assert flows[d] is ccf
