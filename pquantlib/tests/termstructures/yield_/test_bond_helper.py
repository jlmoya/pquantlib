"""Tests for BondHelper inspectors.

The cross-validated behaviour (dates + implied quote against C++ v1.43) lives
in ``test_bond_helper_implied_quote.py``. This file keeps the plain-inspector
checks, now over a real ``Bond`` — the earlier version drove a ``_StubBond``
placeholder and asserted that ``implied_quote`` raised "deferred to L3", which
stopped being true once the Bond instrument and DiscountingBondEngine landed.
"""

from __future__ import annotations

from pquantlib.instruments.bonds.zero_coupon_bond import ZeroCouponBond
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.bond_helper import BondHelper, BondPriceType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _bond() -> ZeroCouponBond:
    return ZeroCouponBond(
        2,
        TARGET(),
        100.0,
        Date.from_ymd(15, Month.June, 2030),
        BusinessDayConvention.Following,
        100.0,
        Date.from_ymd(15, Month.June, 2020),
    )


def test_bond_helper_holds_bond_and_price_type() -> None:
    bond = _bond()
    helper = BondHelper(SimpleQuote(100.0), bond)
    assert helper.bond() is bond
    assert helper.price_type() == BondPriceType.Clean


def test_bond_helper_dirty_price_type() -> None:
    helper = BondHelper(SimpleQuote(100.0), _bond(), price_type=BondPriceType.Dirty)
    assert helper.price_type() == BondPriceType.Dirty


def test_bond_helper_latest_date_is_last_cashflow_date() -> None:
    """C++ takes the last CASHFLOW date, which the Following roll pushes past
    the 15 June 2030 maturity (a Saturday)."""
    helper = BondHelper(SimpleQuote(100.0), _bond())
    assert helper.latest_date() == helper.bond().cashflows()[-1].date()
    assert helper.latest_date() > helper.bond().maturity_date()
