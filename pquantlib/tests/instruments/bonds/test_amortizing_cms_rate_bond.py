"""Cross-validate AmortizingCmsRateBond against ``v143/inst/bondsamort``.

The amortisation is swept three ways — a sinking-fund schedule
(``sinking_notionals``), a hand-written per-period notional vector and a single
constant notional — and the per-coupon nominal is asserted against C++ in every
case. A dropped notional vector would still price (as a bullet), so the nominal
listing is the assertion that catches it.

Every other optional argument gets its own non-default test, as for
:mod:`test_cms_rate_bond`.

Tolerance: LOOSE — see :mod:`test_cms_rate_bond` for the rationale (Hagan
static replication). Structural assertions are exact.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise
from typing import Any

import pytest

from pquantlib.cashflows.coupon_pricer import set_coupon_pricer
from pquantlib.instruments.bonds.amortizing_cms_rate_bond import AmortizingCmsRateBond
from pquantlib.instruments.bonds.amortizing_fixed_rate_bond import sinking_notionals
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

from ._bondsamort import (
    CMS_START,
    CUSTOM_NOTIONALS,
    DC_30360,
    TODAY,
    CmsMarket,
    build_market,
    cms_schedule,
    compare_bond,
    load_reference,
    pinned_evaluation_date,
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return load_reference()


@pytest.fixture(scope="module")
def market() -> CmsMarket:
    return build_market()


@pytest.fixture
def pinned_today() -> Iterator[None]:
    with pinned_evaluation_date(TODAY):
        yield


def sinking_vector() -> list[float]:
    """Probe ``sinkingVector()`` — French amortisation, trailing zero dropped.

    ``sinking_notionals`` returns ``n_periods + 1`` entries whose last is 0.0;
    the leg only ever reads indices ``[0, n)`` and ``FloatingLeg`` rejects more
    than ``n`` nominals, so the unread trailing zero is dropped.
    """
    return sinking_notionals(Period(10, TimeUnit.Years), Frequency.Annual, 0.05, 100.0)[
        :-1
    ]


def _bond(market: CmsMarket, **overrides: Any) -> AmortizingCmsRateBond:
    """Probe ``emitAmCmsBond`` — defaults, with one argument overridden."""
    args: dict[str, Any] = {
        "settlement_days": 2,
        "notionals": [100.0],
        "schedule": cms_schedule(),
        "swap_index": market.swap_index,
        "payment_day_counter": DC_30360,
    }
    args.update(overrides)
    bond = AmortizingCmsRateBond(**args)
    bond.set_pricing_engine(market.engine)
    set_coupon_pricer(bond.cashflows(), market.cms_pricer)
    return bond


# --- the amortisation sweep -------------------------------------------------


def test_constant_notional(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A single-entry notional vector is a bullet: one redemption, flat nominal."""
    ref = cpp["amcms_constant"]
    bond = _bond(market)
    compare_bond(bond, ref, loose)
    assert len(bond.redemptions()) == 1
    nominals = {c["nominal"] for c in ref["cashflows"] if c["is_coupon"]}
    assert nominals == {100.0}


def test_sinking_notionals(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """French (sinking-fund) amortisation: every coupon nominal pinned to C++."""
    ref = cpp["amcms_sinking"]
    bond = _bond(market, notionals=sinking_vector())
    compare_bond(bond, ref, loose)
    # Ten distinct, strictly decreasing nominals — this is exactly what a
    # dropped amortisation vector would collapse to a constant.
    nominals = [c["nominal"] for c in ref["cashflows"] if c["is_coupon"]]
    assert len(nominals) == 10
    assert all(a > b for a, b in pairwise(nominals))


def test_custom_notionals(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A hand-written per-period notional vector."""
    ref = cpp["amcms_custom"]
    bond = _bond(market, notionals=CUSTOM_NOTIONALS)
    compare_bond(bond, ref, loose)
    assert [c["nominal"] for c in ref["cashflows"] if c["is_coupon"]] == CUSTOM_NOTIONALS
    # Nine amortising payments + one final redemption.
    assert len(bond.redemptions()) == 10


def test_redemptions_vector(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``redemptions=[100, 101, 102]`` (default ``[100]``).

    Each amortisation step is redeemed at its own price; the last entry repeats
    for the remaining steps (bond.cpp ``addRedemptionsToCashflows``).
    """
    ref = cpp["amcms_redemptions"]
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS, redemptions=[100.0, 101.0, 102.0]),
        ref,
        loose,
    )
    base = cpp["amcms_custom"]
    assert ref["npv"] != base["npv"]


# --- one non-default per remaining optional argument ------------------------


def test_payment_convention(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``payment_convention=Preceding`` (default Following)."""
    ref = cpp["amcms_payment_convention"]
    compare_bond(
        _bond(
            market,
            notionals=CUSTOM_NOTIONALS,
            payment_convention=BusinessDayConvention.Preceding,
        ),
        ref,
        loose,
    )
    base_dates = [c["date_serial"] for c in cpp["amcms_custom"]["cashflows"]]
    assert [c["date_serial"] for c in ref["cashflows"]] != base_dates


def test_fixing_days(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``fixing_days=10`` (default: the index's own 2)."""
    ref = cpp["amcms_fixing_days"]
    compare_bond(_bond(market, notionals=CUSTOM_NOTIONALS, fixing_days=10), ref, loose)
    base_fixings = [c.get("fixing_serial") for c in cpp["amcms_custom"]["cashflows"]]
    assert [c.get("fixing_serial") for c in ref["cashflows"]] != base_fixings


def test_gearings(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``gearings=[0.8]`` (default ``[1.0]``)."""
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS, gearings=[0.8]),
        cpp["amcms_gearings"],
        loose,
    )


def test_spreads(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``spreads=[0.002]`` (default ``[0.0]``)."""
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS, spreads=[0.002]),
        cpp["amcms_spreads"],
        loose,
    )


def test_caps(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``caps=[0.035]`` (default: none)."""
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS, caps=[0.035]),
        cpp["amcms_caps"],
        loose,
    )


def test_floors(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``floors=[0.045]`` (default: none)."""
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS, floors=[0.045]),
        cpp["amcms_floors"],
        loose,
    )


def test_in_arrears(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``in_arrears=True`` moves every fixing date to the accrual end."""
    ref = cpp["amcms_in_arrears"]
    compare_bond(_bond(market, notionals=CUSTOM_NOTIONALS, in_arrears=True), ref, loose)
    base_fixings = [c.get("fixing_serial") for c in cpp["amcms_custom"]["cashflows"]]
    assert [c.get("fixing_serial") for c in ref["cashflows"]] != base_fixings


def test_issue_date(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``issue_date`` (default: null) clamps settlement dates before issue."""
    ref = cpp["amcms_issue_date"]
    bond = _bond(market, notionals=CUSTOM_NOTIONALS, issue_date=CMS_START)
    compare_bond(bond, ref, loose)
    assert bond.issue_date() == CMS_START
    assert (
        ref["settlement_before_issue_serial"]
        != cpp["amcms_custom"]["settlement_before_issue_serial"]
    )


def test_settlement_days(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``settlement_days=5`` (default 2)."""
    ref = cpp["amcms_settlement_days"]
    compare_bond(_bond(market, notionals=CUSTOM_NOTIONALS, settlement_days=5), ref, loose)
    assert ref["settlement_serial"] != cpp["amcms_custom"]["settlement_serial"]
