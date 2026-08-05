"""Cross-validate CmsRateBond against the ``v143/inst/bondsamort`` probe.

``CmsRateBond`` forwards thirteen arguments into ``CmsLeg``. Every optional one
gets its own test that sets it to a NON-default value and asserts the resulting
bond — headline prices plus the complete cashflow listing — against C++. An
argument that is accepted and silently dropped therefore fails here.

Tolerance: LOOSE. The coupon rate comes from the Hagan static-replication
convexity adjustment (``AnalyticHaganPricer``), which integrates a swaption
smile and solves for the underlying par swap rate; the existing
``tests/pricingengines/test_conundrum_pricer.py`` cross-validates that pricer
at the same tier. The structural assertions (dates, nominals, accrual periods,
cashflow counts) are exact regardless.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.coupon_pricer import set_coupon_pricer
from pquantlib.instruments.bonds.cms_rate_bond import CmsRateBond
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date

from ._bondsamort import (
    CMS_START,
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


def _bond(market: CmsMarket, **overrides: Any) -> CmsRateBond:
    """Probe ``emitCmsRateBond`` — defaults, with one argument overridden."""
    args: dict[str, Any] = {
        "settlement_days": 2,
        "face_amount": 100.0,
        "schedule": cms_schedule(),
        "swap_index": market.swap_index,
        "payment_day_counter": DC_30360,
    }
    args.update(overrides)
    bond = CmsRateBond(**args)
    bond.set_pricing_engine(market.engine)
    # C++'s CmsLeg attaches no pricer, so the caller must — as
    # test-suite/assetswap.cpp does with an AnalyticHaganPricer.
    set_coupon_pricer(bond.cashflows(), market.cms_pricer)
    return bond


def test_base(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """All-defaults bond: bullet notional, one redemption, 10 CMS coupons."""
    compare_bond(_bond(market), cpp["cms_base"], loose)


def test_payment_convention(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``payment_convention=Preceding`` (default Following).

    22-Aug falls on a weekend in 2020/2021/2026/2027, so the payment dates for
    those periods roll the other way.
    """
    ref = cpp["cms_payment_convention"]
    bond = _bond(market, payment_convention=BusinessDayConvention.Preceding)
    compare_bond(bond, ref, loose)
    base_dates = [c["date_serial"] for c in cpp["cms_base"]["cashflows"]]
    assert [c["date_serial"] for c in ref["cashflows"]] != base_dates


def test_fixing_days(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``fixing_days=10`` (default: the index's own 2)."""
    ref = cpp["cms_fixing_days"]
    compare_bond(_bond(market, fixing_days=10), ref, loose)
    base_fixings = [c.get("fixing_serial") for c in cpp["cms_base"]["cashflows"]]
    assert [c.get("fixing_serial") for c in ref["cashflows"]] != base_fixings


def test_gearings_scalar(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``gearings=[0.8]`` (default ``[1.0]``) — every coupon geared down."""
    compare_bond(_bond(market, gearings=[0.8]), cpp["cms_gearings"], loose)


def test_gearings_vector(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A per-period gearing vector shorter than the leg: the last entry repeats."""
    compare_bond(
        _bond(market, gearings=[0.8, 0.9, 1.1]), cpp["cms_gearings_vector"], loose
    )


def test_spreads_scalar(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``spreads=[0.002]`` (default ``[0.0]``)."""
    compare_bond(_bond(market, spreads=[0.002]), cpp["cms_spreads"], loose)


def test_spreads_vector(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A per-period spread vector shorter than the leg."""
    compare_bond(
        _bond(market, spreads=[0.001, 0.002, 0.003]), cpp["cms_spreads_vector"], loose
    )


def test_caps(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``caps=[0.035]`` (default: none) — coupons become capped CMS coupons."""
    compare_bond(_bond(market, caps=[0.035]), cpp["cms_caps"], loose)


def test_floors(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``floors=[0.045]`` (default: none)."""
    compare_bond(_bond(market, floors=[0.045]), cpp["cms_floors"], loose)


def test_collar(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """Cap and floor together — the assetswap.cpp 5.5%/2.5% collar."""
    compare_bond(_bond(market, caps=[0.055], floors=[0.025]), cpp["cms_collar"], loose)


def test_in_arrears(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``in_arrears=True`` moves every fixing date to the accrual end."""
    ref = cpp["cms_in_arrears"]
    compare_bond(_bond(market, in_arrears=True), ref, loose)
    base_fixings = [c.get("fixing_serial") for c in cpp["cms_base"]["cashflows"]]
    assert [c.get("fixing_serial") for c in ref["cashflows"]] != base_fixings


def test_redemption(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``redemption=102.0`` (default 100) — the final redemption flow scales."""
    compare_bond(_bond(market, redemption=102.0), cpp["cms_redemption"], loose)


def test_issue_date(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``issue_date`` (default: null) clamps settlement dates before issue."""
    ref = cpp["cms_issue_date"]
    bond = _bond(market, issue_date=CMS_START)
    compare_bond(bond, ref, loose)
    assert bond.issue_date() == CMS_START
    assert (
        ref["settlement_before_issue_serial"]
        != cpp["cms_base"]["settlement_before_issue_serial"]
    )


def test_settlement_days(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``settlement_days=5`` (default 2) moves the settlement date."""
    ref = cpp["cms_settlement_days"]
    compare_bond(_bond(market, settlement_days=5), ref, loose)
    assert ref["settlement_serial"] != cpp["cms_base"]["settlement_serial"]


def test_face_amount(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``face_amount=250`` scales every nominal and the redemption."""
    compare_bond(_bond(market, face_amount=250.0), cpp["cms_face_amount"], loose)


def test_zero_gearing_degenerates_to_fixed(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A zero gearing turns every coupon into a fixed coupon paying the
    cap/floor-clamped spread (cashflowvectors.hpp:133-141).

    That is the one place where gearings, spreads and caps interact, so it is
    pinned separately: ``min(0.035, max(-inf, 0.04)) == 0.035``.
    """
    ref = cpp["cms_zero_gearing"]
    bond = _bond(market, gearings=[0.0], spreads=[0.04], caps=[0.035])
    compare_bond(bond, ref, loose)
    for entry in ref["cashflows"]:
        if entry["is_coupon"]:
            assert entry["rate"] == 0.035
            assert entry["fixing_serial"] == 0


def test_maturity_is_schedule_end_date(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``maturity_date`` is ``schedule.end_date``, not the last payment date.

    C++ assigns ``maturityDate_ = schedule.endDate()`` before building the leg
    (cmsratebond.cpp:47), so a payment date rolled by the payment convention
    does not move the maturity.
    """
    bond = _bond(market, payment_convention=BusinessDayConvention.Preceding)
    assert bond.maturity_date() == Date(
        int(cpp["cms_payment_convention"]["maturity_serial"])
    )
