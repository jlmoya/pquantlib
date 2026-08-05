"""Cross-validate AmortizingFloatingRateBond against ``v143/inst/bondsamort``.

This is the widest constructor of the four — nineteen arguments, thirteen of
them optional, all forwarded into ``IborLeg``. Each optional one has a test
that sets it to a NON-default value and asserts the whole bond against C++,
including the payment lag and all four ex-coupon settings, which are exactly
the kind of argument that gets accepted and dropped.

The amortisation is swept the same three ways as
:mod:`test_amortizing_cms_rate_bond`.

Tolerance: TIGHT. Unlike the CMS bonds, an IBOR leg's rate is a short chain of
arithmetic over discount factors (the par-coupon forecast) or a historic
fixing, with no solver or quadrature in the path. The capped/floored cases add
one Black formula evaluation, which is still closed form.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise
from typing import Any, cast

import pytest

from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer, set_coupon_pricer
from pquantlib.instruments.bonds.amortizing_floating_rate_bond import (
    AmortizingFloatingRateBond,
)
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

from ._bondsamort import (
    CMS_START,
    CUSTOM_NOTIONALS_FRN,
    DC_360,
    TODAY,
    CmsMarket,
    build_market,
    compare_bond,
    frn_schedule,
    load_reference,
    pinned_evaluation_date,
)
from .test_amortizing_cms_rate_bond import sinking_vector


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

#: Probe ``AmFrnSpec::pricer`` — 0 leaves the pricer ``IborLeg`` attaches,
#: 1 attaches a 20%-vol Black pricer, 2 a 0%-vol one.
_NO_PRICER = 0
_VOL_20 = 1
_VOL_0 = 2


def _bond(
    market: CmsMarket, pricer: int = _NO_PRICER, **overrides: Any
) -> AmortizingFloatingRateBond:
    """Probe ``emitAmFrnBond`` — defaults, with one argument overridden."""
    args: dict[str, Any] = {
        "settlement_days": 2,
        "notionals": [100.0],
        "schedule": frn_schedule(),
        "ibor_index": cast(IborIndexProtocol, market.ibor),
        "payment_day_counter": DC_360,
    }
    args.update(overrides)
    bond = AmortizingFloatingRateBond(**args)
    bond.set_pricing_engine(market.engine)
    # C++'s IborLeg attaches a default BlackIborCouponPricer only when the leg
    # has no caps, no floors and is not in arrears; otherwise the caller must.
    if pricer == _VOL_20:
        set_coupon_pricer(bond.cashflows(), BlackIborCouponPricer(market.optionlet_vol_20))
    elif pricer == _VOL_0:
        set_coupon_pricer(bond.cashflows(), BlackIborCouponPricer(market.optionlet_vol_0))
    return bond


# --- the amortisation sweep -------------------------------------------------


def test_constant_notional(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A single-entry notional vector is a bullet: one redemption, flat nominal."""
    ref = cpp["amfrn_constant"]
    bond = _bond(market)
    compare_bond(bond, ref, tight)
    assert len(bond.redemptions()) == 1
    assert {c["nominal"] for c in ref["cashflows"] if c["is_coupon"]} == {100.0}


def test_sinking_notionals(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """French (sinking-fund) amortisation over the semiannual leg."""
    ref = cpp["amfrn_sinking"]
    compare_bond(_bond(market, notionals=sinking_vector()), ref, tight)
    nominals = [c["nominal"] for c in ref["cashflows"] if c["is_coupon"]]
    assert len(nominals) == 10
    assert all(a > b for a, b in pairwise(nominals))


def test_custom_notionals(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A hand-written per-period notional vector."""
    ref = cpp["amfrn_custom"]
    bond = _bond(market, notionals=CUSTOM_NOTIONALS_FRN)
    compare_bond(bond, ref, tight)
    assert [
        c["nominal"] for c in ref["cashflows"] if c["is_coupon"]
    ] == CUSTOM_NOTIONALS_FRN
    assert len(bond.redemptions()) == 10


def test_redemptions_vector(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``redemptions=[100, 101, 102]`` (default ``[100]``)."""
    ref = cpp["amfrn_redemptions"]
    compare_bond(
        _bond(
            market, notionals=CUSTOM_NOTIONALS_FRN, redemptions=[100.0, 101.0, 102.0]
        ),
        ref,
        tight,
    )
    assert ref["npv"] != cpp["amfrn_custom"]["npv"]


# --- one non-default per remaining optional argument ------------------------


def test_payment_convention(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``payment_convention=Preceding`` (default Following)."""
    ref = cpp["amfrn_payment_convention"]
    compare_bond(
        _bond(
            market,
            notionals=CUSTOM_NOTIONALS_FRN,
            payment_convention=BusinessDayConvention.Preceding,
        ),
        ref,
        tight,
    )
    base_dates = [c["date_serial"] for c in cpp["amfrn_custom"]["cashflows"]]
    assert [c["date_serial"] for c in ref["cashflows"]] != base_dates


def test_fixing_days(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``fixing_days=10`` (default: the index's own 2)."""
    ref = cpp["amfrn_fixing_days"]
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS_FRN, fixing_days=10), ref, tight
    )
    base_fixings = [c.get("fixing_serial") for c in cpp["amfrn_custom"]["cashflows"]]
    assert [c.get("fixing_serial") for c in ref["cashflows"]] != base_fixings


def test_gearings(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``gearings=[0.8]`` (default ``[1.0]``)."""
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS_FRN, gearings=[0.8]),
        cpp["amfrn_gearings"],
        tight,
    )


def test_spreads(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``spreads=[0.002]`` (default ``[0.0]``)."""
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS_FRN, spreads=[0.002]),
        cpp["amfrn_spreads"],
        tight,
    )


def test_caps(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``caps=[0.026]`` (default: none) — binds against a ~3% forward."""
    compare_bond(
        _bond(market, _VOL_20, notionals=CUSTOM_NOTIONALS_FRN, caps=[0.026]),
        cpp["amfrn_caps"],
        tight,
    )


def test_floors(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``floors=[0.032]`` (default: none) — binds against a ~3% forward."""
    compare_bond(
        _bond(market, _VOL_20, notionals=CUSTOM_NOTIONALS_FRN, floors=[0.032]),
        cpp["amfrn_floors"],
        tight,
    )


def test_in_arrears(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``in_arrears=True`` moves every fixing date to the accrual end.

    Priced at zero optionlet vol: C++ applies an in-arrears convexity
    adjustment that PQuantLib's ``BlackIborCouponPricer`` documents as a
    carve-out, and at zero variance that adjustment is identically zero — so
    the test isolates the fixing-date move rather than re-testing the carve-out.
    """
    ref = cpp["amfrn_in_arrears"]
    compare_bond(
        _bond(market, _VOL_0, notionals=CUSTOM_NOTIONALS_FRN, in_arrears=True),
        ref,
        tight,
    )
    base_fixings = [c.get("fixing_serial") for c in cpp["amfrn_custom"]["cashflows"]]
    assert [c.get("fixing_serial") for c in ref["cashflows"]] != base_fixings


def test_issue_date(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``issue_date`` (default: null) clamps settlement dates before issue."""
    ref = cpp["amfrn_issue_date"]
    bond = _bond(market, notionals=CUSTOM_NOTIONALS_FRN, issue_date=CMS_START)
    compare_bond(bond, ref, tight)
    assert bond.issue_date() == CMS_START
    assert (
        ref["settlement_before_issue_serial"]
        != cpp["amfrn_custom"]["settlement_before_issue_serial"]
    )


def test_settlement_days(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """``settlement_days=5`` (default 2)."""
    ref = cpp["amfrn_settlement_days"]
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS_FRN, settlement_days=5), ref, tight
    )
    assert ref["settlement_serial"] != cpp["amfrn_custom"]["settlement_serial"]


def test_payment_lag(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """``payment_lag=3`` (default 0) — every payment date moves three days out.

    This is the exact defect class the swap ports suffered from: a payment lag
    that was stored and never forwarded to the leg builder.
    """
    ref = cpp["amfrn_payment_lag"]
    compare_bond(
        _bond(market, notionals=CUSTOM_NOTIONALS_FRN, payment_lag=3), ref, tight
    )
    base_dates = [c["date_serial"] for c in cpp["amfrn_custom"]["cashflows"]]
    lagged = [c["date_serial"] for c in ref["cashflows"]]
    assert lagged != base_dates
    assert all(b <= a for a, b in zip(lagged, base_dates, strict=True))


def test_ex_coupon(pinned_today: None, market: CmsMarket, cpp: dict[str, Any]) -> None:
    """A 5-day ex-coupon period on TARGET, Preceding (default: no ex-coupon)."""
    ref = cpp["amfrn_ex_coupon"]
    compare_bond(
        _bond(
            market,
            notionals=CUSTOM_NOTIONALS_FRN,
            ex_coupon_period=Period(5, TimeUnit.Days),
            ex_coupon_calendar=TARGET(),
            ex_coupon_convention=BusinessDayConvention.Preceding,
        ),
        ref,
        tight,
    )
    assert all(c["ex_coupon_serial"] == 0 for c in cpp["amfrn_custom"]["cashflows"])
    assert any(c["ex_coupon_serial"] != 0 for c in ref["cashflows"])


def test_ex_coupon_end_of_month(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A 1-month ex-coupon period on NullCalendar, Following, end_of_month=True.

    All four ex-coupon arguments differ from :func:`test_ex_coupon`, so the
    resulting ex-dates pin every one of them.
    """
    ref = cpp["amfrn_ex_coupon_eom"]
    compare_bond(
        _bond(
            market,
            notionals=CUSTOM_NOTIONALS_FRN,
            ex_coupon_period=Period(1, TimeUnit.Months),
            ex_coupon_calendar=NullCalendar(),
            ex_coupon_convention=BusinessDayConvention.Following,
            ex_coupon_end_of_month=True,
        ),
        ref,
        tight,
    )
    five_day = [c["ex_coupon_serial"] for c in cpp["amfrn_ex_coupon"]["cashflows"]]
    assert [c["ex_coupon_serial"] for c in ref["cashflows"]] != five_day


def test_zero_gearing_degenerates_to_fixed(
    pinned_today: None, market: CmsMarket, cpp: dict[str, Any]
) -> None:
    """A zero gearing turns every coupon into a fixed coupon paying the
    floor-clamped spread: ``max(0.045, 0.04) == 0.045``."""
    ref = cpp["amfrn_zero_gearing"]
    compare_bond(
        _bond(
            market,
            notionals=CUSTOM_NOTIONALS_FRN,
            gearings=[0.0],
            spreads=[0.04],
            floors=[0.045],
        ),
        ref,
        tight,
    )
    for entry in ref["cashflows"]:
        if entry["is_coupon"]:
            assert entry["rate"] == 0.045
            assert entry["fixing_serial"] == 0
