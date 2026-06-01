"""Tests for capped/floored + digital coupon families (Phase 11 W12-B).

# C++ parity:
#   ql/cashflows/capflooredcoupon.hpp + .cpp
#   ql/cashflows/digitalcoupon.hpp + .cpp
#   ql/cashflows/replication.hpp + .cpp
#   ql/cashflows/digitaliborcoupon.hpp / digitalcmscoupon.hpp
#   ql/experimental/coupons/strippedcapflooredcoupon.hpp + .cpp

Cross-validates against ``migration-harness/references/cluster/w12b.json``,
which reproduces the canonical ``digitalcoupon.cpp`` setup (Euribor6M on a flat
5% curve, fixingDays 2, 10Y-forward coupon) with a ``ConstantOptionletVolatility``
+ ``BlackIborCouponPricer``, pinned to a deterministic 15-Jan-2024 eval date.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.capped_floored_coupon import (
    CappedFlooredCmsCoupon,
    CappedFlooredIborCoupon,
)
from pquantlib.cashflows.cms_coupon import CmsCoupon
from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer
from pquantlib.cashflows.digital_cms_coupon import DigitalCmsCoupon, digital_cms_leg
from pquantlib.cashflows.digital_coupon import DigitalCoupon
from pquantlib.cashflows.digital_ibor_coupon import DigitalIborCoupon, digital_ibor_leg
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.replication import DigitalReplication, Replication
from pquantlib.cashflows.stripped_capped_floored_coupon import (
    StrippedCappedFlooredCoupon,
    stripped_capped_floored_coupon_leg,
)
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.position import PositionType
from pquantlib.pricingengines.conundrum_pricer import (
    AnalyticHaganPricer,
    YieldCurveModel,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_EVAL = Date.from_ymd(15, Month.January, 2024)
_FIXING_DAYS = 2
_NOMINAL = 1_000_000.0
_CAPLET_VOL = 0.15


def _nn(x: float | None) -> float:
    """Assert a sign-aware accessor returned a value (not None) and narrow it."""
    assert x is not None
    return x


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w12b")


@pytest.fixture(autouse=True)
def _eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    cal = TARGET()
    ObservableSettings().evaluation_date = cal.adjust(_EVAL)


# ---------------------------------------------------------------------------
# Shared fixtures: the digitalcoupon.cpp CommonVars setup
# ---------------------------------------------------------------------------


def _today() -> Date:
    return TARGET().adjust(_EVAL)


def _settlement() -> Date:
    return TARGET().advance(_today(), _FIXING_DAYS, TimeUnit.Days)


def _curve() -> FlatForward:
    return FlatForward.from_rate(_settlement(), 0.05, Actual365Fixed())


def _index() -> Euribor:
    return Euribor.six_months(_curve())


def _caplet_vol(v: float = _CAPLET_VOL) -> ConstantOptionletVolatility:
    return ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.Following,
        volatility=v,
        calendar=TARGET(),
        day_counter=Actual360(),
        reference_date=_today(),
    )


def _pricer() -> BlackIborCouponPricer:
    return BlackIborCouponPricer(_caplet_vol())


def _coupon_dates() -> tuple[Date, Date, Date]:
    cal = TARGET()
    start = cal.advance(_settlement(), 10, TimeUnit.Years)
    end = cal.advance(_settlement(), 11, TimeUnit.Years)
    return start, end, end  # start, end, payment


def _ibor_underlying(gearing: float = 1.0, spread: float = 0.0) -> IborCoupon:
    start, end, payment = _coupon_dates()
    u = IborCoupon(payment, _NOMINAL, start, end, _FIXING_DAYS, _index(), gearing, spread)
    u.set_pricer(_pricer())
    return u


# ---------------------------------------------------------------------------
# DigitalReplication — deterministic gap placement (TIGHT)
# ---------------------------------------------------------------------------


def test_replication_defaults() -> None:
    """Central, gap=1e-4 are the C++ defaults."""
    r = DigitalReplication()
    assert r.replication_type() == Replication.Central
    tight(r.gap(), 1e-4)


def test_replication_rejects_nonpositive_gap() -> None:
    with pytest.raises(LibraryException, match="Non positive epsilon"):
        DigitalReplication(Replication.Central, 0.0)


def test_replication_central_gap_split() -> None:
    """Central replication splits the gap symmetrically: left = right = gap/2.

    TIGHT: deterministic strike offsets (no pricing).
    """
    gap = 1e-4
    u = _ibor_underlying()
    dig = DigitalCoupon(
        u, 0.04, PositionType.Long, False, None, None, PositionType.Long, False, None,
        DigitalReplication(Replication.Central, gap),
    )
    # White-box check of the replication-gap placement (C++ digitalcoupon.cpp:59).
    tight(dig._call_left_eps, gap / 2.0)  # pyright: ignore[reportPrivateUsage]
    tight(dig._call_right_eps, gap / 2.0)  # pyright: ignore[reportPrivateUsage]


def test_replication_sub_super_gap_placement() -> None:
    """Sub/Super place the whole gap on one side (long call).

    TIGHT: deterministic strike offsets. Sub long-call → left=0, right=gap;
    Super long-call → left=gap, right=0 (C++ digitalcoupon.cpp:111-170).
    """
    gap = 1e-4
    u = _ibor_underlying()
    sub = DigitalCoupon(
        u, 0.04, PositionType.Long, False, None, None, PositionType.Long, False, None,
        DigitalReplication(Replication.Sub, gap),
    )
    tight(sub._call_left_eps, 0.0)  # pyright: ignore[reportPrivateUsage]
    tight(sub._call_right_eps, gap)  # pyright: ignore[reportPrivateUsage]
    sup = DigitalCoupon(
        u, 0.04, PositionType.Long, False, None, None, PositionType.Long, False, None,
        DigitalReplication(Replication.Super, gap),
    )
    tight(sup._call_left_eps, gap)  # pyright: ignore[reportPrivateUsage]
    tight(sup._call_right_eps, 0.0)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# CappedFlooredIborCoupon — cap / floor / collar rate + effective strikes
# ---------------------------------------------------------------------------


def test_capped_floored_underlying_rate(reference_data: dict[str, Any]) -> None:
    """The plain (uncapped) Ibor swaplet rate anchors the cap/floor tests.

    LOOSE: par-coupon-adjusted Black forecast off a flat curve.
    """
    loose(_ibor_underlying().rate(), reference_data["cfi_underlying_rate"])


def test_capped_ibor_rate(reference_data: dict[str, Any]) -> None:
    """Capped coupon rate = swaplet - capletRate (Black caplet).

    LOOSE: Black-vol-adjusted optionlet.
    """
    start, end, payment = _coupon_dates()
    c = CappedFlooredIborCoupon(
        payment, _NOMINAL, start, end, _FIXING_DAYS, _index(), 1.0, 0.0, 0.04, None
    )
    c.set_pricer(_pricer())
    loose(c.rate(), reference_data["cfi_capped_rate"])
    tight(_nn(c.effective_cap()), reference_data["cfi_capped_effcap"])
    # Capping reduces the rate below the uncapped swaplet.
    assert c.rate() < reference_data["cfi_underlying_rate"]


def test_floored_ibor_rate(reference_data: dict[str, Any]) -> None:
    """Floored coupon rate = swaplet + floorletRate (Black floorlet).

    LOOSE: Black-vol-adjusted optionlet.
    """
    start, end, payment = _coupon_dates()
    c = CappedFlooredIborCoupon(
        payment, _NOMINAL, start, end, _FIXING_DAYS, _index(), 1.0, 0.0, None, 0.03
    )
    c.set_pricer(_pricer())
    loose(c.rate(), reference_data["cfi_floored_rate"])
    tight(_nn(c.effective_floor()), reference_data["cfi_floored_efffloor"])
    # Flooring raises the rate above the uncapped swaplet.
    assert c.rate() > reference_data["cfi_underlying_rate"]


def test_collared_ibor_rate(reference_data: dict[str, Any]) -> None:
    """Collar rate = swaplet + floorletRate - capletRate.

    LOOSE: Black-vol-adjusted optionlets.
    """
    start, end, payment = _coupon_dates()
    c = CappedFlooredIborCoupon(
        payment, _NOMINAL, start, end, _FIXING_DAYS, _index(), 1.0, 0.0, 0.04, 0.03
    )
    c.set_pricer(_pricer())
    loose(c.rate(), reference_data["cfi_collar_rate"])
    assert c.is_capped()
    assert c.is_floored()


def test_capped_floored_geared_effective_strikes(reference_data: dict[str, Any]) -> None:
    """Effective strikes apply (strike - spread)/gearing.

    TIGHT: pure arithmetic; the rate itself is LOOSE (Black).
    """
    start, end, payment = _coupon_dates()
    c = CappedFlooredIborCoupon(
        payment, _NOMINAL, start, end, _FIXING_DAYS, _index(), 2.0, 0.005, 0.04, 0.03
    )
    c.set_pricer(_pricer())
    tight(_nn(c.effective_cap()), reference_data["cfi_geared_effcap"])
    tight(_nn(c.effective_floor()), reference_data["cfi_geared_efffloor"])
    loose(c.rate(), reference_data["cfi_geared_rate"])


def test_capped_floored_negative_gearing_swaps_roles() -> None:
    """Negative gearing swaps cap↔floor storage (C++ capflooredcoupon.cpp:55-64).

    With gearing < 0 the user-passed cap becomes the floor slot and vice versa,
    but the sign-aware ``cap()`` / ``floor()`` accessors reverse the swap.
    """
    start, end, payment = _coupon_dates()
    c = CappedFlooredIborCoupon(
        payment, _NOMINAL, start, end, _FIXING_DAYS, _index(), -1.0, 0.0, 0.04, 0.03
    )
    # cap(): gearing<0 + is_floored → returns the stored floor_ (== passed cap).
    tight(_nn(c.cap()), 0.04)
    tight(_nn(c.floor()), 0.03)


# ---------------------------------------------------------------------------
# DigitalCoupon — call/put option rate via replication
# ---------------------------------------------------------------------------


def test_digital_call_option_rate(reference_data: dict[str, Any]) -> None:
    """Asset-or-nothing call option rate via call-spread replication.

    LOOSE: finite-difference replication (≈ Cox-Rubinstein asset-or-nothing).
    """
    u = _ibor_underlying()
    rep = DigitalReplication(Replication.Central, 1e-4)
    dig = DigitalCoupon(
        u, 0.04, PositionType.Short, False, None, None, PositionType.Short, False, None, rep
    )
    dig.set_pricer(_pricer())
    loose(dig.call_option_rate(), reference_data["dc_call_option_rate"])


def test_digital_put_option_rate(reference_data: dict[str, Any]) -> None:
    """Asset-or-nothing put option rate via put-spread replication.

    LOOSE: finite-difference replication.
    """
    u = _ibor_underlying()
    rep = DigitalReplication(Replication.Central, 1e-4)
    dig = DigitalCoupon(
        u, None, PositionType.Long, False, None, 0.04, PositionType.Long, False, None, rep
    )
    dig.set_pricer(_pricer())
    loose(dig.put_option_rate(), reference_data["dc_put_option_rate"])


def test_digital_cash_or_nothing_call(reference_data: dict[str, Any]) -> None:
    """Cash-or-nothing call: fixed payoff * Heaviside, via replication.

    LOOSE: finite-difference replication.
    """
    u = _ibor_underlying()
    rep = DigitalReplication(Replication.Central, 1e-4)
    dig = DigitalCoupon(
        u, 0.04, PositionType.Long, False, 0.10, None, PositionType.Long, False, None, rep
    )
    dig.set_pricer(_pricer())
    loose(dig.call_option_rate(), reference_data["dc_cash_call_option_rate"])
    tight(_nn(dig.call_digital_payoff()), 0.10)


def test_digital_replication_type_ordering(reference_data: dict[str, Any]) -> None:
    """Central / Sub / Super call rates match C++ and are monotone Sub<Central<Super.

    LOOSE: finite-difference replication.
    """
    u = _ibor_underlying()

    def call_rate(t: Replication) -> float:
        dig = DigitalCoupon(
            u, 0.04, PositionType.Long, False, None, None, PositionType.Long, False, None,
            DigitalReplication(t, 1e-4),
        )
        dig.set_pricer(_pricer())
        return dig.call_option_rate()

    central = call_rate(Replication.Central)
    sub = call_rate(Replication.Sub)
    sup = call_rate(Replication.Super)
    loose(central, reference_data["dc_central_call_rate"])
    loose(sub, reference_data["dc_sub_call_rate"])
    loose(sup, reference_data["dc_super_call_rate"])
    assert sub < central < sup


def test_digital_inspectors() -> None:
    """has_call / has_put / has_collar / is_long_* discriminants."""
    u = _ibor_underlying()
    rep = DigitalReplication()
    collar = DigitalCoupon(
        u, 0.05, PositionType.Long, False, None, 0.03, PositionType.Short, False, None, rep
    )
    assert collar.has_call()
    assert collar.has_put()
    assert collar.has_collar()
    assert collar.is_long_call()
    assert not collar.is_long_put()
    tight(_nn(collar.call_strike()), 0.05)
    tight(_nn(collar.put_strike()), 0.03)


# ---------------------------------------------------------------------------
# StrippedCappedFlooredCoupon — extracted optionality value
# ---------------------------------------------------------------------------


def _capped_floored(cap: float | None, floor: float | None) -> CappedFlooredIborCoupon:
    start, end, payment = _coupon_dates()
    c = CappedFlooredIborCoupon(
        payment, _NOMINAL, start, end, _FIXING_DAYS, _index(), 1.0, 0.0, cap, floor
    )
    c.set_pricer(_pricer())
    return c


def test_stripped_capped_value(reference_data: dict[str, Any]) -> None:
    """Stripped long-cap value = capletRate = swaplet - cappedCoupon.rate().

    TIGHT: the stripper just re-reads the underlying pricer's caplet rate; the
    identity stripped = underlying_swaplet - capped_rate is exact.
    """
    cfc = _capped_floored(0.04, None)
    s = StrippedCappedFlooredCoupon(cfc)
    s.set_pricer(_pricer())
    loose(s.rate(), reference_data["scf_capped_rate"])
    # Exact identity: stripped == swaplet - capped_coupon_rate.
    tight(s.rate(), reference_data["cfi_underlying_rate"] - cfc.rate())
    assert s.is_cap()
    assert not s.is_floor()


def test_stripped_floored_value(reference_data: dict[str, Any]) -> None:
    """Stripped long-floor value = floorletRate = flooredCoupon.rate() - swaplet.

    TIGHT identity.
    """
    cfc = _capped_floored(None, 0.03)
    s = StrippedCappedFlooredCoupon(cfc)
    s.set_pricer(_pricer())
    loose(s.rate(), reference_data["scf_floored_rate"])
    tight(s.rate(), cfc.rate() - reference_data["cfi_underlying_rate"])
    assert s.is_floor()
    assert not s.is_cap()


def test_stripped_collar_value(reference_data: dict[str, Any]) -> None:
    """Stripped collar value = floorletRate - capletRate.

    LOOSE (Black optionlets). is_collar() true; cap/floor accessors pass through.
    """
    cfc = _capped_floored(0.04, 0.03)
    s = StrippedCappedFlooredCoupon(cfc)
    s.set_pricer(_pricer())
    loose(s.rate(), reference_data["scf_collar_rate"])
    assert s.is_collar()
    tight(_nn(s.cap()), reference_data["scf_collar_cap"])
    tight(_nn(s.floor()), reference_data["scf_collar_floor"])


def test_stripped_leg_passthrough() -> None:
    """stripped_capped_floored_coupon_leg wraps CappedFloored, passes others."""
    cfc = _capped_floored(0.04, None)
    plain = _ibor_underlying()
    leg = stripped_capped_floored_coupon_leg([cfc, plain])
    assert isinstance(leg[0], StrippedCappedFlooredCoupon)
    assert leg[1] is plain


# ---------------------------------------------------------------------------
# CappedFlooredCmsCoupon — uncapped Hagan CMS swaplet
# ---------------------------------------------------------------------------


def _swap_index() -> SwapIndex:
    ibor = _index()
    return SwapIndex(
        "EuriborSwapIsdaFixA",
        Period(10, TimeUnit.Years),
        ibor.fixing_days(),
        EURCurrency(),
        ibor.fixing_calendar(),
        Period(1, TimeUnit.Years),
        BusinessDayConvention.Unadjusted,
        ibor.day_counter(),
        ibor,
    )


def _swaption_vol(v: float = 0.16) -> SwaptionConstantVolatility:
    return SwaptionConstantVolatility(
        reference_date=_today(),
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.Following,
        volatility=v,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.ShiftedLognormal,
    )


def _hagan_pricer() -> AnalyticHaganPricer:
    zero_mean_rev = SimpleQuote(0.0)
    return AnalyticHaganPricer(_swaption_vol(), YieldCurveModel.Standard, zero_mean_rev)


def test_capped_floored_cms_uncapped_equals_plain(reference_data: dict[str, Any]) -> None:
    """Uncapped CappedFlooredCmsCoupon degenerates to the plain Hagan CMS rate.

    LOOSE: Hagan convexity-adjusted CMS swaplet (static replication). With no
    CMS cap/floor pricer, only the uncapped branch is exercisable.
    """
    curve = _curve()
    swap_index = _swap_index()
    start = curve.reference_date() + Period(20, TimeUnit.Years)
    payment = start + Period(1, TimeUnit.Years)
    end = payment

    cfc = CappedFlooredCmsCoupon(
        payment, 1.0, start, end, swap_index.fixing_days(), swap_index,
        1.0, 0.0, None, None, start, end, _index().day_counter(),
    )
    cfc.set_pricer(_hagan_pricer())
    loose(cfc.rate(), reference_data["cfcms_uncapped_rate"])

    plain = CmsCoupon(
        payment, 1.0, start, end, swap_index.fixing_days(), swap_index,
        1.0, 0.0, start, end, _index().day_counter(),
    )
    plain.set_pricer(_hagan_pricer())
    loose(cfc.rate(), plain.rate())
    loose(plain.rate(), reference_data["cfcms_plain_rate"])


# ---------------------------------------------------------------------------
# Leg builders
# ---------------------------------------------------------------------------


def _schedule() -> Schedule:
    start = TARGET().advance(_settlement(), 10, TimeUnit.Years)
    end = TARGET().advance(_settlement(), 12, TimeUnit.Years)
    return Schedule.from_rule(
        start,
        end,
        Period(6, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Forward,
        False,
    )


def test_digital_ibor_leg_builds_digital_coupons() -> None:
    """digital_ibor_leg builds a leg of DigitalIborCoupon at the call strike."""
    leg = digital_ibor_leg(
        _schedule(),
        _index(),
        _NOMINAL,
        call_strikes=0.04,
        long_call_option=PositionType.Long,
    )
    assert len(leg) >= 1
    assert all(isinstance(c, DigitalIborCoupon) for c in leg)
    first = leg[0]
    assert isinstance(first, DigitalIborCoupon)
    assert first.has_call()
    tight(_nn(first.call_strike()), 0.04)


def test_digital_cms_leg_builds_digital_cms_coupons() -> None:
    """digital_cms_leg builds a leg of DigitalCmsCoupon at the call strike."""
    leg = digital_cms_leg(
        _schedule(),
        _swap_index(),
        _NOMINAL,
        call_strikes=0.05,
        long_call_option=PositionType.Long,
    )
    assert len(leg) >= 1
    assert all(isinstance(c, DigitalCmsCoupon) for c in leg)
    first = leg[0]
    assert isinstance(first, DigitalCmsCoupon)
    tight(_nn(first.call_strike()), 0.05)
