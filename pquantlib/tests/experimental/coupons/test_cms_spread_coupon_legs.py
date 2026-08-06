"""Cross-validation for the ql/experimental/coupons CMS-spread coupon family.

# C++ parity:
#   ql/experimental/coupons/cmsspreadcoupon.{hpp,cpp}          (v1.43)
#   ql/experimental/coupons/digitalcmsspreadcoupon.{hpp,cpp}   (v1.43)
#   ql/experimental/coupons/strippedcapflooredcoupon.{hpp,cpp} (v1.43)

Every expected value in this module comes from running C++ QuantLib v1.43 —
``migration-harness/cpp/probes/v143_experimental_coupons/probe.cpp`` →
``migration-harness/references/v143/experimental/coupons.json``.

Tolerance tiers used here
-------------------------
* ``exact`` — everything integral or discrete: date serial numbers, leg sizes,
  the per-period coupon-type dispatch, booleans, and the cap / floor / strike /
  payoff values, which are echoes (or one multiply-and-add) of the constructor
  arguments and agree bit-for-bit with C++.
* ``tight`` — day-count arithmetic (accrual periods), nominals, and the
  underlying par-swap and spread-index fixings.
* ``loose`` — anything produced by
  :class:`~pquantlib.cashflows.lognormal_cms_spread_pricer.LognormalCmsSpreadPricer`
  (rates, amounts, option rates, convexity adjustments). Reason: the value is a
  Hagan static replication feeding a 16-point Gauss-Hermite quadrature; the C++
  ``-O3`` build contracts several of those products into FMAs, so agreement
  lands around 1e-13 relative, degrading to ~3e-12 on the collar payoffs where
  two option values of order 1e-2 cancel down to order 1e-4. Multiplying by the
  1e6 nominal then removes the TIGHT tier's 1e-14 absolute floor. This is the
  tier the repo already applies to this pricer
  (``tests/cashflows/test_lognormal_cms_spread_pricer.py``). Largest deviation
  observed across all 466 numeric comparisons in this module: 2.8e-12 relative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pquantlib.cashflows.capped_floored_coupon import CappedFlooredCoupon
from pquantlib.cashflows.cms_spread_coupon import CmsSpreadCoupon
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.digital_coupon import DigitalCoupon
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.lognormal_cms_spread_pricer import LognormalCmsSpreadPricer
from pquantlib.cashflows.replication import DigitalReplication, Replication
from pquantlib.cashflows.stripped_capped_floored_coupon import StrippedCappedFlooredCoupon
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.coupons.cms_spread_coupon import (
    CappedFlooredCmsSpreadCoupon,
    CmsSpreadLeg,
)
from pquantlib.experimental.coupons.digital_cms_spread_coupon import (
    DigitalCmsSpreadCoupon,
    DigitalCmsSpreadLeg,
)
from pquantlib.experimental.coupons.stripped_capfloored_coupon import (
    StrippedCappedFlooredCouponLeg,
)
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.indexes.swap_spread_index import SwapSpreadIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.position import PositionType
from pquantlib.pricingengines.conundrum_pricer import AnalyticHaganPricer, YieldCurveModel
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.cashflows.coupon_pricer import FloatingRateCouponPricer

_TODAY = Date.from_ymd(15, Month.January, 2024)


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return load_reference("v143/experimental/coupons")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the evaluation date to the probe's, then restore it."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _TODAY
    yield
    settings.evaluation_date = previous


# ---------------------------------------------------------------------------
# Market — a transcription of the probe's makeMarket()
# ---------------------------------------------------------------------------


class _Market:
    """Flat 5% curve, 10Y/2Y EuriborSwapIsdaFixA-style indexes, 16% lognormal vol."""

    def __init__(self, correlation: float = 0.5) -> None:
        self.curve = FlatForward.from_rate(_TODAY, 0.05, Actual365Fixed())
        self.ibor = Euribor.six_months(self.curve)
        self.s10 = self._swap_index(Period(10, TimeUnit.Years))
        self.s2 = self._swap_index(Period(2, TimeUnit.Years))
        self.ssi = SwapSpreadIndex("CMS10Y-2Y", self.s10, self.s2, 1.0, -1.0)
        vol = SwaptionConstantVolatility(
            reference_date=_TODAY,
            calendar=TARGET(),
            business_day_convention=BusinessDayConvention.Following,
            volatility=0.16,
            day_counter=Actual365Fixed(),
            volatility_type=VolatilityType.ShiftedLognormal,
        )
        cms_pricer = AnalyticHaganPricer(vol, YieldCurveModel.Standard, SimpleQuote(0.0))
        self.pricer: FloatingRateCouponPricer = LognormalCmsSpreadPricer(
            cms_pricer, SimpleQuote(correlation), self.curve, 16
        )

    def _swap_index(self, tenor: Period) -> SwapIndex:
        ibor = self.ibor
        return SwapIndex(
            "EuriborSwapIsdaFixA",
            tenor,
            ibor.fixing_days(),
            ibor.currency(),
            ibor.fixing_calendar(),
            Period(1, TimeUnit.Years),
            BusinessDayConvention.Unadjusted,
            ibor.day_counter(),
            ibor,
        )

    # The single-coupon block's dates: start = reference + 20Y, 1Y accrual.
    def coupon_dates(self) -> tuple[Date, Date]:
        start = self.curve.reference_date() + Period(20, TimeUnit.Years)
        return start, start + Period(1, TimeUnit.Years)

    def regular_schedule(self) -> Schedule:
        start = self.curve.reference_date() + Period(2, TimeUnit.Years)
        return Schedule.from_rule(
            start,
            start + Period(5, TimeUnit.Years),
            Period(1, TimeUnit.Years),
            TARGET(),
            BusinessDayConvention.ModifiedFollowing,
            BusinessDayConvention.ModifiedFollowing,
            DateGeneration.Forward,
            False,
        )

    def stub_schedule(self) -> Schedule:
        """A schedule whose first period is a short stub (isRegular(1) is False)."""
        start = self.curve.reference_date() + Period(2, TimeUnit.Years)
        return Schedule.from_rule(
            start,
            start + Period(3, TimeUnit.Years) + Period(5, TimeUnit.Months),
            Period(1, TimeUnit.Years),
            TARGET(),
            BusinessDayConvention.ModifiedFollowing,
            BusinessDayConvention.ModifiedFollowing,
            DateGeneration.Forward,
            False,
            start + Period(5, TimeUnit.Months),
        )


@pytest.fixture
def market() -> _Market:
    return _Market()


def _set_pricer(leg: Sequence[CashFlow], pricer: FloatingRateCouponPricer) -> None:
    """Attach the pricer to every floating coupon.

    # C++ parity note: ``setCouponPricer(leg, pricer)`` dispatches through the
    # ``PricerSetter`` AcyclicVisitor, which has no ``CmsSpreadCoupon`` case, so
    # it silently does nothing for these legs. The probe walks the leg by hand
    # for the same reason.
    """
    for cf in leg:
        if isinstance(cf, FloatingRateCoupon):
            cf.set_pricer(pricer)


def _type_tag(cf: CashFlow) -> str:
    """Most-derived-first type tag, mirroring the probe's ``typeTag``."""
    if isinstance(cf, StrippedCappedFlooredCoupon):
        return "StrippedCappedFlooredCoupon"
    if isinstance(cf, DigitalCmsSpreadCoupon):
        return "DigitalCmsSpreadCoupon"
    if isinstance(cf, DigitalCoupon):
        return "DigitalCoupon"
    if isinstance(cf, CappedFlooredCmsSpreadCoupon):
        return "CappedFlooredCmsSpreadCoupon"
    if isinstance(cf, CappedFlooredCoupon):
        return "CappedFlooredCoupon"
    if isinstance(cf, CmsSpreadCoupon):
        return "CmsSpreadCoupon"
    if isinstance(cf, FixedRateCoupon):
        return "FixedRateCoupon"
    return "CashFlow"


def _assert_leg(ref: dict[str, Any], tag: str, leg: Sequence[CashFlow]) -> None:
    """Assert a whole leg against the reference: dispatch, dates, and values."""
    assert len(leg) == ref[f"{tag}_size"]
    assert ",".join(_type_tag(cf) for cf in leg) == ref[f"{tag}_types"]

    for cf, expected in zip(leg, ref[f"{tag}_dates"], strict=True):
        exact(cf.date().serial, expected)

    coupons = [cf for cf in leg if isinstance(cf, Coupon)]
    for key, actual_dates in (
        ("accrual_start", [c.accrual_start_date().serial for c in coupons]),
        ("accrual_end", [c.accrual_end_date().serial for c in coupons]),
        ("ref_start", [c.reference_period_start().serial for c in coupons]),
        ("ref_end", [c.reference_period_end().serial for c in coupons]),
    ):
        for actual, expected in zip(actual_dates, ref[f"{tag}_{key}"], strict=True):
            exact(actual, expected)

    for c, expected in zip(coupons, ref[f"{tag}_accrual_periods"], strict=True):
        tight(c.accrual_period(), expected)
    for c, expected in zip(coupons, ref[f"{tag}_nominals"], strict=True):
        tight(c.nominal(), expected)
    for cf, expected in zip(leg, ref[f"{tag}_amounts"], strict=True):
        loose(cf.amount(), expected)
    for c, expected in zip(coupons, ref[f"{tag}_rates"], strict=True):
        loose(c.rate(), expected)


def _assert_opt_rate(ref: dict[str, Any], key: str, actual: float | None) -> None:
    """Assert a ``Rate`` that may be C++ ``Null<Rate>()`` (Python ``None``)."""
    assert (actual is None) is ref[f"{key}_is_null"]
    if actual is not None:
        exact(actual, ref[key])


# ---------------------------------------------------------------------------
# Block A — the market itself
# ---------------------------------------------------------------------------


def test_market_setup(ref: dict[str, Any], market: _Market) -> None:
    """The curve / index / date scaffolding every other test stands on."""
    start, end = market.coupon_dates()
    probe = CmsSpreadCoupon(
        end, 1.0, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.0,
        start, end, market.ibor.day_counter(),
    )
    exact(_TODAY.serial, ref["A_today"])
    exact(market.curve.reference_date().serial, ref["A_reference_date"])
    exact(start.serial, ref["A_start"])
    exact(end.serial, ref["A_end"])
    exact(probe.fixing_date().serial, ref["A_fixing_date"])
    exact(market.ssi.fixing_days(), ref["A_ssi_fixing_days"])
    exact(market.ssi.gearing1(), ref["A_gearing1"])
    exact(market.ssi.gearing2(), ref["A_gearing2"])

    fixing_date = probe.fixing_date()
    tight(market.s10.fixing(fixing_date), ref["A_fix1_10y"])
    tight(market.s2.fixing(fixing_date), ref["A_fix2_2y"])
    # TIGHT holds on its 1e-14 absolute floor: the spread is the difference of
    # two par swap rates near 0.0506, so it cancels down to ~1.4e-6 and the
    # 1.5e-16 absolute disagreement is ~3 ULP at the inputs' scale.
    tight(market.ssi.fixing(fixing_date), ref["A_ssi_fixing"])
    tight(probe.accrual_period(), ref["A_accrual_period"])


# ---------------------------------------------------------------------------
# Block B — plain CmsSpreadCoupon (the base the ported classes wrap)
# ---------------------------------------------------------------------------


def test_cms_spread_coupon_priced(ref: dict[str, Any], market: _Market) -> None:
    """The spread coupon the capped/floored and digital variants are built on."""
    start, end = market.coupon_dates()
    coupon = CmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.001,
        start, end, market.ibor.day_counter(),
    )
    coupon.set_pricer(market.pricer)

    loose(coupon.rate(), ref["B_rate"])
    loose(coupon.amount(), ref["B_amount"])
    loose(coupon.adjusted_fixing(), ref["B_adjusted_fixing"])
    loose(coupon.convexity_adjustment(), ref["B_convexity_adjustment"])
    tight(coupon.index_fixing(), ref["B_index_fixing"])
    tight(coupon.accrual_period(), ref["B_accrual_period"])
    exact(coupon.date().serial, ref["B_payment_date"])
    exact(coupon.accrual_start_date().serial, ref["B_accrual_start"])
    exact(coupon.accrual_end_date().serial, ref["B_accrual_end"])
    exact(coupon.fixing_date().serial, ref["B_fixing_date"])


def test_cms_spread_coupon_gearing_and_spread(ref: dict[str, Any], market: _Market) -> None:
    """Gearing 2 / spread -20bp pass through into rate and adjusted fixing."""
    start, end = market.coupon_dates()
    coupon = CmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 2.0, -0.002,
        start, end, market.ibor.day_counter(),
    )
    coupon.set_pricer(market.pricer)
    loose(coupon.rate(), ref["B_geared_rate"])
    loose(coupon.amount(), ref["B_geared_amount"])
    loose(coupon.adjusted_fixing(), ref["B_geared_adjusted_fixing"])
    loose(coupon.convexity_adjustment(), ref["B_geared_convexity_adjustment"])


# ---------------------------------------------------------------------------
# Block C — CappedFlooredCmsSpreadCoupon
# ---------------------------------------------------------------------------

_CAPFLOOR_CASES = [
    # tag, cap, floor, gearing, spread, in_arrears
    ("cap_only", 0.008, None, 1.0, 0.0, False),
    ("floor_only", None, 0.004, 1.0, 0.0, False),
    ("collar", 0.008, 0.004, 1.0, 0.0, False),
    # gearing < 0 makes CappedFlooredCoupon swap the cap and floor roles.
    ("neg_gearing_collar", 0.020, 0.010, -1.0, 0.02, False),
    ("in_arrears_cap", 0.010, None, 1.0, 0.0015, True),
]


@pytest.mark.parametrize(("tag", "cap", "floor", "gearing", "spread", "arrears"), _CAPFLOOR_CASES)
def test_capped_floored_cms_spread_coupon(
    ref: dict[str, Any],
    market: _Market,
    tag: str,
    cap: float | None,
    floor: float | None,
    gearing: float,
    spread: float,
    arrears: bool,
) -> None:
    """Capped / floored / collared CMS-spread coupons, including negative gearing."""
    start, end = market.coupon_dates()
    coupon = CappedFlooredCmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, gearing, spread,
        cap, floor, start, end, market.ibor.day_counter(), arrears,
    )
    coupon.set_pricer(market.pricer)

    loose(coupon.rate(), ref[f"C_{tag}_rate"])
    loose(coupon.amount(), ref[f"C_{tag}_amount"])
    loose(coupon.convexity_adjustment(), ref[f"C_{tag}_convexity_adjustment"])
    assert coupon.is_capped() is ref[f"C_{tag}_is_capped"]
    assert coupon.is_floored() is ref[f"C_{tag}_is_floored"]
    _assert_opt_rate(ref, f"C_{tag}_cap", coupon.cap())
    _assert_opt_rate(ref, f"C_{tag}_floor", coupon.floor())
    _assert_opt_rate(ref, f"C_{tag}_effective_cap", coupon.effective_cap())
    _assert_opt_rate(ref, f"C_{tag}_effective_floor", coupon.effective_floor())
    exact(coupon.date().serial, ref[f"C_{tag}_payment_date"])
    exact(coupon.accrual_start_date().serial, ref[f"C_{tag}_accrual_start"])
    exact(coupon.accrual_end_date().serial, ref[f"C_{tag}_accrual_end"])
    tight(coupon.accrual_period(), ref[f"C_{tag}_accrual_period"])
    assert coupon.swap_spread_index() is market.ssi


def test_capped_floored_negative_gearing_swaps_cap_and_floor(
    ref: dict[str, Any], market: _Market
) -> None:
    """With gearing < 0 the reported cap/floor are NOT the constructor arguments.

    # C++ parity: ql/cashflows/capflooredcoupon.cpp:46-64 — the ``gearing_ > 0``
    # else-branch stores ``cap`` into ``floor_`` and ``floor`` into ``cap_``.
    """
    start, end = market.coupon_dates()
    coupon = CappedFlooredCmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, -1.0, 0.02,
        0.020, 0.010, start, end, market.ibor.day_counter(),
    )
    # constructor cap = 0.020, floor = 0.010 — but:
    exact(coupon.cap() if coupon.cap() is not None else 0.0, ref["C_neg_gearing_collar_cap"])
    exact(coupon.floor() if coupon.floor() is not None else 0.0, ref["C_neg_gearing_collar_floor"])
    assert coupon.cap() == 0.020
    assert coupon.floor() == 0.010
    # effectiveCap = (cap_ - spread)/gearing = (0.010 - 0.02)/(-1) = 0.010
    assert coupon.effective_cap() == 0.010


def test_capped_floored_in_arrears_fixing_date(ref: dict[str, Any], market: _Market) -> None:
    """In-arrears shifts the fixing date to the accrual end."""
    start, end = market.coupon_dates()
    coupon = CappedFlooredCmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.0015,
        0.010, None, start, end, market.ibor.day_counter(), True,
    )
    assert coupon.is_in_arrears() is ref["C_in_arrears_cap_is_in_arrears"]
    exact(coupon.fixing_date().serial, ref["C_in_arrears_cap_fixing_date"])


def test_capped_floored_cap_below_floor_raises(market: _Market) -> None:
    """# C++ parity: ql/cashflows/capflooredcoupon.cpp:66-70 QL_REQUIRE."""
    start, end = market.coupon_dates()
    with pytest.raises(LibraryException, match="less than floor level"):
        CappedFlooredCmsSpreadCoupon(
            end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.0,
            0.002, 0.008, start, end, market.ibor.day_counter(),
        )


# ---------------------------------------------------------------------------
# Block D — StrippedCappedFlooredCoupon over the CMS-spread family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "cap", "floor"),
    [("cap_only", 0.008, None), ("floor_only", None, 0.004), ("collar", 0.008, 0.004)],
)
def test_stripped_capped_floored_cms_spread_coupon(
    ref: dict[str, Any], market: _Market, tag: str, cap: float | None, floor: float | None
) -> None:
    """The embedded optionality alone: collar → floorlet - caplet."""
    start, end = market.coupon_dates()
    underlying = CappedFlooredCmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.0,
        cap, floor, start, end, market.ibor.day_counter(),
    )
    stripped = StrippedCappedFlooredCoupon(underlying)
    stripped.set_pricer(market.pricer)

    loose(stripped.rate(), ref[f"D_{tag}_rate"])
    loose(stripped.amount(), ref[f"D_{tag}_amount"])
    loose(stripped.convexity_adjustment(), ref[f"D_{tag}_convexity_adjustment"])
    assert stripped.is_cap() is ref[f"D_{tag}_is_cap"]
    assert stripped.is_floor() is ref[f"D_{tag}_is_floor"]
    assert stripped.is_collar() is ref[f"D_{tag}_is_collar"]
    _assert_opt_rate(ref, f"D_{tag}_cap", stripped.cap())
    _assert_opt_rate(ref, f"D_{tag}_floor", stripped.floor())
    _assert_opt_rate(ref, f"D_{tag}_effective_cap", stripped.effective_cap())
    _assert_opt_rate(ref, f"D_{tag}_effective_floor", stripped.effective_floor())
    exact(stripped.date().serial, ref[f"D_{tag}_payment_date"])
    tight(stripped.accrual_period(), ref[f"D_{tag}_accrual_period"])
    assert stripped.underlying() is underlying


# ---------------------------------------------------------------------------
# Block E — DigitalCmsSpreadCoupon
# ---------------------------------------------------------------------------

_CENTRAL = DigitalReplication(Replication.Central, 1e-4)
_SUB = DigitalReplication(Replication.Sub, 1e-4)
_SUPER = DigitalReplication(Replication.Super, 1e-4)

_DIGITAL_CASES = [
    # tag, call_strike, call_pos, call_atm, call_payoff,
    #      put_strike, put_pos, put_atm, put_payoff, replication, naked
    ("con_long_call", 0.005, PositionType.Long, False, 0.03,
     None, PositionType.Long, False, None, _CENTRAL, False),
    ("con_long_put", None, PositionType.Long, False, None,
     0.007, PositionType.Long, False, 0.02, _CENTRAL, False),
    ("con_collar", 0.005, PositionType.Short, False, 0.03,
     0.007, PositionType.Long, False, 0.02, _CENTRAL, False),
    ("aon_long_call", 0.005, PositionType.Long, False, None,
     None, PositionType.Long, False, None, _CENTRAL, False),
    ("aon_long_put", None, PositionType.Long, False, None,
     0.007, PositionType.Long, False, None, _CENTRAL, False),
    ("con_long_call_sub", 0.005, PositionType.Long, False, 0.03,
     None, PositionType.Long, False, None, _SUB, False),
    ("con_long_call_super", 0.005, PositionType.Long, False, 0.03,
     None, PositionType.Long, False, None, _SUPER, False),
    ("con_long_call_naked", 0.005, PositionType.Long, False, 0.03,
     None, PositionType.Long, False, None, _CENTRAL, True),
    # replication=None → DigitalCoupon substitutes DigitalReplication()
    # (Central, gap 1e-4), matching C++ digitalcoupon.cpp:54-55.
    ("con_long_call_default_replication", 0.005, PositionType.Long, False, 0.03,
     None, PositionType.Long, False, None, None, False),
]


@pytest.mark.parametrize(
    (
        "tag", "call_strike", "call_position", "call_atm", "call_payoff",
        "put_strike", "put_position", "put_atm", "put_payoff", "replication", "naked",
    ),
    _DIGITAL_CASES,
)
def test_digital_cms_spread_coupon(
    ref: dict[str, Any],
    market: _Market,
    tag: str,
    call_strike: float | None,
    call_position: PositionType,
    call_atm: bool,
    call_payoff: float | None,
    put_strike: float | None,
    put_position: PositionType,
    put_atm: bool,
    put_payoff: float | None,
    replication: DigitalReplication | None,
    naked: bool,
) -> None:
    """Digital CMS-spread coupons: cash- and asset-or-nothing, Sub/Central/Super."""
    start, end = market.coupon_dates()
    underlying = CmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.0,
        start, end, market.ibor.day_counter(),
    )
    coupon = DigitalCmsSpreadCoupon(
        underlying, call_strike, call_position, call_atm, call_payoff,
        put_strike, put_position, put_atm, put_payoff, replication, naked,
    )
    coupon.set_pricer(market.pricer)

    loose(coupon.rate(), ref[f"E_{tag}_rate"])
    loose(coupon.amount(), ref[f"E_{tag}_amount"])
    loose(coupon.convexity_adjustment(), ref[f"E_{tag}_convexity_adjustment"])
    assert coupon.has_call() is ref[f"E_{tag}_has_call"]
    assert coupon.has_put() is ref[f"E_{tag}_has_put"]
    assert coupon.has_collar() is ref[f"E_{tag}_has_collar"]
    assert coupon.is_long_call() is ref[f"E_{tag}_is_long_call"]
    assert coupon.is_long_put() is ref[f"E_{tag}_is_long_put"]
    _assert_opt_rate(ref, f"E_{tag}_call_strike", coupon.call_strike())
    _assert_opt_rate(ref, f"E_{tag}_put_strike", coupon.put_strike())
    _assert_opt_rate(ref, f"E_{tag}_call_digital_payoff", coupon.call_digital_payoff())
    _assert_opt_rate(ref, f"E_{tag}_put_digital_payoff", coupon.put_digital_payoff())
    loose(coupon.call_option_rate(), ref[f"E_{tag}_call_option_rate"])
    loose(coupon.put_option_rate(), ref[f"E_{tag}_put_option_rate"])
    exact(coupon.date().serial, ref[f"E_{tag}_payment_date"])
    tight(coupon.accrual_period(), ref[f"E_{tag}_accrual_period"])
    assert coupon.underlying() is underlying


def test_digital_naked_option_drops_the_swaplet(ref: dict[str, Any], market: _Market) -> None:
    """``naked_option`` leaves only the option value: rate == callOptionRate."""
    loose(ref["E_con_long_call_naked_rate"], ref["E_con_long_call_naked_call_option_rate"])
    # and the non-naked variant is that plus the underlying spread rate
    start, end = market.coupon_dates()
    underlying = CmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.0,
        start, end, market.ibor.day_counter(),
    )
    underlying.set_pricer(market.pricer)
    loose(
        ref["E_con_long_call_rate"] - ref["E_con_long_call_call_option_rate"],
        underlying.rate(),
    )


def test_digital_payoff_without_strike_raises(market: _Market) -> None:
    """# C++ parity: ql/cashflows/digitalcoupon.cpp:61-69 QL_REQUIRE."""
    start, end = market.coupon_dates()
    underlying = CmsSpreadCoupon(
        end, 1.0e6, start, end, market.ssi.fixing_days(), market.ssi, 1.0, 0.0,
        start, end, market.ibor.day_counter(),
    )
    with pytest.raises(LibraryException, match="Call Cash rate non allowed"):
        DigitalCmsSpreadCoupon(underlying, None, PositionType.Long, False, 0.03)


# ---------------------------------------------------------------------------
# Block F — CmsSpreadLeg
# ---------------------------------------------------------------------------


def test_cms_spread_leg_schedule(ref: dict[str, Any], market: _Market) -> None:
    schedule = market.regular_schedule()
    exact(len(schedule), ref["F_schedule_size"])
    for i, expected in enumerate(ref["F_schedule_dates"]):
        exact(schedule.date(i).serial, expected)


def test_cms_spread_leg_plain(ref: dict[str, Any], market: _Market) -> None:
    """Uncapped leg: five plain CmsSpreadCoupons."""
    leg = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F1_plain", leg)


def test_cms_spread_leg_capped(ref: dict[str, Any], market: _Market) -> None:
    """A scalar cap turns every period into a CappedFlooredCmsSpreadCoupon."""
    leg = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_caps(0.008)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F2_capped", leg)


def test_cms_spread_leg_mixed_vectors(ref: dict[str, Any], market: _Market) -> None:
    """Per-period vectors, a short notionals vector (last entry repeats), and
    ``None`` cap/floor entries that switch individual periods back to plain
    CmsSpreadCoupons.
    """
    leg = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals([1.0e6, 2.0e6, 3.0e6])
        .with_payment_day_counter(Actual360())
        .with_fixing_days(2)
        .with_gearings([1.0, 1.5, 0.5, 2.0, 1.0])
        .with_spreads([0.0, 0.001, -0.001, 0.0005, 0.0])
        .with_caps([0.008, None, 0.010, None, 0.012])
        .with_floors([0.002, None, None, 0.001, 0.003])
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F3_mixed", leg)


def test_cms_spread_leg_zero_gearing_gives_fixed_coupons(
    ref: dict[str, Any], market: _Market
) -> None:
    """gearing == 0 → FixedRateCoupon at ``effectiveFixedRate(spread, cap, floor)``.

    Period 0 has spread 3% capped at 2% → 2%; period 2 has spread 3% floored at
    5% → 5%; period 4 is the unclamped 3%.
    """
    leg = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_gearings([0.0, 1.0, 0.0, 1.0, 0.0])
        .with_spreads([0.03, 0.001, 0.03, 0.001, 0.03])
        .with_caps([0.02, None, None, None, None])
        .with_floors([None, None, 0.05, None, None])
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F4_fixed_mix", leg)
    exact(leg[0].rate(), 0.02)  # type: ignore[attr-defined]
    exact(leg[2].rate(), 0.05)  # type: ignore[attr-defined]
    exact(leg[4].rate(), 0.03)  # type: ignore[attr-defined]


def test_cms_spread_leg_zero_payments(ref: dict[str, Any], market: _Market) -> None:
    """``with_zero_payments`` moves every payment to the schedule's last date.

    That is not cosmetic: the Hagan convexity adjustment is computed against the
    payment date, so the coupon rates move too.
    """
    leg = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_zero_payments(True)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F5_zero", leg)
    assert len({cf.date().serial for cf in leg}) == 1


def test_cms_spread_leg_payment_adjustment_and_arrears(
    ref: dict[str, Any], market: _Market
) -> None:
    leg = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(BusinessDayConvention.Preceding)
        .in_arrears(True)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F6_preceding_arrears", leg)


def test_cms_spread_leg_default_day_counter(ref: dict[str, Any], market: _Market) -> None:
    """No payment day counter → the coupons fall back to the index's."""
    leg = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi).with_notionals(1.0e6).leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F7_default_daycounter", leg)
    first = leg[0]
    assert isinstance(first, Coupon)
    assert first.day_counter().name() == ref["F7_daycounter_name"]


def test_cms_spread_leg_irregular_first_stub(ref: dict[str, Any], market: _Market) -> None:
    """A short first period rolls its reference start back one full tenor.

    # C++ parity: cashflowvectors.hpp:114-117 — only when the schedule carries
    # both ``isRegular`` and a tenor, and the period is irregular.
    """
    stub = market.stub_schedule()
    for i, expected in enumerate(ref["F8_schedule_dates"]):
        exact(stub.date(i).serial, expected)
    assert stub.has_is_regular() is ref["F8_schedule_has_is_regular"]
    assert [
        1 if stub.is_regular_at(i) else 0 for i in range(1, len(stub))
    ] == ref["F8_schedule_is_regular"]

    leg = (
        CmsSpreadLeg(stub, market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "F8_stub", leg)
    first = leg[0]
    assert isinstance(first, Coupon)
    # the rolled reference start precedes the accrual start
    assert first.reference_period_start() < first.accrual_start_date()


def test_cms_spread_leg_call_syntax_matches_leg(market: _Market) -> None:
    """``builder()`` is the C++ ``operator Leg()``; same result as ``.leg()``."""
    builder = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
    )
    assert [cf.date() for cf in builder()] == [cf.date() for cf in builder.leg()]


def test_cms_spread_leg_validation(market: _Market) -> None:
    """# C++ parity: the QL_REQUIRE block at cashflowvectors.hpp:79-98."""
    schedule = market.regular_schedule()
    with pytest.raises(LibraryException, match="no notional given"):
        CmsSpreadLeg(schedule, market.ssi).leg()
    with pytest.raises(LibraryException, match="too many nominals"):
        CmsSpreadLeg(schedule, market.ssi).with_notionals([1.0] * 6).leg()
    with pytest.raises(LibraryException, match="too many caps"):
        CmsSpreadLeg(schedule, market.ssi).with_notionals(1.0).with_caps([0.01] * 6).leg()
    with pytest.raises(LibraryException, match="not compatible"):
        (
            CmsSpreadLeg(schedule, market.ssi)
            .with_notionals(1.0)
            .in_arrears(True)
            .with_zero_payments(True)
            .leg()
        )


# ---------------------------------------------------------------------------
# Block G — DigitalCmsSpreadLeg
# ---------------------------------------------------------------------------


def _digital_call_leg(market: _Market) -> list[CashFlow]:
    leg = (
        DigitalCmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_call_strikes(0.005)
        .with_long_call_option(PositionType.Long)
        .with_call_payoffs(0.03)
        .with_replication(_CENTRAL)
        .with_naked_option(False)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    return leg


def _digital_collar_leg(market: _Market) -> list[CashFlow]:
    leg = (
        DigitalCmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_fixing_days(2)
        .with_spreads(0.0005)
        .with_call_strikes([0.004, 0.005, 0.006])
        .with_long_call_option(PositionType.Short)
        .with_call_atm(True)
        .with_call_payoffs([0.03, 0.02])
        .with_put_strikes([0.007, 0.008, 0.009, 0.010, 0.011])
        .with_long_put_option(PositionType.Long)
        .with_put_atm(False)
        .with_put_payoffs([0.01])
        .with_replication(_CENTRAL)
        .with_naked_option(False)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    return leg


def test_digital_cms_spread_leg_call(ref: dict[str, Any], market: _Market) -> None:
    _assert_leg(ref, "G1_call", _digital_call_leg(market))


def test_digital_cms_spread_leg_call_option_rates(ref: dict[str, Any], market: _Market) -> None:
    """The per-period replicated option rates (the leg's actual content)."""
    leg = _digital_call_leg(market)
    for cf, expected in zip(leg, ref["G1_call_call_option_rates"], strict=True):
        assert isinstance(cf, DigitalCoupon)
        loose(cf.call_option_rate(), expected)
    for cf, expected in zip(leg, ref["G1_call_put_option_rates"], strict=True):
        assert isinstance(cf, DigitalCoupon)
        loose(cf.put_option_rate(), expected)


def test_digital_cms_spread_leg_collar(ref: dict[str, Any], market: _Market) -> None:
    """Short call + long put with per-period strike/payoff vectors of different
    lengths — each one's last entry repeats to the end of the schedule.
    """
    leg = _digital_collar_leg(market)
    _assert_leg(ref, "G2_collar", leg)
    digitals = [cf for cf in leg if isinstance(cf, DigitalCoupon)]
    assert [1 if d.is_long_call() else 0 for d in digitals] == ref["G2_is_long_call"]
    assert [1 if d.is_long_put() else 0 for d in digitals] == ref["G2_is_long_put"]
    for d, expected in zip(digitals, ref["G2_call_strikes"], strict=True):
        strike = d.call_strike()
        assert strike is not None
        exact(strike, expected)
    for d, expected in zip(digitals, ref["G2_put_strikes"], strict=True):
        strike = d.put_strike()
        assert strike is not None
        exact(strike, expected)
    for d, expected in zip(digitals, ref["G2_call_payoffs"], strict=True):
        payoff = d.call_digital_payoff()
        assert payoff is not None
        exact(payoff, expected)
    for d, expected in zip(digitals, ref["G2_put_payoffs"], strict=True):
        payoff = d.put_digital_payoff()
        assert payoff is not None
        exact(payoff, expected)


def test_digital_cms_spread_leg_collar_option_rates(
    ref: dict[str, Any], market: _Market
) -> None:
    leg = _digital_collar_leg(market)
    for cf, expected in zip(leg, ref["G2_collar_call_option_rates"], strict=True):
        assert isinstance(cf, DigitalCoupon)
        loose(cf.call_option_rate(), expected)
    for cf, expected in zip(leg, ref["G2_collar_put_option_rates"], strict=True):
        assert isinstance(cf, DigitalCoupon)
        loose(cf.put_option_rate(), expected)


def test_digital_cms_spread_leg_naked_in_arrears(ref: dict[str, Any], market: _Market) -> None:
    leg = (
        DigitalCmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .in_arrears(True)
        .with_call_strikes(0.005)
        .with_call_payoffs(0.03)
        .with_replication(_CENTRAL)
        .with_naked_option(True)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "G3_naked_arrears", leg)


def test_digital_cms_spread_leg_fixed_fallback_defaults_to_one(
    ref: dict[str, Any], market: _Market
) -> None:
    """A gearing-zero period with no spreads vector pays 100%.

    # C++ parity: cashflowvectors.hpp:281-288 — ``FloatingDigitalLeg``'s fixed
    # fallback is ``detail::get(spreads, i, 1.0)``, default ONE, unlike
    # ``FloatingLeg``'s ``effectiveFixedRate`` whose spread default is 0.0. That
    # asymmetry is in v1.43 and is reproduced, not repaired.
    """
    leg = (
        DigitalCmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_gearings([0.0, 1.0, 0.0, 1.0, 1.0])
        .with_call_strikes(0.005)
        .with_call_payoffs(0.03)
        .with_replication(_CENTRAL)
        .with_naked_option(False)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "G4_fixed_default_one", leg)
    exact(leg[0].rate(), 1.0)  # type: ignore[attr-defined]
    exact(leg[2].rate(), 1.0)  # type: ignore[attr-defined]


def test_digital_cms_spread_leg_fixed_fallback_uses_spread(
    ref: dict[str, Any], market: _Market
) -> None:
    """With a spreads vector the gearing-zero periods pay the raw spread —
    NOT clamped by any cap/floor (``FloatingDigitalLeg`` has none).
    """
    leg = (
        DigitalCmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_gearings([0.0, 1.0, 0.0, 1.0, 1.0])
        .with_spreads([0.02, 0.001, 0.03, 0.001, 0.0])
        .with_call_strikes(0.005)
        .with_call_payoffs(0.03)
        .with_replication(_CENTRAL)
        .with_naked_option(False)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "G4_fixed_from_spreads", leg)
    exact(leg[0].rate(), 0.02)  # type: ignore[attr-defined]
    exact(leg[2].rate(), 0.03)  # type: ignore[attr-defined]


def test_digital_cms_spread_leg_irregular_first_stub(
    ref: dict[str, Any], market: _Market
) -> None:
    leg = (
        DigitalCmsSpreadLeg(market.stub_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_call_strikes(0.005)
        .with_call_payoffs(0.03)
        .with_replication(_CENTRAL)
        .with_naked_option(False)
        .leg()
    )
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "G5_stub", leg)


def test_digital_cms_spread_leg_naked_option_defaults_false(market: _Market) -> None:
    """A leg that never calls ``with_naked_option`` behaves as ``False``.

    # C++ parity note: ``DigitalCmsSpreadLeg::nakedOption_`` is never
    # initialised in v1.43 (digitalcmsspreadcoupon.hpp:104, .cpp:52-54), so C++
    # reads an indeterminate bool here — observed as ``true`` in one build of
    # the probe and ``false`` in another. PQuantLib initialises it to ``False``
    # (the documented default of every ``withNakedOption`` overload and of the
    # ``DigitalCoupon`` constructor) rather than reproducing the UB. The two
    # legs below are therefore required to agree, which is what the C++ author
    # plainly intended and what C++ cannot guarantee.
    """

    def build(set_flag: bool) -> list[CashFlow]:
        builder = (
            DigitalCmsSpreadLeg(market.regular_schedule(), market.ssi)
            .with_notionals(1.0e6)
            .with_payment_day_counter(Actual360())
            .with_call_strikes(0.005)
            .with_call_payoffs(0.03)
            .with_replication(_CENTRAL)
        )
        if set_flag:
            builder = builder.with_naked_option(False)
        leg = builder.leg()
        _set_pricer(leg, market.pricer)
        return leg

    unset, explicit = build(False), build(True)
    for a, b in zip(unset, explicit, strict=True):
        exact(a.amount(), b.amount())


def test_digital_cms_spread_leg_validation(market: _Market) -> None:
    """# C++ parity: the QL_REQUIRE block at cashflowvectors.hpp:244-261."""
    schedule = market.regular_schedule()
    with pytest.raises(LibraryException, match="no notional given"):
        DigitalCmsSpreadLeg(schedule, market.ssi).leg()
    with pytest.raises(LibraryException, match="too many call rates"):
        (
            DigitalCmsSpreadLeg(schedule, market.ssi)
            .with_notionals(1.0)
            .with_call_strikes([0.01] * 6)
            .leg()
        )
    with pytest.raises(LibraryException, match="too many put rates"):
        (
            DigitalCmsSpreadLeg(schedule, market.ssi)
            .with_notionals(1.0)
            .with_put_strikes([0.01] * 6)
            .leg()
        )


# ---------------------------------------------------------------------------
# Block H — StrippedCappedFlooredCouponLeg
# ---------------------------------------------------------------------------


def test_stripped_leg_strips_every_capped_coupon(ref: dict[str, Any], market: _Market) -> None:
    base = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_caps(0.008)
        .with_floors(0.004)
        .leg()
    )
    leg = StrippedCappedFlooredCouponLeg(base).leg()
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "H1_all_stripped", leg)


def test_stripped_leg_leaves_plain_coupons_alone(ref: dict[str, Any], market: _Market) -> None:
    """Only the capped/floored periods are stripped; the rest pass through."""
    base = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_caps([0.008, None, 0.010, None, 0.012])
        .leg()
    )
    assert ",".join(_type_tag(cf) for cf in base) == ref["H2_base_types"]
    leg = StrippedCappedFlooredCouponLeg(base).leg()
    _set_pricer(leg, market.pricer)
    _assert_leg(ref, "H2_mixed", leg)


def test_stripped_leg_passthrough_preserves_identity(
    ref: dict[str, Any], market: _Market
) -> None:
    """Non-capped entries are the SAME objects, not copies.

    # C++ parity: strippedcapflooredcoupon.cpp:122-128 pushes the original
    # ``shared_ptr`` back, so the two legs share those cash flows.
    """
    base = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .leg()
    )
    leg = StrippedCappedFlooredCouponLeg(base).leg()
    assert [1 if leg[i] is base[i] else 0 for i in range(len(base))] == ref[
        "H3_passthrough_identity"
    ]
    assert ",".join(_type_tag(cf) for cf in leg) == ref["H3_types"]
    exact(len(leg), ref["H3_size"])


def test_stripped_leg_empty() -> None:
    ref_size = 0
    assert len(StrippedCappedFlooredCouponLeg([]).leg()) == ref_size


def test_stripped_leg_call_syntax(market: _Market) -> None:
    base = (
        CmsSpreadLeg(market.regular_schedule(), market.ssi)
        .with_notionals(1.0e6)
        .with_payment_day_counter(Actual360())
        .with_caps(0.008)
        .leg()
    )
    wrapper = StrippedCappedFlooredCouponLeg(base)
    assert [_type_tag(cf) for cf in wrapper()] == [_type_tag(cf) for cf in wrapper.leg()]
    assert [cf.date() for cf in wrapper.underlying_leg()] == [cf.date() for cf in base]
