"""Cross-validate ``MakeCms`` against C++ QuantLib v1.43.

Probe: ``v143/inst/makeoptions`` (``cms_*`` keys).

``MakeCms`` has twenty-four chained setters over two independently-scheduled
legs, so each test moves exactly one off its default and compares **both**
complete leg listings against C++. Where a setter needs a particular date to
bite (a calendar disagreement, a month-end roll, a non-business termination, a
schedule stub), the probe emits a matching baseline case built from the same
effective date so the delta isolates that one setter.

NPV tolerance: the swap NPV is a difference of two legs of order 0.14 that
lands near -2.5e-4, so ~1e-15 relative agreement per coupon shows up as ~1e-10
relative on the total. LOOSE (abs 1e-8) covers that cancellation comfortably;
the per-coupon rates and amounts, which are what actually pin the port, are
compared at TIGHT.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cms_coupon_pricer import CmsCouponPricer
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.instruments.make_cms import MakeCms
from pquantlib.instruments.swap import Swap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.conundrum_pricer import (
    AnalyticHaganPricer,
    YieldCurveModel,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.January, 2024)
_SPREAD = 0.0010
_5Y = Period(5, TimeUnit.Years)
_2Y = Period(2, TimeUnit.Years)
_6M = Period(6, TimeUnit.Months)
_26FEB = Date.from_ymd(26, Month.February, 2024)
_31MAY = Date.from_ymd(31, Month.May, 2024)
_3JUN = Date.from_ymd(3, Month.June, 2024)
_29FEB = Date.from_ymd(29, Month.February, 2024)
_17JAN = Date.from_ymd(17, Month.January, 2024)
_NPV_REASON = (
    "CMS swap NPV is a ~1e-4 residual of two ~0.14 legs; per-coupon agreement "
    "of ~1e-15 relative is amplified by the cancellation to ~1e-10 relative "
    "(absolute difference stays below 1e-16)."
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/makeoptions")


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = _TODAY
    yield
    s.evaluation_date = old


def _curve(rate: float = 0.03) -> YieldTermStructureProtocol:
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(_TODAY, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual),
    )


def _swap_index() -> EuriborSwapIsdaFixA:
    return EuriborSwapIsdaFixA(Period(10, TimeUnit.Years), _curve())


def _ibor_index() -> Euribor:
    return Euribor.three_months(_curve())


def _pricer() -> CmsCouponPricer:
    vol = SwaptionConstantVolatility(
        reference_date=_TODAY,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.Following,
        volatility=0.16,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.ShiftedLognormal,
    )
    return AnalyticHaganPricer(vol, YieldCurveModel.Standard, SimpleQuote(0.01))


def _make(*, with_pricer: bool = True) -> MakeCms:
    mk = MakeCms(_5Y, _swap_index(), _ibor_index(), _SPREAD)
    if with_pricer:
        mk.with_cms_coupon_pricer(_pricer())
    return mk


def _coupons(leg: Sequence[CashFlow]) -> list[Coupon]:
    """Narrow a built leg to Coupons — every flow on a CMS swap is one."""
    out: list[Coupon] = []
    for cf in leg:
        assert isinstance(cf, Coupon)
        out.append(cf)
    return out


def _assert_leg(leg: list[Any], ref_leg: list[dict[str, Any]]) -> None:
    assert len(leg) == len(ref_leg)
    for cf, ref in zip(leg, ref_leg, strict=True):
        assert cf.date().serial_number() == ref["payment_date"]
        assert cf.accrual_start_date().serial_number() == ref["accrual_start"]
        assert cf.accrual_end_date().serial_number() == ref["accrual_end"]
        tight(cf.nominal(), ref["nominal"])
        tight(cf.accrual_period(), ref["accrual_period"])
        assert cf.fixing_date().serial_number() == ref["fixing_date"]
        if ref["rate"] is None:
            # C++ could not price it either (no coupon pricer attached).
            with pytest.raises(LibraryException):
                cf.rate()
        else:
            tight(cf.rate(), ref["rate"])
            tight(cf.amount(), ref["amount"])


def _check(swap: Swap, ref: dict[str, Any]) -> None:
    if ref["npv"] is None:
        with pytest.raises(LibraryException):
            swap.npv()
    else:
        loose(swap.npv(), ref["npv"], reason=_NPV_REASON)
    _assert_leg(swap.leg(0), ref["leg0"])
    _assert_leg(swap.leg(1), ref["leg1"])


# ---------------------------------------------------------------------------
# Constructors + pricing wiring
# ---------------------------------------------------------------------------


def test_default_without_pricer(cpp: dict[str, Any]) -> None:
    """No CMS coupon pricer: C++ cannot produce a rate either, and says so."""
    _check(_make(with_pricer=False).build(), cpp["cms_default_no_pricer"])


def test_with_cms_coupon_pricer(cpp: dict[str, Any]) -> None:
    swap = _make().build()
    ref = cpp["cms_default"]
    # The pricer is what turns an unpriceable leg into a priced one.
    assert ref["leg0"][0]["rate"] is not None
    assert cpp["cms_default_no_pricer"]["leg0"][0]["rate"] is None
    _check(swap, ref)


def test_constructor_without_ibor_index(cpp: dict[str, Any]) -> None:
    """The 3-argument C++ constructor takes the IBOR index off the swap index."""
    mk = MakeCms(_5Y, _swap_index(), None, _SPREAD).with_cms_coupon_pricer(_pricer())
    _check(mk.build(), cpp["cms_ctor_without_ibor_index"])


# ---------------------------------------------------------------------------
# Top-level setters
# ---------------------------------------------------------------------------


def test_receive_cms(cpp: dict[str, Any]) -> None:
    swap = _make().receive_cms(True).build()
    ref = cpp["cms_receive"]
    # Leg order flips: the float leg becomes leg 0.
    assert ref["leg0"] == cpp["cms_default"]["leg1"]
    assert ref["leg1"] == cpp["cms_default"]["leg0"]
    _check(swap, ref)


def test_with_nominal(cpp: dict[str, Any]) -> None:
    swap = _make().with_nominal(1.0e6).build()
    assert _coupons(swap.leg(0))[0].nominal() == 1.0e6
    _check(swap, cpp["cms_nominal"])


def test_with_effective_date(cpp: dict[str, Any]) -> None:
    swap = _make().with_effective_date(Date.from_ymd(20, Month.March, 2024)).build()
    assert _coupons(swap.leg(0))[0].accrual_start_date() == Date.from_ymd(20, Month.March, 2024)
    _check(swap, cpp["cms_effective_date"])


# ---------------------------------------------------------------------------
# CMS-leg setters
# ---------------------------------------------------------------------------


def test_with_cms_leg_tenor(cpp: dict[str, Any]) -> None:
    swap = _make().with_cms_leg_tenor(_6M).build()
    assert len(swap.leg(0)) == 10
    assert len(swap.leg(1)) == 20
    _check(swap, cpp["cms_leg_tenor_6m"])


def test_with_cms_leg_calendar(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_26FEB).build()
    _check(base, cpp["cms_effective_26feb"])
    swap = _make().with_effective_date(_26FEB).with_cms_leg_calendar(UnitedKingdom()).build()
    assert [c.date() for c in swap.leg(0)] != [c.date() for c in base.leg(0)]
    # The float leg is untouched.
    assert [c.date() for c in swap.leg(1)] == [c.date() for c in base.leg(1)]
    _check(swap, cpp["cms_leg_calendar_uk"])


def test_with_cms_leg_convention(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_31MAY).build()
    _check(base, cpp["cms_effective_31may"])
    swap = (
        _make().with_effective_date(_31MAY).with_cms_leg_convention(BusinessDayConvention.Following).build()
    )
    assert [c.date() for c in swap.leg(0)] != [c.date() for c in base.leg(0)]
    _check(swap, cpp["cms_leg_convention_following"])


def test_with_cms_leg_termination_date_convention(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_3JUN).build()
    _check(base, cpp["cms_effective_3jun"])
    swap = (
        _make()
        .with_effective_date(_3JUN)
        .with_cms_leg_termination_date_convention(BusinessDayConvention.Unadjusted)
        .build()
    )
    assert _coupons(swap.leg(0))[-1].accrual_end_date() != _coupons(base.leg(0))[-1].accrual_end_date()
    _check(swap, cpp["cms_leg_termination_convention_unadjusted"])


def test_with_cms_leg_rule(cpp: dict[str, Any]) -> None:
    base = _make().with_cms_leg_tenor(_2Y).build()
    _check(base, cpp["cms_leg_rule_backward_2y"])
    swap = _make().with_cms_leg_tenor(_2Y).with_cms_leg_rule(DateGeneration.Forward).build()
    # Backward puts the 1Y stub first, Forward puts it last.
    assert _coupons(base.leg(0))[0].accrual_period() < 1.5
    assert _coupons(swap.leg(0))[-1].accrual_period() < 1.5
    _check(swap, cpp["cms_leg_rule_forward"])


def test_with_cms_leg_end_of_month(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_29FEB).build()
    _check(base, cpp["cms_effective_29feb"])
    swap = _make().with_effective_date(_29FEB).with_cms_leg_end_of_month(True).build()
    cal = TARGET()
    assert all(cal.is_end_of_month(c.accrual_end_date()) for c in _coupons(swap.leg(0)))
    _check(swap, cpp["cms_leg_end_of_month"])


def test_with_cms_leg_first_date(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_17JAN).build()
    _check(base, cpp["cms_effective_17jan"])
    swap = (
        _make()
        .with_effective_date(_17JAN)
        .with_cms_leg_first_date(Date.from_ymd(17, Month.March, 2024))
        .build()
    )
    assert _coupons(swap.leg(0))[0].accrual_period() < _coupons(base.leg(0))[0].accrual_period()
    _check(swap, cpp["cms_leg_first_date"])


def test_with_cms_leg_next_to_last_date(cpp: dict[str, Any]) -> None:
    swap = (
        _make()
        .with_effective_date(_17JAN)
        .with_cms_leg_next_to_last_date(Date.from_ymd(17, Month.November, 2028))
        .build()
    )
    _check(swap, cpp["cms_leg_next_to_last_date"])


def test_with_cms_leg_day_count(cpp: dict[str, Any]) -> None:
    swap = _make().with_cms_leg_day_count(Thirty360(Thirty360Convention.BondBasis)).build()
    # 30/360 accruals differ from the Act/360 default.
    assert _coupons(swap.leg(0))[0].accrual_period() != _coupons(swap.leg(1))[0].accrual_period()
    _check(swap, cpp["cms_leg_day_count"])


# ---------------------------------------------------------------------------
# Floating-leg setters
# ---------------------------------------------------------------------------


def test_with_floating_leg_tenor(cpp: dict[str, Any]) -> None:
    swap = _make().with_floating_leg_tenor(_6M).build()
    assert len(swap.leg(1)) == 10
    assert len(swap.leg(0)) == 20
    _check(swap, cpp["cms_float_leg_tenor_6m"])


def test_with_floating_leg_calendar(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_26FEB).build()
    swap = _make().with_effective_date(_26FEB).with_floating_leg_calendar(UnitedKingdom()).build()
    assert [c.date() for c in swap.leg(1)] != [c.date() for c in base.leg(1)]
    _check(swap, cpp["cms_float_leg_calendar_uk"])


def test_with_floating_leg_convention(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_31MAY).build()
    swap = (
        _make()
        .with_effective_date(_31MAY)
        .with_floating_leg_convention(BusinessDayConvention.Following)
        .build()
    )
    assert [c.date() for c in swap.leg(1)] != [c.date() for c in base.leg(1)]
    _check(swap, cpp["cms_float_leg_convention_following"])


def test_with_floating_leg_termination_date_convention(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_3JUN).build()
    swap = (
        _make()
        .with_effective_date(_3JUN)
        .with_floating_leg_termination_date_convention(BusinessDayConvention.Unadjusted)
        .build()
    )
    assert _coupons(swap.leg(1))[-1].accrual_end_date() != _coupons(base.leg(1))[-1].accrual_end_date()
    _check(swap, cpp["cms_float_leg_termination_convention_unadjusted"])


def test_with_floating_leg_rule(cpp: dict[str, Any]) -> None:
    base = _make().with_floating_leg_tenor(_2Y).build()
    _check(base, cpp["cms_float_leg_rule_backward_2y"])
    swap = _make().with_floating_leg_tenor(_2Y).with_floating_leg_rule(DateGeneration.Forward).build()
    assert _coupons(base.leg(1))[0].accrual_period() < 1.5
    assert _coupons(swap.leg(1))[-1].accrual_period() < 1.5
    _check(swap, cpp["cms_float_leg_rule_forward"])


def test_with_floating_leg_end_of_month(cpp: dict[str, Any]) -> None:
    swap = _make().with_effective_date(_29FEB).with_floating_leg_end_of_month(True).build()
    cal = TARGET()
    assert all(cal.is_end_of_month(c.accrual_end_date()) for c in _coupons(swap.leg(1)))
    _check(swap, cpp["cms_float_leg_end_of_month"])


def test_with_floating_leg_first_date(cpp: dict[str, Any]) -> None:
    base = _make().with_effective_date(_17JAN).build()
    swap = (
        _make()
        .with_effective_date(_17JAN)
        .with_floating_leg_first_date(Date.from_ymd(17, Month.March, 2024))
        .build()
    )
    assert _coupons(swap.leg(1))[0].accrual_period() < _coupons(base.leg(1))[0].accrual_period()
    _check(swap, cpp["cms_float_leg_first_date"])


def test_with_floating_leg_next_to_last_date(cpp: dict[str, Any]) -> None:
    swap = (
        _make()
        .with_effective_date(_17JAN)
        .with_floating_leg_next_to_last_date(Date.from_ymd(17, Month.November, 2028))
        .build()
    )
    _check(swap, cpp["cms_float_leg_next_to_last_date"])


def test_with_floating_leg_day_count(cpp: dict[str, Any]) -> None:
    swap = _make().with_floating_leg_day_count(Actual365Fixed()).build()
    # Act/365F accruals differ from the index's Act/360 default.
    assert _coupons(swap.leg(1))[0].accrual_period() != _coupons(swap.leg(0))[0].accrual_period()
    _check(swap, cpp["cms_float_leg_day_count"])


# ---------------------------------------------------------------------------
# Pricing setters
# ---------------------------------------------------------------------------


def test_with_discounting_term_structure(cpp: dict[str, Any]) -> None:
    swap = _make().with_discounting_term_structure(_curve(0.025)).build()
    ref = cpp["cms_discount_curve"]
    # A different discount curve must move the NPV (the legs do not move).
    assert ref["leg0"] == cpp["cms_default"]["leg0"]
    assert ref["npv"] != cpp["cms_default"]["npv"]
    _check(swap, ref)


def test_with_atm_spread(cpp: dict[str, Any]) -> None:
    """The solved spread zeroes the swap NPV and shows up on every float coupon."""
    swap = _make().with_atm_spread(True).build()
    ref = cpp["cms_atm_spread"]
    assert ref["leg1"] != cpp["cms_default"]["leg1"]
    assert abs(swap.npv()) < 1e-14
    _check(swap, ref)


def test_atm_spread_without_pricer_raises(cpp: dict[str, Any]) -> None:
    assert cpp["cms_atm_spread_without_pricer_raises"]["raises"] is True
    mk = _make(with_pricer=False).with_atm_spread(True)
    with pytest.raises(LibraryException, match="CmsCouponPricer"):
        mk.build()


def test_setters_are_chainable() -> None:
    """Every ``with_*`` / ``receive_cms`` returns the builder, as C++ does."""
    mk = _make()
    assert mk.receive_cms(False) is mk
    assert mk.with_nominal(1.0) is mk
    assert mk.with_effective_date(_17JAN) is mk
    assert mk.with_cms_leg_tenor(_6M) is mk
    assert mk.with_cms_leg_calendar(TARGET()) is mk
    assert mk.with_cms_leg_convention(BusinessDayConvention.Following) is mk
    assert mk.with_cms_leg_termination_date_convention(BusinessDayConvention.Following) is mk
    assert mk.with_cms_leg_rule(DateGeneration.Backward) is mk
    assert mk.with_cms_leg_end_of_month(False) is mk
    assert mk.with_cms_leg_first_date(_17JAN) is mk
    assert mk.with_cms_leg_next_to_last_date(_17JAN) is mk
    assert mk.with_cms_leg_day_count(Actual365Fixed()) is mk
    assert mk.with_floating_leg_tenor(_6M) is mk
    assert mk.with_floating_leg_calendar(TARGET()) is mk
    assert mk.with_floating_leg_convention(BusinessDayConvention.Following) is mk
    assert mk.with_floating_leg_termination_date_convention(BusinessDayConvention.Following) is mk
    assert mk.with_floating_leg_rule(DateGeneration.Backward) is mk
    assert mk.with_floating_leg_end_of_month(False) is mk
    assert mk.with_floating_leg_first_date(_17JAN) is mk
    assert mk.with_floating_leg_next_to_last_date(_17JAN) is mk
    assert mk.with_floating_leg_day_count(Actual365Fixed()) is mk
    assert mk.with_atm_spread(False) is mk
    assert mk.with_discounting_term_structure(_curve()) is mk
    assert mk.with_cms_coupon_pricer(_pricer()) is mk
