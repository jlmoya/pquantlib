"""Cross-validate ``MakeYoYInflationCapFloor`` against C++ QuantLib v1.43.

Probe: ``v143/inst/makeoptions`` (``yoy_*`` keys).

Every chained setter gets its own case, moved off its default, and every case
compares the complete leg listing — payment date, amount, nominal, accrual
start/end, accrual period, fixing date and rate for each coupon — plus the
strikes, the start/maturity dates and the NPV.

The YoY curve is flat at 2.5%, which makes the inflation forward exactly the
curve rate at every fixing. That is deliberate: it makes the ATM strike exactly
2.5% by construction, so ``with_atm_strike`` pins ``atmRate`` itself rather
than the curve interpolator.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest

from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.eu_hicp import YoYEUHICP
from pquantlib.instruments.make_yoy_inflation_cap_floor import (
    MakeYoYInflationCapFloor,
    make_yoy_inflation_cap_floor,
    yoy_inflation_leg,
)
from pquantlib.instruments.yoy_inflation_capfloor import (
    YoYInflationCapFloor,
    YoYInflationCapFloorType,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.inflation.yoy_inflation_capfloor_engine import (
    YoYInflationBlackCapFloorEngine,
)
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.termstructures.inflation.interpolated_yoy_inflation_curve import (
    InterpolatedYoYInflationCurve,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.inflation.constant_yoy_optionlet_volatility import (
    ConstantYoYOptionletVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.January, 2024)
_LAG = Period(3, TimeUnit.Months)
_STRIKE = 0.02
_YOY_RATE = 0.025


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


def _nominal_curve(rate: float = 0.03) -> YieldTermStructureProtocol:
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(_TODAY, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual),
    )


def _yoy_curve() -> InterpolatedYoYInflationCurve:
    return InterpolatedYoYInflationCurve(
        reference_date=_TODAY,
        dates=[_TODAY - _LAG, _TODAY + Period(30, TimeUnit.Years)],
        rates=[_YOY_RATE, _YOY_RATE],
        frequency=Frequency.Monthly,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        observation_lag=_LAG,
    )


def _index() -> YoYEUHICP:
    return YoYEUHICP(ts=_yoy_curve())


def _engine() -> PricingEngine:
    vol = ConstantYoYOptionletVolatility(
        vol=0.15,
        settlement_days=0,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Actual360(),
        observation_lag=_LAG,
        frequency=Frequency.Monthly,
        index_is_interpolated=False,
        volatility_type=VolatilityType.ShiftedLognormal,
        displacement=0.0,
    )
    return YoYInflationBlackCapFloorEngine(_index(), vol, _nominal_curve())


def _make(
    cap_floor_type: YoYInflationCapFloorType = YoYInflationCapFloorType.Cap,
    length: int = 5,
) -> MakeYoYInflationCapFloor:
    return MakeYoYInflationCapFloor(
        cap_floor_type, _index(), length, TARGET(), _LAG, InterpolationType.AsIndex
    )


def _assert_leg(leg: list[Any], ref_leg: list[dict[str, Any]]) -> None:
    assert len(leg) == len(ref_leg)
    for cf, ref in zip(leg, ref_leg, strict=True):
        assert cf.date().serial_number() == ref["payment_date"]
        tight(cf.amount(), ref["amount"])
        tight(cf.nominal(), ref["nominal"])
        assert cf.accrual_start_date().serial_number() == ref["accrual_start"]
        assert cf.accrual_end_date().serial_number() == ref["accrual_end"]
        tight(cf.accrual_period(), ref["accrual_period"])
        assert cf.fixing_date().serial_number() == ref["fixing_date"]
        tight(cf.rate(), ref["rate"])


def _check(cap_floor: YoYInflationCapFloor, ref: dict[str, Any], *, npv: bool = True) -> None:
    assert int(cap_floor.type()) == ref["type"]
    for got, want in zip(cap_floor.cap_rates(), ref["cap_rates"], strict=True):
        tight(got, want)
    for got, want in zip(cap_floor.floor_rates(), ref["floor_rates"], strict=True):
        tight(got, want)
    assert cap_floor.start_date().serial_number() == ref["start_date"]
    assert cap_floor.maturity_date().serial_number() == ref["maturity_date"]
    if npv:
        tight(cap_floor.npv(), ref["npv"])
    _assert_leg(cap_floor.yoy_leg(), ref["leg"])


# ---------------------------------------------------------------------------
# Constructor arguments
# ---------------------------------------------------------------------------


def test_default(cpp: dict[str, Any]) -> None:
    cf = _make().with_strike(_STRIKE).with_pricing_engine(_engine()).build()
    assert len(cf.yoy_leg()) == 5
    _check(cf, cpp["yoy_default"])


def test_floor(cpp: dict[str, Any]) -> None:
    cf = _make(YoYInflationCapFloorType.Floor).with_strike(_STRIKE).with_pricing_engine(_engine()).build()
    assert cf.type() == YoYInflationCapFloorType.Floor
    assert cf.cap_rates() == []
    _check(cf, cpp["yoy_floor"])


def test_length(cpp: dict[str, Any]) -> None:
    cf = _make(length=3).with_strike(_STRIKE).with_pricing_engine(_engine()).build()
    assert len(cf.yoy_leg()) == 3
    _check(cf, cpp["yoy_length_3"])


# ---------------------------------------------------------------------------
# One test per chained setter, each set to a non-default value.
# ---------------------------------------------------------------------------


def test_with_nominal(cpp: dict[str, Any]) -> None:
    cf = _make().with_strike(_STRIKE).with_nominal(2.5e6).with_pricing_engine(_engine()).build()
    assert cf.yoy_leg()[0].nominal() == 2.5e6
    assert cf.npv() > cpp["yoy_default"]["npv"]
    _check(cf, cpp["yoy_nominal"])


def test_with_effective_date(cpp: dict[str, Any]) -> None:
    cf = (
        _make()
        .with_strike(_STRIKE)
        .with_effective_date(Date.from_ymd(20, Month.March, 2024))
        .with_pricing_engine(_engine())
        .build()
    )
    assert cf.yoy_leg()[0].accrual_start_date() == Date.from_ymd(20, Month.March, 2024)
    _check(cf, cpp["yoy_effective_date"])


def test_with_fixing_days_moves_spot_but_not_coupon_fixings(
    cpp: dict[str, Any],
) -> None:
    """C++ never forwards ``fixingDays_`` to the leg — only the spot date moves."""
    cf = _make().with_strike(_STRIKE).with_fixing_days(2).with_pricing_engine(_engine()).build()
    ref = cpp["yoy_fixing_days"]
    base = cpp["yoy_default"]
    # The schedule shifts by two business days...
    assert ref["start_date"] != base["start_date"]
    # ...and every coupon fixing date stays accrual_end - observation_lag.
    for row in ref["leg"]:
        assert row["fixing_date"] == (Date(row["accrual_end"]) - _LAG).serial_number()
    _check(cf, ref)


def test_with_forward_start(cpp: dict[str, Any]) -> None:
    cf = (
        _make()
        .with_strike(_STRIKE)
        .with_forward_start(Period(1, TimeUnit.Years))
        .with_pricing_engine(_engine())
        .build()
    )
    assert cf.start_date() > _make().with_strike(_STRIKE).build().start_date()
    _check(cf, cpp["yoy_forward_start"])


def test_with_payment_day_counter(cpp: dict[str, Any]) -> None:
    cf = (
        _make()
        .with_strike(_STRIKE)
        .with_payment_day_counter(Actual360())
        .with_pricing_engine(_engine())
        .build()
    )
    # Act/360 accruals are not the 30/360 default's exact 1.0.
    assert cf.yoy_leg()[0].accrual_period() != 1.0
    _check(cf, cpp["yoy_payment_day_counter"])


def test_with_payment_adjustment(cpp: dict[str, Any]) -> None:
    cf = (
        _make()
        .with_strike(_STRIKE)
        .with_payment_adjustment(BusinessDayConvention.Preceding)
        .with_pricing_engine(_engine())
        .build()
    )
    ref = cpp["yoy_payment_adjustment_preceding"]
    assert [row["payment_date"] for row in ref["leg"]] != [
        row["payment_date"] for row in cpp["yoy_default"]["leg"]
    ]
    _check(cf, ref)


def test_as_optionlet(cpp: dict[str, Any]) -> None:
    cf = _make().with_strike(_STRIKE).as_optionlet(True).with_pricing_engine(_engine()).build()
    assert len(cf.yoy_leg()) == 1
    _check(cf, cpp["yoy_optionlet"])


def test_with_first_caplet_excluded(cpp: dict[str, Any]) -> None:
    """Not expressible in C++ (declared, never defined) — checked structurally.

    The C++ ``firstCapletExcluded_`` flag and the ``leg.erase(leg.begin())``
    that consumes it both exist, so the expected leg is exactly the default
    leg minus its first coupon; that reference *is* C++ data.
    """
    cf = _make().with_strike(_STRIKE).with_first_caplet_excluded().with_pricing_engine(_engine()).build()
    ref = cpp["yoy_default"]
    assert len(cf.yoy_leg()) == 4
    _assert_leg(cf.yoy_leg(), ref["leg"][1:])
    assert cf.start_date().serial_number() == ref["leg"][1]["accrual_start"]


def test_with_pricing_engine(cpp: dict[str, Any]) -> None:
    """No engine → no NPV; C++ emits null for exactly that case."""
    assert cpp["yoy_no_engine"]["npv"] is None
    cf = _make().with_strike(_STRIKE).build()
    _check(cf, cpp["yoy_no_engine"], npv=False)
    with pytest.raises(LibraryException):
        cf.npv()


# ---------------------------------------------------------------------------
# Strike setters and their QL_REQUIRE branches
# ---------------------------------------------------------------------------


def test_with_atm_strike(cpp: dict[str, Any]) -> None:
    cf = _make().with_atm_strike(_nominal_curve()).with_pricing_engine(_engine()).build()
    ref = cpp["yoy_atm_strike"]
    tight(cf.cap_rates()[0], ref["cap_rates"][0])
    # A flat 2.5% YoY curve with gearing 1 / spread 0 makes the par rate the
    # curve rate exactly, which is what makes this a check of atmRate.
    tight(cf.cap_rates()[0], _YOY_RATE)
    assert cf.cap_rates()[0] != _STRIKE
    _check(cf, ref)


def test_strike_after_atm_raises(cpp: dict[str, Any]) -> None:
    assert cpp["yoy_strike_after_atm_raises"]["raises"] is True
    mk = _make().with_atm_strike(_nominal_curve())
    with pytest.raises(LibraryException, match="ATM strike already given"):
        mk.with_strike(_STRIKE)


def test_atm_after_strike_raises(cpp: dict[str, Any]) -> None:
    assert cpp["yoy_atm_after_strike_raises"]["raises"] is True
    mk = _make().with_strike(_STRIKE)
    with pytest.raises(LibraryException, match="explicit strike already given"):
        mk.with_atm_strike(_nominal_curve())


def test_no_strike_at_all_raises() -> None:
    """Neither setter called: C++ would compute atmRate off an empty handle."""
    with pytest.raises(LibraryException, match="no strike given"):
        _make().build()


def test_setters_are_chainable() -> None:
    """Every ``with_*`` returns the builder, as the C++ reference does."""
    mk = _make()
    assert mk.with_nominal(1.0) is mk
    assert mk.with_effective_date(_TODAY) is mk
    assert mk.with_first_caplet_excluded() is mk
    assert mk.with_payment_day_counter(Actual360()) is mk
    assert mk.with_payment_adjustment(BusinessDayConvention.Following) is mk
    assert mk.with_fixing_days(0) is mk
    assert mk.with_pricing_engine(_engine()) is mk
    assert mk.as_optionlet(False) is mk
    assert mk.with_forward_start(Period(0, TimeUnit.Days)) is mk
    assert mk.with_strike(_STRIKE) is mk


# ---------------------------------------------------------------------------
# The keyword-argument façade and the leg builder
# ---------------------------------------------------------------------------


def test_free_function_matches_builder(cpp: dict[str, Any]) -> None:
    """``make_yoy_inflation_cap_floor`` is a thin shim over the builder."""
    cf = make_yoy_inflation_cap_floor(
        YoYInflationCapFloorType.Cap,
        _index(),
        5,
        TARGET(),
        _LAG,
        InterpolationType.AsIndex,
        strike=_STRIKE,
        pricing_engine=_engine(),
    )
    _check(cf, cpp["yoy_default"])


def test_free_function_forwards_every_keyword(cpp: dict[str, Any]) -> None:
    cf = make_yoy_inflation_cap_floor(
        YoYInflationCapFloorType.Floor,
        _index(),
        3,
        TARGET(),
        _LAG,
        InterpolationType.AsIndex,
        strike=_STRIKE,
        nominal=2.5e6,
        effective_date=Date.from_ymd(20, Month.March, 2024),
        fixing_days=2,
        payment_day_counter=Actual360(),
        payment_adjustment=BusinessDayConvention.Preceding,
        first_caplet_excluded=True,
        pricing_engine=_engine(),
    )
    assert cf.type() == YoYInflationCapFloorType.Floor
    assert len(cf.yoy_leg()) == 2  # 3 annual periods minus the first caplet
    assert cf.yoy_leg()[0].nominal() == 2.5e6
    assert cf.yoy_leg()[0].accrual_period() != 1.0  # Act/360, not 30/360
    assert cf.floor_rates()[0] == _STRIKE
    assert cf.npv() > 0.0


def test_free_function_atm_keyword() -> None:
    cf = make_yoy_inflation_cap_floor(
        YoYInflationCapFloorType.Cap,
        _index(),
        5,
        TARGET(),
        _LAG,
        InterpolationType.AsIndex,
        atm_nominal_term_structure=_nominal_curve(),
        pricing_engine=_engine(),
    )
    tight(cf.cap_rates()[0], _YOY_RATE)


def test_yoy_inflation_leg_direct() -> None:
    cal = TARGET()
    start = cal.advance(_TODAY, 2, TimeUnit.Days)
    end = cal.advance(start, 3, TimeUnit.Years)
    schedule = Schedule.from_rule(
        start,
        end,
        Period(1, TimeUnit.Years),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward,
        False,
    )
    leg = yoy_inflation_leg(
        schedule,
        cal,
        _index(),
        _LAG,
        InterpolationType.AsIndex,
        notional=1000.0,
        payment_day_counter=Actual360(),
    )
    assert len(leg) == 3
    assert all(c.nominal() == 1000.0 for c in leg)
    # C++ attaches a YoYInflationCouponPricer when the leg has no caps/floors,
    # so the coupons are priceable straight out of the builder.
    for cf in leg:
        assert isinstance(cf, YoYInflationCoupon)
        tight(cf.rate(), _YOY_RATE)
