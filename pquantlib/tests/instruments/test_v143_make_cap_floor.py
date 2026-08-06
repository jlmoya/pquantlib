"""Cross-validate ``MakeCapFloor`` against C++ QuantLib v1.43.

Probe: ``v143/inst/makeoptions`` (``capfloor_*`` keys).

Every chained setter gets its own case, moved off its default, and every case
compares the **complete leg listing** — payment date, amount, nominal, accrual
start/end, accrual period, fixing date and rate for each coupon — not just the
NPV. An NPV can match while two errors cancel; a dropped setter cannot hide
inside the structure.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.cap_floor import CapFloor, CapFloorType
from pquantlib.instruments.make_cap_floor import MakeCapFloor
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.capfloor.bachelier_capfloor_engine import (
    BachelierCapFloorEngine,
)
from pquantlib.pricingengines.capfloor.black_capfloor_engine import BlackCapFloorEngine
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
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
_STRIKE = 0.03
_5Y = Period(5, TimeUnit.Years)


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


def _index() -> Euribor:
    return Euribor.six_months(_curve())


def _black(vol: float = 0.20) -> PricingEngine:
    return BlackCapFloorEngine(_curve(), vol, Actual365Fixed())


def _bachelier(vol: float = 0.01) -> PricingEngine:
    return BachelierCapFloorEngine(_curve(), vol, Actual365Fixed())


def _coupons(leg: Sequence[CashFlow]) -> list[Coupon]:
    """Narrow a built leg to Coupons — every flow here is one."""
    out: list[Coupon] = []
    for cf in leg:
        assert isinstance(cf, Coupon)
        out.append(cf)
    return out


def _assert_leg(leg: list[Any], ref_leg: list[dict[str, Any]]) -> None:
    """Compare a built leg flow-by-flow against the C++ listing."""
    assert len(leg) == len(ref_leg)
    for cf, ref in zip(leg, ref_leg, strict=True):
        assert cf.date().serial_number() == ref["payment_date"]
        tight(cf.amount(), ref["amount"])
        tight(cf.nominal(), ref["nominal"])
        assert cf.accrual_start_date().serial_number() == ref["accrual_start"]
        assert cf.accrual_end_date().serial_number() == ref["accrual_end"]
        tight(cf.accrual_period(), ref["accrual_period"])
        if ref["fixing_date"] is not None:
            assert cf.fixing_date().serial_number() == ref["fixing_date"]
        tight(cf.rate(), ref["rate"])


def _check(cap_floor: CapFloor, ref: dict[str, Any]) -> None:
    assert int(cap_floor.type()) == ref["type"]
    assert len(cap_floor.cap_rates()) == len(ref["cap_rates"])
    for got, want in zip(cap_floor.cap_rates(), ref["cap_rates"], strict=True):
        tight(got, want)
    assert len(cap_floor.floor_rates()) == len(ref["floor_rates"])
    for got, want in zip(cap_floor.floor_rates(), ref["floor_rates"], strict=True):
        tight(got, want)
    assert cap_floor.start_date().serial_number() == ref["start_date"]
    assert cap_floor.maturity_date().serial_number() == ref["maturity_date"]
    tight(cap_floor.npv(), ref["npv"])
    _assert_leg(cap_floor.floating_leg(), ref["leg"])


# ---------------------------------------------------------------------------
# Constructor arguments
# ---------------------------------------------------------------------------


def test_default(cpp: dict[str, Any]) -> None:
    """A zero forward start excludes the first caplet: 9 flows, not 10."""
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_pricing_engine(_black())
    cf = mk.build()
    assert len(cf.floating_leg()) == 9
    _check(cf, cpp["capfloor_default"])


def test_forward_start_keeps_first_caplet(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE, Period(1, TimeUnit.Years))
    mk.with_pricing_engine(_black())
    cf = mk.build()
    assert len(cf.floating_leg()) == 10
    _check(cf, cpp["capfloor_forward_start_1y"])


def test_floor_type(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Floor, _5Y, _index(), _STRIKE)
    mk.with_pricing_engine(_black())
    cf = mk.build()
    assert cf.type() == CapFloorType.Floor
    assert cf.cap_rates() == []
    _check(cf, cpp["capfloor_floor"])


# ---------------------------------------------------------------------------
# One test per chained setter, each set to a non-default value.
# ---------------------------------------------------------------------------


def test_with_nominal(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_nominal(2.0e6).with_pricing_engine(_black())
    cf = mk.build()
    assert _coupons(cf.floating_leg())[0].nominal() == 2.0e6
    _check(cf, cpp["capfloor_nominal"])


def test_with_effective_date_first_caplet_included(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(20, Month.March, 2024), False)
    mk.with_pricing_engine(_black())
    cf = mk.build()
    assert len(cf.floating_leg()) == 10
    _check(cf, cpp["capfloor_effective_date_incl"])


def test_with_effective_date_first_caplet_excluded(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(20, Month.March, 2024), True)
    mk.with_pricing_engine(_black())
    cf = mk.build()
    assert len(cf.floating_leg()) == 9
    _check(cf, cpp["capfloor_effective_date_excl"])


def test_with_tenor(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_tenor(Period(3, TimeUnit.Months)).with_pricing_engine(_black())
    cf = mk.build()
    # Quarterly instead of semi-annual, first caplet dropped.
    assert len(cf.floating_leg()) == 19
    _check(cf, cpp["capfloor_tenor_3m"])


def test_with_calendar(cpp: dict[str, Any]) -> None:
    """Sunday 26-May-2024 rolls to Monday 27-May — a UK holiday, not a TARGET one."""
    base = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    base.with_effective_date(Date.from_ymd(26, Month.February, 2024), False)
    base.with_tenor(Period(3, TimeUnit.Months)).with_pricing_engine(_black())
    _check(base.build(), cpp["capfloor_calendar_target"])

    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(26, Month.February, 2024), False)
    mk.with_tenor(Period(3, TimeUnit.Months)).with_calendar(UnitedKingdom())
    mk.with_pricing_engine(_black())
    cf = mk.build()
    assert [c.date() for c in cf.floating_leg()] != [c.date() for c in base.build().floating_leg()]
    _check(cf, cpp["capfloor_calendar_uk"])


def test_with_convention(cpp: dict[str, Any]) -> None:
    """Following vs ModifiedFollowing only differ across a month end."""
    base = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    base.with_effective_date(Date.from_ymd(31, Month.May, 2024), False)
    base.with_pricing_engine(_black())
    _check(base.build(), cpp["capfloor_convention_modified_following"])

    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(31, Month.May, 2024), False)
    mk.with_convention(BusinessDayConvention.Following).with_pricing_engine(_black())
    cf = mk.build()
    assert [c.date() for c in cf.floating_leg()] != [c.date() for c in base.build().floating_leg()]
    _check(cf, cpp["capfloor_convention_following"])


def test_with_termination_date_convention(cpp: dict[str, Any]) -> None:
    """A Monday effective date after 29-Feb-2024 matures on a Sunday five years on."""
    base = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    base.with_effective_date(Date.from_ymd(3, Month.June, 2024), False)
    base.with_pricing_engine(_black())
    _check(base.build(), cpp["capfloor_termination_convention_modified_following"])

    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(3, Month.June, 2024), False)
    mk.with_termination_date_convention(BusinessDayConvention.Unadjusted)
    mk.with_pricing_engine(_black())
    cf = mk.build()
    # The final accrual end moves (the payment date is rolled by the payment
    # convention either way, so it is not what shows the difference).
    base_leg = _coupons(base.build().floating_leg())
    leg = _coupons(cf.floating_leg())
    assert leg[-1].accrual_end_date() != base_leg[-1].accrual_end_date()
    assert [c.accrual_end_date() for c in leg[:-1]] == [c.accrual_end_date() for c in base_leg[:-1]]
    _check(cf, cpp["capfloor_termination_convention_unadjusted"])


def test_with_rule(cpp: dict[str, Any]) -> None:
    """Backward puts the 5Y/2Y stub first, Forward puts it last."""
    base = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    base.with_effective_date(Date.from_ymd(31, Month.January, 2024), False)
    base.with_tenor(Period(2, TimeUnit.Years)).with_pricing_engine(_black())
    backward = base.build()
    _check(backward, cpp["capfloor_rule_backward"])

    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(31, Month.January, 2024), False)
    mk.with_tenor(Period(2, TimeUnit.Years)).with_rule(DateGeneration.Forward)
    mk.with_pricing_engine(_black())
    forward = mk.build()
    # Stub first vs stub last.
    assert _coupons(backward.floating_leg())[0].accrual_period() < 1.5
    assert _coupons(forward.floating_leg())[-1].accrual_period() < 1.5
    _check(forward, cpp["capfloor_rule_forward"])


def test_with_end_of_month(cpp: dict[str, Any]) -> None:
    base = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    base.with_effective_date(Date.from_ymd(29, Month.February, 2024), False)
    base.with_pricing_engine(_black())
    _check(base.build(), cpp["capfloor_no_end_of_month"])

    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(29, Month.February, 2024), False)
    mk.with_end_of_month(True).with_pricing_engine(_black())
    cf = mk.build()
    # Every accrual end lands on a month end.
    cal = TARGET()
    assert all(cal.is_end_of_month(c.accrual_end_date()) for c in _coupons(cf.floating_leg()))
    _check(cf, cpp["capfloor_end_of_month"])


def test_with_first_date(cpp: dict[str, Any]) -> None:
    base = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    base.with_effective_date(Date.from_ymd(17, Month.January, 2024), False)
    base.with_pricing_engine(_black())
    _check(base.build(), cpp["capfloor_stub_baseline"])

    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(17, Month.January, 2024), False)
    mk.with_first_date(Date.from_ymd(17, Month.April, 2024))
    mk.with_pricing_engine(_black())
    cf = mk.build()
    # Short first (stub) period: ~3M rather than ~6M.
    assert _coupons(cf.floating_leg())[0].accrual_period() < 0.3
    _check(cf, cpp["capfloor_first_date"])


def test_with_next_to_last_date(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_effective_date(Date.from_ymd(17, Month.January, 2024), False)
    mk.with_next_to_last_date(Date.from_ymd(17, Month.October, 2028))
    mk.with_pricing_engine(_black())
    cf = mk.build()
    assert _coupons(cf.floating_leg())[-1].accrual_period() < 0.3
    _check(cf, cpp["capfloor_next_to_last_date"])


def test_with_day_count(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_day_count(Actual365Fixed()).with_pricing_engine(_black())
    cf = mk.build()
    ref = cpp["capfloor_day_count"]
    # Act/365F accruals differ from the index's Act/360 default.
    default_accrual: float = cpp["capfloor_default"]["leg"][0]["accrual_period"]
    assert _coupons(cf.floating_leg())[0].accrual_period() != default_accrual
    _check(cf, ref)


def test_as_optionlet(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.as_optionlet(True).with_pricing_engine(_black())
    cf = mk.build()
    assert len(cf.floating_leg()) == 1
    _check(cf, cpp["capfloor_optionlet"])


def test_with_pricing_engine_moves_npv(cpp: dict[str, Any]) -> None:
    """The engine is not decorative: a different vol must move the NPV."""
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    mk.with_pricing_engine(_black(0.40))
    cf = mk.build()
    _check(cf, cpp["capfloor_engine_vol_40"])
    assert cf.npv() > cpp["capfloor_default"]["npv"]


# ---------------------------------------------------------------------------
# Null strike → ATM, and its failure branch.
# ---------------------------------------------------------------------------


def test_atm_strike_from_black_engine(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index())
    mk.with_pricing_engine(_black())
    cf = mk.build()
    ref = cpp["capfloor_atm_black"]
    tight(cf.cap_rates()[0], ref["cap_rates"][0])
    _check(cf, ref)


def test_atm_strike_from_bachelier_engine(cpp: dict[str, Any]) -> None:
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index())
    mk.with_pricing_engine(_bachelier())
    cf = mk.build()
    _check(cf, cpp["capfloor_atm_bachelier"])


def test_atm_without_engine_raises(cpp: dict[str, Any]) -> None:
    assert cpp["capfloor_atm_without_engine_raises"]["raises"] is True
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index())
    with pytest.raises(LibraryException, match="cannot calculate ATM"):
        mk.build()


def test_setters_are_chainable() -> None:
    """Every ``with_*`` returns the builder, as the C++ ``MakeCapFloor&`` does."""
    mk = MakeCapFloor(CapFloorType.Cap, _5Y, _index(), _STRIKE)
    assert mk.with_nominal(1.0) is mk
    assert mk.with_effective_date(_TODAY, False) is mk
    assert mk.with_tenor(Period(6, TimeUnit.Months)) is mk
    assert mk.with_calendar(TARGET()) is mk
    assert mk.with_convention(BusinessDayConvention.Following) is mk
    assert mk.with_termination_date_convention(BusinessDayConvention.Following) is mk
    assert mk.with_rule(DateGeneration.Backward) is mk
    assert mk.with_end_of_month(False) is mk
    assert mk.with_first_date(_TODAY) is mk
    assert mk.with_next_to_last_date(_TODAY) is mk
    assert mk.with_day_count(Actual365Fixed()) is mk
    assert mk.as_optionlet(False) is mk
    assert mk.with_pricing_engine(_black()) is mk
