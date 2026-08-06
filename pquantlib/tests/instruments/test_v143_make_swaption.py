"""Cross-validate ``MakeSwaption`` against C++ QuantLib v1.43.

Probe: ``v143/inst/makeoptions`` (``swaption_*`` keys).

Every chained setter gets its own case, moved off its default, and every case
compares both complete legs of the underlying swap — payment date, amount,
nominal, accrual start/end, accrual period, fixing date, rate — plus the
exercise date, the settlement type/method, the strike and the NPV.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.instruments.make_swaption import MakeSwaption
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    Swaption,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.swaption.black_swaption_engine import BlackSwaptionEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.January, 2024)
_STRIKE = 0.03
_5Y = Period(5, TimeUnit.Years)
_10Y = Period(10, TimeUnit.Years)
_19W = Period(19, TimeUnit.Weeks)


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
    return EuriborSwapIsdaFixA(_10Y, _curve())


def _engine() -> PricingEngine:
    return BlackSwaptionEngine(_curve(), 0.16, Actual365Fixed())


def _assert_leg(leg: list[Any], ref_leg: list[dict[str, Any]]) -> None:
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


def _check(swaption: Swaption, ref: dict[str, Any], *, npv: bool = True) -> None:
    assert int(swaption.settlement_type) == ref["settlement_type"]
    assert int(swaption.settlement_method) == ref["settlement_method"]
    assert int(swaption.type()) == ref["swap_type"]
    assert swaption.exercise().dates()[0].serial_number() == ref["exercise_date"]
    underlying = swaption.underlying_swap()
    tight(underlying.fixed_rate(), ref["strike"])
    tight(underlying.nominal(), ref["nominal"])
    if npv:
        tight(swaption.npv(), ref["npv"])
    _assert_leg(underlying.fixed_leg(), ref["fixed_leg"])
    _assert_leg(underlying.floating_leg(), ref["floating_leg"])


def _default() -> MakeSwaption:
    return MakeSwaption(_swap_index(), _5Y, _STRIKE).with_pricing_engine(_engine())


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def test_default(cpp: dict[str, Any]) -> None:
    _check(_default().build(), cpp["swaption_default"])


def test_atm_strike(cpp: dict[str, Any]) -> None:
    """A null strike strikes at the index's own underlying-swap fair rate."""
    mk = MakeSwaption(_swap_index(), _5Y).with_pricing_engine(_engine())
    sw = mk.build()
    ref = cpp["swaption_atm"]
    tight(sw.underlying_swap().fixed_rate(), ref["strike"])
    assert sw.underlying_swap().fixed_rate() != _STRIKE
    _check(sw, ref)


def test_fixing_date_constructor(cpp: dict[str, Any]) -> None:
    mk = MakeSwaption.from_fixing_date(_swap_index(), Date.from_ymd(20, Month.June, 2029), _STRIKE)
    mk.with_pricing_engine(_engine())
    sw = mk.build()
    ref = cpp["swaption_fixing_date_ctor"]
    assert sw.exercise().dates()[0].serial_number() == ref["exercise_date"]
    assert ref["exercise_date"] != cpp["swaption_default"]["exercise_date"]
    _check(sw, ref)


# ---------------------------------------------------------------------------
# One test per chained setter, each set to a non-default value.
# ---------------------------------------------------------------------------


def test_with_nominal(cpp: dict[str, Any]) -> None:
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE).with_nominal(1.0e6)
    mk.with_pricing_engine(_engine())
    sw = mk.build()
    assert sw.underlying_swap().nominal() == 1.0e6
    assert sw.npv() > _default().build().npv()
    _check(sw, cpp["swaption_nominal"])


def test_with_settlement_type_and_method_cash_par_yield(cpp: dict[str, Any]) -> None:
    """Cash/ParYieldCurve reaches the instrument; only its NPV is out of reach.

    ``BlackSwaptionEngine`` has a documented carve-out for the ParYieldCurve
    annuity (black_swaption_engine.py), so the C++ NPV cannot be reproduced
    yet — the setters themselves are still cross-validated through the built
    instrument and both legs, and the carve-out is pinned as a raise.
    """
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    mk.with_settlement_type(SettlementType.Cash)
    mk.with_settlement_method(SettlementMethod.ParYieldCurve)
    mk.with_pricing_engine(_engine())
    sw = mk.build()
    assert sw.settlement_type == SettlementType.Cash
    assert sw.settlement_method == SettlementMethod.ParYieldCurve
    _check(sw, cpp["swaption_cash_par_yield"], npv=False)
    # In C++ the cash annuity moves the price off the physical-settlement one.
    assert cpp["swaption_cash_par_yield"]["npv"] != cpp["swaption_default"]["npv"]
    with pytest.raises(LibraryException, match="ParYieldCurve"):
        sw.npv()


def test_with_settlement_method_collateralized(cpp: dict[str, Any]) -> None:
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    mk.with_settlement_type(SettlementType.Cash)
    mk.with_settlement_method(SettlementMethod.CollateralizedCashPrice)
    mk.with_pricing_engine(_engine())
    sw = mk.build()
    ref = cpp["swaption_cash_collateralized"]
    assert ref["npv"] != cpp["swaption_cash_par_yield"]["npv"]
    _check(sw, ref)


def test_with_option_convention(cpp: dict[str, Any]) -> None:
    """Preceding vs ModifiedFollowing moves the fixing date off a holiday."""
    mk = MakeSwaption(_swap_index(), Period(6, TimeUnit.Years) + Period(3, TimeUnit.Months), _STRIKE)
    mk.with_option_convention(BusinessDayConvention.Preceding)
    mk.with_pricing_engine(_engine())
    sw = mk.build()
    ref = cpp["swaption_option_convention_preceding"]
    assert ref["exercise_date"] != cpp["swaption_default"]["exercise_date"]
    _check(sw, ref)


def test_with_exercise_date(cpp: dict[str, Any]) -> None:
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    mk.with_exercise_date(Date.from_ymd(10, Month.January, 2029))
    mk.with_pricing_engine(_engine())
    sw = mk.build()
    ref = cpp["swaption_exercise_date"]
    # The exercise moves but the underlying (driven by the fixing date) does not.
    assert ref["exercise_date"] != cpp["swaption_default"]["exercise_date"]
    assert ref["floating_leg"] == cpp["swaption_default"]["floating_leg"]
    assert sw.npv() != _default().build().npv()
    _check(sw, ref)


def test_with_exercise_calendar(cpp: dict[str, Any]) -> None:
    """15-Jan-2024 + 19 weeks = 27-May-2024, a UK holiday but a TARGET business day."""
    base = MakeSwaption(_swap_index(), _19W, _STRIKE).with_pricing_engine(_engine())
    _check(base.build(), cpp["swaption_exercise_calendar_target"])

    mk = MakeSwaption(_swap_index(), _19W, _STRIKE)
    mk.with_exercise_calendar(UnitedKingdom()).with_pricing_engine(_engine())
    sw = mk.build()
    assert sw.exercise().dates()[0] != base.build().exercise().dates()[0]
    _check(sw, cpp["swaption_exercise_calendar_uk"])


def test_with_underlying_type(cpp: dict[str, Any]) -> None:
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    mk.with_underlying_type(SwapType.Receiver).with_pricing_engine(_engine())
    sw = mk.build()
    assert sw.type() == SwapType.Receiver
    assert sw.npv() != _default().build().npv()
    _check(sw, cpp["swaption_receiver"])


def test_with_indexed_coupons(cpp: dict[str, Any]) -> None:
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    mk.with_indexed_coupons(True).with_pricing_engine(_engine())
    sw = mk.build()
    ref = cpp["swaption_indexed_coupons"]
    # Indexed coupons genuinely differ from the library default.
    assert ref["floating_leg"] != cpp["swaption_default"]["floating_leg"]
    _check(sw, ref)


def test_with_at_par_coupons(cpp: dict[str, Any]) -> None:
    """At-par is the library default, so C++ makes this a no-op — pinned as such."""
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    mk.with_at_par_coupons(True).with_pricing_engine(_engine())
    sw = mk.build()
    ref = cpp["swaption_at_par_coupons"]
    assert ref["floating_leg"] == cpp["swaption_default"]["floating_leg"]
    assert ref["floating_leg"] != cpp["swaption_indexed_coupons"]["floating_leg"]
    _check(sw, ref)


def test_with_pricing_engine_moves_npv() -> None:
    """No engine → no NPV; a different vol → a different NPV."""
    no_engine = MakeSwaption(_swap_index(), _5Y, _STRIKE).build()
    with pytest.raises(LibraryException):
        no_engine.npv()
    hi = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    hi.with_pricing_engine(BlackSwaptionEngine(_curve(), 0.40, Actual365Fixed()))
    assert hi.build().npv() > _default().build().npv()


# ---------------------------------------------------------------------------
# QL_REQUIRE branches
# ---------------------------------------------------------------------------


def test_exercise_after_fixing_raises(cpp: dict[str, Any]) -> None:
    assert cpp["swaption_exercise_after_fixing_raises"]["raises"] is True
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    mk.with_exercise_date(Date.from_ymd(15, Month.January, 2035))
    with pytest.raises(LibraryException, match="exercise date"):
        mk.build()


def test_atm_without_curve_raises(cpp: dict[str, Any]) -> None:
    assert cpp["swaption_atm_without_curve_raises"]["raises"] is True
    bare = EuriborSwapIsdaFixA(_10Y)
    mk = MakeSwaption(bare, _5Y)
    with pytest.raises(LibraryException, match="null term structure"):
        mk.build()


def test_fixing_date_is_cached_across_builds() -> None:
    """C++ caches the derived fixing date on the (mutable) builder."""
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE).with_pricing_engine(_engine())
    first = mk.build().exercise().dates()[0]
    ObservableSettings().evaluation_date = Date.from_ymd(15, Month.July, 2024)
    second = mk.build().exercise().dates()[0]
    assert second == first


def test_setters_are_chainable() -> None:
    """Every ``with_*`` returns the builder, as the C++ ``MakeSwaption&`` does."""
    mk = MakeSwaption(_swap_index(), _5Y, _STRIKE)
    assert mk.with_nominal(1.0) is mk
    assert mk.with_settlement_type(SettlementType.Physical) is mk
    assert mk.with_settlement_method(SettlementMethod.PhysicalOTC) is mk
    assert mk.with_option_convention(BusinessDayConvention.Following) is mk
    assert mk.with_exercise_date(_TODAY) is mk
    assert mk.with_exercise_calendar(TARGET()) is mk
    assert mk.with_underlying_type(SwapType.Payer) is mk
    assert mk.with_indexed_coupons(None) is mk
    assert mk.with_at_par_coupons(True) is mk
    assert mk.with_pricing_engine(_engine()) is mk
