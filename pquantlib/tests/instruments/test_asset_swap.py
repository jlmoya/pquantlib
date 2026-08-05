"""Cross-validate AssetSwap against C++ QuantLib v1.43.

Probe: ``v143/inst/swaps`` (key ``asset_swap``).

Every case pins the headline results *and* the complete cashflow listing of
both legs — payment-date serial and amount for every flow, plus nominal,
accrual start/end serials, accrual period and rate for every coupon.  An NPV
can match while two errors cancel; a leg listing cannot.

Every optional constructor argument gets its own case at a NON-default value:

===========================  ===============================================
argument                     case
===========================  ===============================================
``pay_bond_coupon``          ``base_pay_bond_coupon`` / ``base_receive_bond_coupon``
``float_schedule``           ``float_schedule_3m``
``floating_day_count``       ``float_daycount_30360``
``par_asset_swap``           ``market_asset_swap``
``gearing``                  ``gearing_0_9``
``non_par_repayment``        ``non_par_repayment_102``
``deal_maturity``            ``deal_maturity_2027``
overnight-index branch       ``overnight_sofr`` / ``overnight_market_gearing``
===========================  ===============================================
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.asset_swap import AssetSwap
from pquantlib.instruments.bond import Bond
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# Pinned market data — mirrors migration-harness/cpp/probes/v143_inst_swaps.
_EVAL = Date.from_ymd(17, Month.January, 2024)
_FORWARD_RATE = 0.035
_DISCOUNT_RATE = 0.030
_CLEAN_PRICE = 102.5
_SPREAD = 0.0025


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _EVAL


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/swaps")


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(_EVAL, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)


def _bond() -> Bond:
    schedule = Schedule.from_rule(
        Date.from_ymd(15, Month.January, 2021),
        Date.from_ymd(15, Month.January, 2031),
        Period(1, TimeUnit.Years),
        TARGET(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        False,
    )
    return FixedRateBond(
        3,
        100.0,
        schedule,
        [0.05],
        Thirty360(Thirty360Convention.BondBasis),
        BusinessDayConvention.Following,
        100.0,
        Date.from_ymd(15, Month.January, 2021),
    )


def _float_schedule_3m() -> Schedule:
    return Schedule.from_rule(
        Date.from_ymd(22, Month.January, 2024),
        Date.from_ymd(15, Month.January, 2031),
        Period(3, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )


def _float_schedule_6m() -> Schedule:
    return Schedule.from_rule(
        Date.from_ymd(22, Month.January, 2024),
        Date.from_ymd(15, Month.January, 2031),
        Period(6, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )


def _overnight_schedule() -> Schedule:
    return Schedule.from_rule(
        Date.from_ymd(22, Month.January, 2024),
        Date.from_ymd(22, Month.January, 2026),
        Period(3, TimeUnit.Months),
        UnitedStates(UnitedStates.Market.SOFR),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )


def _euribor6m() -> IborIndex:
    return Euribor(Period(6, TimeUnit.Months), _flat(_FORWARD_RATE))


def _sofr() -> IborIndex:
    return Sofr(_flat(_FORWARD_RATE))


class _Case:
    """One probe case: the constructor arguments swept for it."""

    def __init__(
        self,
        *,
        pay_bond_coupon: bool = True,
        float_schedule: Schedule | None = None,
        floating_day_count: DayCounter | None = None,
        par_asset_swap: bool = True,
        gearing: float = 1.0,
        non_par_repayment: float | None = None,
        deal_maturity: Date | None = None,
        overnight: bool = False,
    ) -> None:
        self.pay_bond_coupon = pay_bond_coupon
        self.float_schedule = float_schedule
        self.floating_day_count = floating_day_count
        self.par_asset_swap = par_asset_swap
        self.gearing = gearing
        self.non_par_repayment = non_par_repayment
        self.deal_maturity = deal_maturity
        self.overnight = overnight

    def build(self) -> AssetSwap:
        index = _sofr() if self.overnight else _euribor6m()
        swap = AssetSwap(
            self.pay_bond_coupon,
            _bond(),
            _CLEAN_PRICE,
            index,
            _SPREAD,
            self.float_schedule,
            self.floating_day_count,
            self.par_asset_swap,
            self.gearing,
            self.non_par_repayment,
            self.deal_maturity,
        )
        swap.set_pricing_engine(DiscountingSwapEngine(_flat(_DISCOUNT_RATE)))
        return swap


_CASES: dict[str, _Case] = {
    "base_pay_bond_coupon": _Case(),
    "base_receive_bond_coupon": _Case(pay_bond_coupon=False),
    "float_schedule_3m": _Case(float_schedule=_float_schedule_3m()),
    "float_daycount_30360": _Case(floating_day_count=Thirty360(Thirty360Convention.BondBasis)),
    "market_asset_swap": _Case(par_asset_swap=False),
    "gearing_0_9": _Case(gearing=0.9),
    "non_par_repayment_102": _Case(non_par_repayment=102.0),
    "deal_maturity_2027": _Case(deal_maturity=Date.from_ymd(15, Month.July, 2027)),
    "overnight_sofr": _Case(float_schedule=_overnight_schedule(), overnight=True),
    "overnight_market_gearing": _Case(
        pay_bond_coupon=False,
        float_schedule=_overnight_schedule(),
        floating_day_count=Actual365Fixed(),
        par_asset_swap=False,
        gearing=1.1,
        overnight=True,
    ),
}


def _check_leg(leg: list[CashFlow], ref: list[dict[str, Any]], what: str) -> None:
    assert len(leg) == len(ref), f"{what}: flow count"
    for i, (cf, r) in enumerate(zip(leg, ref, strict=True)):
        assert cf.date().serial_number() == r["date_serial"], f"{what}[{i}]: payment date"
        tight(cf.amount(), r["amount"], reason=f"{what}[{i}] amount")
        assert isinstance(cf, Coupon) == r["is_coupon"], f"{what}[{i}]: coupon-ness"
        if not r["is_coupon"]:
            continue
        assert isinstance(cf, Coupon)
        tight(cf.nominal(), r["nominal"], reason=f"{what}[{i}] nominal")
        assert cf.accrual_start_date().serial_number() == r["accrual_start_serial"]
        assert cf.accrual_end_date().serial_number() == r["accrual_end_serial"]
        tight(cf.accrual_period(), r["accrual_period"], reason=f"{what}[{i}] accrual period")
        tight(cf.rate(), r["rate"], reason=f"{what}[{i}] rate")


@pytest.mark.parametrize("key", list(_CASES))
def test_asset_swap_case(cpp: dict[str, Any], key: str) -> None:
    ref = cpp["asset_swap"][key]
    swap = _CASES[key].build()

    tight(swap.npv(), ref["npv"], reason=f"{key} NPV")
    tight(swap.fair_clean_price(), ref["fair_clean_price"], reason=f"{key} fair clean price")
    tight(
        swap.fair_non_par_repayment(),
        ref["fair_non_par_repayment"],
        reason=f"{key} fair non-par repayment",
    )
    tight(swap.fair_spread(), ref["fair_spread"], reason=f"{key} fair spread")
    tight(swap.floating_leg_bps(), ref["floating_leg_bps"], reason=f"{key} floating-leg BPS")
    tight(swap.floating_leg_npv(), ref["floating_leg_npv"], reason=f"{key} floating-leg NPV")

    assert swap.par_swap() == ref["par_swap"]
    tight(swap.spread(), ref["spread"])
    tight(swap.clean_price(), ref["clean_price"])
    tight(swap.non_par_repayment(), ref["non_par_repayment"])
    assert swap.pay_bond_coupon() == ref["pay_bond_coupon"]
    assert swap.start_date().serial_number() == ref["start_date_serial"]
    assert swap.maturity_date().serial_number() == ref["maturity_date_serial"]

    _check_leg(swap.bond_leg(), ref["bond_leg"], f"{key}.bond_leg")
    _check_leg(swap.floating_leg(), ref["floating_leg"], f"{key}.floating_leg")


def test_optional_arguments_change_the_result(cpp: dict[str, Any]) -> None:
    """Every optional argument must move a number relative to the base case.

    This is the direct guard against the "argument accepted and dropped"
    failure mode: if any of these compared equal, the argument would have been
    silently ignored.
    """
    base = cpp["asset_swap"]["base_pay_bond_coupon"]
    for key in (
        "float_schedule_3m",
        "float_daycount_30360",
        "market_asset_swap",
        "gearing_0_9",
        "non_par_repayment_102",
        "deal_maturity_2027",
        "overnight_sofr",
    ):
        assert cpp["asset_swap"][key]["npv"] != base["npv"], key
        assert _CASES[key].build().npv() != _CASES["base_pay_bond_coupon"].build().npv(), key


def test_non_par_repayment_is_stored(cpp: dict[str, Any]) -> None:
    """``non_par_repayment=None`` defaults to 100; a supplied value is kept."""
    assert cpp["asset_swap"]["base_pay_bond_coupon"]["non_par_repayment"] == 100.0
    assert cpp["asset_swap"]["non_par_repayment_102"]["non_par_repayment"] == 102.0
    assert _CASES["base_pay_bond_coupon"].build().non_par_repayment() == 100.0
    assert _CASES["non_par_repayment_102"].build().non_par_repayment() == 102.0


def test_overnight_index_needs_a_float_schedule(cpp: dict[str, Any]) -> None:
    assert cpp["asset_swap_raises"]["overnight_without_schedule"]["raises"] is True
    with pytest.raises(LibraryException, match="floating schedule is needed"):
        AssetSwap(True, _bond(), _CLEAN_PRICE, _sofr(), _SPREAD)


def test_deal_maturity_after_schedule_back_raises(cpp: dict[str, Any]) -> None:
    assert cpp["asset_swap_raises"]["deal_maturity_after_schedule_back"]["raises"] is True
    with pytest.raises(LibraryException, match="cannot be later than"):
        AssetSwap(
            True,
            _bond(),
            _CLEAN_PRICE,
            _euribor6m(),
            _SPREAD,
            _float_schedule_6m(),
            None,
            True,
            1.0,
            None,
            Date.from_ymd(15, Month.January, 2035),
        )


def test_deal_maturity_before_schedule_front_raises(cpp: dict[str, Any]) -> None:
    assert cpp["asset_swap_raises"]["deal_maturity_before_schedule_front"]["raises"] is True
    with pytest.raises(LibraryException, match="must be later than"):
        AssetSwap(
            True,
            _bond(),
            _CLEAN_PRICE,
            _euribor6m(),
            _SPREAD,
            _float_schedule_6m(),
            None,
            True,
            1.0,
            None,
            Date.from_ymd(22, Month.January, 2024),
        )
