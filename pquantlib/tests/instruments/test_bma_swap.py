"""Cross-validate BMASwap against C++ QuantLib v1.43.

Probe: ``v143/inst/swaps`` (key ``bma_swap``).

Both sides of ``SwapType`` are pinned, together with the full cashflow listing
of both legs.  Note that C++ names the fair-spread accessor ``fairLiborSpread``
(bmaswap.cpp:129) — it is the spread on the *Libor* leg, so the port keeps that
name rather than the "BMA spread" one might expect from the leg ordering.

Non-default constructor arguments exercised:

- ``type_`` — ``Payer`` and ``Receiver`` (the sign flip is observable on
  every NPV/BPS).
- ``libor_fraction`` = 0.67 (a gearing != 1 must reach the Libor coupons,
  so every Libor coupon rate is checked against C++).
- ``libor_spread`` = 10bp (likewise; also drives ``fair_libor_fraction``).
- ``libor_day_count`` = Actual/360 vs ``bma_day_count`` = Actual/365F — two
  different day counters, so a swapped pair changes every accrual period.
- The two schedules carry different calendars and conventions, and each leg
  must pick up *its own* schedule's convention as the payment adjustment.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.bma_index import BMAIndex
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.bma_swap import BMASwap
from pquantlib.instruments.swap import SwapType
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

_EVAL = Date.from_ymd(17, Month.January, 2024)
_LIBOR_FORWARD_RATE = 0.035
_BMA_FORWARD_RATE = 0.025
_DISCOUNT_RATE = 0.030
_START = Date.from_ymd(1, Month.February, 2024)
_END = Date.from_ymd(1, Month.February, 2027)
_NOMINAL = 1.0e6
_LIBOR_FRACTION = 0.67
_LIBOR_SPREAD = 0.0010

_TYPES: dict[str, SwapType] = {"payer": SwapType.Payer, "receiver": SwapType.Receiver}


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _EVAL


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/swaps")


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(_EVAL, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)


def _libor_index() -> IborIndex:
    return Euribor(Period(3, TimeUnit.Months), _flat(_LIBOR_FORWARD_RATE))


def _bma_index() -> BMAIndex:
    return BMAIndex(_flat(_BMA_FORWARD_RATE))


def _libor_schedule() -> Schedule:
    return Schedule.from_rule(
        _START,
        _END,
        Period(3, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )


def _bma_schedule() -> Schedule:
    return Schedule.from_rule(
        _START,
        _END,
        Period(3, TimeUnit.Months),
        UnitedStates(UnitedStates.Market.GovernmentBond),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        False,
    )


def _build(type_: SwapType) -> BMASwap:
    swap = BMASwap(
        type_,
        _NOMINAL,
        _libor_schedule(),
        _LIBOR_FRACTION,
        _LIBOR_SPREAD,
        _libor_index(),
        Actual360(),
        _bma_schedule(),
        _bma_index(),
        Actual365Fixed(),
    )
    swap.set_pricing_engine(DiscountingSwapEngine(_flat(_DISCOUNT_RATE)))
    return swap


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


@pytest.mark.parametrize("key", list(_TYPES))
def test_bma_swap(cpp: dict[str, Any], key: str) -> None:
    ref = cpp["bma_swap"][key]
    swap = _build(_TYPES[key])

    tight(swap.npv(), ref["npv"], reason=f"{key} NPV")
    tight(
        swap.fair_libor_fraction(),
        ref["fair_libor_fraction"],
        reason=f"{key} fair libor fraction",
    )
    tight(swap.fair_libor_spread(), ref["fair_libor_spread"], reason=f"{key} fair libor spread")
    tight(swap.libor_leg_bps(), ref["libor_leg_bps"], reason=f"{key} libor-leg BPS")
    tight(swap.bma_leg_bps(), ref["bma_leg_bps"], reason=f"{key} BMA-leg BPS")
    tight(swap.libor_leg_npv(), ref["libor_leg_npv"], reason=f"{key} libor-leg NPV")
    tight(swap.bma_leg_npv(), ref["bma_leg_npv"], reason=f"{key} BMA-leg NPV")

    tight(swap.libor_fraction(), ref["libor_fraction"])
    tight(swap.libor_spread(), ref["libor_spread"])
    tight(swap.nominal(), ref["nominal"])
    assert int(swap.type()) == ref["type"]

    _check_leg(swap.libor_leg(), ref["libor_leg"], f"{key}.libor_leg")
    _check_leg(swap.bma_leg(), ref["bma_leg"], f"{key}.bma_leg")


def test_payer_and_receiver_are_mirrored(cpp: dict[str, Any]) -> None:
    """``Payer`` / ``Receiver`` refers to the BMA leg — every result flips sign."""
    payer = cpp["bma_swap"]["payer"]
    receiver = cpp["bma_swap"]["receiver"]
    for field in ("npv", "libor_leg_bps", "bma_leg_bps", "libor_leg_npv", "bma_leg_npv"):
        tight(payer[field], -receiver[field], reason=f"C++ {field} sign symmetry")
    tight(_build(SwapType.Payer).npv(), -_build(SwapType.Receiver).npv())


def test_libor_fraction_and_spread_reach_the_coupons(cpp: dict[str, Any]) -> None:
    """A gearing != 1 and a non-zero spread must land on the Libor coupons.

    The C++ reference already encodes ``rate = fraction * fixing + spread``
    coupon by coupon; here we assert the same identity holds in the port, so a
    dropped ``libor_fraction`` or ``libor_spread`` cannot pass.
    """
    swap = _build(SwapType.Payer)
    ref_rates = [r["rate"] for r in cpp["bma_swap"]["payer"]["libor_leg"]]
    assert len(ref_rates) > 0
    for cf, ref_rate in zip(swap.libor_leg(), ref_rates, strict=True):
        assert isinstance(cf, Coupon)
        tight(cf.rate(), ref_rate)
        # Ungeared, unspread the reference rate: the residual is the raw
        # index fixing, which must be far from the geared value.
        raw = (ref_rate - _LIBOR_SPREAD) / _LIBOR_FRACTION
        assert abs(raw - ref_rate) > 1e-3
