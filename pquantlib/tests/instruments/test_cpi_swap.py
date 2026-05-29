"""CPISwap smoke + structural tests.

CPISwap is a complex multi-leg instrument that depends on L7-C's CPILeg +
IborLeg builders for inflation. Until those land, the constructor takes
pre-built cashflow legs. These tests verify the structural shell.
"""

from __future__ import annotations

from typing import cast

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.uk_rpi import UKRPI
from pquantlib.instruments.cpi_swap import CPISwap
from pquantlib.instruments.swap import SwapType
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
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


def _build_swap(type_: SwapType = SwapType.Payer) -> CPISwap:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    cal = TARGET()
    start = cal.advance(eval_date, 2, TimeUnit.Days)
    end = cal.advance(start, 5, TimeUnit.Years)
    fixed_schedule = Schedule.from_rule(
        start, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    float_schedule = Schedule.from_rule(
        start, end, Period(3, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )

    forwarding_curve = FlatForward.from_rate(
        eval_date, 0.03, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    float_index = cast(
        IborIndexProtocol,
        Euribor(Period(3, TimeUnit.Months), forwarding_curve),
    )

    # Pre-build minimal legs (L7-D scaffold seam — L7-C will replace).
    cpi_leg: list[CashFlow] = [SimpleCashFlow(1_000.0, end)]
    float_leg: list[CashFlow] = [SimpleCashFlow(500.0, end)]

    return CPISwap(
        type_=type_,
        nominal=1_000_000.0,
        subtract_inflation_nominal=False,
        spread=0.0,
        float_day_count=Actual360(),
        float_schedule=float_schedule,
        float_payment_roll=BusinessDayConvention.ModifiedFollowing,
        fixing_days=2,
        float_index=float_index,
        fixed_rate=0.025,
        base_cpi=100.0,
        fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
        fixed_schedule=fixed_schedule,
        fixed_payment_roll=BusinessDayConvention.ModifiedFollowing,
        observation_lag=Period(3, TimeUnit.Months),
        fixed_index=UKRPI(),
        observation_interpolation=InterpolationType.AsIndex,
        cpi_leg=cpi_leg,
        float_leg=float_leg,
    )


def test_cpi_swap_has_two_legs() -> None:
    swap = _build_swap()
    assert swap.number_of_legs() == 2
    assert len(swap.cpi_leg()) == 1
    assert len(swap.float_leg()) == 1


def test_cpi_swap_payer_inverts_payment_signs() -> None:
    payer = _build_swap(SwapType.Payer)
    receiver = _build_swap(SwapType.Receiver)
    # Payer = pays floating (per CPISwap convention); legs[1] = float.
    # C++ semantics: payer means leg 0 (cpi) = +1.0, leg 1 (float) = -1.0.
    assert payer.payer(0) is False  # cpi leg is received (sign +1)
    assert payer.payer(1) is True  # float leg is paid (sign -1)
    assert receiver.payer(0) is True
    assert receiver.payer(1) is False


def test_cpi_swap_inspectors() -> None:
    swap = _build_swap()
    assert swap.type() == SwapType.Payer
    assert swap.nominal() == 1_000_000.0
    assert swap.fixed_rate() == 0.025
    assert swap.base_cpi() == 100.0
    assert swap.observation_lag() == Period(3, TimeUnit.Months)
    assert swap.observation_interpolation() == InterpolationType.AsIndex
    assert swap.inflation_nominal() == 1_000_000.0


def test_cpi_swap_requires_pre_built_legs_until_l7c() -> None:
    """The L7-D scaffold seam requires explicit cpi_leg + float_leg."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    cal = TARGET()
    start = cal.advance(eval_date, 2, TimeUnit.Days)
    end = cal.advance(start, 5, TimeUnit.Years)
    fixed_schedule = Schedule.from_rule(
        start, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    float_schedule = fixed_schedule  # reuse for brevity
    forwarding_curve = FlatForward.from_rate(
        eval_date, 0.03, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    float_index = cast(
        IborIndexProtocol,
        Euribor(Period(3, TimeUnit.Months), forwarding_curve),
    )
    with pytest.raises(Exception, match="L7-D CPISwap requires pre-built"):
        CPISwap(
            type_=SwapType.Payer,
            nominal=1.0,
            subtract_inflation_nominal=False,
            spread=0.0,
            float_day_count=Actual360(),
            float_schedule=float_schedule,
            float_payment_roll=BusinessDayConvention.ModifiedFollowing,
            fixing_days=2,
            float_index=float_index,
            fixed_rate=0.025,
            base_cpi=100.0,
            fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
            fixed_schedule=fixed_schedule,
            fixed_payment_roll=BusinessDayConvention.ModifiedFollowing,
            observation_lag=Period(3, TimeUnit.Months),
            fixed_index=UKRPI(),
        )
