"""Swaption smoke tests.

These tests validate that ``Swaption`` can be constructed around a
``VanillaSwap`` + ``EuropeanExercise``, that the settlement enums are
exposed correctly, and that ``setup_arguments`` populates a
``SwaptionArguments`` carrier with the right fields. Engine-backed
NPV cross-validation lives in the engine tests (test_black_swaption_engine.py,
test_jamshidian_swaption_engine.py, etc.).

Reference values for the underlying setup live at
``migration-harness/references/cluster/l4e.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exercise import EuropeanExercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    Swaption,
    SwaptionArguments,
    check_settlement_type_and_method_consistency,
)
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
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

_REF_PATH = (
    Path(__file__).resolve().parents[3] / "migration-harness/references/cluster/l4e.json"
)


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


def _five_by_ten_receiver_swaption() -> Swaption:
    """Build the 5y10y receiver swaption used by the C++ probe."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    cal = TARGET()
    settle = cal.advance(eval_date, 5, TimeUnit.Years)
    end = cal.advance(settle, 10, TimeUnit.Years)
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    fixed_sched = Schedule.from_rule(
        settle, end, Period(6, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    float_sched = Schedule.from_rule(
        settle, end, Period(3, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    swap = VanillaSwap(
        SwapType.Receiver, 1_000_000.0,
        fixed_sched, 0.03, Thirty360(Convention.BondBasis),
        float_sched, idx, 0.0, idx.day_counter(),
    )
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    exercise = EuropeanExercise(settle)
    return Swaption(swap, exercise)


# --- Construction + accessors ---------------------------------------------


def test_swaption_constructs_with_defaults() -> None:
    swaption = _five_by_ten_receiver_swaption()
    assert swaption.settlement_type == SettlementType.Physical
    assert swaption.settlement_method == SettlementMethod.PhysicalOTC


def test_swaption_underlying_type_matches_swap() -> None:
    """Swaption.type() returns the underlying VanillaSwap's payer/receiver side."""
    swaption = _five_by_ten_receiver_swaption()
    assert swaption.type() == SwapType.Receiver


def test_swaption_underlying_swap_round_trip() -> None:
    swaption = _five_by_ten_receiver_swaption()
    swap = swaption.underlying_swap()
    assert swap.fixed_rate() == 0.03
    assert swap.nominal() == 1_000_000.0


def test_swaption_is_not_expired_by_default() -> None:
    """Mirrors VanillaOption: pquantlib defers is_expired to False because
    Settings.evaluation_date is a deferred wiring (see swaption.py docstring)."""
    swaption = _five_by_ten_receiver_swaption()
    assert swaption.is_expired() is False


# --- setup_arguments fills the swap + swaption fields ---------------------


def test_swaption_setup_arguments_populates_carrier(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    swaption = _five_by_ten_receiver_swaption()
    args = SwaptionArguments()
    swaption.setup_arguments(args)

    expected_setup = cluster_refs["setup"]
    assert args.swap is swaption.underlying_swap()
    assert args.exercise is swaption.exercise()
    assert args.settlement_type == SettlementType.Physical
    assert args.settlement_method == SettlementMethod.PhysicalOTC
    # Per probe: fixed leg has 20 periods, float leg has 40 (Backward
    # generation collects extra short-stubs).
    assert len(args.legs[0]) == expected_setup["fixed_leg_size"]
    assert len(args.legs[1]) == expected_setup["floating_leg_size"]
    # payoff is nulled out — engines should never read it.
    assert args.payoff is None


def test_swaption_arguments_validate_requires_swap() -> None:
    args = SwaptionArguments()
    args.exercise = EuropeanExercise(Date.from_ymd(17, Month.January, 2030))
    args.legs = [[], []]
    args.payer = [-1.0, 1.0]
    with pytest.raises(Exception, match="swap"):
        args.validate()


# --- Settlement consistency -----------------------------------------------


def test_settlement_consistency_physical_otc() -> None:
    """(Physical, PhysicalOTC) is allowed."""
    check_settlement_type_and_method_consistency(
        SettlementType.Physical, SettlementMethod.PhysicalOTC
    )


def test_settlement_consistency_cash_par_yield() -> None:
    """(Cash, ParYieldCurve) is allowed."""
    check_settlement_type_and_method_consistency(
        SettlementType.Cash, SettlementMethod.ParYieldCurve
    )


def test_settlement_consistency_physical_with_cash_method_rejected() -> None:
    with pytest.raises(Exception, match="invalid settlement method for physical"):
        check_settlement_type_and_method_consistency(
            SettlementType.Physical, SettlementMethod.CollateralizedCashPrice
        )


def test_settlement_consistency_cash_with_physical_method_rejected() -> None:
    with pytest.raises(Exception, match="invalid settlement method for cash"):
        check_settlement_type_and_method_consistency(
            SettlementType.Cash, SettlementMethod.PhysicalOTC
        )
