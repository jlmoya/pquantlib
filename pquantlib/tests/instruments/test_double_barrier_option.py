"""Tests for DoubleBarrierOption + DoubleBarrierType.

# C++ parity: ql/instruments/doublebarrieroption.{hpp,cpp} +
# ql/instruments/doublebarriertype.hpp @ v1.42.1.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _expiry() -> Date:
    return Date.from_ymd(15, Month.June, 2027)


def test_double_barrier_type_int_values_match_cpp() -> None:
    """C++ ``DoubleBarrier::Type``: KnockIn=0, KnockOut=1, KIKO=2, KOKI=3."""
    assert int(DoubleBarrierType.KnockIn) == 0
    assert int(DoubleBarrierType.KnockOut) == 1
    assert int(DoubleBarrierType.KIKO) == 2
    assert int(DoubleBarrierType.KOKI) == 3


def test_double_barrier_option_holds_fields() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockOut, 80.0, 120.0, 1.5, payoff, exercise
    )
    assert opt.barrier_type() == DoubleBarrierType.KnockOut
    assert opt.barrier_lo() == 80.0
    assert opt.barrier_hi() == 120.0
    assert opt.rebate() == 1.5


def test_double_barrier_option_is_not_expired() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = DoubleBarrierOption(
        DoubleBarrierType.KnockOut, 80.0, 120.0, 0.0, payoff, exercise
    )
    assert opt.is_expired() is False


def test_double_barrier_option_setup_arguments_populates_fields() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = DoubleBarrierOption(
        DoubleBarrierType.KIKO, 80.0, 120.0, 2.0, payoff, exercise
    )
    args = DoubleBarrierOptionArguments()
    opt.setup_arguments(args)
    assert args.barrier_type == DoubleBarrierType.KIKO
    assert args.barrier_lo == 80.0
    assert args.barrier_hi == 120.0
    assert args.rebate == 2.0
    assert args.payoff is payoff
    assert args.exercise is exercise


def test_double_barrier_arguments_validate_rejects_missing_lo() -> None:
    args = DoubleBarrierOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(_expiry())
    args.barrier_type = DoubleBarrierType.KnockOut
    # barrier_lo intentionally left as None
    args.barrier_hi = 120.0
    args.rebate = 0.0
    with pytest.raises(LibraryException, match="no low barrier"):
        args.validate()


def test_double_barrier_arguments_validate_rejects_missing_hi() -> None:
    args = DoubleBarrierOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(_expiry())
    args.barrier_type = DoubleBarrierType.KnockOut
    args.barrier_lo = 80.0
    # barrier_hi intentionally left as None
    args.rebate = 0.0
    with pytest.raises(LibraryException, match="no high barrier"):
        args.validate()


def test_double_barrier_arguments_validate_rejects_missing_rebate() -> None:
    args = DoubleBarrierOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(_expiry())
    args.barrier_type = DoubleBarrierType.KnockOut
    args.barrier_lo = 80.0
    args.barrier_hi = 120.0
    # rebate intentionally left as None
    with pytest.raises(LibraryException, match="no rebate"):
        args.validate()
