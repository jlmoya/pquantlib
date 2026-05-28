"""Tests for BarrierOption + BarrierType.

# C++ parity: ql/instruments/barrieroption.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.barrier_option import (
    BarrierOption,
    BarrierOptionArguments,
    BarrierType,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _expiry() -> Date:
    return Date.from_ymd(15, Month.June, 2027)


def test_barrier_type_int_values_match_cpp() -> None:
    """C++ ``Barrier::Type``: DownIn=0, UpIn=1, DownOut=2, UpOut=3."""
    assert int(BarrierType.DownIn) == 0
    assert int(BarrierType.UpIn) == 1
    assert int(BarrierType.DownOut) == 2
    assert int(BarrierType.UpOut) == 3


def test_barrier_option_holds_fields() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = BarrierOption(BarrierType.DownOut, 95.0, 3.0, payoff, exercise)
    assert opt.barrier_type() == BarrierType.DownOut
    assert opt.barrier() == 95.0
    assert opt.rebate() == 3.0


def test_barrier_option_setup_arguments_populates_fields() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = BarrierOption(BarrierType.DownOut, 95.0, 3.0, payoff, exercise)
    args = BarrierOptionArguments()
    opt.setup_arguments(args)
    assert args.barrier_type == BarrierType.DownOut
    assert args.barrier == 95.0
    assert args.rebate == 3.0
    assert args.payoff is payoff
    assert args.exercise is exercise


def test_barrier_arguments_validate_rejects_zero_barrier() -> None:
    args = BarrierOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(_expiry())
    args.barrier_type = BarrierType.DownOut
    args.barrier = 0.0
    args.rebate = 0.0
    with pytest.raises(LibraryException, match=r"barrier .* must be positive"):
        args.validate()


def test_barrier_arguments_validate_rejects_negative_rebate() -> None:
    args = BarrierOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(_expiry())
    args.barrier_type = BarrierType.DownOut
    args.barrier = 95.0
    args.rebate = -1.0
    with pytest.raises(LibraryException, match="negative rebate"):
        args.validate()
