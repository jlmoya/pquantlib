"""Tests for ``DiscreteAveragingAsianOption`` instrument.

# C++ parity: ql/instruments/asianoption.{hpp,cpp} (v1.42.1).
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import (
    DiscreteAveragingAsianOption,
    DiscreteAveragingAsianOptionArguments,
)
from pquantlib.instruments.average_type import AverageType
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _fixtures() -> tuple[list[Date], PlainVanillaPayoff, EuropeanExercise]:
    fixings = [Date.from_ymd(15, Month.June, 2026) + i * 30 for i in range(1, 13)]
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(fixings[-1])
    return fixings, payoff, exercise


def test_construct_arithmetic_unseasoned_overrides_accumulator_to_zero() -> None:
    fixings, payoff, exercise = _fixtures()
    # Pass a positive accumulator with past_fixings=0; should be forced to 0.
    opt = DiscreteAveragingAsianOption(
        AverageType.Arithmetic, 999.0, 0, fixings, payoff, exercise
    )
    assert opt.running_accumulator() == 0.0


def test_construct_geometric_unseasoned_overrides_accumulator_to_one() -> None:
    fixings, payoff, exercise = _fixtures()
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 999.0, 0, fixings, payoff, exercise
    )
    assert opt.running_accumulator() == 1.0


def test_construct_seasoned_keeps_accumulator() -> None:
    fixings, payoff, exercise = _fixtures()
    opt = DiscreteAveragingAsianOption(
        AverageType.Arithmetic, 50.0, 3, fixings, payoff, exercise
    )
    assert opt.running_accumulator() == 50.0
    assert opt.past_fixings() == 3


def test_fixing_dates_are_sorted() -> None:
    _, payoff, _exercise = _fixtures()
    unsorted = [Date.from_ymd(15, Month.October, 2026), Date.from_ymd(15, Month.July, 2026)]
    opt = DiscreteAveragingAsianOption(
        AverageType.Arithmetic, 0.0, 0, unsorted, payoff, EuropeanExercise(unsorted[0])
    )
    fds = opt.fixing_dates()
    assert fds[0] <= fds[1]


def test_setup_arguments_populates_extra_fields() -> None:
    fixings, payoff, exercise = _fixtures()
    opt = DiscreteAveragingAsianOption(
        AverageType.Arithmetic, 0.0, 0, fixings, payoff, exercise
    )
    args = DiscreteAveragingAsianOptionArguments()
    opt.setup_arguments(args)
    assert args.average_type == AverageType.Arithmetic
    assert args.running_accumulator == 0.0
    assert args.past_fixings == 0
    assert args.payoff is payoff
    assert args.exercise is exercise
    assert len(args.fixing_dates) == 12


def test_arguments_validate_rejects_negative_arithmetic_sum() -> None:
    args = DiscreteAveragingAsianOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(Date.from_ymd(15, Month.July, 2026))
    args.average_type = AverageType.Arithmetic
    args.running_accumulator = -1.0
    args.past_fixings = 1
    with pytest.raises(LibraryException, match="non negative running sum"):
        args.validate()


def test_arguments_validate_rejects_nonpositive_geometric_product() -> None:
    args = DiscreteAveragingAsianOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(Date.from_ymd(15, Month.July, 2026))
    args.average_type = AverageType.Geometric
    args.running_accumulator = 0.0
    args.past_fixings = 1
    with pytest.raises(LibraryException, match="positive running product"):
        args.validate()
