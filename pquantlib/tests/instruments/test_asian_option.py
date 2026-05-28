"""Tests for AsianOption + ContinuousAveragingAsianOption +
DiscreteAveragingAsianOption.

# C++ parity: ql/instruments/asianoption.{hpp,cpp} @ v1.42.1.

The instrument layer carries arguments and forwards them to the
engine; cross-validation against C++ analytic values is exercised by
the engine tests (``test_analytic_continuous_geometric_average_price_engine.py``,
``test_analytic_discrete_geometric_average_price_engine.py``).
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOption,
    ContinuousAveragingAsianOptionArguments,
    DiscreteAveragingAsianOption,
    DiscreteAveragingAsianOptionArguments,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _expiry() -> Date:
    return Date.from_ymd(15, Month.June, 2027)


def test_average_type_int_values_match_cpp() -> None:
    """C++ ``Average::Type``: Arithmetic=0, Geometric=1."""
    assert int(AverageType.Arithmetic) == 0
    assert int(AverageType.Geometric) == 1


# --- ContinuousAveragingAsianOption ----------------------------------------


def test_continuous_asian_holds_average_type() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    assert opt.average_type() == AverageType.Geometric


def test_continuous_asian_unseasoned_has_default_start_date() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    # Default Date() == minimum (= 1/Jan/1901).
    assert opt.start_date() == Date()


def test_continuous_asian_seasoned_preserves_start_date() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    start = Date.from_ymd(1, Month.January, 2026)
    opt = ContinuousAveragingAsianOption(
        AverageType.Geometric, payoff, exercise, start_date=start
    )
    assert opt.start_date() == start


def test_continuous_asian_setup_arguments_populates_fields() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    args = ContinuousAveragingAsianOptionArguments()
    opt.setup_arguments(args)
    assert args.average_type == AverageType.Geometric
    assert args.payoff is payoff
    assert args.exercise is exercise


def test_continuous_asian_arguments_validate_rejects_missing_type() -> None:
    args = ContinuousAveragingAsianOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(_expiry())
    args.average_type = None
    with pytest.raises(LibraryException, match="no average type"):
        args.validate()


# --- DiscreteAveragingAsianOption ------------------------------------------


def test_discrete_asian_holds_running_accumulator_and_past_fixings() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    fixings = [Date.from_ymd(15, Month.December, 2026), _expiry()]
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
    )
    assert opt.average_type() == AverageType.Geometric
    assert opt.running_accumulator() == 1.0
    assert opt.past_fixings() == 0
    assert len(opt.fixing_dates()) == 2


def test_discrete_asian_setup_arguments_populates_fields() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(_expiry())
    fixings = [Date.from_ymd(15, Month.December, 2026), _expiry()]
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
    )
    args = DiscreteAveragingAsianOptionArguments()
    opt.setup_arguments(args)
    assert args.average_type == AverageType.Geometric
    assert args.running_accumulator == 1.0
    assert args.past_fixings == 0
    assert args.fixing_dates == fixings


def test_discrete_asian_arguments_validate_rejects_empty_fixings() -> None:
    args = DiscreteAveragingAsianOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(_expiry())
    args.average_type = AverageType.Geometric
    args.running_accumulator = 1.0
    args.past_fixings = 0
    args.fixing_dates = []
    with pytest.raises(LibraryException, match="no fixing dates"):
        args.validate()
