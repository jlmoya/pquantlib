"""Tests for ContinuousFloatingLookbackOption + ContinuousFixedLookbackOption.

# C++ parity: ql/instruments/lookbackoption.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.lookback_option import (
    ContinuousFixedLookbackOption,
    ContinuousFixedLookbackOptionArguments,
    ContinuousFloatingLookbackOption,
    ContinuousFloatingLookbackOptionArguments,
)
from pquantlib.payoffs import FloatingTypePayoff, OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _expiry() -> Date:
    return Date.from_ymd(15, Month.June, 2027)


# --- ContinuousFloatingLookbackOption --------------------------------------


def test_floating_lookback_holds_minmax() -> None:
    payoff = FloatingTypePayoff(OptionType.Call)
    ex = EuropeanExercise(_expiry())
    opt = ContinuousFloatingLookbackOption(100.0, payoff, ex)
    assert opt.minmax() == 100.0


def test_floating_lookback_setup_arguments_populates_minmax() -> None:
    payoff = FloatingTypePayoff(OptionType.Call)
    ex = EuropeanExercise(_expiry())
    opt = ContinuousFloatingLookbackOption(100.0, payoff, ex)
    args = ContinuousFloatingLookbackOptionArguments()
    opt.setup_arguments(args)
    assert args.minmax == 100.0
    assert args.payoff is payoff


def test_floating_lookback_arguments_validate_rejects_negative_minmax() -> None:
    args = ContinuousFloatingLookbackOptionArguments()
    args.payoff = FloatingTypePayoff(OptionType.Call)
    args.exercise = EuropeanExercise(_expiry())
    args.minmax = -1.0
    with pytest.raises(LibraryException, match="nonnegative prior extremum"):
        args.validate()


def test_floating_lookback_arguments_validate_rejects_null_minmax() -> None:
    args = ContinuousFloatingLookbackOptionArguments()
    args.payoff = FloatingTypePayoff(OptionType.Call)
    args.exercise = EuropeanExercise(_expiry())
    args.minmax = None
    with pytest.raises(LibraryException, match="null prior extremum"):
        args.validate()


# --- ContinuousFixedLookbackOption -----------------------------------------


def test_fixed_lookback_holds_minmax_and_strike() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    ex = EuropeanExercise(_expiry())
    opt = ContinuousFixedLookbackOption(95.0, payoff, ex)
    assert opt.minmax() == 95.0


def test_fixed_lookback_setup_arguments_populates_minmax() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    ex = EuropeanExercise(_expiry())
    opt = ContinuousFixedLookbackOption(95.0, payoff, ex)
    args = ContinuousFixedLookbackOptionArguments()
    opt.setup_arguments(args)
    assert args.minmax == 95.0
    assert args.payoff is payoff
