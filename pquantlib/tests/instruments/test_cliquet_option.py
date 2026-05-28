"""Tests for CliquetOption.

# C++ parity: ql/instruments/cliquetoption.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.cliquet_option import (
    CliquetOption,
    CliquetOptionArguments,
)
from pquantlib.payoffs import OptionType, PercentageStrikePayoff, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _maturity() -> EuropeanExercise:
    return EuropeanExercise(Date.from_ymd(15, Month.June, 2027))


def _reset_dates() -> list[Date]:
    return [
        Date.from_ymd(1, Month.January, 2027),
        Date.from_ymd(1, Month.April, 2027),
    ]


def test_cliquet_holds_reset_dates() -> None:
    payoff = PercentageStrikePayoff(OptionType.Call, 0.90)
    cl = CliquetOption(payoff, _maturity(), _reset_dates())
    assert cl.reset_dates() == _reset_dates()


def test_cliquet_setup_arguments_populates_reset_dates() -> None:
    payoff = PercentageStrikePayoff(OptionType.Call, 0.90)
    cl = CliquetOption(payoff, _maturity(), _reset_dates())
    args = CliquetOptionArguments()
    cl.setup_arguments(args)
    assert args.reset_dates == _reset_dates()


def test_cliquet_arguments_validate_passes_with_valid_setup() -> None:
    args = CliquetOptionArguments()
    args.payoff = PercentageStrikePayoff(OptionType.Call, 0.90)
    args.exercise = _maturity()
    args.reset_dates = _reset_dates()
    args.validate()  # no raise


def test_cliquet_arguments_validate_rejects_wrong_payoff_type() -> None:
    args = CliquetOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)  # wrong type
    args.exercise = _maturity()
    args.reset_dates = _reset_dates()
    with pytest.raises(LibraryException, match="wrong payoff type"):
        args.validate()


def test_cliquet_arguments_validate_rejects_zero_moneyness() -> None:
    args = CliquetOptionArguments()
    args.payoff = PercentageStrikePayoff(OptionType.Call, 0.0)
    args.exercise = _maturity()
    args.reset_dates = _reset_dates()
    with pytest.raises(LibraryException, match="negative or zero moneyness"):
        args.validate()


def test_cliquet_arguments_validate_rejects_empty_reset_dates() -> None:
    args = CliquetOptionArguments()
    args.payoff = PercentageStrikePayoff(OptionType.Call, 0.90)
    args.exercise = _maturity()
    args.reset_dates = []
    with pytest.raises(LibraryException, match="no reset dates"):
        args.validate()


def test_cliquet_arguments_validate_rejects_reset_after_maturity() -> None:
    args = CliquetOptionArguments()
    args.payoff = PercentageStrikePayoff(OptionType.Call, 0.90)
    args.exercise = _maturity()
    args.reset_dates = [Date.from_ymd(15, Month.December, 2028)]  # past maturity
    with pytest.raises(LibraryException, match="reset date greater"):
        args.validate()


def test_cliquet_arguments_validate_rejects_unsorted_reset_dates() -> None:
    args = CliquetOptionArguments()
    args.payoff = PercentageStrikePayoff(OptionType.Call, 0.90)
    args.exercise = _maturity()
    args.reset_dates = [
        Date.from_ymd(1, Month.April, 2027),
        Date.from_ymd(1, Month.January, 2027),  # earlier — bad
    ]
    with pytest.raises(LibraryException, match="unsorted reset dates"):
        args.validate()
