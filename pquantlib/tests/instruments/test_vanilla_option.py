"""Tests for VanillaOption + EuropeanOption.

The pricing-engine integration is exercised in
``pquantlib/tests/pricingengines/vanilla/`` once the engines land.
This file checks the type-discrimination + non-expired default.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# --- VanillaOption ---------------------------------------------------------


def test_vanilla_option_is_a_one_asset_option() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(Date.from_ymd(15, Month.June, 2027))
    opt = VanillaOption(payoff, exercise)
    assert isinstance(opt, OneAssetOption)


def test_vanilla_option_accepts_american_exercise() -> None:
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    earliest = Date.from_ymd(15, Month.June, 2026)
    latest = Date.from_ymd(15, Month.June, 2027)
    exercise = AmericanExercise(earliest, latest)
    opt = VanillaOption(payoff, exercise)
    assert opt.exercise() is exercise


def test_vanilla_option_not_expired_by_default() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(Date.from_ymd(15, Month.June, 2027))
    opt = VanillaOption(payoff, exercise)
    assert opt.is_expired() is False


def test_vanilla_option_payoff_and_exercise() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(Date.from_ymd(15, Month.June, 2027))
    opt = VanillaOption(payoff, exercise)
    assert opt.payoff() is payoff
    assert opt.exercise() is exercise


# --- EuropeanOption --------------------------------------------------------


def test_european_option_is_a_vanilla_option() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(Date.from_ymd(15, Month.June, 2027))
    opt = EuropeanOption(payoff, exercise)
    assert isinstance(opt, VanillaOption)


def test_european_option_rejects_american_exercise() -> None:
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    earliest = Date.from_ymd(15, Month.June, 2026)
    latest = Date.from_ymd(15, Month.June, 2027)
    exercise = AmericanExercise(earliest, latest)
    with pytest.raises(LibraryException, match="EuropeanOption requires European exercise"):
        EuropeanOption(payoff, exercise)


def test_european_option_inspectors() -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(Date.from_ymd(15, Month.June, 2027))
    opt = EuropeanOption(payoff, exercise)
    assert opt.payoff() is payoff
    assert opt.exercise() is exercise
