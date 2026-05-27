"""Tests for the Option abstract + OptionArguments + Greeks/MoreGreeks results."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.option import Greeks, MoreGreeks, Option, OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _payoff() -> PlainVanillaPayoff:
    return PlainVanillaPayoff(OptionType.Call, 100.0)


def _exercise() -> EuropeanExercise:
    return EuropeanExercise(Date.from_ymd(15, Month.June, 2027))


def test_cannot_instantiate_option_directly() -> None:
    with pytest.raises(TypeError):
        Option(_payoff(), _exercise())  # type: ignore[abstract]


def test_option_arguments_validates_payoff_required() -> None:
    args = OptionArguments()
    args.exercise = _exercise()
    with pytest.raises(LibraryException, match="no payoff given"):
        args.validate()


def test_option_arguments_validates_exercise_required() -> None:
    args = OptionArguments()
    args.payoff = _payoff()
    with pytest.raises(LibraryException, match="no exercise given"):
        args.validate()


def test_option_arguments_validates_when_complete() -> None:
    args = OptionArguments()
    args.payoff = _payoff()
    args.exercise = _exercise()
    # No exception.
    args.validate()


# --- Greeks --------------------------------------------------------------


def test_greeks_initialized_with_none_fields() -> None:
    g = Greeks()
    assert g.delta is None
    assert g.gamma is None
    assert g.theta is None
    assert g.vega is None
    assert g.rho is None
    assert g.dividend_rho is None


def test_greeks_reset_clears_all_fields() -> None:
    g = Greeks()
    g.value = 42.0
    g.delta = 0.5
    g.gamma = 0.02
    g.reset()
    assert g.value is None
    assert g.delta is None
    assert g.gamma is None


# --- MoreGreeks ----------------------------------------------------------


def test_more_greeks_initialized_with_none_fields() -> None:
    mg = MoreGreeks()
    assert mg.itm_cash_probability is None
    assert mg.delta_forward is None
    assert mg.elasticity is None
    assert mg.theta_per_day is None
    assert mg.strike_sensitivity is None


def test_more_greeks_reset_clears_all_fields() -> None:
    mg = MoreGreeks()
    mg.delta_forward = 0.4
    mg.elasticity = 0.8
    mg.reset()
    assert mg.delta_forward is None
    assert mg.elasticity is None
