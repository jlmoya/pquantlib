"""Tests for the four continuous lookback option instruments.

# C++ parity: ql/instruments/lookbackoption.{hpp,cpp} @ v1.43.

Structural coverage only — argument plumbing, inspectors and every
``validate()`` branch. The priced cross-validation of the two partial-time
variants lives with their engines, in
``tests/pricingengines/lookback/test_analytic_continuous_partial_lookback_engines.py``.
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
    ContinuousPartialFixedLookbackOption,
    ContinuousPartialFixedLookbackOptionArguments,
    ContinuousPartialFloatingLookbackOption,
    ContinuousPartialFloatingLookbackOptionArguments,
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


# --- ContinuousPartialFloatingLookbackOption -------------------------------
#
# The two extra arguments (``lambda_`` and ``lookback_period_end``) are the
# whole difference from the full-window variant, so each is set to a
# NON-DEFAULT value below and asserted to reach both the inspector and the
# engine argument carrier.


def _lookback_end() -> Date:
    return Date.from_ymd(15, Month.December, 2026)


def _partial_floating(
    option_type: OptionType = OptionType.Call,
    minmax: float = 100.0,
    lambda_: float = 1.4,
    lookback_end: Date | None = None,
) -> ContinuousPartialFloatingLookbackOption:
    return ContinuousPartialFloatingLookbackOption(
        minmax,
        lambda_,
        _lookback_end() if lookback_end is None else lookback_end,
        FloatingTypePayoff(option_type),
        EuropeanExercise(_expiry()),
    )


def test_partial_floating_inspectors_return_the_extra_arguments() -> None:
    opt = _partial_floating()
    assert opt.minmax() == 100.0
    assert opt.lambda_() == 1.4
    assert opt.lookback_period_end() == _lookback_end()


def test_partial_floating_setup_arguments_populates_lambda_and_end_date() -> None:
    opt = _partial_floating()
    args = ContinuousPartialFloatingLookbackOptionArguments()
    opt.setup_arguments(args)
    assert args.minmax == 100.0
    assert args.lambda_ == 1.4
    assert args.lookback_period_end == _lookback_end()


def test_partial_floating_setup_arguments_rejects_base_argument_type() -> None:
    """The base carrier has nowhere to put lambda / the end date."""
    opt = _partial_floating()
    with pytest.raises(LibraryException, match="wrong argument type"):
        opt.setup_arguments(ContinuousFloatingLookbackOptionArguments())


def _partial_floating_args(
    option_type: OptionType = OptionType.Call,
    minmax: float = 100.0,
    lambda_: float = 1.4,
    lookback_end: Date | None = None,
) -> ContinuousPartialFloatingLookbackOptionArguments:
    args = ContinuousPartialFloatingLookbackOptionArguments()
    _partial_floating(option_type, minmax, lambda_, lookback_end).setup_arguments(args)
    return args


def test_partial_floating_validate_accepts_the_base_case() -> None:
    _partial_floating_args().validate()


def test_partial_floating_validate_rejects_lambda_below_one_for_calls() -> None:
    args = _partial_floating_args(OptionType.Call, lambda_=0.9)
    with pytest.raises(LibraryException, match="greater than or equal to 1 for calls"):
        args.validate()


def test_partial_floating_validate_rejects_lambda_above_one_for_puts() -> None:
    args = _partial_floating_args(OptionType.Put, lambda_=1.1)
    with pytest.raises(LibraryException, match="smaller than or equal to 1 for puts"):
        args.validate()


@pytest.mark.parametrize("option_type", [OptionType.Call, OptionType.Put])
def test_partial_floating_validate_accepts_lambda_exactly_one(
    option_type: OptionType,
) -> None:
    """Both constraints are non-strict, so lambda == 1 suits either type."""
    _partial_floating_args(option_type, lambda_=1.0).validate()


def test_partial_floating_validate_rejects_end_date_after_exercise() -> None:
    args = _partial_floating_args(lookback_end=_expiry() + 1)
    with pytest.raises(LibraryException, match="lookback start date must be earlier"):
        args.validate()


def test_partial_floating_validate_accepts_end_date_on_exercise() -> None:
    """The comparison is ``<=``: a window running exactly to expiry is legal."""
    _partial_floating_args(lookback_end=_expiry()).validate()


def test_partial_floating_validate_still_rejects_negative_minmax() -> None:
    """The base class's prior-extremum check is inherited, not replaced."""
    args = _partial_floating_args(minmax=-1.0)
    with pytest.raises(LibraryException, match="nonnegative prior extremum"):
        args.validate()


# --- ContinuousPartialFixedLookbackOption ----------------------------------


def _partial_fixed(
    option_type: OptionType = OptionType.Call,
    strike: float = 100.0,
    lookback_start: Date | None = None,
) -> ContinuousPartialFixedLookbackOption:
    return ContinuousPartialFixedLookbackOption(
        _lookback_end() if lookback_start is None else lookback_start,
        PlainVanillaPayoff(option_type, strike),
        EuropeanExercise(_expiry()),
    )


def test_partial_fixed_inspector_returns_the_start_date() -> None:
    assert _partial_fixed().lookback_period_start() == _lookback_end()


def test_partial_fixed_has_no_running_extremum() -> None:
    """C++ forwards a literal ``0`` to the base class — there is no argument."""
    assert _partial_fixed().minmax() == 0.0


def test_partial_fixed_setup_arguments_populates_start_date_and_zero_minmax() -> None:
    opt = _partial_fixed()
    args = ContinuousPartialFixedLookbackOptionArguments()
    opt.setup_arguments(args)
    assert args.lookback_period_start == _lookback_end()
    assert args.minmax == 0.0


def test_partial_fixed_setup_arguments_rejects_base_argument_type() -> None:
    opt = _partial_fixed()
    with pytest.raises(LibraryException, match="wrong argument type"):
        opt.setup_arguments(ContinuousFixedLookbackOptionArguments())


def _partial_fixed_args(
    lookback_start: Date | None = None,
) -> ContinuousPartialFixedLookbackOptionArguments:
    args = ContinuousPartialFixedLookbackOptionArguments()
    _partial_fixed(lookback_start=lookback_start).setup_arguments(args)
    return args


def test_partial_fixed_validate_accepts_the_base_case() -> None:
    _partial_fixed_args().validate()


def test_partial_fixed_validate_rejects_start_date_after_exercise() -> None:
    args = _partial_fixed_args(lookback_start=_expiry() + 1)
    with pytest.raises(LibraryException, match="lookback start date must be earlier"):
        args.validate()


def test_partial_fixed_validate_accepts_start_date_on_exercise() -> None:
    _partial_fixed_args(lookback_start=_expiry()).validate()
