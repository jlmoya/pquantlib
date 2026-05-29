"""Tests for :class:`VanillaStorageOption` and :class:`VanillaSwingOption`.

These instruments are scaffolds used by the W5-B FD engines (storage
+ swing). Construction + argument-validation are tested here; full
pricing requires the W5-A FD operators.

# C++ parity reference: ql/instruments/vanillastorageoption.{hpp,cpp}
# + ql/instruments/vanillaswingoption.{hpp,cpp} (v1.42.1).
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import BermudanExercise
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.experimental.finitedifferences.vanilla_storage_option import (
    VanillaStorageOption,
    VanillaStorageOptionArguments,
)
from pquantlib.experimental.finitedifferences.vanilla_swing_option import (
    VanillaSwingOption,
    VanillaSwingOptionArguments,
)
from pquantlib.payoffs import NullPayoff, OptionType, PlainVanillaPayoff
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def today() -> Date:
    return Date.from_ymd(18, Month.December, 2011)


@pytest.fixture(scope="module")
def bermudan(today: Date) -> BermudanExercise:
    dates = [today + i for i in (1, 30, 60, 90)]
    return BermudanExercise(dates)


@pytest.fixture(scope="module")
def swing(today: Date) -> SwingExercise:
    return SwingExercise.from_range(today, today + 1, 3600)


# ---------- VanillaStorageOption -------------------------------------------


def test_storage_option_accessors(bermudan: BermudanExercise) -> None:
    opt = VanillaStorageOption(
        exercise=bermudan, capacity=50.0, load=10.0, change_rate=5.0
    )
    tight(opt.capacity(), 50.0)
    tight(opt.load(), 10.0)
    tight(opt.change_rate(), 5.0)
    assert opt.exercise() is bermudan
    assert isinstance(opt.payoff(), NullPayoff)
    assert not opt.is_expired()


def test_storage_option_setup_arguments_roundtrip(
    bermudan: BermudanExercise,
) -> None:
    opt = VanillaStorageOption(
        exercise=bermudan, capacity=50.0, load=10.0, change_rate=5.0
    )
    args = VanillaStorageOptionArguments()
    opt.setup_arguments(args)
    assert args.capacity == 50.0
    assert args.load == 10.0
    assert args.change_rate == 5.0
    assert args.exercise is bermudan
    assert isinstance(args.payoff, NullPayoff)


def test_storage_arguments_validate_requires_positive_fields(
    bermudan: BermudanExercise,
) -> None:
    args = VanillaStorageOptionArguments()
    args.payoff = NullPayoff()
    args.exercise = bermudan
    args.capacity = 50.0
    args.load = 10.0
    args.change_rate = 5.0
    args.validate()  # no raise
    # capacity must be positive.
    args.capacity = -1.0
    with pytest.raises(LibraryException):
        args.validate()
    # load must be non-negative.
    args.capacity = 50.0
    args.load = -1.0
    with pytest.raises(LibraryException):
        args.validate()
    # change_rate must be positive.
    args.load = 10.0
    args.change_rate = -1.0
    with pytest.raises(LibraryException):
        args.validate()


def test_storage_arguments_validate_requires_load_le_capacity(
    bermudan: BermudanExercise,
) -> None:
    args = VanillaStorageOptionArguments()
    args.payoff = NullPayoff()
    args.exercise = bermudan
    args.capacity = 50.0
    args.load = 100.0  # > capacity
    args.change_rate = 5.0
    with pytest.raises(LibraryException):
        args.validate()


# ---------- VanillaSwingOption ---------------------------------------------


def test_swing_option_accessors(swing: SwingExercise) -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 30.0)
    opt = VanillaSwingOption(
        payoff=payoff, exercise=swing,
        min_exercise_rights=0, max_exercise_rights=5,
    )
    assert opt.min_exercise_rights() == 0
    assert opt.max_exercise_rights() == 5
    assert opt.payoff() is payoff
    assert opt.exercise() is swing


def test_swing_option_setup_arguments_roundtrip(swing: SwingExercise) -> None:
    payoff = PlainVanillaPayoff(OptionType.Put, 30.0)
    opt = VanillaSwingOption(
        payoff=payoff, exercise=swing,
        min_exercise_rights=1, max_exercise_rights=3,
    )
    args = VanillaSwingOptionArguments()
    opt.setup_arguments(args)
    assert args.min_exercise_rights == 1
    assert args.max_exercise_rights == 3
    assert args.payoff is payoff
    assert args.exercise is swing


def test_swing_arguments_validate_requires_min_le_max(
    swing: SwingExercise,
) -> None:
    args = VanillaSwingOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 30.0)
    args.exercise = swing
    args.min_exercise_rights = 3
    args.max_exercise_rights = 1
    with pytest.raises(LibraryException):
        args.validate()


def test_swing_arguments_validate_rejects_too_many_rights(
    swing: SwingExercise,
) -> None:
    args = VanillaSwingOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 30.0)
    args.exercise = swing
    args.min_exercise_rights = 0
    # swing has 48 exercise dates; requesting 100 rights is too many.
    args.max_exercise_rights = 100_000
    with pytest.raises(LibraryException):
        args.validate()
