"""Tests for :class:`VanillaVPPOption` and its arguments class.

# C++ parity reference:
# ql/experimental/finitedifferences/vanillavppoption.{hpp,cpp} (v1.42.1).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.experimental.finitedifferences.vanilla_vpp_option import (
    VanillaVPPOption,
    VanillaVPPOptionArguments,
)
from pquantlib.instruments.basket_option import AverageBasketPayoff
from pquantlib.instruments.multi_asset_option import MultiAssetOption
from pquantlib.option import OptionArguments
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def today() -> Date:
    return Date.from_ymd(18, Month.December, 2011)


@pytest.fixture(scope="module")
def short_exercise(today: Date) -> SwingExercise:
    """Hourly swing exercise across (today, today + 1)."""
    return SwingExercise.from_range(today, today + 1, 3600)


def test_construction_sets_payoff_to_avg_basket_with_heatrate_weights(
    short_exercise: SwingExercise,
) -> None:
    """The embedded payoff is :class:`AverageBasketPayoff` with weights
    ``[1, -heat_rate]`` over an identity base payoff — matching C++.
    """
    heat_rate = 2.5
    opt = VanillaVPPOption(
        heat_rate=heat_rate,
        p_min=8,
        p_max=40,
        t_min_up=2,
        t_min_down=2,
        start_up_fuel=20,
        start_up_fix_cost=100,
        exercise=short_exercise,
    )
    assert isinstance(opt, MultiAssetOption)
    payoff = opt.payoff()
    assert isinstance(payoff, AverageBasketPayoff)
    weights = payoff.weights()
    assert weights.tolist() == [1.0, -heat_rate]


def test_accessors_roundtrip(short_exercise: SwingExercise) -> None:
    opt = VanillaVPPOption(
        heat_rate=2.5,
        p_min=8,
        p_max=40,
        t_min_up=2,
        t_min_down=2,
        start_up_fuel=20,
        start_up_fix_cost=100,
        exercise=short_exercise,
        n_starts=3,
    )
    exact(opt.heat_rate(), 2.5)
    exact(opt.p_min(), 8.0)
    exact(opt.p_max(), 40.0)
    assert opt.t_min_up() == 2
    assert opt.t_min_down() == 2
    exact(opt.start_up_fuel(), 20.0)
    exact(opt.start_up_fix_cost(), 100.0)
    assert opt.n_starts() == 3
    assert opt.n_running_hours() is None
    assert not opt.is_expired()


def test_setup_arguments_round_trip_matches_cpp(
    cpp_ref: dict[str, Any],
    short_exercise: SwingExercise,
) -> None:
    """``setup_arguments`` populates the engine arguments with the same
    field values as the C++ ``VanillaVPPOption::setupArguments``.
    """
    ref = cpp_ref["setup_arguments_roundtrip"]
    opt = VanillaVPPOption(
        heat_rate=ref["heat_rate"],
        p_min=ref["p_min"],
        p_max=ref["p_max"],
        t_min_up=ref["t_min_up"],
        t_min_down=ref["t_min_down"],
        start_up_fuel=ref["start_up_fuel"],
        start_up_fix_cost=ref["start_up_fix_cost"],
        exercise=short_exercise,
        n_starts=ref["n_starts"],
    )
    args = VanillaVPPOptionArguments()
    opt.setup_arguments(args)
    assert args.heat_rate is not None
    tight(args.heat_rate, ref["heat_rate"])
    assert args.p_min is not None
    tight(args.p_min, ref["p_min"])
    assert args.p_max is not None
    tight(args.p_max, ref["p_max"])
    assert args.t_min_up == ref["t_min_up"]
    assert args.t_min_down == ref["t_min_down"]
    assert args.start_up_fuel is not None
    tight(args.start_up_fuel, ref["start_up_fuel"])
    assert args.start_up_fix_cost is not None
    tight(args.start_up_fix_cost, ref["start_up_fix_cost"])
    assert args.n_starts == ref["n_starts"]
    # n_running_hours_is_null: True ↔ args.n_running_hours is None.
    assert (args.n_running_hours is None) == ref["n_running_hours_is_null"]


def test_arguments_validate_rejects_both_limits_set(
    short_exercise: SwingExercise,
) -> None:
    """``validate()`` raises when both ``n_starts`` and ``n_running_hours``
    are set — matching the C++ invariant.
    """
    args = VanillaVPPOptionArguments()
    args.exercise = short_exercise
    args.n_starts = 3
    args.n_running_hours = 100
    with pytest.raises(LibraryException):
        args.validate()


def test_arguments_validate_accepts_exactly_one_limit(
    short_exercise: SwingExercise,
) -> None:
    args = VanillaVPPOptionArguments()
    args.exercise = short_exercise
    # No limits: OK.
    args.validate()
    args.n_starts = 3
    args.validate()
    args.n_starts = None
    args.n_running_hours = 100
    args.validate()


def test_arguments_validate_rejects_missing_exercise() -> None:
    args = VanillaVPPOptionArguments()
    with pytest.raises(LibraryException):
        args.validate()


def test_setup_arguments_rejects_wrong_type(short_exercise: SwingExercise) -> None:
    """Calling ``setup_arguments`` with a non-VPP args class raises."""
    opt = VanillaVPPOption(
        heat_rate=2.0,
        p_min=8,
        p_max=40,
        t_min_up=2,
        t_min_down=2,
        start_up_fuel=20,
        start_up_fix_cost=100,
        exercise=short_exercise,
    )
    plain = OptionArguments()
    with pytest.raises(LibraryException):
        opt.setup_arguments(plain)
