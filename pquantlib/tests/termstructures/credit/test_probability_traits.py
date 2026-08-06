"""Smoke tests for credit-curve bootstrap traits.

# C++ parity: ql/termstructures/credit/probabilitytraits.hpp.

Validates the per-trait API: initial_value, guess, min_value_after,
max_value_after, update_guess. No C++ probe reference — the traits
are pure static helpers; we cross-check by reproducing the C++ logic
arithmetic here.
"""

from __future__ import annotations

import math
import sys

from pquantlib.termstructures.credit.probability_traits import (
    DefaultDensity,
    HazardRate,
    SurvivalProbability,
)

# --- SurvivalProbability ---------------------------------------------------


def test_sp_initial_value_is_one() -> None:
    assert SurvivalProbability.initial_value() == 1.0


def test_sp_guess_first_pillar() -> None:
    # i=1, valid_data=False → 1/(1 + 0.01*0.25).
    expected = 1.0 / (1.0 + 0.01 * 0.25)
    assert SurvivalProbability.guess(1, [1.0, 0.0], [0.0, 1.0], False) == expected


def test_sp_min_value_after_invalid_data() -> None:
    # i=1, valid_data=False, times=[0,1], data=[1.0,...] →
    # data[i-1] * exp(-1.0 * (times[i] - times[i-1])).
    times = [0.0, 1.0]
    data = [1.0, 0.0]
    expected = 1.0 * math.exp(-1.0 * (times[1] - times[0]))
    assert SurvivalProbability.min_value_after(1, data, times, False) == expected


def test_sp_max_value_after_is_previous_pillar() -> None:
    data = [1.0, 0.98, 0.95]
    assert SurvivalProbability.max_value_after(2, data, [], False) == 0.98


def test_sp_update_guess_writes_pillar() -> None:
    data = [1.0, 0.0, 0.0]
    SurvivalProbability.update_guess(data, 0.96, 1)
    assert data[1] == 0.96
    assert data[0] == 1.0  # first point untouched (unlike HazardRate)


# --- HazardRate ------------------------------------------------------------


def test_hr_initial_value_is_avg_hazard() -> None:
    assert HazardRate.initial_value() == 0.01


def test_hr_max_value_after_invalid_is_max_hazard() -> None:
    # data ignored when valid_data=False.
    assert HazardRate.max_value_after(1, [], [], False) == 1.0


def test_hr_min_value_after_invalid_is_epsilon() -> None:
    assert HazardRate.min_value_after(1, [], [], False) == sys.float_info.epsilon


def test_hr_update_guess_propagates_to_first_pillar() -> None:
    """HazardRate trait writes both data[1] and data[0] on the first solve."""
    data = [0.01, 0.0, 0.0]
    HazardRate.update_guess(data, 0.025, 1)
    assert data[0] == 0.025
    assert data[1] == 0.025


# --- DefaultDensity --------------------------------------------------------


def test_dd_initial_value_is_avg_hazard() -> None:
    assert DefaultDensity.initial_value() == 0.01


def test_dd_max_value_after_valid_data_doubles_max() -> None:
    data = [0.01, 0.012, 0.015]
    assert DefaultDensity.max_value_after(0, data, [], True) == 0.030


def test_dd_min_value_after_valid_data_halves_min() -> None:
    data = [0.01, 0.012, 0.015]
    assert DefaultDensity.min_value_after(0, data, [], True) == 0.005


def test_dd_update_guess_propagates_to_first() -> None:
    data = [0.01, 0.0, 0.0]
    DefaultDensity.update_guess(data, 0.013, 1)
    assert data[0] == 0.013
    assert data[1] == 0.013
