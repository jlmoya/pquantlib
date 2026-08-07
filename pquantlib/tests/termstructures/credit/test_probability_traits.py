"""Smoke tests for credit-curve bootstrap traits.

# C++ parity: ql/termstructures/credit/probabilitytraits.hpp (v1.43).

Validates the per-trait API: initial_value, guess, min_value_after,
max_value_after, update_guess. No C++ probe reference — the traits
are pure static helpers; we cross-check by reproducing the C++ logic
arithmetic here.

``guess`` / ``min_value_after`` / ``max_value_after`` take the CURVE, as
C++ does (``const C* c``), plus the trailing ``firstAliveHelper`` index.
:class:`_StubCurve` supplies just the accessors the traits read.
"""

from __future__ import annotations

import math
import sys

from pquantlib.termstructures.credit.probability_traits import (
    DefaultDensity,
    HazardRate,
    SurvivalProbability,
)
from pquantlib.time.date import Date


class _StubCurve:
    """Minimal stand-in for ``const C* c`` — data / times / dates only."""

    def __init__(
        self,
        data: list[float],
        times: list[float] | None = None,
        dates: list[Date] | None = None,
    ) -> None:
        self._data = data
        self._times = times if times is not None else []
        self._dates = dates if dates is not None else []

    def data(self) -> list[float]:
        return self._data

    def times(self) -> list[float]:
        return self._times

    def dates(self) -> list[Date]:
        return self._dates


# --- SurvivalProbability ---------------------------------------------------


def test_sp_initial_value_is_one() -> None:
    assert SurvivalProbability.initial_value() == 1.0


def test_sp_guess_first_pillar() -> None:
    # i=1, valid_data=False → 1/(1 + 0.01*0.25).
    expected = 1.0 / (1.0 + 0.01 * 0.25)
    c = _StubCurve([1.0, 0.0], times=[0.0, 1.0])
    assert SurvivalProbability.guess(1, c, False, 0) == expected


def test_sp_min_value_after_invalid_data() -> None:
    # i=1, valid_data=False, times=[0,1], data=[1.0,...] →
    # data[i-1] * exp(-1.0 * (times[i] - times[i-1])).
    times = [0.0, 1.0]
    data = [1.0, 0.0]
    expected = 1.0 * math.exp(-1.0 * (times[1] - times[0]))
    c = _StubCurve(data, times=times)
    assert SurvivalProbability.min_value_after(1, c, False, 0) == expected


def test_sp_max_value_after_is_previous_pillar() -> None:
    c = _StubCurve([1.0, 0.98, 0.95])
    assert SurvivalProbability.max_value_after(2, c, False, 0) == 0.98


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
    assert HazardRate.max_value_after(1, _StubCurve([]), False, 0) == 1.0


def test_hr_min_value_after_invalid_is_epsilon() -> None:
    assert (
        HazardRate.min_value_after(1, _StubCurve([]), False, 0)
        == sys.float_info.epsilon
    )


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
    c = _StubCurve([0.01, 0.012, 0.015])
    assert DefaultDensity.max_value_after(0, c, True, 0) == 0.030


def test_dd_min_value_after_valid_data_halves_min() -> None:
    c = _StubCurve([0.01, 0.012, 0.015])
    assert DefaultDensity.min_value_after(0, c, True, 0) == 0.005


def test_dd_update_guess_propagates_to_first() -> None:
    data = [0.01, 0.0, 0.0]
    DefaultDensity.update_guess(data, 0.013, 1)
    assert data[0] == 0.013
    assert data[1] == 0.013
