"""Tests for ZeroInflationTraits + YoYInflationTraits.

# C++ parity: ql/termstructures/inflation/inflationtraits.hpp (v1.42.1).

Constants and trait-algebra behaviour are compared against C++ probe
values from `migration-harness/references/cluster/l7b.json` (`traits`
key) and against documented C++ formulas.
"""

from __future__ import annotations

import json
from pathlib import Path

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.termstructures.inflation.interpolated_yoy_inflation_curve import (
    InterpolatedYoYInflationCurve,
)
from pquantlib.termstructures.inflation.interpolated_zero_inflation_curve import (
    InterpolatedZeroInflationCurve,
)
from pquantlib.termstructures.inflation.yoy_inflation_traits import YoYInflationTraits
from pquantlib.termstructures.inflation.zero_inflation_traits import (
    AVG_INFLATION,
    MAX_INFLATION,
    MAX_ITERATIONS,
    ZeroInflationTraits,
)
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

_HARNESS_REF = (
    Path(__file__).parents[4]
    / "migration-harness"
    / "references"
    / "cluster"
    / "l7b.json"
)


def _load_ref() -> dict[str, object]:
    return json.loads(_HARNESS_REF.read_text())


# ---- constants match the C++ probe ----------------------------------


def test_constants_match_cpp_probe() -> None:
    """Constants AVG_INFLATION = 0.02, MAX_INFLATION = 0.5, MAX_ITERATIONS = 40."""
    ref = _load_ref()["traits"]
    assert isinstance(ref, dict)
    exact(AVG_INFLATION, float(ref["avg_inflation"]))  # type: ignore[arg-type]
    exact(MAX_INFLATION, float(ref["max_inflation"]))  # type: ignore[arg-type]
    assert int(ref["max_iterations"]) == MAX_ITERATIONS  # type: ignore[arg-type]
    # The YoY traits re-export the same constants.
    yt = YoYInflationTraits()
    assert yt.max_iterations() == MAX_ITERATIONS


# ---- ZeroInflationTraits algebra ------------------------------------


def _build_zero_curve() -> InterpolatedZeroInflationCurve:
    """Build a small zero curve for trait initial-date/value tests."""
    today = Date.from_ymd(15, Month.January, 2020)
    return InterpolatedZeroInflationCurve(
        reference_date=today,
        dates=[
            Date.from_ymd(1, Month.January, 2020),
            Date.from_ymd(1, Month.January, 2025),
        ],
        rates=[0.015, 0.025],
        frequency=Frequency.Monthly,
        day_counter=Actual360(),
    )


def test_zero_initial_date_is_base_date() -> None:
    """C++ parity: ``ZeroInflationTraits::initialDate(ts)`` returns ``ts.baseDate()``."""
    t = ZeroInflationTraits()
    ts = _build_zero_curve()
    assert t.initial_date(ts) == ts.base_date()


def test_zero_initial_value_returns_avg_inflation() -> None:
    """C++ parity: ZeroInflationTraits seeds with detail::avgInflation = 0.02."""
    t = ZeroInflationTraits()
    ts = _build_zero_curve()
    exact(t.initial_value(ts), 0.02)


def test_zero_guess_returns_data_i_when_valid() -> None:
    """C++ parity: ``valid_data`` ⇒ guess(i, data, true) == data[i]."""
    t = ZeroInflationTraits()
    data = [0.02, 0.03, 0.04, 0.05]
    exact(t.guess(2, data, valid_data=True), 0.04)
    # And the AVG_INFLATION fallback for non-valid data.
    exact(t.guess(2, data, valid_data=False), AVG_INFLATION)


def test_zero_min_max_value_after_no_valid_data() -> None:
    """Without valid prior data, bounds = ±MAX_INFLATION."""
    t = ZeroInflationTraits()
    data = [0.0, 0.0]
    exact(t.min_value_after(1, data, valid_data=False), -MAX_INFLATION)
    exact(t.max_value_after(1, data, valid_data=False), MAX_INFLATION)


def test_zero_min_max_value_after_valid_data_all_positive() -> None:
    """Positive data → min/2, max*2 — # C++ parity inflationtraits.hpp."""
    t = ZeroInflationTraits()
    data = [0.02, 0.03, 0.025, 0.022]
    # min over [0.02, 0.03, 0.025, 0.022] = 0.02 → halve to 0.01.
    tight(t.min_value_after(2, data, valid_data=True), 0.01)
    # max = 0.03 → double to 0.06.
    tight(t.max_value_after(2, data, valid_data=True), 0.06)


def test_zero_min_max_value_after_valid_data_with_negative() -> None:
    """Negative data → flips the bound (r*2 if r<0, r/2 otherwise)."""
    t = ZeroInflationTraits()
    data = [-0.01, 0.02, -0.03]
    # min = -0.03 → r<0 → r*2 = -0.06.
    tight(t.min_value_after(1, data, valid_data=True), -0.06)
    # max = 0.02 → r>0 → r*2 = 0.04.
    tight(t.max_value_after(1, data, valid_data=True), 0.04)


def test_zero_update_guess_propagates_to_data_zero_when_i_is_one() -> None:
    """C++ parity: ZeroInflationTraits::updateGuess sets data[0] = level when i == 1."""
    t = ZeroInflationTraits()
    data = [99.0, 99.0, 99.0]
    t.update_guess(data, 0.025, 1)
    exact(data[1], 0.025)
    exact(data[0], 0.025)  # propagated.


def test_zero_update_guess_no_propagation_for_i_above_one() -> None:
    """For i > 1 only data[i] is set; data[0] is preserved."""
    t = ZeroInflationTraits()
    data = [99.0, 99.0, 99.0]
    t.update_guess(data, 0.03, 2)
    exact(data[2], 0.03)
    exact(data[0], 99.0)


def test_zero_max_iterations_constant() -> None:
    t = ZeroInflationTraits()
    assert t.max_iterations() == 40


# ---- YoYInflationTraits algebra -------------------------------------


def _build_yoy_curve(base_rate: float = 0.02) -> InterpolatedYoYInflationCurve:
    today = Date.from_ymd(15, Month.January, 2020)
    return InterpolatedYoYInflationCurve(
        reference_date=today,
        dates=[
            Date.from_ymd(1, Month.January, 2020),
            Date.from_ymd(1, Month.January, 2025),
        ],
        rates=[base_rate, 0.025],
        frequency=Frequency.Monthly,
        day_counter=Actual360(),
    )


def test_yoy_initial_date_is_base_date() -> None:
    t = YoYInflationTraits()
    ts = _build_yoy_curve()
    assert t.initial_date(ts) == ts.base_date()


def test_yoy_initial_value_returns_base_rate_not_constant() -> None:
    """C++ parity: YoYInflationTraits seeds from ts.baseRate(), not AVG_INFLATION.

    Key difference from ZeroInflationTraits.
    """
    t = YoYInflationTraits()
    ts = _build_yoy_curve(base_rate=0.018)
    tight(t.initial_value(ts), 0.018)


def test_yoy_update_guess_does_not_propagate_to_data_zero() -> None:
    """C++ parity: YoYInflationTraits::updateGuess never touches data[0].

    Key difference from ZeroInflationTraits — YoY's base rate is user-set.
    """
    t = YoYInflationTraits()
    data = [0.018, 99.0, 99.0]
    t.update_guess(data, 0.027, 1)
    exact(data[1], 0.027)
    # data[0] preserved — this is the user-supplied base YoY rate.
    exact(data[0], 0.018)


def test_yoy_guess_and_bounds_same_shape_as_zero() -> None:
    """YoY guess/min_value_after/max_value_after share the algebra with zero."""
    t = YoYInflationTraits()
    data = [0.02, 0.03, 0.025]
    exact(t.guess(2, data, valid_data=True), 0.025)
    exact(t.guess(2, data, valid_data=False), AVG_INFLATION)
    tight(t.min_value_after(1, data, valid_data=True), 0.01)
    tight(t.max_value_after(1, data, valid_data=True), 0.06)
