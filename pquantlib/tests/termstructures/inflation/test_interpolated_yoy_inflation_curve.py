"""Tests for InterpolatedYoYInflationCurve — Linear interp matches C++."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.inflation.interpolated_yoy_inflation_curve import (
    InterpolatedYoYInflationCurve,
)
from pquantlib.testing.tolerance import tight
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


def _build_curve() -> InterpolatedYoYInflationCurve:
    full = _load_ref()
    ref = full["interpolated_yoy"]
    assert isinstance(ref, dict)
    nodes = cast(list[dict[str, float | int]], ref["nodes"])
    assert isinstance(nodes, list)
    dates = [Date(int(n["date_serial"])) for n in nodes]  # type: ignore[index, call-overload]
    rates = [float(n["rate"]) for n in nodes]  # type: ignore[index, call-overload]
    # Same reference date as the zero-curve probe block (today = 15 Jan 2020).
    reference = Date(int(full["interpolated_zero"]["reference_serial"]))  # type: ignore[index, call-overload]
    return InterpolatedYoYInflationCurve(
        reference_date=reference,
        dates=dates,
        rates=rates,
        frequency=Frequency.Monthly,
        day_counter=Actual360(),
    )


def test_base_rate_and_max_date_match_cpp() -> None:
    """C++ parity: baseRate() = rates[0], maxDate() = dates[-1]."""
    ref = _load_ref()["interpolated_yoy"]
    assert isinstance(ref, dict)
    curve = _build_curve()
    tight(curve.base_rate(), float(ref["base_rate"]))  # type: ignore[arg-type]
    assert curve.max_date() == Date(int(ref["max_serial"]))  # type: ignore[arg-type]
    assert curve.base_date() == Date(int(ref["base_serial"]))  # type: ignore[arg-type]


def test_yoy_rate_at_intermediate_dates_matches_cpp() -> None:
    """Linear interpolation between known YoY rates matches C++.

    Tolerance TIGHT: same algorithm + shared time arithmetic.
    """
    ref = _load_ref()["interpolated_yoy"]
    assert isinstance(ref, dict)
    curve = _build_curve()
    samples = cast(list[dict[str, float]], ref["samples"])
    assert isinstance(samples, list)
    for s in samples:
        d = Date(int(s["date_serial"]))  # type: ignore[index, call-overload]
        expected = float(s["yoy_rate"])  # type: ignore[index, call-overload]
        actual = curve.yoy_rate(d)
        tight(actual, expected)


def test_yoy_rate_at_anchor_nodes_returns_input_rate() -> None:
    """At each anchor date the curve returns the input rate exactly."""
    ref = _load_ref()["interpolated_yoy"]
    assert isinstance(ref, dict)
    curve = _build_curve()
    nodes = cast(list[dict[str, float | int]], ref["nodes"])
    assert isinstance(nodes, list)
    for node in nodes:
        d = Date(int(node["date_serial"]))  # type: ignore[index, call-overload]
        expected = float(node["rate"])  # type: ignore[index, call-overload]
        actual = curve.yoy_rate(d)
        tight(actual, expected)


def test_yoy_rate_below_minus_one_raises() -> None:
    """C++ parity: YoY rates may be negative but must be > -1."""
    d0 = Date.from_ymd(1, Month.January, 2020)
    d1 = Date.from_ymd(1, Month.January, 2021)
    with pytest.raises(LibraryException, match="year-on-year inflation data"):
        InterpolatedYoYInflationCurve(
            reference_date=d0,
            dates=[d0, d1],
            rates=[0.02, -1.5],
            frequency=Frequency.Monthly,
            day_counter=Actual360(),
        )


def test_inspectors_return_independent_copies() -> None:
    curve = _build_curve()
    assert curve.dates() == curve.dates()
    assert curve.dates() is not curve.dates()
    assert curve.data() == curve.rates()
    nodes = curve.nodes()
    assert len(nodes) == len(curve.dates())
