"""Tests for InterpolatedZeroInflationCurve — Linear interp matches C++."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.inflation.interpolated_zero_inflation_curve import (
    InterpolatedZeroInflationCurve,
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


def _build_curve() -> InterpolatedZeroInflationCurve:
    ref = _load_ref()["interpolated_zero"]
    assert isinstance(ref, dict)
    nodes = ref["nodes"]
    assert isinstance(nodes, list)
    dates = [Date(int(n["date_serial"])) for n in nodes]  # type: ignore[index, call-overload]
    rates = [float(n["rate"]) for n in nodes]  # type: ignore[index, call-overload]
    reference = Date(int(ref["reference_serial"]))  # type: ignore[arg-type]
    return InterpolatedZeroInflationCurve(
        reference_date=reference,
        dates=dates,
        rates=rates,
        frequency=Frequency.Monthly,
        day_counter=Actual360(),
    )


def test_base_and_max_date_match_cpp() -> None:
    """C++ parity: baseDate() = dates[0], maxDate() = dates[-1]."""
    ref = _load_ref()["interpolated_zero"]
    assert isinstance(ref, dict)
    curve = _build_curve()
    assert curve.base_date() == Date(int(ref["base_serial"]))  # type: ignore[arg-type]
    assert curve.max_date() == Date(int(ref["max_serial"]))  # type: ignore[arg-type]


def test_zero_rate_at_anchor_dates_returns_input_rate() -> None:
    """At anchor dates the curve returns the input rates exactly.

    Tolerance TIGHT: linear interpolation through (t, r) is exact at the nodes.
    """
    ref = _load_ref()["interpolated_zero"]
    assert isinstance(ref, dict)
    curve = _build_curve()
    nodes = ref["nodes"]
    assert isinstance(nodes, list)
    # First node coincides with the curve base date — skip it because
    # the interpolation is exact there but inflation_period bucketing
    # forces the period start, which in January 2020 = same date.
    for node in nodes:
        d = Date(int(node["date_serial"]))  # type: ignore[index, call-overload]
        expected = float(node["rate"])  # type: ignore[index, call-overload]
        actual = curve.zero_rate(d)
        tight(actual, expected)


def test_zero_rate_at_intermediate_dates_matches_cpp() -> None:
    """Linear interpolation between known nodes matches C++ verbatim.

    Tolerance TIGHT: the same C++ Linear interpolation algorithm is
    reproduced (we share the same time arithmetic + linear formula).
    """
    ref = _load_ref()["interpolated_zero"]
    assert isinstance(ref, dict)
    curve = _build_curve()
    samples = ref["samples"]
    assert isinstance(samples, list)
    for s in samples:
        d = Date(int(s["date_serial"]))  # type: ignore[index, call-overload]
        expected = float(s["zero_rate"])  # type: ignore[index, call-overload]
        actual = curve.zero_rate(d)
        tight(actual, expected)


def test_too_few_dates_raises() -> None:
    """C++ parity: dates.size() > 1 required."""
    d0 = Date.from_ymd(15, Month.January, 2020)
    with pytest.raises(LibraryException, match="too few dates"):
        InterpolatedZeroInflationCurve(
            reference_date=d0,
            dates=[d0],
            rates=[0.02],
            frequency=Frequency.Monthly,
            day_counter=Actual360(),
        )


def test_count_mismatch_raises() -> None:
    """C++ parity: rates count must equal dates count."""
    d0 = Date.from_ymd(15, Month.January, 2020)
    d1 = Date.from_ymd(15, Month.January, 2021)
    with pytest.raises(LibraryException, match="indices/dates count mismatch"):
        InterpolatedZeroInflationCurve(
            reference_date=d0,
            dates=[d0, d1],
            rates=[0.02],
            frequency=Frequency.Monthly,
            day_counter=Actual360(),
        )


def test_rate_below_minus_one_raises() -> None:
    """C++ parity: rates must be > -1 (i.e. > -100%)."""
    d0 = Date.from_ymd(15, Month.January, 2020)
    d1 = Date.from_ymd(15, Month.January, 2021)
    with pytest.raises(LibraryException, match="zero inflation data"):
        InterpolatedZeroInflationCurve(
            reference_date=d0,
            dates=[d0, d1],
            rates=[0.02, -1.5],
            frequency=Frequency.Monthly,
            day_counter=Actual360(),
        )


def test_inspectors_return_independent_copies() -> None:
    """C++ parity: ``dates()`` / ``times()`` / ``data()`` accessors.

    Python idiom: each returns a fresh list so callers can't mutate state.
    """
    curve = _build_curve()
    d1 = curve.dates()
    d2 = curve.dates()
    assert d1 == d2
    assert d1 is not d2
    assert len(curve.times()) == len(curve.dates())
    assert curve.data() == curve.rates()
    nodes = curve.nodes()
    assert all(isinstance(n, tuple) and len(n) == 2 for n in nodes)
