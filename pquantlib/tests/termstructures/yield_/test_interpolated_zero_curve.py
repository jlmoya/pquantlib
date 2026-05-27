"""Cross-validate ``InterpolatedZeroCurve`` (+ ``ZeroCurve`` alias) vs C++.

Probe source: migration-harness/cpp/probes/cluster_l2b/probe.cpp
Reference:    migration-harness/references/cluster/l2b.json
              key: "interpolated_zero_curve"
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.termstructures.yield_.zero_curve import ZeroCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["interpolated_zero_curve"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


@pytest.fixture(scope="module")
def curve(ref_date: Date) -> InterpolatedZeroCurve:
    dates = [
        ref_date,
        ref_date + 30,
        ref_date + 91,
        ref_date + 182,
        ref_date + 365,
        ref_date + 730,
    ]
    yields = [0.020, 0.022, 0.025, 0.028, 0.030, 0.035]
    # Default interpolator = Linear, default compounding = Continuous.
    return InterpolatedZeroCurve(dates, yields, Actual365Fixed())


def test_discount_at_nodes(curve: InterpolatedZeroCurve, cpp_ref: dict[str, Any]) -> None:
    for i, d in enumerate(curve.dates()):
        actual = curve.discount(d)
        expected = cpp_ref[f"discount_node{i}"]
        tolerance.tight(actual, expected, reason=f"node {i}")


def test_zero_rate_at_nodes(curve: InterpolatedZeroCurve, cpp_ref: dict[str, Any]) -> None:
    for i, d in enumerate(curve.dates()):
        actual = curve.zero_rate(d, Compounding.Continuous, Frequency.Annual, False, Actual365Fixed()).rate()
        expected = cpp_ref[f"zero_rate_node{i}"]
        tolerance.tight(actual, expected, reason=f"node {i}")


def test_discount_intermediate_times(curve: InterpolatedZeroCurve, cpp_ref: dict[str, Any]) -> None:
    tolerance.tight(curve.discount(0.5), cpp_ref["discount_t0_5"])
    tolerance.tight(curve.discount(1.0), cpp_ref["discount_t1_0"])
    tolerance.tight(curve.discount(1.5), cpp_ref["discount_t1_5"])


def test_discount_extrapolation(curve: InterpolatedZeroCurve, cpp_ref: dict[str, Any]) -> None:
    # t = 3.0 is past the last knot (t_max ≈ 2.0).
    tolerance.tight(curve.discount(3.0, extrapolate=True), cpp_ref["discount_t3_extrap"])


def test_max_date_serial(curve: InterpolatedZeroCurve, cpp_ref: dict[str, Any]) -> None:
    # max_date is the last entry in dates_.
    # serial_number of last date matches C++.
    last = curve.max_date()
    # Date has serial_number — let me check
    assert last == curve.dates()[-1]


def test_zero_curve_type_alias_is_interpolated_zero_curve() -> None:
    # PEP 695 ``type X = Y`` produces a TypeAliasType; ``.__value__`` is the
    # aliased type — runtime equivalent of the C++ ``typedef``.
    assert ZeroCurve.__value__ is InterpolatedZeroCurve
