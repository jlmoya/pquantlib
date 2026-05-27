"""Cross-validate ``InterpolatedDiscountCurve`` (+ ``DiscountCurve`` alias).

Probe key: "interpolated_discount_curve".
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.yield_.discount_curve import DiscountCurve
from pquantlib.termstructures.yield_.interpolated_discount_curve import InterpolatedDiscountCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["interpolated_discount_curve"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


@pytest.fixture(scope="module")
def curve(ref_date: Date) -> InterpolatedDiscountCurve:
    dates = [
        ref_date,
        ref_date + 30,
        ref_date + 91,
        ref_date + 182,
        ref_date + 365,
        ref_date + 730,
    ]
    dfs = [1.0, 0.998, 0.993, 0.985, 0.970, 0.930]
    return InterpolatedDiscountCurve(dates, dfs, Actual365Fixed())


def test_discount_at_nodes(curve: InterpolatedDiscountCurve, cpp_ref: dict[str, Any]) -> None:
    for i, d in enumerate(curve.dates()):
        actual = curve.discount(d)
        expected = cpp_ref[f"discount_node{i}"]
        tolerance.tight(actual, expected, reason=f"node {i}")


def test_discount_intermediate(curve: InterpolatedDiscountCurve, cpp_ref: dict[str, Any]) -> None:
    tolerance.tight(curve.discount(0.2), cpp_ref["discount_t0_2"])
    tolerance.tight(curve.discount(1.0), cpp_ref["discount_t1_0"])


def test_discount_extrap(curve: InterpolatedDiscountCurve, cpp_ref: dict[str, Any]) -> None:
    tolerance.tight(curve.discount(3.0, extrapolate=True), cpp_ref["discount_t3_extrap"])


def test_first_discount_must_be_one(ref_date: Date) -> None:
    dates = [ref_date, ref_date + 365]
    dfs = [0.99, 0.95]  # first not 1.0 → raises
    with pytest.raises(LibraryException):
        InterpolatedDiscountCurve(dates, dfs, Actual365Fixed())


def test_negative_discount_rejected(ref_date: Date) -> None:
    dates = [ref_date, ref_date + 365]
    dfs = [1.0, -0.5]
    with pytest.raises(LibraryException):
        InterpolatedDiscountCurve(dates, dfs, Actual365Fixed())


def test_discount_curve_type_alias() -> None:
    assert DiscountCurve.__value__ is InterpolatedDiscountCurve
