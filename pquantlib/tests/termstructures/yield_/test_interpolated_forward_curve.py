"""Cross-validate ``InterpolatedForwardCurve`` (+ ``ForwardCurve`` alias).

Probe key: "interpolated_forward_curve".
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.yield_.forward_curve import ForwardCurve
from pquantlib.termstructures.yield_.interpolated_forward_curve import InterpolatedForwardCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["interpolated_forward_curve"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


@pytest.fixture(scope="module")
def curve(ref_date: Date) -> InterpolatedForwardCurve:
    dates = [
        ref_date,
        ref_date + 30,
        ref_date + 91,
        ref_date + 182,
        ref_date + 365,
        ref_date + 730,
    ]
    fwds = [0.020, 0.022, 0.025, 0.028, 0.030, 0.035]
    # Default interpolator = BackwardFlat.
    return InterpolatedForwardCurve(dates, fwds, Actual365Fixed())


def test_discount_intermediate(curve: InterpolatedForwardCurve, cpp_ref: dict[str, Any]) -> None:
    tolerance.tight(curve.discount(0.2), cpp_ref["discount_t0_2"])
    tolerance.tight(curve.discount(0.5), cpp_ref["discount_t0_5"])
    tolerance.tight(curve.discount(1.0), cpp_ref["discount_t1_0"])
    tolerance.tight(curve.discount(1.5), cpp_ref["discount_t1_5"])


def test_discount_at_last_node(curve: InterpolatedForwardCurve, cpp_ref: dict[str, Any]) -> None:
    last_d = curve.dates()[-1]
    tolerance.tight(curve.discount(last_d), cpp_ref["discount_t2_0_extrap"])


def test_discount_extrap(curve: InterpolatedForwardCurve, cpp_ref: dict[str, Any]) -> None:
    tolerance.tight(curve.discount(3.0, extrapolate=True), cpp_ref["discount_t3_extrap"])


def test_forward_curve_type_alias() -> None:
    assert ForwardCurve.__value__ is InterpolatedForwardCurve
