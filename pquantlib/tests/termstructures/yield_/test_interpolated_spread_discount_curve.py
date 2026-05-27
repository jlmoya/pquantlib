"""Cross-validate ``InterpolatedSpreadDiscountCurve`` vs C++.

Probe key: "interpolated_spread_discount".
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.termstructures.yield_.discount_spreaded_term_structure import (
    DiscountSpreadedTermStructure,
    SpreadDiscountCurve,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.interpolated_spread_discount_curve import (
    InterpolatedSpreadDiscountCurve,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["interpolated_spread_discount"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


@pytest.fixture(scope="module")
def curve(ref_date: Date) -> InterpolatedSpreadDiscountCurve:
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    sdates = [ref_date, ref_date + 182, ref_date + 365, ref_date + 730]
    sdfs = [1.0, 0.995, 0.985, 0.965]
    return InterpolatedSpreadDiscountCurve(base, sdates, sdfs)


def test_discount_at_nodes(curve: InterpolatedSpreadDiscountCurve, cpp_ref: dict[str, Any]) -> None:
    dates = curve.dates()
    tolerance.tight(curve.discount(dates[1]), cpp_ref["discount_at_node_ref_plus_182"])
    tolerance.tight(curve.discount(dates[2]), cpp_ref["discount_at_node_ref_plus_365"])
    tolerance.tight(curve.discount(dates[3]), cpp_ref["discount_at_node_ref_plus_730"])


def test_discount_intermediate(
    curve: InterpolatedSpreadDiscountCurve, cpp_ref: dict[str, Any]
) -> None:
    tolerance.tight(curve.discount(0.5), cpp_ref["discount_t0_5"])
    tolerance.tight(curve.discount(1.0), cpp_ref["discount_t1_0"])


def test_discount_extrap(curve: InterpolatedSpreadDiscountCurve, cpp_ref: dict[str, Any]) -> None:
    tolerance.tight(curve.discount(2.0, extrapolate=True), cpp_ref["discount_t2_0_extrap"])


def test_discount_spreaded_term_structure_alias() -> None:
    # Cluster-scope alias; SpreadDiscountCurve is the C++ canonical typedef.
    assert DiscountSpreadedTermStructure.__value__ is InterpolatedSpreadDiscountCurve
    assert SpreadDiscountCurve.__value__ is InterpolatedSpreadDiscountCurve
