"""Cross-validation of ``FdmAffineModelTermStructure`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmaffinemodeltermstructure.{hpp,cpp}
# @ v1.43 (6b57206e0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.methods.finitedifferences.utilities.fdm_affine_model_term_structure import (
    FdmAffineModelTermStructure,
)
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


@pytest.fixture
def flat_curve(today: Date, day_counter: DayCounter) -> YieldTermStructure:
    """Probe setup: flat 5% continuous curve anchored at ``today``."""
    return FlatForward.from_rate(today, 0.05, day_counter)


def test_hull_white_dates(
    reference_data: dict[str, Any],
    flat_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    ts = FdmAffineModelTermStructure(
        np.array([0.04]),
        calendar,
        day_counter,
        today + 180,
        today,
        HullWhite(flat_curve, 0.1, 0.01),
    )
    assert ts.reference_date().serial_number() == int(
        reference_data["affine_ts_hw_reference_serial"]
    )
    assert ts.max_date().serial_number() == int(reference_data["affine_ts_hw_max_date_serial"])


@pytest.mark.parametrize(
    ("t", "key"),
    [
        (0.5, "affine_ts_hw_discount_05"),
        (1.0, "affine_ts_hw_discount_10"),
        (3.0, "affine_ts_hw_discount_30"),
    ],
)
def test_hull_white_discounts(
    reference_data: dict[str, Any],
    flat_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
    t: float,
    key: str,
) -> None:
    """``discount(T) = model.discount_bond(t0, T + t0, r)`` with ``t0`` the anchor."""
    ts = FdmAffineModelTermStructure(
        np.array([0.04]),
        calendar,
        day_counter,
        today + 180,
        today,
        HullWhite(flat_curve, 0.1, 0.01),
    )
    tight(ts.discount(t), float(reference_data[key]))


def test_hull_white_set_variable(
    reference_data: dict[str, Any],
    flat_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    """``setVariable`` swaps the state in place — the curve object is reused."""
    ts = FdmAffineModelTermStructure(
        np.array([0.04]),
        calendar,
        day_counter,
        today + 180,
        today,
        HullWhite(flat_curve, 0.1, 0.01),
    )
    before = ts.discount(1.0)
    ts.set_variable(np.array([0.06]))
    after = ts.discount(1.0)
    assert after != before
    tight(after, float(reference_data["affine_ts_hw_discount_10_after_set"]))


@pytest.mark.parametrize(
    ("t", "key"),
    [(0.5, "affine_ts_g2_discount_05"), (1.0, "affine_ts_g2_discount_10")],
)
def test_g2_discounts(
    reference_data: dict[str, Any],
    flat_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
    t: float,
    key: str,
) -> None:
    ts = FdmAffineModelTermStructure(
        np.array([0.01, 0.02]),
        calendar,
        day_counter,
        today + 180,
        today,
        G2(flat_curve, 0.1, 0.01, 0.1, 0.012, -0.75),
    )
    tight(ts.discount(t), float(reference_data[key]))


def test_g2_set_variable(
    reference_data: dict[str, Any],
    flat_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    ts = FdmAffineModelTermStructure(
        np.array([0.01, 0.02]),
        calendar,
        day_counter,
        today + 180,
        today,
        G2(flat_curve, 0.1, 0.01, 0.1, 0.012, -0.75),
    )
    ts.set_variable(np.array([-0.005, 0.03]))
    tight(ts.discount(1.0), float(reference_data["affine_ts_g2_discount_10_after_set"]))
