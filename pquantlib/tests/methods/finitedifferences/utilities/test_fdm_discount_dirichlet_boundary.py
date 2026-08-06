"""Cross-validation of ``FdmDiscountDirichletBoundary`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmdiscountdirichletboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).

The boundary value is ``cashFlow * P(maturityTime) / P(t)`` on a flat 5%
curve, i.e. two exponentials and a division — TIGHT tier is comfortable
(measured deviation 0).
"""

from __future__ import annotations

from typing import Any

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_discount_dirichlet_boundary import (
    FdmDiscountDirichletBoundary,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date

from .conftest import as_floats, ramp


def test_upper_direction_0_at_t04(
    reference_data: dict[str, Any],
    small_mesher: FdmMesherComposite,
    today: Date,
    day_counter: DayCounter,
) -> None:
    r_ts = FlatForward.from_rate(today, 0.05, day_counter)
    bc = FdmDiscountDirichletBoundary(
        small_mesher, r_ts, 1.0, 100.0, 0, BoundaryConditionSide.UPPER
    )
    bc.set_time(0.4)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_applying(a)
    expected = as_floats(reference_data["discount_dirichlet_d0_upper_t04"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_upper_direction_0_at_t0_via_after_solving(
    reference_data: dict[str, Any],
    small_mesher: FdmMesherComposite,
    today: Date,
    day_counter: DayCounter,
) -> None:
    r_ts = FlatForward.from_rate(today, 0.05, day_counter)
    bc = FdmDiscountDirichletBoundary(
        small_mesher, r_ts, 1.0, 100.0, 0, BoundaryConditionSide.UPPER
    )
    bc.set_time(0.0)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_solving(a)
    expected = as_floats(reference_data["discount_dirichlet_d0_upper_t0"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_lower_direction_1_forward_discounting(
    reference_data: dict[str, Any],
    small_mesher: FdmMesherComposite,
    today: Date,
    day_counter: DayCounter,
) -> None:
    """``t > maturityTime`` compounds the cash flow *up*, so the sign survives."""
    r_ts = FlatForward.from_rate(today, 0.05, day_counter)
    bc = FdmDiscountDirichletBoundary(
        small_mesher, r_ts, 2.0, -5.0, 1, BoundaryConditionSide.LOWER
    )
    bc.set_time(1.25)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_applying(a)
    expected = as_floats(reference_data["discount_dirichlet_d1_lower_t125"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_before_hooks_are_no_ops(
    small_mesher: FdmMesherComposite, today: Date, day_counter: DayCounter
) -> None:
    """# C++ parity: both ``applyBefore*`` forward to the held bc's empty bodies."""
    r_ts = FlatForward.from_rate(today, 0.05, day_counter)
    bc = FdmDiscountDirichletBoundary(
        small_mesher, r_ts, 1.0, 100.0, 0, BoundaryConditionSide.UPPER
    )
    bc.set_time(0.4)
    a = ramp(small_mesher.layout().size())
    before = a.copy()
    bc.apply_before_applying(object())
    bc.apply_before_solving(object(), a)
    assert list(a) == list(before)
