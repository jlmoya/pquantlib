"""Cross-validation of ``FdmDirichletBoundary`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmdirichletboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).

All values are constants copied into the array (no arithmetic), so TIGHT is
the natural tier — the measured deviation is exactly zero.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_dirichlet_boundary import (
    FdmDirichletBoundary,
)
from pquantlib.testing.tolerance import tight

from .conftest import as_floats, ramp


def test_apply_after_applying_lower(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite
) -> None:
    bc = FdmDirichletBoundary(small_mesher, 3.5, 0, BoundaryConditionSide.LOWER)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_applying(a)
    expected = as_floats(reference_data["dirichlet_d0_lower_after_applying"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_apply_after_solving_lower(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite
) -> None:
    bc = FdmDirichletBoundary(small_mesher, 3.5, 0, BoundaryConditionSide.LOWER)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_solving(a)
    expected = as_floats(reference_data["dirichlet_d0_lower_after_solving"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_apply_after_applying_upper_direction_1(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite
) -> None:
    bc = FdmDirichletBoundary(small_mesher, -1.25, 1, BoundaryConditionSide.UPPER)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_applying(a)
    expected = as_floats(reference_data["dirichlet_d1_upper_after_applying"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


@pytest.mark.parametrize(
    ("x", "key"),
    [
        (-3.0, "dirichlet_d0_lower_scalar_below"),
        (-2.0, "dirichlet_d0_lower_scalar_at"),
        (0.5, "dirichlet_d0_lower_scalar_above"),
    ],
)
def test_scalar_overload_lower(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite, x: float, key: str
) -> None:
    bc = FdmDirichletBoundary(small_mesher, 3.5, 0, BoundaryConditionSide.LOWER)
    tight(bc.apply_after_applying_value(x, 7.0), float(reference_data[key]))


@pytest.mark.parametrize(
    ("x", "key"),
    [
        (2.0, "dirichlet_d1_upper_scalar_above"),
        (1.0, "dirichlet_d1_upper_scalar_at"),
        (0.0, "dirichlet_d1_upper_scalar_below"),
    ],
)
def test_scalar_overload_upper_direction_1(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite, x: float, key: str
) -> None:
    """C++ picks ``locations(1)[dim[1]-1]`` — a *flat* index — as the extreme.

    For direction 1 that flat index still has coordinate 0 along direction 1,
    so ``xExtreme_`` lands on the first y node (-1.0) and every probed x is
    "above" it. The port reproduces that verbatim; the reference values
    confirm all three cases collapse to the boundary value.
    """
    bc = FdmDirichletBoundary(small_mesher, -1.25, 1, BoundaryConditionSide.UPPER)
    tight(bc.apply_after_applying_value(x, 7.0), float(reference_data[key]))


def test_none_side_rejected(small_mesher: FdmMesherComposite) -> None:
    """# C++ parity: ``QL_FAIL("internal error")`` for a side other than Lower/Upper."""
    with pytest.raises(LibraryException):
        FdmDirichletBoundary(small_mesher, 1.0, 0, BoundaryConditionSide.NONE)


def test_before_hooks_are_no_ops(small_mesher: FdmMesherComposite) -> None:
    """# C++ parity: both ``applyBefore*`` bodies are empty."""
    bc = FdmDirichletBoundary(small_mesher, 3.5, 0, BoundaryConditionSide.LOWER)
    a = ramp(small_mesher.layout().size())
    before = a.copy()
    bc.apply_before_applying(object())
    bc.apply_before_solving(object(), a)
    bc.set_time(0.5)
    assert list(a) == list(before)
