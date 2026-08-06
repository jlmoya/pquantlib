"""Cross-validation of ``FdmTimeDepDirichletBoundary`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmtimedepdirichletboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_time_dep_dirichlet_boundary import (
    FdmTimeDepDirichletBoundary,
)
from pquantlib.testing.tolerance import tight

from .conftest import as_floats, ramp


def _affine_of_time(t: float) -> float:
    """# C++ parity: the probe's ``AffineOfTime`` functor ``1 + 2t``."""
    return 1.0 + 2.0 * t


def _vector_of_time(t: float) -> Array:
    """# C++ parity: the probe's ``VectorOfTime{5}`` functor ``(i+1) * t``."""
    return np.array([(i + 1.0) * t for i in range(5)], dtype=np.float64)


@pytest.mark.parametrize(
    ("t", "key"),
    [(0.3, "time_dep_scalar_d0_lower_t03"), (0.75, "time_dep_scalar_d0_lower_t075")],
)
def test_scalar_boundary_function(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite, t: float, key: str
) -> None:
    bc = FdmTimeDepDirichletBoundary(
        small_mesher, 0, BoundaryConditionSide.LOWER, value_on_boundary=_affine_of_time
    )
    bc.set_time(t)
    a = ramp(small_mesher.layout().size())
    # ``applyAfterSolving`` forwards to ``applyAfterApplying`` in C++, so the
    # probe used one for each t; both are exercised here through the same path.
    bc.apply_after_applying(a)
    for actual_v, expected_v in zip(a, as_floats(reference_data[key]), strict=True):
        tight(float(actual_v), expected_v)


def test_apply_after_solving_matches_apply_after_applying(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite
) -> None:
    bc = FdmTimeDepDirichletBoundary(
        small_mesher, 0, BoundaryConditionSide.LOWER, value_on_boundary=_affine_of_time
    )
    bc.set_time(0.75)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_solving(a)
    expected = as_floats(reference_data["time_dep_scalar_d0_lower_t075"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_vector_boundary_function(
    reference_data: dict[str, Any], small_mesher: FdmMesherComposite
) -> None:
    bc = FdmTimeDepDirichletBoundary(
        small_mesher, 1, BoundaryConditionSide.UPPER, values_on_boundary=_vector_of_time
    )
    bc.set_time(0.4)
    a = ramp(small_mesher.layout().size())
    bc.apply_after_applying(a)
    expected = as_floats(reference_data["time_dep_vector_d1_upper_t04"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_size_mismatch_rejected(small_mesher: FdmMesherComposite) -> None:
    """# C++ parity: ``QL_REQUIRE(indices_.size() == values_.size(), ...)``."""
    bc = FdmTimeDepDirichletBoundary(
        small_mesher,
        1,
        BoundaryConditionSide.UPPER,
        values_on_boundary=lambda t: np.array([t, 2.0 * t], dtype=np.float64),
    )
    bc.set_time(0.4)
    with pytest.raises(LibraryException, match="does not match hypersurface size"):
        bc.apply_after_applying(ramp(small_mesher.layout().size()))


def test_exactly_one_boundary_function_required(small_mesher: FdmMesherComposite) -> None:
    """Python guard replacing C++'s "which ``std::function`` is engaged?" test."""
    with pytest.raises(LibraryException):
        FdmTimeDepDirichletBoundary(small_mesher, 0, BoundaryConditionSide.LOWER)
    with pytest.raises(LibraryException):
        FdmTimeDepDirichletBoundary(
            small_mesher,
            0,
            BoundaryConditionSide.LOWER,
            value_on_boundary=_affine_of_time,
            values_on_boundary=_vector_of_time,
        )
