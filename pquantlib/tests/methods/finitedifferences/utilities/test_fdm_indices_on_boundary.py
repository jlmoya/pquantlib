"""Cross-validation of ``FdmIndicesOnBoundary`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmindicesonboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).

The indices are integers, so the assertions are plain equality — the EXACT
tier by construction, no floating-point tolerance involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.methods.finitedifferences.utilities.fdm_indices_on_boundary import (
    FdmIndicesOnBoundary,
)


@pytest.mark.parametrize(
    ("dim", "direction", "side", "key"),
    [
        ((5, 4), 0, BoundaryConditionSide.LOWER, "indices_2d_d0_lower"),
        ((5, 4), 0, BoundaryConditionSide.UPPER, "indices_2d_d0_upper"),
        ((5, 4), 1, BoundaryConditionSide.LOWER, "indices_2d_d1_lower"),
        ((5, 4), 1, BoundaryConditionSide.UPPER, "indices_2d_d1_upper"),
        ((3, 2, 4), 1, BoundaryConditionSide.LOWER, "indices_3d_d1_lower"),
        ((3, 2, 4), 1, BoundaryConditionSide.UPPER, "indices_3d_d1_upper"),
        ((3, 2, 4), 2, BoundaryConditionSide.UPPER, "indices_3d_d2_upper"),
        ((6,), 0, BoundaryConditionSide.LOWER, "indices_1d_lower"),
        ((6,), 0, BoundaryConditionSide.UPPER, "indices_1d_upper"),
    ],
)
def test_indices_match_cpp(
    reference_data: dict[str, Any],
    dim: tuple[int, ...],
    direction: int,
    side: BoundaryConditionSide,
    key: str,
) -> None:
    layout = FdmLinearOpLayout(dim)
    indices = FdmIndicesOnBoundary(layout, direction, side).get_indices()
    assert list(indices) == reference_data[key]


def test_hypersurface_size_is_layout_size_over_direction_dim() -> None:
    """The boundary of direction ``d`` holds ``size / dim[d]`` nodes.

    # C++ parity: ``hyperSize = accumulate(newDim, 1, multiplies<>())`` with
    # ``newDim[direction] = 1``.
    """
    layout = FdmLinearOpLayout((3, 2, 4))
    for direction in range(3):
        expected = layout.size() // layout.dim()[direction]
        for side in (BoundaryConditionSide.LOWER, BoundaryConditionSide.UPPER):
            assert len(FdmIndicesOnBoundary(layout, direction, side).get_indices()) == expected
