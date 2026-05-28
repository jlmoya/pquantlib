"""Tests for UniformGridMesher (multi-D uniform mesher).

# C++ parity: ql/methods/finitedifferences/meshers/uniformgridmesher.{hpp,cpp}
# @ v1.42.1.

Cross-validates the 1-D path against ``uniform_grid_mesher_1d`` in
``migration-harness/references/cluster/l5d.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.uniform_grid_mesher import (
    UniformGridMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l5d")


def _build_mesh_1d() -> UniformGridMesher:
    layout = FdmLinearOpLayout((11,))
    return UniformGridMesher(layout, [(-2.0, 2.0)])


def test_locations_1d_match_reference(reference_data: dict[str, Any]) -> None:
    m = _build_mesh_1d()
    locs = m.locations(0)
    expected = reference_data["uniform_grid_mesher_1d"]["locations"]
    # TIGHT (not EXACT): C++ at -O3 may fuse ``start + j * dx`` into
    # FMA, producing ULP-level differences at some indices. See
    # ``test_uniform_1d_mesher::test_locations_match_reference`` for
    # the full rationale.
    for actual_v, expected_v in zip(locs, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_location_at_idx_5(reference_data: dict[str, Any]) -> None:
    m = _build_mesh_1d()
    # Build an iterator at coord 5 (index 5 in 1-D).
    iter_ = next(it for it in m.layout().iter() if it.index == 5)
    tight(
        m.location(iter_, 0),
        float(reference_data["uniform_grid_mesher_1d"]["location_at_idx5"]),
    )


def test_dplus_at_idx_5(reference_data: dict[str, Any]) -> None:
    m = _build_mesh_1d()
    iter_ = next(it for it in m.layout().iter() if it.index == 5)
    tight(
        m.dplus(iter_, 0),
        float(reference_data["uniform_grid_mesher_1d"]["dplus_at_idx5"]),
    )


def test_dminus_at_idx_5(reference_data: dict[str, Any]) -> None:
    m = _build_mesh_1d()
    iter_ = next(it for it in m.layout().iter() if it.index == 5)
    tight(
        m.dminus(iter_, 0),
        float(reference_data["uniform_grid_mesher_1d"]["dminus_at_idx5"]),
    )


def test_boundaries_size_mismatch_raises() -> None:
    """Layout has 1 axis but 2 boundary pairs given — must raise."""
    layout = FdmLinearOpLayout((5,))
    with pytest.raises(LibraryException):
        UniformGridMesher(layout, [(-1.0, 1.0), (0.0, 2.0)])
