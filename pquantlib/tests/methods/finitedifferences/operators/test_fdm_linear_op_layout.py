"""Tests for FdmLinearOpLayout (1-D + multi-D index layout).

# C++ parity: ql/methods/finitedifferences/operators/fdmlinearoplayout.{hpp,cpp}
# @ v1.42.1.

Cross-validates the 1-D layout against ``linear_op_layout_1d``
section of ``migration-harness/references/cluster/l5d.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.testing.reference_reader import load as load_reference


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l5d")


def test_layout_size_1d(reference_data: dict[str, Any]) -> None:
    layout = FdmLinearOpLayout((8,))
    assert layout.size() == reference_data["linear_op_layout_1d"]["size"]


def test_layout_dim_1d(reference_data: dict[str, Any]) -> None:
    layout = FdmLinearOpLayout((8,))
    assert list(layout.dim()) == reference_data["linear_op_layout_1d"]["dim"]


def test_layout_spacing_1d(reference_data: dict[str, Any]) -> None:
    layout = FdmLinearOpLayout((8,))
    assert list(layout.spacing()) == reference_data["linear_op_layout_1d"]["spacing"]


def test_layout_index_at_3_1d(reference_data: dict[str, Any]) -> None:
    layout = FdmLinearOpLayout((8,))
    assert layout.index((3,)) == reference_data["linear_op_layout_1d"]["index_at_3"]


def test_layout_iter_yields_in_row_major_order() -> None:
    """1-D iter just walks the flat range."""
    # EXACT: integer iteration order — bit-identical via int equality.
    layout = FdmLinearOpLayout((4,))
    items = [(it.index, it.coordinates[0]) for it in layout.iter()]
    assert items == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_layout_iter_2d_axis_0_fastest() -> None:
    """Multi-D iter walks axis 0 fastest."""
    layout = FdmLinearOpLayout((3, 2))
    coords = [it.coordinates for it in layout.iter()]
    assert coords == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_layout_neighbourhood_clamps_at_boundary() -> None:
    layout = FdmLinearOpLayout((5,))
    first = next(layout.iter())
    # Going below 0 clamps to 0.
    assert layout.neighbourhood(first, 0, -1) == 0
    # Going up by 1 from index 0 is index 1.
    assert layout.neighbourhood(first, 0, +1) == 1


def test_layout_index_round_trip() -> None:
    """For 1-D, index((i,)) == i. EXACT via int equality."""
    layout = FdmLinearOpLayout((10,))
    for i in range(10):
        assert layout.index((i,)) == i
