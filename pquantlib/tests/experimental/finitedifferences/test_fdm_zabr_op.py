"""Tests for FdmZabrOp.

# C++ parity: ql/experimental/finitedifferences/fdmzabrop.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``fdm_zabr_op_apply`` section of
``migration-harness/references/cluster/w5c.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.experimental.finitedifferences.fdm_zabr_op import FdmZabrOp
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
    return load_reference("cluster/w5c")["fdm_zabr_op_apply"]


def _build_mesh(reference_data: dict[str, Any]) -> UniformGridMesher:
    layout = FdmLinearOpLayout((reference_data["n_fwd"], reference_data["n_vol"]))
    return UniformGridMesher(
        layout,
        [
            (reference_data["fwd_min"], reference_data["fwd_max"]),
            (reference_data["vol_min"], reference_data["vol_max"]),
        ],
    )


def test_zabr_apply_full_matches_cpp(reference_data: dict[str, Any]) -> None:
    """Full apply ``L @ u`` matches the C++ probe stencil.

    TIGHT-tier: the ZABR coefficients are products of locations and
    constants — bit-identical with C++ assuming numpy's ``power`` and
    C++'s ``std::pow`` agree on those inputs (they do, both call libm).
    """
    mesh = _build_mesh(reference_data)
    op = FdmZabrOp(
        mesh,
        reference_data["beta"],
        reference_data["nu"],
        reference_data["rho"],
        reference_data["gamma"],
    )
    u = np.array(reference_data["u"], dtype=np.float64)
    out = op.apply(u)
    expected = reference_data["apply"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_zabr_apply_direction_x_matches_cpp(reference_data: dict[str, Any]) -> None:
    """Direction-0 (forward) apply matches the C++ probe."""
    mesh = _build_mesh(reference_data)
    op = FdmZabrOp(
        mesh,
        reference_data["beta"],
        reference_data["nu"],
        reference_data["rho"],
        reference_data["gamma"],
    )
    u = np.array(reference_data["u"], dtype=np.float64)
    out = op.apply_direction(0, u)
    expected = reference_data["apply_dx"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_zabr_apply_direction_y_matches_cpp(reference_data: dict[str, Any]) -> None:
    """Direction-1 (vol) apply matches the C++ probe."""
    mesh = _build_mesh(reference_data)
    op = FdmZabrOp(
        mesh,
        reference_data["beta"],
        reference_data["nu"],
        reference_data["rho"],
        reference_data["gamma"],
    )
    u = np.array(reference_data["u"], dtype=np.float64)
    out = op.apply_direction(1, u)
    expected = reference_data["apply_dy"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_zabr_apply_mixed_matches_cpp(reference_data: dict[str, Any]) -> None:
    """Mixed-derivative apply matches the C++ probe."""
    mesh = _build_mesh(reference_data)
    op = FdmZabrOp(
        mesh,
        reference_data["beta"],
        reference_data["nu"],
        reference_data["rho"],
        reference_data["gamma"],
    )
    u = np.array(reference_data["u"], dtype=np.float64)
    out = op.apply_mixed(u)
    expected = reference_data["apply_mixed"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_zabr_size_is_two() -> None:
    """ZABR op has 2 directions per C++ contract."""
    layout = FdmLinearOpLayout((5, 4))
    mesher = UniformGridMesher(layout, [(0.02, 0.08), (0.01, 0.05)])
    op = FdmZabrOp(mesher, beta=0.5, nu=0.4, rho=-0.2, gamma=0.7)
    assert op.size() == 2


def test_zabr_apply_decomposes_as_dx_plus_dy_plus_dxy(reference_data: dict[str, Any]) -> None:
    """``apply == apply_direction(0) + apply_direction(1) + apply_mixed``."""
    mesh = _build_mesh(reference_data)
    op = FdmZabrOp(
        mesh,
        reference_data["beta"],
        reference_data["nu"],
        reference_data["rho"],
        reference_data["gamma"],
    )
    u = np.array(reference_data["u"], dtype=np.float64)
    full = op.apply(u)
    decomposed = op.apply_direction(0, u) + op.apply_direction(1, u) + op.apply_mixed(u)
    for actual_v, expected_v in zip(decomposed, full, strict=True):
        tight(float(actual_v), float(expected_v))


def test_zabr_apply_direction_out_of_range_raises() -> None:
    """Direction > 1 raises (only 2 directions exist)."""
    layout = FdmLinearOpLayout((5, 4))
    mesher = UniformGridMesher(layout, [(0.02, 0.08), (0.01, 0.05)])
    op = FdmZabrOp(mesher, beta=0.5, nu=0.4, rho=-0.2, gamma=0.7)
    u = np.zeros(20, dtype=np.float64)
    with pytest.raises(ValueError, match="direction too large"):
        op.apply_direction(2, u)
