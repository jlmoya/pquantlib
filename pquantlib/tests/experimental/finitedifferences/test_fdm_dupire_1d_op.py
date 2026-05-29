"""Tests for FdmDupire1dOp.

# C++ parity: ql/experimental/finitedifferences/fdmdupire1dop.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``fdm_dupire_1d_op_apply`` section of
``migration-harness/references/cluster/w5c.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.experimental.finitedifferences.fdm_dupire_1d_op import FdmDupire1dOp
from pquantlib.methods.finitedifferences.meshers.uniform_grid_mesher import (
    UniformGridMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w5c")["fdm_dupire_1d_op_apply"]


def test_dupire_apply_matches_cpp(reference_data: dict[str, Any]) -> None:
    """Stencil-level parity: ``L @ u`` matches the C++ probe.

    TIGHT-tier: the second-derivative coefficients and the local-vol
    scaling are pure floating-point arithmetic with no transcendentals
    or path dependence — the Python operator should agree with C++ to
    TIGHT (1e-14 abs).
    """
    n = reference_data["n"]
    layout = FdmLinearOpLayout((n,))
    mesher = UniformGridMesher(layout, [(reference_data["s_min"], reference_data["s_max"])])
    local_vol = np.array(reference_data["local_vol"], dtype=np.float64)
    op = FdmDupire1dOp(mesher, local_vol)

    u = np.array(reference_data["u"], dtype=np.float64)
    out = op.apply(u)
    expected = reference_data["apply"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_dupire_size_is_one() -> None:
    """1-D Dupire op has size 1 per C++ contract."""
    layout = FdmLinearOpLayout((5,))
    mesher = UniformGridMesher(layout, [(80.0, 120.0)])
    local_vol = np.full(5, 0.2, dtype=np.float64)
    op = FdmDupire1dOp(mesher, local_vol)
    assert op.size() == 1


def test_dupire_set_time_is_noop() -> None:
    """``set_time`` is a no-op (the constructor captures the vol slice)."""
    layout = FdmLinearOpLayout((5,))
    mesher = UniformGridMesher(layout, [(80.0, 120.0)])
    local_vol = np.full(5, 0.2, dtype=np.float64)
    op = FdmDupire1dOp(mesher, local_vol)
    before = op.apply(np.ones(5, dtype=np.float64))
    op.set_time(0.0, 1.0)
    after = op.apply(np.ones(5, dtype=np.float64))
    for b, a in zip(before, after, strict=True):
        tight(float(a), float(b))


def test_dupire_apply_direction_zero_matches_apply() -> None:
    """``apply_direction(0, r)`` should equal ``apply(r)`` for the 1-D op."""
    layout = FdmLinearOpLayout((5,))
    mesher = UniformGridMesher(layout, [(80.0, 120.0)])
    local_vol = np.full(5, 0.2, dtype=np.float64)
    op = FdmDupire1dOp(mesher, local_vol)
    r = np.array([1.0, 2.5, 3.0, 5.5, 7.0], dtype=np.float64)
    a = op.apply(r)
    d = op.apply_direction(0, r)
    for av, dv in zip(a, d, strict=True):
        tight(float(dv), float(av))


def test_dupire_apply_direction_one_raises() -> None:
    """``apply_direction(1, r)`` raises on a 1-D op (no second direction)."""
    layout = FdmLinearOpLayout((5,))
    mesher = UniformGridMesher(layout, [(80.0, 120.0)])
    local_vol = np.full(5, 0.2, dtype=np.float64)
    op = FdmDupire1dOp(mesher, local_vol)
    with pytest.raises(ValueError, match="direction too large"):
        op.apply_direction(1, np.zeros(5, dtype=np.float64))


def test_dupire_solve_splitting_inverts_implicit_step() -> None:
    """``(I + dt * L)^{-1} (I + dt * L) r == r`` to TIGHT tolerance."""
    layout = FdmLinearOpLayout((11,))
    mesher = UniformGridMesher(layout, [(80.0, 120.0)])
    local_vol = np.full(11, 0.2, dtype=np.float64)
    op = FdmDupire1dOp(mesher, local_vol)
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float64)
    dt = 0.01
    forward = r + dt * op.apply(r)
    inverse = op.solve_splitting(0, forward, dt)
    # LOOSE-tier: Thomas algorithm round-trip accumulates O(N * dt^2)
    # rounding error; LOOSE (1e-8 abs/rel) is the right gate.
    for actual_v, expected_v in zip(inverse, r, strict=True):
        loose(float(actual_v), float(expected_v))
