"""Tests for FdmExtOUJumpOp.

# C++ parity: ql/experimental/finitedifferences/fdmextoujumpop.hpp.

Reference values: migration-harness/references/cluster/w5a.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_op import (
    FdmExtOUJumpOp,
)
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date, Month


@pytest.fixture(scope="module")
def refs() -> dict[str, Any]:
    return reference_reader.load("cluster/w5a")


@pytest.fixture(scope="module")
def jump_op_2d() -> tuple[FdmMesherComposite, FdmExtOUJumpOp]:
    """Match C++ probe: ExtOU (1.0, 0.3, 0, b=0) + jumps (Y0=0, beta=4, lam=2, eta=4).

    Mesh: Uniform1d([-2, 2], 7) x Uniform1d([0, 2], 5).
    """
    ou_process = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    kluge = ExtOUWithJumpsProcess(ou_process, y0=0.0, beta=4.0, jump_intensity=2.0, eta=4.0)

    mx = Uniform1dMesher(-2.0, 2.0, 7)
    my = Uniform1dMesher(0.0, 2.0, 5)
    mesher = FdmMesherComposite(mx, my)

    today = Date.from_ymd(15, Month.January, 2024)
    r_ts = FlatForward.from_rate(today, 0.05, Actual365Fixed())

    op = FdmExtOUJumpOp(mesher, kluge, r_ts, integro_integration_order=16)
    op.set_time(0.0, 0.5)
    return mesher, op


def test_jump_op_apply_const(
    refs: dict[str, Any], jump_op_2d: tuple[FdmMesherComposite, FdmExtOUJumpOp]
) -> None:
    """apply(ones) at four sample positions.

    TIGHT: discretisation matches C++ (Gauss-Laguerre nodes via
    scipy.special.roots_laguerre — same nodes as QL's
    GaussLaguerreIntegration since both use Golub-Welsch on the
    Laguerre 3-term recurrence).
    """
    mesher, op = jump_op_2d
    n = mesher.layout().size()
    ones = np.ones(n, dtype=np.float64)
    out = op.apply(ones)
    tolerance.tight(float(out[0]), refs["jump_op_const_apply_0"])
    tolerance.tight(float(out[n // 4]), refs["jump_op_const_apply_size_4"])
    tolerance.tight(float(out[n // 2]), refs["jump_op_const_apply_size_2"])
    tolerance.tight(float(out[3 * n // 4]), refs["jump_op_const_apply_size_3_4"])


def test_jump_op_apply_direction_y(
    refs: dict[str, Any], jump_op_2d: tuple[FdmMesherComposite, FdmExtOUJumpOp]
) -> None:
    """apply_direction(1, ones) — pure dyMap contribution.

    TIGHT.
    """
    mesher, op = jump_op_2d
    n = mesher.layout().size()
    ones = np.ones(n, dtype=np.float64)
    out = op.apply_direction(1, ones)
    tolerance.tight(float(out[0]), refs["jump_op_const_apply_dir1_0"])
    tolerance.tight(float(out[n // 2]), refs["jump_op_const_apply_dir1_size_2"])


def test_jump_op_apply_mixed(
    refs: dict[str, Any], jump_op_2d: tuple[FdmMesherComposite, FdmExtOUJumpOp]
) -> None:
    """apply_mixed(ones) — pure integro contribution.

    TIGHT.
    """
    mesher, op = jump_op_2d
    n = mesher.layout().size()
    ones = np.ones(n, dtype=np.float64)
    out = op.apply_mixed(ones)
    tolerance.tight(float(out[0]), refs["jump_op_const_apply_mixed_0"])
    tolerance.tight(float(out[n // 2]), refs["jump_op_const_apply_mixed_size_2"])


def test_jump_op_size(
    jump_op_2d: tuple[FdmMesherComposite, FdmExtOUJumpOp],
) -> None:
    """``size()`` matches the mesh dimensionality."""
    _, op = jump_op_2d
    assert op.size() == 2


def test_jump_op_solve_splitting_other_dirs(
    jump_op_2d: tuple[FdmMesherComposite, FdmExtOUJumpOp],
) -> None:
    """solve_splitting on unknown directions returns r unchanged.

    # C++ parity.
    """
    mesher, op = jump_op_2d
    n = mesher.layout().size()
    r = np.arange(n, dtype=np.float64)
    out = op.solve_splitting(5, r, 0.1)
    assert np.array_equal(out, r)
