"""Tests for FdmKlugeExtOUOp.

# C++ parity: ql/experimental/finitedifferences/fdmklugeextouop.hpp.

Reference values: migration-harness/references/cluster/w5a.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.finitedifferences.fdm_kluge_ext_ou_op import (
    FdmKlugeExtOUOp,
)
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
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
def kluge_op_3d() -> tuple[FdmMesherComposite, FdmKlugeExtOUOp]:
    """Match C++ probe: rho=0.4, sigma_x=0.3, sigma_u=0.25.

    Mesh: Uniform1d([-2, 2], 5) x Uniform1d([0, 2], 4) x Uniform1d([-2, 2], 5).
    """
    ext_ou_a = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    kluge = ExtOUWithJumpsProcess(ext_ou_a, y0=0.0, beta=4.0, jump_intensity=2.0, eta=4.0)
    ext_ou_b = ExtendedOrnsteinUhlenbeckProcess(0.5, 0.25, 0.0, lambda _t: 0.0)
    kluge_ext_ou = KlugeExtOUProcess(0.4, kluge, ext_ou_b)

    mx = Uniform1dMesher(-2.0, 2.0, 5)
    my = Uniform1dMesher(0.0, 2.0, 4)
    mu = Uniform1dMesher(-2.0, 2.0, 5)
    mesher = FdmMesherComposite(mx, my, mu)

    today = Date.from_ymd(15, Month.January, 2024)
    r_ts = FlatForward.from_rate(today, 0.05, Actual365Fixed())

    op = FdmKlugeExtOUOp(mesher, kluge_ext_ou, r_ts, integro_integration_order=16)
    op.set_time(0.0, 0.5)
    return mesher, op


def test_kluge_op_apply_const(
    refs: dict[str, Any], kluge_op_3d: tuple[FdmMesherComposite, FdmKlugeExtOUOp]
) -> None:
    """apply(ones) at four sample positions.

    TIGHT.
    """
    mesher, op = kluge_op_3d
    n = mesher.layout().size()
    ones = np.ones(n, dtype=np.float64)
    out = op.apply(ones)
    tolerance.tight(float(out[0]), refs["kluge_op_const_apply_0"])
    tolerance.tight(float(out[n // 4]), refs["kluge_op_const_apply_size_4"])
    tolerance.tight(float(out[n // 2]), refs["kluge_op_const_apply_size_2"])
    tolerance.tight(float(out[3 * n // 4]), refs["kluge_op_const_apply_size_3_4"])


def test_kluge_op_apply_linear_u(
    refs: dict[str, Any], kluge_op_3d: tuple[FdmMesherComposite, FdmKlugeExtOUOp]
) -> None:
    """apply(r) where r = u-axis location at each node.

    TIGHT.
    """
    mesher, op = kluge_op_3d
    n = mesher.layout().size()
    r = np.empty(n, dtype=np.float64)
    for iter_ in mesher.layout().iter():
        r[iter_.index] = mesher.location(iter_, 2)
    out = op.apply(r)
    tolerance.tight(float(out[n // 2]), refs["kluge_op_lin_apply_size_2"])


def test_kluge_op_apply_direction_u(
    refs: dict[str, Any], kluge_op_3d: tuple[FdmMesherComposite, FdmKlugeExtOUOp]
) -> None:
    """apply_direction(2, ones) — pure ouOp contribution along U axis.

    TIGHT.
    """
    mesher, op = kluge_op_3d
    n = mesher.layout().size()
    ones = np.ones(n, dtype=np.float64)
    out = op.apply_direction(2, ones)
    tolerance.tight(float(out[n // 2]), refs["kluge_op_const_apply_dir2_size_2"])


def test_kluge_op_size(
    kluge_op_3d: tuple[FdmMesherComposite, FdmKlugeExtOUOp],
) -> None:
    """size() = 3."""
    _, op = kluge_op_3d
    assert op.size() == 3
