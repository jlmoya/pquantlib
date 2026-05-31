"""Tests for FdmExtendedOrnsteinUhlenbeckOp.

# C++ parity: ql/experimental/finitedifferences/fdmextendedornsteinuhlenbeckop.hpp.

Reference values: migration-harness/references/cluster/w5a.json.

Tolerance: TIGHT for the apply / solve_splitting outputs — they
should match C++ to floating-point precision since the discretisation
is identical (triple-band stencils built from the same mesher).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.finitedifferences.fdm_extended_ornstein_uhlenbeck_op import (
    FdmExtendedOrnsteinUhlenbeckOp,
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


def _make_op() -> tuple[FdmMesherComposite, FdmExtendedOrnsteinUhlenbeckOp]:
    """Match C++ probe: speed=1, sigma=0.3, x0=0, b(t)=0; r=5% flat forward.

    Mesher: Uniform1d([-2, 2], 9 nodes).
    """
    process = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    m1 = Uniform1dMesher(-2.0, 2.0, 9)
    mesher = FdmMesherComposite(m1)
    today = Date.from_ymd(15, Month.January, 2024)
    r_ts = FlatForward.from_rate(today, 0.05, Actual365Fixed())
    op = FdmExtendedOrnsteinUhlenbeckOp(mesher, process, r_ts)
    op.set_time(0.0, 0.5)
    return mesher, op


def test_ext_ou_op_const_apply(refs: dict[str, Any]) -> None:
    """L @ 1 = -r everywhere at interior nodes.

    TIGHT: 1-D triple-band apply on constant vector should match
    C++ floating-point exactly (same multiplications + adds).
    """
    _, op = _make_op()
    ones = np.ones(9, dtype=np.float64)
    out = op.apply(ones)
    tolerance.tight(float(out[0]), refs["ou_op_const_apply_0"])
    tolerance.tight(float(out[3]), refs["ou_op_const_apply_3"])
    tolerance.tight(float(out[4]), refs["ou_op_const_apply_4"])
    tolerance.tight(float(out[5]), refs["ou_op_const_apply_5"])
    tolerance.tight(float(out[8]), refs["ou_op_const_apply_8"])


def test_ext_ou_op_linear_apply(refs: dict[str, Any]) -> None:
    """L @ x — captures the drift + rate term + second-derivative on the linear.

    TIGHT.
    """
    mesher, op = _make_op()
    locations = mesher.locations(0)
    out = op.apply(locations.astype(np.float64))
    tolerance.tight(float(out[3]), refs["ou_op_lin_apply_3"])
    tolerance.tight(float(out[4]), refs["ou_op_lin_apply_4"])
    tolerance.tight(float(out[5]), refs["ou_op_lin_apply_5"])


def test_ext_ou_op_quadratic_apply(refs: dict[str, Any]) -> None:
    """L @ x^2 — exercises the second-derivative term.

    TIGHT.
    """
    mesher, op = _make_op()
    locations = mesher.locations(0)
    out = op.apply((locations * locations).astype(np.float64))
    tolerance.tight(float(out[3]), refs["ou_op_quad_apply_3"])
    tolerance.tight(float(out[4]), refs["ou_op_quad_apply_4"])
    tolerance.tight(float(out[5]), refs["ou_op_quad_apply_5"])


def test_ext_ou_op_solve_splitting(refs: dict[str, Any]) -> None:
    """solve_splitting (a=0.1) on ones.

    TIGHT: tridiagonal Thomas solver should match C++ exactly.
    """
    _, op = _make_op()
    ones = np.ones(9, dtype=np.float64)
    solved = op.solve_splitting(0, ones, 0.1)
    tolerance.tight(float(solved[4]), refs["ou_op_solve_splitting_4"])


def test_ext_ou_op_apply_mixed_is_zero() -> None:
    """apply_mixed returns zeros (no cross-direction term).

    # C++ parity: ``apply_mixed`` returns ``Array(r.size(), 0.0)``.
    """
    _, op = _make_op()
    ones = np.ones(9, dtype=np.float64)
    out = op.apply_mixed(ones)
    assert np.allclose(out, 0.0)


def test_ext_ou_op_apply_direction_off_axis_is_zero() -> None:
    """apply_direction along direction != active returns zeros.

    # C++ parity: ``apply_direction``.
    """
    _, op = _make_op()
    ones = np.ones(9, dtype=np.float64)
    out = op.apply_direction(1, ones)  # active dir is 0
    assert np.allclose(out, 0.0)
