"""Tests for FdmOrnsteinUhlenbeckOp.

# C++ parity: ql/methods/finitedifferences/operators/fdmornsteinuhlenbeckop.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``fdm_ornstein_uhlenbeck_op_apply`` section of
``migration-harness/references/cluster/w5c.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.finitedifferences.meshers.uniform_grid_mesher import (
    UniformGridMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.methods.finitedifferences.operators.fdm_ornstein_uhlenbeck_op import (
    FdmOrnsteinUhlenbeckOp,
)
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date, Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w5c")["fdm_ornstein_uhlenbeck_op_apply"]


def test_ou_op_apply_matches_cpp(reference_data: dict[str, Any]) -> None:
    """OU operator ``L @ u`` matches the C++ probe.

    TIGHT-tier: the FD coefficient build is pure floating-point
    arithmetic; the only divergence sources would be the
    ``forward_rate`` computation (same closed form in both libs) and
    the OU drift evaluation (same closed form).
    """
    n = reference_data["n"]
    layout = FdmLinearOpLayout((n,))
    mesher = UniformGridMesher(
        layout, [(reference_data["x_min"], reference_data["x_max"])]
    )
    process = OrnsteinUhlenbeckProcess(
        speed=reference_data["speed"],
        vol=reference_data["vol"],
        x0=reference_data["x0"],
        level=reference_data["level"],
    )
    today = Date.from_ymd(15, Month.May, 2026)
    rTS = FlatForward.from_rate(today, reference_data["flat_rate"], Actual365Fixed())  # noqa: N806
    op = FdmOrnsteinUhlenbeckOp(mesher, process, rTS, direction=0)
    op.set_time(reference_data["t1"], reference_data["t2"])

    u = np.array(reference_data["u"], dtype=np.float64)
    out = op.apply(u)
    expected = reference_data["apply"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_ou_op_size_is_dimensions() -> None:
    """Per C++ contract, ``size()`` returns the number of mesher dimensions.

    For the 1-D OU op this equals 1.
    """
    layout = FdmLinearOpLayout((5,))
    mesher = UniformGridMesher(layout, [(-1.0, 1.0)])
    process = OrnsteinUhlenbeckProcess(speed=0.5, vol=0.1, x0=0.0, level=0.0)
    today = Date.from_ymd(15, Month.May, 2026)
    rTS = FlatForward.from_rate(today, 0.0, Actual365Fixed())  # noqa: N806
    op = FdmOrnsteinUhlenbeckOp(mesher, process, rTS, direction=0)
    assert op.size() == 1


def test_ou_op_apply_mixed_is_zero() -> None:
    """The OU op has no mixed-derivative term — apply_mixed returns zeros."""
    layout = FdmLinearOpLayout((5,))
    mesher = UniformGridMesher(layout, [(-1.0, 1.0)])
    process = OrnsteinUhlenbeckProcess(speed=0.5, vol=0.1, x0=0.0, level=0.0)
    today = Date.from_ymd(15, Month.May, 2026)
    rTS = FlatForward.from_rate(today, 0.0, Actual365Fixed())  # noqa: N806
    op = FdmOrnsteinUhlenbeckOp(mesher, process, rTS, direction=0)
    op.set_time(0.0, 0.1)
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    out = op.apply_mixed(r)
    for actual_v in out:
        tight(float(actual_v), 0.0)


def test_ou_op_apply_direction_zero_matches_apply(reference_data: dict[str, Any]) -> None:
    """``apply_direction(0, r)`` matches ``apply(r)`` along the op's direction."""
    n = reference_data["n"]
    layout = FdmLinearOpLayout((n,))
    mesher = UniformGridMesher(
        layout, [(reference_data["x_min"], reference_data["x_max"])]
    )
    process = OrnsteinUhlenbeckProcess(
        speed=reference_data["speed"],
        vol=reference_data["vol"],
        x0=reference_data["x0"],
        level=reference_data["level"],
    )
    today = Date.from_ymd(15, Month.May, 2026)
    rTS = FlatForward.from_rate(today, reference_data["flat_rate"], Actual365Fixed())  # noqa: N806
    op = FdmOrnsteinUhlenbeckOp(mesher, process, rTS, direction=0)
    op.set_time(reference_data["t1"], reference_data["t2"])
    u = np.array(reference_data["u"], dtype=np.float64)
    a = op.apply(u)
    d = op.apply_direction(0, u)
    for av, dv in zip(a, d, strict=True):
        tight(float(dv), float(av))


def test_ou_op_solve_splitting_inverts_implicit_step(reference_data: dict[str, Any]) -> None:
    """``(I + dt * L)^{-1} (I + dt * L) r ≈ r`` round-trip."""
    n = reference_data["n"]
    layout = FdmLinearOpLayout((n,))
    mesher = UniformGridMesher(
        layout, [(reference_data["x_min"], reference_data["x_max"])]
    )
    process = OrnsteinUhlenbeckProcess(
        speed=reference_data["speed"],
        vol=reference_data["vol"],
        x0=reference_data["x0"],
        level=reference_data["level"],
    )
    today = Date.from_ymd(15, Month.May, 2026)
    rTS = FlatForward.from_rate(today, reference_data["flat_rate"], Actual365Fixed())  # noqa: N806
    op = FdmOrnsteinUhlenbeckOp(mesher, process, rTS, direction=0)
    op.set_time(reference_data["t1"], reference_data["t2"])

    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    dt = 0.001
    forward = r + dt * op.apply(r)
    inverse = op.solve_splitting(0, forward, dt)
    for actual_v, expected_v in zip(inverse, r, strict=True):
        loose(float(actual_v), float(expected_v))
