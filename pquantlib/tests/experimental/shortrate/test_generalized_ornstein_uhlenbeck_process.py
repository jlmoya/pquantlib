"""Tests for GeneralizedOrnsteinUhlenbeckProcess.

Cross-validates against ``migration-harness/references/cluster/w8d.json``.

C++ parity: ql/experimental/shortrate/generalizedornsteinuhlenbeckprocess.{hpp,cpp}
@ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.shortrate.generalized_ornstein_uhlenbeck_process import (
    GeneralizedOrnsteinUhlenbeckProcess,
)
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8d")


# Constant coefficients used in the probe.
_A, _SIG, _X0, _LEVEL = 0.3, 0.15, 0.05, 0.04


def _const_process() -> GeneralizedOrnsteinUhlenbeckProcess:
    return GeneralizedOrnsteinUhlenbeckProcess(
        lambda _t: _A, lambda _t: _SIG, _X0, _LEVEL
    )


def test_const_inspectors(ref: dict[str, Any]) -> None:
    g = _const_process()
    tight(g.x0(), ref["gou_const_x0"])
    tight(g.speed(0.7), ref["gou_const_speed"])
    tight(g.volatility(0.7), ref["gou_const_vol"])
    tight(g.level(), ref["gou_const_level"])
    tight(g.drift_1d(0.7, 0.06), ref["gou_const_drift"])
    tight(g.diffusion_1d(0.7, 0.06), ref["gou_const_diffusion"])


def test_const_expectation_variance(ref: dict[str, Any]) -> None:
    g = _const_process()
    tight(g.expectation_1d(0.0, _X0, 1.5), ref["gou_const_expectation"])
    tight(g.variance_1d(0.0, _X0, 1.5), ref["gou_const_variance"])
    tight(g.std_deviation_1d(0.0, _X0, 1.5), ref["gou_const_stddev"])


def test_const_equals_plain_ou(ref: dict[str, Any]) -> None:
    """Constant coefficients reproduce the plain OrnsteinUhlenbeckProcess.

    EXACT-tier: both code paths evaluate the identical closed form with
    the same scalar constants, so the results are bit-identical.
    """
    g = _const_process()
    ou = OrnsteinUhlenbeckProcess(_A, _SIG, _X0, _LEVEL)
    # bit-identical to the plain OU
    assert g.expectation_1d(0.0, _X0, 1.5) == ou.expectation_1d(0.0, _X0, 1.5)
    assert g.variance_1d(0.0, _X0, 1.5) == ou.variance_1d(0.0, _X0, 1.5)
    # and both reproduce the C++ probe (one anchor value each)
    tight(ou.expectation_1d(0.0, _X0, 1.5), ref["ou_expectation"])
    tight(ou.variance_1d(0.0, _X0, 1.5), ref["ou_variance"])


def test_time_varying_coefficients(ref: dict[str, Any]) -> None:
    """Linear-in-t speed / vol — coefficients evaluated at the step start."""
    g = GeneralizedOrnsteinUhlenbeckProcess(
        lambda t: 0.2 + 0.1 * t, lambda t: 0.10 + 0.05 * t, _X0, _LEVEL
    )
    tight(g.speed(2.0), ref["gou_tv_speed_t2"])
    tight(g.volatility(2.0), ref["gou_tv_vol_t2"])
    tight(g.drift_1d(2.0, 0.06), ref["gou_tv_drift"])
    tight(g.diffusion_1d(2.0, 0.06), ref["gou_tv_diffusion"])
    tight(g.expectation_1d(2.0, _X0, 0.5), ref["gou_tv_expectation"])
    tight(g.variance_1d(2.0, _X0, 0.5), ref["gou_tv_variance"])
    tight(g.std_deviation_1d(2.0, _X0, 0.5), ref["gou_tv_stddev"])


def test_small_speed_algebraic_limit(ref: dict[str, Any]) -> None:
    """``a(t0) -> 0`` falls back to the algebraic variance ``sigma^2 dt``."""
    g = GeneralizedOrnsteinUhlenbeckProcess(
        lambda _t: 0.0, lambda _t: 0.2, _X0, _LEVEL
    )
    tight(g.variance_1d(0.0, _X0, 2.0), ref["gou_smallspeed_variance"])


def test_rejects_negative_x0() -> None:
    with pytest.raises(LibraryException):
        GeneralizedOrnsteinUhlenbeckProcess(lambda _t: 0.3, lambda _t: 0.15, -1.0, 0.0)


def test_rejects_negative_level() -> None:
    with pytest.raises(LibraryException):
        GeneralizedOrnsteinUhlenbeckProcess(lambda _t: 0.3, lambda _t: 0.15, 0.0, -1.0)
