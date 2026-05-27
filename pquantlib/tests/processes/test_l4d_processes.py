"""Tests for the L4-D process layer.

Covers OrnsteinUhlenbeckProcess + CoxIngersollRossProcess (1-D),
G2Process + G2ForwardProcess (multi-D), and HullWhiteForwardProcess.

Cross-validates against ``migration-harness/references/cluster/l4d.json``.

C++ parity: ql/processes/{ornsteinuhlenbeckprocess,coxingersollrossprocess,
            g2process,hullwhiteprocess,forwardmeasureprocess}.{hpp,cpp}
            @ v1.42.1 (099987f0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
from pquantlib.processes.forward_measure_process import (
    ForwardMeasureProcess,
    ForwardMeasureProcess1D,
)
from pquantlib.processes.g2_forward_process import G2ForwardProcess
from pquantlib.processes.g2_process import G2Process
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4d")


@pytest.fixture(scope="module")
def flat_5_curve() -> FlatForward:
    """5% Act/365F flat-forward curve at 15-Jun-2026 (matches the probe)."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    return FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)


# --- helpers ---------------------------------------------------------------


def _tight_arr(actual: np.ndarray, expected: list[Any]) -> None:
    """Element-wise TIGHT check for an Array (1-D or 2-D)."""
    arr = np.array(expected, dtype=np.float64)
    assert actual.shape == arr.shape, f"shape {actual.shape} != {arr.shape}"
    flat_a = actual.ravel()
    flat_e = arr.ravel()
    for a, e in zip(flat_a, flat_e, strict=True):
        tight(float(a), float(e))


# --- OrnsteinUhlenbeckProcess ----------------------------------------------


class TestOrnsteinUhlenbeckProcess:
    """Cross-validate the OU process closed-forms against the C++ probe."""

    @pytest.fixture
    def ou(self) -> OrnsteinUhlenbeckProcess:
        return OrnsteinUhlenbeckProcess(0.3, 0.05, 0.02, 0.0)

    def test_is_stochastic_process_1d(self, ou: OrnsteinUhlenbeckProcess) -> None:
        assert isinstance(ou, StochasticProcess1D)

    def test_inspectors(
        self, ou: OrnsteinUhlenbeckProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["ornstein_uhlenbeck"]
        tight(ou.x0(), ref["x0"])
        tight(ou.speed(), ref["speed"])
        tight(ou.volatility(), ref["volatility"])
        tight(ou.level(), ref["level"])

    def test_drift_diffusion(
        self, ou: OrnsteinUhlenbeckProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["ornstein_uhlenbeck"]
        tight(ou.drift_1d(1.0, 0.02), ref["drift_t1_x0_02"])
        tight(ou.diffusion_1d(1.0, 0.02), ref["diffusion_t1_x0_02"])

    def test_expectation_variance_stddev(
        self, ou: OrnsteinUhlenbeckProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["ornstein_uhlenbeck"]
        tight(ou.expectation_1d(0.0, 0.02, 1.0), ref["expectation_t0_x0_02_dt1"])
        tight(ou.variance_1d(0.0, 0.02, 1.0), ref["variance_t0_x0_02_dt1"])
        tight(ou.std_deviation_1d(0.0, 0.02, 1.0), ref["stdDeviation_t0_x0_02_dt1"])

    def test_negative_volatility_rejected(self) -> None:
        # C++ parity: ornsteinuhlenbeckprocess.cpp:31 — QL_REQUIRE.
        with pytest.raises(Exception, match="negative volatility"):
            OrnsteinUhlenbeckProcess(0.3, -0.05, 0.02, 0.0)

    def test_variance_small_speed_algebraic_limit(self) -> None:
        # C++ parity: ornsteinuhlenbeckprocess.cpp:35-37 — for tiny speed,
        # variance = sigma^2 * dt. Use speed ~ 0 to trigger the branch.
        ou = OrnsteinUhlenbeckProcess(speed=0.0, vol=0.1, x0=0.0, level=0.0)
        tight(ou.variance_1d(0.0, 0.0, 2.0), 0.1 * 0.1 * 2.0)


# --- CoxIngersollRossProcess -----------------------------------------------


class TestCoxIngersollRossProcess:
    """Cross-validate CIR process closed-forms against the C++ probe.

    Note: the C++ ``variance(t0, x0, dt)`` uses the stored ``x0_`` rather
    than the ``x0`` argument; the Python port preserves that quirk
    exactly (see module docstring of cox_ingersoll_ross_process.py).
    """

    @pytest.fixture
    def cir(self) -> CoxIngersollRossProcess:
        return CoxIngersollRossProcess(0.5, 0.1, 0.05, 0.04)

    def test_is_stochastic_process_1d(self, cir: CoxIngersollRossProcess) -> None:
        assert isinstance(cir, StochasticProcess1D)

    def test_inspectors(
        self, cir: CoxIngersollRossProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["cir"]
        tight(cir.x0(), ref["x0"])
        tight(cir.speed(), ref["speed"])
        tight(cir.volatility(), ref["volatility"])
        tight(cir.level(), ref["level"])

    def test_drift_diffusion(
        self, cir: CoxIngersollRossProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["cir"]
        tight(cir.drift_1d(1.0, 0.05), ref["drift_t1_x0_05"])
        tight(cir.diffusion_1d(1.0, 0.05), ref["diffusion_t1_x0_05"])

    def test_expectation_variance_stddev(
        self, cir: CoxIngersollRossProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["cir"]
        tight(cir.expectation_1d(0.0, 0.05, 1.0), ref["expectation_t0_x0_05_dt1"])
        tight(cir.variance_1d(0.0, 0.05, 1.0), ref["variance_t0_x0_05_dt1"])
        tight(cir.std_deviation_1d(0.0, 0.05, 1.0), ref["stdDeviation_t0_x0_05_dt1"])

    def test_negative_volatility_rejected(self) -> None:
        # C++ parity: coxingersollrossprocess.cpp:29 — QL_REQUIRE.
        with pytest.raises(Exception, match="negative volatility"):
            CoxIngersollRossProcess(0.5, -0.1, 0.05, 0.04)

    def test_evolve_smoke(self) -> None:
        # Smoke: ``evolve_1d`` produces a finite non-negative draw at
        # both QE branches (small psi and large psi). We don't pin the
        # exact value because the C++ probe doesn't cross-validate it
        # either; the QE scheme is bit-stable across implementations
        # but path-dependent, so a smoke check is the right tier.
        cir = CoxIngersollRossProcess(0.5, 0.1, 0.05, 0.04)
        for dw in (-1.0, 0.0, 1.0):
            v = cir.evolve_1d(0.0, 0.05, 1.0, dw)
            assert math.isfinite(v)
            assert v >= 0.0


# --- G2Process -------------------------------------------------------------


class TestG2Process:
    """Cross-validate G2Process drift/diffusion/expectation against the C++ probe."""

    @pytest.fixture
    def g2(self) -> G2Process:
        return G2Process(0.1, 0.01, 0.1, 0.01, -0.75)

    def test_is_stochastic_process(self, g2: G2Process) -> None:
        assert isinstance(g2, StochasticProcess)

    def test_inspectors(self, g2: G2Process, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_process"]
        assert g2.size() == ref["size"]
        tight(g2.a(), ref["a"])
        tight(g2.sigma(), ref["sigma"])
        tight(g2.b(), ref["b"])
        tight(g2.eta(), ref["eta"])
        tight(g2.rho(), ref["rho"])
        _tight_arr(g2.initial_values(), ref["initial_values"])

    def test_drift_origin(self, g2: G2Process, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2.drift(1.0, x0), ref["drift_t1_origin"])

    def test_drift_nonzero(self, g2: G2Process, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_process"]
        # ``drift_t2_x_001_neg_0005``: drift at t=2 with x=0.01, y=-0.005.
        _tight_arr(g2.drift(2.0, np.array([0.01, -0.005])), ref["drift_t2_x_001_neg_0005"])

    def test_diffusion(self, g2: G2Process, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2.diffusion(1.0, x0), ref["diffusion_t1_origin"])

    def test_expectation(self, g2: G2Process, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2.expectation(0.0, x0, 1.0), ref["expectation_t0_origin_dt1"])

    def test_std_deviation(self, g2: G2Process, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2.std_deviation(0.0, x0, 1.0), ref["std_deviation_t0_origin_dt1"])

    def test_covariance(self, g2: G2Process, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2.covariance(0.0, x0, 1.0), ref["covariance_t0_origin_dt1"])


# --- G2ForwardProcess ------------------------------------------------------


class TestG2ForwardProcess:
    """G2 process under the T-forward measure."""

    @pytest.fixture
    def g2f(self) -> G2ForwardProcess:
        p = G2ForwardProcess(0.1, 0.01, 0.1, 0.01, -0.75)
        p.set_forward_measure_time(10.0)
        return p

    def test_is_forward_measure_process(self, g2f: G2ForwardProcess) -> None:
        assert isinstance(g2f, ForwardMeasureProcess)

    def test_inspectors(
        self, g2f: G2ForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["g2_forward_process"]
        assert g2f.size() == ref["size"]
        tight(g2f.get_forward_measure_time(), ref["forward_T"])

    def test_drift(self, g2f: G2ForwardProcess, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2_forward_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2f.drift(1.0, x0), ref["drift_t1_origin"])

    def test_diffusion(
        self, g2f: G2ForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["g2_forward_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2f.diffusion(1.0, x0), ref["diffusion_t1_origin"])

    def test_expectation(
        self, g2f: G2ForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["g2_forward_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2f.expectation(0.0, x0, 1.0), ref["expectation_t0_origin_dt1"])

    def test_std_deviation(
        self, g2f: G2ForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["g2_forward_process"]
        x0 = np.array([0.0, 0.0])
        _tight_arr(g2f.std_deviation(0.0, x0, 1.0), ref["std_deviation_t0_origin_dt1"])


# --- HullWhiteForwardProcess -----------------------------------------------


class TestHullWhiteForwardProcess:
    """HW process under the T-forward measure on a 5% flat-forward curve."""

    @pytest.fixture
    def hwf(self, flat_5_curve: FlatForward) -> HullWhiteForwardProcess:
        p = HullWhiteForwardProcess(flat_5_curve, 0.1, 0.01)
        p.set_forward_measure_time(10.0)
        return p

    def test_is_forward_measure_process_1d(
        self, hwf: HullWhiteForwardProcess
    ) -> None:
        assert isinstance(hwf, ForwardMeasureProcess1D)

    def test_inspectors(
        self, hwf: HullWhiteForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["hull_white_forward_process"]
        tight(hwf.a(), ref["a"])
        tight(hwf.sigma(), ref["sigma"])
        tight(hwf.get_forward_measure_time(), ref["forward_T"])
        tight(hwf.x0(), ref["x0"])

    def test_closed_form_helpers(
        self, hwf: HullWhiteForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["hull_white_forward_process"]
        tight(hwf.alpha(1.0), ref["alpha_at_1"])
        tight(hwf.alpha(5.0), ref["alpha_at_5"])
        tight(hwf.B(1.0, 10.0), ref["B_t1_T10"])
        tight(hwf.M_T(0.0, 1.0, 10.0), ref["M_T_s0_t1_T10"])

    def test_drift_diffusion(
        self, hwf: HullWhiteForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["hull_white_forward_process"]
        x = hwf.x0()
        tight(hwf.drift_1d(1.0, x), ref["drift_t1_x0"])
        tight(hwf.diffusion_1d(1.0, x), ref["diffusion_t1_x0"])

    def test_expectation_variance_stddev(
        self, hwf: HullWhiteForwardProcess, reference_data: dict[str, Any]
    ) -> None:
        ref = reference_data["hull_white_forward_process"]
        x = hwf.x0()
        tight(hwf.expectation_1d(0.0, x, 1.0), ref["expectation_t0_x0_dt1"])
        tight(hwf.variance_1d(0.0, x, 1.0), ref["variance_t0_x0_dt1"])
        tight(hwf.std_deviation_1d(0.0, x, 1.0), ref["stdDeviation_t0_x0_dt1"])

    def test_B_small_a_limit(self, flat_5_curve: FlatForward) -> None:  # noqa: N802 — math symbol
        # C++ parity: hullwhiteprocess.cpp:148-152 — a -> 0 branch
        # returns T-t.
        hwf = HullWhiteForwardProcess(flat_5_curve, 0.0, 0.01)
        tight(hwf.B(1.0, 5.0), 4.0)

    def test_M_T_small_a_limit(self, flat_5_curve: FlatForward) -> None:  # noqa: N802 — math symbol
        # C++ parity: hullwhiteprocess.cpp:142-145 — a -> 0 branch
        # returns (sigma^2/2)*(t-s)*(2T - t - s).
        hwf = HullWhiteForwardProcess(flat_5_curve, 0.0, 0.01)
        expected = 0.5 * 0.01 * 0.01 * 1.0 * (2.0 * 10.0 - 1.0 - 0.0)
        tight(hwf.M_T(0.0, 1.0, 10.0), expected)
