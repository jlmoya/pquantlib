"""BatesDetJumpModel behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/w1c.json``.

C++ parity: ql/models/equity/batesmodel.{hpp,cpp} @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.bates_det_jump_model import BatesDetJumpModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.bates_process import BatesProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/w1c")


@pytest.fixture
def bates_process() -> BatesProcess:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    return BatesProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        lambda_=0.1,
        nu=-0.05,
        delta=0.1,
    )


def test_parameter_accessors(
    bates_process: BatesProcess, cpp_refs: dict[str, Any]
) -> None:
    model = BatesDetJumpModel(
        bates_process, kappa_lambda=1.0, theta_lambda=0.1
    )
    m = cpp_refs["bates_det_jump_model"]
    exact(model.theta(), m["theta"])
    exact(model.kappa(), m["kappa"])
    exact(model.sigma(), m["sigma"])
    exact(model.rho(), m["rho"])
    exact(model.v0(), m["v0"])
    exact(model.nu(), m["nu"])
    exact(model.delta(), m["delta"])
    exact(model.lambda_(), m["lambda"])
    exact(model.kappa_lambda(), m["kappaLambda"])
    exact(model.theta_lambda(), m["thetaLambda"])


def test_arguments_has_ten_slots(
    bates_process: BatesProcess, cpp_refs: dict[str, Any]
) -> None:
    """BatesDetJumpModel extends BatesModel's 8 args to 10.

    # C++ parity: batesmodel.cpp:51 — arguments_.resize(10).
    """
    model = BatesDetJumpModel(bates_process)
    assert len(model.arguments) == 10
    assert len(model.arguments) == cpp_refs["bates_det_jump_model"]["n_args"]
    for arg in model.arguments:
        assert isinstance(arg, ConstantParameter)


def test_params_vector_order(bates_process: BatesProcess) -> None:
    """params() = [theta, kappa, sigma, rho, v0, nu, delta, lambda, kL, tL].

    # C++ parity: batesmodel.cpp:53-56 — slots 8, 9 in that order, after
    # the 8 inherited Bates slots.
    """
    model = BatesDetJumpModel(
        bates_process, kappa_lambda=1.5, theta_lambda=0.2
    )
    p = model.params()
    assert p.shape == (10,)
    # Inherited 8 slots
    exact(float(p[0]), 0.04)  # theta
    exact(float(p[1]), 2.0)  # kappa
    exact(float(p[2]), 0.3)  # sigma
    exact(float(p[3]), -0.7)  # rho
    exact(float(p[4]), 0.04)  # v0
    exact(float(p[5]), -0.05)  # nu
    exact(float(p[6]), 0.1)  # delta
    exact(float(p[7]), 0.1)  # lambda
    # New OU-intensity slots
    exact(float(p[8]), 1.5)  # kappaLambda
    exact(float(p[9]), 0.2)  # thetaLambda


def test_constraint_rejects_negative_kappa_lambda(
    bates_process: BatesProcess,
) -> None:
    """kappaLambda must be Positive.

    # C++ parity: batesmodel.cpp:54.
    """
    model = BatesDetJumpModel(bates_process)
    c = model.constraint
    p = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, -0.05, 0.1, 0.1, -0.5, 0.1],
        dtype=np.float64,
    )
    assert c.test(p) is False


def test_constraint_rejects_negative_theta_lambda(
    bates_process: BatesProcess,
) -> None:
    """thetaLambda must be Positive.

    # C++ parity: batesmodel.cpp:56.
    """
    model = BatesDetJumpModel(bates_process)
    c = model.constraint
    p = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, -0.05, 0.1, 0.1, 1.0, -0.05],
        dtype=np.float64,
    )
    assert c.test(p) is False


def test_set_params_propagates_to_inherited_slots(
    bates_process: BatesProcess,
) -> None:
    """set_params triggers generate_arguments → new BatesProcess.

    The deterministic-intensity slots are calibration-only (no impact
    on the underlying BatesProcess), so we only check the Bates ones
    here.
    """
    model = BatesDetJumpModel(bates_process)
    new_params = np.array(
        [0.05, 1.5, 0.2, -0.5, 0.03, -0.02, 0.15, 0.2, 2.0, 0.3],
        dtype=np.float64,
    )
    model.set_params(new_params)
    new_process = model.process()
    assert isinstance(new_process, BatesProcess)
    exact(new_process.theta, 0.05)
    exact(new_process.lambda_, 0.2)
    exact(new_process.nu, -0.02)
    exact(new_process.delta, 0.15)
    # New det-intensity slots come through accessors.
    exact(model.kappa_lambda(), 2.0)
    exact(model.theta_lambda(), 0.3)
