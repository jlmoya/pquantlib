"""BatesDoubleExpDetJumpModel behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/w1c.json``.

C++ parity: ql/models/equity/batesmodel.{hpp,cpp} @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.bates_double_exp_det_jump_model import (
    BatesDoubleExpDetJumpModel,
)
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.heston_process import HestonProcess
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
def heston_process() -> HestonProcess:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    return HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
    )


def test_parameter_accessors(
    heston_process: HestonProcess, cpp_refs: dict[str, Any]
) -> None:
    model = BatesDoubleExpDetJumpModel(
        heston_process,
        lambda_=0.1,
        nu_up=0.05,
        nu_down=0.05,
        p=0.5,
        kappa_lambda=1.0,
        theta_lambda=0.1,
    )
    m = cpp_refs["bates_double_exp_det_jump_model"]
    exact(model.theta(), m["theta"])
    exact(model.kappa(), m["kappa"])
    exact(model.sigma(), m["sigma"])
    exact(model.rho(), m["rho"])
    exact(model.v0(), m["v0"])
    exact(model.p(), m["p"])
    exact(model.nu_down(), m["nuDown"])
    exact(model.nu_up(), m["nuUp"])
    exact(model.lambda_(), m["lambda"])
    exact(model.kappa_lambda(), m["kappaLambda"])
    exact(model.theta_lambda(), m["thetaLambda"])


def test_arguments_has_eleven_slots(
    heston_process: HestonProcess, cpp_refs: dict[str, Any]
) -> None:
    """BatesDoubleExpDetJumpModel extends DoubleExp's 9 args to 11.

    # C++ parity: batesmodel.cpp:79 — arguments_.resize(11).
    """
    model = BatesDoubleExpDetJumpModel(heston_process)
    assert len(model.arguments) == 11
    assert (
        len(model.arguments)
        == cpp_refs["bates_double_exp_det_jump_model"]["n_args"]
    )
    for arg in model.arguments:
        assert isinstance(arg, ConstantParameter)


def test_params_vector_order(heston_process: HestonProcess) -> None:
    """params() = [...DoubleExp.., kappaLambda, thetaLambda]."""
    model = BatesDoubleExpDetJumpModel(
        heston_process,
        lambda_=0.2,
        nu_up=0.06,
        nu_down=0.07,
        p=0.45,
        kappa_lambda=1.5,
        theta_lambda=0.3,
    )
    p = model.params()
    assert p.shape == (11,)
    exact(float(p[0]), 0.04)  # theta
    exact(float(p[1]), 2.0)  # kappa
    exact(float(p[2]), 0.3)  # sigma
    exact(float(p[3]), -0.7)  # rho
    exact(float(p[4]), 0.04)  # v0
    exact(float(p[5]), 0.45)  # p
    exact(float(p[6]), 0.07)  # nuDown
    exact(float(p[7]), 0.06)  # nuUp
    exact(float(p[8]), 0.2)  # lambda
    exact(float(p[9]), 1.5)  # kappaLambda
    exact(float(p[10]), 0.3)  # thetaLambda


def test_constraint_rejects_negative_det_intensity(
    heston_process: HestonProcess,
) -> None:
    """kappaLambda, thetaLambda must be Positive.

    # C++ parity: batesmodel.cpp:81-84.
    """
    model = BatesDoubleExpDetJumpModel(heston_process)
    c = model.constraint
    # negative kappaLambda
    p_bad_kl = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 0.5, 0.1, 0.1, 0.1, -1.0, 0.1],
        dtype=np.float64,
    )
    assert c.test(p_bad_kl) is False
    # negative thetaLambda
    p_bad_tl = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 0.5, 0.1, 0.1, 0.1, 1.0, -0.05],
        dtype=np.float64,
    )
    assert c.test(p_bad_tl) is False


def test_inherits_double_exp_constraints(
    heston_process: HestonProcess,
) -> None:
    """The 5..8 slot constraints from BatesDoubleExpModel are inherited.

    # C++ parity: the composite per-Parameter constraint covers all 11
    # slots since each ConstantParameter brings its own Constraint.
    """
    model = BatesDoubleExpDetJumpModel(heston_process)
    c = model.constraint
    # p out of [0,1] should be rejected even at the 11-vector level.
    p = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 1.5, 0.1, 0.1, 0.1, 1.0, 0.1],
        dtype=np.float64,
    )
    assert c.test(p) is False
