"""BatesDoubleExpModel behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/w1c.json``.

C++ parity: ql/models/equity/batesmodel.{hpp,cpp} @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.bates_double_exp_model import BatesDoubleExpModel
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
    model = BatesDoubleExpModel(
        heston_process, lambda_=0.1, nu_up=0.05, nu_down=0.05, p=0.5
    )
    m = cpp_refs["bates_double_exp_model"]
    exact(model.theta(), m["theta"])
    exact(model.kappa(), m["kappa"])
    exact(model.sigma(), m["sigma"])
    exact(model.rho(), m["rho"])
    exact(model.v0(), m["v0"])
    exact(model.p(), m["p"])
    exact(model.nu_down(), m["nuDown"])
    exact(model.nu_up(), m["nuUp"])
    exact(model.lambda_(), m["lambda"])


def test_arguments_has_nine_slots(
    heston_process: HestonProcess, cpp_refs: dict[str, Any]
) -> None:
    """BatesDoubleExpModel extends HestonModel's 5 args to 9.

    # C++ parity: batesmodel.cpp:64 — arguments_.resize(9).
    """
    model = BatesDoubleExpModel(heston_process)
    assert len(model.arguments) == 9
    assert len(model.arguments) == cpp_refs["bates_double_exp_model"]["n_args"]
    for arg in model.arguments:
        assert isinstance(arg, ConstantParameter)


def test_params_vector_order(heston_process: HestonProcess) -> None:
    """params() = [theta, kappa, sigma, rho, v0, p, nuDown, nuUp, lambda].

    # C++ parity: batesmodel.cpp:66-70 — slots 5..8 in that order.
    """
    model = BatesDoubleExpModel(
        heston_process, lambda_=0.2, nu_up=0.07, nu_down=0.08, p=0.45
    )
    p = model.params()
    assert p.shape == (9,)
    exact(float(p[0]), 0.04)  # theta
    exact(float(p[1]), 2.0)  # kappa
    exact(float(p[2]), 0.3)  # sigma
    exact(float(p[3]), -0.7)  # rho
    exact(float(p[4]), 0.04)  # v0
    exact(float(p[5]), 0.45)  # p
    exact(float(p[6]), 0.08)  # nuDown
    exact(float(p[7]), 0.07)  # nuUp
    exact(float(p[8]), 0.2)  # lambda


def test_constraint_p_in_unit_interval(heston_process: HestonProcess) -> None:
    """p must be in [0, 1] (BoundaryConstraint).

    # C++ parity: batesmodel.cpp:66-67 — BoundaryConstraint(0.0, 1.0).
    """
    model = BatesDoubleExpModel(heston_process)
    c = model.constraint
    # p = 1.5 outside upper bound — rejected.
    p_bad_high = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 1.5, 0.05, 0.05, 0.1], dtype=np.float64
    )
    assert c.test(p_bad_high) is False
    # p = -0.1 outside lower bound — rejected.
    p_bad_low = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, -0.1, 0.05, 0.05, 0.1], dtype=np.float64
    )
    assert c.test(p_bad_low) is False
    # p = 0.7 inside — accepted.
    p_good = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 0.7, 0.05, 0.05, 0.1], dtype=np.float64
    )
    assert c.test(p_good) is True


def test_constraint_rejects_negative_jump_means(
    heston_process: HestonProcess,
) -> None:
    """nuUp, nuDown must be Positive.

    # C++ parity: batesmodel.cpp:68-69.
    """
    model = BatesDoubleExpModel(heston_process)
    c = model.constraint
    p_bad_nu_down = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 0.5, -0.05, 0.05, 0.1], dtype=np.float64
    )
    assert c.test(p_bad_nu_down) is False
    p_bad_nu_up = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 0.5, 0.05, -0.05, 0.1], dtype=np.float64
    )
    assert c.test(p_bad_nu_up) is False


def test_constraint_rejects_negative_lambda(
    heston_process: HestonProcess,
) -> None:
    """lambda must be Positive.

    # C++ parity: batesmodel.cpp:70.
    """
    model = BatesDoubleExpModel(heston_process)
    c = model.constraint
    p = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, 0.5, 0.05, 0.05, -0.1], dtype=np.float64
    )
    assert c.test(p) is False


def test_set_params_propagates_to_heston_process(
    heston_process: HestonProcess,
) -> None:
    """set_params + generate_arguments rebuilds the underlying HestonProcess.

    The double-exp jump slots are model-level (not in the process).
    """
    model = BatesDoubleExpModel(heston_process)
    new_params = np.array(
        [0.05, 1.5, 0.2, -0.5, 0.03, 0.7, 0.08, 0.06, 0.15], dtype=np.float64
    )
    model.set_params(new_params)
    new_process = model.process()
    exact(new_process.theta, 0.05)
    exact(new_process.kappa, 1.5)
    exact(new_process.sigma, 0.2)
    exact(new_process.rho, -0.5)
    exact(new_process.v0, 0.03)
    # Jump params come back through model accessors.
    exact(model.p(), 0.7)
    exact(model.nu_down(), 0.08)
    exact(model.nu_up(), 0.06)
    exact(model.lambda_(), 0.15)
