"""BatesModel behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/l4c.json``.

C++ parity: ql/models/equity/batesmodel.{hpp,cpp} @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.bates_model import BatesModel
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
    return load_reference("cluster/l4c")


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
    model = BatesModel(bates_process)
    b = cpp_refs["bates_model"]
    exact(model.theta(), b["theta"])
    exact(model.kappa(), b["kappa"])
    exact(model.sigma(), b["sigma"])
    exact(model.rho(), b["rho"])
    exact(model.v0(), b["v0"])
    exact(model.nu(), b["nu"])
    exact(model.delta(), b["delta"])
    exact(model.lambda_(), b["lambda"])


def test_arguments_has_eight_slots(bates_process: BatesProcess) -> None:
    """BatesModel extends HestonModel's 5 args to 8.

    # C++ parity: batesmodel.cpp:27 — arguments_.resize(8).
    """
    model = BatesModel(bates_process)
    assert len(model.arguments) == 8
    for arg in model.arguments:
        assert isinstance(arg, ConstantParameter)


def test_params_vector_order(bates_process: BatesProcess) -> None:
    """params() = [theta, kappa, sigma, rho, v0, nu, delta, lambda].

    # C++ parity: batesmodel.cpp:29-34 — slots 5, 6, 7 in that order.
    """
    model = BatesModel(bates_process)
    p = model.params()
    assert p.shape == (8,)
    exact(float(p[0]), 0.04)  # theta
    exact(float(p[1]), 2.0)  # kappa
    exact(float(p[2]), 0.3)  # sigma
    exact(float(p[3]), -0.7)  # rho
    exact(float(p[4]), 0.04)  # v0
    exact(float(p[5]), -0.05)  # nu
    exact(float(p[6]), 0.1)  # delta
    exact(float(p[7]), 0.1)  # lambda


def test_set_params_rebuilds_bates_process(bates_process: BatesProcess) -> None:
    """set_params triggers generate_arguments → new BatesProcess."""
    model = BatesModel(bates_process)
    original_process = model.process()
    new_params = np.array(
        [0.05, 1.5, 0.2, -0.5, 0.03, -0.02, 0.15, 0.2], dtype=np.float64
    )
    model.set_params(new_params)
    new_process = model.process()
    assert new_process is not original_process
    assert isinstance(new_process, BatesProcess)
    exact(new_process.theta, 0.05)
    exact(new_process.lambda_, 0.2)
    exact(new_process.nu, -0.02)
    exact(new_process.delta, 0.15)


def test_underlying_process_is_bates(bates_process: BatesProcess) -> None:
    """``model.process()`` returns a ``BatesProcess`` (not just HestonProcess).

    Bates models calibrated with jump-aware engines need the underlying
    process to carry the jump parameters, not just the Heston ones.
    """
    model = BatesModel(bates_process)
    p = model.process()
    assert isinstance(p, BatesProcess)
    exact(p.lambda_, 0.1)
    exact(p.nu, -0.05)
    exact(p.delta, 0.1)


def test_inherits_heston_constraint_behavior(bates_process: BatesProcess) -> None:
    """The composite per-Parameter constraint covers all 8 slots.

    # C++ parity: arguments_[5] = NoConstraint(); [6]/[7] = Positive.
    """
    model = BatesModel(bates_process)
    c = model.constraint
    # nu can be negative — verify NoConstraint on slot 5.
    p_negative_nu = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, -0.2, 0.1, 0.1], dtype=np.float64
    )
    assert c.test(p_negative_nu) is True
    # delta must be positive — reject negative.
    p_negative_delta = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, -0.05, -0.1, 0.1], dtype=np.float64
    )
    assert c.test(p_negative_delta) is False
    # lambda must be positive — reject negative.
    p_negative_lambda = np.array(
        [0.04, 2.0, 0.3, -0.7, 0.04, -0.05, 0.1, -0.1], dtype=np.float64
    )
    assert c.test(p_negative_lambda) is False
