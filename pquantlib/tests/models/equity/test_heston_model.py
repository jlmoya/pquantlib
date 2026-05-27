"""HestonModel behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/l4c.json``.

C++ parity: ql/models/equity/hestonmodel.{hpp,cpp} @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.heston_model import FellerConstraint, HestonModel
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
    return load_reference("cluster/l4c")


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
    model = HestonModel(heston_process)
    m = cpp_refs["heston_model"]
    exact(model.theta(), m["theta"])
    exact(model.kappa(), m["kappa"])
    exact(model.sigma(), m["sigma"])
    exact(model.rho(), m["rho"])
    exact(model.v0(), m["v0"])


def test_arguments_have_five_slots(heston_process: HestonProcess) -> None:
    model = HestonModel(heston_process)
    assert len(model.arguments) == 5
    for arg in model.arguments:
        assert isinstance(arg, ConstantParameter)


def test_params_vector_order_matches_cpp_layout(
    heston_process: HestonProcess,
) -> None:
    """params() returns [theta, kappa, sigma, rho, v0] in that order.

    # C++ parity: hestonmodel.cpp:28-37 — arguments_[0]=theta, [1]=kappa,
    # [2]=sigma, [3]=rho, [4]=v0.
    """
    model = HestonModel(heston_process)
    p = model.params()
    assert p.shape == (5,)
    exact(float(p[0]), 0.04)  # theta
    exact(float(p[1]), 2.0)  # kappa
    exact(float(p[2]), 0.3)  # sigma
    exact(float(p[3]), -0.7)  # rho
    exact(float(p[4]), 0.04)  # v0


def test_set_params_rebuilds_process(heston_process: HestonProcess) -> None:
    """set_params triggers generate_arguments which rebuilds the process."""
    model = HestonModel(heston_process)
    original_process = model.process()
    new_params = np.array([0.05, 1.5, 0.2, -0.5, 0.03], dtype=np.float64)
    model.set_params(new_params)
    # generate_arguments must have rebuilt the process.
    new_process = model.process()
    assert new_process is not original_process
    exact(new_process.theta, 0.05)
    exact(new_process.kappa, 1.5)
    exact(new_process.sigma, 0.2)
    exact(new_process.rho, -0.5)
    exact(new_process.v0, 0.03)


def test_default_constraint_passes_physical_params(
    heston_process: HestonProcess,
) -> None:
    """The composite per-Parameter constraint passes physical Heston params.

    # C++ parity: hestonmodel.hpp:33-37 — composite of 4 PositiveConstraint
    # + 1 BoundaryConstraint(-1, 1).
    """
    model = HestonModel(heston_process)
    c = model.constraint
    # Initial params: (0.04, 2.0, 0.3, -0.7, 0.04) — all physical.
    p = np.array([0.04, 2.0, 0.3, -0.7, 0.04], dtype=np.float64)
    assert c.test(p) is True


def test_default_constraint_rejects_invalid_rho(
    heston_process: HestonProcess,
) -> None:
    """The composite constraint rejects |rho| > 1."""
    model = HestonModel(heston_process)
    c = model.constraint
    p = np.array([0.04, 2.0, 0.3, -1.5, 0.04], dtype=np.float64)
    assert c.test(p) is False


def test_default_constraint_rejects_negative_kappa(
    heston_process: HestonProcess,
) -> None:
    """The composite constraint rejects kappa <= 0."""
    model = HestonModel(heston_process)
    c = model.constraint
    p = np.array([0.04, -0.5, 0.3, -0.7, 0.04], dtype=np.float64)
    assert c.test(p) is False


def test_feller_constraint_accepts_in_region(
    heston_process: HestonProcess,
) -> None:
    """Feller-condition constraint: sigma^2 < 2*kappa*theta.

    # C++ parity: ``HestonModel::FellerConstraint`` in
    # hestonmodel.hpp:66-82.

    With kappa=2, theta=0.04, sigma=0.3:
    sigma^2 = 0.09; 2*kappa*theta = 0.16; 0.09 < 0.16 → in-region.
    """
    fc = FellerConstraint()
    p = np.array([0.04, 2.0, 0.3, -0.7, 0.04], dtype=np.float64)
    assert fc.test(p) is True


def test_feller_constraint_rejects_out_of_region(
    heston_process: HestonProcess,
) -> None:
    """sigma=0.5 with kappa=2, theta=0.04: sigma^2 = 0.25 > 0.16. Reject."""
    fc = FellerConstraint()
    p = np.array([0.04, 2.0, 0.5, -0.7, 0.04], dtype=np.float64)
    assert fc.test(p) is False


def test_feller_constraint_rejects_negative_sigma() -> None:
    """sigma must be non-negative."""
    fc = FellerConstraint()
    p = np.array([0.04, 2.0, -0.1, -0.7, 0.04], dtype=np.float64)
    assert fc.test(p) is False
