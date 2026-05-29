"""GjrGarchModel behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/w1d.json``.

C++ parity: ql/models/equity/gjrgarchmodel.{hpp,cpp} @ v1.42.1 (099987f0).

Tolerance choice:

* Parameter accessors: EXACT — straight passthrough.
* set_params round-trip: EXACT — the optimizer view back-propagates
  unchanged through the ConstantParameter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.gjr_garch_model import (
    GjrGarchModel,
    VolatilityConstraint,
)
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.gjr_garch_process import GjrGarchProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/w1d")


@pytest.fixture
def gjr_process() -> GjrGarchProcess:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    return GjrGarchProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.000160,
        omega=0.000002,
        alpha=0.024,
        beta=0.93,
        gamma=0.059,
        lambda_=0.2,
        days_per_year=252.0,
    )


def test_parameter_accessors(
    gjr_process: GjrGarchProcess, cpp_refs: dict[str, Any]
) -> None:
    model = GjrGarchModel(gjr_process)
    m = cpp_refs["gjr_garch_model"]
    exact(model.omega(), m["omega"])
    exact(model.alpha(), m["alpha"])
    exact(model.beta(), m["beta"])
    exact(model.gamma(), m["gamma"])
    exact(model.lambda_(), m["lambda"])
    exact(model.v0(), m["v0"])


def test_arguments_have_six_slots(gjr_process: GjrGarchProcess) -> None:
    model = GjrGarchModel(gjr_process)
    assert len(model.arguments) == 6
    for arg in model.arguments:
        assert isinstance(arg, ConstantParameter)


def test_params_vector_order_matches_cpp_layout(
    gjr_process: GjrGarchProcess,
) -> None:
    """params() returns [omega, alpha, beta, gamma, lambda, v0].

    # C++ parity: gjrgarchmodel.cpp:46-56.
    """
    model = GjrGarchModel(gjr_process)
    p = model.params()
    assert p.shape == (6,)
    exact(float(p[0]), 0.000002)
    exact(float(p[1]), 0.024)
    exact(float(p[2]), 0.93)
    exact(float(p[3]), 0.059)
    exact(float(p[4]), 0.2)
    exact(float(p[5]), 0.000160)


def test_set_params_rebuilds_process(gjr_process: GjrGarchProcess) -> None:
    """set_params triggers generate_arguments which rebuilds the process."""
    model = GjrGarchModel(gjr_process)
    original = model.process()
    new_params = np.array(
        [3e-6, 0.05, 0.90, 0.02, 0.15, 0.00018], dtype=np.float64
    )
    model.set_params(new_params)
    new_process = model.process()
    assert new_process is not original
    exact(new_process.omega, 3e-6)
    exact(new_process.alpha, 0.05)
    exact(new_process.beta, 0.90)
    exact(new_process.gamma, 0.02)
    exact(new_process.lambda_, 0.15)
    exact(new_process.v0, 0.00018)


def test_volatility_constraint_passes_in_region() -> None:
    """beta + gamma >= 0 — region-positive at canonical params.

    # C++ parity: gjrgarchmodel.cpp:26-40.
    """
    vc = VolatilityConstraint()
    p = np.array([2e-6, 0.024, 0.93, 0.059, 0.2, 0.00016], dtype=np.float64)
    assert vc.test(p) is True


def test_volatility_constraint_rejects_out_of_region() -> None:
    """beta + gamma < 0 → reject."""
    vc = VolatilityConstraint()
    # gamma sufficiently negative to push beta + gamma < 0.
    p = np.array([2e-6, 0.024, 0.10, -0.20, 0.2, 0.00016], dtype=np.float64)
    assert vc.test(p) is False


def test_default_constraint_passes_canonical_params(
    gjr_process: GjrGarchProcess,
) -> None:
    """Composite constraint accepts canonical params."""
    model = GjrGarchModel(gjr_process)
    c = model.constraint
    p = np.array(
        [2e-6, 0.024, 0.93, 0.059, 0.2, 0.00016], dtype=np.float64
    )
    assert c.test(p) is True


def test_default_constraint_rejects_alpha_above_one(
    gjr_process: GjrGarchProcess,
) -> None:
    """alpha > 1 → reject (BoundaryConstraint(0, 1))."""
    model = GjrGarchModel(gjr_process)
    c = model.constraint
    p = np.array([2e-6, 1.5, 0.93, 0.059, 0.2, 0.00016], dtype=np.float64)
    assert c.test(p) is False


def test_default_constraint_rejects_negative_v0(
    gjr_process: GjrGarchProcess,
) -> None:
    """v0 < 0 → reject (PositiveConstraint)."""
    model = GjrGarchModel(gjr_process)
    c = model.constraint
    p = np.array(
        [2e-6, 0.024, 0.93, 0.059, 0.2, -0.0001], dtype=np.float64
    )
    assert c.test(p) is False
