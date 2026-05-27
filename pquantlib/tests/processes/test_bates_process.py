"""BatesProcess behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/l4c.json``.

C++ parity: ql/processes/batesprocess.{hpp,cpp} @ v1.42.1 (099987f0).

Bates = Heston + Merton lognormal jumps. The drift on S gets a
martingale correction ``-lambda * m`` where ``m = exp(nu+0.5*delta^2) - 1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.bates_process import BatesProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/l4c")


@pytest.fixture
def process() -> BatesProcess:
    """The Bates testbed with lambda=0.1, nu=-0.05, delta=0.1."""
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


def test_size_unchanged_from_heston(process: BatesProcess, cpp_refs: dict[str, Any]) -> None:
    """State dimension stays 2 (S, V); jumps don't add state."""
    b = cpp_refs["bates_process"]
    assert process.size() == b["size"]


def test_factors_includes_jump_brownians(
    process: BatesProcess, cpp_refs: dict[str, Any]
) -> None:
    """Bates ``factors() == HestonFactors + 2 == 4`` for the MC path.

    # C++ parity: batesprocess.cpp:65-67 — adds 2 extra factors for
    # the Poisson arrival + log-jump-size Brownians.
    """
    b = cpp_refs["bates_process"]
    assert process.factors() == b["factors"]


def test_jump_parameter_accessors(
    process: BatesProcess, cpp_refs: dict[str, Any]
) -> None:
    b = cpp_refs["bates_process"]
    exact(process.lambda_, b["lambda"])
    exact(process.nu, b["nu"])
    exact(process.delta, b["delta"])


def test_m_jump_multiplier(process: BatesProcess, cpp_refs: dict[str, Any]) -> None:
    """m = exp(nu + 0.5 * delta^2) - 1.

    # C++ parity: batesprocess.cpp:37 — pre-computed at construction.
    """
    b = cpp_refs["bates_process"]
    tight(process.m, b["m"])


def test_drift_applies_jump_correction(
    process: BatesProcess, cpp_refs: dict[str, Any]
) -> None:
    """Bates drift_s = Heston drift_s - lambda * m.

    # C++ parity: batesprocess.cpp:40-44.
    """
    b = cpp_refs["bates_process"]
    x = np.array([100.0, 0.04], dtype=np.float64)
    d = process.drift(0.0, x)
    tight(float(d[0]), b["drift_s"])
    tight(float(d[1]), b["drift_v"])


def test_drift_with_zero_lambda_reduces_to_heston() -> None:
    """When lambda=0 the Bates drift equals the Heston drift exactly."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)

    bates = BatesProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        lambda_=0.0,
        nu=-0.05,
        delta=0.1,
    )
    heston = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
    )
    x = np.array([100.0, 0.04], dtype=np.float64)
    bd = bates.drift(0.0, x)
    hd = heston.drift(0.0, x)
    exact(float(bd[0]), float(hd[0]))
    exact(float(bd[1]), float(hd[1]))


def test_inherits_heston_diffusion(process: BatesProcess) -> None:
    """Diffusion matrix is unchanged — jumps only affect drift + MC evolve.

    # C++ parity: BatesProcess does not override diffusion(), so it
    # inherits HestonProcess::diffusion verbatim.
    """
    x = np.array([100.0, 0.04], dtype=np.float64)
    diff = process.diffusion(0.0, x)
    # Cholesky form with vol=0.2, sigma=0.3, rho=-0.7:
    #   [[0.2, 0], [-0.7*0.06, sqrt(1-0.49)*0.06]]
    tight(float(diff[0, 0]), 0.2)
    exact(float(diff[0, 1]), 0.0)
    tight(float(diff[1, 0]), -0.7 * 0.3 * 0.2)
    tight(float(diff[1, 1]), np.sqrt(1.0 - 0.49) * 0.3 * 0.2)
