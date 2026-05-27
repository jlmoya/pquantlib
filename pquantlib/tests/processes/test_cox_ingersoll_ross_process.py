"""Tests for CoxIngersollRossProcess.

Cross-validates against ``migration-harness/references/cluster/l4b.json``.

C++ parity: ql/processes/coxingersollrossprocess.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4b")


def _build_cir() -> CoxIngersollRossProcess:
    """CIR with k=0.5, sigma=0.1, x0=0.05, theta=0.06 (Feller OK)."""
    return CoxIngersollRossProcess(speed=0.5, vol=0.1, x0=0.05, level=0.06)


def test_cir_inspectors(reference_data: dict[str, Any]) -> None:
    """CIR process inspector round-trip."""
    ref = reference_data["cox_ingersoll_ross"]
    p = _build_cir()
    tight(p.x0(), ref["x0"])
    tight(p.speed(), ref["speed"])
    tight(p.volatility(), ref["volatility"])
    tight(p.level(), ref["level"])


def test_cir_drift_diffusion(reference_data: dict[str, Any]) -> None:
    """Drift = k * (theta - x); diffusion = sigma (constant, per C++)."""
    ref = reference_data["cox_ingersoll_ross"]
    p = _build_cir()
    tight(p.drift_1d(0.0, 0.05), ref["drift_at_x0"])
    # NOTE: C++ returns ``sigma`` for diffusion, NOT ``sigma * sqrt(x)``.
    # The sqrt(x) factor is folded into the Andersen QE evolve scheme.
    tight(p.diffusion_1d(0.0, 0.05), ref["diffusion_at_x0"])


def test_cir_expectation(reference_data: dict[str, Any]) -> None:
    """E[x_{dt}] = theta + (x0 - theta) * exp(-k*dt)."""
    ref = reference_data["cox_ingersoll_ross"]
    p = _build_cir()
    tight(p.expectation_1d(0.0, 0.05, 1.0), ref["expectation_dt_1"])


def test_cir_variance(reference_data: dict[str, Any]) -> None:
    """CIR variance uses ctor-time x0_ and level_ (C++ idiosyncrasy)."""
    ref = reference_data["cox_ingersoll_ross"]
    p = _build_cir()
    tight(p.variance_1d(0.0, 0.05, 1.0), ref["variance_dt_1"])
    tight(p.variance_1d(0.0, 0.05, 5.0), ref["variance_dt_5"])


def test_cir_negative_volatility_rejected() -> None:
    """Negative volatility is rejected at construction."""
    with pytest.raises(LibraryException, match="negative volatility"):
        CoxIngersollRossProcess(speed=0.5, vol=-0.1)


def test_cir_evolve_returns_finite_nonnegative() -> None:
    """Evolution step under QE scheme returns a finite, non-negative state.

    No probe value for ``evolve`` — the QE scheme depends on a CumulativeNormal
    branch which is sensitive to dw, so the contract test is the property of
    non-negativity (CIR samples are non-negative by construction).
    """
    p = _build_cir()
    for dw in (-2.0, -0.5, 0.0, 0.5, 2.0):
        result = p.evolve_1d(0.0, 0.05, 0.1, dw)
        assert result >= 0.0
        assert result < 1.0  # sanity: tiny dt won't blow up
