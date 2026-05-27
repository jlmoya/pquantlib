"""Tests for OrnsteinUhlenbeckProcess.

Cross-validates against ``migration-harness/references/cluster/l4b.json``.

C++ parity: ql/processes/ornsteinuhlenbeckprocess.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4b")


def test_ornstein_uhlenbeck_inspectors(reference_data: dict[str, Any]) -> None:
    """OU process inspector round-trip via the C++ ctor (a, sigma, x0, level)."""
    ref = reference_data["ornstein_uhlenbeck"]
    p = OrnsteinUhlenbeckProcess(speed=0.5, vol=0.02, x0=0.03, level=0.05)
    tight(p.x0(), ref["x0"])
    tight(p.speed(), ref["speed"])
    tight(p.volatility(), ref["volatility"])
    tight(p.level(), ref["level"])


def test_ornstein_uhlenbeck_drift_diffusion(reference_data: dict[str, Any]) -> None:
    """Drift = a * (level - x); diffusion = sigma (independent of x)."""
    ref = reference_data["ornstein_uhlenbeck"]
    p = OrnsteinUhlenbeckProcess(speed=0.5, vol=0.02, x0=0.03, level=0.05)
    tight(p.drift_1d(0.0, 0.03), ref["drift_at_0_03"])
    tight(p.drift_1d(1.0, 0.08), ref["drift_at_0_08"])
    tight(p.diffusion_1d(0.0, 0.03), ref["diffusion_at_0_03"])


def test_ornstein_uhlenbeck_expectation(reference_data: dict[str, Any]) -> None:
    """E[x_{dt}] = level + (x0 - level) * exp(-speed*dt)."""
    ref = reference_data["ornstein_uhlenbeck"]
    p = OrnsteinUhlenbeckProcess(speed=0.5, vol=0.02, x0=0.03, level=0.05)
    tight(p.expectation_1d(0.0, 0.03, 1.0), ref["expectation_dt_1"])


def test_ornstein_uhlenbeck_variance(reference_data: dict[str, Any]) -> None:
    """Var[x_{dt}] = sigma^2/(2a) * (1 - exp(-2a*dt))."""
    ref = reference_data["ornstein_uhlenbeck"]
    p = OrnsteinUhlenbeckProcess(speed=0.5, vol=0.02, x0=0.03, level=0.05)
    tight(p.variance_1d(0.0, 0.03, 1.0), ref["variance_dt_1"])
    tight(p.variance_1d(0.0, 0.03, 5.0), ref["variance_dt_5"])
    tight(p.std_deviation_1d(0.0, 0.03, 1.0), ref["std_deviation_dt_1"])


def test_ornstein_uhlenbeck_zero_speed_limit(reference_data: dict[str, Any]) -> None:
    """Algebraic limit Var = sigma^2 * dt when |a| < sqrt(QL_EPSILON).

    # C++ parity: ornsteinuhlenbeckprocess.cpp:36-44 — the algebraic
    # ``Var = sigma^2 * dt`` branch for the small mean-reversion case.
    """
    ref = reference_data["ornstein_uhlenbeck_zero_speed"]
    p = OrnsteinUhlenbeckProcess(speed=1e-12, vol=0.02, x0=0.03, level=0.05)
    tight(p.variance_1d(0.0, 0.03, 1.0), ref["variance_dt_1"])
    tight(p.variance_1d(0.0, 0.03, 5.0), ref["variance_dt_5"])


def test_ornstein_uhlenbeck_negative_volatility_rejected() -> None:
    """Negative volatility is rejected at construction.

    # C++ parity: ornsteinuhlenbeckprocess.cpp:31-33 — ``QL_REQUIRE``.
    """
    with pytest.raises(LibraryException, match="negative volatility"):
        OrnsteinUhlenbeckProcess(speed=0.5, vol=-0.01)
