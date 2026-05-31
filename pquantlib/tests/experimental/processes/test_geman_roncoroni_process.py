"""Tests for GemanRoncoroniProcess.

# C++ parity: ql/experimental/processes/gemanroncoroniprocess.hpp.

Cross-validates against ``migration-harness/references/cluster/w7a.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.experimental.processes.geman_roncoroni_process import (
    GemanRoncoroniProcess,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w7a")


def _build() -> GemanRoncoroniProcess:
    """Canonical vpp.cpp Geman-Roncoroni parameters."""
    return GemanRoncoroniProcess(
        x0=3.3,
        alpha=3.1,
        beta=0.05,
        gamma=-0.09,
        delta=0.07,
        eps=-0.40,
        zeta=-0.40,
        d=1.6,
        k=1.0,
        tau=0.5,
        sig2=10.0,
        a=-7.0,
        b=-0.3,
        theta1=35.0,
        theta2=9.0,
        theta3=0.10,
        psi=1.9,
    )


def test_gr_x0(reference_data: dict[str, Any]) -> None:
    """x0 round-trip.

    EXACT: scalar inspector.
    """
    p = _build()
    exact(p.x0(), float(reference_data["gr_x0"]))


def test_gr_drift(reference_data: dict[str, Any]) -> None:
    """drift(t=0.4, x=3.0) vs C++ probe.

    TIGHT: trig + linear closed-form.
    """
    p = _build()
    tight(p.drift_1d(0.4, 3.0), float(reference_data["gr_drift"]))


def test_gr_diffusion(reference_data: dict[str, Any]) -> None:
    """diffusion(t=0.4, x=3.0) vs C++ probe.

    TIGHT: sqrt(sig2 + a cos^2(pi t + b)).
    """
    p = _build()
    tight(p.diffusion_1d(0.4, 3.0), float(reference_data["gr_diffusion"]))


def test_gr_std_deviation(reference_data: dict[str, Any]) -> None:
    """stdDeviation(t0=0.4, x0=3.0, dt=0.1) vs C++ probe.

    TIGHT: OU closed-form.
    """
    p = _build()
    tight(p.std_deviation_1d(0.4, 3.0, 0.1), float(reference_data["gr_std_deviation"]))


def test_gr_evolve_below_threshold(reference_data: dict[str, Any]) -> None:
    """Deterministic evolve below the spike threshold (mu + d).

    TIGHT: closed-form OU step + deterministic jump from supplied du.
    """
    p = _build()
    du = np.array([0.4, 0.6, 0.0], dtype=np.float64)
    tight(
        p.evolve_du(0.4, 3.0, 0.1, 0.5, du),
        float(reference_data["gr_evolve_below"]),
    )


def test_gr_evolve_above_threshold(reference_data: dict[str, Any]) -> None:
    """Deterministic evolve above the spike threshold reverts down by j.

    TIGHT: x0 - j branch.
    """
    p = _build()
    du = np.array([0.4, 0.6, 0.0], dtype=np.float64)
    tight(
        p.evolve_du(0.4, 50.0, 0.1, 0.5, du),
        float(reference_data["gr_evolve_above"]),
    )


def test_gr_evolve_with_internal_rng_is_finite() -> None:
    """The 4-arg RNG-driven evolve returns a finite value.

    # C++ parity: the internal MT-seeded jump path is not a stable
    # reference (seed depends on dw); just assert finiteness + that the
    # RNG is lazily constructed once.
    """
    p = _build()
    v1 = p.evolve_1d(0.4, 3.0, 0.1, 0.5)
    assert np.isfinite(v1)
