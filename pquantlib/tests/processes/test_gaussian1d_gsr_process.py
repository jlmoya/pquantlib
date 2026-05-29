"""Tests for the Gaussian1dGsrProcess adapter.

Cross-validates against ``migration-harness/references/cluster/w1a.json``.

The adapter is a Gaussian1d-style wrapper around ``GsrProcess``. The
expected behaviour is that under same-reversion configuration the
adapter reproduces ``GsrProcess`` evolution and inspectors bit-for-bit
(TIGHT) — the adapter is delegation, not re-implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.gaussian1d_gsr_process import Gaussian1dGsrProcess
from pquantlib.processes.gsr_process import GsrProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w1a")


def _build_curve() -> FlatForward:
    """Flat 3% curve on 14-May-2026 (same as probe)."""
    return FlatForward(
        Date.from_ymd(14, Month.May, 2026),
        SimpleQuote(0.03),
        Actual365Fixed(),
    )


def _build_const_adapter() -> Gaussian1dGsrProcess:
    """Single-piece Gaussian1dGsrProcess: sigma=0.01, kappa=0.05, T=60."""
    return Gaussian1dGsrProcess(
        term_structure=_build_curve(),
        volstepdates=[],
        volatilities=[0.01],
        reversion=0.05,
        T=60.0,
    )


def test_gaussian1d_gsr_process_inspectors(reference_data: dict[str, Any]) -> None:
    """Adapter forwards sigma/reversion/x0 to GsrProcess."""
    ref = reference_data["gaussian1d_gsr_process"]
    p = _build_const_adapter()
    tight(p.x0(), ref["x0"])
    tight(p.sigma(0.5), ref["sigma_0_5"])
    tight(p.reversion(0.5), ref["reversion_0_5"])


def test_gaussian1d_gsr_process_diffusion(reference_data: dict[str, Any]) -> None:
    """Diffusion equals sigma(t) — x-independent."""
    ref = reference_data["gaussian1d_gsr_process"]
    p = _build_const_adapter()
    tight(p.diffusion_1d(0.5, 0.0), ref["diffusion_0_5"])
    tight(p.diffusion_1d(0.5, 0.7), ref["diffusion_0_5"])


def test_gaussian1d_gsr_process_variance(reference_data: dict[str, Any]) -> None:
    """Variance matches the underlying GsrProcess (constant-rev integral)."""
    ref = reference_data["gaussian1d_gsr_process"]
    p = _build_const_adapter()
    tight(p.variance_1d(0.0, 0.0, 1.0), ref["variance_0_1"])
    tight(p.variance_1d(0.0, 0.0, 5.0), ref["variance_0_5"])
    tight(p.std_deviation_1d(0.0, 0.0, 5.0), ref["std_deviation_0_5"])


def test_gaussian1d_gsr_process_expectation(reference_data: dict[str, Any]) -> None:
    """Expectation matches GsrProcess (with forward-measure correction)."""
    ref = reference_data["gaussian1d_gsr_process"]
    p = _build_const_adapter()
    tight(p.expectation_1d(0.0, 0.0, 1.0), ref["expectation_0_0_1"])
    tight(p.expectation_1d(1.0, 0.0, 1.0), ref["expectation_1_0_2"])


def test_gaussian1d_gsr_process_y_and_g(reference_data: dict[str, Any]) -> None:
    """y(t) and G(t, w) GSR-specific inspectors forwarded correctly."""
    ref = reference_data["gaussian1d_gsr_process"]
    p = _build_const_adapter()
    tight(p.y(5.0), ref["y_5"])
    tight(p.G(0.0, 5.0), ref["G_0_5"])


def test_gaussian1d_gsr_process_equivalence() -> None:
    """Adapter and plain GsrProcess produce identical SDE evolution.

    Same-reversion config: the adapter is delegation, no math
    divergence is possible. EXACT-tier check.
    """
    adapter = _build_const_adapter()
    direct = GsrProcess(
        times=np.array([], dtype=np.float64),
        vols=np.array([0.01], dtype=np.float64),
        reversions=np.array([0.05], dtype=np.float64),
        T=60.0,
    )

    # Sample a grid of (t0, x0, dt) triples.
    for t0 in [0.0, 1.0, 3.0]:
        for x0 in [0.0, 0.3, -0.4]:
            for dt in [0.5, 1.0, 2.0]:
                # Drift / diffusion are independent of x0 in GSR. Sample
                # them at one point only.
                if x0 == 0.0:
                    tight(
                        adapter.drift_1d(t0, x0),
                        direct.drift_1d(t0, x0),
                    )
                    tight(
                        adapter.diffusion_1d(t0, x0),
                        direct.diffusion_1d(t0, x0),
                    )
                tight(
                    adapter.expectation_1d(t0, x0, dt),
                    direct.expectation_1d(t0, x0, dt),
                )
                tight(
                    adapter.variance_1d(t0, x0, dt),
                    direct.variance_1d(t0, x0, dt),
                )


def test_gaussian1d_gsr_process_inner_accessor() -> None:
    """``inner()`` exposes the wrapped GsrProcess for GSR-specific calls."""
    p = _build_const_adapter()
    inner = p.inner()
    assert isinstance(inner, GsrProcess)
    # Round-trip sigma through both surfaces.
    assert p.sigma(0.5) == inner.sigma(0.5)


def test_gaussian1d_gsr_process_piecewise_reversion() -> None:
    """Reversion can be passed as a list[float] aligned to volstepdates."""
    curve = _build_curve()
    today = curve.reference_date()
    p = Gaussian1dGsrProcess(
        term_structure=curve,
        volstepdates=[today + 365, today + 730],
        volatilities=[0.01, 0.012, 0.014],
        reversion=[0.03, 0.05, 0.07],
        T=60.0,
    )
    # The first piece (t < 1Y) uses reversion[0] = 0.03.
    tight(p.reversion(0.5), 0.03)
    # The middle piece uses 0.05.
    tight(p.reversion(1.5), 0.05)
    # The last piece uses 0.07.
    tight(p.reversion(2.5), 0.07)
