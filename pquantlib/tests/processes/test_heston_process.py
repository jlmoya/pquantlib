"""HestonProcess behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/l4c.json``.

C++ parity: ql/processes/hestonprocess.{hpp,cpp} @ v1.42.1 (099987f0).

Tolerance choice:

* Scalar drift / diffusion / apply: TIGHT — these are direct
  arithmetic that should match within float64 rounding.
* ``initial_values`` and parameter accessors: EXACT — straight passthrough.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/l4c")


@pytest.fixture
def process() -> HestonProcess:
    """The canonical Albrecher-Mayer-Schoutens-Tistaert testbed."""
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


def test_size_and_factors(process: HestonProcess, cpp_refs: dict[str, Any]) -> None:
    p = cpp_refs["process"]
    assert process.size() == p["size"]
    assert process.factors() == p["factors"]


def test_initial_values(process: HestonProcess, cpp_refs: dict[str, Any]) -> None:
    p = cpp_refs["process"]
    iv = process.initial_values()
    exact(float(iv[0]), p["initial_s"])
    exact(float(iv[1]), p["initial_v"])


def test_parameter_accessors(process: HestonProcess, cpp_refs: dict[str, Any]) -> None:
    p = cpp_refs["process"]
    exact(process.v0, p["v0"])
    exact(process.kappa, p["kappa"])
    exact(process.theta, p["theta"])
    exact(process.sigma, p["sigma"])
    exact(process.rho, p["rho"])


def test_drift_at_initial_state(process: HestonProcess, cpp_refs: dict[str, Any]) -> None:
    p = cpp_refs["process"]
    x = np.array([100.0, 0.04], dtype=np.float64)
    d = process.drift(0.0, x)
    tight(
        float(d[0]),
        p["drift_s"],
        reason="r - q - 0.5*V at initial state; tight rounding",
    )
    tight(
        float(d[1]),
        p["drift_v"],
        reason="kappa*(theta - V) — should be ~0 at V=theta",
    )


def test_diffusion_matrix(process: HestonProcess, cpp_refs: dict[str, Any]) -> None:
    p = cpp_refs["process"]
    x = np.array([100.0, 0.04], dtype=np.float64)
    diff = process.diffusion(0.0, x)
    tight(float(diff[0, 0]), p["diffusion_00"])
    exact(float(diff[0, 1]), p["diffusion_01"])
    tight(float(diff[1, 0]), p["diffusion_10"])
    tight(float(diff[1, 1]), p["diffusion_11"])


def test_apply_increment(process: HestonProcess, cpp_refs: dict[str, Any]) -> None:
    p = cpp_refs["process"]
    x0 = np.array([100.0, 0.04], dtype=np.float64)
    dx = np.array([0.01, 0.01], dtype=np.float64)
    applied = process.apply(x0, dx)
    tight(
        float(applied[0]),
        p["apply_s"],
        reason="S = S0 * exp(0.01) = 101.00501670841679",
    )
    exact(float(applied[1]), p["apply_v"])


def test_diffusion_clamps_negative_variance(process: HestonProcess) -> None:
    """When V <= 0, diffusion preserves correlation via 1e-8 sentinel.

    # C++ parity: hestonprocess.cpp:92-95 — "set vol to (almost) zero
    # but still expose some correlation information".
    """
    x_neg = np.array([100.0, -0.001], dtype=np.float64)
    diff = process.diffusion(0.0, x_neg)
    # diffusion[0][0] should be near zero (vol = 1e-8) but non-zero.
    assert 0.0 < float(diff[0, 0]) <= 1e-7
    # diffusion[1][1] proportional to sigma * vol → nonzero.
    assert float(diff[1, 1]) > 0.0


def test_drift_truncates_negative_variance(process: HestonProcess) -> None:
    """When V < 0 in drift, vol is 0 (FullTruncation).

    Tolerance is LOOSE here because the rate component uses a finite-
    difference forward over a 1e-4 window (the L3-D YieldTermStructure
    workaround), which introduces ~1e-13 noise relative to the exact
    flat-rate value. The drift_v term is exact (no rate lookup).
    """
    x_neg = np.array([100.0, -0.5], dtype=np.float64)
    d = process.drift(0.0, x_neg)
    # vol = 0 → drift_s = r - q - 0 = 0.05
    # drift_v = kappa*(theta - 0) = 2*0.04 = 0.08
    loose(float(d[0]), 0.05, reason="r - q with V truncated; FD-forward noise")
    tight(float(d[1]), 0.08, reason="kappa*theta with V truncated to 0")


def test_time_to_expiry_matches_curve_daycounter(
    process: HestonProcess, cpp_refs: dict[str, Any]
) -> None:
    p = cpp_refs["process"]
    expiry = Date.from_ymd(15, Month.June, 2026) + 365
    exact(process.time(expiry), p["time_to_expiry"])


def test_quotes_and_curves_observable(process: HestonProcess) -> None:
    """The process registers itself as an observer of its market data.

    Smoke test — when the spot quote changes, the process should
    receive an update (observable plumbing matters for downstream
    engines like AnalyticHestonEngine).
    """
    spot = process.s0()
    assert isinstance(spot, SimpleQuote)
    # Mutating the quote should not raise — observability plumbing OK.
    spot.set_value(105.0)
    iv = process.initial_values()
    exact(float(iv[0]), 105.0)
