"""GjrGarchProcess behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/w1d.json``.

C++ parity: ql/processes/gjrgarchprocess.{hpp,cpp} @ v1.42.1 (099987f0).

Tolerance choice:

* Scalar parameter accessors: EXACT — straight passthrough.
* Drift / diffusion entries: TIGHT — closed-form scalar arithmetic.
* initial_values: EXACT (multiplication daysPerYear * v0 is a single op).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.gjr_garch_process import Discretization, GjrGarchProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/w1d")


@pytest.fixture
def process() -> GjrGarchProcess:
    """Duan et al. (2006) testbed with daily constants."""
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
        discretization=Discretization.FullTruncation,
    )


def test_size_and_factors(process: GjrGarchProcess) -> None:
    assert process.size() == 2
    assert process.factors() == 2


def test_parameter_accessors(
    process: GjrGarchProcess, cpp_refs: dict[str, Any]
) -> None:
    p = cpp_refs["gjr_garch_process"]
    exact(process.v0, p["v0"])
    exact(process.omega, p["omega"])
    exact(process.alpha, p["alpha"])
    exact(process.beta, p["beta"])
    exact(process.gamma, p["gamma"])
    exact(process.lambda_, p["lambda"])
    exact(process.days_per_year, p["daysPerYear"])


def test_initial_values(
    process: GjrGarchProcess, cpp_refs: dict[str, Any]
) -> None:
    """initial_values = (S0, daysPerYear * v0).

    # C++ parity: gjrgarchprocess.cpp:47-49.
    """
    p = cpp_refs["gjr_garch_process"]
    iv = process.initial_values()
    exact(float(iv[0]), p["initial_values_S"])
    exact(float(iv[1]), p["initial_values_V"])


def test_drift_at_t_half_year(
    process: GjrGarchProcess, cpp_refs: dict[str, Any]
) -> None:
    """drift at t=0.5y with x=(100, daysPerYear*v0).

    # C++ parity: gjrgarchprocess.cpp:51-69.
    """
    p = cpp_refs["gjr_garch_process"]
    x = np.array([100.0, p["initial_values_V"]], dtype=np.float64)
    d = process.drift(0.5, x)
    tight(
        float(d[0]),
        p["drift_S"],
        reason="closed-form scalar arithmetic",
    )
    tight(
        float(d[1]),
        p["drift_V"],
        reason="closed-form scalar arithmetic",
    )


def test_diffusion_at_t_half_year(
    process: GjrGarchProcess, cpp_refs: dict[str, Any]
) -> None:
    """diffusion at t=0.5y with x=(100, daysPerYear*v0).

    Lower-triangular: (0,1) entry is exactly 0.

    # C++ parity: gjrgarchprocess.cpp:71-101.
    """
    p = cpp_refs["gjr_garch_process"]
    x = np.array([100.0, p["initial_values_V"]], dtype=np.float64)
    diff = process.diffusion(0.5, x)
    tight(
        float(diff[0, 0]),
        p["diffusion_00"],
        reason="vol = sqrt(V)",
    )
    exact(float(diff[0, 1]), p["diffusion_01"])
    tight(
        float(diff[1, 0]),
        p["diffusion_10"],
        reason="rho1 — closed-form scalar combination",
    )
    tight(
        float(diff[1, 1]),
        p["diffusion_11"],
        reason="rho2 — closed-form scalar combination",
    )


def test_apply_log_step_on_spot() -> None:
    """apply: S = S0 * exp(dx_S); V = V0 + dx_V.

    # C++ parity: gjrgarchprocess.cpp:103-105.
    """
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    proc = GjrGarchProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.000160,
        omega=0.000002,
        alpha=0.024,
        beta=0.93,
        gamma=0.059,
        lambda_=0.2,
    )
    x0 = np.array([100.0, 0.04032], dtype=np.float64)
    dx = np.array([0.01, 0.001], dtype=np.float64)
    out = proc.apply(x0, dx)
    exact(float(out[0]), 100.0 * np.exp(0.01))
    exact(float(out[1]), 0.04032 + 0.001)


def test_discretization_default_is_full_truncation() -> None:
    """Default discretization is ``FullTruncation``.

    # C++ parity: gjrgarchprocess.hpp:76 — d = FullTruncation.
    """
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    proc = GjrGarchProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.000160,
        omega=0.000002,
        alpha=0.024,
        beta=0.93,
        gamma=0.059,
        lambda_=0.2,
    )
    assert proc.discretization == Discretization.FullTruncation
