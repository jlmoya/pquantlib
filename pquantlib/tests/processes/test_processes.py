"""Tests for the L3-D processes module.

Cross-validates against ``migration-harness/references/cluster/l3d.json``.

C++ parity: ql/processes/{blackscholesprocess,eulerdiscretization}.{hpp,cpp}
            @ v1.42.1 (099987f0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.processes.black_process import BlackProcess
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.black_scholes_process import BlackScholesProcess
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_constant_vol import (
    LocalConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# --- fixtures --------------------------------------------------------------


@pytest.fixture
def reference_data() -> dict[str, Any]:
    """Load the probe-emitted JSON once per test session."""
    return load_reference("cluster/l3d")


def _build_setup() -> tuple[GeneralizedBlackScholesProcess, Date, Date]:
    """Build the (S=100, K=100, T=1, r=5%, q=2%, sigma=20%) GBSM process.

    Matches the C++ probe setup exactly.
    """
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365  # 1 year under Actual/365 Fixed

    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)

    process = GeneralizedBlackScholesProcess(
        x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, ref, expiry


def _build_no_div_setup() -> BlackScholesProcess:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    return BlackScholesProcess(x0=SimpleQuote(100.0), risk_free_ts=rf, black_vol_ts=vol)


def _build_black_setup() -> BlackProcess:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    return BlackProcess(x0=SimpleQuote(100.0), risk_free_ts=rf, black_vol_ts=vol)


def _build_bsm_setup() -> BlackScholesMertonProcess:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    return BlackScholesMertonProcess(
        x0=SimpleQuote(100.0), dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )


# --- GeneralizedBlackScholesProcess ----------------------------------------


def test_gbsm_x0(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    tight(p.x0(), float(reference_data["gbsm_process"]["x0"]))


def test_gbsm_diffusion(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    tight(p.diffusion_1d(0.0, 100.0), float(reference_data["gbsm_process"]["diffusion_t0"]))


def test_gbsm_drift(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    # drift = (r - q) - 0.5 * sigma^2 = (0.05 - 0.02) - 0.5 * 0.04 = 0.01.
    tight(p.drift_1d(0.0, 100.0), float(reference_data["gbsm_process"]["drift_t0"]))


def test_gbsm_expectation(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    # expectation = S * exp((r-q) * 1) = 100 * exp(0.03).
    tight(p.expectation_1d(0.0, 100.0, 1.0), float(reference_data["gbsm_process"]["expectation_t0_dt1"]))


def test_gbsm_variance(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    tight(p.variance_1d(0.0, 100.0, 1.0), float(reference_data["gbsm_process"]["variance_t0_dt1"]))


def test_gbsm_std_deviation(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    tight(
        p.std_deviation_1d(0.0, 100.0, 1.0),
        float(reference_data["gbsm_process"]["std_deviation_t0_dt1"]),
    )


def test_gbsm_evolve_dw0(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    tight(p.evolve_1d(0.0, 100.0, 1.0, 0.0), float(reference_data["gbsm_process"]["evolve_t0_dt1_dw0"]))


def test_gbsm_evolve_dw1(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    tight(p.evolve_1d(0.0, 100.0, 1.0, 1.0), float(reference_data["gbsm_process"]["evolve_t0_dt1_dw1"]))


def test_gbsm_apply(reference_data: dict[str, Any]) -> None:
    p, _, _ = _build_setup()
    tight(p.apply_1d(100.0, 0.2), float(reference_data["gbsm_process"]["apply_x100_dx0p2"]))


def test_gbsm_time_at_expiry(reference_data: dict[str, Any]) -> None:
    p, _, expiry = _build_setup()
    tight(p.time(expiry), float(reference_data["gbsm_process"]["time_at_expiry"]))


def test_gbsm_size_is_1() -> None:
    p, _, _ = _build_setup()
    assert p.size() == 1


def test_gbsm_initial_values_is_vector() -> None:
    p, _, _ = _build_setup()
    iv = p.initial_values()
    assert iv.shape == (1,)
    tight(float(iv[0]), 100.0)


def test_gbsm_inspectors() -> None:
    p, _, _ = _build_setup()
    assert p.state_variable().value() == 100.0
    assert p.dividend_yield().reference_date() == Date.from_ymd(15, Month.June, 2026)
    assert p.risk_free_rate().reference_date() == Date.from_ymd(15, Month.June, 2026)
    assert isinstance(p.local_volatility(), LocalConstantVol)


# --- BlackScholesProcess (no dividends) ------------------------------------


def test_bs_process_x0(reference_data: dict[str, Any]) -> None:
    p = _build_no_div_setup()
    tight(p.x0(), float(reference_data["bs_process"]["x0"]))


def test_bs_process_drift(reference_data: dict[str, Any]) -> None:
    p = _build_no_div_setup()
    # drift = (r - 0) - 0.5 * sigma^2 = 0.05 - 0.02 = 0.03
    tight(p.drift_1d(0.0, 100.0), float(reference_data["bs_process"]["drift_t0"]))


def test_bs_process_expectation(reference_data: dict[str, Any]) -> None:
    p = _build_no_div_setup()
    # expectation = S * exp(r * 1)
    tight(
        p.expectation_1d(0.0, 100.0, 1.0),
        float(reference_data["bs_process"]["expectation_t0_dt1"]),
    )


def test_bs_process_variance(reference_data: dict[str, Any]) -> None:
    p = _build_no_div_setup()
    tight(p.variance_1d(0.0, 100.0, 1.0), float(reference_data["bs_process"]["variance_t0_dt1"]))


# --- BlackProcess (forwards / futures: r = q) ------------------------------


def test_black_process_drift(reference_data: dict[str, Any]) -> None:
    p = _build_black_setup()
    # drift = 0 - 0.5 * sigma^2 = -0.02
    tight(p.drift_1d(0.0, 100.0), float(reference_data["black_process"]["drift_t0"]))


def test_black_process_expectation(reference_data: dict[str, Any]) -> None:
    p = _build_black_setup()
    # expectation = S * exp((r-r) * 1) = S = 100
    tight(
        p.expectation_1d(0.0, 100.0, 1.0),
        float(reference_data["black_process"]["expectation_t0_dt1"]),
    )


# --- BlackScholesMertonProcess (alias for GBSM) ----------------------------


def test_bsm_process_drift(reference_data: dict[str, Any]) -> None:
    p = _build_bsm_setup()
    tight(p.drift_1d(0.0, 100.0), float(reference_data["bsm_process"]["drift_t0"]))


def test_bsm_process_expectation(reference_data: dict[str, Any]) -> None:
    p = _build_bsm_setup()
    tight(
        p.expectation_1d(0.0, 100.0, 1.0),
        float(reference_data["bsm_process"]["expectation_t0_dt1"]),
    )


# --- EulerDiscretization ---------------------------------------------------


def test_euler_drift_scalar(reference_data: dict[str, Any]) -> None:
    p = _build_black_setup()
    euler = EulerDiscretization()
    # drift * dt = (-0.02) * 0.25 = -0.005
    tight(
        euler.drift(p, 0.0, 100.0, 0.25),
        float(reference_data["euler_discretization"]["drift_dt_quarter"]),
    )


def test_euler_diffusion_scalar(reference_data: dict[str, Any]) -> None:
    p = _build_black_setup()
    euler = EulerDiscretization()
    # sigma * sqrt(dt) = 0.20 * 0.5 = 0.10
    tight(
        euler.diffusion(p, 0.0, 100.0, 0.25),
        float(reference_data["euler_discretization"]["diffusion_dt_quarter"]),
    )


def test_euler_variance_scalar(reference_data: dict[str, Any]) -> None:
    p = _build_black_setup()
    euler = EulerDiscretization()
    # sigma^2 * dt = 0.04 * 0.25 = 0.01
    tight(
        euler.variance(p, 0.0, 100.0, 0.25),
        float(reference_data["euler_discretization"]["variance_dt_quarter"]),
    )


def test_euler_drift_array() -> None:
    """The array overload should produce a length-1 array matching the scalar."""
    p = _build_black_setup()
    euler = EulerDiscretization()
    x = np.array([100.0])
    result = euler.drift(p, 0.0, x, 0.25)
    assert isinstance(result, np.ndarray)
    tight(float(result[0]), -0.005)


# --- StochasticProcess1D base dispatchers (vector → scalar) ----------------


def test_stochastic_process_1d_size_and_factors() -> None:
    p, _, _ = _build_setup()
    assert p.size() == 1
    assert p.factors() == 1


def test_stochastic_process_1d_vector_drift_dispatch() -> None:
    """Vector ``drift(t, x_array)`` should dispatch through ``drift_1d``."""
    p, _, _ = _build_setup()
    x = np.array([100.0])
    drift_vec = p.drift(0.0, x)
    drift_scalar = p.drift_1d(0.0, 100.0)
    assert drift_vec.shape == (1,)
    tight(float(drift_vec[0]), drift_scalar)


def test_stochastic_process_1d_vector_diffusion_dispatch() -> None:
    """Vector ``diffusion(t, x_array)`` should return a 1x1 matrix."""
    p, _, _ = _build_setup()
    x = np.array([100.0])
    diff_mat = p.diffusion(0.0, x)
    diff_scalar = p.diffusion_1d(0.0, 100.0)
    assert diff_mat.shape == (1, 1)
    tight(float(diff_mat[0, 0]), diff_scalar)


def test_stochastic_process_1d_vector_evolve_dispatch() -> None:
    """Vector ``evolve(t0, x0, dt, dw)`` should dispatch through ``evolve_1d``."""
    p, _, _ = _build_setup()
    x0 = np.array([100.0])
    dw = np.array([1.0])
    evolved_vec = p.evolve(0.0, x0, 1.0, dw)
    evolved_scalar = p.evolve_1d(0.0, 100.0, 1.0, 1.0)
    tight(float(evolved_vec[0]), evolved_scalar)


# --- base error paths ------------------------------------------------------


class _MinimalProcess1D(StochasticProcess1D):
    """A minimal-impl 1-D process without a discretization, to test errors."""

    def x0(self) -> float:
        return 1.0

    def drift_1d(self, t: float, x: float) -> float:
        _ = t, x
        return 0.0

    def diffusion_1d(self, t: float, x: float) -> float:
        _ = t, x
        return 1.0


def test_no_discretization_raises_on_expectation() -> None:
    p = _MinimalProcess1D()
    with pytest.raises(LibraryException, match="no 1-D discretization"):
        p.expectation_1d(0.0, 1.0, 1.0)


def test_base_time_raises() -> None:
    p = _MinimalProcess1D()
    with pytest.raises(LibraryException, match="date/time conversion not supported"):
        p.time(Date.from_ymd(15, Month.June, 2026))


def test_force_discretization_falls_back_to_euler() -> None:
    """With ``force_discretization=True`` the GBSM should use Euler-based
    variance / std_dev / evolve, but ``expectation_1d`` still raises
    ``not implemented`` to match C++."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    p = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=div,
        risk_free_ts=rf,
        black_vol_ts=vol,
        force_discretization=True,
    )
    # Euler-based variance over dt=1.0: sigma(t=0, x=spot)^2 * dt = 0.04.
    tight(p.variance_1d(0.0, 100.0, 1.0), 0.04)
    # Evolution with dw=1.0 against the Euler step:
    # drift = (r-q) - 0.5*sigma^2 = 0.01, sigma = 0.20.
    # Euler drift*dt = 0.01 * 1.0 = 0.01. std_dev = 0.2.
    # x0 = 100, apply = 100 * exp(0.01 + 0.2 * 1.0).
    expected = 100.0 * math.exp(0.01 + 0.20)
    tight(p.evolve_1d(0.0, 100.0, 1.0, 1.0), expected)
    # expectation still raises (matches C++).
    with pytest.raises(LibraryException, match="not implemented"):
        p.expectation_1d(0.0, 100.0, 1.0)


def test_update_resets_local_vol_cache() -> None:
    """After an update, the cached local volatility should be re-derived."""
    p, _, _ = _build_setup()
    lv1 = p.local_volatility()
    p.update()
    lv2 = p.local_volatility()
    # New instance after update.
    assert lv1 is not lv2
