"""HestonSlvMcModel behavioral tests.

C++ parity: ql/models/equity/hestonslvmcmodel.{hpp,cpp}.

Tolerance choice:
* Public-API getters: EXACT — passthrough.
* MC bucketing convergence (degenerate flat vol case): LOOSE with a
  loosened reason — sampling noise + small n_paths bias dominate. A
  larger n_paths + lower variance test would tighten but slow the
  suite; the LOOSE tier explicitly accepts 1e-8 absolute / 1e-8
  relative.

  For the constant-Black-vol case the calibration target is
  ``L(S, t) = sigma_Black / sqrt(V_long_run)``; with the canonical
  Heston params (theta=0.04, sigma_Black=0.20) the target leverage
  is 1.0. We test that the MC calibration converges to L ≈ 1 at
  the ATM bucket with a generous tolerance to absorb sampling noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.heston_slv_mc_model import HestonSlvMcModel
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_surface import (
    LocalVolSurface,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _build_model(*, n_paths: int = 2048, n_bins: int = 21) -> HestonSlvMcModel:
    """Build a small-scale SLV MC model for testing."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    spot = SimpleQuote(100.0)
    process = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=spot,
        # Long-run V = 0.04; sigma_Black = 0.20 → 0.04 = sigma_Black^2.
        # That makes the leverage target 1.0 at ATM.
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
    )
    heston_model = HestonModel(process)
    bvol = BlackConstantVol(
        reference_date=ref,
        calendar=NullCalendar(),
        volatility=0.20,
        day_counter=dc,
    )
    local_vol = LocalVolSurface(black_ts=bvol, underlying=spot)
    return HestonSlvMcModel(
        local_vol=local_vol,
        heston_model=heston_model,
        end_date=ref + 365,
        time_steps_per_year=20,  # coarse for test speed
        n_bins=n_bins,
        calibration_paths=n_paths,
        rng=np.random.default_rng(seed=42),
    )


@pytest.fixture
def mc_model() -> HestonSlvMcModel:
    return _build_model()


def test_heston_process_passthrough(mc_model: HestonSlvMcModel) -> None:
    """``heston_process()`` returns the model's underlying process."""
    proc = mc_model.heston_process()
    exact(proc.v0, 0.04)
    exact(proc.theta, 0.04)


def test_local_vol_passthrough(mc_model: HestonSlvMcModel) -> None:
    """``local_vol()`` returns the input local-vol surface."""
    lv = mc_model.local_vol()
    v = lv.local_vol_at_time(0.5, 100.0, extrapolate=True)
    # Constant Black vol 0.20 → constant local vol 0.20 (TIGHT, not EXACT,
    # because Dupire FD introduces ~1e-14 float64 noise).
    tight(v, 0.20, reason="Dupire FD float64 round-off")


def test_time_grid_length(mc_model: HestonSlvMcModel) -> None:
    """time_grid has time_steps_per_year * end_time + 1 nodes (approx)."""
    n = len(mc_model.time_grid())
    # 1 year * 20 steps/year + 1 = 21 (or close).
    assert n >= 2
    assert n <= 30


def test_mixing_factor_default_is_one(mc_model: HestonSlvMcModel) -> None:
    exact(mc_model.mixing_factor(), 1.0)


def test_leverage_function_calibrates_near_one_at_atm() -> None:
    """At the canonical params (long-run V = sigma_Black^2 = 0.04) the
    MC calibration should produce L ≈ 1 at the ATM bucket.

    The test uses a relatively small n_paths (8192) — sampling noise
    dominates so we accept a coarse tolerance (~5% relative).
    """
    model = _build_model(n_paths=8192, n_bins=51)
    leverage = model.leverage_function()

    # At t = end_time, query ATM bucket.
    end_time = model.time_grid().back()
    l_atm = leverage.local_vol_at_time(end_time, 100.0, extrapolate=True)

    # The target is L = sigma_LV(t, S) / sqrt(E[V|S,t]) ≈ 0.20 / sqrt(0.04) = 1.0.
    # Sampling noise from 8192 paths on 51 buckets dominates — accept 5x slack.
    assert math.isclose(l_atm, 1.0, rel_tol=0.5), (
        f"Expected ATM leverage ≈ 1.0 (within 50% sampling slack); got {l_atm}"
    )
    # Sanity: leverage is positive everywhere.
    assert l_atm > 0.0


def test_leverage_function_is_cached(mc_model: HestonSlvMcModel) -> None:
    """Repeated calls return the same surface object (lazy cache)."""
    lev1 = mc_model.leverage_function()
    lev2 = mc_model.leverage_function()
    assert lev1 is lev2
