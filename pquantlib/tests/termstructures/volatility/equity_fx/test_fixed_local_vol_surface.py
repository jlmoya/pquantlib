"""FixedLocalVolSurface behavioral tests.

C++ parity: ql/termstructures/volatility/equityfx/fixedlocalvolsurface.hpp.

Tolerance choice:
* Pillar evaluation: EXACT — matrix lookup is exact.
* Interpolated values: TIGHT — linear interp introduces float64 noise only.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def surface() -> FixedLocalVolSurface:
    """A 3-times x 3-strikes surface centred on spot=100."""
    ref = Date.from_ymd(15, Month.June, 2026)
    dc = Actual365Fixed()
    times = [0.0, 0.5, 1.0]
    strikes = [[80.0, 100.0, 120.0]] * 3
    mat = np.array(
        [
            [0.25, 0.22, 0.20],  # strike 80
            [0.20, 0.18, 0.16],  # strike 100
            [0.22, 0.20, 0.18],  # strike 120
        ],
        dtype=np.float64,
    )
    return FixedLocalVolSurface(
        reference_date=ref,
        times=times,
        strikes=strikes,
        local_vol_matrix=mat,
        day_counter=dc,
    )


def test_pillar_lookup_exact(surface: FixedLocalVolSurface) -> None:
    """Exact match at a grid corner."""
    v = surface.local_vol_at_time(0.0, 100.0, extrapolate=True)
    exact(v, 0.20)
    v = surface.local_vol_at_time(0.5, 80.0, extrapolate=True)
    exact(v, 0.22)
    v = surface.local_vol_at_time(1.0, 120.0, extrapolate=True)
    exact(v, 0.18)


def test_interp_strike_midpoint(surface: FixedLocalVolSurface) -> None:
    """Linear interp midpoint of strike axis at fixed t.

    At t=0, strikes 80→0.25, 100→0.20. Midpoint K=90: (0.25 + 0.20)/2 = 0.225.
    """
    v = surface.local_vol_at_time(0.0, 90.0, extrapolate=True)
    tight(v, 0.225, reason="midpoint linear strike interp")


def test_interp_time_midpoint(surface: FixedLocalVolSurface) -> None:
    """Linear interp midpoint of time axis at fixed K=100.

    At t=0, K=100: 0.20; at t=0.5, K=100: 0.18. Midpoint t=0.25: 0.19.
    """
    v = surface.local_vol_at_time(0.25, 100.0, extrapolate=True)
    tight(v, 0.19, reason="midpoint linear time interp")


def test_extrapolation_constant_in_strike(surface: FixedLocalVolSurface) -> None:
    """Constant extrapolation below min strike."""
    v = surface.local_vol_at_time(0.0, 50.0, extrapolate=True)
    exact(v, 0.25)


def test_extrapolation_constant_in_time(surface: FixedLocalVolSurface) -> None:
    """Constant extrapolation after max time."""
    v = surface.local_vol_at_time(2.0, 100.0, extrapolate=True)
    exact(v, 0.16)  # value at t=1.0, K=100


def test_set_column_updates_surface(surface: FixedLocalVolSurface) -> None:
    """set_column overwrites a time-slice."""
    new_strikes = [70.0, 100.0, 130.0]
    new_col = np.array([0.30, 0.25, 0.28], dtype=np.float64)
    surface.set_column(1, new_strikes, new_col)
    v = surface.local_vol_at_time(0.5, 100.0, extrapolate=True)
    exact(v, 0.25)
    # Min strike should have refreshed to 70.
    exact(surface.min_strike(), 70.0)
