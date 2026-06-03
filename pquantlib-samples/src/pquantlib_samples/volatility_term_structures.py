"""VolatilityTermStructures sample — explores the volatility term structures.

Port of ``org.jquantlib.samples.VolatilityTermStructures`` (Java AllSamples kept
it in ``complete``). Exercises the equity/FX volatility term structures pquantlib
ships:

* :class:`BlackConstantVol`     — constant Black vol;
* :class:`BlackVarianceCurve`   — term structure of Black vols (linear variance);
* :class:`BlackVarianceSurface` — (date x strike) Black-vol surface;
* :class:`LocalConstantVol`     — constant local vol;
* :class:`LocalVolCurve`        — local vol implied from a variance curve;
* :class:`LocalVolSurface`      — local vol implied from a Black-vol surface.

For each it prints a representative ``blackVol`` / ``blackForwardVol`` /
``blackVariance`` (Black structures) or ``localVol`` (local structures).

Divergence: the Java original also touched ``ImpliedVolTermStructure``, which
pquantlib does not port; that section is omitted (documented inline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_surface import (
    BlackVarianceSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_constant_vol import (
    LocalConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_curve import LocalVolCurve
from pquantlib.termstructures.volatility.equity_fx.local_vol_surface import (
    LocalVolSurface,
)
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib_samples.util.stop_clock import StopClock


@dataclass(frozen=True, slots=True)
class VolResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    black_constant_vol: float
    black_constant_forward_vol: float
    black_constant_variance: float
    black_variance_curve_vol: float
    black_variance_surface_vol: float
    local_constant_vol: float
    local_vol_curve_vol: float
    local_vol_surface_vol: float


def compute() -> VolResult:
    dc = Actual365Fixed()
    nyse = UnitedStates(UnitedStates.Market.NYSE)
    today = Date.todays_date()

    d10 = today + 10
    d12 = today + 12
    d15 = today + 15
    d20 = today + 20
    d25 = today + 25
    d30 = today + 30
    d40 = today + 40
    d50 = today + 50

    # --- BlackConstantVol ------------------------------------------------
    constant_vol = BlackConstantVol(reference_date=today, calendar=nyse, day_counter=dc, volatility=0.30)
    bc_vol = constant_vol.black_vol(d10, 20.0)
    bc_fwd = constant_vol.black_forward_vol(d10, d15, 20.0, True)
    bc_var = constant_vol.black_variance(d10, 20.0)

    # --- BlackVarianceCurve ----------------------------------------------
    bvc = BlackVarianceCurve(
        reference_date=today,
        dates=[d10, d20, d30, d40, d50],
        black_vol_curve=[0.1, 0.2, 0.3, 0.4, 0.5],
        day_counter=dc,
        force_monotone_variance=False,
    )
    bvc_vol = bvc.black_vol(d25, 30.0, True)

    # --- BlackVarianceSurface --------------------------------------------
    strikes = [20.0, 30.0, 40.0]
    dates = [d12, d25, d40]
    # rows = strikes, cols = dates (BlackVarianceSurface convention).
    vol_matrix = np.array(
        [
            [0.10, 0.12, 0.15],
            [0.20, 0.22, 0.25],
            [0.30, 0.32, 0.35],
        ],
        dtype=np.float64,
    )
    bvs = BlackVarianceSurface(
        reference_date=today,
        calendar=NullCalendar(),
        dates=dates,
        strikes=strikes,
        black_vol_matrix=vol_matrix,
        day_counter=dc,
    )
    bvs_vol = bvs.black_vol(d20, 30.0, True)

    # --- LocalConstantVol ------------------------------------------------
    lcv = LocalConstantVol(reference_date=today, volatility=0.30, day_counter=dc)
    lcv_vol = lcv.local_vol(d10, 20.0)

    # --- LocalVolCurve (implied from a variance curve) -------------------
    lvc = LocalVolCurve(bvc)
    lvc_vol = lvc.local_vol(d25, 30.0, True)

    # --- LocalVolSurface (implied from a Black-vol surface) --------------
    lvs = LocalVolSurface(black_ts=bvs, underlying=SimpleQuote(30.0))
    lvs_vol = lvs.local_vol(d20, 30.0, True)

    return VolResult(
        black_constant_vol=bc_vol,
        black_constant_forward_vol=bc_fwd,
        black_constant_variance=bc_var,
        black_variance_curve_vol=bvc_vol,
        black_variance_surface_vol=bvs_vol,
        local_constant_vol=lcv_vol,
        local_vol_curve_vol=lvc_vol,
        local_vol_surface_vol=lvs_vol,
    )


def run() -> None:
    print("::::: VolatilityTermStructures :::::")

    clock = StopClock()
    clock.start_clock()

    r = compute()
    print("//===============================BlackConstantVol===============================")
    print(f"BlackVolatility        = {r.black_constant_vol}")
    print(f"BlackForwardVolatility = {r.black_constant_forward_vol}")
    print(f"BlackVariance          = {r.black_constant_variance}")
    print("//===============================BlackVarianceCurve=============================")
    print(f"BlackVolatility (25d, K=30) = {r.black_variance_curve_vol}")
    print("//===============================BlackVarianceSurface===========================")
    print(f"BlackVolatility (20d, K=30) = {r.black_variance_surface_vol}")
    print("//===============================LocalConstantVol===============================")
    print(f"LocalVolatility = {r.local_constant_vol}")
    print("//===============================LocalVolCurve==================================")
    print(f"LocalVolatility (25d, S=30) = {r.local_vol_curve_vol}")
    print("//===============================LocalVolSurface================================")
    print(f"LocalVolatility (20d, S=30) = {r.local_vol_surface_vol}")

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
