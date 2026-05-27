"""Tests for LocalVolSurface — cross-validated against L2-E probe.

L2-E ports the flat-curve simplification of LocalVolSurface (zero
risk-free + zero dividend ⇒ forward = spot). The C++ probe used
``FlatForward(0%)`` curves on both sides, so the reference values
agree with our flat-curve impl.
"""

from __future__ import annotations

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.equity_fx.black_variance_surface import (
    BlackVarianceSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_surface import LocalVolSurface
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l2e")
_LVS = _REF["local_vol_surface"]


def _make_local_vol_surface() -> LocalVolSurface:
    surf = BlackVarianceSurface(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        dates=[
            Date.from_ymd(15, Month.September, 2026),
            Date.from_ymd(15, Month.December, 2026),
            Date.from_ymd(15, Month.June, 2027),
            Date.from_ymd(15, Month.June, 2028),
        ],
        strikes=[80.0, 100.0, 120.0],
        black_vol_matrix=np.asarray(
            [
                [0.20, 0.21, 0.22, 0.23],
                [0.10, 0.15, 0.20, 0.25],
                [0.20, 0.21, 0.22, 0.23],
            ],
            dtype=np.float64,
        ),
        day_counter=Actual365Fixed(),
    )
    return LocalVolSurface(black_ts=surf, underlying=100.0)


def test_local_vol_surface_delegates_reference_date() -> None:
    lvs = _make_local_vol_surface()
    assert lvs.reference_date() == Date.from_ymd(15, Month.June, 2026)


def test_local_vol_surface_delegates_max_date() -> None:
    lvs = _make_local_vol_surface()
    assert lvs.max_date() == Date.from_ymd(15, Month.June, 2028)


def test_local_vol_surface_delegates_min_max_strike() -> None:
    lvs = _make_local_vol_surface()
    assert lvs.min_strike() == 80.0
    assert lvs.max_strike() == 120.0


def test_local_vol_at_t_0p5_s100_matches_cpp() -> None:
    lvs = _make_local_vol_surface()
    # Dupire on a bilinear-variance surface with central FD in K and t.
    # LOOSE tier (1e-8) — finite-difference epsilon^2 error dominates,
    # and Python/C++ may sequence the same FP ops slightly differently
    # (the reference values come from the C++ probe with FlatForward(0%)
    # for both risk-free and dividend).
    tolerance.loose(
        lvs.local_vol_at_time(0.5, 100.0, extrapolate=True), _LVS["local_vol_t_0p5_s100"]
    )


def test_local_vol_at_t_0p75_s100_matches_cpp() -> None:
    lvs = _make_local_vol_surface()
    tolerance.loose(
        lvs.local_vol_at_time(0.75, 100.0, extrapolate=True),
        _LVS["local_vol_t_0p75_s100"],
    )


def test_local_vol_at_t_1p0_s100_matches_cpp() -> None:
    lvs = _make_local_vol_surface()
    tolerance.loose(
        lvs.local_vol_at_time(1.0, 100.0, extrapolate=True),
        _LVS["local_vol_t_1p0_s100"],
    )
