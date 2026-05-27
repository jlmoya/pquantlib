"""Tests for LocalVolCurve — cross-validated against L2-E probe."""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_curve import LocalVolCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l2e")
_LVC = _REF["local_vol_curve"]


def _make_local_vol_curve() -> LocalVolCurve:
    curve = BlackVarianceCurve(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        dates=[
            Date.from_ymd(15, Month.September, 2026),
            Date.from_ymd(15, Month.December, 2026),
            Date.from_ymd(15, Month.June, 2027),
            Date.from_ymd(15, Month.June, 2028),
        ],
        black_vol_curve=[0.10, 0.15, 0.20, 0.25],
        day_counter=Actual365Fixed(),
    )
    return LocalVolCurve(curve)


def test_local_vol_curve_delegates_reference_date() -> None:
    lvc = _make_local_vol_curve()
    assert lvc.reference_date() == Date.from_ymd(15, Month.June, 2026)


def test_local_vol_curve_delegates_max_date() -> None:
    lvc = _make_local_vol_curve()
    assert lvc.max_date() == Date.from_ymd(15, Month.June, 2028)


def test_min_max_strike_are_infinite() -> None:
    lvc = _make_local_vol_curve()
    assert lvc.min_strike() == -math.inf
    assert lvc.max_strike() == math.inf


def test_local_vol_at_t_0p5_matches_cpp() -> None:
    lvc = _make_local_vol_curve()
    # Tolerance: the Dupire estimator uses one-sided FD with dt=1/365.
    # Cross-validation against C++ should be tight since both sides do
    # identical arithmetic.
    tolerance.tight(lvc.local_vol_at_time(0.5, 100.0), _LVC["local_vol_t_0p5"])


def test_local_vol_at_t_0p75_matches_cpp() -> None:
    lvc = _make_local_vol_curve()
    tolerance.tight(lvc.local_vol_at_time(0.75, 100.0), _LVC["local_vol_t_0p75"])


def test_local_vol_at_t_1p5_matches_cpp() -> None:
    lvc = _make_local_vol_curve()
    tolerance.tight(lvc.local_vol_at_time(1.5, 100.0), _LVC["local_vol_t_1p5"])
