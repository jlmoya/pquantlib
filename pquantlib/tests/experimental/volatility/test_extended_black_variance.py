"""Tests for ExtendedBlackVarianceCurve / ExtendedBlackVarianceSurface.

# C++ parity: ql/experimental/volatility/extendedblackvariance{curve,surface}.*
# (v1.42.1).

Cross-validated against the W6-B C++ probe (``cluster/w6b``):

- Curve: black vol / variance at the input pillar dates reproduce the
  input vols (TIGHT); interpolated points match C++ Linear (LOOSE);
  flat-variance extrapolation beyond the last pillar (LOOSE).
- Surface: pillar vols reproduce inputs (TIGHT); interior bilinear
  point matches the documented-correct C++ reference (LOOSE).

The C++ ``ExtendedBlackVarianceSurface`` aborts on construction (OOB bug),
so the surface reference values were computed inline in the probe using
the documented-correct variance grid + Bilinear interpolation — exactly
what the Python port implements.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.volatility.extended_black_variance_curve import (
    ExtendedBlackVarianceCurve,
)
from pquantlib.experimental.volatility.extended_black_variance_surface import (
    ExtendedBlackVarianceSurface,
    Extrapolation,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/w6b")
_CURVE = _REF["extended_black_variance_curve"]
_SURFACE = _REF["extended_black_variance_surface"]

_TODAY = Date.from_ymd(15, Month.May, 2026)
_DC = Actual365Fixed()


# --- ExtendedBlackVarianceCurve ----------------------------------------


def _build_curve() -> ExtendedBlackVarianceCurve:
    dates = [
        _TODAY + Period(6, TimeUnit.Months),
        _TODAY + Period(1, TimeUnit.Years),
        _TODAY + Period(2, TimeUnit.Years),
        _TODAY + Period(3, TimeUnit.Years),
    ]
    vols = [SimpleQuote(v) for v in _CURVE["input_vols"]]
    return ExtendedBlackVarianceCurve(
        reference_date=_TODAY,
        dates=dates,
        volatilities=vols,
        day_counter=_DC,
        force_monotone_variance=True,
    )


def test_extended_black_variance_curve_pillars_reproduce_inputs() -> None:
    curve = _build_curve()
    for t, vol_ref, var_ref in zip(
        _CURVE["pillar_times"],
        _CURVE["pillar_vols"],
        _CURVE["pillar_variances"],
        strict=True,
    ):
        # at the pillar times the vol must equal the input vol exactly.
        tolerance.tight(curve.black_vol_at_time(t, 0.0, True), vol_ref)
        tolerance.tight(curve.black_variance_at_time(t, 0.0, True), var_ref)


def test_extended_black_variance_curve_interpolated_points() -> None:
    curve = _build_curve()
    for t, vol_ref, var_ref in zip(
        _CURVE["interp_times"],
        _CURVE["interp_vols"],
        _CURVE["interp_variances"],
        strict=True,
    ):
        # interpolation arm — LOOSE vs C++ Linear interpolation.
        tolerance.loose(curve.black_vol_at_time(t, 0.0, True), vol_ref)
        tolerance.loose(curve.black_variance_at_time(t, 0.0, True), var_ref)


def test_extended_black_variance_curve_extrapolation() -> None:
    curve = _build_curve()
    var = curve.black_variance_at_time(_CURVE["extrap_t"], 0.0, True)
    tolerance.loose(var, _CURVE["extrap_variance"])


def test_extended_black_variance_curve_monotone_check() -> None:
    dates = [_TODAY + Period(1, TimeUnit.Years), _TODAY + Period(2, TimeUnit.Years)]
    # decreasing variance: vol drops enough that t*vol^2 decreases.
    vols = [SimpleQuote(0.40), SimpleQuote(0.10)]
    with pytest.raises(LibraryException):
        ExtendedBlackVarianceCurve(
            reference_date=_TODAY,
            dates=dates,
            volatilities=vols,
            day_counter=_DC,
            force_monotone_variance=True,
        )


def test_extended_black_variance_curve_quote_update_refreshes() -> None:
    dates = [_TODAY + Period(1, TimeUnit.Years), _TODAY + Period(2, TimeUnit.Years)]
    q0 = SimpleQuote(0.20)
    q1 = SimpleQuote(0.22)
    curve = ExtendedBlackVarianceCurve(
        reference_date=_TODAY,
        dates=dates,
        volatilities=[q0, q1],
        day_counter=_DC,
    )
    t1 = _DC.year_fraction(_TODAY, dates[0])
    tolerance.tight(curve.black_vol_at_time(t1, 0.0, True), 0.20)
    # bump the first quote -> the curve refreshes via observer wiring.
    q0.set_value(0.30)
    tolerance.tight(curve.black_vol_at_time(t1, 0.0, True), 0.30)


# --- ExtendedBlackVarianceSurface --------------------------------------


def _build_surface() -> ExtendedBlackVarianceSurface:
    dates = [_TODAY + Period(1, TimeUnit.Years), _TODAY + Period(2, TimeUnit.Years)]
    strikes = list(_SURFACE["strikes"])
    # input_vols_rowmajor is [strike_i][date_j] flattened.
    flat = _SURFACE["input_vols_rowmajor"]
    n_dates = len(dates)
    grid = [
        [SimpleQuote(flat[i * n_dates + j]) for j in range(n_dates)]
        for i in range(len(strikes))
    ]
    return ExtendedBlackVarianceSurface(
        reference_date=_TODAY,
        calendar=TARGET(),
        dates=dates,
        strikes=strikes,
        vol_matrix=grid,
        day_counter=_DC,
        lower_extrapolation=Extrapolation.ConstantExtrapolation,
        upper_extrapolation=Extrapolation.ConstantExtrapolation,
    )


def test_extended_black_variance_surface_pillars_reproduce_inputs() -> None:
    surface = _build_surface()
    strikes = list(_SURFACE["strikes"])
    times = list(_SURFACE["pillar_times"])
    # probe order: outer loop strikes, inner loop dates.
    idx = 0
    for k in strikes:
        for t in times:
            tolerance.tight(
                surface.black_vol_at_time(t, k, True), _SURFACE["pillar_vols"][idx]
            )
            tolerance.tight(
                surface.black_variance_at_time(t, k, True),
                _SURFACE["pillar_variances"][idx],
            )
            idx += 1


def test_extended_black_variance_surface_interior_point() -> None:
    surface = _build_surface()
    tolerance.loose(
        surface.black_vol_at_time(_SURFACE["interior_t"], _SURFACE["interior_k"], True),
        _SURFACE["interior_vol"],
    )
    tolerance.loose(
        surface.black_variance_at_time(
            _SURFACE["interior_t"], _SURFACE["interior_k"], True
        ),
        _SURFACE["interior_variance"],
    )


def test_extended_black_variance_surface_constant_strike_extrapolation() -> None:
    surface = _build_surface()
    strikes = list(_SURFACE["strikes"])
    t = _SURFACE["pillar_times"][0]
    # strike below the grid clamps to the lowest strike's vol.
    below = surface.black_vol_at_time(t, strikes[0] - 50.0, True)
    at_low = surface.black_vol_at_time(t, strikes[0], True)
    tolerance.tight(below, at_low)
    # strike above the grid clamps to the highest strike's vol.
    above = surface.black_vol_at_time(t, strikes[-1] + 50.0, True)
    at_high = surface.black_vol_at_time(t, strikes[-1], True)
    tolerance.tight(above, at_high)
