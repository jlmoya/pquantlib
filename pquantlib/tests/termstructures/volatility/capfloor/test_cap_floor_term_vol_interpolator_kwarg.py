"""L9-A opt-in: interpolator= kwarg wired into CapFloorTermVolCurve + Surface.

Smoke-test the L9-A opt-in upgrade to the L8-C capfloor surfaces.
Backward compatibility: the default kwargs (Linear / Bilinear) reproduce
the L8-C behaviour to TIGHT (verified by the unchanged L8-C test
suites). These tests verify that passing the cubic-family interpolators
from L9-A succeeds, evaluates within the LOOSE neighborhood of the
linear default at pillar nodes (which both interpolators pass through),
and produces sensibly-different intermediate values.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.math.interpolations.cubic_interpolation import (
    CubicNaturalSpline,
    MonotonicCubicNaturalSpline,
)
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_curve import (
    CapFloorTermVolCurve,
)
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_surface import (
    CapFloorTermVolSurface,
)
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _ref_date() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


def _curve_inputs() -> dict[str, object]:
    return {
        "business_day_convention": BusinessDayConvention.ModifiedFollowing,
        "option_tenors": [
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(3, TimeUnit.Years),
            Period(5, TimeUnit.Years),
        ],
        "vols": [0.18, 0.19, 0.20, 0.22],
        "calendar": TARGET(),
        "day_counter": Actual365Fixed(),
        "reference_date": _ref_date(),
    }


@pytest.mark.parametrize(
    "interp_cls",
    [CubicNaturalSpline, MonotonicCubicNaturalSpline],
)
def test_curve_accepts_cubic_interpolators(
    interp_cls: type[CubicNaturalSpline] | type[MonotonicCubicNaturalSpline],
) -> None:
    curve = CapFloorTermVolCurve(**_curve_inputs(), interpolator=interp_cls)  # type: ignore[arg-type]
    # At each pillar, all interpolators pass through the input vols.
    d2y = TARGET().advance_period(_ref_date(), Period(2, TimeUnit.Years))
    expected_2y = 0.19
    actual_2y = curve.volatility(d2y, 0.05, extrapolate=True)
    tolerance.tight(actual_2y, expected_2y)


def test_curve_default_is_linear() -> None:
    # Backward compatibility — default kwargs reproduce L8-C exactly.
    default_curve = CapFloorTermVolCurve(**_curve_inputs())  # type: ignore[arg-type]
    # Intermediate maturity between 2y and 3y pillars.
    d_mid = TARGET().advance_period(_ref_date(), Period(30, TimeUnit.Months))
    # Linear interpolation at the midpoint of 0.19 and 0.20 should be
    # approximately their average (the year-fractions are not exactly
    # symmetric so we tier LOOSE).
    v_mid = default_curve.volatility(d_mid, 0.05, extrapolate=True)
    assert 0.185 < v_mid < 0.205


def test_curve_cubic_off_pillar_differs_from_linear() -> None:
    # Same curve, two interpolators — across off-pillar points the
    # cubic spline must produce some non-trivial difference vs the
    # linear curve (otherwise the kwarg wiring is silently broken).
    inputs = _curve_inputs()
    linear_curve = CapFloorTermVolCurve(**inputs)  # type: ignore[arg-type]
    cubic_curve = CapFloorTermVolCurve(**inputs, interpolator=CubicNaturalSpline)  # type: ignore[arg-type]
    # Sample multiple off-pillar maturities; with a 4-point input the
    # cubic and linear should diverge at least at one point — pick a
    # point far from the midpoint where linear and cubic coincide.
    d_mid_a = TARGET().advance_period(_ref_date(), Period(18, TimeUnit.Months))
    d_mid_b = TARGET().advance_period(_ref_date(), Period(42, TimeUnit.Months))
    diff_a = abs(
        linear_curve.volatility(d_mid_a, 0.05, extrapolate=True)
        - cubic_curve.volatility(d_mid_a, 0.05, extrapolate=True)
    )
    diff_b = abs(
        linear_curve.volatility(d_mid_b, 0.05, extrapolate=True)
        - cubic_curve.volatility(d_mid_b, 0.05, extrapolate=True)
    )
    # At least one off-pillar point must show a measurable difference.
    assert max(diff_a, diff_b) > 1e-7, (
        f"cubic and linear shouldn't agree at all off-pillar points "
        f"(got diffs {diff_a:.3e}, {diff_b:.3e})"
    )
    # Both should stay near linear (smooth vol curve).
    assert max(diff_a, diff_b) < 0.05, "cubic should stay near linear"


def _surface_inputs() -> dict[str, object]:
    # 4-tenor x 4-strike grid — bicubic needs at least 4 in each axis.
    return {
        "business_day_convention": BusinessDayConvention.ModifiedFollowing,
        "option_tenors": [
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(3, TimeUnit.Years),
            Period(5, TimeUnit.Years),
        ],
        "strikes": [0.02, 0.03, 0.04, 0.05],
        "volatilities": np.array(
            [
                [0.20, 0.19, 0.18, 0.17],
                [0.22, 0.21, 0.20, 0.19],
                [0.24, 0.23, 0.22, 0.21],
                [0.28, 0.27, 0.26, 0.25],
            ],
            dtype=np.float64,
        ),
        "calendar": TARGET(),
        "day_counter": Actual365Fixed(),
        "reference_date": _ref_date(),
    }


def test_surface_accepts_bicubic_interpolator() -> None:
    surf = CapFloorTermVolSurface(**_surface_inputs(), interpolator=BicubicSpline)  # type: ignore[arg-type]
    # At pillar (2y, 0.03) the surface must roundtrip to the input vol 0.21.
    d2y = TARGET().advance_period(_ref_date(), Period(2, TimeUnit.Years))
    v = surf.volatility(d2y, 0.03, extrapolate=True)
    tolerance.tight(v, 0.21)


def test_surface_default_is_bilinear() -> None:
    surf = CapFloorTermVolSurface(**_surface_inputs())  # type: ignore[arg-type]
    d2y = TARGET().advance_period(_ref_date(), Period(2, TimeUnit.Years))
    v = surf.volatility(d2y, 0.03, extrapolate=True)
    tolerance.tight(v, 0.21)


def test_surface_bicubic_off_pillar_differs_from_bilinear() -> None:
    inputs = _surface_inputs()
    bilinear_surf = CapFloorTermVolSurface(**inputs)  # type: ignore[arg-type]
    bicubic_surf = CapFloorTermVolSurface(**inputs, interpolator=BicubicSpline)  # type: ignore[arg-type]
    # Sample multiple off-pillar (time, strike) points; at least one
    # must show a measurable bicubic-vs-bilinear difference.
    d_18m = TARGET().advance_period(_ref_date(), Period(18, TimeUnit.Months))
    d_42m = TARGET().advance_period(_ref_date(), Period(42, TimeUnit.Months))
    diffs: list[float] = []
    for d in (d_18m, d_42m):
        for k in (0.025, 0.035, 0.045):
            diffs.append(
                abs(
                    bilinear_surf.volatility(d, k, extrapolate=True)
                    - bicubic_surf.volatility(d, k, extrapolate=True)
                )
            )
    # At least one off-pillar (time, strike) shows a non-trivial diff.
    assert max(diffs) > 1e-7, (
        f"bicubic and bilinear shouldn't agree at all off-pillar points "
        f"(max diff {max(diffs):.3e})"
    )
    # All stay near (smooth surface).
    assert max(diffs) < 0.05, "bicubic should stay near bilinear"
