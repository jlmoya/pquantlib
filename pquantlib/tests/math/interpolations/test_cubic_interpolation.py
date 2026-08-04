"""Cross-validate CubicNaturalSpline / MonotonicCubicNaturalSpline against L9-A.

Reference: ``migration-harness/references/cluster/l9a.json`` —
``cubic_natural_spline`` and ``monotonic_cubic_natural_spline`` sections.

The probe uses 5 sorted x-knots at integer positions and assorted
y-values. C++ values at the pillar nodes equal the input y to EXACT;
this port reproduces y[i] to TIGHT (round-off). Intermediate values agree
TIGHT for both splines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    CubicNaturalSpline,
    DerivativeApprox,
    MonotonicCubicNaturalSpline,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l9a")


# ---------------------------------------------------------------------------
# CubicNaturalSpline — Spline + Natural BC + non-monotonic.
# ---------------------------------------------------------------------------


def _make_natural(cpp: dict[str, Any]) -> CubicNaturalSpline:
    block = cpp["cubic_natural_spline"]
    xs = np.asarray(block["xs"], dtype=np.float64)
    ys = np.asarray(block["ys"], dtype=np.float64)
    return CubicNaturalSpline(xs, ys)


def test_cubic_natural_spline_pillars_tight(cpp: dict[str, Any]) -> None:
    # At pillar nodes the spline reproduces the input y to TIGHT.
    # (Bit-EXACT isn't promised because scipy's natural-BC tridiagonal
    # solve has accumulated round-off; we observe agreement to ~1e-15.)
    block = cpp["cubic_natural_spline"]
    xs = block["xs"]
    ys = block["ys"]
    interp = _make_natural(cpp)
    for x, y in zip(xs, ys, strict=True):
        tolerance.tight(interp(float(x)), float(y))


def test_cubic_natural_spline_pillars_match_cpp_tight(cpp: dict[str, Any]) -> None:
    # scipy and C++ agree at pillars to TIGHT.
    block = cpp["cubic_natural_spline"]
    xs = block["xs"]
    pillars = block["pillars"]
    interp = _make_natural(cpp)
    for x, p in zip(xs, pillars, strict=True):
        tolerance.tight(interp(float(x)), float(p))


def test_cubic_natural_spline_intermediates_match_cpp_tight(cpp: dict[str, Any]) -> None:
    # Both C++ and scipy solve the same Natural-BC tridiagonal system;
    # values at intermediate x agree to TIGHT (~1e-14).
    block = cpp["cubic_natural_spline"]
    mids_x = block["mids_x"]
    mids_y = block["mids_y"]
    interp = _make_natural(cpp)
    for x, y in zip(mids_x, mids_y, strict=True):
        tolerance.tight(interp(float(x)), float(y))


def test_cubic_natural_spline_derivative_tight(cpp: dict[str, Any]) -> None:
    interp = _make_natural(cpp)
    block = cpp["cubic_natural_spline"]
    tolerance.tight(interp.derivative(1.5), float(block["derivative_at_1_5"]))


def test_cubic_natural_spline_second_derivative_tight(cpp: dict[str, Any]) -> None:
    interp = _make_natural(cpp)
    block = cpp["cubic_natural_spline"]
    tolerance.tight(interp.second_derivative(1.5), float(block["second_derivative_at_1_5"]))


def test_cubic_natural_spline_primitive_tight(cpp: dict[str, Any]) -> None:
    # primitive(x) = ∫_{x0}^{x} s(t) dt with primitiveConst_[0] = 0.
    interp = _make_natural(cpp)
    block = cpp["cubic_natural_spline"]
    tolerance.tight(interp.primitive(2.5), float(block["primitive_at_2_5"]))


def test_cubic_natural_spline_update_refreshes() -> None:
    # Mutating the underlying arrays + calling update() rebuilds the spline.
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)  # y = x^2
    interp = CubicNaturalSpline(xs, ys)
    v_before = interp(1.5)
    # Mutate y in place and rebuild — exercise the update() hook.
    ys2 = np.array([0.0, 2.0, 8.0, 18.0], dtype=np.float64)  # 2 * x^2
    interp._ys[:] = ys2  # pyright: ignore[reportPrivateUsage]
    interp.update()
    v_after = interp(1.5)
    # v_after should be 2 * v_before (linear in y by spline linearity).
    tolerance.tight(v_after, 2.0 * v_before)


# ---------------------------------------------------------------------------
# MonotonicCubicNaturalSpline — natural cubic spline + Hyman 1983 filter.
# ---------------------------------------------------------------------------


def _make_monotonic(cpp: dict[str, Any]) -> MonotonicCubicNaturalSpline:
    block = cpp["monotonic_cubic_natural_spline"]
    xs = np.asarray(block["xs"], dtype=np.float64)
    ys = np.asarray(block["ys"], dtype=np.float64)
    return MonotonicCubicNaturalSpline(xs, ys)


def test_monotonic_cubic_pillars_tight(cpp: dict[str, Any]) -> None:
    block = cpp["monotonic_cubic_natural_spline"]
    xs = block["xs"]
    ys = block["ys"]
    interp = _make_monotonic(cpp)
    for x, y in zip(xs, ys, strict=True):
        tolerance.tight(interp(float(x)), float(y))


def test_monotonic_cubic_pillars_match_cpp_tight(cpp: dict[str, Any]) -> None:
    block = cpp["monotonic_cubic_natural_spline"]
    xs = block["xs"]
    pillars = block["pillars"]
    interp = _make_monotonic(cpp)
    for x, p in zip(xs, pillars, strict=True):
        tolerance.tight(interp(float(x)), float(p))


def test_monotonic_cubic_intermediates_match_cpp_tight(cpp: dict[str, Any]) -> None:
    # TIGHT. Off-pillar values are where the monotonic cubic's *algorithm*
    # shows, and this port now runs C++'s own one — the natural-spline
    # tridiagonal solve plus the Hyman 1983 filter. It previously delegated
    # to scipy's Fritsch-Carlson PCHIP, which shares the pillars but is a
    # different function between them (relative error up to ~0.2 here), so
    # this assertion was a 0.2-wide envelope. It is now plain agreement.
    block = cpp["monotonic_cubic_natural_spline"]
    mids_x = block["mids_x"]
    mids_y = block["mids_y"]
    interp = _make_monotonic(cpp)
    for x, y_cpp in zip(mids_x, mids_y, strict=True):
        tolerance.tight(interp(float(x)), float(y_cpp))


def test_monotonic_cubic_preserves_monotonicity() -> None:
    # The filter's contract: monotone-increasing y → strictly increasing
    # values on a fine grid (no overshoot / no oscillation).
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    interp = MonotonicCubicNaturalSpline(xs, ys)
    fine = np.linspace(0.0, 4.0, 401)
    vals = [interp(float(xi)) for xi in fine]
    diffs = np.diff(vals)
    assert (diffs >= -1e-15).all(), f"non-monotonic: min diff {diffs.min()}"


# ---------------------------------------------------------------------------
# CubicInterpolation parameter validation.
# ---------------------------------------------------------------------------


def test_cubic_interpolation_default_is_natural() -> None:
    # Default kwargs produce a natural cubic spline.
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    a = CubicInterpolation(xs, ys)
    b = CubicNaturalSpline(xs, ys)
    for x in (0.25, 1.5, 2.7):
        tolerance.tight(a(x), b(x))


def test_cubic_interpolation_monotonic_matches_convenience_class() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    a = CubicInterpolation(xs, ys, monotonic=True)
    b = MonotonicCubicNaturalSpline(xs, ys)
    for x in (0.5, 1.25, 2.7, 3.4):
        tolerance.tight(a(x), b(x))


@pytest.mark.parametrize(
    "da",
    [
        DerivativeApprox.SplineOM1,
        DerivativeApprox.SplineOM2,
        DerivativeApprox.FourthOrder,
        DerivativeApprox.Parabolic,
        DerivativeApprox.FritschButland,
        DerivativeApprox.Akima,
        DerivativeApprox.Kruger,
        DerivativeApprox.Harmonic,
    ],
)
def test_cubic_interpolation_unimplemented_derivative_raises(da: DerivativeApprox) -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="not implemented in this port"):
        CubicInterpolation(xs, ys, derivative_approx=da)


@pytest.mark.parametrize(
    "bc",
    [
        BoundaryCondition.FirstDerivative,
        BoundaryCondition.Periodic,
        BoundaryCondition.Lagrange,
    ],
)
def test_cubic_interpolation_unimplemented_boundary_raises(bc: BoundaryCondition) -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="not implemented in this port"):
        CubicInterpolation(
            xs, ys, left_condition=bc, right_condition=bc, left_value=0.0, right_value=0.0
        )


@pytest.mark.parametrize(
    "bc",
    [
        BoundaryCondition.NotAKnot,
        BoundaryCondition.FirstDerivative,
        BoundaryCondition.Periodic,
        BoundaryCondition.Lagrange,
    ],
)
def test_cubic_interpolation_mixed_boundary_conditions_raise(bc: BoundaryCondition) -> None:
    """Two different conditions across the two ends stay carved out."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="mixed boundary conditions"):
        CubicInterpolation(xs, ys, left_condition=bc)
    with pytest.raises(LibraryException, match="mixed boundary conditions"):
        CubicInterpolation(xs, ys, right_condition=bc)


def test_not_a_knot_is_supported_and_ignores_the_end_condition_value() -> None:
    """C++ ignores the end-condition value for NotAKnot; so does this port.

    The spline must still pass through every knot, and — unlike a natural
    spline — its endpoint second derivative is nonzero. That difference is
    what makes it the discriminating underlying for ``FlatExtrapolator``.
    """
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([5.0, 3.0, 4.0, 2.0, 1.0], dtype=np.float64)
    interp = CubicInterpolation(
        xs,
        ys,
        left_condition=BoundaryCondition.NotAKnot,
        left_value=123.0,  # ignored
        right_condition=BoundaryCondition.NotAKnot,
        right_value=-7.0,  # ignored
    )
    for i in range(xs.size):
        tolerance.tight(interp(float(xs[i])), float(ys[i]))
    assert abs(interp.second_derivative(0.0)) > 1.0
    assert abs(interp.second_derivative(4.0)) > 1.0


def test_not_a_knot_with_monotonic_raises() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="Hyman filter runs on the natural-BC spline"):
        CubicInterpolation(
            xs,
            ys,
            monotonic=True,
            left_condition=BoundaryCondition.NotAKnot,
            right_condition=BoundaryCondition.NotAKnot,
        )


def test_cubic_interpolation_nonzero_second_derivative_raises() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="not implemented in this port"):
        CubicInterpolation(xs, ys, left_value=1.0)
    with pytest.raises(LibraryException, match="not implemented in this port"):
        CubicInterpolation(xs, ys, right_value=1.0)


def test_cubic_interpolation_extrapolation_guard() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    interp = CubicNaturalSpline(xs, ys)
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(5.0)
    # allow_extrapolation kwarg bypasses the check.
    _ = interp(5.0, allow_extrapolation=True)
