"""Cross-validate BicubicSpline against the L9-A C++ probe.

Reference: ``migration-harness/references/cluster/l9a.json`` —
``bicubic_spline`` section. The probe uses a 4x4 grid with
``z[j, i] = sin(xs[i]) + cos(ys[j])`` and evaluates four off-grid
points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l9a")


def _make(cpp: dict[str, Any]) -> BicubicSpline:
    block = cpp["bicubic_spline"]
    xs = np.asarray(block["xs"], dtype=np.float64)
    ys = np.asarray(block["ys"], dtype=np.float64)
    z = np.asarray(block["z"], dtype=np.float64)  # shape (len(ys), len(xs))
    return BicubicSpline(xs, ys, z)


def test_bicubic_pillars_roundtrip_tight(cpp: dict[str, Any]) -> None:
    # At pillar nodes the spline must reproduce the input z to TIGHT
    # (scipy's RectBivariateSpline interpolates exactly at the knots).
    block = cpp["bicubic_spline"]
    xs = block["xs"]
    ys = block["ys"]
    z = block["z"]
    interp = _make(cpp)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            tolerance.tight(interp(float(x), float(y)), float(z[j][i]))


def test_bicubic_pillars_match_cpp_tight(cpp: dict[str, Any]) -> None:
    # C++ pillar values also roundtrip to TIGHT; cross-validate both
    # implementations recover the input.
    block = cpp["bicubic_spline"]
    xs = block["xs"]
    ys = block["ys"]
    pillars = block["pillars"]
    interp = _make(cpp)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            tolerance.tight(interp(float(x), float(y)), float(pillars[j][i]))


def test_bicubic_intermediates_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Off-pillar values now match C++ to TIGHT.

    This assertion used to be a 0.15 relative-error "qualitative agreement"
    envelope, because the port delegated to
    ``scipy.interpolate.RectBivariateSpline`` — a tensor-product B-spline with
    not-a-knot end conditions — where C++ composes *natural* 1-D cubic splines
    row-then-column. Same pillars, different function in between: the observed
    gap on this 4x4 grid was ~10 %. The composition is now transcribed from
    ``bicubicsplineinterpolation.hpp``, so the surfaces agree everywhere and
    the envelope is replaced by the tier.
    """
    block = cpp["bicubic_spline"]
    mids_xy = block["mids_xy"]
    mids_z = block["mids_z"]
    interp = _make(cpp)
    for (x, y), expected in zip(mids_xy, mids_z, strict=True):
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_bicubic_update_refreshes() -> None:
    xs = np.linspace(0.0, 3.0, 4)
    ys = np.linspace(0.0, 3.0, 4)
    z = np.array([[float(i + j) for i in range(4)] for j in range(4)], dtype=np.float64)
    interp = BicubicSpline(xs, ys, z)
    v_before = interp(1.5, 1.5)
    z2 = 2.0 * z
    interp._z[:] = z2  # pyright: ignore[reportPrivateUsage]
    interp.update()
    v_after = interp(1.5, 1.5)
    tolerance.tight(v_after, 2.0 * v_before)


def test_bicubic_extrapolation_guard() -> None:
    xs = np.linspace(0.0, 3.0, 4)
    ys = np.linspace(0.0, 3.0, 4)
    z = np.array([[float(i + j) for i in range(4)] for j in range(4)], dtype=np.float64)
    interp = BicubicSpline(xs, ys, z)
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(5.0, 1.5)
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(1.5, 5.0)
    _ = interp(5.0, 1.5, allow_extrapolation=True)


def test_bicubic_inspectors() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    z = np.zeros((4, 4), dtype=np.float64)
    interp = BicubicSpline(xs, ys, z)
    tolerance.exact(interp.x_min, 0.0)
    tolerance.exact(interp.x_max, 3.0)
    tolerance.exact(interp.y_min, 0.0)
    tolerance.exact(interp.y_max, 3.0)
    assert interp.is_in_range(1.5, 1.5)
    assert not interp.is_in_range(5.0, 1.5)
    assert not interp.is_in_range(1.5, 5.0)
