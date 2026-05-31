"""Cross-validate Polynomial2DSpline against the W7-D C++ probe.

Probe source: migration-harness/cpp/probes/cluster_w7d/probe.cpp
Reference:    migration-harness/references/cluster/w7d.json

The probe grid is ``x = [1,2,3,4,5]`` (maturities, spline direction),
``y = [0.01,0.02,0.03,0.04]`` (strikes, parabolic direction), with
``z[i][k] = 100*y[i]^2 + 5*x[k] + 2*y[i]*x[k]`` (rows = y, cols = x).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.polynomial_2d_spline import Polynomial2DSpline
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/w7d")


def _make() -> Polynomial2DSpline:
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    ys = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    z = np.empty((ys.shape[0], xs.shape[0]), dtype=np.float64)
    for i, y in enumerate(ys):
        for k, x in enumerate(xs):
            z[i, k] = 100.0 * y * y + 5.0 * x + 2.0 * y * x
    return Polynomial2DSpline(xs, ys, z)


def test_pillars_tight(cpp: dict[str, Any]) -> None:
    # At the grid pillars, the parabolic-then-spline construction must
    # reproduce the input surface to TIGHT (both 1-D interpolations are
    # exact at their nodes).
    s = _make()
    tolerance.tight(s(1.0, 0.01), cpp["poly2d_pillar_x1_y0.01"])
    tolerance.tight(s(3.0, 0.02), cpp["poly2d_pillar_x3_y0.02"])
    tolerance.tight(s(5.0, 0.04), cpp["poly2d_pillar_x5_y0.04"])


def test_interior_loose(cpp: dict[str, Any]) -> None:
    # Interior (off-grid) points: LOOSE — the exact value depends on the
    # parabolic Hermite slopes and the natural-cubic-spline coefficients,
    # which we reproduce from the C++ algorithm but accumulate fp
    # differently (numpy vectorised vs C++ scalar; scipy spline solve).
    s = _make()
    tolerance.loose(s(2.5, 0.025), cpp["poly2d_interior_x2.5_y0.025"])
    tolerance.loose(s(4.2, 0.015), cpp["poly2d_interior_x4.2_y0.015"])
    tolerance.loose(s(1.5, 0.035), cpp["poly2d_interior_x1.5_y0.035"])


def test_pillars_reproduce_input() -> None:
    # Independent of the probe: pillar reproduction across the whole grid.
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [0.01, 0.02, 0.03, 0.04]
    s = _make()
    for y in ys:
        for x in xs:
            expected = 100.0 * y * y + 5.0 * x + 2.0 * y * x
            tolerance.loose(s(x, y), expected)


def test_extrapolation_allowed_by_default_in_value() -> None:
    # The C++ surface always evaluates with extrapolation enabled in both
    # directions; our base ``Interpolation2D`` still guards the public
    # ``__call__`` range check unless extrapolation is enabled.
    s = _make()
    # In-range query works without enabling extrapolation.
    _ = s(2.0, 0.02)
    # Out-of-range raises unless extrapolation is enabled.
    with pytest.raises(LibraryException):
        _ = s(6.0, 0.02)
    s.enable_extrapolation()
    _ = s(6.0, 0.02)  # no raise once enabled
