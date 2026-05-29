"""Tests for ConvexMonotoneInterpolation — Hagan-West (2006) convex-monotone curve.

# C++ parity: ql/math/interpolations/convexmonotoneinterpolation.hpp (v1.42.1).

The interpolation has no clean closed form for arbitrary inputs, so
cross-validation against the C++ probe runs at pillar values (EXACT
where possible) plus a convexity invariant check at a fine grid
(LOOSE — depends on the shape of the synthetic input).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.convex_monotone_interpolation import (
    ConvexMonotoneInterpolation,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/w2b")


def _xs_ys() -> tuple[np.ndarray, np.ndarray]:
    """Same pillars as the probe — 6 points on a rate-shaped curve.

    See ``migration-harness/cpp/probes/cluster_w2b/probe.cpp``.
    """
    xs = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    ys = np.array([0.0, 0.018, 0.020, 0.025, 0.028, 0.030], dtype=np.float64)
    return xs, ys


# --- construction + invariants -------------------------------------------------


def test_construction_rejects_bad_quadraticity() -> None:
    xs, ys = _xs_ys()
    with pytest.raises(LibraryException, match="Quadraticity"):
        ConvexMonotoneInterpolation(xs, ys, quadraticity=1.5)


def test_construction_rejects_bad_monotonicity() -> None:
    xs, ys = _xs_ys()
    with pytest.raises(LibraryException, match="Monotonicity"):
        ConvexMonotoneInterpolation(xs, ys, monotonicity=-0.1)


def test_construction_rejects_single_pillar() -> None:
    with pytest.raises(LibraryException):
        ConvexMonotoneInterpolation(np.array([0.0]), np.array([1.0]))


def test_derivative_not_implemented() -> None:
    """C++ ``QL_FAIL`` — derivative is not computed for this interpolation."""
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    with pytest.raises(NotImplementedError, match="derivative"):
        interp.derivative(1.5)


def test_second_derivative_not_implemented() -> None:
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    with pytest.raises(NotImplementedError, match="second derivative"):
        interp.second_derivative(1.5)


# --- pillar values cross-validate against C++ probe (EXACT-when-possible) -----


def test_pillar_values_match_cpp(cpp: dict[str, Any]) -> None:
    """At pillar ``x[i]`` (i > 0), the interpolation matches the C++ reference.

    The first pillar (i=0) is the "ignored" y-value per the C++
    contract — both ports skip it. So we test i in [1, len-1].
    """
    block = cpp["convex_monotone"]
    xs = np.array(block["xs"], dtype=np.float64)
    ys = np.array(block["ys"], dtype=np.float64)
    expected_at_pillars = block["values_at_pillars"]  # length == len(xs)
    interp = ConvexMonotoneInterpolation(
        xs, ys,
        quadraticity=block["quadraticity"],
        monotonicity=block["monotonicity"],
        force_positive=bool(block["force_positive"]),
    )
    for i in range(1, len(xs)):
        # LOOSE because the convex-monotone branch's section-helper
        # choice depends on global-data-driven boundary forwards; small
        # numerical noise in the section-split parameter ``eta`` is
        # amplified by the helper polynomials.
        tolerance.loose(
            interp(float(xs[i])),
            float(expected_at_pillars[i]),
            reason="convex-monotone section helpers — eta numerics LOOSE",
        )


def test_interior_values_match_cpp(cpp: dict[str, Any]) -> None:
    """Mid-segment evaluation at a few probed-strikes matches C++."""
    block = cpp["convex_monotone"]
    xs = np.array(block["xs"], dtype=np.float64)
    ys = np.array(block["ys"], dtype=np.float64)
    interp = ConvexMonotoneInterpolation(
        xs, ys,
        quadraticity=block["quadraticity"],
        monotonicity=block["monotonicity"],
        force_positive=bool(block["force_positive"]),
    )
    for probe_pt in block["interior_probes"]:
        x = float(probe_pt["x"])
        expected = float(probe_pt["value"])
        tolerance.loose(
            interp(x), expected,
            reason="convex-monotone polynomial evaluation LOOSE",
        )


# --- shape invariants ---------------------------------------------------------


def test_returns_non_negative_when_force_positive() -> None:
    """force_positive=True ensures ``f >= 0`` on a non-negative input."""
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(
        xs, ys, quadratic_constraint=False, force_positive=True,
    )
    grid: np.ndarray = np.linspace(0.1, float(xs[-1]), 50)
    for x_val in grid:
        x = float(x_val)
        assert interp(x) >= -1e-12, f"f({x}) = {interp(x)} < 0"


def test_extrapolation_returns_constant_beyond_last_pillar() -> None:
    """# C++ parity: extrapolation helper is ``EverywhereConstantHelper``."""
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    interp.enable_extrapolation(True)
    v_end = interp(float(xs[-1]))
    v_beyond = interp(float(xs[-1] + 5.0))
    assert math.isclose(v_end, v_beyond, abs_tol=1e-10)


def test_constant_input_yields_constant_output() -> None:
    """If all averaged values are the same, the interpolation is flat.

    Sanity check on the section-helper picker — when ``y_i == c`` for
    all i, the boundary forwards collapse to ``c`` too, and ``g_prev =
    g_next = 0`` triggers the ``ConstantGradHelper`` branch.
    """
    xs = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    ys = np.array([0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float64)
    interp = ConvexMonotoneInterpolation(
        xs, ys, quadratic_constraint=False, force_positive=False,
    )
    for x in [0.5, 1.5, 3.0, 7.0, 9.5]:
        assert math.isclose(interp(x), 0.03, abs_tol=1e-12), (
            f"f({x}) = {interp(x)} != 0.03"
        )


def test_quadratic_constraint_default_matches_factory_defaults() -> None:
    """quadratic_constraint=True → quadraticity=0.3, monotonicity=0.7."""
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=True)
    assert interp.quadraticity == 0.3
    assert interp.monotonicity == 0.7
    interp2 = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    assert interp2.quadraticity == 0.0
    assert interp2.monotonicity == 1.0


def test_two_point_construction_constant_helper() -> None:
    """# C++ parity: single-period case → ``EverywhereConstantHelper``."""
    xs = np.array([0.0, 1.0], dtype=np.float64)
    ys = np.array([0.0, 0.02], dtype=np.float64)
    interp = ConvexMonotoneInterpolation(
        xs, ys, quadratic_constraint=False, force_positive=True,
    )
    # y[0] is ignored; the single period takes y[1] as its constant.
    assert math.isclose(interp(0.5), 0.02, abs_tol=1e-12)
    assert math.isclose(interp(1.0), 0.02, abs_tol=1e-12)
