"""Cross-validate HymanFilteredCubic against the L10-C C++ probe.

Reference: ``migration-harness/references/cluster/l10c.json`` —
``hyman_filtered_cubic`` section. Knots y = (0, 0.5, 1.5, 3, 3.2) at
x = 0..4 — strictly monotone-increasing input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.interpolations.cubic_interpolation import (
    MonotonicCubicNaturalSpline,
)
from pquantlib.math.interpolations.hyman_filter import (
    HymanFilteredCubic,
    is_monotone,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l10c")


def _make() -> HymanFilteredCubic:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    return HymanFilteredCubic(xs, ys)


def test_pillars_exact(cpp: dict[str, Any]) -> None:
    """Interpolation must pass through every input pillar exactly.

    EXACT tier: the Hyman filter does not perturb knot heights — the
    natural-spline + filter algorithm reproduces y[i] bit-for-bit at
    each x[i].
    """
    block = cpp["hyman_filtered_cubic"]
    xs = [float(e) for e in block["xs"]]
    ys = [float(e) for e in block["ys"]]
    interp = _make()
    for x, y in zip(xs, ys, strict=True):
        tolerance.exact(interp(x), y)


def test_intermediates_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Intermediate values match C++ to TIGHT — same algorithm both sides.

    Both Python and C++ port the *same* Hyman-1983 algorithm: solve the
    natural-spline tridiagonal, apply the monotonicity filter, emit the
    cubic-Hermite coefficients. The only differences come from
    floating-point order-of-operations between numpy's BLAS-routed
    triangular solve and the C++ TridiagonalOperator. TIGHT is
    appropriate.
    """
    block = cpp["hyman_filtered_cubic"]
    mids_x = [float(e) for e in block["mids_x"]]
    mids_y = [float(e) for e in block["mids_y"]]
    interp = _make()
    for x, y in zip(mids_x, mids_y, strict=True):
        tolerance.tight(interp(x), y)


def test_derivative_at_1_5(cpp: dict[str, Any]) -> None:
    """First derivative at x = 1.5 matches C++ to TIGHT."""
    block = cpp["hyman_filtered_cubic"]
    interp = _make()
    tolerance.tight(interp.derivative(1.5), float(block["derivative_at_1_5"]))


def test_second_derivative_at_1_5(cpp: dict[str, Any]) -> None:
    """Second derivative at x = 1.5 matches C++ to TIGHT."""
    block = cpp["hyman_filtered_cubic"]
    interp = _make()
    tolerance.tight(
        interp.second_derivative(1.5), float(block["second_derivative_at_1_5"])
    )


def test_no_overshoot_on_monotone_input() -> None:
    """The Hyman filter must produce a monotonic-on-its-input cubic.

    On any monotone-increasing input the interpolant on intermediate
    points must remain in the [y[i], y[i+1]] interval — that's the
    whole point of the Hyman filter. We sample 40 points across the
    domain and assert monotonicity.
    """
    interp = _make()
    samples = np.linspace(0.0, 4.0, 41)
    values = np.array([interp(float(x)) for x in samples])
    assert is_monotone(values, tol=1e-12), (
        f"Hyman-filtered cubic is not monotonic: {values}"
    )


def test_primitive_at_pillar_zero() -> None:
    """Primitive at x = x[0] must be zero (F(x0) = 0)."""
    interp = _make()
    tolerance.exact(interp.primitive(0.0), 0.0)


def test_primitive_monotone_for_positive_y() -> None:
    """Primitive is monotonically increasing for a positive-y curve.

    The input is y = (0, 0.5, 1.5, 3.0, 3.2), all >= 0, so the
    integral F(x) is monotone non-decreasing.
    """
    interp = _make()
    prev = -1.0
    for x in np.linspace(0.0, 4.0, 21):
        val = interp.primitive(float(x))
        assert val >= prev - 1e-14
        prev = val


def test_versus_pchip_differs_on_intermediates() -> None:
    """Hyman-natural-spline differs from scipy PCHIP at intermediate points.

    Both algorithms produce monotone-preserving cubics through the
    same knots, but PCHIP (Fritsch-Carlson) derives slopes directly
    from one-sided three-point formulas, while
    HymanFilteredCubic solves the natural-spline tridiagonal first
    then filters. The two diverge at intermediate points, documented
    in the cubic_interpolation module docstring.

    This test certifies the divergence is *observable* (not a near-
    equality) on the L10-C reference grid.
    """
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    hyman = HymanFilteredCubic(xs, ys)
    pchip = MonotonicCubicNaturalSpline(xs, ys)
    diffs = [abs(hyman(x) - pchip(x)) for x in (0.5, 1.25, 2.7, 3.4)]
    # At least one intermediate point should differ by >= 1e-3 to
    # confirm we're not accidentally reproducing PCHIP.
    assert max(diffs) >= 1.0e-3, f"Hyman and PCHIP unexpectedly agree: {diffs}"


def test_update_with_different_data() -> None:
    """Constructing two HymanFilteredCubic instances with different ys
    produces independent interpolators.

    The C++ ``update()`` method is exercised internally by the
    constructor; we test that two instances with different y-data return
    independent values.
    """
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    # Linear input — at x = 1.5 the value should be 1.5.
    interp_a = HymanFilteredCubic(xs, np.array([0.0, 1.0, 2.0, 3.0]))
    tolerance.tight(interp_a(1.5), 1.5)
    # Slope-2 input — at x = 1.5 the value should be 3.0.
    interp_b = HymanFilteredCubic(xs, np.array([0.0, 2.0, 4.0, 6.0]))
    tolerance.tight(interp_b(1.5), 3.0)
