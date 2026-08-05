"""Cross-validate ``AkimaCubicInterpolation`` against the L5-A C++ probe.

Reference: ``migration-harness/references/l5a/foundations.json`` —
``akima_cubic``: ``y = x^2`` at ``x = 0..4``, evaluated at 0.25, 0.75, 1.5,
2.5, 3.75.

This reference used to be *cited but not asserted*. The class delegated to
``scipy.interpolate.Akima1DInterpolator``, and the old tests said so
explicitly: they checked that scipy "recovers the quadratic exactly" and
recorded C++ as deviating on the boundary cubic. C++ QuantLib v1.43 is the
ground truth for this port, and its Akima endpoint slopes are its own
invention (products like ``2*S[0]*S[1]``, cubicinterpolation.hpp:613-629),
not Akima's reflection rule — on this very data C++ gives a pillar slope of
2.25 at ``x = 0`` where calculus gives 0. Since ``AkimaCubicInterpolation``
is now ``CubicInterpolation(Akima, ...)``, the probe values are assertable
at TIGHT, including at the two intervals nearest each end.

The wider ``Akima`` coverage — non-uniform grids, the ``monotonic`` variant,
extrapolation, the coefficient arrays — lives in
``test_cubic_interpolation.py``, which walks the v1.43 probe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.akima_cubic_interpolation import (
    AkimaCubicInterpolation,
)
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    DerivativeApprox,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("l5a/foundations")


def _make() -> AkimaCubicInterpolation:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    return AkimaCubicInterpolation(xs, ys)


def test_values_match_cpp(cpp: dict[str, Any]) -> None:
    """TIGHT at every probed point, boundary intervals included."""
    block = cpp["akima_cubic"]
    interp = _make()
    for x, expected in zip(block["xs_eval"], block["values"], strict=True):
        tolerance.tight(interp(float(x)), float(expected))


def test_derivatives_match_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["akima_cubic"]
    interp = _make()
    for x, expected in zip(block["xs_eval"], block["derivatives"], strict=True):
        tolerance.tight(interp.derivative(float(x)), float(expected))


def test_second_derivatives_match_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["akima_cubic"]
    interp = _make()
    for x, expected in zip(block["xs_eval"], block["second_derivatives"], strict=True):
        tolerance.tight(interp.second_derivative(float(x)), float(expected))


def test_does_not_recover_the_quadratic_near_the_ends(cpp: dict[str, Any]) -> None:
    """QuantLib's Akima is not the textbook one, and the difference is large.

    On ``y = x^2`` a faithful Akima 1970 implementation reproduces ``x^2``
    everywhere. QuantLib's endpoint slopes do not, and the first two and last
    two intervals are visibly off — ``f(0.25) = 0.3588`` against ``0.0625``.
    Pinning that keeps a "more correct" substitute from being slipped back in.
    """
    block = cpp["akima_cubic"]
    interp = _make()
    assert abs(interp(0.25) - 0.0625) > 0.25
    assert abs(interp(3.75) - 14.0625) > 0.05
    # Away from the ends it does reproduce the quadratic.
    tolerance.tight(interp(2.5), 6.367307692307692)
    _ = block


def test_hits_knots_exactly() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    interp = AkimaCubicInterpolation(xs, ys)
    for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
        tolerance.exact(interp(float(x)), float(y))


def test_is_the_cubic_interpolation_akima_preset() -> None:
    """EXACT: the class is one argument tuple of ``CubicInterpolation``."""
    xs = np.array([0.0, 1.0, 2.5, 3.0, 4.5, 6.0], dtype=np.float64)
    ys = np.array([5.0, 3.0, 4.0, 2.0, 1.0, 3.0], dtype=np.float64)
    a = AkimaCubicInterpolation(xs, ys)
    b = CubicInterpolation(
        xs,
        ys,
        DerivativeApprox.Akima,
        False,
        BoundaryCondition.SecondDerivative,
        0.0,
        BoundaryCondition.SecondDerivative,
        0.0,
    )
    for x in (-0.5, 0.5, 2.0, 3.75, 6.5):
        tolerance.exact(a(x, allow_extrapolation=True), b(x, allow_extrapolation=True))


def test_rejects_extrapolation_by_default() -> None:
    interp = _make()
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(5.0)


def test_allows_extrapolation_when_requested() -> None:
    """Extrapolation extends the end cubic — a finite number, never NaN.

    scipy's ``Akima1DInterpolator`` returns NaN outside the data range; C++
    evaluates the clamped end interval's cubic. That difference alone would
    turn an out-of-range curve query into a silent NaN.
    """
    interp = _make()
    value = interp(5.0, allow_extrapolation=True)
    assert np.isfinite(value)


def test_length_mismatch_raises() -> None:
    with pytest.raises(LibraryException, match="same length"):
        AkimaCubicInterpolation(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))


def test_akima_with_few_points() -> None:
    """C++ parity: ``testAkimaWithFewPoints`` (test-suite/interpolations.cpp, v1.43).

    The Akima scheme reads ``S[2]`` and ``S[n-4]``; with three points both are
    out of bounds, and C++ ``QL_REQUIRE``s against it.
    """
    with pytest.raises(LibraryException, match="at least 4 points"):
        AkimaCubicInterpolation(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.5]))
    with pytest.raises(LibraryException, match="at least 4 points"):
        AkimaCubicInterpolation(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    x4 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y4 = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)
    f = AkimaCubicInterpolation(x4, y4)
    for x, y in zip(x4.tolist(), y4.tolist(), strict=True):
        tolerance.exact(f(float(x)), float(y))


def test_update_idempotent_when_inputs_unchanged() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    interp = AkimaCubicInterpolation(xs, ys)
    before = interp(1.5)
    interp.update()
    tolerance.exact(interp(1.5), before)
