"""Cross-validate FlatExtrapolator2D against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``flat_extrapolator_2d`` section. A :class:`BicubicSpline` over the 5x4 grid,
wrapped and then evaluated across all eight out-of-range directions plus the
corners. The reference also pins, per point, the clamped ``(x, y)`` and the
decorated surface's value there, so the tests can assert the *clamp* rather
than just reproduce a number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.math.interpolations.flat_extrapolation_2d import FlatExtrapolator2D
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _decorated(block: dict[str, Any]) -> BicubicSpline:
    return BicubicSpline(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )


def _make(block: dict[str, Any]) -> FlatExtrapolator2D:
    interp = FlatExtrapolator2D(_decorated(block))
    interp.enable_extrapolation()
    return interp


def test_decorated_surface_is_the_probe_s(cpp: dict[str, Any]) -> None:
    """Guard: the probe wrapped a BicubicSpline, and so does this test."""
    assert cpp["flat_extrapolator_2d"]["decorated"] == "BicubicSpline"


def test_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Values match C++ to TIGHT, inside and outside the grid on all sides."""
    block = cpp["flat_extrapolator_2d"]
    interp = _make(block)
    for x, y, expected in block["evals"]:
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_value_equals_decorated_at_the_clamped_point(cpp: dict[str, Any]) -> None:
    """The decorator IS the clamp — nothing else.

    C++ pins ``decoratedInterp_(bindX(x), bindY(y))`` alongside the
    decorator's own answer; asserting they agree proves the clamp, whereas
    matching the number alone would also pass for a decorator that happened
    to extrapolate to the same value.
    """
    block = cpp["flat_extrapolator_2d"]
    assert block["clamped_columns"] == "x,y,bound_x,bound_y,decorated_at_bound"
    interp = _make(block)
    decorated = _decorated(block)
    for x, y, bound_x, bound_y, at_bound in block["clamped"]:
        tolerance.tight(interp(float(x), float(y)), float(at_bound))
        tolerance.tight(
            decorated(float(bound_x), float(bound_y), allow_extrapolation=True),
            float(at_bound),
        )


def test_bounds_delegate_to_the_decorated_surface(cpp: dict[str, Any]) -> None:
    """xMin/xMax/yMin/yMax forward rather than being recomputed."""
    block = cpp["flat_extrapolator_2d"]
    interp = _make(block)
    tolerance.exact(interp.x_min, float(block["x_min"]))
    tolerance.exact(interp.x_max, float(block["x_max"]))
    tolerance.exact(interp.y_min, float(block["y_min"]))
    tolerance.exact(interp.y_max, float(block["y_max"]))
    assert interp.decorated_interpolation is not None


def test_decorator_carries_its_own_extrapolation_flag(cpp: dict[str, Any]) -> None:
    """Enabling extrapolation on the decorated surface does not propagate.

    ``is_in_range`` delegates, but the range *check* runs against the
    decorator's own flag, so an unwrapped-enabled inner surface still raises
    through the wrapper.
    """
    block = cpp["flat_extrapolator_2d"]
    decorated = _decorated(block)
    decorated.enable_extrapolation()
    interp = FlatExtrapolator2D(decorated)
    out_x = float(block["x_max"]) + 5.0
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(out_x, 0.0)
    # Explicit per-call opt-in works, as does enabling the decorator's flag.
    _ = interp(out_x, 0.0, allow_extrapolation=True)
    interp.enable_extrapolation()
    _ = interp(out_x, 0.0)


def test_flat_beyond_every_edge(cpp: dict[str, Any]) -> None:
    """Past an edge the value stops changing — that is what "flat" means."""
    block = cpp["flat_extrapolator_2d"]
    interp = _make(block)
    x_min = float(block["x_min"])
    x_max = float(block["x_max"])
    y_min = float(block["y_min"])
    y_max = float(block["y_max"])
    y_mid = 0.5 * (y_min + y_max)
    x_mid = 0.5 * (x_min + x_max)
    tolerance.exact(interp(x_min - 1.0, y_mid), interp(x_min - 100.0, y_mid))
    tolerance.exact(interp(x_max + 1.0, y_mid), interp(x_max + 100.0, y_mid))
    tolerance.exact(interp(x_mid, y_min - 1.0), interp(x_mid, y_min - 100.0))
    tolerance.exact(interp(x_mid, y_max + 1.0), interp(x_mid, y_max + 100.0))
    # Corner: clamps in both coordinates at once.
    tolerance.exact(interp(x_min - 7.0, y_max + 9.0), interp(x_min, y_max))
