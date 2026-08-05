"""Cross-validate BilinearInterpolation + the Bilinear factory against v1.43 C++.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``bilinear`` section, on the same non-square, non-uniform 5x4 grid as the
bicubic probe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.bilinear import Bilinear, BilinearInterpolation
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _make(block: dict[str, Any]) -> BilinearInterpolation:
    interp = BilinearInterpolation(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )
    interp.enable_extrapolation()
    return interp


def test_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Values match C++ to TIGHT at nodes, midpoints and outside the grid.

    Outside the grid ``locate_x`` / ``locate_y`` clamp to the edge cell and
    ``t`` / ``u`` simply leave ``[0, 1]``, so the surface extends linearly
    rather than flat — the probe points beyond both edges pin that.
    """
    block = cpp["bilinear"]
    interp = _make(block)
    for x, y, expected in block["evals"]:
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_pillars_are_exact(cpp: dict[str, Any]) -> None:
    """At a node ``t == u == 0``, so the formula collapses to ``1*1*z1``.

    Bit-exact, not merely tight: no rounding can enter a multiplication by
    an exact 1.0 and additions of exact zeros.
    """
    block = cpp["bilinear"]
    interp = _make(block)
    z = np.asarray(block["z"], dtype=np.float64)
    for j, y in enumerate(block["ys"]):
        for i, x in enumerate(block["xs"]):
            tolerance.exact(interp(float(x), float(y)), float(z[j, i]))


def test_factory_matches_direct_construction(cpp: dict[str, Any]) -> None:
    """``Bilinear().interpolate(...)`` is the C++ traits path; same numbers."""
    block = cpp["bilinear"]
    interp = Bilinear().interpolate(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )
    interp.enable_extrapolation()
    for x, y, expected in block["factory_evals"]:
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_is_an_interpolation_2d(cpp: dict[str, Any]) -> None:
    """C++ ``BilinearInterpolation : public Interpolation2D``; so is the port.

    It used to be a standalone class duplicating the base's accessors, which
    meant it could not be handed to ``FlatExtrapolator2D``.
    """
    assert isinstance(_make(cpp["bilinear"]), Interpolation2D)


def test_extrapolation_guard(cpp: dict[str, Any]) -> None:
    """Without the flag, out-of-range calls raise (C++ ``checkRange``)."""
    block = cpp["bilinear"]
    interp = BilinearInterpolation(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(10.0, 0.0)
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(1.0, 10.0)
    _ = interp(10.0, 0.0, allow_extrapolation=True)


def test_locate_matches_cpp_clamping(cpp: dict[str, Any]) -> None:
    """``locateX`` / ``locateY`` clamp to ``[0, n-2]`` at both ends."""
    block = cpp["bilinear"]
    interp = _make(block)
    xs = [float(v) for v in block["xs"]]
    ys = [float(v) for v in block["ys"]]
    assert interp.locate_x(xs[0] - 1.0) == 0
    assert interp.locate_x(xs[-1] + 1.0) == len(xs) - 2
    assert interp.locate_x(xs[-1]) == len(xs) - 2
    assert interp.locate_y(ys[0] - 1.0) == 0
    assert interp.locate_y(ys[-1] + 1.0) == len(ys) - 2
    for i in range(len(xs) - 1):
        assert interp.locate_x(xs[i]) == i
