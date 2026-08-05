"""Cross-validate BackwardflatLinearInterpolation against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``backwardflat_linear`` section, on the same non-square, non-uniform 5x4 grid
as the bilinear and bicubic probes.

The probe sweeps x through every branch of the selection: below the first
node, one ULP-ish below each node, exactly on each node, one ULP-ish above
each node, between nodes, and past the last node. That matters because the
middle branch is an exact ``==`` comparison against the abscissa, so a port
that used a tolerance would return a different column for arguments 1e-9 off
a node — invisible on a coarse sweep, wrong in production.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.interpolations.backwardflat_linear_interpolation import (
    BackwardflatLinear,
    BackwardflatLinearInterpolation,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _make(block: dict[str, Any]) -> BackwardflatLinearInterpolation:
    interp = BackwardflatLinearInterpolation(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )
    interp.enable_extrapolation()
    return interp


def test_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Every (x, y) in the branch sweep matches C++ to TIGHT.

    The arithmetic is a single linear blend of two grid entries, so the two
    sides agree bit-for-bit in practice; TIGHT rather than EXACT only because
    ``u`` is a quotient and the compiler is free to contract the blend into
    an FMA.
    """
    block = cpp["backwardflat_linear"]
    interp = _make(block)
    for x, y, expected in block["evals"]:
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_flat_to_the_left_and_right(cpp: dict[str, Any]) -> None:
    """x below the first node uses column 0; x past the last uses column n-1.

    Both are *flat*, unlike the y direction which extrapolates linearly.
    """
    block = cpp["backwardflat_linear"]
    interp = _make(block)
    xs = [float(v) for v in block["xs"]]
    y = float(block["ys"][0])
    tolerance.exact(interp(xs[0] - 100.0, y), interp(xs[0], y))
    tolerance.exact(interp(xs[-1] + 100.0, y), interp(xs[-1] + 1.0, y))


def test_node_takes_its_own_column_not_the_next(cpp: dict[str, Any]) -> None:
    """Exactly on a node the value is that node's column, not the one right of it.

    Just above the node it is the next column. This is the discontinuity the
    ``x == xBegin_[i]`` branch creates and is the whole character of
    backward-flat.
    """
    block = cpp["backwardflat_linear"]
    interp = _make(block)
    xs = [float(v) for v in block["xs"]]
    ys = [float(v) for v in block["ys"]]
    z = np.asarray(block["z"], dtype=np.float64)
    y = ys[0]
    # Interior node: on it -> column i; a hair above -> column i+1.
    i = 1
    tolerance.exact(interp(xs[i], y), float(z[0, i]))
    tolerance.exact(interp(xs[i] + 1e-9, y), float(z[0, i + 1]))
    # A hair BELOW the node still resolves to column i (locate_x gives i-1,
    # the equality fails, so the i-1+1 == i column is used).
    tolerance.exact(interp(xs[i] - 1e-9, y), float(z[0, i]))


def test_linear_in_y(cpp: dict[str, Any]) -> None:
    """Along y the surface is the straight line between two grid rows."""
    block = cpp["backwardflat_linear"]
    interp = _make(block)
    x = float(block["xs"][2])
    ys = [float(v) for v in block["ys"]]
    lo, hi = ys[1], ys[2]
    mid = 0.5 * (lo + hi)
    tolerance.tight(interp(x, mid), 0.5 * (interp(x, lo) + interp(x, hi)))


def test_factory_matches_direct_construction(cpp: dict[str, Any]) -> None:
    """``BackwardflatLinear().interpolate(...)`` is the C++ traits path."""
    block = cpp["backwardflat_linear"]
    interp = BackwardflatLinear().interpolate(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )
    interp.enable_extrapolation()
    for x, y, expected in block["factory_evals"]:
        tolerance.tight(interp(float(x), float(y)), float(expected))
