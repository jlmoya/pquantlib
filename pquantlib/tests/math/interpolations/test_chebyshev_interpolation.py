"""Cross-validate ChebyshevInterpolation against the L10-C C++ probe.

Reference: ``migration-harness/references/cluster/l10c.json`` —
``chebyshev_interpolation`` section. n=10 nodes, SecondKind, function
sin remapped to ``[0, pi]``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.chebyshev_interpolation import (
    ChebyshevInterpolation,
    PointsType,
    chebyshev_nodes_canonical,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l10c")


def test_canonical_nodes_first_kind() -> None:
    """FirstKind nodes match C++ ``-cos((i+0.5)*pi/n)``."""
    nodes = chebyshev_nodes_canonical(10, PointsType.FirstKind)
    for i in range(10):
        tolerance.tight(float(nodes[i]), -math.cos((i + 0.5) * math.pi / 10))


def test_canonical_nodes_second_kind_match_cpp(cpp: dict[str, Any]) -> None:
    """SecondKind nodes match the C++ probe's canonical nodes."""
    block = cpp["chebyshev_interpolation"]
    expected = [float(e) for e in block["nodes_canonical"]]
    nodes = chebyshev_nodes_canonical(10, PointsType.SecondKind)
    for i in range(10):
        tolerance.tight(float(nodes[i]), expected[i])


def test_remapped_nodes_match_cpp(cpp: dict[str, Any]) -> None:
    """Remapped nodes equal C++ probe's remapped nodes."""
    block = cpp["chebyshev_interpolation"]
    expected = [float(e) for e in block["nodes_remapped"]]
    interp = ChebyshevInterpolation(
        n=10, f=math.sin, points_type=PointsType.SecondKind,
        x_min=0.0, x_max=math.pi,
    )
    nodes = interp.nodes()
    for i in range(10):
        tolerance.tight(float(nodes[i]), expected[i])


def test_pillar_values_match_cpp(cpp: dict[str, Any]) -> None:
    """At each (remapped) Chebyshev node the interp returns ``sin(node)``.

    EXACT tier: the barycentric form has the removable-singularity
    short-circuit at node coincidence; both Python and C++ return the
    stored ``y[i]`` directly.
    """
    block = cpp["chebyshev_interpolation"]
    expected_pillars = [float(e) for e in block["pillar_values"]]
    interp = ChebyshevInterpolation(
        n=10, f=math.sin, points_type=PointsType.SecondKind,
        x_min=0.0, x_max=math.pi,
    )
    nodes = interp.nodes()
    for i, node in enumerate(nodes):
        # The pillar value equals sin(node) by construction.
        tolerance.tight(interp(float(node)), expected_pillars[i])


def test_intermediates_match_cpp(cpp: dict[str, Any]) -> None:
    """Intermediate values match C++ to TIGHT.

    TIGHT tier — both implementations use the same barycentric formula
    with the same closed-form weights. The C++ side evaluates at
    canonical-domain points (no remap on the C++ side); we evaluate at
    the equivalent remapped x = mid + t * half_span.
    """
    block = cpp["chebyshev_interpolation"]
    mids_remapped = [float(e) for e in block["mids_remapped"]]
    expected = [float(e) for e in block["interp_at_mids"]]
    interp = ChebyshevInterpolation(
        n=10, f=math.sin, points_type=PointsType.SecondKind,
        x_min=0.0, x_max=math.pi,
    )
    for x, y in zip(mids_remapped, expected, strict=True):
        tolerance.tight(interp(x), y)


def test_close_to_sin_approximation_error(cpp: dict[str, Any]) -> None:
    """Approximation error vs true ``sin`` is bounded.

    With n=10 SecondKind nodes on ``[0, pi]`` the Chebyshev polynomial
    approximation of ``sin`` has roughly 1e-7 error at off-pillar
    points. This is the *approximation* error (the gap between the
    interpolating polynomial and the true function), not the
    implementation error (the gap between Python and C++ interpolants
    of the same data — that one is TIGHT).
    """
    block = cpp["chebyshev_interpolation"]
    mids_remapped = [float(e) for e in block["mids_remapped"]]
    expected_sin = [float(e) for e in block["sin_at_remapped_mids"]]
    interp = ChebyshevInterpolation(
        n=10, f=math.sin, points_type=PointsType.SecondKind,
        x_min=0.0, x_max=math.pi,
    )
    for x, y in zip(mids_remapped, expected_sin, strict=True):
        tolerance.custom(
            interp(x), y,
            abs_tol=1.0e-6, rel_tol=1.0e-6,
            reason=(
                "n=10 Chebyshev polynomial approximation error to sin "
                "on [0, pi] is ~1e-7 at off-pillar samples"
            ),
        )


def test_values_argument_constructor() -> None:
    """Pass explicit y-values instead of a callable ``f``."""
    nodes = chebyshev_nodes_canonical(5, PointsType.SecondKind)
    ys = np.sin(nodes)  # sin(t) for t in [-1, 1]
    interp = ChebyshevInterpolation(values=ys, points_type=PointsType.SecondKind)
    for i, t in enumerate(nodes):
        tolerance.tight(interp(float(t)), float(ys[i]))


def test_update_y_replaces_values() -> None:
    interp = ChebyshevInterpolation(n=5, f=lambda x: x, x_min=-1.0, x_max=1.0)
    # Identity function: value at node t is t.
    nodes = interp.nodes()
    for t in nodes:
        tolerance.tight(interp(float(t)), float(t))
    # Replace y-values with all-1s.
    interp.update_y(np.ones(5))
    for t in nodes:
        tolerance.tight(interp(float(t)), 1.0)


def test_first_kind_nodes_exclude_endpoints() -> None:
    """FirstKind nodes lie strictly inside ``(-1, 1)``."""
    nodes = chebyshev_nodes_canonical(7, PointsType.FirstKind)
    assert float(nodes[0]) > -1.0
    assert float(nodes[-1]) < 1.0


def test_second_kind_nodes_include_endpoints() -> None:
    """SecondKind nodes include ``-1`` and ``+1``."""
    nodes = chebyshev_nodes_canonical(7, PointsType.SecondKind)
    tolerance.tight(float(nodes[0]), -1.0)
    tolerance.tight(float(nodes[-1]), 1.0)


def test_construct_from_n_only_zeros_y() -> None:
    """Without ``f`` or ``values`` the y-array seeds to zero.

    The barycentric formula can yield ``-0.0`` instead of ``+0.0``
    depending on the sign of the weights — both are representations
    of zero. Use TIGHT instead of EXACT.
    """
    interp = ChebyshevInterpolation(n=5)
    for t in (-0.8, -0.3, 0.0, 0.4, 0.9):
        tolerance.tight(interp(t), 0.0)


def test_update_y_length_mismatch_raises() -> None:
    interp = ChebyshevInterpolation(n=5, f=math.sin)
    with pytest.raises(LibraryException):
        interp.update_y(np.array([1.0, 2.0]))  # wrong length


def test_x_min_greater_than_x_max_raises() -> None:
    with pytest.raises(LibraryException):
        ChebyshevInterpolation(n=5, f=math.sin, x_min=1.0, x_max=0.0)


def test_n_below_2_raises() -> None:
    with pytest.raises(LibraryException):
        ChebyshevInterpolation(n=1)
