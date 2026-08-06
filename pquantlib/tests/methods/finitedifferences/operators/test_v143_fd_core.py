"""Cross-validate the FD framework core primitives against C++ v1.43.

Reference: ``migration-harness/references/v143/methods/core.json``, emitted by
``v143_methods_core_probe``.

Two pre-existing divergences are pinned here:

1. ``FdmLinearOpLayout.neighbourhood`` **clamped** at the grid boundary
   (``coordinate 0`` with offset ``-1`` mapped back to ``0``). C++
   ``FdmLinearOpLayout::neighbourhood`` *reflects*
   (``coorOffset = -coorOffset`` below zero,
   ``coorOffset = 2*(dim-1) - coorOffset`` above the top). The two agree only
   when the corresponding band coefficient happens to be zero — which is true
   for ``FirstDerivativeOp``/``SecondDerivativeOp`` but not for operators that
   patch boundary rows, and never for ``to_matrix()``.

2. ``TripleBandLinearOp.reverse_index`` was hard-coded to the identity with a
   comment claiming multi-D was "deferred". The identity is only correct for
   ``direction == 0``; along any other direction the Thomas sweep in
   ``solve_splitting`` then walks the wrong lines. C++ builds the permutation
   by swapping axis 0 with ``direction`` in ``dim``, taking that layout's
   ``spacing``, swapping the same two entries back, and re-indexing.

``multR`` and ``ModTripleBandLinearOp`` were simply missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/methods/core")


def _assert_array(actual: Array, expected: list[float]) -> None:
    assert actual.size == len(expected)
    for a, e in zip(actual.tolist(), expected, strict=True):
        tight(float(a), float(e))


# ---------------------------------------------------------------------------
# FdmLinearOpLayout — the boundary rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-2, -1, 1, 2])
def test_layout_1d_neighbourhood_matches_cpp(cpp: dict[str, Any], offset: int) -> None:
    layout = FdmLinearOpLayout((5,))
    actual = [layout.neighbourhood(it, 0, offset) for it in layout.iter()]
    assert actual == cpp[f"layout_1d_5_neighbourhood_off{offset}"]


def test_layout_2d_shape_matches_cpp(cpp: dict[str, Any]) -> None:
    layout = FdmLinearOpLayout((4, 3))
    assert list(layout.dim()) == cpp["layout_2d_4x3_dim"]
    assert list(layout.spacing()) == cpp["layout_2d_4x3_spacing"]


@pytest.mark.parametrize("direction", [0, 1])
@pytest.mark.parametrize("offset", [-1, 1])
def test_layout_2d_neighbourhood_matches_cpp(
    cpp: dict[str, Any], direction: int, offset: int
) -> None:
    layout = FdmLinearOpLayout((4, 3))
    actual = [layout.neighbourhood(it, direction, offset) for it in layout.iter()]
    assert actual == cpp[f"layout_2d_4x3_nb_dir{direction}_off{offset}"]


@pytest.mark.parametrize("o0", [-1, 1])
@pytest.mark.parametrize("o1", [-1, 1])
def test_layout_2d_neighbourhood2_matches_cpp(cpp: dict[str, Any], o0: int, o1: int) -> None:
    layout = FdmLinearOpLayout((4, 3))
    actual = [layout.neighbourhood2(it, 0, o0, 1, o1) for it in layout.iter()]
    assert actual == cpp[f"layout_2d_4x3_nb2_o0{o0}_o1{o1}"]


def test_layout_2d_iter_neighbourhood_matches_cpp(cpp: dict[str, Any]) -> None:
    layout = FdmLinearOpLayout((4, 3))
    nb = [layout.iter_neighbourhood(it, 0, -1) for it in layout.iter()]
    assert [n.index for n in nb] == cpp["layout_2d_4x3_iternb_dir0_offm1_index"]
    assert [n.coordinates[0] for n in nb] == cpp["layout_2d_4x3_iternb_dir0_offm1_c0"]
    assert [n.coordinates[1] for n in nb] == cpp["layout_2d_4x3_iternb_dir0_offm1_c1"]


# ---------------------------------------------------------------------------
# TripleBandLinearOp / ModTripleBandLinearOp on a 2-D mesh
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(-1.0, 2.0, 4),
        Uniform1dMesher(0.5, 3.5, 3),
    )


def _probe_vector(n: int, seed: float) -> Array:
    """Reproduce the probe's synthetic vector exactly (same closed form)."""
    i = np.arange(n, dtype=np.float64)
    return np.sin(seed + 0.37 * i) + 1.5 + 0.011 * i * i


def test_probe_vectors_reproduce(cpp: dict[str, Any], mesher: FdmMesherComposite) -> None:
    """The synthetic inputs must match bit-for-bit before the outputs mean anything."""
    n = mesher.layout().size()
    _assert_array(_probe_vector(n, 0.3), cpp["tb_v"])
    _assert_array(_probe_vector(n, 1.7), cpp["tb_w"])


@pytest.mark.parametrize("direction", [0, 1])
def test_triple_band_ops_match_cpp(
    cpp: dict[str, Any], mesher: FdmMesherComposite, direction: int
) -> None:
    n = mesher.layout().size()
    v = _probe_vector(n, 0.3)
    w = _probe_vector(n, 1.7)
    d = FirstDerivativeOp(direction, mesher)
    p = f"tb_dir{direction}_"

    _assert_array(d.apply(v), cpp[p + "apply"])
    _assert_array(d.solve_splitting(v, 0.4, 1.0), cpp[p + "solve_splitting_a0.4_b1"])
    _assert_array(d.solve_splitting(v, -0.25, 2.0), cpp[p + "solve_splitting_am0.25_b2"])
    _assert_array(d.mult(w).apply(v), cpp[p + "mult_apply"])
    _assert_array(d.mult_r(w).apply(v), cpp[p + "multR_apply"])
    _assert_array(d.add(d.mult(w)).apply(v), cpp[p + "add_op_apply"])
    _assert_array(d.add(w).apply(v), cpp[p + "add_array_apply"])


@pytest.mark.parametrize("direction", [0, 1])
def test_triple_band_axpyb_matches_cpp(
    cpp: dict[str, Any], mesher: FdmMesherComposite, direction: int
) -> None:
    n = mesher.layout().size()
    v = _probe_vector(n, 0.3)
    w = _probe_vector(n, 1.7)
    d = FirstDerivativeOp(direction, mesher)
    x = d.mult(w)
    p = f"tb_dir{direction}_"

    t = TripleBandLinearOp(direction, mesher)
    t.axpyb(np.array([0.75]), x, d, np.array([-0.2]))
    _assert_array(t.apply(v), cpp[p + "axpyb_scalar_apply"])

    t2 = TripleBandLinearOp(direction, mesher)
    t2.axpyb(w, x, d, v)
    _assert_array(t2.apply(v), cpp[p + "axpyb_array_apply"])
