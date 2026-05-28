"""Tests for FirstDerivativeOp + SecondDerivativeOp on a uniform 1-D grid.

# C++ parity: ql/methods/finitedifferences/operators/firstderivativeop.{hpp,cpp},
# ql/methods/finitedifferences/operators/secondderivativeop.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``deriv_ops_x_squared`` + ``deriv_ops_x_cubed``
sections of ``migration-harness/references/cluster/l5d.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.uniform_grid_mesher import (
    UniformGridMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l5d")


def _build_mesh() -> UniformGridMesher:
    layout = FdmLinearOpLayout((11,))
    return UniformGridMesher(layout, [(-2.0, 2.0)])


def test_first_derivative_x_squared_central_nodes(reference_data: dict[str, Any]) -> None:
    """Central first-derivative applied to f(x)=x^2 — interior gives 2x.

    TIGHT-tier: the closed-form coefficients and uniform mesh use
    identical floating-point operations in C++ and Python, so the
    output should agree to TIGHT (1e-14 abs).
    """
    mesh = _build_mesh()
    x = mesh.locations(0)
    fx2 = x * x
    d1 = FirstDerivativeOp(0, mesh)
    out = d1.apply(fx2)
    expected = reference_data["deriv_ops_x_squared"]["d1_apply_x_squared"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_second_derivative_x_squared(reference_data: dict[str, Any]) -> None:
    """Central second-derivative applied to f(x)=x^2 — interior is exactly 2."""
    mesh = _build_mesh()
    x = mesh.locations(0)
    fx2 = x * x
    d2 = SecondDerivativeOp(0, mesh)
    out = d2.apply(fx2)
    expected = reference_data["deriv_ops_x_squared"]["d2_apply_x_squared"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_first_derivative_x_cubed(reference_data: dict[str, Any]) -> None:
    """First-derivative applied to f(x)=x^3 — interior LOOSE (O(h^2) error).

    TIGHT-tier here: the *raw* stencil output (3 x^2 + O(h^2)) is
    *bit-identical* between C++ and Python because both use the same
    closed-form coefficients applied to the same uniform mesh — we're
    not measuring discretisation accuracy, we're measuring stencil
    parity.
    """
    mesh = _build_mesh()
    x = mesh.locations(0)
    fx3 = x**3
    d1 = FirstDerivativeOp(0, mesh)
    out = d1.apply(fx3)
    expected = reference_data["deriv_ops_x_cubed"]["d1_apply_x_cubed"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_second_derivative_x_cubed(reference_data: dict[str, Any]) -> None:
    """Second-derivative applied to f(x)=x^3 — interior approximates 6x."""
    mesh = _build_mesh()
    x = mesh.locations(0)
    fx3 = x**3
    d2 = SecondDerivativeOp(0, mesh)
    out = d2.apply(fx3)
    expected = reference_data["deriv_ops_x_cubed"]["d2_apply_x_cubed"]
    for actual_v, expected_v in zip(out, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_first_derivative_boundary_upwinding() -> None:
    """At index 0 (lower boundary) the first-derivative uses
    *upwinding* — lower=0, diag=-1/hp, upper=1/hp.

    Applied to f(x)=x^2 at the boundary: diag*f[0] + upper*f[1] =
    -1/hp * 4 + 1/hp * 2.56 = (2.56-4)/0.4 = -3.6.
    """
    mesh = _build_mesh()
    x = mesh.locations(0)
    fx2 = x * x
    d1 = FirstDerivativeOp(0, mesh)
    out = d1.apply(fx2)
    tight(float(out[0]), -3.6)


def test_second_derivative_boundary_is_zero() -> None:
    """At boundary nodes the second-derivative stencil is identically zero."""
    mesh = _build_mesh()
    fx_any = np.ones(11, dtype=np.float64)
    d2 = SecondDerivativeOp(0, mesh)
    out = d2.apply(fx_any)
    tight(float(out[0]), 0.0)
    tight(float(out[10]), 0.0)


def test_inconsistent_input_length_raises() -> None:
    """apply with wrong-length input vector must raise."""
    mesh = _build_mesh()
    d1 = FirstDerivativeOp(0, mesh)
    bad = np.zeros(5, dtype=np.float64)
    with pytest.raises(LibraryException):
        d1.apply(bad)


def test_to_matrix_round_trip_apply() -> None:
    """``to_matrix() @ r`` must agree with ``apply(r)`` for any r.

    LOOSE: dense .dot() of a sparse matrix produces slightly different
    floating-point ordering vs the hand-rolled fancy-indexing in
    ``apply``; the discrepancy is O(eps*N) — within LOOSE bound.
    """
    mesh = _build_mesh()
    d2 = SecondDerivativeOp(0, mesh)
    r = np.linspace(1.0, 11.0, 11, dtype=np.float64)
    via_apply = d2.apply(r)
    # scipy sparse @ dense returns a numpy array (no type stubs — cast).
    via_matrix = np.asarray(d2.to_matrix() @ r, dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]
    for a, b in zip(via_apply, via_matrix, strict=True):
        loose(float(a), float(b))
