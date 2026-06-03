"""Unit tests for the legacy FD boundary conditions (WS3-FD2).

# Retired-API compat layer — see package docstring.

Dirichlet/Neumann have no standalone C++ test; their behaviour is well-defined
and is checked here against hand-derived expectations (operator-row rewrites +
array-entry pins), plus an end-to-end consistency check that a Dirichlet-pinned
``MixedScheme`` step preserves the pinned boundary value.
"""

from __future__ import annotations

import numpy as np

from pquantlib.testing import tolerance
from pquantlib_helpers.methods.finitedifferences.boundary_condition import (
    BoundaryConditionLike,
    BoundaryConditionSet,
    DirichletBC,
    NeumannBC,
    Side,
)
from pquantlib_helpers.methods.finitedifferences.crank_nicolson import CrankNicolson
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)


def _op(n: int) -> TridiagonalOperator:
    return TridiagonalOperator(
        low=np.full(n - 1, 2.0),
        mid=np.full(n, 3.0),
        high=np.full(n - 1, 4.0),
    )


# --- Dirichlet --------------------------------------------------------------


def test_dirichlet_lower_pins_value_after_applying() -> None:
    a = np.array([7.0, 1.0, 2.0, 3.0, 4.0])
    DirichletBC(99.0, Side.Lower).apply_after_applying(a)
    tolerance.exact(float(a[0]), 99.0)
    # interior untouched
    tolerance.exact(float(a[2]), 2.0)


def test_dirichlet_upper_pins_value_after_applying() -> None:
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    DirichletBC(55.0, Side.Upper).apply_after_applying(a)
    tolerance.exact(float(a[-1]), 55.0)


def test_dirichlet_lower_rewrites_operator_first_row_to_identity() -> None:
    op = _op(5)
    DirichletBC(99.0, Side.Lower).apply_before_applying(op)
    tolerance.exact(float(op.diagonal()[0]), 1.0)
    tolerance.exact(float(op.upper_diagonal()[0]), 0.0)


def test_dirichlet_upper_rewrites_operator_last_row_to_identity() -> None:
    op = _op(5)
    DirichletBC(99.0, Side.Upper).apply_before_applying(op)
    tolerance.exact(float(op.diagonal()[-1]), 1.0)
    tolerance.exact(float(op.lower_diagonal()[-1]), 0.0)


def test_dirichlet_before_solving_sets_identity_row_and_rhs() -> None:
    op = _op(5)
    rhs = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    bc = DirichletBC(42.0, Side.Lower)
    bc.apply_before_solving(op, rhs)
    tolerance.exact(float(op.diagonal()[0]), 1.0)
    tolerance.exact(float(op.upper_diagonal()[0]), 0.0)
    tolerance.exact(float(rhs[0]), 42.0)
    # solving the identity-pinned system returns the pinned value at the boundary
    x = op.solve_for(rhs)
    tolerance.tight(float(x[0]), 42.0)


# --- Neumann ----------------------------------------------------------------


def test_neumann_lower_imposes_first_derivative_after_applying() -> None:
    # u[0] = u[1] - value
    a = np.array([0.0, 5.0, 6.0, 7.0])
    NeumannBC(2.0, Side.Lower).apply_after_applying(a)
    tolerance.exact(float(a[0]), 3.0)


def test_neumann_upper_imposes_first_derivative_after_applying() -> None:
    # u[n-1] = u[n-2] + value
    a = np.array([1.0, 2.0, 3.0, 10.0])
    NeumannBC(4.0, Side.Upper).apply_after_applying(a)
    tolerance.exact(float(a[-1]), 7.0)


def test_neumann_lower_rewrites_operator_to_difference_row() -> None:
    # C++/Java setFirstRow(-1.0, 1.0) -> diagonal[0]=-1, upper[0]=1
    op = _op(5)
    NeumannBC(2.0, Side.Lower).apply_before_applying(op)
    tolerance.exact(float(op.diagonal()[0]), -1.0)
    tolerance.exact(float(op.upper_diagonal()[0]), 1.0)


def test_neumann_upper_rewrites_operator_to_difference_row() -> None:
    # C++/Java setLastRow(-1.0, 1.0) -> lower[n-2]=-1, diagonal[n-1]=1
    op = _op(5)
    NeumannBC(2.0, Side.Upper).apply_before_applying(op)
    tolerance.exact(float(op.diagonal()[-1]), 1.0)
    tolerance.exact(float(op.lower_diagonal()[-1]), -1.0)


def test_neumann_before_solving_sets_difference_row_and_rhs() -> None:
    op = _op(5)
    rhs = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    NeumannBC(2.5, Side.Lower).apply_before_solving(op, rhs)
    tolerance.exact(float(op.diagonal()[0]), -1.0)
    tolerance.exact(float(op.upper_diagonal()[0]), 1.0)
    tolerance.exact(float(rhs[0]), 2.5)


def test_neumann_after_solving_is_noop() -> None:
    a = np.array([1.0, 2.0, 3.0])
    NeumannBC(2.0, Side.Lower).apply_after_solving(a)
    assert np.array_equal(a, np.array([1.0, 2.0, 3.0]))


# --- BoundaryConditionSet ---------------------------------------------------


def test_boundary_condition_set_stores_and_returns_lists_in_order() -> None:
    bc_set: BoundaryConditionSet = BoundaryConditionSet()
    lo: list[BoundaryConditionLike] = [DirichletBC(1.0, Side.Lower)]
    up: list[BoundaryConditionLike] = [DirichletBC(2.0, Side.Upper)]
    bc_set.push_back(lo)
    bc_set.push_back(up)
    assert bc_set.get(0) is lo
    assert bc_set.get(1) is up


# --- End-to-end: Dirichlet BC through a real MixedScheme step ---------------


def test_dirichlet_lower_preserved_through_crank_nicolson_step() -> None:
    """A DirichletBC registered with CrankNicolson pins the boundary entry.

    This is the end-to-end consistency check advertised in the module docstring:
    after a full Crank-Nicolson ``step()``, the lower-boundary entry must equal
    the Dirichlet value regardless of the interior evolution.

    The four BC hooks fire in order:
      explicit half: apply_before_applying → apply_to → apply_after_applying
      implicit half: apply_before_solving  → solve_for → apply_after_solving

    ``apply_after_applying`` hard-pins ``a[0]`` to ``value`` on the explicit
    half; ``apply_before_solving`` rewrites the operator's first row to the
    identity and sets ``rhs[0] = value``, so ``solve_for`` returns ``value``
    at index 0 exactly. We verify both the lower boundary is pinned AND the
    interior nodes have changed (i.e. the interior evolution did actually run).
    """
    pinned_value = 42.0
    dt = 0.01

    # 5-node grid.  Off-diagonals have length 4 (= n - 1).
    # The boundary rows will be overwritten by the BC hooks.
    op = TridiagonalOperator(
        low=np.array([1.0, 1.0, 1.0, 1.0]),
        mid=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        high=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    bc = DirichletBC(pinned_value, Side.Lower)
    scheme = CrankNicolson(op, bcs=[bc])
    scheme.set_step(dt)

    # Initial array: all ones except the boundary node.
    a = np.array([99.0, 1.0, 1.0, 1.0, 1.0])
    a_out = scheme.step(a, dt)

    # Lower boundary must equal the Dirichlet value exactly.
    tolerance.exact(float(a_out[0]), pinned_value)

    # Sanity: interior nodes must have changed (scheme actually ran).
    assert float(a_out[1]) != 1.0, (
        "interior node 1 was not updated — the scheme may not have run"
    )
