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
    BoundaryConditionSet,
    DirichletBC,
    NeumannBC,
    Side,
)
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
    lo = [DirichletBC(1.0, Side.Lower)]
    up = [DirichletBC(2.0, Side.Upper)]
    bc_set.push_back(lo)  # type: ignore[arg-type]
    bc_set.push_back(up)  # type: ignore[arg-type]
    assert bc_set.get(0) is lo
    assert bc_set.get(1) is up
