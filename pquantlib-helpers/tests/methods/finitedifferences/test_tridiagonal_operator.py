"""Cross-validation tests for the legacy TridiagonalOperator (WS3-FD1).

# Retired-API compat layer — see package docstring.

PRIMARY gate: ``migration-harness/references/cluster/ws3fd1.json`` holds
applyTo / solveFor / SOR / identity-algebra results from genuine C++ v1.42.1
``TridiagonalOperator`` arithmetic. Python must reproduce them TIGHT.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)

_REF: dict[str, Any] = reference_reader.load("cluster/ws3fd1")


def _floats(block: str, key: str) -> list[float]:
    return [float(x) for x in _REF[block][key]]


def _generic() -> TridiagonalOperator:
    return TridiagonalOperator(
        low=np.full(4, -1.0),
        mid=np.full(5, 2.0),
        high=np.full(4, -1.0),
    )


def _v() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# --- construction / inspectors ---------------------------------------------


def test_size_matches_diagonal_length() -> None:
    assert _generic().size() == 5


def test_size_constructor_zero_initialises() -> None:
    op = TridiagonalOperator(5)
    assert op.size() == 5
    assert np.all(op.diagonal() == 0.0)
    assert np.all(op.lower_diagonal() == 0.0)
    assert np.all(op.upper_diagonal() == 0.0)


def test_size_one_is_rejected() -> None:
    # C++ requires size null or >= 2.
    with pytest.raises(LibraryException):
        TridiagonalOperator(1)


def test_explicit_diagonals_size_mismatch_rejected() -> None:
    with pytest.raises(LibraryException):
        TridiagonalOperator(low=np.zeros(3), mid=np.zeros(5), high=np.zeros(4))


def test_not_time_dependent_by_default() -> None:
    assert _generic().is_time_dependent() is False


# --- applyTo / solveFor / SOR vs C++ ---------------------------------------


def test_apply_to_matches_cpp() -> None:
    expected = _floats("tridiag_generic", "applyTo")
    result = _generic().apply_to(_v())
    for got, want in zip(result, expected, strict=True):
        tolerance.tight(float(got), want)


def test_solve_for_matches_cpp() -> None:
    expected = _floats("tridiag_generic", "solveFor")
    result = _generic().solve_for(_v())
    for got, want in zip(result, expected, strict=True):
        tolerance.tight(float(got), want)


def test_sor_matches_cpp_loose() -> None:
    # SOR is an iterative solver to a tolerance; the C++ probe used tol=1e-13,
    # so the residual (not the values) is bounded at ~1e-13. We assert LOOSE
    # (1e-8) — the per-element iterate agreement is well inside that.
    expected = _floats("tridiag_generic", "SOR")
    result = _generic().sor(_v(), 1e-13)
    for got, want in zip(result, expected, strict=True):
        tolerance.loose(float(got), want)


def test_solve_for_inverts_apply_to() -> None:
    # round-trip: solveFor(applyTo(v)) == v (well-posed for this SPD-ish op).
    op = _generic()
    v = _v()
    back = op.solve_for(op.apply_to(v))
    for got, want in zip(back, v, strict=True):
        tolerance.tight(float(got), float(want))


# --- operator algebra + identity vs C++ ------------------------------------


def test_identity_is_unit_diagonal() -> None:
    ident = _generic().identity(5)
    assert np.all(ident.diagonal() == 1.0)
    assert np.all(ident.lower_diagonal() == 0.0)
    assert np.all(ident.upper_diagonal() == 0.0)


def test_algebra_a_apply_matches_cpp() -> None:
    # A = I + 0.5*T ; emit A.applyTo(v).
    op = _generic()
    a = op.identity(5).add(op.multiply(0.5))
    # coefficient bands
    for got, want in zip(a.diagonal(), _floats("algebra", "A_diagonal"), strict=True):
        tolerance.tight(float(got), want)
    for got, want in zip(a.lower_diagonal(), _floats("algebra", "A_lower"), strict=True):
        tolerance.tight(float(got), want)
    for got, want in zip(a.upper_diagonal(), _floats("algebra", "A_upper"), strict=True):
        tolerance.tight(float(got), want)
    for got, want in zip(a.apply_to(_v()), _floats("algebra", "A_applyTo"), strict=True):
        tolerance.tight(float(got), want)


def test_algebra_b_solve_matches_cpp() -> None:
    # B = I + 0.25*T ; emit B.solveFor(v).
    op = _generic()
    b = op.identity(5).add(op.multiply(0.25))
    for got, want in zip(b.solve_for(_v()), _floats("algebra", "B_solveFor"), strict=True):
        tolerance.tight(float(got), want)


def test_solve_for_into_can_alias_rhs() -> None:
    # C++ contract: rhs and result may be the same Array.
    op = _generic()
    v = _v()
    expected = op.solve_for(v)
    op.solve_for_into(v, v)  # alias
    for got, want in zip(v, expected, strict=True):
        tolerance.tight(float(got), float(want))
