"""Cross-validate SymmetricSchurDecomposition against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``symmetric_schur`` section. The probe pins the **full eigenvector matrix**,
signs included, not a ``U D U^T`` residual: a residual is blind to exactly the
failure modes this port has to avoid (LAPACK ordering, LAPACK signs, a
different basis of a degenerate eigenspace).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.symmetric_schur_decomposition import (
    SymmetricSchurDecomposition,
)
from pquantlib.testing import reference_reader, tolerance

# The eigenvectors of a repeated eigenvalue are only determined up to a
# rotation inside its eigenspace, so the basis the Jacobi sweep lands on for
# this case is decided by rounding rather than by the algorithm. Its
# eigenvalues are still pinned; see ``test_degenerate_eigenspace_eigenvalues``.
_DEGENERATE_NULLSPACE = "rank_one_3x3"


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def _cases(cpp: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cpp["symmetric_schur"])


def test_eigenvalues_match_cpp(cpp: dict[str, Any]) -> None:
    """Eigenvalues match C++ to TIGHT, in the C++ decreasing order.

    Both sides run the same cyclic-Jacobi sweep in the same order; the only
    divergence is FMA contraction inside the compiled C++ rotations, which is
    a per-rotation relative perturbation of order ``eps``. Jacobi is backward
    stable, so the eigenvalues inherit ``O(n * eps * ||A||)`` — comfortably
    inside TIGHT for these matrices.
    """
    for case in _cases(cpp):
        dec = SymmetricSchurDecomposition(np.asarray(case["matrix"], dtype=np.float64))
        actual = dec.eigenvalues()
        expected = [float(e) for e in case["eigenvalues"]]
        assert actual.shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.tight(float(actual[i]), exp, reason=f"{case['name']}[{i}]")


def test_eigenvectors_match_cpp(cpp: dict[str, Any]) -> None:
    """Every eigenvector column matches C++ to TIGHT, sign included.

    The sign is not incidental: ``SymmetricSchurDecomposition`` pins the first
    component of each column non-negative, and ``rankReducedSqrt`` propagates
    those signs into the pseudo-root that drives the BGM diffusion term.
    """
    for case in _cases(cpp):
        if case["name"] == _DEGENERATE_NULLSPACE:
            continue
        dec = SymmetricSchurDecomposition(np.asarray(case["matrix"], dtype=np.float64))
        actual = dec.eigenvectors()
        expected = np.asarray(case["eigenvectors"], dtype=np.float64)
        assert actual.shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(actual[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_first_eigenvector_component_is_non_negative(cpp: dict[str, Any]) -> None:
    """The C++ sign convention holds for every column of every case."""
    for case in _cases(cpp):
        dec = SymmetricSchurDecomposition(np.asarray(case["matrix"], dtype=np.float64))
        ev = dec.eigenvectors()
        for j in range(ev.shape[1]):
            assert float(ev[0, j]) >= 0.0, f"{case['name']} column {j}"


def test_eigenvalues_are_decreasing(cpp: dict[str, Any]) -> None:
    """C++ sorts descending; LAPACK's ``eigh`` would sort ascending."""
    for case in _cases(cpp):
        dec = SymmetricSchurDecomposition(np.asarray(case["matrix"], dtype=np.float64))
        values = [float(v) for v in dec.eigenvalues()]
        assert values == sorted(values, reverse=True), case["name"]


def test_degenerate_eigenspace_eigenvalues(cpp: dict[str, Any]) -> None:
    """The rank-one case: eigenvalues pinned, eigenvector basis not.

    ``[[1,2,3],[2,4,6],[3,6,9]]`` has a two-dimensional null space, and the
    ``|lambda / lambda_max| < 1e-16`` guard snaps both of its eigenvalues to
    exactly ``0.0``. Any orthonormal basis of that plane is a correct answer,
    so the two null columns are compared for *span*, not entry by entry, while
    the eigenvalues (including the exact zeros) are pinned.
    """
    case = next(c for c in _cases(cpp) if c["name"] == _DEGENERATE_NULLSPACE)
    matrix = np.asarray(case["matrix"], dtype=np.float64)
    dec = SymmetricSchurDecomposition(matrix)
    expected = [float(e) for e in case["eigenvalues"]]
    for i, exp in enumerate(expected):
        tolerance.tight(float(dec.eigenvalues()[i]), exp, reason=f"eigenvalue[{i}]")
    tolerance.exact(float(dec.eigenvalues()[1]), 0.0)
    tolerance.exact(float(dec.eigenvalues()[2]), 0.0)
    # The null columns still have to be an orthonormal basis of ker(matrix).
    ev = dec.eigenvectors()
    for j in (1, 2):
        column = ev[:, j]
        for value in matrix @ column:
            tolerance.tight(float(value), 0.0)
        tolerance.tight(float(np.dot(column, column)), 1.0)


def test_reconstruction(cpp: dict[str, Any]) -> None:
    """``U D U^T == S`` for every case — a sanity check *on top of* the pins."""
    for case in _cases(cpp):
        matrix = np.asarray(case["matrix"], dtype=np.float64)
        dec = SymmetricSchurDecomposition(matrix)
        rebuilt = dec.eigenvectors() @ np.diag(dec.eigenvalues()) @ dec.eigenvectors().T
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                tolerance.loose(float(rebuilt[i, j]), float(matrix[i, j]))


def test_non_square_raises() -> None:
    with pytest.raises(LibraryException, match="square"):
        SymmetricSchurDecomposition(np.zeros((2, 3), dtype=np.float64))


def test_null_matrix_raises() -> None:
    with pytest.raises(LibraryException, match="null matrix"):
        SymmetricSchurDecomposition(np.zeros((0, 0), dtype=np.float64))
