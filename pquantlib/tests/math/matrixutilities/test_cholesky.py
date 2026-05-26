"""Cross-validate Cholesky decomposition against the L1-E C++ probe.

Reference: ``migration-harness/references/cluster/e.json`` (``cholesky`` key).

Probe matrix: ``[[4, 2, 2], [2, 3, 1], [2, 1, 5]]`` — symmetric and
positive-definite. Expected lower-triangular factor is given in the JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.cholesky import cholesky_decomposition
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/e")


def test_cholesky_matches_cpp(cpp: dict[str, Any]) -> None:
    m = np.array([[4.0, 2.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0]])
    factor = cholesky_decomposition(m)
    expected = np.array(cpp["cholesky"])
    for i in range(3):
        for j in range(3):
            tolerance.tight(float(factor[i, j]), float(expected[i][j]))


def test_cholesky_reconstructs_matrix() -> None:
    m = np.array([[4.0, 2.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0]])
    factor = cholesky_decomposition(m)
    reconstructed = factor @ factor.T
    for i in range(3):
        for j in range(3):
            tolerance.tight(float(reconstructed[i, j]), float(m[i, j]))


def test_cholesky_is_lower_triangular() -> None:
    m = np.array([[4.0, 2.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0]])
    factor = cholesky_decomposition(m)
    # Upper triangle (strict) must be zero.
    for i in range(3):
        for j in range(i + 1, 3):
            tolerance.exact(float(factor[i, j]), 0.0)


def test_cholesky_identity() -> None:
    eye = np.eye(4)
    factor = cholesky_decomposition(eye)
    for i in range(4):
        for j in range(4):
            tolerance.exact(float(factor[i, j]), float(eye[i, j]))


def test_cholesky_non_square_raises() -> None:
    with pytest.raises(LibraryException, match="square"):
        cholesky_decomposition(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_cholesky_not_2d_raises() -> None:
    with pytest.raises(LibraryException, match="2-D"):
        cholesky_decomposition(np.array([1.0, 2.0, 3.0]))


def test_cholesky_non_psd_raises() -> None:
    # Indefinite (eigenvalues 1 and -1).
    m = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(LibraryException, match="failed"):
        cholesky_decomposition(m)


def test_cholesky_flexible_branch_deferred() -> None:
    m = np.array([[4.0, 2.0], [2.0, 3.0]])
    with pytest.raises(LibraryException, match="deferred"):
        cholesky_decomposition(m, flexible=True)
