"""Cross-validate Cholesky decomposition against the C++ probes.

References:

* ``migration-harness/references/cluster/e.json`` (``cholesky`` key) — the
  original L1-E probe, a single SPD 3x3.
* ``migration-harness/references/v143/math/matrixutilities.json``
  (``cholesky`` section) — the v1.43 probe, which additionally pins the
  ``flexible=True`` semi-definite arm, the ``CholeskySolveFor`` triangular
  solve, and the fact that only the **upper** triangle of the input is read.

That last case is what retired the previous ``scipy.linalg.cholesky(m,
lower=True)`` delegation: LAPACK reads the opposite triangle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.cholesky import (
    cholesky_decomposition,
    cholesky_solve_for,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/e")


@pytest.fixture(scope="module")
def cpp143() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def test_cholesky_matches_cpp(cpp: dict[str, Any]) -> None:
    m = np.array([[4.0, 2.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0]])
    factor = cholesky_decomposition(m)
    expected = np.array(cpp["cholesky"])
    for i in range(3):
        for j in range(3):
            tolerance.tight(float(factor[i, j]), float(expected[i][j]))


def test_factor_matches_v143_probe(cpp143: dict[str, Any]) -> None:
    """Every case's ``L`` matches C++ entry by entry, both ``flexible`` arms.

    Both sides run the same Cholesky-Banachiewicz loop in the same order, so
    the divergence is FMA contraction inside the compiled C++ — an ``O(eps)``
    relative perturbation on a backward-stable factorisation.
    """
    for case in cpp143["cholesky"]:
        factor = cholesky_decomposition(
            np.asarray(case["matrix"], dtype=np.float64), bool(case["flexible"])
        )
        expected = np.asarray(case["L"], dtype=np.float64)
        assert factor.shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(factor[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_solve_for_matches_v143_probe(cpp143: dict[str, Any]) -> None:
    """``CholeskySolveFor`` matches C++ where the factor is non-singular.

    The probe records ``null`` for the semi-definite cases, where the solve
    divides by a zero diagonal and is not defined.
    """
    for case in cpp143["cholesky"]:
        if case["solve_for"] is None:
            continue
        factor = cholesky_decomposition(
            np.asarray(case["matrix"], dtype=np.float64), bool(case["flexible"])
        )
        actual = cholesky_solve_for(factor, np.asarray(case["b"], dtype=np.float64))
        expected = [float(e) for e in case["solve_for"]]
        assert actual.shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.tight(float(actual[i]), exp, reason=f"{case['name']}[{i}]")


def test_only_the_upper_triangle_is_read(cpp143: dict[str, Any]) -> None:
    """A matrix whose lower triangle is garbage factors as if it were symmetric.

    ``scipy.linalg.cholesky(..., lower=True)`` reads the lower triangle and
    would return a different factor (or raise) here. This is the case that
    proves the old delegation's premise false.
    """
    asymmetric = next(c for c in cpp143["cholesky"] if c["name"] == "upper_triangle_only")
    spd = next(c for c in cpp143["cholesky"] if c["name"] == "spd_3x3")
    from_asymmetric = cholesky_decomposition(np.asarray(asymmetric["matrix"], dtype=np.float64))
    from_spd = cholesky_decomposition(np.asarray(spd["matrix"], dtype=np.float64))
    for i in range(3):
        for j in range(3):
            tolerance.exact(float(from_asymmetric[i, j]), float(from_spd[i, j]))


def test_flexible_clamps_a_semi_definite_pivot(cpp143: dict[str, Any]) -> None:
    """``flexible=True`` zeroes a non-positive pivot and the column below it."""
    case = next(c for c in cpp143["cholesky"] if c["name"] == "psd_rank2_3x3")
    m = np.asarray(case["matrix"], dtype=np.float64)
    factor = cholesky_decomposition(m, flexible=True)
    tolerance.exact(float(factor[2, 2]), 0.0)
    rebuilt = factor @ factor.T
    for i in range(3):
        for j in range(3):
            tolerance.tight(float(rebuilt[i, j]), float(m[i, j]))


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


def test_solve_for_round_trip() -> None:
    """``L L.T x == b`` for the SPD probe matrix."""
    m = np.array([[4.0, 2.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0]])
    b = np.array([1.0, 2.0, 3.0])
    x = cholesky_solve_for(cholesky_decomposition(m), b)
    for i, value in enumerate(m @ x):
        tolerance.tight(float(value), float(b[i]))


def test_cholesky_non_square_raises() -> None:
    with pytest.raises(LibraryException, match="square"):
        cholesky_decomposition(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_cholesky_not_2d_raises() -> None:
    with pytest.raises(LibraryException, match="2-D"):
        cholesky_decomposition(np.array([1.0, 2.0, 3.0]))


def test_cholesky_non_psd_raises() -> None:
    # Indefinite (eigenvalues 1 and -1).
    m = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(LibraryException, match="not positive definite"):
        cholesky_decomposition(m)


def test_cholesky_flexible_accepts_indefinite(cpp143: dict[str, Any]) -> None:
    """The indefinite 2x2 that raises without ``flexible`` factors to zeros."""
    case = next(c for c in cpp143["cholesky"] if c["name"] == "indefinite_2x2")
    factor = cholesky_decomposition(np.asarray(case["matrix"], dtype=np.float64), flexible=True)
    expected = np.asarray(case["L"], dtype=np.float64)
    for i in range(2):
        for j in range(2):
            tolerance.exact(float(factor[i, j]), float(expected[i, j]))


def test_solve_for_size_mismatch_raises() -> None:
    with pytest.raises(LibraryException, match="does not match"):
        cholesky_solve_for(np.eye(3), np.array([1.0, 2.0]))
