"""Cross-validate TqrEigenDecomposition against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``tqr_eigen`` section. Every ``EigenVectorCalculation`` x ``ShiftStrategy``
combination is walked, and each case pins the eigenvalues, the **full**
eigenvector block (signs included) and the iteration count — the last of which
is what actually separates this implicit-shift sweep from LAPACK's
``eigh_tridiagonal``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.tqr_eigen_decomposition import (
    EigenVectorCalculation,
    ShiftStrategy,
    TqrEigenDecomposition,
)
from pquantlib.testing import reference_reader, tolerance

_CALCS: dict[str, EigenVectorCalculation] = {
    "WithEigenVector": EigenVectorCalculation.WITH_EIGEN_VECTOR,
    "WithoutEigenVector": EigenVectorCalculation.WITHOUT_EIGEN_VECTOR,
    "OnlyFirstRowEigenVector": EigenVectorCalculation.ONLY_FIRST_ROW_EIGEN_VECTOR,
}

_SHIFTS: dict[str, ShiftStrategy] = {
    "NoShift": ShiftStrategy.NO_SHIFT,
    "Overrelaxation": ShiftStrategy.OVERRELAXATION,
    "CloseEigenValue": ShiftStrategy.CLOSE_EIGEN_VALUE,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def _build(case: dict[str, Any]) -> TqrEigenDecomposition:
    return TqrEigenDecomposition(
        np.asarray(case["diag"], dtype=np.float64),
        np.asarray(case["sub"], dtype=np.float64),
        _CALCS[case["calc"]],
        _SHIFTS[case["strategy"]],
    )


def _label(case: dict[str, Any]) -> str:
    return f"{case['name']}/{case['calc']}/{case['strategy']}"


def test_iterations_match_cpp_exactly(cpp: dict[str, Any]) -> None:
    """The QR sweep count is an integer and must agree exactly.

    It is the sharpest single discriminator available: a different shift
    strategy, a different convergence test or a different deflation order all
    change it, while leaving the eigenvalues correct.
    """
    for case in cpp["tqr_eigen"]:
        dec = _build(case)
        assert dec.iterations() == int(case["iterations"]), _label(case)


def test_eigenvalues_match_cpp(cpp: dict[str, Any]) -> None:
    """Eigenvalues match C++ to TIGHT, in the C++ decreasing order.

    Both sides run the identical sequence of Givens rotations; the divergence
    is FMA contraction inside the compiled C++, i.e. an ``O(eps)`` relative
    perturbation per rotation on a backward-stable iteration.
    """
    for case in cpp["tqr_eigen"]:
        dec = _build(case)
        expected = [float(e) for e in case["eigenvalues"]]
        actual = dec.eigenvalues()
        assert actual.shape[0] == len(expected), _label(case)
        for i, exp in enumerate(expected):
            tolerance.tight(float(actual[i]), exp, reason=f"{_label(case)}[{i}]")


def test_eigenvectors_match_cpp(cpp: dict[str, Any]) -> None:
    """The accumulated eigenvector block matches C++ to TIGHT, sign included.

    ``WithoutEigenVector`` yields a zero-row block, which the C++ probe
    serialises as an empty JSON array.
    """
    for case in cpp["tqr_eigen"]:
        dec = _build(case)
        actual = dec.eigenvectors()
        rows = list(case["eigenvectors"])
        assert actual.shape[0] == len(rows), _label(case)
        for i, row in enumerate(rows):
            expected_row = [float(e) for e in row]
            assert actual.shape[1] == len(expected_row), _label(case)
            for j, exp in enumerate(expected_row):
                tolerance.tight(
                    float(actual[i, j]), exp, reason=f"{_label(case)}[{i}][{j}]"
                )


def test_eigenvector_block_shape_follows_calc_mode(cpp: dict[str, Any]) -> None:
    """0 / 1 / n rows for WithoutEigenVector / OnlyFirstRow / WithEigenVector."""
    expected_rows = {
        "WithoutEigenVector": 0,
        "OnlyFirstRowEigenVector": 1,
    }
    for case in cpp["tqr_eigen"]:
        dec = _build(case)
        n = len(case["diag"])
        assert dec.eigenvectors().shape == (expected_rows.get(case["calc"], n), n), _label(case)


def test_eigenvalues_are_decreasing(cpp: dict[str, Any]) -> None:
    """C++ sorts descending; ``eigh_tridiagonal`` would sort ascending."""
    for case in cpp["tqr_eigen"]:
        values = [float(v) for v in _build(case).eigenvalues()]
        assert values == sorted(values, reverse=True), _label(case)


def test_first_eigenvector_component_is_non_negative(cpp: dict[str, Any]) -> None:
    """C++ pins the first stored component of every eigenvector non-negative."""
    for case in cpp["tqr_eigen"]:
        ev = _build(case).eigenvectors()
        if ev.shape[0] == 0:
            continue
        for j in range(ev.shape[1]):
            assert float(ev[0, j]) >= 0.0, f"{_label(case)} column {j}"


def test_golub_welsch_first_row_is_what_quadrature_consumes(cpp: dict[str, Any]) -> None:
    """The Gauss-Hermite(5) Jacobi matrix reproduces the analytic rule.

    ``GaussianQuadrature`` asks for ``OnlyFirstRowEigenVector`` +
    ``Overrelaxation`` and then forms ``w_i = mu_0 * ev[0][i]**2``. With
    ``alpha == 0`` and ``beta(i) == i/2`` the nodes are the roots of the
    physicists' Hermite polynomial ``H_5`` and the weights are the classical
    Gauss-Hermite weights, so this checks the port end to end against a
    closed-form rule rather than only against the probe.
    """
    case = next(
        c
        for c in cpp["tqr_eigen"]
        if c["name"] == "gauss_hermite_5"
        and c["calc"] == "OnlyFirstRowEigenVector"
        and c["strategy"] == "Overrelaxation"
    )
    dec = _build(case)
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    # numpy returns the nodes ascending; C++ returns them descending.
    for i in range(5):
        tolerance.loose(float(dec.eigenvalues()[i]), float(nodes[4 - i]))
    mu_0 = np.sqrt(np.pi)
    for i in range(5):
        w = mu_0 * float(dec.eigenvectors()[0, i]) ** 2
        tolerance.loose(w, float(weights[4 - i]))


def test_wrong_dimensions_raises() -> None:
    with pytest.raises(LibraryException, match="Wrong dimensions"):
        TqrEigenDecomposition(
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
