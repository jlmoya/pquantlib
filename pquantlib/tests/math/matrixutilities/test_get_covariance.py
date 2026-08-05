"""Cross-validate get_covariance / CovarianceDecomposition against the C++ probe.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``covariance`` section. The two are inverses of each other, and the probe pins
the round trip in both directions plus the explicit symmetrisation
``0.5 * (c[i][j] + c[j][i])`` that ``get_covariance`` performs on an input that
is only symmetric to within tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.get_covariance import (
    CovarianceDecomposition,
    get_covariance,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def test_get_covariance_matches_cpp(cpp: dict[str, Any]) -> None:
    """``covariance[i][j] == sd_i sd_j * 0.5 (c[i][j] + c[j][i])``, entry by entry."""
    for case in cpp["covariance"]:
        if not case["std_devs"]:
            continue
        actual = get_covariance(
            [float(v) for v in case["std_devs"]],
            np.asarray(case["correlation"], dtype=np.float64),
        )
        expected = np.asarray(case["covariance"], dtype=np.float64)
        assert actual.shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(actual[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_covariance_decomposition_matches_cpp(cpp: dict[str, Any]) -> None:
    """Variances, standard deviations and the correlation matrix all match."""
    for case in cpp["covariance"]:
        dec = CovarianceDecomposition(np.asarray(case["covariance"], dtype=np.float64))
        for i, exp in enumerate([float(e) for e in case["variances"]]):
            tolerance.tight(float(dec.variances()[i]), exp, reason=f"{case['name']}[{i}]")
        for i, exp in enumerate([float(e) for e in case["standard_deviations"]]):
            tolerance.tight(
                float(dec.standard_deviations()[i]), exp, reason=f"{case['name']}[{i}]"
            )
        expected = np.asarray(case["correlation_matrix"], dtype=np.float64)
        assert dec.correlation_matrix().shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(dec.correlation_matrix()[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_decomposition_diagonal_is_exactly_one(cpp: dict[str, Any]) -> None:
    """C++ writes ``1.0`` on the diagonal rather than computing ``c/(sd sd)``."""
    for case in cpp["covariance"]:
        dec = CovarianceDecomposition(np.asarray(case["covariance"], dtype=np.float64))
        corr = dec.correlation_matrix()
        for i in range(corr.shape[0]):
            tolerance.exact(float(corr[i, i]), 1.0)


def test_round_trip(cpp: dict[str, Any]) -> None:
    """``get_covariance(decompose(cov)) == cov`` — the C++ cross-check."""
    for case in cpp["covariance"]:
        cov = np.asarray(case["covariance"], dtype=np.float64)
        dec = CovarianceDecomposition(cov)
        rebuilt = get_covariance(
            [float(v) for v in dec.standard_deviations()], dec.correlation_matrix()
        )
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                tolerance.loose(float(rebuilt[i, j]), float(cov[i, j]), reason=case["name"])


def test_nearly_symmetric_correlation_is_averaged(cpp: dict[str, Any]) -> None:
    """A 4e-13 asymmetry inside the 1e-12 tolerance is averaged, not rejected.

    ``covariance[i][j]`` must land on ``sd_i sd_j * (0.5 + 2e-13)`` — strictly
    between the two input entries — which is what distinguishes the C++
    formula from simply reading the lower triangle.
    """
    case = next(c for c in cpp["covariance"] if c["name"] == "nearly_symmetric")
    corr = np.asarray(case["correlation"], dtype=np.float64)
    sd = [float(v) for v in case["std_devs"]]
    cov = get_covariance(sd, corr)
    averaged = 0.5 * (float(corr[0, 1]) + float(corr[1, 0]))
    assert float(corr[0, 1]) < averaged < float(corr[1, 0])
    tolerance.tight(float(cov[1, 0]), sd[0] * sd[1] * averaged)


def test_asymmetric_correlation_raises() -> None:
    corr = np.array([[1.0, 0.5], [0.4, 1.0]], dtype=np.float64)
    with pytest.raises(LibraryException, match="not symmetric"):
        get_covariance([0.2, 0.3], corr)


def test_non_unit_diagonal_raises() -> None:
    corr = np.array([[1.0, 0.5], [0.5, 0.9]], dtype=np.float64)
    with pytest.raises(LibraryException, match="invalid correlation matrix"):
        get_covariance([0.2, 0.3], corr)


def test_dimension_mismatch_raises() -> None:
    corr = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    with pytest.raises(LibraryException, match="dimension mismatch"):
        get_covariance([0.2, 0.3, 0.4], corr)


def test_non_square_covariance_raises() -> None:
    with pytest.raises(LibraryException, match="must be square"):
        CovarianceDecomposition(np.zeros((2, 3), dtype=np.float64))


def test_asymmetric_covariance_raises() -> None:
    cov = np.array([[0.04, 0.01], [0.02, 0.09]], dtype=np.float64)
    with pytest.raises(LibraryException, match="invalid covariance matrix"):
        CovarianceDecomposition(cov)
