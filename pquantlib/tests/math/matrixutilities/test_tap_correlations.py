"""Cross-validate the TAP correlation parametrisations and FrobeniusCostFunction.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``tap_correlations`` and ``frobenius_cost_function`` sections.

The sharpest thing to pin about :class:`FrobeniusCostFunction` is that it
overrides ``value`` with the plain **sum of squares** of the residuals, not the
``sqrt(mean(...))`` its ``CostFunction`` base returns. Two calibrations that
differ only in that factor converge to different points under a
tolerance-driven optimizer, so the test asserts the override explicitly rather
than only comparing numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.math.matrixutilities.tap_correlations import (
    FrobeniusCostFunction,
    Parametrisation,
    lmm_triangular_angles_parametrization,
    lmm_triangular_angles_parametrization_unconstrained,
    triangular_angles_parametrization,
    triangular_angles_parametrization_rank_three,
    triangular_angles_parametrization_rank_three_vectorial,
    triangular_angles_parametrization_unconstrained,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def _rank_three(x: Array, matrix_size: int, rank: int) -> Matrix:
    del rank
    return triangular_angles_parametrization_rank_three_vectorial(x, matrix_size)


_PARAMETRISATIONS: dict[str, Parametrisation] = {
    "triangular_unconstrained": triangular_angles_parametrization_unconstrained,
    "lmm_triangular_unconstrained": lmm_triangular_angles_parametrization_unconstrained,
    "rank_three_vectorial": _rank_three,
}


def _assert_matrix(actual: Matrix, expected: list[list[float]], label: str) -> None:
    exp = np.asarray(expected, dtype=np.float64)
    assert actual.shape == exp.shape, label
    for i in range(exp.shape[0]):
        for j in range(exp.shape[1]):
            tolerance.tight(float(actual[i, j]), float(exp[i, j]), reason=f"{label}[{i}][{j}]")


def test_triangular_matches_cpp(cpp: dict[str, Any]) -> None:
    """Equation (24), rank 3 out of 5, entry by entry."""
    block = cpp["tap_correlations"]
    angles = np.asarray(block["angles7"], dtype=np.float64)
    _assert_matrix(
        triangular_angles_parametrization(angles, 5, 3), block["triangular_5_3"], "tri_5_3"
    )


def test_triangular_full_rank_matches_cpp(cpp: dict[str, Any]) -> None:
    """``rank == matrixSize`` needs ``n(n-1)/2`` angles and fills the triangle."""
    block = cpp["tap_correlations"]
    angles = np.asarray(block["angles10"], dtype=np.float64)
    _assert_matrix(
        triangular_angles_parametrization(angles, 5, 5), block["triangular_5_5"], "tri_5_5"
    )


def test_triangular_unconstrained_matches_cpp(cpp: dict[str, Any]) -> None:
    """``theta_i = pi/2 - arctan(x_i)`` then equation (24)."""
    block = cpp["tap_correlations"]
    x = np.asarray(block["x7"], dtype=np.float64)
    _assert_matrix(
        triangular_angles_parametrization_unconstrained(x, 5, 3),
        block["triangular_unconstrained_5_3"],
        "tri_unc_5_3",
    )


def test_lmm_triangular_matches_cpp(cpp: dict[str, Any]) -> None:
    """The LMM variant: one angle per row, applied cumulatively down the rows."""
    block = cpp["tap_correlations"]
    angles = np.asarray(block["angles4"], dtype=np.float64)
    _assert_matrix(
        lmm_triangular_angles_parametrization(angles, 5, 5), block["lmm_triangular_5"], "lmm_5"
    )


def test_lmm_triangular_unconstrained_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["tap_correlations"]
    x = np.asarray(block["x4"], dtype=np.float64)
    _assert_matrix(
        lmm_triangular_angles_parametrization_unconstrained(x, 5, 5),
        block["lmm_triangular_unconstrained_5"],
        "lmm_unc_5",
    )


def test_rank_three_matches_cpp(cpp: dict[str, Any]) -> None:
    """Equation (32): the 3-D spherical spiral, ``(nbRows, 3)``."""
    block = cpp["tap_correlations"]
    _assert_matrix(
        triangular_angles_parametrization_rank_three(0.6, 2.5, -0.35, 6),
        block["rank_three_6"],
        "rank_three_6",
    )


def test_rank_three_vectorial_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["tap_correlations"]
    params = np.asarray(block["rank3_params"], dtype=np.float64)
    _assert_matrix(
        triangular_angles_parametrization_rank_three_vectorial(params, 6),
        block["rank_three_vectorial_6"],
        "rank_three_vectorial_6",
    )


def test_pseudo_root_gives_a_unit_diagonal_correlation(cpp: dict[str, Any]) -> None:
    """``B B^T`` has a unit diagonal for every angle vector — the point of (24)."""
    block = cpp["tap_correlations"]
    angles = np.asarray(block["angles7"], dtype=np.float64)
    m = triangular_angles_parametrization(angles, 5, 3)
    corr = m @ m.T
    for i in range(corr.shape[0]):
        tolerance.tight(float(corr[i, i]), 1.0)


def test_wrong_angle_count_raises() -> None:
    with pytest.raises(LibraryException, match=r"angles\.size"):
        triangular_angles_parametrization(np.zeros(3, dtype=np.float64), 5, 3)


def test_rank_three_vectorial_wrong_size_raises() -> None:
    with pytest.raises(LibraryException, match="exactly 3 values"):
        triangular_angles_parametrization_rank_three_vectorial(
            np.zeros(2, dtype=np.float64), 6
        )


def _cost(case: dict[str, Any], target: Matrix) -> FrobeniusCostFunction:
    return FrobeniusCostFunction(
        target,
        _PARAMETRISATIONS[case["parametrisation"]],
        int(case["matrix_size"]),
        int(case["rank"]),
    )


def test_frobenius_values_match_cpp(cpp: dict[str, Any]) -> None:
    """The residual vector is the strictly-lower triangle of ``B B^T - target``."""
    block = cpp["frobenius_cost_function"]
    target = np.asarray(block[0]["target"], dtype=np.float64)
    for case in block[1:]:
        cf = _cost(case, target)
        actual = cf.values(np.asarray(case["x"], dtype=np.float64))
        expected = [float(e) for e in case["values"]]
        assert actual.shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.tight(float(actual[i]), exp, reason=f"{case['name']}[{i}]")


def test_frobenius_value_match_cpp(cpp: dict[str, Any]) -> None:
    """``value`` matches C++ to TIGHT."""
    block = cpp["frobenius_cost_function"]
    target = np.asarray(block[0]["target"], dtype=np.float64)
    for case in block[1:]:
        cf = _cost(case, target)
        actual = cf.value(np.asarray(case["x"], dtype=np.float64))
        tolerance.tight(actual, float(case["value"]), reason=case["name"])


def test_value_is_the_sum_of_squares_not_the_base_class_rms(cpp: dict[str, Any]) -> None:
    """``value`` overrides ``CostFunction.value``: ``sum(v**2)``, not ``rms(v)``.

    The base class returns ``sqrt(mean(v**2))``. Keeping that default here
    would rescale the objective by ``sqrt(len(v))`` and take a square root,
    which changes both the gradient direction reported by a finite-difference
    optimizer and the point at which its function-value criterion fires.
    """
    block = cpp["frobenius_cost_function"]
    target = np.asarray(block[0]["target"], dtype=np.float64)
    case = block[1]
    cf = _cost(case, target)
    x = np.asarray(case["x"], dtype=np.float64)
    residuals = cf.values(x)
    tolerance.tight(cf.value(x), float(np.dot(residuals, residuals)))
    base_rms = float(np.sqrt(np.sum(residuals * residuals) / residuals.size))
    assert abs(cf.value(x) - base_rms) > 1e-6
