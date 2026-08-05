"""Cross-validate HouseholderTransformation / HouseholderReflection against C++.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``householder_transformation`` and ``householder_reflection`` sections. The
reflection cases are chosen to land in each of the three branches of
``reflectionVector``: the exact-parallel zero return, the fourth-order series
in ``eps``, and the direct ``(a - |a| e) / |a - |a| e|`` formula. The series
branch is the one no library routine will reproduce — it exists precisely to
avoid the cancellation the direct formula suffers near ``a || e``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.householder import (
    HouseholderReflection,
    HouseholderTransformation,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def test_transformation_matrix_matches_cpp(cpp: dict[str, Any]) -> None:
    """``getMatrix()`` is ``I - 2 y y^T`` with ``y = v / |v|``, entry by entry."""
    for case in cpp["householder_transformation"]:
        h = HouseholderTransformation(np.asarray(case["v"], dtype=np.float64))
        expected = np.asarray(case["matrix"], dtype=np.float64)
        assert h.get_matrix().shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(h.get_matrix()[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_transformation_apply_matches_cpp(cpp: dict[str, Any]) -> None:
    """``h(x) == x - 2 (v.x) v`` — note this does **not** normalise ``v``."""
    for case in cpp["householder_transformation"]:
        h = HouseholderTransformation(np.asarray(case["v"], dtype=np.float64))
        actual = h(np.asarray(case["x"], dtype=np.float64))
        expected = [float(e) for e in case["applied"]]
        for i, exp in enumerate(expected):
            tolerance.tight(float(actual[i]), exp, reason=f"{case['name']}[{i}]")


def test_reflection_vector_matches_cpp(cpp: dict[str, Any]) -> None:
    """``reflectionVector`` matches C++ to TIGHT in all three branches."""
    for case in cpp["householder_reflection"]:
        h = HouseholderReflection(np.asarray(case["e"], dtype=np.float64))
        actual = h.reflection_vector(np.asarray(case["a"], dtype=np.float64))
        expected = [float(e) for e in case["reflection_vector"]]
        assert actual.shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.tight(float(actual[i]), exp, reason=f"{case['name']}[{i}]")


def test_reflection_apply_matches_cpp(cpp: dict[str, Any]) -> None:
    """``h(a)`` matches C++ to TIGHT."""
    for case in cpp["householder_reflection"]:
        h = HouseholderReflection(np.asarray(case["e"], dtype=np.float64))
        actual = h(np.asarray(case["a"], dtype=np.float64))
        expected = [float(e) for e in case["applied"]]
        for i, exp in enumerate(expected):
            tolerance.tight(float(actual[i]), exp, reason=f"{case['name']}[{i}]")


def test_parallel_input_returns_the_zero_vector(cpp: dict[str, Any]) -> None:
    """``a`` exactly parallel to ``e`` gives ``eps == 0`` and a zero ``v``."""
    case = next(c for c in cpp["householder_reflection"] if c["name"] == "parallel_3")
    h = HouseholderReflection(np.asarray(case["e"], dtype=np.float64))
    for value in h.reflection_vector(np.asarray(case["a"], dtype=np.float64)):
        tolerance.exact(float(value), 0.0)


def test_reflection_maps_a_onto_the_target_direction(cpp: dict[str, Any]) -> None:
    """``h(a) == |a| e`` — the property the whole class exists for.

    Holds in all three branches, including the anti-projected case
    (``a . e < 0``): the reflection about ``a - |a| e`` always sends ``a`` to
    ``+|a| e``. Asserted at LOOSE because the series branch is an
    approximation rather than an identity — its truncation error is
    ``O(eps**5)``, which at ``eps ~ 7e-5`` (``series_edge_3``) is around
    ``2e-21`` on a unit vector, so LOOSE is generous; the direct branch is
    exact to rounding.
    """
    for case in cpp["householder_reflection"]:
        e = np.asarray(case["e"], dtype=np.float64)
        a = np.asarray(case["a"], dtype=np.float64)
        h = HouseholderReflection(e)
        expected = float(np.linalg.norm(a)) * e
        for i, value in enumerate(h(a)):
            tolerance.loose(float(value), float(expected[i]), reason=case["name"])


def test_zero_vector_raises() -> None:
    h = HouseholderReflection(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    with pytest.raises(LibraryException, match="vector of length zero"):
        h.reflection_vector(np.zeros(3, dtype=np.float64))
