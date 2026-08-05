"""Cross-validate SVD (Golub-Reinsch, JAMA/TNT) against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``svd`` section. Both singular-vector matrices are pinned entry by entry, signs
included, because sign and shape are precisely where ``numpy.linalg.svd``
diverges: it returns the *full* ``U`` and ``V^T`` with its own sign choice,
while this returns the thin factors and swaps the accessors for a wide input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.matrixutilities.svd import SVD
from pquantlib.testing import reference_reader, tolerance

# The third singular value of this case is ~2e-15, i.e. at the noise floor of a
# 4x3 matrix whose third column is the sum of the other two. Its singular
# *vector* is therefore fixed by rounding, not by the algorithm, and C++ and
# Python disagree about it at ``O(1)``. Everything else about the case — the
# two well-determined columns, the singular values, the rank and the
# pseudo-inverse solve — is still pinned; see ``test_null_singular_vector``.
_NOISE_FLOOR_NULL_VECTOR = "rank_deficient_4x3"


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def _cases(cpp: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cpp["svd"])


def _matrix(case: dict[str, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(case["matrix"], dtype=np.float64)


def test_singular_values_match_cpp(cpp: dict[str, Any]) -> None:
    """Singular values match C++ to TIGHT, descending.

    Both sides run the same Householder bidiagonalisation and the same
    implicit-shift QR sweep; the divergence is FMA contraction inside the
    compiled C++, an ``O(eps)`` relative perturbation on a backward-stable
    factorisation.
    """
    for case in _cases(cpp):
        svd = SVD(_matrix(case))
        expected = [float(e) for e in case["singular_values"]]
        assert svd.singular_values().shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.tight(
                float(svd.singular_values()[i]), exp, reason=f"{case['name']}[{i}]"
            )
        values = [float(v) for v in svd.singular_values()]
        assert values == sorted(values, reverse=True), case["name"]


def test_u_matches_cpp(cpp: dict[str, Any]) -> None:
    """``U()`` matches C++ entry by entry, sign included."""
    for case in _cases(cpp):
        if case["name"] == _NOISE_FLOOR_NULL_VECTOR:
            continue
        svd = SVD(_matrix(case))
        expected = np.asarray(case["U"], dtype=np.float64)
        assert svd.u().shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(svd.u()[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_v_matches_cpp(cpp: dict[str, Any]) -> None:
    """``V()`` matches C++ entry by entry, sign included — every case."""
    for case in _cases(cpp):
        svd = SVD(_matrix(case))
        expected = np.asarray(case["V"], dtype=np.float64)
        assert svd.v().shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(svd.v()[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_null_singular_vector(cpp: dict[str, Any]) -> None:
    """The rank-deficient case: everything but its noise-floor ``U`` column.

    ``s[2] ~ 2e-15`` against ``s[0] ~ 22``, i.e. a ratio of ``1e-16`` — one
    unit in the last place of the leading singular value. The two
    well-determined columns of ``U`` are pinned to TIGHT; the third is only
    required to be a unit vector orthogonal to them, which is all the
    algorithm actually determines.
    """
    case = next(c for c in _cases(cpp) if c["name"] == _NOISE_FLOOR_NULL_VECTOR)
    svd = SVD(_matrix(case))
    expected = np.asarray(case["U"], dtype=np.float64)
    for j in range(svd.rank()):
        for i in range(expected.shape[0]):
            tolerance.tight(float(svd.u()[i, j]), float(expected[i, j]), reason=f"[{i}][{j}]")
    null_column = svd.u()[:, 2]
    tolerance.loose(float(np.dot(null_column, null_column)), 1.0)
    for j in range(svd.rank()):
        tolerance.loose(float(np.dot(null_column, svd.u()[:, j])), 0.0)


def test_s_matrix_matches_cpp(cpp: dict[str, Any]) -> None:
    """``S()`` is the diagonal ``n x n`` matrix of the singular values."""
    for case in _cases(cpp):
        svd = SVD(_matrix(case))
        expected = np.asarray(case["S"], dtype=np.float64)
        assert svd.s().shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(svd.s()[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']}[{i}][{j}]",
                )


def test_norm2_cond_rank_match_cpp(cpp: dict[str, Any]) -> None:
    """``norm2``, ``cond`` and ``rank`` match C++.

    ``rank`` is an integer and must agree exactly — it drives how many
    reciprocals ``solve_for`` builds. ``cond`` is only a reproducible quantity
    when the matrix is numerically full rank: otherwise it is
    ``s[0] / (noise)``, and the noise is one unit in the last place of ``s[0]``,
    so the quotient is meaningful only as an order of magnitude. C++ records
    ``null`` where the quotient is outright infinite.
    """
    for case in _cases(cpp):
        svd = SVD(_matrix(case))
        tolerance.tight(svd.norm2(), float(case["norm2"]), reason=case["name"])
        assert svd.rank() == int(case["rank"]), case["name"]
        full_rank = svd.rank() == svd.singular_values().shape[0]
        if case["cond"] is None:
            assert not np.isfinite(svd.cond()), case["name"]
        elif full_rank:
            tolerance.loose(svd.cond(), float(case["cond"]), reason=case["name"])
        else:
            assert svd.cond() > 1e14, case["name"]
            assert float(case["cond"]) > 1e14, case["name"]


def test_solve_for_matches_cpp(cpp: dict[str, Any]) -> None:
    """``solveFor`` matches C++ to LOOSE.

    The pseudo-inverse is ``V W U^T`` with ``W = diag(1/s_i)`` over the
    numerical rank, so the componentwise error is amplified by the condition
    number ``s[0]/s[rank-1]``. The Hilbert(4) case has ``cond ~ 1.6e4``, which
    turns the ``O(eps)`` factor differences into ``O(1e-12)`` relative on the
    solution — outside TIGHT by construction, and the amplification is the
    reason, not an empirical fit.
    """
    for case in _cases(cpp):
        svd = SVD(_matrix(case))
        actual = svd.solve_for(np.asarray(case["b"], dtype=np.float64))
        expected = [float(e) for e in case["solve_for"]]
        assert actual.shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.loose(float(actual[i]), exp, reason=f"{case['name']}[{i}]")


def test_thin_shapes(cpp: dict[str, Any]) -> None:
    """The decomposition is thin, and the accessors swap for a wide input.

    For ``M`` of shape ``(r, c)`` with ``k = min(r, c)``: ``U`` is ``(r, k)``,
    ``V`` is ``(c, k)``. ``numpy.linalg.svd`` would hand back a square ``(r, r)``
    ``U`` unless asked otherwise.
    """
    for case in _cases(cpp):
        m = _matrix(case)
        svd = SVD(m)
        k = min(m.shape[0], m.shape[1])
        assert svd.u().shape == (m.shape[0], k), case["name"]
        assert svd.v().shape == (m.shape[1], k), case["name"]


def test_reconstruction(cpp: dict[str, Any]) -> None:
    """``U S V^T == M`` — a sanity check *on top of* the pinned factors."""
    for case in _cases(cpp):
        m = _matrix(case)
        svd = SVD(m)
        rebuilt = svd.u() @ svd.s() @ svd.v().T
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                tolerance.loose(float(rebuilt[i, j]), float(m[i, j]))
