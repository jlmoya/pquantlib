"""Cross-validate SparseILUPreconditioner against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``sparse_ilu`` section. Both factors are pinned **entry by entry**, dense dumps
included: a residual check on ``L U ~ A`` is exactly what an incomplete
factorisation is *not* supposed to satisfy, so it would tell us nothing. The
level-of-fill sweep is what decides which entries exist at all, and ``lfil``
is varied across the cases to move that frontier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.sparse_ilu_preconditioner import SparseILUPreconditioner
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def _build(case: dict[str, Any]) -> SparseILUPreconditioner:
    return SparseILUPreconditioner(
        np.asarray(case["matrix"], dtype=np.float64), int(case["lfil"])
    )


def test_l_factor_matches_cpp(cpp: dict[str, Any]) -> None:
    """Every entry of ``L`` matches C++ to TIGHT, structural zeros included.

    The zeros carry as much information as the values: they are where the
    level-of-fill rule refused to fill in.
    """
    for case in cpp["sparse_ilu"]:
        ilu = _build(case)
        expected = np.asarray(case["L"], dtype=np.float64)
        assert ilu.l().shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(ilu.l()[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']} L[{i}][{j}]",
                )


def test_u_factor_matches_cpp(cpp: dict[str, Any]) -> None:
    """Every entry of ``U`` matches C++ to TIGHT, structural zeros included."""
    for case in cpp["sparse_ilu"]:
        ilu = _build(case)
        expected = np.asarray(case["U"], dtype=np.float64)
        assert ilu.u().shape == expected.shape, case["name"]
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                tolerance.tight(
                    float(ilu.u()[i, j]),
                    float(expected[i, j]),
                    reason=f"{case['name']} U[{i}][{j}]",
                )


def test_l_is_unit_lower_triangular(cpp: dict[str, Any]) -> None:
    """``L`` has an exact unit diagonal and nothing above it."""
    for case in cpp["sparse_ilu"]:
        low = _build(case).l()
        for i in range(low.shape[0]):
            tolerance.exact(float(low[i, i]), 1.0)
            for j in range(i + 1, low.shape[1]):
                tolerance.exact(float(low[i, j]), 0.0)


def test_u_is_upper_triangular(cpp: dict[str, Any]) -> None:
    """``U`` has nothing below the diagonal."""
    for case in cpp["sparse_ilu"]:
        up = _build(case).u()
        for i in range(up.shape[0]):
            for j in range(i):
                tolerance.exact(float(up[i, j]), 0.0)


def test_apply_matches_cpp(cpp: dict[str, Any]) -> None:
    """``apply(b)`` — forward then backward band-limited substitution — to TIGHT.

    The band-limited solves visit only the offsets populated during the
    factorisation, ``lBands_`` descending and ``uBands_`` ascending, so the
    summation order is fixed and both sides differ only by FMA contraction.
    """
    for case in cpp["sparse_ilu"]:
        ilu = _build(case)
        for k, rhs in enumerate(case["rhs"]):
            actual = ilu.apply(np.asarray(rhs, dtype=np.float64))
            expected = [float(e) for e in case["applied"][k]]
            assert actual.shape[0] == len(expected), case["name"]
            for i, exp in enumerate(expected):
                tolerance.tight(
                    float(actual[i]), exp, reason=f"{case['name']}#{k}[{i}]"
                )


def test_lfil_widens_the_fill_frontier(cpp: dict[str, Any]) -> None:
    """A higher ``lfil`` admits at least as many entries into ``U``.

    For the 2-D Laplacian, ``lfil = 1`` drops the fill that ``lfil = 2``
    keeps, so the two factorisations must differ — which is also why the probe
    records both.
    """
    one = _build(next(c for c in cpp["sparse_ilu"] if c["name"] == "laplacian_3x3"))
    two = _build(next(c for c in cpp["sparse_ilu"] if c["name"] == "laplacian_3x3_lfil2"))
    assert int(np.count_nonzero(two.u())) > int(np.count_nonzero(one.u()))


def test_tridiagonal_ilu_is_the_exact_lu(cpp: dict[str, Any]) -> None:
    """On a tridiagonal matrix ILU has nothing to drop, so ``L U == A``.

    That makes this case a check of the *arithmetic* rather than of the fill
    rule, and it pins the transcription against the closed-form Thomas
    factorisation.
    """
    case = next(c for c in cpp["sparse_ilu"] if c["name"] == "tridiag_6")
    a = np.asarray(case["matrix"], dtype=np.float64)
    ilu = _build(case)
    rebuilt = ilu.l() @ ilu.u()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tolerance.tight(float(rebuilt[i, j]), float(a[i, j]))


def test_diagonal_matrix_factors_trivially(cpp: dict[str, Any]) -> None:
    """A diagonal input gives ``L == I`` and ``U == A``."""
    case = next(c for c in cpp["sparse_ilu"] if c["name"] == "diagonal_4")
    a = np.asarray(case["matrix"], dtype=np.float64)
    ilu = _build(case)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tolerance.exact(float(ilu.l()[i, j]), 1.0 if i == j else 0.0)
            tolerance.exact(float(ilu.u()[i, j]), float(a[i, j]))


def test_non_square_raises() -> None:
    with pytest.raises(LibraryException, match="square matrices"):
        SparseILUPreconditioner(np.zeros((2, 3), dtype=np.float64))
