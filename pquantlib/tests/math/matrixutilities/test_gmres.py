"""Cross-validate GMRES against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``gmres`` section. The pinned quantity is the **per-iteration error history**
that ``GMRESResult`` carries, together with the solution and the norm of every
argument handed to the operator and the preconditioner.
``scipy.sparse.linalg.gmres`` returns a solution and an info flag, and its
restart, breakdown and preconditioning conventions differ; the error list is
where that shows up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.gmres import GMRES
from pquantlib.testing import reference_reader, tolerance

from ._krylov_helpers import Recorded, matvec, x0_of


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def _solve(case: dict[str, Any]) -> tuple[Any, Recorded]:
    ops = Recorded(case)
    solver = GMRES(ops.apply_a, int(case["max_iter"]), float(case["rel_tol"]), ops.apply_m)
    b = np.asarray(case["b"], dtype=np.float64)
    restart = int(case["restart"])
    if restart == 0:
        return solver.solve(b, x0_of(case)), ops
    return solver.solve_with_restart(restart, b, x0_of(case)), ops


def test_error_history_length_matches_cpp(cpp: dict[str, Any]) -> None:
    """The history has exactly as many entries as C++ produced.

    The length encodes where the Arnoldi loop stopped — on the relative-
    tolerance test, on the ``h[j+1][j] < QL_EPSILON**2`` breakdown, or on
    ``maxIter`` — and, for the restarted case, how many cycles ran.
    """
    for case in cpp["gmres"]:
        result, _ = _solve(case)
        assert len(result.errors) == len(case["errors"]), case["name"]


def test_error_history_matches_cpp(cpp: dict[str, Any]) -> None:
    """Every entry of the error history matches C++ to TIGHT.

    The entries are ``|z[j+1]| / ||b||`` straight out of the Givens rotations,
    so they are products of ``O(1)`` sines and cosines rather than differences
    of near-equal numbers: the recurrence carries no cancellation of its own
    and TIGHT holds even where the value itself has decayed to ``1e-12``.
    """
    for case in cpp["gmres"]:
        result, _ = _solve(case)
        expected = [float(e) for e in case["errors"]]
        for i, exp in enumerate(expected):
            tolerance.tight(float(result.errors[i]), exp, reason=f"{case['name']}[{i}]")


def test_error_history_is_the_convergence_record(cpp: dict[str, Any]) -> None:
    """The last entry is below ``rel_tol`` — C++ raises otherwise."""
    for case in cpp["gmres"]:
        result, _ = _solve(case)
        assert result.errors[-1] < float(case["rel_tol"]), case["name"]


def test_solution_matches_cpp(cpp: dict[str, Any]) -> None:
    """The solution vector matches C++ to TIGHT."""
    for case in cpp["gmres"]:
        result, _ = _solve(case)
        expected = [float(e) for e in case["x"]]
        assert result.x.shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.tight(float(result.x[i]), exp, reason=f"{case['name']}[{i}]")


def test_operator_call_trace_matches_cpp(cpp: dict[str, Any]) -> None:
    """Every operator and preconditioner call, in order, on the same vector.

    Asserted at LOOSE. The Arnoldi vector at step ``j`` is what is left of
    ``A M^-1 v[j-1]`` after classical Gram-Schmidt against ``v[0..j]``, so its
    norm carries a cancellation amplification of
    ``||A M^-1 v[j-1]|| / h[j+1][j]``. That ratio is the reciprocal of the
    residual decay, which for the ILU-preconditioned 2-D Laplacian reaches
    ``1e4`` by the fifth step — turning the ``O(eps)`` FMA difference into
    ``1e-11`` relative, outside TIGHT by construction. The number of calls and
    their order are exact regardless.
    """
    for case in cpp["gmres"]:
        _, ops = _solve(case)
        expected_a = [float(e) for e in case["a_call_norms"]]
        expected_m = [float(e) for e in case["m_call_norms"]]
        assert len(ops.a_norms) == len(expected_a), case["name"]
        assert len(ops.m_norms) == len(expected_m), case["name"]
        for i, exp in enumerate(expected_a):
            tolerance.loose(ops.a_norms[i], exp, reason=f"{case['name']} A[{i}]")
        for i, exp in enumerate(expected_m):
            tolerance.loose(ops.m_norms[i], exp, reason=f"{case['name']} M[{i}]")


def test_zero_rhs_short_circuits(cpp: dict[str, Any]) -> None:
    """``b == 0`` returns ``b`` with a one-entry ``[0.0]`` history, no A call."""
    case = next(c for c in cpp["gmres"] if c["name"] == "zero_rhs")
    result, ops = _solve(case)
    assert result.errors == [0.0]
    assert ops.a_norms == []
    for value in result.x:
        tolerance.exact(float(value), 0.0)


def test_restart_concatenates_cycles(cpp: dict[str, Any]) -> None:
    """``solve_with_restart`` glues each cycle's history onto the previous one.

    With ``max_iter = 3`` on a 6x6 system no single cycle can converge, so the
    recorded history is strictly longer than one cycle's and its length is a
    multiple-of-cycles structure that a ``restart=`` keyword would not
    reproduce.
    """
    case = next(c for c in cpp["gmres"] if c["name"] == "tridiag_6_restart")
    result, _ = _solve(case)
    assert len(result.errors) == len(case["errors"])
    assert len(result.errors) > int(case["max_iter"]) + 1


def test_residual_actually_solves_the_system(cpp: dict[str, Any]) -> None:
    """``A x == b`` to the requested tolerance — a check on top of the pins."""
    for case in cpp["gmres"]:
        if case["name"] == "zero_rhs":
            continue
        result, _ = _solve(case)
        a = np.asarray(case["matrix"], dtype=np.float64)
        b = np.asarray(case["b"], dtype=np.float64)
        residual = float(np.linalg.norm(b - matvec(a, result.x)))
        assert residual / float(np.linalg.norm(b)) < float(case["rel_tol"]), case["name"]


def test_zero_max_iter_raises() -> None:
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    with pytest.raises(LibraryException, match="maxIter must be greater than zero"):
        GMRES(lambda x: matvec(a, x), 0, 1e-8)


def test_non_convergence_raises() -> None:
    """C++ raises rather than returning an info flag; so does the port."""
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    solver = GMRES(lambda x: matvec(a, x), 1, 1e-30)
    with pytest.raises(LibraryException, match="could not converge"):
        solver.solve(np.array([1.0, 2.0], dtype=np.float64))
