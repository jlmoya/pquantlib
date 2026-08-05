"""Cross-validate BiCGstab against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/matrixutilities.json`` —
``bicgstab`` section. Each case pins the iteration count, the final relative
error, the whole solution vector **and** the norm of every argument the solver
handed to its operator and preconditioner — the trajectory, not just the
endpoint. ``scipy.sparse.linalg.bicgstab`` would reach a comparable solution
through a different sequence of operator applications and a different stopping
rule; the trace is what makes that visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrixutilities.bicgstab import BiCGstab
from pquantlib.testing import reference_reader, tolerance

from ._krylov_helpers import Recorded, matvec, x0_of


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/matrixutilities")


def _solve(case: dict[str, Any]) -> tuple[Any, Recorded]:
    ops = Recorded(case)
    solver = BiCGstab(ops.apply_a, int(case["max_iter"]), float(case["rel_tol"]), ops.apply_m)
    result = solver.solve(np.asarray(case["b"], dtype=np.float64), x0_of(case))
    return result, ops


def test_iterations_match_cpp_exactly(cpp: dict[str, Any]) -> None:
    """The iteration count is an integer and must agree exactly.

    It moves the moment the convergence test, the breakdown test or the
    ``||s|| < relTol ||b||`` early exit differ from C++'s.
    """
    for case in cpp["bicgstab"]:
        result, _ = _solve(case)
        assert result.iterations == int(case["iterations"]), case["name"]


def test_solution_matches_cpp(cpp: dict[str, Any]) -> None:
    """The solution vector matches C++ to TIGHT.

    Both sides apply the same operator with the same summation order (see
    ``_krylov_helpers.matvec``) and run the same recurrence, so the only
    divergence is FMA contraction inside the compiled C++ dot products,
    amplified by the condition number of the (well-conditioned) test systems.
    """
    for case in cpp["bicgstab"]:
        result, _ = _solve(case)
        expected = [float(e) for e in case["x"]]
        assert result.x.shape[0] == len(expected), case["name"]
        for i, exp in enumerate(expected):
            tolerance.tight(float(result.x[i]), exp, reason=f"{case['name']}[{i}]")


def test_final_error_matches_cpp(cpp: dict[str, Any]) -> None:
    """The reported relative residual matches C++ to TIGHT.

    TIGHT passes here on its *absolute* arm, and that is the honest reading:
    ``error = ||r|| / ||b||`` is carried by a recurrence whose terms are
    ``O(1)`` while ``||r||`` decays to ``1e-10`` or below, so the quantity is
    cancellation-dominated and its **relative** accuracy is only
    ``eps * ||b|| / ||r||``. What is reproducible — and what the caller acts
    on — is its absolute size, which agrees to well inside ``1e-14``.
    """
    for case in cpp["bicgstab"]:
        result, _ = _solve(case)
        tolerance.tight(result.error, float(case["error"]), reason=case["name"])
        assert result.error < float(case["rel_tol"]) or result.error == 0.0, case["name"]


def test_operator_call_trace_matches_cpp(cpp: dict[str, Any]) -> None:
    """Every operator and preconditioner call is made, in order, on the same vector.

    Asserted at LOOSE: the later Krylov directions are formed by differencing
    nearly-parallel vectors, so their norms carry a cancellation amplification
    of ``||A x|| / ||r_k||``, which grows as the residual decays. The
    *sequence* — how many calls, in what order, with what magnitudes — is the
    discriminating content and is preserved far beyond ``1e-8``.
    """
    for case in cpp["bicgstab"]:
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
    """``b == 0`` returns ``b`` untouched with zero iterations and no A call."""
    case = next(c for c in cpp["bicgstab"] if c["name"] == "zero_rhs")
    result, ops = _solve(case)
    assert result.iterations == 0
    tolerance.exact(result.error, 0.0)
    assert ops.a_norms == []
    for value in result.x:
        tolerance.exact(float(value), 0.0)


def test_residual_actually_solves_the_system(cpp: dict[str, Any]) -> None:
    """``A x == b`` to the requested tolerance — a check on top of the pins."""
    for case in cpp["bicgstab"]:
        if case["name"] == "zero_rhs":
            continue
        result, _ = _solve(case)
        a = np.asarray(case["matrix"], dtype=np.float64)
        b = np.asarray(case["b"], dtype=np.float64)
        residual = float(np.linalg.norm(b - matvec(a, result.x)))
        assert residual / float(np.linalg.norm(b)) < float(case["rel_tol"]), case["name"]


def test_non_convergence_raises() -> None:
    """C++ raises rather than returning an info flag; so does the port."""
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    solver = BiCGstab(lambda x: matvec(a, x), 1, 1e-30)
    with pytest.raises(LibraryException, match=r"could not converge|max number of iterations"):
        solver.solve(np.array([1.0, 2.0], dtype=np.float64))
