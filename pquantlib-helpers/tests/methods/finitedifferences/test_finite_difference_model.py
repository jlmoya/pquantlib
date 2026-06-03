"""Cross-validation tests for the FD model + schemes (WS3-FD1).

# Retired-API compat layer — see package docstring.

PRIMARY gate: ``cluster/ws3fd1.json`` (``fd_rollback`` block) holds a
Crank-Nicolson rollback of a BSM operator over a payoff vector, produced by the
genuine C++ v1.42.1 ``FiniteDifferenceModel<CrankNicolson<TridiagonalOperator>>``.
This transitively validates MixedScheme.step/setStep, CrankNicolson, and the
operator algebra used to assemble the implicit/explicit halves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib_helpers.methods.finitedifferences.bsm_operator import BSMOperator
from pquantlib_helpers.methods.finitedifferences.crank_nicolson import CrankNicolson
from pquantlib_helpers.methods.finitedifferences.explicit_euler import ExplicitEuler
from pquantlib_helpers.methods.finitedifferences.finite_difference_model import (
    FiniteDifferenceModel,
    StandardFiniteDifferenceModel,
)
from pquantlib_helpers.methods.finitedifferences.mixed_scheme import MixedScheme

_REF: dict[str, Any] = reference_reader.load("cluster/ws3fd1")


def _expected_result() -> list[float]:
    return [float(x) for x in _REF["fd_rollback"]["result"]]


def _payoff() -> np.ndarray:
    # centered ramp payoff max(i-3, 0) on 7 nodes.
    return np.array([max(float(i) - 3.0, 0.0) for i in range(7)])


def test_crank_nicolson_rollback_matches_cpp() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    model = FiniteDifferenceModel(CrankNicolson(op))
    result = model.rollback(_payoff(), 1.0, 0.0, 10)
    for got, want in zip(result, _expected_result(), strict=True):
        tolerance.tight(float(got), want)


def test_standard_model_matches_cpp() -> None:
    # StandardFiniteDifferenceModel wires the same CrankNicolson evolver.
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    model = StandardFiniteDifferenceModel(op)
    result = model.rollback(_payoff(), 1.0, 0.0, 10)
    for got, want in zip(result, _expected_result(), strict=True):
        tolerance.tight(float(got), want)


def test_rollback_from_before_to_is_rejected() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    model = FiniteDifferenceModel(CrankNicolson(op))
    with pytest.raises(LibraryException):
        model.rollback(_payoff(), 0.0, 1.0, 10)


def test_rollback_endpoints_are_preserved() -> None:
    # the BSM operator's zero endpoint rows mean the boundary nodes are
    # unchanged by an interior-only Crank-Nicolson step.
    expected = _expected_result()
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    model = StandardFiniteDifferenceModel(op)
    result = model.rollback(_payoff(), 1.0, 0.0, 10)
    tolerance.tight(float(result[0]), expected[0])
    tolerance.tight(float(result[-1]), expected[-1])


def test_stopping_times_are_sorted_and_deduplicated() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    model = FiniteDifferenceModel(CrankNicolson(op), [0.5, 0.2, 0.5, 0.8])
    assert model.stopping_times() == [0.2, 0.5, 0.8]


# --- scheme weighting (observable behaviour) --------------------------------
#
# A single MixedScheme.step ending at time t over a *time-constant* operator L:
#   theta=0   (explicit): a <- (I - dt*L) a
#   theta=1   (implicit): a <- (I + dt*L)^{-1} a   (solveFor)
#   theta=0.5 (CN)      : a <- (I + 0.5*dt*L)^{-1} (I - 0.5*dt*L) a
# We assert these against operators assembled by hand from L's algebra.


def _state() -> np.ndarray:
    return np.array([1.0, 2.0, 1.5, 0.5, 1.0, 2.5, 3.0])


def test_explicit_euler_step_applies_i_minus_dt_l() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    dt = 0.1
    scheme = ExplicitEuler(op)
    scheme.set_step(dt)
    got = scheme.step(_state(), 0.5)
    ident = op.identity(op.size())
    expected = ident.subtract(op.multiply(dt)).apply_to(_state())
    for g, w in zip(got, expected, strict=True):
        tolerance.tight(float(g), float(w))


def test_implicit_step_solves_i_plus_dt_l() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    dt = 0.1
    scheme = MixedScheme(op, 1.0)
    scheme.set_step(dt)
    got = scheme.step(_state(), 0.5)
    ident = op.identity(op.size())
    expected = ident.add(op.multiply(dt)).solve_for(_state())
    for g, w in zip(got, expected, strict=True):
        tolerance.tight(float(g), float(w))


def test_crank_nicolson_step_combines_both_halves() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    dt = 0.1
    scheme = CrankNicolson(op)
    scheme.set_step(dt)
    got = scheme.step(_state(), 0.5)
    ident = op.identity(op.size())
    intermediate = ident.subtract(op.multiply(0.5 * dt)).apply_to(_state())
    expected = ident.add(op.multiply(0.5 * dt)).solve_for(intermediate)
    for g, w in zip(got, expected, strict=True):
        tolerance.tight(float(g), float(w))
