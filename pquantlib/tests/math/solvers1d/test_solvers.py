"""Cross-validate 1-D solvers against the L1-C cluster C++ probe.

Probe source: migration-harness/cpp/probes/cluster_c/probe.cpp
Reference:    migration-harness/references/cluster/c.json

Test function: f(x) = (x - 2)(x - 5), roots at x=2 and x=5.
Bracket [3, 7] surrounds the root at x=5; guess = 4; accuracy = 1e-12.
Unbracketed solvers (Newton, Halley) start from guess=4 with step=0.1.

Note: the Secant solver, given the smaller |f| at xMin=3 and the bracket
shape, drifts toward the *other* root at x=2 — this is faithful to C++
behavior and is asserted directly against the probe value.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.solvers1d.bisection import Bisection
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.math.solvers1d.false_position import FalsePosition
from pquantlib.math.solvers1d.finite_difference_newton_safe import FiniteDifferenceNewtonSafe
from pquantlib.math.solvers1d.halley import Halley
from pquantlib.math.solvers1d.newton import Newton
from pquantlib.math.solvers1d.newton_safe import NewtonSafe
from pquantlib.math.solvers1d.ridder import Ridder
from pquantlib.math.solvers1d.secant import Secant
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/c")


class _Quadratic:
    """Test function f(x) = (x - 2)(x - 5) with first + second derivatives."""

    def __call__(self, x: float) -> float:
        return (x - 2.0) * (x - 5.0)

    def derivative(self, x: float) -> float:
        return 2.0 * x - 7.0

    def second_derivative(self, x: float) -> float:
        # x is unused; the Halley protocol requires this method to take an x.
        del x
        return 2.0


_ACCURACY = 1e-12
_GUESS = 4.0
_X_MIN = 3.0
_X_MAX = 7.0


# --- bracketed solvers --------------------------------------------------


def test_bisection_finds_root(cpp: dict[str, Any]) -> None:
    root = Bisection().solve(_Quadratic(), _ACCURACY, _GUESS, _X_MIN, _X_MAX)
    tolerance.tight(root, float(cpp["solvers"]["bisection"]))


def test_brent_finds_root(cpp: dict[str, Any]) -> None:
    root = Brent().solve(_Quadratic(), _ACCURACY, _GUESS, _X_MIN, _X_MAX)
    tolerance.tight(root, float(cpp["solvers"]["brent"]))


def test_false_position_finds_root(cpp: dict[str, Any]) -> None:
    root = FalsePosition().solve(_Quadratic(), _ACCURACY, _GUESS, _X_MIN, _X_MAX)
    tolerance.tight(root, float(cpp["solvers"]["false_position"]))


def test_ridder_finds_root(cpp: dict[str, Any]) -> None:
    root = Ridder().solve(_Quadratic(), _ACCURACY, _GUESS, _X_MIN, _X_MAX)
    tolerance.tight(root, float(cpp["solvers"]["ridder"]))


def test_secant_finds_other_root(cpp: dict[str, Any]) -> None:
    # Secant drifts toward x=2 from bracket [3,7] — see module docstring.
    root = Secant().solve(_Quadratic(), _ACCURACY, _GUESS, _X_MIN, _X_MAX)
    tolerance.tight(root, float(cpp["solvers"]["secant"]))


def test_newton_safe_finds_root(cpp: dict[str, Any]) -> None:
    root = NewtonSafe().solve(_Quadratic(), _ACCURACY, _GUESS, _X_MIN, _X_MAX)
    tolerance.tight(root, float(cpp["solvers"]["newton_safe"]))


def test_finite_difference_newton_safe_finds_root(cpp: dict[str, Any]) -> None:
    root = FiniteDifferenceNewtonSafe().solve(_Quadratic(), _ACCURACY, _GUESS, _X_MIN, _X_MAX)
    tolerance.tight(root, float(cpp["solvers"]["fd_newton_safe"]))


# --- unbracketed (Newton-style) solvers --------------------------------


def test_newton_finds_root(cpp: dict[str, Any]) -> None:
    root = Newton().solve(_Quadratic(), _ACCURACY, _GUESS, 0.1)
    tolerance.tight(root, float(cpp["solvers"]["newton"]))


def test_halley_finds_root(cpp: dict[str, Any]) -> None:
    root = Halley().solve(_Quadratic(), _ACCURACY, _GUESS, 0.1)
    tolerance.tight(root, float(cpp["solvers"]["halley"]))


# --- closed-form sanity checks -----------------------------------------


def test_bisection_closed_form_root() -> None:
    """Bisection on (x-2)(x-5) over [3,7] should land on x=5 exactly enough."""
    root = Bisection().solve(_Quadratic(), 1e-12, 4.0, 3.0, 7.0)
    assert abs(root - 5.0) < 1e-10


def test_set_max_evaluations_caps_iterations() -> None:
    """Capping max_evaluations forces failure when the bracket cannot collapse fast enough."""

    # Use cos(x) = 0 on [0, 3] — root ~ 1.5707963. f(root) is not exactly 0.
    def cos_minus_zero(x: float) -> float:
        return math.cos(x)

    solver = Bisection()
    solver.set_max_evaluations(2)
    with pytest.raises(LibraryException):
        # accuracy super-tight, only 2 iterations allowed.
        solver.solve(cos_minus_zero, 1e-30, 1.0, 0.0, 3.0)


def test_invalid_accuracy_raises() -> None:
    """``accuracy <= 0`` must be rejected."""
    with pytest.raises(LibraryException):
        Bisection().solve(_Quadratic(), 0.0, 4.0, 3.0, 7.0)


def test_invalid_bracket_raises() -> None:
    """``xMin >= xMax`` must be rejected."""
    with pytest.raises(LibraryException):
        Bisection().solve(_Quadratic(), 1e-12, 4.0, 7.0, 3.0)


def test_root_not_bracketed_raises() -> None:
    """Bracket where f(xMin) and f(xMax) have the same sign must be rejected."""

    def f(x: float) -> float:
        return x * x + 1.0  # always > 0

    with pytest.raises(LibraryException):
        Bisection().solve(f, 1e-12, 0.0, -1.0, 1.0)


def test_lower_bound_enforced() -> None:
    """A bracket below the enforced lower bound must be rejected."""
    solver = Bisection()
    solver.set_lower_bound(0.0)
    with pytest.raises(LibraryException):
        solver.solve(_Quadratic(), 1e-12, 4.0, -1.0, 7.0)


def test_upper_bound_enforced() -> None:
    """A bracket above the enforced upper bound must be rejected."""
    solver = Bisection()
    solver.set_upper_bound(6.0)
    with pytest.raises(LibraryException):
        solver.solve(_Quadratic(), 1e-12, 4.0, 3.0, 10.0)


def test_newton_without_derivative_raises() -> None:
    """Newton must reject integrands lacking ``derivative()``."""

    def f(x: float) -> float:
        return (x - 2.0) * (x - 5.0)

    with pytest.raises(LibraryException):
        Newton().solve(f, 1e-12, 4.0, 0.1)


def test_halley_without_second_derivative_raises() -> None:
    """Halley must reject integrands lacking ``second_derivative()``."""

    class OnlyFirst:
        def __call__(self, x: float) -> float:
            return (x - 2.0) * (x - 5.0)

        def derivative(self, x: float) -> float:
            return 2.0 * x - 7.0

    with pytest.raises(LibraryException):
        Halley().solve(OnlyFirst(), 1e-12, 4.0, 0.1)


def test_unbracketed_newton_finds_root_close_to_guess() -> None:
    """Newton's auto-bracket should successfully find x=5 when starting near it."""
    root = Newton().solve(_Quadratic(), 1e-12, 4.0, 0.1)
    assert abs(root - 5.0) < 1e-10
