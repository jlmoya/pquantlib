"""Cross-validate ``LeastSquareFunction`` / ``NonLinearLeastSquare`` against C++.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block G. The problem is an 8-point exponential fit ``a*exp(b*t)``.

The pointwise assertions are EXACT: ``value``, ``values`` and ``gradient``
are a fixed sequence of arithmetic on the target/model vectors with no
iteration in between. They also pin the two shapes that are easy to get
wrong when reading the header rather than the source:

- ``values`` returns ``diff * diff`` — the ELEMENTWISE square, not the
  residual. A least-squares method fed this object therefore minimizes the
  sum of FOURTH powers.
- ``value`` returns ``diff . diff`` — the sum of squares, not its square
  root and not its mean.

The solver assertions are EXACT too, because the default optimizer is
``ConjugateGradient`` over an ANALYTIC gradient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.optimization.conjugate_gradient import ConjugateGradient
from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.least_square import (
    LeastSquareFunction,
    LeastSquareProblem,
    NonLinearLeastSquare,
)
from pquantlib.math.optimization.simplex import Simplex
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import EXP_T, EXP_Y, ExpFitLSP


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def test_least_square_function_value_is_the_sum_of_squares(
    cpp: dict[str, Any],
) -> None:
    lsf = LeastSquareFunction(ExpFitLSP())
    x = np.array(cpp["lsf_x"], dtype=np.float64)
    tolerance.exact(lsf.value(x), float(cpp["lsf_value"]))


def test_least_square_function_values_are_elementwise_squares(
    cpp: dict[str, Any],
) -> None:
    """``values`` is ``diff*diff``, NOT ``diff`` — pinned against C++ and re-derived."""
    lsf = LeastSquareFunction(ExpFitLSP())
    x = np.array(cpp["lsf_x"], dtype=np.float64)
    got = lsf.values(x)
    for a, b in zip(got, cpp["lsf_values"], strict=True):
        tolerance.exact(float(a), float(b))
    # Independent re-derivation: every entry is non-negative and is the square
    # of the corresponding residual.
    diff = EXP_Y - x[0] * np.exp(x[1] * EXP_T)
    for a, b in zip(got, diff * diff, strict=True):
        tolerance.exact(float(a), float(b))


def test_least_square_function_gradient(cpp: dict[str, Any]) -> None:
    lsf = LeastSquareFunction(ExpFitLSP())
    x = np.array(cpp["lsf_x"], dtype=np.float64)
    grad = np.zeros(2, dtype=np.float64)
    lsf.gradient(grad, x)
    for a, b in zip(grad, cpp["lsf_gradient"], strict=True):
        tolerance.exact(float(a), float(b))


def test_least_square_function_value_and_gradient(cpp: dict[str, Any]) -> None:
    """One ``target_value_and_gradient`` call must feed both outputs."""
    lsf = LeastSquareFunction(ExpFitLSP())
    x = np.array(cpp["lsf_x"], dtype=np.float64)
    grad = np.zeros(2, dtype=np.float64)
    value = lsf.value_and_gradient(grad, x)
    tolerance.exact(value, float(cpp["lsf_value_and_gradient_value"]))
    for a, b in zip(grad, cpp["lsf_value_and_gradient_grad"], strict=True):
        tolerance.exact(float(a), float(b))


@pytest.mark.parametrize(
    ("tag", "accuracy", "maxiter", "use_simplex"),
    [
        ("default", None, None, False),
        ("tight", 1e-8, 200, False),
        ("simplex", 1e-8, 200, True),
    ],
)
def test_non_linear_least_square_matches_cpp(
    tag: str,
    accuracy: float | None,
    maxiter: int | None,
    use_simplex: bool,
    cpp: dict[str, Any],
) -> None:
    if accuracy is None or maxiter is None:
        nlls = NonLinearLeastSquare(NoConstraint())
    elif use_simplex:
        nlls = NonLinearLeastSquare(
            NoConstraint(), accuracy, maxiter, Simplex(lambda_=0.1)
        )
    else:
        nlls = NonLinearLeastSquare(NoConstraint(), accuracy, maxiter)
    nlls.set_initial_value(np.array([1.0, -1.0], dtype=np.float64))
    results = nlls.perform(ExpFitLSP())

    for got, expected in zip(results, cpp[f"nlls_{tag}_x"], strict=True):
        tolerance.exact(float(got), float(expected))
    tolerance.exact(nlls.residual_norm, float(cpp[f"nlls_{tag}_resnorm"]))
    tolerance.exact(nlls.last_value, float(cpp[f"nlls_{tag}_lastvalue"]))
    assert nlls.exit_flag == cpp[f"nlls_{tag}_exitflag"]


def test_non_linear_least_square_defaults() -> None:
    """Accuracy 1e-4, maxiter 100, ConjugateGradient — asserted behaviourally.

    # C++ parity: leastsquare.hpp:100-102 and leastsquare.cpp:79-85.

    The three defaults have no accessors in C++ either, so they are pinned by
    running the default constructor against an explicit one built from the
    documented values and requiring bit-identical results.
    """
    default = NonLinearLeastSquare(NoConstraint())
    assert default.exit_flag == -1
    explicit = NonLinearLeastSquare(NoConstraint(), 1e-4, 100, ConjugateGradient())
    for solver in (default, explicit):
        solver.set_initial_value(np.array([1.0, -1.0], dtype=np.float64))
        solver.perform(ExpFitLSP())
    for a, b in zip(default.results, explicit.results, strict=True):
        tolerance.exact(float(a), float(b))
    tolerance.exact(default.residual_norm, explicit.residual_norm)
    assert default.exit_flag == explicit.exit_flag
    assert default.iterations_number == 0


def test_exit_flag_is_an_end_criteria_type(cpp: dict[str, Any]) -> None:
    """The C++ field is an ``Integer`` assigned straight from ``minimize``."""
    assert Type(cpp["nlls_default_exitflag"]) in set(Type)


def test_results_are_stored_and_returned(cpp: dict[str, Any]) -> None:
    nlls = NonLinearLeastSquare(NoConstraint(), 1e-8, 200)
    nlls.set_initial_value(np.array([1.0, -1.0], dtype=np.float64))
    returned = nlls.perform(ExpFitLSP())
    for a, b in zip(returned, nlls.results, strict=True):
        tolerance.exact(float(a), float(b))


def test_least_square_problem_is_abstract() -> None:
    with pytest.raises(TypeError):
        LeastSquareProblem()  # type: ignore[abstract]
