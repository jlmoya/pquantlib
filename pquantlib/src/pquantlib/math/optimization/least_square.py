"""Least-square problem abstraction and its non-linear solver.

# C++ parity: ql/math/optimization/leastsquare.{hpp,cpp} (v1.43).

A ``LeastSquareProblem`` supplies, for a parameter vector ``x``, both a
target vector ``b`` and the model values ``phi(x)`` (plus, optionally,
the Jacobian of ``phi``). ``LeastSquareFunction`` turns that into a
``CostFunction``, and ``NonLinearLeastSquare`` drives an
``OptimizationMethod`` (conjugate gradient by default) over it.

The one thing to get exactly right — and the reason this is not a thin
shell around ``scipy.optimize.least_squares`` — is what
``LeastSquareFunction`` reports:

- ``value(x)``   == ``diff . diff``           (the SUM of squares)
- ``values(x)``  == ``diff * diff``           (the ELEMENTWISE square,
                                               not the residual itself)
- ``gradient``   == ``-2 * J^T diff``

where ``diff = target - phi(x)``. So a gradient-based method
(``ConjugateGradient``) minimizes the sum of squares, while a
least-squares method fed the same object (``LevenbergMarquardt``, which
reads ``values``) minimizes the sum of the FOURTH powers. That asymmetry
is in v1.43 and is reproduced here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.conjugate_gradient import ConjugateGradient
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.line_search import dot_product
from pquantlib.math.optimization.problem import Problem

if TYPE_CHECKING:
    from pquantlib.math.optimization.constraint import Constraint
    from pquantlib.math.optimization.optimization_method import OptimizationMethod


def _fill_neg_two_jt_diff(
    grad: npt.NDArray[np.float64],
    grad_fct2fit: npt.NDArray[np.float64],
    diff: npt.NDArray[np.float64],
) -> None:
    """Write ``-2 * J^T diff`` into ``grad``, column by column.

    # C++ parity: ``grad_f = -2.0*(transpose(grad_fct2fit)*diff)``
    # (leastsquare.cpp:60 and :74). QuantLib's ``Matrix * Array`` is a
    # per-row ``std::inner_product``, i.e. a strictly sequential
    # accumulation, so this uses :func:`dot_product` rather than a BLAS
    # ``matvec`` whose summation order differs.
    """
    for j in range(grad.size):
        grad[j] = -2.0 * dot_product(grad_fct2fit[:, j], diff)


class LeastSquareProblem(ABC):
    """Target vector + model values (+ their derivatives) at a parameter vector.

    # C++ parity: ``class LeastSquareProblem`` in
    # ql/math/optimization/leastsquare.hpp:38-54 (v1.43).

    ``target`` and ``fct2fit`` are C++ out-parameters (``Array&``); the
    Python port keeps them as pre-allocated arrays the implementation
    fills in place, so a subclass reads as a direct transcription of its
    C++ counterpart.
    """

    @abstractmethod
    def size(self) -> int:
        """Length of the target vector."""
        ...

    @abstractmethod
    def target_and_value(
        self,
        x: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        fct2fit: npt.NDArray[np.float64],
    ) -> None:
        """Fill ``target`` with the data and ``fct2fit`` with the model at ``x``."""
        ...

    @abstractmethod
    def target_value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
        grad_fct2fit: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        fct2fit: npt.NDArray[np.float64],
    ) -> None:
        """Also fill ``grad_fct2fit[i][j]`` with d fct2fit_i / d x_j."""
        ...


class LeastSquareFunction(CostFunction):
    """``CostFunction`` view of a :class:`LeastSquareProblem`.

    # C++ parity: ``class LeastSquareFunction`` in
    # ql/math/optimization/leastsquare.{hpp,cpp} (v1.43).
    """

    __slots__ = ("_lsp",)

    def __init__(self, lsp: LeastSquareProblem) -> None:
        # C++ parity: leastsquare.hpp:63 — stored by reference.
        self._lsp: LeastSquareProblem = lsp

    def value(self, x: npt.NDArray[np.float64]) -> float:
        """Sum of squared residuals ``diff . diff``.

        # C++ parity: leastsquare.cpp:28-37.
        """
        n = self._lsp.size()
        target = np.zeros(n, dtype=np.float64)
        fct2fit = np.zeros(n, dtype=np.float64)
        self._lsp.target_and_value(x, target, fct2fit)
        diff = target - fct2fit
        return dot_product(diff, diff)

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ELEMENTWISE squared residuals ``diff * diff`` — not ``diff``.

        # C++ parity: leastsquare.cpp:39-47.
        """
        n = self._lsp.size()
        target = np.zeros(n, dtype=np.float64)
        fct2fit = np.zeros(n, dtype=np.float64)
        self._lsp.target_and_value(x, target, fct2fit)
        diff = target - fct2fit
        return diff * diff

    def gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> None:
        """Analytic gradient ``-2 * J^T diff``, into ``grad``.

        # C++ parity: leastsquare.cpp:49-61.
        """
        n = self._lsp.size()
        target = np.zeros(n, dtype=np.float64)
        fct2fit = np.zeros(n, dtype=np.float64)
        grad_fct2fit = np.zeros((n, x.size), dtype=np.float64)
        self._lsp.target_value_and_gradient(x, grad_fct2fit, target, fct2fit)
        diff = target - fct2fit
        _fill_neg_two_jt_diff(grad, grad_fct2fit, diff)

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        """Gradient into ``grad`` and ``diff . diff`` returned, in ONE pass.

        # C++ parity: leastsquare.cpp:63-77 — a single
        # ``targetValueAndGradient`` call, not ``gradient`` then ``value``.
        """
        n = self._lsp.size()
        target = np.zeros(n, dtype=np.float64)
        fct2fit = np.zeros(n, dtype=np.float64)
        grad_fct2fit = np.zeros((n, x.size), dtype=np.float64)
        self._lsp.target_value_and_gradient(x, grad_fct2fit, target, fct2fit)
        diff = target - fct2fit
        _fill_neg_two_jt_diff(grad, grad_fct2fit, diff)
        return dot_product(diff, diff)


class NonLinearLeastSquare:
    """Non-linear least-square solver over a :class:`LeastSquareProblem`.

    # C++ parity: ``class NonLinearLeastSquare`` in
    # ql/math/optimization/leastsquare.{hpp,cpp} (v1.43).

    Minimizes ``r(x) = |f(x)|^2`` with a supplied ``OptimizationMethod``
    (default: :class:`~pquantlib.math.optimization.conjugate_gradient.ConjugateGradient`).
    The end criteria are built internally as
    ``EndCriteria(maxiter, min(maxiter//2, 100), acc, acc, acc)``
    (leastsquare.cpp:103-105), so ``accuracy`` sets all three epsilons.
    """

    __slots__ = (
        "_accuracy",
        "_best_accuracy",
        "_c",
        "_exit_flag",
        "_initial_value",
        "_max_iterations",
        "_nb_iterations",
        "_om",
        "_resnorm",
        "_results",
    )

    def __init__(
        self,
        c: Constraint,
        accuracy: float = 1e-4,
        maxiter: int = 100,
        om: OptimizationMethod | None = None,
    ) -> None:
        # C++ parity: leastsquare.cpp:79-91 — two constructors, the first
        # of which defaults the method to a fresh ConjugateGradient.
        self._c: Constraint = c
        self._accuracy: float = accuracy
        self._max_iterations: int = maxiter
        self._om: OptimizationMethod = ConjugateGradient() if om is None else om
        self._exit_flag: int = -1
        self._results: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._initial_value: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._resnorm: float = 0.0
        self._best_accuracy: float = 0.0
        self._nb_iterations: int = 0

    def set_initial_value(self, initial_value: npt.NDArray[np.float64]) -> None:
        """Set the starting parameter vector. # C++ parity: leastsquare.hpp:114-116."""
        self._initial_value = initial_value.astype(np.float64, copy=True)

    @property
    def results(self) -> npt.NDArray[np.float64]:
        """Solution vector. # C++ parity: leastsquare.hpp:119."""
        return self._results

    @property
    def residual_norm(self) -> float:
        """Least-square residual norm. # C++ parity: leastsquare.hpp:122."""
        return self._resnorm

    @property
    def last_value(self) -> float:
        """Last function value. # C++ parity: leastsquare.hpp:125."""
        return self._best_accuracy

    @property
    def exit_flag(self) -> int:
        """The ``EndCriteria.Type`` the optimizer returned, as an int.

        # C++ parity: leastsquare.hpp:128 — ``Integer exitFlag()``; the
        # field is assigned straight from ``om_->minimize(...)``, so it is
        # an ``EndCriteria::Type`` widened to an int, and -1 before the
        # first ``perform``.
        """
        return self._exit_flag

    @property
    def iterations_number(self) -> int:
        """Iterations performed.

        # C++ parity: leastsquare.hpp:131 — the assignment that would fill
        # this is COMMENTED OUT in leastsquare.cpp:109, so C++ returns an
        # uninitialised value. pquantlib reports 0 rather than garbage.
        """
        return self._nb_iterations

    def perform(self, ls_problem: LeastSquareProblem) -> npt.NDArray[np.float64]:
        """Solve ``ls_problem`` and return (and store) the solution vector.

        # C++ parity: leastsquare.cpp:93-116.
        """
        eps = self._accuracy

        # Wrap the least-square problem in an optimization function.
        lsf = LeastSquareFunction(ls_problem)

        # Define the optimization problem.
        p = Problem(lsf, self._c, self._initial_value)

        # Minimize.
        ec = EndCriteria(
            self._max_iterations,
            min(self._max_iterations // 2, 100),
            eps,
            eps,
            eps,
        )
        self._exit_flag = int(self._om.minimize(p, ec))

        # Summarize the results of the minimization.
        self._results = p.current_value.astype(np.float64, copy=True)
        self._resnorm = p.function_value
        self._best_accuracy = p.function_value

        return self._results
