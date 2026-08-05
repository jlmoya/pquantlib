"""Line-search based optimization method (base of SteepestDescent/CG/BFGS).

# C++ parity: ql/math/optimization/linesearchbasedmethod.{hpp,cpp} (v1.43).

The whole of ``SteepestDescent``, ``ConjugateGradient`` and ``BFGS`` is
this one ``minimize`` loop plus a single overridden
``get_updated_direction``. Transcribing the loop is therefore the entire
content of those three classes, and the details below are the ones a
generic descent loop would get wrong:

- **The line search owns the objective.** ``minimize`` never evaluates
  ``f`` itself after the initial ``valueAndGradient``; it reads
  ``last_function_value`` / ``last_gradient_norm2`` back out of the line
  search. So the evaluation counters are entirely the line search's.
- **The convergence test is on the RELATIVE function change**, in the
  *Numerical Recipes* form ``2|f_new - f_old| / (|f_new| + |f_old| + eps)``
  compared against ``function_epsilon`` — not on ``|x|``, not on the
  gradient.
- **The published minimum lags one iteration.** On the converging exit
  the method returns WITHOUT calling ``set_current_value``, so
  ``Problem.current_value`` still holds the iterate from the previous
  round while ``Problem.function_value`` already holds the new one. That
  is v1.43 behaviour (linesearchbasedmethod.cpp:98-104), not an
  oversight in this port.
- **The end-criteria type is manufactured.** ``checkStationaryFunctionValue``
  is called with ``(0.0, 0.0)`` and with the counter SEEDED to
  ``max_stationary_state``, which makes it fire unconditionally and yields
  ``StationaryFunctionValue``; a following ``checkMaxIterations`` may
  overwrite it with ``MaxIterations``.
- **The same EndCriteria caps the inner backtracking loop.** It is handed
  straight to the line search, whose own ``checkMaxIterations`` uses the
  backtracking counter. A small ``max_iterations`` therefore starves the
  line search into ``succeed == False`` as well.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.optimization.armijo import ArmijoLineSearch
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.line_search import dot_product
from pquantlib.math.optimization.optimization_method import OptimizationMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.line_search import LineSearch
    from pquantlib.math.optimization.problem import Problem


class LineSearchBasedMethod(OptimizationMethod):
    """Descent method driven by a pluggable line search.

    # C++ parity: ``class LineSearchBasedMethod`` in
    # ql/math/optimization/linesearchbasedmethod.{hpp,cpp} (v1.43).

    A ``None`` line search selects a default-constructed
    ``ArmijoLineSearch``, matching linesearchbasedmethod.cpp:29-33.
    """

    __slots__ = ("_line_search",)

    def __init__(self, line_search: LineSearch | None = None) -> None:
        self._line_search: LineSearch = (
            ArmijoLineSearch() if line_search is None else line_search
        )

    @property
    def line_search(self) -> LineSearch:
        """The line search this method drives. # C++ parity: ``lineSearch_``."""
        return self._line_search

    @abstractmethod
    def get_updated_direction(
        self,
        problem: Problem,
        gold2: float,
        gradient: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the next search direction.

        # C++ parity: linesearchbasedmethod.hpp:47-49 —
        # ``getUpdatedDirection(const Problem&, Real gold2, const Array&)``.

        ``gold2`` is the squared gradient norm from BEFORE the line search
        (the "orthogonalization coefficient"); ``gradient`` is the gradient
        from before it. The new gradient is read from
        ``self.line_search.last_gradient``.
        """
        ...

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        # C++ parity: linesearchbasedmethod.cpp:35-114.
        ftol = end_criteria.function_epsilon
        max_stationary_state_iterations = end_criteria.max_stationary_state
        ec_type = Type.None_
        problem.reset()
        x = problem.current_value.astype(np.float64, copy=True)
        iteration_number = 0
        # C++ parity: linesearchbasedmethod.cpp:47 — the direction is
        # sized (uninitialized) here purely so ``sz`` can be read off it.
        self._line_search.search_direction = np.empty(x.size, dtype=np.float64)
        done = False

        # Classical initial value for the line-search step.
        t = 1.0
        sz = self._line_search.search_direction.size
        prev_gradient = np.empty(sz, dtype=np.float64)
        problem.set_function_value(problem.value_and_gradient(prev_gradient, x))
        problem.set_gradient_norm_value(dot_product(prev_gradient, prev_gradient))
        self._line_search.search_direction = -prev_gradient

        first_time = True
        while True:
            if not first_time:
                # Deep copy: BFGS differences this against the NEW gradient,
                # which is the same object the line search keeps mutating.
                prev_gradient = self._line_search.last_gradient.astype(
                    np.float64, copy=True
                )
            t, ec_type = self._line_search(problem, ec_type, end_criteria, t)
            # C++ deliberately does NOT throw when the line search fails:
            # it can fail simply because maxIterations was exceeded
            # (linesearchbasedmethod.cpp:71-72).
            if self._line_search.succeed:
                x = self._line_search.last_x.astype(np.float64, copy=True)
                fold = problem.function_value
                problem.set_function_value(self._line_search.last_function_value)

                # Orthogonalization coefficient.
                gold2 = problem.gradient_norm_value
                problem.set_gradient_norm_value(self._line_search.last_gradient_norm2)

                direction = self.get_updated_direction(problem, gold2, prev_gradient)
                # C++ computes ``sddiff = direction - searchDirection()`` here
                # (linesearchbasedmethod.cpp:91) and never reads it; the dead
                # store is not reproduced.
                self._line_search.search_direction = direction

                # Numerical Recipes exit strategy on f(x) (NR in C++, p.423).
                fnew = problem.function_value
                fdiff = (
                    2.0 * abs(fnew - fold) / (abs(fnew) + abs(fold) + QL_EPSILON)
                )
                if fdiff < ftol or end_criteria.check_max_iterations(iteration_number):
                    # Seeding the counter with the maximum makes this fire
                    # unconditionally -> StationaryFunctionValue.
                    _, hit = end_criteria.check_stationary_function_value(
                        0.0, 0.0, max_stationary_state_iterations
                    )
                    if hit is not None:
                        ec_type = hit
                    hit = end_criteria.check_max_iterations(iteration_number)
                    if hit is not None:
                        ec_type = hit
                    return ec_type
                problem.set_current_value(x)
                iteration_number += 1
                first_time = False
            else:
                done = True
            if done:
                break
        problem.set_current_value(x)
        return ec_type
