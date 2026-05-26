"""Optimization problem bundle (cost + constraint + current state).

# C++ parity: ql/math/optimization/problem.hpp (v1.42.1).

C++ stores the cost function and constraint by reference (warning in
``problem.hpp:37``), with mutable state for the current minimum, the
function value, the gradient-norm squared value, and evaluation
counters. Each ``value`` / ``values`` / ``gradient`` call increments
its counter.

The Python port mirrors that mutability with a plain (non-frozen)
class. The L1-D spec suggests ``@dataclass(frozen=True, slots=True)``
with ``object.__setattr__`` in ``__post_init__`` as an alternative;
we chose the plain class because the C++ semantics demand mutation
through methods, and the frozen-with-bypass pattern reads as a
worked-around workaround rather than honest C++ parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pquantlib.math.optimization.constraint import Constraint
    from pquantlib.math.optimization.cost_function import CostFunction


class Problem:
    """Constrained optimization problem.

    # C++ parity: ``class Problem`` in
    # ql/math/optimization/problem.hpp:42-113 (v1.42.1).

    Stores cost function and constraint as references (Python: as
    attributes — identity-preserving). The current value, function
    value, gradient-norm squared, and evaluation counters are
    mutable; mutation happens through ``value`` / ``values`` /
    ``gradient`` / ``value_and_gradient`` / ``reset`` /
    ``set_current_value`` / ``set_function_value`` /
    ``set_gradient_norm_value``.
    """

    __slots__ = (
        "_constraint",
        "_cost_function",
        "_current_value",
        "_function_evaluation",
        "_function_value",
        "_gradient_evaluation",
        "_squared_norm",
    )

    def __init__(
        self,
        cost_function: CostFunction,
        constraint: Constraint,
        initial_value: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self._cost_function: CostFunction = cost_function
        self._constraint: Constraint = constraint
        # C++ parity: problem.hpp:47 — default-construct an empty Array
        # when no initial value is supplied.
        self._current_value: npt.NDArray[np.float64] = (
            initial_value.astype(np.float64, copy=True)
            if initial_value is not None
            else np.empty(0, dtype=np.float64)
        )
        # Mutable per-call state.
        # NaN is the closest Python analogue to C++ ``Null<Real>()``,
        # which QuantLib defines as ``std::numeric_limits<Real>::max()``.
        # Either sentinel works because the only check ever applied is
        # ``isNull(x)``; pquantlib will fully port ``Null<>`` in a later
        # cluster (carry-out documented).
        self._function_value: float = float("nan")
        self._squared_norm: float = float("nan")
        self._function_evaluation: int = 0
        self._gradient_evaluation: int = 0

    # --- inspectors -----------------------------------------------------

    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def cost_function(self) -> CostFunction:
        return self._cost_function

    @property
    def current_value(self) -> npt.NDArray[np.float64]:
        return self._current_value

    @property
    def function_value(self) -> float:
        return self._function_value

    @property
    def gradient_norm_value(self) -> float:
        return self._squared_norm

    @property
    def function_evaluation(self) -> int:
        return self._function_evaluation

    @property
    def gradient_evaluation(self) -> int:
        return self._gradient_evaluation

    # --- setters --------------------------------------------------------

    def set_current_value(self, current_value: npt.NDArray[np.float64]) -> None:
        self._current_value = current_value.astype(np.float64, copy=True)

    def set_function_value(self, function_value: float) -> None:
        self._function_value = function_value

    def set_gradient_norm_value(self, squared_norm: float) -> None:
        self._squared_norm = squared_norm

    # --- counted invokers ----------------------------------------------

    def value(self, x: npt.NDArray[np.float64]) -> float:
        """Increment function-evaluation counter, return cost-function value at ``x``."""
        # C++ parity: problem.hpp:116-119.
        self._function_evaluation += 1
        return self._cost_function.value(x)

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Increment function-evaluation counter, return cost-function residuals at ``x``."""
        # C++ parity: problem.hpp:121-124.
        self._function_evaluation += 1
        return self._cost_function.values(x)

    def gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        """Increment gradient-evaluation counter, store cost-function gradient at ``x``."""
        # C++ parity: problem.hpp:126-130.
        self._gradient_evaluation += 1
        self._cost_function.gradient(grad, x)

    def value_and_gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
        """Increment both counters, return ``value(x)`` and store ``gradient`` at ``x``."""
        # C++ parity: problem.hpp:132-137.
        self._function_evaluation += 1
        self._gradient_evaluation += 1
        self._cost_function.gradient(grad, x)
        return self._cost_function.value(x)

    def reset(self) -> None:
        """Zero the counters, NaN-out the cached function and gradient values."""
        # C++ parity: problem.hpp:139-142.
        self._function_evaluation = 0
        self._gradient_evaluation = 0
        self._function_value = float("nan")
        self._squared_norm = float("nan")
