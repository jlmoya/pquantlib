"""Armijo backtracking line search.

# C++ parity: ql/math/optimization/armijo.{hpp,cpp} (v1.43).

Let ``alpha`` and ``beta`` be two scalars in [0, 1], ``x`` the current
iterate, ``d`` the search direction and ``t`` the step. The search stops
when ``t`` satisfies both

    f(x + t*d) - f(x) <= -alpha * t * f'(x + t*d)

and

    f(x + t/beta * d) - f(x) > -alpha/beta * t * f'(x + t*d)

(Polak, *Algorithms and Consistent Approximations*, Springer 1997).

Two details that a from-scratch backtracking loop would get wrong:

1. ``qpt_`` — the directional derivative the sufficient-decrease test is
   measured against — is taken from the gradient left over by the
   PREVIOUS call (``-g_prev . d``), not from a fresh gradient at the
   current point. Only on the very first call, when the stored gradient
   is still empty, does it fall back to ``Problem::gradientNormValue()``.
2. the gradient is evaluated TWICE per accepted step: once at the end of
   each backtracking iteration and once again unconditionally after the
   loop. That doubling is visible in
   ``Problem::gradientEvaluation()`` and is reproduced here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib.math.optimization.line_search import LineSearch, dot_product

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria, Type
    from pquantlib.math.optimization.problem import Problem


class ArmijoLineSearch(LineSearch):
    """Armijo line search.

    # C++ parity: ``class ArmijoLineSearch`` in
    # ql/math/optimization/armijo.hpp:48-63 (v1.43). Constructor defaults
    # ``eps = 1e-8``, ``alpha = 0.05``, ``beta = 0.65``; ``eps`` is
    # forwarded to ``LineSearch`` which discards it.
    """

    __slots__ = ("_alpha", "_beta")

    def __init__(self, eps: float = 1e-8, alpha: float = 0.05, beta: float = 0.65) -> None:
        super().__init__(eps)
        self._alpha: float = alpha
        self._beta: float = beta

    @property
    def alpha(self) -> float:
        """Sufficient-decrease coefficient. # C++ parity: armijo.hpp:62."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Backtracking shrink factor. # C++ parity: armijo.hpp:62."""
        return self._beta

    def __call__(
        self,
        problem: Problem,
        ec_type: Type,
        end_criteria: EndCriteria,
        t_ini: float,
    ) -> tuple[float, Type]:
        # C++ parity: armijo.cpp:26-85.
        constraint = problem.constraint
        self._succeed = True
        max_iter = False
        t = t_ini
        loop_number = 0

        q0 = problem.function_value
        qp0 = problem.gradient_norm_value

        self._qt = q0
        # C++ parity: armijo.cpp:42 — the directional derivative comes from
        # the gradient stored by the PREVIOUS call; ``gradientNormValue()``
        # is only the first-call fallback.
        self._qpt = (
            qp0
            if self._gradient.size == 0
            else -dot_product(self._gradient, self._search_direction)
        )

        # C++ parity: armijo.cpp:45 — ``Array(n)`` leaves the elements
        # uninitialized; every one is overwritten by ``P.gradient`` before
        # being read, so ``np.empty`` is the faithful analogue.
        self._gradient = np.empty(problem.current_value.size, dtype=np.float64)
        self._xtd = problem.current_value.astype(np.float64, copy=True)
        t = self.update(self._xtd, self._search_direction, t, constraint)
        self._qt = problem.value(self._xtd)

        if (self._qt - q0) > -self._alpha * t * self._qpt:
            while True:
                loop_number += 1
                # Decrease step.
                t *= self._beta
                qtold = self._qt
                self._xtd = problem.current_value.astype(np.float64, copy=True)
                t = self.update(self._xtd, self._search_direction, t, constraint)

                self._qt = problem.value(self._xtd)
                problem.gradient(self._gradient, self._xtd)
                hit = end_criteria.check_max_iterations(loop_number)
                max_iter = hit is not None
                if hit is not None:
                    ec_type = hit
                # C++ parity: armijo.cpp:69-72 — do/while condition.
                keep_going = (
                    (self._qt - q0) > (-self._alpha * t * self._qpt)
                    or (qtold - q0) <= (-self._alpha * t * self._qpt / self._beta)
                ) and not max_iter
                if not keep_going:
                    break

        if max_iter:
            self._succeed = False

        # C++ parity: armijo.cpp:79-81 — the gradient is recomputed here
        # even when it was already computed at the end of the loop.
        problem.gradient(self._gradient, self._xtd)
        self._qpt = dot_product(self._gradient, self._gradient)

        return t, ec_type
