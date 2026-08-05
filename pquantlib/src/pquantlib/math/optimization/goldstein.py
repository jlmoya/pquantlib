"""Goldstein and Price line search.

# C++ parity: ql/math/optimization/goldstein.{hpp,cpp} (v1.43).

Unlike Armijo, which only ever shrinks the step, Goldstein brackets it:
it maintains ``tl`` (a step known to be too short) and ``tr`` (one known
to be too long) and accepts ``t`` only when the decrease sits between the
two Goldstein bounds

    -beta * t * f'  <=  f(x + t*d) - f(x)  <=  -alpha * t * f'.

While no upper bracket has been found (``tr`` still exactly zero, tested
with ``close_enough``) the step is EXPANDED by ``extrapolation_``;
afterwards it bisects ``(tl + tr) / 2``.

Both the ``qpt_``-from-the-previous-gradient rule and the double gradient
evaluation described in :mod:`pquantlib.math.optimization.armijo` apply
here too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib.math.closeness import close_enough
from pquantlib.math.optimization.line_search import LineSearch, dot_product

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria, Type
    from pquantlib.math.optimization.problem import Problem


class GoldsteinLineSearch(LineSearch):
    """Goldstein and Price line search.

    # C++ parity: ``class GoldsteinLineSearch`` in
    # ql/math/optimization/goldstein.hpp:30-48 (v1.43). Constructor
    # defaults ``eps = 1e-8``, ``alpha = 0.05``, ``beta = 0.65``,
    # ``extrapolation = 1.5``; ``eps`` is forwarded to ``LineSearch``
    # which discards it.
    """

    __slots__ = ("_alpha", "_beta", "_extrapolation")

    def __init__(
        self,
        eps: float = 1e-8,
        alpha: float = 0.05,
        beta: float = 0.65,
        extrapolation: float = 1.5,
    ) -> None:
        super().__init__(eps)
        self._alpha: float = alpha
        self._beta: float = beta
        self._extrapolation: float = extrapolation

    @property
    def alpha(self) -> float:
        """Upper Goldstein coefficient. # C++ parity: goldstein.hpp:46."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Lower Goldstein coefficient. # C++ parity: goldstein.hpp:46."""
        return self._beta

    @property
    def extrapolation(self) -> float:
        """Step expansion factor used before a bracket exists.

        # C++ parity: goldstein.hpp:47.
        """
        return self._extrapolation

    def __call__(
        self,
        problem: Problem,
        ec_type: Type,
        end_criteria: EndCriteria,
        t_ini: float,
    ) -> tuple[float, Type]:
        # C++ parity: goldstein.cpp:26-91.
        constraint = problem.constraint
        self._succeed = True
        max_iter = False
        t = t_ini
        loop_number = 0

        q0 = problem.function_value
        qp0 = problem.gradient_norm_value

        tl = 0.0
        tr = 0.0

        self._qt = q0
        self._qpt = (
            qp0
            if self._gradient.size == 0
            else -dot_product(self._gradient, self._search_direction)
        )

        self._gradient = np.empty(problem.current_value.size, dtype=np.float64)
        self._xtd = problem.current_value.astype(np.float64, copy=True)
        t = self.update(self._xtd, self._search_direction, t, constraint)
        self._qt = problem.value(self._xtd)

        while (self._qt - q0) < -self._beta * t * self._qpt or (
            self._qt - q0
        ) > -self._alpha * t * self._qpt:
            if (self._qt - q0) > -self._alpha * t * self._qpt:
                tr = t
            else:
                tl = t
            loop_number += 1

            # C++ parity: goldstein.cpp:62-65 — expand while no upper
            # bracket exists, bisect once one does.
            if close_enough(tr, 0.0):
                t *= self._extrapolation
            else:
                t = (tl + tr) / 2.0

            self._xtd = problem.current_value.astype(np.float64, copy=True)
            t = self.update(self._xtd, self._search_direction, t, constraint)

            self._qt = problem.value(self._xtd)
            problem.gradient(self._gradient, self._xtd)
            hit = end_criteria.check_max_iterations(loop_number)
            max_iter = hit is not None
            if hit is not None:
                ec_type = hit

            if max_iter:
                break

        if max_iter:
            self._succeed = False

        # C++ parity: goldstein.cpp:85-87.
        problem.gradient(self._gradient, self._xtd)
        self._qpt = dot_product(self._gradient, self._gradient)

        return t, ec_type
