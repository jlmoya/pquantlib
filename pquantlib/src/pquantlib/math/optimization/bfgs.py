"""Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method.

# C++ parity: ql/math/optimization/bfgs.{hpp,cpp} (v1.43).

Adapted from *Numerical Recipes in C*, 2nd edition. The class keeps an
explicit inverse-Hessian approximation and returns
``d = -H * g_new`` as the search direction.

Three details worth spelling out, because the textbook BFGS update does
NOT look like this one:

1. **The rank-two correction uses ``fae``, not ``1/fae``.** ``fad`` is set
   to ``1/fae`` and used on the middle term, while the third term is
   scaled by the raw ``fae`` (bfgs.cpp:71). That is the *Numerical
   Recipes* ``dfpmin`` formulation; it is not a typo to "fix".
2. **The update is skipped when the curvature condition is weak** —
   ``fac > sqrt(1e-8 * sumdg * sumxi)`` (bfgs.cpp:58). When it is
   skipped, ``H`` is left untouched and the direction is still computed
   from the stale ``H``.
3. **``inverse_hessian`` persists across ``minimize`` calls.** C++ stores
   it as a member and only rebuilds it when ``rows() == 0``, so reusing a
   ``BFGS`` instance for a second problem carries the first problem's
   curvature over. Reproduced here; construct a fresh ``BFGS`` per
   problem if that is not what you want.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.line_search_based_method import LineSearchBasedMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.line_search import LineSearch
    from pquantlib.math.optimization.problem import Problem


class BFGS(LineSearchBasedMethod):
    """BFGS with an explicit inverse-Hessian approximation.

    # C++ parity: ``class BFGS`` in ql/math/optimization/bfgs.hpp:39-51
    # (v1.43).
    """

    __slots__ = ("_inverse_hessian",)

    def __init__(self, line_search: LineSearch | None = None) -> None:
        super().__init__(line_search)
        # C++ parity: bfgs.hpp:50 — a default-constructed Matrix, i.e.
        # zero rows, which is the "not yet initialised" sentinel.
        self._inverse_hessian: npt.NDArray[np.float64] = np.empty(
            (0, 0), dtype=np.float64
        )

    @property
    def inverse_hessian(self) -> npt.NDArray[np.float64]:
        """Current inverse-Hessian approximation. # C++ parity: ``inverseHessian_``."""
        return self._inverse_hessian

    def get_updated_direction(
        self,
        problem: Problem,
        gold2: float,
        gradient: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # C++ parity: bfgs.cpp:26-87. The second parameter is unnamed in C++.
        del gold2
        n = problem.current_value.size
        if self._inverse_hessian.shape[0] == 0:
            # C++ parity: bfgs.cpp:29-36 — zero matrix with a unit diagonal.
            self._inverse_hessian = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                self._inverse_hessian[i][i] = 1.0

        inv_h = self._inverse_hessian
        search_direction = self._line_search.search_direction

        diff_gradient = self._line_search.last_gradient - gradient
        diff_gradient_with_hessian_applied = np.zeros(n, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc += float(inv_h[i][j]) * float(diff_gradient[j])
            diff_gradient_with_hessian_applied[i] = acc

        fac = 0.0
        fae = 0.0
        sumdg = 0.0
        sumxi = 0.0
        for i in range(n):
            fac += float(diff_gradient[i]) * float(search_direction[i])
            fae += float(diff_gradient[i]) * float(diff_gradient_with_hessian_applied[i])
            sumdg += float(diff_gradient[i]) ** 2.0
            sumxi += float(search_direction[i]) ** 2.0

        # Skip the update when ``fac`` is not sufficiently positive.
        if fac > np.sqrt(1e-8 * sumdg * sumxi):
            fac = 1.0 / fac
            fad = 1.0 / fae

            for i in range(n):
                diff_gradient[i] = (
                    fac * search_direction[i]
                    - fad * diff_gradient_with_hessian_applied[i]
                )

            for i in range(n):
                for j in range(n):
                    inv_h[i][j] += fac * search_direction[i] * search_direction[j]
                    inv_h[i][j] -= (
                        fad
                        * diff_gradient_with_hessian_applied[i]
                        * diff_gradient_with_hessian_applied[j]
                    )
                    inv_h[i][j] += fae * diff_gradient[i] * diff_gradient[j]
        # C++ has a commented-out ``else throw "BFGS: FAC not sufficiently
        # positive"``; the skip is silent.

        last_gradient = self._line_search.last_gradient
        direction = np.empty(n, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc -= float(inv_h[i][j]) * float(last_gradient[j])
            direction[i] = acc

        return direction
