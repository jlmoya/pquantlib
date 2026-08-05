"""Multi-dimensional conjugate-gradient method.

# C++ parity: ql/math/optimization/conjugategradient.{hpp,cpp} (v1.43).

Fletcher-Reeves-Polak-Ribiere, adapted from *Numerical Recipes in C*,
2nd edition. The search direction is

    d_i = -f'(x_i) + c_i * d_{i-1},   c_i = |f'(x_i)|^2 / |f'(x_{i-1})|^2

with ``d_1 = -f'(x_1)``. In the v1.43 implementation ``c_i`` is
``P.gradientNormValue() / gold2``, where the numerator has ALREADY been
updated to the new squared norm by ``LineSearchBasedMethod::minimize``
and ``gold2`` is the pre-line-search value it saved. That makes this the
Fletcher-Reeves form; the ``oldGradient`` parameter is accepted and
unused (only ``BFGS`` reads it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.math.optimization.line_search_based_method import LineSearchBasedMethod

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from pquantlib.math.optimization.problem import Problem


class ConjugateGradient(LineSearchBasedMethod):
    """Conjugate gradient: ``d = -g_new + (|g_new|^2 / |g_old|^2) * d_old``.

    # C++ parity: ``class ConjugateGradient`` in
    # ql/math/optimization/conjugategradient.hpp:47-57 (v1.43).
    """

    __slots__ = ()

    def get_updated_direction(
        self,
        problem: Problem,
        gold2: float,
        gradient: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # C++ parity: conjugategradient.cpp:30-35 — the third parameter is
        # unnamed in C++.
        del gradient
        return (
            -self._line_search.last_gradient
            + (problem.gradient_norm_value / gold2) * self._line_search.search_direction
        )
