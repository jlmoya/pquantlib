"""Multi-dimensional steepest-descent method.

# C++ parity: ql/math/optimization/steepestdescent.{hpp,cpp} (v1.43).

The entire class is one line: the search direction is minus the gradient
at the point the line search just accepted. Everything else — the
iteration loop, the convergence test, the end-criteria bookkeeping —
lives in :class:`~pquantlib.math.optimization.line_search_based_method.LineSearchBasedMethod`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.math.optimization.line_search_based_method import LineSearchBasedMethod

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from pquantlib.math.optimization.problem import Problem


class SteepestDescent(LineSearchBasedMethod):
    """Steepest descent: ``d = -f'(x)``.

    # C++ parity: ``class SteepestDescent`` in
    # ql/math/optimization/steepestdescent.hpp:37-47 (v1.43).
    """

    __slots__ = ()

    def get_updated_direction(
        self,
        problem: Problem,
        gold2: float,
        gradient: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # C++ parity: steepestdescent.cpp:27-31 — all three parameters are
        # unnamed in C++; only the line search's last gradient is used.
        del problem, gold2, gradient
        return -self._line_search.last_gradient
