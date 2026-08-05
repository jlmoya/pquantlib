"""Cost function restricted to a subset of its parameters.

# C++ parity: ql/math/optimization/projectedcostfunction.{hpp,cpp} (v1.43).

``ProjectedCostFunction`` presents an ``n``-parameter cost function to an
optimizer as a ``k``-parameter one (``k`` = number of free parameters),
scattering the free values back into the full vector before every
evaluation. C++ achieves this by inheriting from BOTH ``CostFunction``
and ``Projection``; the Python port does the same, so ``project`` and
``include`` remain available on the instance.

Only ``value`` and ``values`` are overridden. ``gradient``, ``jacobian``
and friends fall through to the ``CostFunction`` finite-difference
defaults, which then differentiate with respect to the free coordinates.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.projection import Projection


class ProjectedCostFunction(CostFunction, Projection):
    """Cost function over the free subset of a parameter vector.

    # C++ parity: ``class ProjectedCostFunction`` in
    # ql/math/optimization/projectedcostfunction.{hpp,cpp} (v1.43).
    """

    __slots__ = ("_cost_function",)

    def __init__(
        self,
        cost_function: CostFunction,
        parameter_values: npt.NDArray[np.float64] | None = None,
        fix_parameters: list[bool] | None = None,
        projection: Projection | None = None,
    ) -> None:
        """Build from a base vector + mask, or from an existing ``Projection``.

        # C++ parity: projectedcostfunction.cpp:26-35 — two constructors,
        # one taking ``(parameterValues, fixParameters)`` and one taking a
        # ready-made ``Projection`` (which it COPIES). Python has no
        # overloading, so pass ``projection=`` to select the second form;
        # ``parameter_values`` and ``fix_parameters`` are then ignored.
        """
        if projection is not None:
            Projection.__init__(
                self, projection.fixed_parameters, projection.fix_parameters
            )
        else:
            if parameter_values is None:
                parameter_values = np.zeros(0, dtype=np.float64)
            Projection.__init__(self, parameter_values, fix_parameters)
        self._cost_function: CostFunction = cost_function

    def value(self, x: npt.NDArray[np.float64]) -> float:
        """Cost at the full vector obtained by scattering ``x`` into the free slots.

        # C++ parity: projectedcostfunction.cpp:37-40. The parameter is
        # named ``freeParameters`` in C++; pquantlib keeps the base class's
        # ``x`` because pyright requires override parameter names to match.
        """
        self._map_free_parameters(x)
        return self._cost_function.value(self._actual_parameters)

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Residuals at the full vector obtained by scattering ``x``.

        # C++ parity: projectedcostfunction.cpp:42-45.
        """
        self._map_free_parameters(x)
        return self._cost_function.values(self._actual_parameters)
