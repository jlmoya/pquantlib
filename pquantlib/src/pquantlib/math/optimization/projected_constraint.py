"""Constraint restricted to a subset of a parameter vector.

# C++ parity: ql/math/optimization/projectedconstraint.hpp (v1.43).

Wraps an ``n``-dimensional constraint so that an optimizer working on the
``k`` free coordinates can test it: ``test`` rebuilds the full vector via
``Projection.include`` and delegates; the bounds are the inner
constraint's bounds at the rebuilt point, projected back down.

Note the consequence of ``include`` restoring the FIXED coordinates from
the vector the projection was constructed with: if one of those fixed
values violates the inner constraint, ``test`` returns ``False`` for
EVERY free vector.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.constraint import Constraint
from pquantlib.math.optimization.projection import Projection


class ProjectedConstraint(Constraint):
    """Constraint seen through a :class:`Projection`.

    # C++ parity: ``class ProjectedConstraint`` in
    # ql/math/optimization/projectedconstraint.hpp:33-73 (v1.43).
    """

    __slots__ = ("_constraint", "_projection")

    def __init__(
        self,
        constraint: Constraint,
        parameter_values: npt.NDArray[np.float64] | None = None,
        fix_parameters: list[bool] | None = None,
        projection: Projection | None = None,
    ) -> None:
        """Build from a base vector + mask, or from an existing ``Projection``.

        # C++ parity: projectedconstraint.hpp:62-72 — two constructors.
        # Pass ``projection=`` to select the second form; ``parameter_values``
        # and ``fix_parameters`` are then ignored.
        """
        self._constraint: Constraint = constraint
        if projection is not None:
            self._projection: Projection = Projection(
                projection.fixed_parameters, projection.fix_parameters
            )
        else:
            if parameter_values is None:
                parameter_values = np.zeros(0, dtype=np.float64)
            self._projection = Projection(parameter_values, fix_parameters)

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        # C++ parity: projectedconstraint.hpp:45-47.
        return self._constraint.test(self._projection.include(params))

    def upper_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: projectedconstraint.hpp:48-50.
        return self._projection.project(
            self._constraint.upper_bound(self._projection.include(params))
        )

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: projectedconstraint.hpp:51-53.
        return self._projection.project(
            self._constraint.lower_bound(self._projection.include(params))
        )
