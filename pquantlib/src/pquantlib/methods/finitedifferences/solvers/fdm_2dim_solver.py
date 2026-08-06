"""Fdm2DimSolver — roll a 2-D FD problem back to t=0 and spline the result.

# C++ parity: ql/methods/finitedifferences/solvers/fdm2dimsolver.{hpp,cpp}
# (v1.43).

The rolled-back solution vector is reshaped into a ``(dim[1], dim[0])``
matrix — rows are the second direction, columns the first, matching the
C++ ``Matrix resultValues_(dim[1], dim[0])`` filled by a flat ``std::copy``
— and wrapped in a ``BicubicSpline`` over the two axis coordinate vectors.
All five partial derivatives come straight off that surface.

The axes are read out of the layout exactly as C++ does: ``x_`` collects
the direction-0 location of every node whose second coordinate is 0, and
``y_`` the direction-1 location of every node whose first coordinate is 0.

``solver_desc.bc_set`` reaches ``FdmBackwardSolver`` just as it does in
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver.Fdm1DimSolver`.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.math.matrix import Matrix
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import NULL_REAL, snapshot_time
from pquantlib.methods.finitedifferences.solvers.fdm_backward_solver import FdmBackwardSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_snapshot_condition import (
    FdmSnapshotCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.patterns.lazy_object import LazyObject


@final
class Fdm2DimSolver(LazyObject):
    """Backward FD solver on a two-dimensional mesh.

    # C++ parity: ``class Fdm2DimSolver : public LazyObject``.
    """

    def __init__(
        self,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
        op: FdmLinearOpComposite,
    ) -> None:
        super().__init__()
        self._solver_desc: FdmSolverDesc = solver_desc
        self._scheme_desc: FdmSchemeDesc = scheme_desc
        self._op: FdmLinearOpComposite = op
        self._theta_condition: FdmSnapshotCondition = FdmSnapshotCondition(snapshot_time(solver_desc))
        self._conditions: FdmStepConditionComposite = FdmStepConditionComposite.join_conditions(
            self._theta_condition, solver_desc.condition
        )

        layout = solver_desc.mesher.layout()
        dim = layout.dim()
        qassert.require(len(dim) == 2, f"Fdm2DimSolver needs a 2-D layout; got {len(dim)}-D")

        self._initial_values: Array = np.empty(layout.size(), dtype=np.float64)
        xs: list[float] = []
        ys: list[float] = []
        for it in layout.iter():
            self._initial_values[it.index] = solver_desc.calculator(it, solver_desc.maturity)
            if it.coordinates[1] == 0:
                xs.append(solver_desc.mesher.location(it, 0))
            if it.coordinates[0] == 0:
                ys.append(solver_desc.mesher.location(it, 1))
        self._x: Array = np.asarray(xs, dtype=np.float64)
        self._y: Array = np.asarray(ys, dtype=np.float64)

        # C++ parity: ``Matrix resultValues_(dim[1], dim[0])`` — rows = y.
        self._result_values: Matrix = np.zeros((dim[1], dim[0]), dtype=np.float64)
        self._interpolation: BicubicSpline | None = None

    def _perform_calculations(self) -> None:
        """# C++ parity: ``Fdm2DimSolver::performCalculations``."""
        rhs = self._initial_values.copy()
        rhs = FdmBackwardSolver(
            self._op, self._conditions, self._scheme_desc, self._solver_desc.bc_set
        ).rollback(
            rhs,
            self._solver_desc.maturity,
            0.0,
            self._solver_desc.time_steps,
            self._solver_desc.damping_steps,
        )
        self._result_values = rhs.reshape(self._result_values.shape)
        self._interpolation = BicubicSpline(self._x, self._y, self._result_values)

    def _surface(self) -> BicubicSpline:
        self.calculate()
        interpolation = self._interpolation
        if interpolation is None:
            qassert.fail("Fdm2DimSolver: the backward rollback produced no interpolation")
        return interpolation

    def interpolate_at(self, x: float, y: float) -> float:
        """Solution at ``(x, y)``. # C++ parity: ``Fdm2DimSolver::interpolateAt``."""
        return self._surface()(x, y)

    def theta_at(self, x: float, y: float) -> float:
        """Backward time-derivative at ``(x, y)``.

        # C++ parity: ``Fdm2DimSolver::thetaAt``.
        """
        if self._conditions.stopping_times()[0] == 0.0:
            return NULL_REAL

        self.calculate()
        theta_values = self._theta_condition.get_values()
        qassert.require(
            theta_values.size == self._result_values.size,
            "Fdm2DimSolver: the theta snapshot was never taken",
        )
        theta_matrix = theta_values.reshape(self._result_values.shape)
        surface = BicubicSpline(self._x, self._y, theta_matrix)
        return (surface(x, y) - self.interpolate_at(x, y)) / self._theta_condition.get_time()

    def derivative_x(self, x: float, y: float) -> float:
        """# C++ parity: ``Fdm2DimSolver::derivativeX``."""
        return self._surface().derivative_x(x, y)

    def derivative_y(self, x: float, y: float) -> float:
        """# C++ parity: ``Fdm2DimSolver::derivativeY``."""
        return self._surface().derivative_y(x, y)

    def derivative_xx(self, x: float, y: float) -> float:
        """# C++ parity: ``Fdm2DimSolver::derivativeXX`` — ``secondDerivativeX``."""
        return self._surface().second_derivative_x(x, y)

    def derivative_yy(self, x: float, y: float) -> float:
        """# C++ parity: ``Fdm2DimSolver::derivativeYY`` — ``secondDerivativeY``."""
        return self._surface().second_derivative_y(x, y)

    def derivative_xy(self, x: float, y: float) -> float:
        """# C++ parity: ``Fdm2DimSolver::derivativeXY``."""
        return self._surface().derivative_xy(x, y)


__all__ = ["Fdm2DimSolver"]
