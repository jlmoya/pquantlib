"""Fdm3DimSolver — roll a 3-D FD problem back to t=0 and interpolate.

# C++ parity: ql/methods/finitedifferences/solvers/fdm3dimsolver.{hpp,cpp}
# (v1.43).

The interpolation is two-stage, exactly as C++ builds it:

1. the rolled-back vector is sliced into ``dim[2]`` matrices of shape
   ``(dim[1], dim[0])`` — one z-slice each, in flat-index order — and each
   slice gets its own ``BicubicSpline`` over ``(x, y)``;
2. evaluating at ``(x, y, z)`` evaluates every slice spline at ``(x, y)``
   and runs a ``MonotonicCubicNaturalSpline`` through those ``dim[2]``
   values over the z axis.

Note the asymmetry that C++ has and this port keeps: the in-slice
interpolation is a *natural* bicubic spline while the cross-slice one is
the Hyman-filtered monotonic cubic.

``solver_desc.bc_set`` reaches ``FdmBackwardSolver`` just as it does in
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver.Fdm1DimSolver`.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.math.interpolations.cubic_interpolation import MonotonicCubicNaturalSpline
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
class Fdm3DimSolver(LazyObject):
    """Backward FD solver on a three-dimensional mesh.

    # C++ parity: ``class Fdm3DimSolver : public LazyObject``.
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
        qassert.require(len(dim) == 3, f"Fdm3DimSolver needs a 3-D layout; got {len(dim)}-D")

        self._initial_values: Array = np.empty(layout.size(), dtype=np.float64)
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for it in layout.iter():
            self._initial_values[it.index] = solver_desc.calculator(it, solver_desc.maturity)
            c = it.coordinates
            if c[1] == 0 and c[2] == 0:
                xs.append(solver_desc.mesher.location(it, 0))
            if c[0] == 0 and c[2] == 0:
                ys.append(solver_desc.mesher.location(it, 1))
            if c[0] == 0 and c[1] == 0:
                zs.append(solver_desc.mesher.location(it, 2))
        self._x: Array = np.asarray(xs, dtype=np.float64)
        self._y: Array = np.asarray(ys, dtype=np.float64)
        self._z: Array = np.asarray(zs, dtype=np.float64)

        # C++ parity: ``vector<Matrix> resultValues_(dim[2], Matrix(dim[1], dim[0]))``.
        self._slice_shape: tuple[int, int] = (dim[1], dim[0])
        self._result_values: list[Matrix] = [
            np.zeros(self._slice_shape, dtype=np.float64) for _ in range(dim[2])
        ]
        self._interpolation: list[BicubicSpline] = []

    def _slices(self, values: Array) -> list[Matrix]:
        """Split a flat solution vector into one ``(dim[1], dim[0])`` matrix per z.

        # C++ parity: the ``std::copy(rhs.begin()+i*y*x, rhs.begin()+(i+1)*y*x, ...)``
        # loop shared by ``performCalculations`` and ``thetaAt``.
        """
        stride = self._slice_shape[0] * self._slice_shape[1]
        return [values[i * stride : (i + 1) * stride].reshape(self._slice_shape) for i in range(len(self._z))]

    def _perform_calculations(self) -> None:
        """# C++ parity: ``Fdm3DimSolver::performCalculations``."""
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
        self._result_values = self._slices(rhs)
        self._interpolation = [BicubicSpline(self._x, self._y, slice_) for slice_ in self._result_values]

    def _z_section(self, splines: list[BicubicSpline], x: float, y: float) -> Array:
        return np.asarray([spline(x, y) for spline in splines], dtype=np.float64)

    def interpolate_at(self, x: float, y: float, z: float) -> float:
        """Solution at ``(x, y, z)``. # C++ parity: ``Fdm3DimSolver::interpolateAt``."""
        self.calculate()
        z_array = self._z_section(self._interpolation, x, y)
        return MonotonicCubicNaturalSpline(self._z, z_array)(z)

    def theta_at(self, x: float, y: float, z: float) -> float:
        """Backward time-derivative at ``(x, y, z)``.

        # C++ parity: ``Fdm3DimSolver::thetaAt``.
        """
        if self._conditions.stopping_times()[0] == 0.0:
            return NULL_REAL

        self.calculate()
        theta_values = self._theta_condition.get_values()
        qassert.require(
            theta_values.size == self._initial_values.size,
            "Fdm3DimSolver: the theta snapshot was never taken",
        )
        splines = [BicubicSpline(self._x, self._y, s) for s in self._slices(theta_values)]
        z_array = self._z_section(splines, x, y)
        theta = MonotonicCubicNaturalSpline(self._z, z_array)(z)
        return (theta - self.interpolate_at(x, y, z)) / self._theta_condition.get_time()


__all__ = ["Fdm3DimSolver"]
