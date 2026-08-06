"""Fdm1DimSolver — roll a 1-D FD problem back to t=0 and spline the result.

# C++ parity: ql/methods/finitedifferences/solvers/fdm1dimsolver.{hpp,cpp}
# (v1.43).

The solver is a ``LazyObject``: the backward rollback runs once, on the
first ``interpolate_at`` / ``theta_at`` / ``derivative_x`` /
``derivative_xx`` call, and the resulting grid is wrapped in a
``MonotonicCubicNaturalSpline`` over the direction-0 mesher locations.

``theta_at`` reads the solution one snapshot-step before t=0 out of an
``FdmSnapshotCondition`` that the constructor glues onto the caller's
step-condition composite (``FdmStepConditionComposite.join_conditions``),
exactly as C++ does, and differences it against the t=0 solution.

``solver_desc.bc_set`` is handed to ``FdmBackwardSolver`` exactly as C++
hands it ``solverDesc.bcSet``, so a Dirichlet face on the mesh is honoured
through every damping and time step.

C++ takes ``calculator`` as an ``FdmInnerValueCalculator`` and calls
``avgInnerValue(iter, t)``; ``FdmSolverDesc.calculator`` is a plain
callable in this port, so callers pass ``calc.avg_inner_value``.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import MonotonicCubicNaturalSpline
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_backward_solver import FdmBackwardSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_snapshot_condition import (
    FdmSnapshotCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.patterns.lazy_object import LazyObject

#: # C++ parity: ``Null<Real>()`` — ``std::numeric_limits<float>::max()``.
NULL_REAL: float = 3.4028234663852886e38


def snapshot_time(solver_desc: FdmSolverDesc) -> float:
    """Time at which the theta snapshot is taken.

    # C++ parity: the ``thetaCondition_`` member initialiser repeated
    # verbatim in fdm1dimsolver.cpp, fdm2dimsolver.cpp, fdm3dimsolver.cpp
    # and fdmndimsolver.hpp::
    #
    #     0.99 * std::min(1.0/365.0,
    #                     condition->stoppingTimes().empty()
    #                         ? maturity : condition->stoppingTimes().front())
    """
    stopping_times = solver_desc.condition.stopping_times()
    first = solver_desc.maturity if not stopping_times else stopping_times[0]
    return 0.99 * min(1.0 / 365.0, first)


@final
class Fdm1DimSolver(LazyObject):
    """Backward FD solver on a one-dimensional mesh.

    # C++ parity: ``class Fdm1DimSolver : public LazyObject``.
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
        size = layout.size()
        self._x: Array = np.empty(size, dtype=np.float64)
        self._initial_values: Array = np.empty(size, dtype=np.float64)
        for it in layout.iter():
            self._initial_values[it.index] = solver_desc.calculator(it, solver_desc.maturity)
            self._x[it.index] = solver_desc.mesher.location(it, 0)

        self._result_values: Array = np.zeros(size, dtype=np.float64)
        self._interpolation: MonotonicCubicNaturalSpline | None = None

    def _perform_calculations(self) -> None:
        """# C++ parity: ``Fdm1DimSolver::performCalculations``."""
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
        self._result_values = rhs
        self._interpolation = MonotonicCubicNaturalSpline(self._x, self._result_values)

    def _spline(self) -> MonotonicCubicNaturalSpline:
        self.calculate()
        interpolation = self._interpolation
        if interpolation is None:
            qassert.fail("Fdm1DimSolver: the backward rollback produced no interpolation")
        return interpolation

    def interpolate_at(self, x: float) -> float:
        """Solution at ``x``. # C++ parity: ``Fdm1DimSolver::interpolateAt``."""
        return self._spline()(x)

    def theta_at(self, x: float) -> float:
        """Backward time-derivative at ``x``.

        # C++ parity: ``Fdm1DimSolver::thetaAt`` — returns ``Null<Real>()``
        # when the first stopping time sits on t=0 (no room for the
        # snapshot step).
        """
        if self._conditions.stopping_times()[0] == 0.0:
            return NULL_REAL

        self.calculate()
        theta_values = self._theta_condition.get_values()
        qassert.require(
            len(theta_values) == len(self._result_values),
            "Fdm1DimSolver: the theta snapshot was never taken",
        )
        temp = MonotonicCubicNaturalSpline(self._x, theta_values)(x)
        return (temp - self.interpolate_at(x)) / self._theta_condition.get_time()

    def derivative_x(self, x: float) -> float:
        """First space derivative. # C++ parity: ``Fdm1DimSolver::derivativeX``."""
        return self._spline().derivative(x)

    def derivative_xx(self, x: float) -> float:
        """Second space derivative. # C++ parity: ``Fdm1DimSolver::derivativeXX``."""
        return self._spline().second_derivative(x)


__all__ = ["NULL_REAL", "Fdm1DimSolver", "snapshot_time"]
