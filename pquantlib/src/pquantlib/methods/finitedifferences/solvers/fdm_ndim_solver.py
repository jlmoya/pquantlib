"""FdmNdimSolver — the n-dimensional backward FD solver.

# C++ parity: ql/methods/finitedifferences/solvers/fdmndimsolver.hpp
# (v1.43) — header-only ``template <Size N> class FdmNdimSolver``.

Where ``Fdm1DimSolver`` / ``Fdm2DimSolver`` / ``Fdm3DimSolver`` hand-roll
their interpolation per rank, this one hands the whole rolled-back grid to
``MultiCubicSpline`` — QuantLib's own tensor cubic interpolation — with
extrapolation disabled on every axis, exactly as the C++ constructor does
(``extrapolation_(std::vector<bool>(N, false))``).

**The C++ template parameter.** ``FdmNdimSolver<N>`` fixes the rank at
compile time and ``QL_REQUIRE``s that the mesher layout has the same rank.
Python has no such parameter, so the rank is read off the layout and the
optional ``n`` argument reproduces the check when a caller wants it.

**Index order.** The C++ ``data_table`` is filled by ``setValue`` so that
``f[c_0][c_1]...[c_{N-1}]`` holds the value at layout coordinates ``c``,
while the layout's flat index runs with the *first* axis fastest. The
numpy equivalent of that data table therefore has shape
``(dim[0], ..., dim[N-1])`` and is filled coordinate-wise, not by a plain
C-order reshape of the solution vector.

``solver_desc.bc_set`` reaches ``FdmBackwardSolver`` just as it does in
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver.Fdm1DimSolver`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.multi_cubic_spline import MultiCubicSpline
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
class FdmNdimSolver(LazyObject):
    """Backward FD solver on a mesh of any rank.

    # C++ parity: ``template <Size N> class FdmNdimSolver : public LazyObject``.
    """

    def __init__(
        self,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
        op: FdmLinearOpComposite,
        n: int | None = None,
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
        # C++ parity: QL_REQUIRE(layout->dim().size() == N, ...).
        qassert.require(
            n is None or len(dim) == n,
            f"solver dim {n} does not fit to layout dim {len(dim)}",
        )
        self._n: int = len(dim)
        self._dim: tuple[int, ...] = dim

        self._initial_values: Array = np.empty(layout.size(), dtype=np.float64)
        axes: list[list[float]] = [[] for _ in range(self._n)]
        for it in layout.iter():
            self._initial_values[it.index] = solver_desc.calculator(it, solver_desc.maturity)
            c = it.coordinates
            total = sum(c)
            for i in range(self._n):
                # C++ parity: ``(accumulate(c) - c[i]) == 0`` — the node is on
                # axis ``i`` with every other coordinate at zero.
                if total - c[i] == 0:
                    axes[i].append(solver_desc.mesher.location(it, i))
        self._x: list[Array] = [np.asarray(a, dtype=np.float64) for a in axes]

        self._interp: MultiCubicSpline | None = None

    @staticmethod
    def set_value(f: np.ndarray, x: Sequence[int], value: float) -> None:
        """Write ``value`` into the data table at layout coordinates ``x``.

        # C++ parity: ``FdmNdimSolver<N>::setValue`` — the recursive
        # ``f[x[x.size()-N]]`` descent, whose base case is
        # ``f[x.back()] = value``. Python indexes the whole coordinate
        # tuple in one go, which is the same address.
        """
        f[tuple(x)] = value

    def _data_table(self, values: Array) -> np.ndarray:
        """Build the ``MultiCubicSpline`` data table from a flat solution vector."""
        f: np.ndarray = np.zeros(self._dim, dtype=np.float64)
        for it in self._solver_desc.mesher.layout().iter():
            FdmNdimSolver.set_value(f, it.coordinates, float(values[it.index]))
        return f

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmNdimSolver<N>::performCalculations``."""
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
        self._interp = MultiCubicSpline(self._x, self._data_table(rhs))

    def interpolate_at(self, x: Sequence[float]) -> float:
        """Solution at ``x``. # C++ parity: ``FdmNdimSolver<N>::interpolateAt``."""
        self.calculate()
        interp = self._interp
        if interp is None:
            qassert.fail("FdmNdimSolver: the backward rollback produced no interpolation")
        return interp(list(x))

    def theta_at(self, x: Sequence[float]) -> float:
        """Backward time-derivative at ``x``.

        # C++ parity: ``FdmNdimSolver<N>::thetaAt``.
        """
        if self._conditions.stopping_times()[0] == 0.0:
            return NULL_REAL

        self.calculate()
        theta_values = self._theta_condition.get_values()
        qassert.require(
            theta_values.size == self._initial_values.size,
            "FdmNdimSolver: the theta snapshot was never taken",
        )
        theta = MultiCubicSpline(self._x, self._data_table(theta_values))(list(x))
        return (theta - self.interpolate_at(x)) / self._theta_condition.get_time()


__all__ = ["FdmNdimSolver"]
