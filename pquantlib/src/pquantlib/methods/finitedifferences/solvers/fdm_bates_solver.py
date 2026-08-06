"""FdmBatesSolver — Bates (Heston + jumps) FD solver.

# C++ parity: ql/methods/finitedifferences/solvers/fdmbatessolver.{hpp,cpp}
# (v1.43).

Same shape as
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_heston_solver.FdmHestonSolver`
— a 2-D (log-spot, variance) mesh read through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver.Fdm2DimSolver`
— but the operator is an ``FdmBatesOp``, whose extra integro term
(a Gauss-Hermite quadrature of the jump distribution over the log-spot
axis) makes the PIDE non-local. ``integro_integration_order`` is that
quadrature's order and defaults to 12, as in C++.

``solver_desc.bc_set`` goes into ``FdmBatesOp`` as well as into the inner
``Fdm2DimSolver``, because the Bates integrand re-applies each Dirichlet
boundary to the interpolated jump tail — exactly as C++ does.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_bates_op import FdmBatesOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.bates_process import BatesProcess


@final
class FdmBatesSolver(LazyObject):
    """Bates FD solver on a (log-spot, variance) mesh.

    # C++ parity: ``class FdmBatesSolver : public LazyObject``.
    """

    def __init__(
        self,
        process: BatesProcess,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
        integro_integration_order: int = 12,
        quanto_helper: FdmQuantoHelper | None = None,
    ) -> None:
        super().__init__()
        self._process: BatesProcess = process
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._integro_integration_order: int = integro_integration_order
        self._quanto_helper: FdmQuantoHelper | None = quanto_helper
        self._solver: Fdm2DimSolver | None = None

        # C++ parity: ``registerWith(process_)`` / ``registerWith(quantoHelper_)``.
        process.register_with(self)
        if quanto_helper is not None:
            quanto_helper.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmBatesSolver::performCalculations``."""
        op = FdmBatesOp(
            self._solver_desc.mesher,
            self._process,
            self._solver_desc.bc_set,  # C++ parity: ``solverDesc_.bcSet``.
            self._integro_integration_order,
            self._quanto_helper,
        )
        self._solver = Fdm2DimSolver(self._solver_desc, self._scheme_desc, op)

    def _inner(self) -> Fdm2DimSolver:
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmBatesSolver: the inner solver was not built")
        return solver

    def value_at(self, s: float, v: float) -> float:
        """# C++ parity: ``FdmBatesSolver::valueAt`` — ``interpolateAt(log s, v)``."""
        return self._inner().interpolate_at(math.log(s), v)

    def theta_at(self, s: float, v: float) -> float:
        """# C++ parity: ``FdmBatesSolver::thetaAt``."""
        return self._inner().theta_at(math.log(s), v)

    def delta_at(self, s: float, v: float) -> float:
        """Spot delta — *not* the model-implied delta.

        # C++ parity: ``FdmBatesSolver::deltaAt``.
        """
        return self._inner().derivative_x(math.log(s), v) / s

    def gamma_at(self, s: float, v: float) -> float:
        """Spot gamma — *not* the model-implied gamma.

        # C++ parity: ``FdmBatesSolver::gammaAt``.
        """
        solver = self._inner()
        x = math.log(s)
        return (solver.derivative_xx(x, v) - solver.derivative_x(x, v)) / (s * s)


__all__ = ["FdmBatesSolver"]
