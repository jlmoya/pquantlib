"""FdmCIRSolver — equity under a CIR stochastic short rate.

# C++ parity: ql/methods/finitedifferences/solvers/fdmcirsolver.{hpp,cpp}
# (v1.43).

The mesh is (log-spot, short rate); the operator ``FdmCIROp`` couples them
through the correlation ``rho`` and discounts at the *stochastic* rate, so
the second axis is genuinely live even for a payoff that only reads the
first. Values come back through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver.Fdm2DimSolver`.

Note the C++ argument order: the constructor takes ``(cirProcess,
bsProcess, ...)`` while the member-initialiser list assigns ``bsProcess_``
first. The Python port keeps the *constructor* order, which is the one
callers see.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_cir_op import FdmCIROp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdmCIRSolver(LazyObject):
    """Equity / CIR-short-rate FD solver.

    # C++ parity: ``class FdmCIRSolver : public LazyObject``.
    """

    def __init__(
        self,
        cir_process: CoxIngersollRossProcess,
        bs_process: GeneralizedBlackScholesProcess,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
        rho: float = 1.0,
        strike: float = 1.0,
    ) -> None:
        super().__init__()
        self._cir_process: CoxIngersollRossProcess = cir_process
        self._bs_process: GeneralizedBlackScholesProcess = bs_process
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._rho: float = rho
        self._strike: float = strike
        self._solver: Fdm2DimSolver | None = None

        # C++ parity: ``registerWith(bsProcess_)`` / ``registerWith(cirProcess_)``.
        bs_process.register_with(self)
        cir_process.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmCIRSolver::performCalculations``."""
        op = FdmCIROp(
            self._solver_desc.mesher,
            self._cir_process,
            self._bs_process,
            self._rho,
            self._strike,
        )
        self._solver = Fdm2DimSolver(self._solver_desc, self._scheme_desc, op)

    def _inner(self) -> Fdm2DimSolver:
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmCIRSolver: the inner solver was not built")
        return solver

    def value_at(self, s: float, r: float) -> float:
        """# C++ parity: ``FdmCIRSolver::valueAt`` — ``interpolateAt(log s, r)``."""
        return self._inner().interpolate_at(math.log(s), r)

    def delta_at(self, s: float, r: float) -> float:
        """# C++ parity: ``FdmCIRSolver::deltaAt``."""
        return self._inner().derivative_x(math.log(s), r) / s

    def gamma_at(self, s: float, r: float) -> float:
        """# C++ parity: ``FdmCIRSolver::gammaAt``."""
        solver = self._inner()
        x = math.log(s)
        return (solver.derivative_xx(x, r) - solver.derivative_x(x, r)) / (s * s)

    def theta_at(self, s: float, r: float) -> float:
        """# C++ parity: ``FdmCIRSolver::thetaAt``."""
        return self._inner().theta_at(math.log(s), r)


__all__ = ["FdmCIRSolver"]
