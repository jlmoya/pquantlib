"""FdmHestonSolver — Heston stochastic-volatility FD solver.

# C++ parity: ql/methods/finitedifferences/solvers/fdmhestonsolver.{hpp,cpp}
# (v1.43).

Builds an ``FdmHestonOp`` on the (log-spot, variance) mesh and reads the
answer back through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver.Fdm2DimSolver`.

Beyond spot delta/gamma the class exposes the *mean-variance* greeks, which
add the ``dV/dv`` sensitivity weighted by ``rho*sigma/s`` — the C++ header
points at Mercurio & Morini, "A Note on Hedging with Local and Stochastic
Volatility Models", for why the plain ``dV/ds`` is not the model-implied
delta.

``leverage_fct`` (the local-volatility multiplier of the Heston-SLV model)
and ``mixing_factor`` are forwarded to the operator unchanged.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_heston_op import FdmHestonOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)


@final
class FdmHestonSolver(LazyObject):
    """Heston FD solver on a (log-spot, variance) mesh.

    # C++ parity: ``class FdmHestonSolver : public LazyObject``.
    """

    def __init__(
        self,
        process: HestonProcess,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
        quanto_helper: FdmQuantoHelper | None = None,
        leverage_fct: LocalVolTermStructure | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self._process: HestonProcess = process
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._quanto_helper: FdmQuantoHelper | None = quanto_helper
        self._leverage_fct: LocalVolTermStructure | None = leverage_fct
        self._mixing_factor: float = mixing_factor
        self._solver: Fdm2DimSolver | None = None

        # C++ parity: ``registerWith(process_)`` / ``registerWith(quantoHelper_)``.
        process.register_with(self)
        if quanto_helper is not None:
            quanto_helper.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmHestonSolver::performCalculations``."""
        op = FdmHestonOp(
            self._solver_desc.mesher,
            self._process,
            self._quanto_helper,
            self._leverage_fct,
            self._mixing_factor,
        )
        self._solver = Fdm2DimSolver(self._solver_desc, self._scheme_desc, op)

    def _inner(self) -> Fdm2DimSolver:
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmHestonSolver: the inner solver was not built")
        return solver

    def value_at(self, s: float, v: float) -> float:
        """# C++ parity: ``FdmHestonSolver::valueAt`` — ``interpolateAt(log s, v)``."""
        return self._inner().interpolate_at(math.log(s), v)

    def theta_at(self, s: float, v: float) -> float:
        """# C++ parity: ``FdmHestonSolver::thetaAt``."""
        return self._inner().theta_at(math.log(s), v)

    def delta_at(self, s: float, v: float) -> float:
        """Spot delta — *not* the model-implied delta.

        # C++ parity: ``FdmHestonSolver::deltaAt``.
        """
        return self._inner().derivative_x(math.log(s), v) / s

    def gamma_at(self, s: float, v: float) -> float:
        """Spot gamma — *not* the model-implied gamma.

        # C++ parity: ``FdmHestonSolver::gammaAt``.
        """
        solver = self._inner()
        x = math.log(s)
        return (solver.derivative_xx(x, v) - solver.derivative_x(x, v)) / (s * s)

    def mean_variance_delta_at(self, s: float, v: float) -> float:
        """# C++ parity: ``FdmHestonSolver::meanVarianceDeltaAt``."""
        solver = self._inner()
        alpha = self._process.rho * self._process.sigma / s
        return self.delta_at(s, v) + alpha * solver.derivative_y(math.log(s), v)

    def mean_variance_gamma_at(self, s: float, v: float) -> float:
        """# C++ parity: ``FdmHestonSolver::meanVarianceGammaAt``."""
        solver = self._inner()
        x = math.log(s)
        alpha = self._process.rho * self._process.sigma / s
        return (
            self.gamma_at(s, v)
            + solver.derivative_yy(x, v) * alpha * alpha
            + 2 * solver.derivative_xy(x, v) * alpha / s
        )


__all__ = ["FdmHestonSolver"]
