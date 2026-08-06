"""Fdm2dBlackScholesSolver — two correlated Black-Scholes assets.

# C++ parity: ql/methods/finitedifferences/solvers/fdm2dblackscholessolver.{hpp,cpp}
# (v1.43).

Builds an ``Fdm2dBlackScholesOp`` (two log-spot directions plus the
correlation cross term) on the description's mesher and reads the answer
back through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver.Fdm2DimSolver`.

All accessors take *spot* coordinates and convert to log space; the greeks
are the C++ chain-rule expressions, including the cross gamma
``V_xy/(u v)``. The default scheme is Hundsdorfer, matching C++ — this is a
genuinely two-directional operator, so an ADI scheme is the natural choice.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_2d_black_scholes_op import (
    Fdm2dBlackScholesOp,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import NULL_REAL
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class Fdm2dBlackScholesSolver(LazyObject):
    """Two-asset Black-Scholes FD solver.

    # C++ parity: ``class Fdm2dBlackScholesSolver : public LazyObject``.
    """

    def __init__(
        self,
        p1: GeneralizedBlackScholesProcess,
        p2: GeneralizedBlackScholesProcess,
        correlation: float,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = -NULL_REAL,
    ) -> None:
        super().__init__()
        self._p1: GeneralizedBlackScholesProcess = p1
        self._p2: GeneralizedBlackScholesProcess = p2
        self._correlation: float = correlation
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._local_vol: bool = local_vol
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        self._solver: Fdm2DimSolver | None = None

        # C++ parity: ``registerWith(p1_)`` / ``registerWith(p2_)``.
        p1.register_with(self)
        p2.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``Fdm2dBlackScholesSolver::performCalculations``."""
        op = Fdm2dBlackScholesOp(
            self._solver_desc.mesher,
            self._p1,
            self._p2,
            self._correlation,
            self._solver_desc.maturity,
            self._local_vol,
            self._illegal_local_vol_overwrite,
        )
        self._solver = Fdm2DimSolver(self._solver_desc, self._scheme_desc, op)

    def _inner(self) -> Fdm2DimSolver:
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("Fdm2dBlackScholesSolver: the inner solver was not built")
        return solver

    def value_at(self, u: float, v: float) -> float:
        """# C++ parity: ``Fdm2dBlackScholesSolver::valueAt``."""
        return self._inner().interpolate_at(math.log(u), math.log(v))

    def theta_at(self, u: float, v: float) -> float:
        """# C++ parity: ``Fdm2dBlackScholesSolver::thetaAt``."""
        return self._inner().theta_at(math.log(u), math.log(v))

    def delta_x_at(self, u: float, v: float) -> float:
        """# C++ parity: ``Fdm2dBlackScholesSolver::deltaXat`` — ``V_x/u``."""
        return self._inner().derivative_x(math.log(u), math.log(v)) / u

    def delta_y_at(self, u: float, v: float) -> float:
        """# C++ parity: ``Fdm2dBlackScholesSolver::deltaYat`` — ``V_y/v``."""
        return self._inner().derivative_y(math.log(u), math.log(v)) / v

    def gamma_x_at(self, u: float, v: float) -> float:
        """# C++ parity: ``Fdm2dBlackScholesSolver::gammaXat``."""
        solver = self._inner()
        x, y = math.log(u), math.log(v)
        return (solver.derivative_xx(x, y) - solver.derivative_x(x, y)) / (u * u)

    def gamma_y_at(self, u: float, v: float) -> float:
        """# C++ parity: ``Fdm2dBlackScholesSolver::gammaYat``."""
        solver = self._inner()
        x, y = math.log(u), math.log(v)
        return (solver.derivative_yy(x, y) - solver.derivative_y(x, y)) / (v * v)

    def gamma_xy_at(self, u: float, v: float) -> float:
        """# C++ parity: ``Fdm2dBlackScholesSolver::gammaXYat`` — ``V_xy/(u v)``."""
        return self._inner().derivative_xy(math.log(u), math.log(v)) / (u * v)


__all__ = ["Fdm2dBlackScholesSolver"]
