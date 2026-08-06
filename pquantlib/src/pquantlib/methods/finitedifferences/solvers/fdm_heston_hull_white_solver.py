"""FdmHestonHullWhiteSolver — Heston equity with a Hull-White short rate.

# C++ parity: ql/methods/finitedifferences/solvers/fdmhestonhullwhitesolver.{hpp,cpp}
# (v1.43).

Three state variables — log-spot, variance, short rate — on an
``FdmHestonHullWhiteOp``, read back through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_3dim_solver.Fdm3DimSolver`.

Unlike the 2-D equity solvers there are no analytic greeks: C++ finite-
differences its own ``valueAt`` with a caller-supplied bump ``eps``, so the
bump size is part of the API. As on
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_heston_solver.FdmHestonSolver`
these are spot greeks, not the model-implied ones.

**Divergence — ``HullWhiteProcess``.** C++ takes a
``Handle<HullWhiteProcess>``. That process is not ported; the operator
module already expresses the dependency structurally (it needs only ``a()``
and ``sigma()``), and this module does the same, so
``HullWhiteForwardProcess`` — or any object with those two accessors —
satisfies it.
"""

from __future__ import annotations

import math
from typing import Protocol, final, runtime_checkable

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_heston_hull_white_op import (
    FdmHestonHullWhiteOp,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_3dim_solver import Fdm3DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.heston_process import HestonProcess


@runtime_checkable
class HullWhiteProcessLike(Protocol):
    """Structural stand-in for C++ ``HullWhiteProcess``.

    # C++ parity: ql/processes/hullwhiteprocess.hpp — ``a()`` and ``sigma()``
    # are the only members ``FdmHestonHullWhiteOp`` reads.
    """

    def a(self) -> float:
        """Mean-reversion speed."""
        ...

    def sigma(self) -> float:
        """Short-rate volatility."""
        ...


@final
class FdmHestonHullWhiteSolver(LazyObject):
    """Heston / Hull-White FD solver on a 3-D mesh.

    # C++ parity: ``class FdmHestonHullWhiteSolver : public LazyObject``.
    """

    def __init__(
        self,
        heston_process: HestonProcess,
        hw_process: HullWhiteProcessLike,
        corr_equity_short_rate: float,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__()
        self._heston_process: HestonProcess = heston_process
        self._hw_process: HullWhiteProcessLike = hw_process
        self._corr_equity_short_rate: float = corr_equity_short_rate
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._solver: Fdm3DimSolver | None = None

        # C++ parity: ``registerWith(hestonProcess)`` / ``registerWith(hwProcess)``.
        heston_process.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmHestonHullWhiteSolver::performCalculations``."""
        op = FdmHestonHullWhiteOp(
            self._solver_desc.mesher,
            self._heston_process,
            self._hw_process,
            self._corr_equity_short_rate,
        )
        self._solver = Fdm3DimSolver(self._solver_desc, self._scheme_desc, op)

    def _inner(self) -> Fdm3DimSolver:
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmHestonHullWhiteSolver: the inner solver was not built")
        return solver

    def value_at(self, s: float, v: float, r: float) -> float:
        """# C++ parity: ``FdmHestonHullWhiteSolver::valueAt`` — ``interpolateAt(log s, v, r)``."""
        return self._inner().interpolate_at(math.log(s), v, r)

    def theta_at(self, s: float, v: float, r: float) -> float:
        """# C++ parity: ``FdmHestonHullWhiteSolver::thetaAt``."""
        return self._inner().theta_at(math.log(s), v, r)

    def delta_at(self, s: float, v: float, r: float, eps: float) -> float:
        """Central difference in ``s``. # C++ parity: ``deltaAt``."""
        return (self.value_at(s + eps, v, r) - self.value_at(s - eps, v, r)) / (2 * eps)

    def gamma_at(self, s: float, v: float, r: float, eps: float) -> float:
        """Second central difference in ``s``. # C++ parity: ``gammaAt``."""
        return (self.value_at(s + eps, v, r) + self.value_at(s - eps, v, r) - 2 * self.value_at(s, v, r)) / (
            eps * eps
        )


__all__ = ["FdmHestonHullWhiteSolver", "HullWhiteProcessLike"]
