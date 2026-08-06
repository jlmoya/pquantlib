"""FdmBlackScholesSolver — 1-D Black-Scholes FD solver in log-spot space.

# C++ parity: ql/methods/finitedifferences/solvers/fdmblackscholessolver.{hpp,cpp}
# (v1.43).

A thin ``LazyObject`` wrapper: it builds an ``FdmBlackScholesOp`` on the
solver description's mesher, hands it to an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver.Fdm1DimSolver`,
and converts between spot and log-spot on the way in and out. The greeks
follow the C++ chain rule verbatim::

    delta(s) = V_x(log s) / s
    gamma(s) = (V_xx(log s) - V_x(log s)) / s^2

**Divergence — local volatility and quanto adjustment.** C++ passes
``localVol``, ``illegalLocalVolOverwrite`` and a ``FdmQuantoHelper`` into
``FdmBlackScholesOp``. The Python ``FdmBlackScholesOp`` (owned by
``operators/``) takes only ``(mesher, process, strike, direction)`` — its
docstring records local vol and quanto as deferred. The two arguments are
kept on this constructor for signature parity and validated: asking for
either raises rather than silently pricing the wrong thing.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import FdmBlackScholesOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import NULL_REAL, Fdm1DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdmBlackScholesSolver(LazyObject):
    """Black-Scholes FD solver on a log-spot mesh.

    # C++ parity: ``class FdmBlackScholesSolver : public LazyObject``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        strike: float,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = -NULL_REAL,
        quanto_helper: FdmQuantoHelper | None = None,
    ) -> None:
        super().__init__()
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()`` default.
        qassert.require(
            not local_vol,
            "FdmBlackScholesSolver(local_vol=True) needs the local-vol branch of "
            "FdmBlackScholesOp, which the Python operator does not implement",
        )
        qassert.require(
            quanto_helper is None,
            "FdmBlackScholesSolver(quanto_helper=...) needs the quanto branch of "
            "FdmBlackScholesOp, which the Python operator does not implement",
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._strike: float = strike
        self._solver_desc: FdmSolverDesc = solver_desc
        self._scheme_desc: FdmSchemeDesc = scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        self._local_vol: bool = local_vol
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        self._quanto_helper: FdmQuantoHelper | None = quanto_helper
        self._solver: Fdm1DimSolver | None = None

        # C++ parity: ``registerWith(process_)`` / ``registerWith(quantoHelper_)``.
        process.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmBlackScholesSolver::performCalculations``."""
        op = FdmBlackScholesOp(self._solver_desc.mesher, self._process, self._strike, 0)
        self._solver = Fdm1DimSolver(self._solver_desc, self._scheme_desc, op)

    def _inner(self) -> Fdm1DimSolver:
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmBlackScholesSolver: the inner solver was not built")
        return solver

    def value_at(self, s: float) -> float:
        """# C++ parity: ``FdmBlackScholesSolver::valueAt`` — ``interpolateAt(log s)``."""
        return self._inner().interpolate_at(math.log(s))

    def delta_at(self, s: float) -> float:
        """# C++ parity: ``FdmBlackScholesSolver::deltaAt`` — ``derivativeX(log s)/s``."""
        return self._inner().derivative_x(math.log(s)) / s

    def gamma_at(self, s: float) -> float:
        """# C++ parity: ``FdmBlackScholesSolver::gammaAt``."""
        solver = self._inner()
        x = math.log(s)
        return (solver.derivative_xx(x) - solver.derivative_x(x)) / (s * s)

    def theta_at(self, s: float) -> float:
        """# C++ parity: ``FdmBlackScholesSolver::thetaAt``.

        C++ does *not* call ``calculate()`` here (the only accessor that
        does not); it relies on a previous ``valueAt``/``deltaAt`` call
        having built the inner solver, and segfaults otherwise. The Python
        port calls ``calculate()`` — a strict improvement that cannot
        change any value C++ would have returned.
        """
        return self._inner().theta_at(math.log(s))


__all__ = ["FdmBlackScholesSolver"]
