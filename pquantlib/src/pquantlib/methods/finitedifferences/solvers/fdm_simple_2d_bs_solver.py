"""FdmSimple2dBSSolver — Black-Scholes on a 2-D mesh (spot x auxiliary state).

# C++ parity: ql/methods/finitedifferences/solvers/fdmsimple2dbssolver.{hpp,cpp}
# (v1.43).

The operator is the ordinary one-direction ``FdmBlackScholesOp`` built on a
two-dimensional mesher, so the second axis carries a path-dependent state
(the running average of an Asian option, say) that the PDE does not diffuse
— it is only advanced by the step conditions. Values come back through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver.Fdm2DimSolver`.

Unlike its 1-D sibling, this class takes both greeks by *finite differencing
its own* ``value_at`` with a caller-supplied ``eps`` — C++ does the same, so
the bump size is part of the API rather than an implementation detail.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import FdmBlackScholesOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdmSimple2dBSSolver(LazyObject):
    """Black-Scholes FD solver on a 2-D mesh.

    # C++ parity: ``class FdmSimple2dBSSolver : public LazyObject``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        strike: float,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__()
        self._process: GeneralizedBlackScholesProcess = process
        self._strike: float = strike
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()`` default.
        self._scheme_desc: FdmSchemeDesc = scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        self._solver: Fdm2DimSolver | None = None

        # C++ parity: ``registerWith(process_)``.
        process.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmSimple2dBSSolver::performCalculations``."""
        op = FdmBlackScholesOp(self._solver_desc.mesher, self._process, self._strike)
        self._solver = Fdm2DimSolver(self._solver_desc, self._scheme_desc, op)

    def _inner(self) -> Fdm2DimSolver:
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmSimple2dBSSolver: the inner solver was not built")
        return solver

    def value_at(self, s: float, a: float) -> float:
        """# C++ parity: ``FdmSimple2dBSSolver::valueAt`` — ``interpolateAt(log s, log a)``."""
        return self._inner().interpolate_at(math.log(s), math.log(a))

    def delta_at(self, s: float, a: float, eps: float) -> float:
        """Central difference in ``s``. # C++ parity: ``FdmSimple2dBSSolver::deltaAt``."""
        return (self.value_at(s + eps, a) - self.value_at(s - eps, a)) / (2 * eps)

    def gamma_at(self, s: float, a: float, eps: float) -> float:
        """Second central difference in ``s``. # C++ parity: ``gammaAt``."""
        return (self.value_at(s + eps, a) + self.value_at(s - eps, a) - 2 * self.value_at(s, a)) / (eps * eps)

    def theta_at(self, s: float, a: float) -> float:
        """# C++ parity: ``FdmSimple2dBSSolver::thetaAt``."""
        return self._inner().theta_at(math.log(s), math.log(a))


__all__ = ["FdmSimple2dBSSolver"]
