"""FdmHullWhiteSolver — one-factor Hull-White short-rate FD solver.

# C++ parity: ql/methods/finitedifferences/solvers/fdmhullwhitesolver.{hpp,cpp}
# (v1.43).

Builds an ``FdmHullWhiteOp`` on direction 0 of the mesher and reads the
answer back through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver.Fdm1DimSolver`.
The mesh coordinate is the model state, so ``value_at(r)`` passes it
through with no transform.

The default scheme is Hundsdorfer even though the operator has a single
direction — C++ declares it that way, and for a one-direction operator the
ADI splitting collapses onto the plain theta scheme anyway.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_hull_white_op import FdmHullWhiteOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import Fdm1DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.lazy_object import LazyObject


@final
class FdmHullWhiteSolver(LazyObject):
    """Hull-White FD solver on the short-rate mesh.

    # C++ parity: ``class FdmHullWhiteSolver : public LazyObject``.
    """

    def __init__(
        self,
        model: HullWhite,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__()
        self._model: HullWhite = model
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._solver: Fdm1DimSolver | None = None

        # C++ parity: ``registerWith(model_)``.
        model.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmHullWhiteSolver::performCalculations`` — direction 0."""
        op = FdmHullWhiteOp(self._solver_desc.mesher, self._model, 0)
        self._solver = Fdm1DimSolver(self._solver_desc, self._scheme_desc, op)

    def value_at(self, r: float) -> float:
        """# C++ parity: ``FdmHullWhiteSolver::valueAt`` — ``interpolateAt(r)``."""
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmHullWhiteSolver: the inner solver was not built")
        return solver.interpolate_at(r)


__all__ = ["FdmHullWhiteSolver"]
