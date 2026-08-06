"""FdmG2Solver — two-factor Gaussian short-rate FD solver.

# C++ parity: ql/methods/finitedifferences/solvers/fdmg2solver.{hpp,cpp}
# (v1.43).

Builds an ``FdmG2Op`` over the two state variables (directions 0 and 1 of
the mesher) and reads the answer back through an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver.Fdm2DimSolver`.
Unlike the equity solvers there is no log transform: the mesh coordinates
*are* the model states, so ``value_at(x, y)`` passes them straight through.

The class exposes only ``value_at``; C++ declares no greeks here.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.methods.finitedifferences.operators.fdm_g2_op import FdmG2Op
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.patterns.lazy_object import LazyObject


@final
class FdmG2Solver(LazyObject):
    """G2++ FD solver on the two-factor state mesh.

    # C++ parity: ``class FdmG2Solver : public LazyObject``.
    """

    def __init__(
        self,
        model: G2,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__()
        self._model: G2 = model
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._solver: Fdm2DimSolver | None = None

        # C++ parity: ``registerWith(model_)``.
        model.register_with(self)

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmG2Solver::performCalculations`` — directions 0 and 1."""
        op = FdmG2Op(self._solver_desc.mesher, self._model, 0, 1)
        self._solver = Fdm2DimSolver(self._solver_desc, self._scheme_desc, op)

    def value_at(self, x: float, y: float) -> float:
        """# C++ parity: ``FdmG2Solver::valueAt`` — ``interpolateAt(x, y)``, no log."""
        self.calculate()
        solver = self._solver
        if solver is None:
            qassert.fail("FdmG2Solver: the inner solver was not built")
        return solver.interpolate_at(x, y)


__all__ = ["FdmG2Solver"]
