"""FdmExtOUJumpSolver — solver wrapper for the FdmExtOUJumpOp.

# C++ parity: ql/experimental/finitedifferences/fdmextoujumpsolver.{hpp,cpp}
# (v1.43).

A lazy-object wrapper around :class:`Fdm2DimSolver`: it builds an
:class:`FdmExtOUJumpOp` over the description's mesher and hands it to the
generic 2-D backward solver, whose bicubic spline answers
``value_at(x, y)``.

The integro-integration order is hardcoded to 32 in C++
(fdmextoujumpsolver.cpp:44) and is not a constructor parameter.
"""

from __future__ import annotations

from typing import final

from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_op import FdmExtOUJumpOp
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

#: # C++ parity: the literal 32 at fdmextoujumpsolver.cpp:44.
_INTEGRO_INTEGRATION_ORDER = 32


@final
class FdmExtOUJumpSolver(LazyObject):
    """Backward FD solver for the ExtOU + jump op.

    # C++ parity: ``class FdmExtOUJumpSolver : public LazyObject``.
    """

    def __init__(
        self,
        process: ExtOUWithJumpsProcess,
        r_ts: YieldTermStructure,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__()
        self._process: ExtOUWithJumpsProcess = process
        self._r_ts: YieldTermStructure = r_ts
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ default argument: FdmSchemeDesc::Hundsdorfer().
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._solver: Fdm2DimSolver | None = None
        # C++ parity divergence: the C++ ctor does registerWith(process_). The
        # Python ExtOUWithJumpsProcess is a plain value object, not an
        # Observable (it has no register_with), so there is nothing to observe
        # and no notification to forward. The process is immutable once built,
        # so no cache can go stale.

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmExtOUJumpSolver::performCalculations``."""
        op = FdmExtOUJumpOp(
            self._solver_desc.mesher,
            self._process,
            self._r_ts,
            _INTEGRO_INTEGRATION_ORDER,
        )
        self._solver = Fdm2DimSolver(self._solver_desc, self._scheme_desc, op)

    def value_at(self, x: float, y: float) -> float:
        """Interpolate the rolled-back value at ``(x, y)``.

        # C++ parity: ``FdmExtOUJumpSolver::valueAt``.
        """
        self.calculate()
        assert self._solver is not None
        return self._solver.interpolate_at(x, y)


__all__ = ["FdmExtOUJumpSolver"]
