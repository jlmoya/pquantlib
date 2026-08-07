"""FdmKlugeExtOUSolver — solver wrapper for the FdmKlugeExtOUOp.

# C++ parity: ql/experimental/finitedifferences/fdmklugeextousolver.hpp
# (v1.43) — header-only ``template <Size N=3> class FdmKlugeExtOUSolver``.

A lazy-object wrapper around :class:`FdmNdimSolver`: it builds an
:class:`FdmKlugeExtOUOp` over the description's mesher and hands it to the
generic n-dimensional backward solver, whose tensor cubic spline answers
``value_at(x)``.

The integro-integration order is hardcoded to 16 in C++
(fdmklugeextousolver.hpp:63) — note this differs from the 32 that
:class:`FdmExtOUJumpSolver` passes — and is not a constructor parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.fdm_kluge_ext_ou_op import (
    FdmKlugeExtOUOp,
)
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_ndim_solver import FdmNdimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

#: # C++ parity: the literal 16 at fdmklugeextousolver.hpp:63.
_INTEGRO_INTEGRATION_ORDER = 16


@final
class FdmKlugeExtOUSolver(LazyObject):
    """Backward FD solver for the correlated Kluge + ExtOU op.

    # C++ parity: ``template <Size N=3> class FdmKlugeExtOUSolver``.

    Python has no template parameter, so ``n`` is an ordinary argument
    defaulting to 3 and reproducing the C++ ``BOOST_STATIC_ASSERT(N >= 3)``
    — the operator writes into three directions and cannot act on a
    lower-rank mesh.
    """

    def __init__(
        self,
        kluge_ext_ou_process: KlugeExtOUProcess,
        r_ts: YieldTermStructure,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc | None = None,
        n: int = 3,
    ) -> None:
        super().__init__()
        qassert.require(n >= 3, f"KlugeExtOU solver requires N >= 3, got {n}")
        self._process: KlugeExtOUProcess = kluge_ext_ou_process
        self._r_ts: YieldTermStructure = r_ts
        self._solver_desc: FdmSolverDesc = solver_desc
        # C++ default argument: FdmSchemeDesc::Hundsdorfer().
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._n: int = n
        self._solver: FdmNdimSolver | None = None
        # C++ parity divergence: the C++ ctor does registerWith(klugeOUProcess_).
        # The Python KlugeExtOUProcess is a plain value object, not an
        # Observable, so there is nothing to observe. See FdmExtOUJumpSolver.

    def _perform_calculations(self) -> None:
        """# C++ parity: ``FdmKlugeExtOUSolver::performCalculations``."""
        op = FdmKlugeExtOUOp(
            self._solver_desc.mesher,
            self._process,
            self._r_ts,
            _INTEGRO_INTEGRATION_ORDER,
        )
        self._solver = FdmNdimSolver(
            self._solver_desc, self._scheme_desc, op, self._n
        )

    def value_at(self, x: Sequence[float]) -> float:
        """Interpolate the rolled-back value at the given multi-D state.

        # C++ parity: ``FdmKlugeExtOUSolver::valueAt(const std::vector<Real>&)``.
        """
        self.calculate()
        assert self._solver is not None
        return self._solver.interpolate_at(x)


__all__ = ["FdmKlugeExtOUSolver"]
