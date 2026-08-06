"""ImplicitEulerScheme — implicit-Euler time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/impliciteulerscheme.{hpp,cpp}
# (v1.43).

One step solves ``(I - theta*dt*L(t)) a_new = a_old``.

When the operator has a single direction that is the tridiagonal
``solve_splitting``. When it has more than one, C++ does **not** call
``solve_splitting(0, ...)`` — it runs a Krylov solve (BiCGstab by default,
GMRES on request) against the full ``I - theta*dt*L`` with
``map->preconditioner`` as preconditioner. The earlier Python port took the
``solve_splitting(0, ...)`` branch unconditionally, which silently solves a
different system for any 2-D or 3-D operator; that is fixed here.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.matrixutilities.bicgstab import BiCGstab
from pquantlib.math.matrixutilities.gmres import GMRES
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    FdmBoundaryConditionSet,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.methods.finitedifferences.schemes.boundary_condition_scheme_helper import (
    BoundaryConditionSchemeHelper,
)


class ImplicitEulerSchemeSolverType(IntEnum):
    """# C++ parity: ``ImplicitEulerScheme::SolverType { BiCGstab, GMRES }``."""

    BiCGstab = 0
    GMRES = 1


@final
class ImplicitEulerScheme:
    """Implicit-Euler one-step evolver.

    # C++ parity: ``class ImplicitEulerScheme``.
    """

    #: # C++ parity: the nested ``SolverType`` enum.
    SolverType = ImplicitEulerSchemeSolverType

    def __init__(
        self,
        op: FdmLinearOpComposite,
        bc_set: FdmBoundaryConditionSet = (),
        rel_tol: float = 1e-8,
        solver_type: ImplicitEulerSchemeSolverType = ImplicitEulerSchemeSolverType.BiCGstab,
    ) -> None:
        self._op: FdmLinearOpComposite = op
        self._dt: float = float("nan")
        self._rel_tol: float = rel_tol
        self._solver_type: ImplicitEulerSchemeSolverType = solver_type
        self._bc_set: BoundaryConditionSchemeHelper = BoundaryConditionSchemeHelper(bc_set)
        self._iterations: int = 0

    def set_step(self, dt: float) -> None:
        """# C++ parity: ``ImplicitEulerScheme::setStep``."""
        self._dt = dt

    def number_of_iterations(self) -> int:
        """Total Krylov iterations accumulated so far.

        # C++ parity: ``ImplicitEulerScheme::numberOfIterations``.
        """
        return self._iterations

    def _apply(self, r: Array, theta: float) -> Array:
        """# C++ parity: ``ImplicitEulerScheme::apply`` — ``r - theta*dt*map->apply(r)``."""
        return r - (theta * self._dt) * self._op.apply(r)

    def step(self, a: Array, t: float, theta: float = 1.0) -> Array:
        """Solve ``(I - theta * dt * L(t)) a_new = a`` and return ``a_new``.

        # C++ parity: ``ImplicitEulerScheme::step(a, t, theta)``. C++ mutates
        # its reference argument; the Python FD package returns the new array.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")
        t1 = max(0.0, t - self._dt)
        self._op.set_time(t1, t)
        self._bc_set.set_time(t1)

        # C++ hands the array to applyBeforeSolving by non-const reference, so
        # a boundary condition may edit it in place before the solve.
        rhs = a.copy()
        self._bc_set.apply_before_solving(self._op, rhs)

        result: Array
        if self._op.size() == 1:
            # solve_splitting solves (b*I + a*L) x = r with a = -theta*dt, b = 1,
            # i.e. x = (I - theta*dt*L)^{-1} r.
            result = self._op.solve_splitting(0, rhs, -theta * self._dt)
        else:
            n = int(rhs.shape[0])

            def preconditioner(x: Array) -> Array:
                return self._op.preconditioner(x, -theta * self._dt)

            def apply_f(x: Array) -> Array:
                return self._apply(x, theta)

            if self._solver_type == ImplicitEulerSchemeSolverType.BiCGstab:
                bicg = BiCGstab(apply_f, max(10, n), self._rel_tol, preconditioner).solve(rhs, rhs)
                self._iterations += bicg.iterations
                result = bicg.x
            elif self._solver_type == ImplicitEulerSchemeSolverType.GMRES:
                gmres = GMRES(apply_f, max(10, n // 10), self._rel_tol, preconditioner).solve(
                    rhs, rhs
                )
                self._iterations += len(gmres.errors)
                result = gmres.x
            else:
                raise LibraryException("unknown/illegal solver type")

        self._bc_set.apply_after_solving(result)
        return result


__all__ = ["ImplicitEulerScheme", "ImplicitEulerSchemeSolverType"]
