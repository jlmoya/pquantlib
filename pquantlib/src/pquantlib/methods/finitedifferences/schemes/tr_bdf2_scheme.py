"""TrBDF2Scheme — trapezoidal / BDF2 composite time stepping.

# C++ parity: ql/methods/finitedifferences/schemes/trbdf2scheme.hpp (v1.43) —
# header-only, ``template <class TrapezoidalScheme> class TrBDF2Scheme``.

One step is a trapezoidal sub-step of length ``alpha*dt`` (delegated to any
scheme with ``set_step`` / ``step``, canonically
:class:`~pquantlib.methods.finitedifferences.schemes.crank_nicolson_scheme.CrankNicolsonScheme`)
followed by one BDF2 solve of ``(I - beta*L) f_new = f`` with
``beta = (1-alpha)/(2-alpha) * dt`` and::

    f = (f*/alpha - (1-alpha)^2/alpha * f_n) / (2 - alpha)

The C++ template parameter becomes a structural :class:`TrapezoidalScheme`
protocol here; Python has no template instantiation, so the trapezoidal half
is just an object satisfying two methods.

The BDF2 solve takes the direct ``solve_splitting`` path when the operator has
a single direction, and otherwise goes through the ported
:class:`~pquantlib.math.matrixutilities.bicgstab.BiCGstab` /
:class:`~pquantlib.math.matrixutilities.gmres.GMRES` Krylov solvers with the
operator's own preconditioner — the same two branches, the same iteration
caps (``max(10, n)`` for BiCGstab, ``max(10, n // 10)`` for GMRES) and the
same iteration accounting as C++.

Naming: the C++ constructor parameter is ``map``; the Python port calls it
``op`` for consistency with the sibling schemes in this package.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Protocol, final

import numpy as np

from pquantlib import qassert
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


class TrBDF2SolverType(IntEnum):
    """Krylov solver used for the BDF2 half in the multi-direction case.

    # C++ parity: ``TrBDF2Scheme<...>::SolverType { BiCGstab, GMRES }``.
    """

    BiCGstab = 0
    GMRES = 1


class TrapezoidalScheme(Protocol):
    """What the C++ ``TrapezoidalScheme`` template parameter has to provide.

    # C++ parity: the template is instantiated in the library with
    # ``CrankNicolsonScheme``; any scheme with these two members works.
    """

    # Positional-only: C++ template instantiation never sees parameter names,
    # so a scheme must not fail to qualify merely because it spells its first
    # argument ``fn`` rather than ``a``.
    def set_step(self, dt: float, /) -> None:
        """Set the sub-step length."""
        ...

    def step(self, a: Array, t: float, /) -> Array:
        """Advance ``a`` from ``t`` by one sub-step."""
        ...


@final
class TrBDF2Scheme:
    """Trapezoidal + BDF2 one-step evolver.

    # C++ parity: ``class TrBDF2Scheme<TrapezoidalScheme>``.
    """

    SolverType = TrBDF2SolverType

    __slots__ = (
        "_alpha",
        "_bc_set",
        "_beta",
        "_dt",
        "_iterations",
        "_op",
        "_rel_tol",
        "_solver_type",
        "_trapezoidal_scheme",
    )

    def __init__(
        self,
        alpha: float,
        op: FdmLinearOpComposite,
        trapezoidal_scheme: TrapezoidalScheme,
        bc_set: FdmBoundaryConditionSet = (),
        rel_tol: float = 1e-8,
        solver_type: TrBDF2SolverType = TrBDF2SolverType.BiCGstab,
    ) -> None:
        # C++ parity: dt_/beta_ start at Null<Real>(); NaN is the Python analogue.
        self._dt: float = float("nan")
        self._beta: float = float("nan")
        # C++ holds ``ext::shared_ptr<Size>`` so copies share the counter; a
        # plain attribute is equivalent here because Python objects are never
        # copied by assignment.
        self._iterations: int = 0
        self._alpha: float = alpha
        self._op: FdmLinearOpComposite = op
        self._trapezoidal_scheme: TrapezoidalScheme = trapezoidal_scheme
        self._bc_set: BoundaryConditionSchemeHelper = BoundaryConditionSchemeHelper(bc_set)
        self._rel_tol: float = rel_tol
        self._solver_type: TrBDF2SolverType = solver_type

    def set_step(self, dt: float) -> None:
        """# C++ parity: ``TrBDF2Scheme::setStep``."""
        self._dt = dt
        self._beta = (1.0 - self._alpha) / (2.0 - self._alpha) * self._dt

    def number_of_iterations(self) -> int:
        """Total Krylov iterations accumulated so far.

        # C++ parity: ``TrBDF2Scheme::numberOfIterations``.
        """
        return self._iterations

    def _apply(self, r: Array) -> Array:
        """# C++ parity: ``TrBDF2Scheme::apply`` — ``r - beta*map->apply(r)``."""
        return r - self._beta * self._op.apply(r)

    def step(self, fn: Array, t: float) -> Array:
        """Advance ``fn`` from ``t`` to ``t - dt`` and return the new array.

        # C++ parity: ``TrBDF2Scheme::step(array_type& fn, Time t)``.
        """
        qassert.require(t - self._dt > -1e-8, "a step towards negative time given")

        intermediate_time_step = self._dt * self._alpha

        # C++ copies ``fn`` into ``fStar`` first, then mutates ``fn`` through
        # the reference via applyBeforeSolving. Python's schemes do not mutate
        # their input, so the copy here is the one C++ makes for ``fn`` itself.
        self._trapezoidal_scheme.set_step(intermediate_time_step)
        f_star = self._trapezoidal_scheme.step(fn, t)

        fn_local = np.array(fn, dtype=np.float64, copy=True)
        self._bc_set.set_time(max(0.0, t - self._dt))
        self._bc_set.apply_before_solving(self._op, fn_local)

        alpha = self._alpha
        f = (1 / alpha * f_star - (1.0 - alpha) * (1.0 - alpha) / alpha * fn_local) / (2 - alpha)

        result: Array
        if self._op.size() == 1:
            result = self._op.solve_splitting(0, f, -self._beta)
        else:
            n = int(f.shape[0])

            def preconditioner(a: Array) -> Array:
                return self._op.preconditioner(a, -self._beta)

            if self._solver_type == TrBDF2SolverType.BiCGstab:
                bicg_result = BiCGstab(self._apply, max(10, n), self._rel_tol, preconditioner).solve(
                    f, f
                )
                self._iterations += bicg_result.iterations
                result = bicg_result.x
            elif self._solver_type == TrBDF2SolverType.GMRES:
                gmres_result = GMRES(self._apply, max(10, n // 10), self._rel_tol, preconditioner).solve(
                    f, f
                )
                self._iterations += len(gmres_result.errors)
                result = gmres_result.x
            else:
                qassert.fail("unknown/illegal solver type")

        self._bc_set.apply_after_solving(result)

        return result


__all__ = ["TrBDF2Scheme", "TrBDF2SolverType", "TrapezoidalScheme"]
