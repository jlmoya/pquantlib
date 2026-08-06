"""FdmBatesOp — Bates (Heston + lognormal jumps) partial-integro operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmbatesop.{hpp,cpp}
# (v1.43).

Bates adds a compound-Poisson jump ``S -> S e^J`` with
``J ~ N(nu, delta^2)`` and intensity ``lambda`` to Heston. On the log-spot
grid the jump generator is the integral operator

.. math::

    (\\mathcal{J}u)(x) = \\lambda \\left(
        \\int_{-\\infty}^{\\infty} u(x + z)\\,\\phi_{\\nu,\\delta}(z)\\,dz - u(x)
    \\right)

which, after the substitution ``z = \\sqrt{2}\\,\\delta y + \\nu``, becomes a
Gauss-Hermite quadrature of :class:`IntegroIntegrand` scaled by
``1/\\sqrt{\\pi}``. The drift compensation ``lambda * m`` (with
``m = e^{\\nu + \\delta^2/2} - 1``) is folded into the dividend curve of
the inner :class:`~pquantlib.methods.finitedifferences.operators.fdm_heston_op.FdmHestonOp`
via a ``ZeroSpreadedTermStructure``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.integrals.gaussian_quadrature import GaussHermiteIntegration
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_heston_op import FdmHestonOp
from pquantlib.processes.bates_process import BatesProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.zero_spreaded_term_structure import (
    ZeroSpreadedTermStructure,
)
from pquantlib.time.compounding import Compounding

# C++ parity: ql/mathconstants.hpp — ``M_SQRT2`` / ``M_1_SQRTPI``. Both
# decimal literals round to the same IEEE-754 double as the computed
# expressions, so the two spellings are interchangeable.
_M_SQRT2: float = 1.41421356237309504880168872420969808
_M_1_SQRTPI: float = 0.564189583547756286948


@runtime_checkable
class _FdmQuantoHelperLike(Protocol):
    """Structural stand-in for ``FdmQuantoHelper``.

    Same contract as the Protocol of the same name in ``fdm_heston_op`` —
    ``FdmBatesOp`` only forwards the helper into its inner ``FdmHestonOp``,
    and the two Protocols are structurally interchangeable. Declared here
    rather than imported so neither module exports a private name.
    """

    def quanto_adjustment_array(self, equity_vol: Array, t1: float, t2: float) -> Array:
        """Quanto drift adjustment for a per-node equity vol vector."""
        ...


@runtime_checkable
class _FdmDirichletBoundaryLike(Protocol):
    """Structural stand-in for ``FdmDirichletBoundary``.

    # C++ parity: ql/methods/finitedifferences/utilities/fdmdirichletboundary.hpp
    # — ``Real applyAfterApplying(Real x, Real value) const``.

    ``FdmBatesOp`` requires every boundary condition in its set to be a
    ``FdmDirichletBoundary`` and calls only the *scalar* overload of
    ``applyAfterApplying``, spelled ``apply_after_applying_value`` in the
    Python port. The dependency is structural so this module does not have
    to import from ``methods.finitedifferences.utilities`` (which imports
    back from ``operators``); ``FdmDirichletBoundary`` satisfies it.
    """

    def apply_after_applying_value(self, x: float, value: float) -> float:
        """Replace ``value`` by the boundary value when ``x`` is outside."""
        ...


class IntegroIntegrand:
    """Gauss-Hermite integrand of the Bates jump integral.

    # C++ parity: ``class FdmBatesOp::IntegroIntegrand``
    # (fdmbatesop.hpp:58-69, fdmbatesop.cpp:61-85). C++ makes it a private
    # nested class; Python declares it at module scope so it can be
    # tested (and named) directly.
    """

    def __init__(
        self,
        interpl: LinearInterpolation,
        bc_set: Sequence[object],
        x: float,
        delta: float,
        nu: float,
    ) -> None:
        # C++ parity: fdmbatesop.cpp:61-66.
        self._x: float = x
        self._delta: float = delta
        self._nu: float = nu
        self._bc_set: Sequence[object] = bc_set
        self._interpl: LinearInterpolation = interpl

    def __call__(self, y: float) -> float:
        """# C++ parity: fdmbatesop.cpp:68-85."""
        x = self._x + _M_SQRT2 * self._delta * y + self._nu
        value_of_derivative = self._interpl(x, allow_extrapolation=True)

        for bc in self._bc_set:
            # C++ dynamic_pointer_casts each boundary condition to
            # FdmDirichletBoundary and QL_REQUIREs the cast succeeded.
            if not isinstance(bc, _FdmDirichletBoundaryLike):
                qassert.fail("FdmBatesOp can only deal with Dirichlet boundary conditions.")
            value_of_derivative = bc.apply_after_applying_value(x, value_of_derivative)

        return math.exp(-y * y) * value_of_derivative


class FdmBatesOp:
    """Composite Bates partial-integro-differential operator (2-D grid).

    # C++ parity: ``class FdmBatesOp : public FdmLinearOpComposite``
    # (fdmbatesop.hpp:36-115, fdmbatesop.cpp:37-124). Python satisfies
    # the ``FdmLinearOpComposite`` Protocol structurally.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        bates_process: BatesProcess,
        bc_set: Sequence[object],
        integro_integration_order: int,
        quanto_helper: _FdmQuantoHelperLike | None = None,
    ) -> None:
        # C++ parity: fdmbatesop.cpp:37-59.
        self._lambda: float = bates_process.lambda_
        self._delta: float = bates_process.delta
        self._nu: float = bates_process.nu
        self._m: float = math.exp(self._nu + 0.5 * self._delta * self._delta) - 1.0
        self._gauss_hermite_integration: GaussHermiteIntegration = GaussHermiteIntegration(
            integro_integration_order
        )
        self._mesher: FdmMesher = mesher
        self._bc_set: Sequence[object] = list(bc_set)
        self._heston_op: FdmHestonOp = FdmHestonOp(
            mesher,
            HestonProcess(
                risk_free_rate=bates_process.risk_free_rate(),
                dividend_yield=ZeroSpreadedTermStructure(
                    bates_process.dividend_yield(),
                    SimpleQuote(self._lambda * self._m),
                    Compounding.Continuous,
                ),
                s0=bates_process.s0(),
                v0=bates_process.v0,
                kappa=bates_process.kappa,
                theta=bates_process.theta,
                sigma=bates_process.sigma,
                rho=bates_process.rho,
            ),
            quanto_helper,
        )

    # --- jump integral ---------------------------------------------------

    def _integro(self, r: Array) -> Array:
        """# C++ parity: fdmbatesop.cpp:87-118."""
        layout = self._mesher.layout()
        dim = layout.dim()
        qassert.require(len(dim) == 2, "invalid layout dimension")

        x: Array = np.zeros(dim[0], dtype=np.float64)
        f: list[Array] = [np.zeros(dim[0], dtype=np.float64) for _ in range(dim[1])]

        for iter_ in layout.iter():
            i = iter_.coordinates[0]
            j = iter_.coordinates[1]
            x[i] = self._mesher.location(iter_, 0)
            f[j][i] = r[iter_.index]

        interpl = [LinearInterpolation(x, row) for row in f]

        integral: Array = np.zeros(r.size, dtype=np.float64)
        for iter_ in layout.iter():
            i = iter_.coordinates[0]
            j = iter_.coordinates[1]
            integral[iter_.index] = _M_1_SQRTPI * self._gauss_hermite_integration(
                IntegroIntegrand(interpl[j], self._bc_set, float(x[i]), self._delta, self._nu)
            )

        return self._lambda * (integral - r)

    # --- FdmLinearOpComposite surface ------------------------------------

    def size(self) -> int:
        """# C++ parity: fdmbatesop.hpp:84-86."""
        return self._heston_op.size()

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmbatesop.hpp:88-90."""
        self._heston_op.set_time(t1, t2)

    def apply(self, r: Array) -> Array:
        """# C++ parity: fdmbatesop.hpp:92-94."""
        return self._heston_op.apply(r) + self._integro(r)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: fdmbatesop.hpp:96-98."""
        return self._heston_op.apply_mixed(r) + self._integro(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: fdmbatesop.hpp:100-103."""
        return self._heston_op.apply_direction(direction, r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: fdmbatesop.hpp:105-109."""
        return self._heston_op.solve_splitting(direction, r, dt)

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: fdmbatesop.hpp:111-114."""
        return self._heston_op.preconditioner(r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: fdmbatesop.cpp:120-122 — ``QL_FAIL``."""
        qassert.fail("not implemented")


__all__ = ["FdmBatesOp", "IntegroIntegrand"]
