"""FdmSquareRootFwdOp — Fokker-Planck operator for a square-root process.

# C++ parity: ql/methods/finitedifferences/operators/fdmsquarerootfwdop.{hpp,cpp}
# (v1.43).

Forward (Fokker-Planck) operator for

.. math::

    dv_t = \\kappa(\\theta - v_t)\\,dt + \\sigma\\sqrt{v_t}\\,dW_t

in one of three coordinate transformations:

``Plain``
    the variance ``v`` itself,
``Power``
    the ``2 \\kappa \\theta / \\sigma^2`` power transform,
``Log``
    ``log v``.

The interior stencil comes from the usual first/second derivative
operators; the two boundary rows are then **overwritten** with a
zero-flux discretisation (``setLowerBC`` / ``setUpperBC``), which is
what ``ModTripleBandLinearOp``'s per-entry setters exist for.

The helper family ``v(i) / h(i) / zeta*(i) / mu(i)`` uses C++'s
one-based, ghost-node-extended indexing: ``v(0)`` and ``v(n+1)`` are
extrapolated ghost values outside the grid, ``v(1) .. v(n)`` are the
real nodes.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import cast, final

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.mod_triple_band_linear_op import (
    ModTripleBandLinearOp,
)
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)


class TransformationType(IntEnum):
    """Coordinate transform of the square-root variable.

    # C++ parity: ``FdmSquareRootFwdOp::TransformationType`` — a nested
    # enum in C++; PQuantLib spells nested C++ enums as module-level
    # ``IntEnum``.
    """

    Plain = 0
    Power = 1
    Log = 2


@final
class FdmSquareRootFwdOp:
    """Square-root-process Fokker-Planck operator.

    # C++ parity: ``class FdmSquareRootFwdOp : public FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        kappa: float,
        theta: float,
        sigma: float,
        direction: int,
        transform: TransformationType = TransformationType.Plain,
    ) -> None:
        self._direction: int = direction
        self._kappa: float = kappa
        self._theta: float = theta
        self._sigma: float = sigma
        self._transform: TransformationType = transform

        layout = mesher.layout()
        size = layout.size()
        loc = mesher.locations(direction)

        base: TripleBandLinearOp
        if transform == TransformationType.Plain:
            base = (
                FirstDerivativeOp(direction, mesher)
                .mult(kappa * (loc - theta) + sigma * sigma)
                .add(
                    SecondDerivativeOp(direction, mesher).mult(
                        0.5 * sigma * sigma * loc
                    )
                )
                .add(np.full(size, kappa, dtype=np.float64))
            )
        elif transform == TransformationType.Power:
            base = (
                SecondDerivativeOp(direction, mesher)
                .mult(0.5 * sigma * sigma * loc)
                .add(
                    FirstDerivativeOp(direction, mesher).mult(kappa * (loc + theta))
                )
                .add(
                    np.full(
                        size,
                        2 * kappa * kappa * theta / (sigma * sigma),
                        dtype=np.float64,
                    )
                )
            )
        else:
            exp_neg = np.exp(-loc)
            base = (
                FirstDerivativeOp(direction, mesher)
                .mult(exp_neg * (-0.5 * sigma * sigma - kappa * theta) + kappa)
                .add(
                    SecondDerivativeOp(direction, mesher).mult(
                        0.5 * sigma * sigma * exp_neg
                    )
                )
                .add(kappa * theta * exp_neg)
            )
        self._map_x: ModTripleBandLinearOp = ModTripleBandLinearOp(op=base)

        self._v_grid: Array = np.zeros(layout.dim()[direction], dtype=np.float64)
        for iter_ in layout.iter():
            self._v_grid[iter_.coordinates[direction]] = mesher.location(
                iter_, direction
            )

        # zero flux boundary condition
        self._set_lower_bc(mesher)
        self._set_upper_bc(mesher)

    # --- grid helpers (C++ one-based, ghost-extended indexing) ----------

    def v(self, i: int) -> float:
        """Node position, with ghost nodes at ``i = 0`` and ``i = n+1``.

        # C++ parity: ``FdmSquareRootFwdOp::v``.
        """
        n = self._v_grid.shape[0]
        if 0 < i <= n:
            return float(self._v_grid[i - 1])
        if i == 0:
            if self._transform == TransformationType.Log:
                return 2 * float(self._v_grid[0]) - float(self._v_grid[1])
            return max(
                0.5 * float(self._v_grid[0]),
                float(self._v_grid[0])
                - 0.01 * (float(self._v_grid[1]) - float(self._v_grid[0])),
            )
        if i == n + 1:
            return float(self._v_grid[-1]) + (
                float(self._v_grid[-1]) - float(self._v_grid[-2])
            )
        return qassert.fail("unknown index")

    def _h(self, i: int) -> float:
        """# C++ parity: ``h(i) = v(i+1) - v(i)``."""
        return self.v(i + 1) - self.v(i)

    def _mu(self, i: int) -> float:
        """# C++ parity: ``mu(i) = kappa*(v(i) - theta) + sigma^2``."""
        return self._kappa * (self.v(i) - self._theta) + self._sigma * self._sigma

    def _zetam(self, i: int) -> float:
        """# C++ parity: ``zetam(i) = h(i-1)*(h(i-1)+h(i))``."""
        return self._h(i - 1) * (self._h(i - 1) + self._h(i))

    def _zeta(self, i: int) -> float:
        """# C++ parity: ``zeta(i) = h(i-1)*h(i)``."""
        return self._h(i - 1) * self._h(i)

    def _zetap(self, i: int) -> float:
        """# C++ parity: ``zetap(i) = h(i)*(h(i-1)+h(i))``."""
        return self._h(i) * (self._h(i - 1) + self._h(i))

    # --- interior stencil coefficients ----------------------------------

    def _get_coeff(self, n: int) -> tuple[float, float, float]:
        """# C++ parity: ``getCoeff`` — dispatch on the transform."""
        if self._transform == TransformationType.Plain:
            return self._get_coeff_plain(n)
        if self._transform == TransformationType.Power:
            return self._get_coeff_power(n)
        return self._get_coeff_log(n)

    def _get_coeff_plain(self, n: int) -> tuple[float, float, float]:
        """# C++ parity: ``getCoeffPlain``."""
        s2 = self._sigma * self._sigma
        alpha = s2 * self.v(n) / self._zetam(n) - self._mu(n) * self._h(n) / self._zetam(n)
        beta = (
            -s2 * self.v(n) / self._zeta(n)
            + self._mu(n) * (self._h(n) - self._h(n - 1)) / self._zeta(n)
            + self._kappa
        )
        gamma = s2 * self.v(n) / self._zetap(n) + self._mu(n) * self._h(n - 1) / self._zetap(n)
        return alpha, beta, gamma

    def _get_coeff_log(self, n: int) -> tuple[float, float, float]:
        """# C++ parity: ``getCoeffLog``."""
        s2 = self._sigma * self._sigma
        mu = (-self._kappa * self._theta - s2 / 2.0) * math.exp(-self.v(n)) + self._kappa
        e = math.exp(-self.v(n))
        alpha = s2 * e / self._zetam(n) - mu * self._h(n) / self._zetam(n)
        beta = (
            -s2 * e / self._zeta(n)
            + mu * (self._h(n) - self._h(n - 1)) / self._zeta(n)
            + self._kappa * self._theta * e
        )
        gamma = s2 * e / self._zetap(n) + mu * self._h(n - 1) / self._zetap(n)
        return alpha, beta, gamma

    def _get_coeff_power(self, n: int) -> tuple[float, float, float]:
        """# C++ parity: ``getCoeffPower``."""
        s2 = self._sigma * self._sigma
        mu = self._kappa * (self._theta + self.v(n))
        alpha = (s2 * self.v(n) - mu * self._h(n)) / self._zetam(n)
        beta = (-s2 * self.v(n) + mu * (self._h(n) - self._h(n - 1))) / self._zeta(
            n
        ) + 2 * self._kappa * self._kappa * self._theta / s2
        gamma = (s2 * self.v(n) + mu * self._h(n - 1)) / self._zetap(n)
        return alpha, beta, gamma

    # --- boundary factors -----------------------------------------------

    def lower_boundary_factor(
        self, transform: TransformationType = TransformationType.Plain
    ) -> float:
        """# C++ parity: ``FdmSquareRootFwdOp::lowerBoundaryFactor``."""
        if transform == TransformationType.Plain:
            return self._f0_plain()
        if transform == TransformationType.Power:
            return self._f0_power()
        if transform == TransformationType.Log:
            return self._f0_log()
        return qassert.fail("unknown transform")

    def upper_boundary_factor(
        self, transform: TransformationType = TransformationType.Plain
    ) -> float:
        """# C++ parity: ``FdmSquareRootFwdOp::upperBoundaryFactor``."""
        if transform == TransformationType.Plain:
            return self._f1_plain()
        if transform == TransformationType.Power:
            return self._f1_power()
        if transform == TransformationType.Log:
            return self._f1_log()
        return qassert.fail("unknown transform")

    def _f0_plain(self) -> float:
        """# C++ parity: ``f0Plain``."""
        n = 1
        s2 = self._sigma * self._sigma
        a = -(2 * self._h(n - 1) + self._h(n)) / self._zetam(n)
        alpha = s2 * self.v(n) / self._zetam(n) - self._mu(n) * self._h(n) / self._zetam(n)
        nu = a * self.v(n - 1) + (2 * self._kappa * (self.v(n - 1) - self._theta) + s2) / s2
        return alpha / nu * self.v(n - 1)

    def _f1_plain(self) -> float:
        """# C++ parity: ``f1Plain``."""
        n = self._v_grid.shape[0]
        s2 = self._sigma * self._sigma
        a = (2 * self._h(n) + self._h(n - 1)) / self._zetap(n)
        gamma = s2 * self.v(n) / self._zetap(n) + self._mu(n) * self._h(n - 1) / self._zetap(n)
        nu = a * self.v(n + 1) + (2 * self._kappa * (self.v(n + 1) - self._theta) + s2) / s2
        return gamma / nu * self.v(n + 1)

    def _f0_power(self) -> float:
        """# C++ parity: ``f0Power``."""
        n = 1
        s2 = self._sigma * self._sigma
        mu = self._kappa * (self.v(n) + self._theta)
        a = -(2 * self._h(n - 1) + self._h(n)) / self._zetam(n)
        alpha = s2 * self.v(n) / self._zetam(n) - mu * self._h(n) / self._zetam(n)
        nu = a * self.v(n - 1) + 2 * (self._kappa * self.v(n - 1) / s2)
        return alpha / nu * self.v(n - 1)

    def _f1_power(self) -> float:
        """# C++ parity: ``f1Power``."""
        n = self._v_grid.shape[0]
        s2 = self._sigma * self._sigma
        mu = self._kappa * (self.v(n) + self._theta)
        a = (2 * self._h(n) + self._h(n - 1)) / self._zetap(n)
        gamma = s2 * self.v(n) / self._zetap(n) + mu * self._h(n - 1) / self._zetap(n)
        nu = a * self.v(n + 1) + 2 * (self._kappa * self.v(n + 1) / s2)
        return gamma / nu * self.v(n + 1)

    def _f0_log(self) -> float:
        """# C++ parity: ``f0Log``."""
        n = 1
        s2 = self._sigma * self._sigma
        mu = (-self._kappa * self._theta - s2 / 2.0) * math.exp(-self.v(1)) + self._kappa
        a = -(2 * self._h(n - 1) + self._h(n)) / self._zetam(n)
        alpha = s2 * math.exp(-self.v(n)) / self._zetam(n) - mu * self._h(n) / self._zetam(n)
        nu = a * math.exp(-self.v(n - 1)) + 2 * self._kappa * (
            1 - self._theta * math.exp(-self.v(n - 1))
        ) / s2
        return alpha / nu * math.exp(-self.v(n - 1))

    def _f1_log(self) -> float:
        """# C++ parity: ``f1Log``."""
        n = self._v_grid.shape[0]
        s2 = self._sigma * self._sigma
        mu = (-self._kappa * self._theta - s2 / 2.0) * math.exp(-self.v(n)) + self._kappa
        a = (2 * self._h(n) + self._h(n - 1)) / self._zetap(n)
        gamma = s2 * math.exp(-self.v(n)) / self._zetap(n) + mu * self._h(n - 1) / self._zetap(n)
        nu = a * math.exp(-self.v(n + 1)) + 2 * self._kappa * (
            1 - self._theta * math.exp(-self.v(n + 1))
        ) / s2
        return gamma / nu * math.exp(-self.v(n + 1))

    # --- zero-flux boundary rows ----------------------------------------

    def _set_lower_bc(self, mesher: FdmMesher) -> None:
        """# C++ parity: ``FdmSquareRootFwdOp::setLowerBC``."""
        n = 1
        _alpha, beta, gamma = self._get_coeff(n)
        f = self.lower_boundary_factor(self._transform)

        b = -(self._h(n - 1) + self._h(n)) / self._zeta(n)
        c = self._h(n - 1) / self._zetap(n)

        for iter_ in mesher.layout().iter():
            if iter_.coordinates[self._direction] == 0:
                idx = iter_.index
                self._map_x.set_diag(idx, beta + f * b)
                self._map_x.set_upper(idx, gamma + f * c)

    def _set_upper_bc(self, mesher: FdmMesher) -> None:
        """# C++ parity: ``FdmSquareRootFwdOp::setUpperBC``."""
        n = self._v_grid.shape[0]
        alpha, beta, _gamma = self._get_coeff(n)
        f = self.upper_boundary_factor(self._transform)

        b = (self._h(n) + self._h(n - 1)) / self._zeta(n)
        c = -self._h(n) / self._zetam(n)

        for iter_ in mesher.layout().iter():
            if iter_.coordinates[self._direction] == n - 1:
                idx = iter_.index
                self._map_x.set_diag(idx, beta + f * b)
                self._map_x.set_lower(idx, alpha + f * c)

    # --- FdmLinearOpComposite surface ------------------------------------

    def size(self) -> int:
        """# C++ parity: ``FdmSquareRootFwdOp::size`` — always 1."""
        return 1

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``FdmSquareRootFwdOp::setTime`` — a no-op.

        The square-root coefficients are time-homogeneous, so the
        operator is fully built in the constructor.
        """

    def apply(self, p: Array) -> Array:
        """# C++ parity: ``FdmSquareRootFwdOp::apply``."""
        return self._map_x.apply(p)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: ``FdmSquareRootFwdOp::apply_mixed`` — zero."""
        return np.zeros_like(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: ``FdmSquareRootFwdOp::apply_direction``."""
        if direction == self._direction:
            return self._map_x.apply(r)
        return np.zeros_like(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmSquareRootFwdOp::solve_splitting``.

        Off-direction returns ``r`` unchanged (not zero).
        """
        if direction == self._direction:
            return self._map_x.solve_splitting(r, dt, 1.0)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmSquareRootFwdOp::preconditioner``."""
        return self.solve_splitting(self._direction, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: ``FdmSquareRootFwdOp::toMatrixDecomp``."""
        return [self._map_x.to_matrix()]

    def to_matrix(self) -> csr_matrix:
        """Sum of the decomposition.

        # C++ parity: ``FdmLinearOpComposite::toMatrix`` — accumulates
        # ``toMatrixDecomp()``.
        """
        decomp = self.to_matrix_decomp()
        total: csr_matrix = decomp[0]
        for m in decomp[1:]:
            total = cast("csr_matrix", total + m)
        return total


__all__ = ["FdmSquareRootFwdOp", "TransformationType"]
