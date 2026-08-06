"""FdmHestonFwdOp — Heston Fokker-Planck (forward) operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmhestonfwdop.{hpp,cpp}
# (v1.43).

Two-dimensional forward operator for the Heston model in log-spot
(direction 0) and variance (direction 1, in one of the
:class:`TransformationType` coordinate systems):

* ``mapX_`` — the log-spot part, rebuilt at every :meth:`set_time`
  because the drift depends on the ``[t1, t2]`` forward rates.
* ``mapY_`` — a :class:`FdmSquareRootFwdOp` on the variance axis.
* ``correlation_`` — the mixed second-derivative stencil.

The constructor patches the ``v``-boundary rows of ``dxxMap_`` (and of
a separate ``boundary_`` operator kept for the leverage-function path)
with a zero-flux term whose weight comes from ``mapY_``'s boundary
factors. That is why both ``dxxMap_`` and ``boundary_`` are
``ModTripleBandLinearOp``.

The optional ``leverage_fct`` turns this into the forward operator of a
stochastic-local-volatility model: the log-spot diffusion is scaled by
``L(t, S)^2`` and the correlation term by ``L(t, S)``.
"""

from __future__ import annotations

import math
from typing import cast, final

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_square_root_fwd_op import (
    FdmSquareRootFwdOp,
    TransformationType,
)
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.mod_triple_band_linear_op import (
    ModTripleBandLinearOp,
)
from pquantlib.methods.finitedifferences.operators.nine_point_linear_op import (
    NinePointLinearOp,
)
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.second_order_mixed_derivative_op import (
    SecondOrderMixedDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.compounding import Compounding


@final
class FdmHestonFwdOp:
    """Heston forward (Fokker-Planck) operator.

    # C++ parity: ``class FdmHestonFwdOp : public FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        process: HestonProcess,
        transform_type: TransformationType = TransformationType.Plain,
        leverage_fct: LocalVolTermStructure | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        self._type: TransformationType = transform_type
        self._kappa: float = process.kappa
        self._theta: float = process.theta
        self._sigma: float = process.sigma
        self._rho: float = process.rho
        self._v0: float = process.v0
        self._mixed_sigma: float = mixing_factor * self._sigma

        self._r_ts = process.risk_free_rate()
        self._q_ts = process.dividend_yield()

        layout = mesher.layout()
        size = layout.size()
        loc1 = mesher.locations(1)
        self._variance_values: Array = 0.5 * loc1

        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(0, mesher)
        dxx_base: TripleBandLinearOp = (
            SecondDerivativeOp(0, mesher).mult(0.5 * np.exp(loc1))
            if transform_type == TransformationType.Log
            else SecondDerivativeOp(0, mesher).mult(0.5 * loc1)
        )
        self._dxx_map: ModTripleBandLinearOp = ModTripleBandLinearOp(op=dxx_base)
        # C++: SecondDerivativeOp(0, mesher).mult(Array(locations(0).size(), 0.0)).
        self._boundary: ModTripleBandLinearOp = ModTripleBandLinearOp(
            op=SecondDerivativeOp(0, mesher).mult(np.zeros(size, dtype=np.float64))
        )
        self._l: Array = np.ones(size, dtype=np.float64)

        self._map_x: TripleBandLinearOp = TripleBandLinearOp(0, mesher)
        self._map_y: FdmSquareRootFwdOp = FdmSquareRootFwdOp(
            mesher, self._kappa, self._theta, self._mixed_sigma, 1, transform_type
        )
        self._correlation: NinePointLinearOp = SecondOrderMixedDerivativeOp(
            0, 1, mesher
        ).mult(
            np.full(size, self._rho * self._mixed_sigma, dtype=np.float64)
            if transform_type == TransformationType.Log
            else self._rho * self._mixed_sigma * loc1
        )

        self._leverage_fct: LocalVolTermStructure | None = leverage_fct
        self._mesher: FdmMesher = mesher

        # --- zero flux boundary condition --------------------------------
        n = layout.dim()[1]
        lower_boundary_factor = self._map_y.lower_boundary_factor(transform_type)
        upper_boundary_factor = self._map_y.upper_boundary_factor(transform_type)

        log_fac_low = (
            math.exp(self._map_y.v(0)) if transform_type == TransformationType.Log else 1.0
        )
        log_fac_upp = (
            math.exp(self._map_y.v(n + 1))
            if transform_type == TransformationType.Log
            else 1.0
        )

        alpha = -2 * self._rho / self._mixed_sigma * lower_boundary_factor * log_fac_low
        beta = -2 * self._rho / self._mixed_sigma * upper_boundary_factor * log_fac_upp

        f_dx = ModTripleBandLinearOp(op=FirstDerivativeOp(0, mesher))

        for iter_ in layout.iter():
            coord1 = iter_.coordinates[1]
            if coord1 == 0:
                idx = iter_.index
                if leverage_fct is None:
                    self._dxx_map.set_upper(
                        idx, self._dxx_map.upper(idx) + alpha * f_dx.upper(idx)
                    )
                    self._dxx_map.set_diag(
                        idx, self._dxx_map.diag(idx) + alpha * f_dx.diag(idx)
                    )
                    self._dxx_map.set_lower(
                        idx, self._dxx_map.lower(idx) + alpha * f_dx.lower(idx)
                    )
                self._boundary.set_upper(idx, alpha * f_dx.upper(idx))
                self._boundary.set_diag(idx, alpha * f_dx.diag(idx))
                self._boundary.set_lower(idx, alpha * f_dx.lower(idx))
            elif coord1 == n - 1:
                idx = iter_.index
                if leverage_fct is None:
                    self._dxx_map.set_upper(
                        idx, self._dxx_map.upper(idx) + beta * f_dx.upper(idx)
                    )
                    self._dxx_map.set_diag(
                        idx, self._dxx_map.diag(idx) + beta * f_dx.diag(idx)
                    )
                    self._dxx_map.set_lower(
                        idx, self._dxx_map.lower(idx) + beta * f_dx.lower(idx)
                    )
                self._boundary.set_upper(idx, beta * f_dx.upper(idx))
                self._boundary.set_diag(idx, beta * f_dx.diag(idx))
                self._boundary.set_lower(idx, beta * f_dx.lower(idx))

    def size(self) -> int:
        """# C++ parity: ``FdmHestonFwdOp::size`` — always 2."""
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``FdmHestonFwdOp::setTime``."""
        r = self._r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        q = self._q_ts.forward_rate(t1, t2, Compounding.Continuous).rate()

        if self._leverage_fct is not None:
            self._l = self._get_leverage_fct_slice(t1, t2)
            l_square = self._l * self._l
            a = np.array([-r + q], dtype=np.float64)
            if self._type == TransformationType.Plain:
                y = (
                    self._dxx_map.mult_r(l_square)
                    .add(self._boundary.mult_r(self._l))
                    .add(self._dx_map.mult_r(self._rho * self._mixed_sigma * self._l))
                    .add(self._dx_map.mult(self._variance_values).mult_r(l_square))
                )
            elif self._type == TransformationType.Power:
                y = (
                    self._dxx_map.mult_r(l_square)
                    .add(self._boundary.mult_r(self._l))
                    .add(
                        self._dx_map.mult_r(
                            self._rho
                            * 2.0
                            * self._kappa
                            * self._theta
                            / (self._mixed_sigma)
                            * self._l
                        )
                    )
                    .add(self._dx_map.mult(self._variance_values).mult_r(l_square))
                )
            else:
                y = (
                    self._dxx_map.mult_r(l_square)
                    .add(self._boundary.mult_r(self._l))
                    .add(
                        self._dx_map.mult(
                            0.5 * np.exp(2.0 * self._variance_values)
                        ).mult_r(l_square)
                    )
                )
            self._map_x.axpyb(a, self._dx_map, y, None)
        else:
            if self._type == TransformationType.Plain:
                a = -r + q + self._rho * self._mixed_sigma + self._variance_values
            elif self._type == TransformationType.Power:
                a = (
                    -r
                    + q
                    + self._rho * 2.0 * self._kappa * self._theta / (self._mixed_sigma)
                    + self._variance_values
                )
            else:
                a = -r + q + 0.5 * np.exp(2.0 * self._variance_values)
            self._map_x.axpyb(a, self._dx_map, self._dxx_map, None)

    def apply(self, u: Array) -> Array:
        """# C++ parity: ``FdmHestonFwdOp::apply``."""
        if self._leverage_fct is not None:
            return (
                self._map_x.apply(u)
                + self._map_y.apply(u)
                + self._correlation.apply(self._l * u)
            )
        return self._map_x.apply(u) + self._map_y.apply(u) + self._correlation.apply(u)

    def apply_mixed(self, u: Array) -> Array:
        """# C++ parity: ``FdmHestonFwdOp::apply_mixed``."""
        if self._leverage_fct is not None:
            return self._correlation.apply(self._l * u)
        return self._correlation.apply(u)

    def apply_direction(self, direction: int, u: Array) -> Array:
        """# C++ parity: ``FdmHestonFwdOp::apply_direction``."""
        if direction == 0:
            return self._map_x.apply(u)
        if direction == 1:
            return self._map_y.apply(u)
        return qassert.fail("direction too large")

    def solve_splitting(self, direction: int, u: Array, dt: float) -> Array:
        """# C++ parity: ``FdmHestonFwdOp::solve_splitting``."""
        if direction == 0:
            return self._map_x.solve_splitting(u, dt, 1.0)
        if direction == 1:
            return self._map_y.solve_splitting(1, u, dt)
        return qassert.fail("direction too large")

    def preconditioner(self, u: Array, dt: float) -> Array:
        """# C++ parity: ``FdmHestonFwdOp::preconditioner`` — solve along 1."""
        return self.solve_splitting(1, u, dt)

    def _get_leverage_fct_slice(self, t1: float, t2: float) -> Array:
        """# C++ parity: ``FdmHestonFwdOp::getLeverageFctSlice``.

        The leverage function is strike- (not variance-) dependent, so
        C++ evaluates it once per log-spot coordinate on the ``v == 0``
        row and copies the value across the variance axis. Because
        ``spacing[0] == 1`` the flat index of the ``v == 0`` row equals
        the log-spot coordinate, which is why the C++ source can index
        the buffer with ``nx`` directly. Reproduced verbatim.
        """
        layout = self._mesher.layout()
        v = np.ones(layout.size(), dtype=np.float64)
        leverage_fct = self._leverage_fct
        if leverage_fct is None:
            return v

        t = 0.5 * (t1 + t2)
        time = min(leverage_fct.max_time(), t)

        for iter_ in layout.iter():
            nx = iter_.coordinates[0]
            if iter_.coordinates[1] == 0:
                x = math.exp(self._mesher.location(iter_, 0))
                spot = min(leverage_fct.max_strike(), max(leverage_fct.min_strike(), x))
                v[nx] = max(0.01, leverage_fct.local_vol_at_time(time, spot, True))
            else:
                v[iter_.index] = v[nx]
        return v

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: ``FdmHestonFwdOp::toMatrixDecomp``."""
        return [
            self._map_x.to_matrix(),
            self._map_y.to_matrix(),
            self._correlation.to_matrix(),
        ]

    def to_matrix(self) -> csr_matrix:
        """# C++ parity: ``FdmLinearOpComposite::toMatrix``."""
        decomp = self.to_matrix_decomp()
        total: csr_matrix = decomp[0]
        for m in decomp[1:]:
            total = cast("csr_matrix", total + m)
        return total


__all__ = ["FdmHestonFwdOp"]
