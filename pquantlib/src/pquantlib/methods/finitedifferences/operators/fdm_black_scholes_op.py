"""FdmBlackScholesOp — 1-D Black-Scholes linear operator (log-spot).

# C++ parity: ql/methods/finitedifferences/operators/fdmblackscholesop.{hpp,cpp}
# (v1.42.1).

The Black-Scholes PDE in log-spot ``x = log S`` is

.. math::

    \\partial_t V + (r - q - 0.5 \\sigma^2) \\partial_x V
                + 0.5 \\sigma^2 \\partial_{xx} V - r V = 0

so the spatial operator is

.. math::

    L = (r - q - 0.5 \\sigma^2) D_x + 0.5 \\sigma^2 D_{xx} - r I

and the backward solver propagates ``V`` from maturity to t=0.

The operator is built once on construction (with first/second
derivative ops); ``set_time(t1, t2)`` recomputes the coefficients
using the forward rates over ``[t1, t2]`` and the forward variance.

**Carve-outs (vs C++):**

* ``local_vol`` is deferred — the engine always uses the
  constant-vol branch.
* ``quanto_helper`` is deferred — multi-currency support lands in
  Phase 6.
* ``apply_mixed`` / ``apply_direction`` / ``preconditioner`` are
  inherited as direct delegations (the 1-D op has a single
  direction = 0).
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding


@final
class FdmBlackScholesOp:
    """1-D Black-Scholes operator in log-spot coordinates.

    # C++ parity: ``class FdmBlackScholesOp : public
    # FdmLinearOpComposite``. The Python port is **not** a subclass of
    # ``FdmLinearOp`` directly — it owns a ``TripleBandLinearOp`` that
    # carries the time-dependent coefficients. The engine + backward
    # solver call ``set_time(t1, t2)``, then ``apply`` / ``solve_splitting``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        process: GeneralizedBlackScholesProcess,
        strike: float,
        direction: int = 0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._direction: int = direction
        self._rts = process.risk_free_rate()
        self._qts = process.dividend_yield()
        self._vol_ts = process.black_volatility()
        self._strike: float = strike
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(direction, mesher)
        self._dxx_map: SecondDerivativeOp = SecondDerivativeOp(direction, mesher)
        # The output operator gets recomputed at each step via set_time.
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)

    def size(self) -> int:
        """Number of directions (always 1 for the 1-D BSM op).

        # C++ parity: ``FdmLinearOpComposite::size``.
        """
        return 1

    def set_time(self, t1: float, t2: float) -> None:
        """Recompute coefficients using the forward rates over [t1, t2].

        # C++ parity: ``FdmBlackScholesOp::setTime``.
        """
        # Forward rates over [t1, t2]: r and q (continuous compounding).
        r = self._rts.forward_rate(t1, t2, Compounding.Continuous).rate()
        q = self._qts.forward_rate(t1, t2, Compounding.Continuous).rate()
        # Forward variance per unit time.
        v = self._vol_ts.black_forward_variance_at_time(t1, t2, self._strike, extrapolate=True) / (t2 - t1)

        # L = (r - q - 0.5 v) * D_x + 0.5 v * D_xx - r * I
        size = self._mesher.layout().size()
        a_scalar = np.array([r - q - 0.5 * v], dtype=np.float64)
        b_scalar = np.array([-r], dtype=np.float64)
        # mult by half * v on the second-derivative op.
        v_arr = np.full(size, v, dtype=np.float64)
        scaled_dxx = self._dxx_map.mult(0.5 * v_arr)
        # axpyb does: self <- a*x + y + b on each band.
        self._map_t.axpyb(a_scalar, self._dx_map, scaled_dxx, b_scalar)

    def apply(self, r: Array) -> Array:
        """Triple-band matrix-vector product ``L @ r``.

        # C++ parity: ``FdmBlackScholesOp::apply``.
        """
        return self._map_t.apply(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """Solve ``(I + dt * L) x = r`` along the given direction.

        # C++ parity: ``FdmBlackScholesOp::solve_splitting``. For the
        # 1-D BSM op only ``direction == 0`` is meaningful — any
        # other direction returns ``r`` unchanged.
        """
        if direction == self._direction:
            return self._map_t.solve_splitting(r, dt, 1.0)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner = solve along the operator's direction.

        # C++ parity: ``FdmBlackScholesOp::preconditioner``.
        """
        return self.solve_splitting(self._direction, r, dt)


__all__ = ["FdmBlackScholesOp"]
