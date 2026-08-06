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

Both C++ branches are here:

* ``local_vol=True`` swaps the single forward variance for a per-node
  local variance sampled at ``0.5*(t1+t2)`` on the *spot* grid
  ``exp(locations)``, with ``illegal_local_vol_overwrite`` standing in
  for a lookup that raises.
* ``quanto_helper`` subtracts the quanto drift adjustment from the
  first-derivative coefficient (the scalar overload in the constant-vol
  branch, the vector one in the local-vol branch).

``apply_mixed`` / ``apply_direction`` / ``preconditioner`` are direct
delegations — the 1-D op has a single direction.
"""

from __future__ import annotations

import math
from typing import final

import numpy as np

from pquantlib.exceptions import LibraryException
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
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
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
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = -1.0,
        direction: int = 0,
        quanto_helper: FdmQuantoHelper | None = None,
    ) -> None:
        """# C++ parity: the seven-argument ``FdmBlackScholesOp`` constructor.

        ``illegal_local_vol_overwrite`` follows C++'s sentinel convention:
        a negative value means "let a failing local-vol lookup propagate",
        any non-negative value is substituted for the vol when the lookup
        raises. C++'s callers pass ``-Null<Real>()`` for the former; any
        negative number selects the same branch.
        """
        self._mesher: FdmMesher = mesher
        self._direction: int = direction
        self._rts = process.risk_free_rate()
        self._qts = process.dividend_yield()
        self._vol_ts = process.black_volatility()
        self._local_vol_ts: LocalVolTermStructure | None = (
            process.local_volatility() if local_vol else None
        )
        # C++ parity: ``x_((localVol) ? Array(Exp(mesher->locations(direction)))
        # : Array())`` — the *spot* level at every node, not the log-spot.
        self._x: Array | None = (
            np.exp(np.asarray(mesher.locations(direction), dtype=np.float64))
            if local_vol
            else None
        )
        self._strike: float = strike
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        self._quanto_helper: FdmQuantoHelper | None = quanto_helper
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(direction, mesher)
        self._dxx_map: SecondDerivativeOp = SecondDerivativeOp(direction, mesher)
        # The output operator gets recomputed at each step via set_time.
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)

    def _local_variances(self, t1: float, t2: float) -> Array:
        """Per-node local variance at the mid-point of ``[t1, t2]``.

        # C++ parity: the ``localVol_ != nullptr`` loop in ``setTime`` —
        # ``squared(localVol_->localVol(0.5*(t1+t2), x_[i], true))``, with
        # the failing lookup replaced by ``squared(illegalLocalVolOverwrite_)``
        # only when that override is non-negative.
        """
        local_vol_ts = self._local_vol_ts
        x = self._x
        assert local_vol_ts is not None
        assert x is not None
        t_mid = 0.5 * (t1 + t2)
        overwrite = self._illegal_local_vol_overwrite
        v: Array = np.empty(self._mesher.layout().size(), dtype=np.float64)
        for i in range(v.size):
            if overwrite < 0.0:
                vol = local_vol_ts.local_vol_at_time(t_mid, float(x[i]), extrapolate=True)
            else:
                try:
                    vol = local_vol_ts.local_vol_at_time(t_mid, float(x[i]), extrapolate=True)
                except LibraryException:
                    vol = overwrite
            v[i] = vol * vol
        return v

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
        quanto = self._quanto_helper
        size = self._mesher.layout().size()
        b_scalar: Array = np.array([-r], dtype=np.float64)

        if self._local_vol_ts is not None:
            # Local vol: the drift and the diffusion are both per-node.
            v = self._local_variances(t1, t2)
            drift: Array = r - q - 0.5 * v
            if quanto is not None:
                drift = drift - quanto.quanto_adjustment_array(np.sqrt(v), t1, t2)
            self._map_t.axpyb(drift, self._dx_map, self._dxx_map.mult(0.5 * v), b_scalar)
            return

        # Constant vol: forward variance per unit time, one scalar.
        vc = self._vol_ts.black_forward_variance_at_time(
            t1, t2, self._strike, extrapolate=True
        ) / (t2 - t1)
        a_scalar: Array = np.array([r - q - 0.5 * vc], dtype=np.float64)
        if quanto is not None:
            a_scalar = a_scalar - quanto.quanto_adjustment(math.sqrt(vc), t1, t2)
        # L = (r - q - 0.5 v) * D_x + 0.5 v * D_xx - r * I
        v_arr: Array = np.full(size, vc, dtype=np.float64)
        scaled_dxx = self._dxx_map.mult(0.5 * v_arr)
        # axpyb does: self <- a*x + y + b on each band.
        self._map_t.axpyb(a_scalar, self._dx_map, scaled_dxx, b_scalar)

    def apply(self, r: Array) -> Array:
        """Triple-band matrix-vector product ``L @ r``.

        # C++ parity: ``FdmBlackScholesOp::apply``.
        """
        return self._map_t.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """Mixed-derivative apply — zero for the 1-D BSM op.

        # C++ parity: ``FdmBlackScholesOp::apply_mixed``.
        """
        return np.zeros_like(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Directional apply — equal to ``apply`` along direction 0, zero else.

        # C++ parity: ``FdmBlackScholesOp::apply_direction``.
        """
        if direction == self._direction:
            return self._map_t.apply(r)
        return np.zeros_like(r)

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
