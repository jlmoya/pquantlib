"""FdmBlackScholesFwdOp — Black-Scholes Fokker-Planck (forward) operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmblackscholesfwdop.{hpp,cpp}
# (v1.43).

The *forward* (Fokker-Planck) equation for the transition density
``p(x, t)`` in log-spot coordinates ``x = log S`` is the formal adjoint
of the backward Black-Scholes operator:

.. math::

    \\partial_t p = \\partial_x\\!\\left[-(r - q - \\tfrac{1}{2}v) p\\right]
                  + \\tfrac{1}{2}\\partial_{xx}\\!\\left[v p\\right]

Discretised, the adjoint shows up as **right** multiplication of the
derivative operators by the (node-dependent) coefficient array —
``multR`` rather than ``mult`` — in the local-vol branch. In the
constant-vol branch the coefficient is uniform, so C++ falls back to a
scalar ``axpyb`` plus ``mult``. Both branches are ported verbatim,
including that asymmetry.
"""

from __future__ import annotations

import sys
from typing import final

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

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
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.compounding import Compounding

# C++ parity: the default ``-Null<Real>()`` sentinel, i.e. ``-QL_MAX_REAL``.
# Only its sign is load-bearing (``illegalLocalVolOverwrite_ < 0.0``).
_NEG_NULL_REAL: float = -sys.float_info.max


@final
class FdmBlackScholesFwdOp:
    """1-D Black-Scholes forward (Fokker-Planck) operator.

    # C++ parity: ``class FdmBlackScholesFwdOp : public FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        process: GeneralizedBlackScholesProcess,
        strike: float,
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = _NEG_NULL_REAL,
        direction: int = 0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._r_ts = process.risk_free_rate()
        self._q_ts = process.dividend_yield()
        self._vol_ts = process.black_volatility()
        self._local_vol: LocalVolTermStructure | None = (
            process.local_volatility() if local_vol else None
        )
        self._x: Array = (
            np.exp(mesher.locations(direction))
            if local_vol
            else np.zeros(0, dtype=np.float64)
        )
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(direction, mesher)
        self._dxx_map: SecondDerivativeOp = SecondDerivativeOp(direction, mesher)
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)
        self._strike: float = strike
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        self._direction: int = direction

    def size(self) -> int:
        """# C++ parity: ``FdmBlackScholesFwdOp::size`` — always 1."""
        return 1

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``FdmBlackScholesFwdOp::setTime``."""
        r = self._r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        q = self._q_ts.forward_rate(t1, t2, Compounding.Continuous).rate()

        local_vol = self._local_vol
        if local_vol is not None:
            size = self._mesher.layout().size()
            v = np.empty(size, dtype=np.float64)
            t_mid = 0.5 * (t1 + t2)
            for i in range(size):
                if self._illegal_local_vol_overwrite < 0.0:
                    lv = local_vol.local_vol_at_time(t_mid, float(self._x[i]), True)
                else:
                    try:
                        lv = local_vol.local_vol_at_time(t_mid, float(self._x[i]), True)
                    except LibraryException:
                        lv = self._illegal_local_vol_overwrite
                v[i] = lv * lv
            # C++: mapT_.axpyb(Array(1, 1.0), dxMap_.multR(-r + q + 0.5*v),
            #                  dxxMap_.multR(0.5*v), Array(1, 0.0));
            self._map_t.axpyb(
                np.array([1.0], dtype=np.float64),
                self._dx_map.mult_r(-r + q + 0.5 * v),
                self._dxx_map.mult_r(0.5 * v),
                np.array([0.0], dtype=np.float64),
            )
        else:
            v_scalar = self._vol_ts.black_forward_variance_at_time(
                t1, t2, self._strike
            ) / (t2 - t1)
            size = self._mesher.layout().size()
            # C++: mapT_.axpyb(Array(1, -r + q + 0.5*v), dxMap_,
            #                  dxxMap_.mult(0.5*Array(size, v)), Array(1, 0.0));
            self._map_t.axpyb(
                np.array([-r + q + 0.5 * v_scalar], dtype=np.float64),
                self._dx_map,
                self._dxx_map.mult(0.5 * np.full(size, v_scalar, dtype=np.float64)),
                np.array([0.0], dtype=np.float64),
            )

    def apply(self, u: Array) -> Array:
        """# C++ parity: ``FdmBlackScholesFwdOp::apply``."""
        return self._map_t.apply(u)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: ``FdmBlackScholesFwdOp::apply_mixed`` — zero."""
        return np.zeros_like(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: ``FdmBlackScholesFwdOp::apply_direction``."""
        if direction == self._direction:
            return self._map_t.apply(r)
        return np.zeros_like(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmBlackScholesFwdOp::solve_splitting``.

        Note the off-direction branch returns ``r`` **unchanged** (not
        zero) — unlike ``apply_direction``.
        """
        if direction == self._direction:
            return self._map_t.solve_splitting(r, dt, 1.0)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmBlackScholesFwdOp::preconditioner``."""
        return self.solve_splitting(self._direction, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: ``FdmBlackScholesFwdOp::toMatrixDecomp``."""
        return [self._map_t.to_matrix()]


__all__ = ["FdmBlackScholesFwdOp"]
