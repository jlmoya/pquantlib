"""FdmOrnsteinUhlenbeckOp — 1-D Ornstein-Uhlenbeck FD operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmornsteinuhlenbeckop.{hpp,cpp}
# (v1.42.1).

The OU PDE for a value ``V(x, t)`` under

.. math::

    dx_t = a (b - x_t) dt + \\sigma dW_t

is

.. math::

    \\partial_t V + a (b - x) \\partial_x V + 0.5 \\sigma^2 \\partial_{xx} V - r V = 0

with the spatial operator

.. math::

    L = a (b - x) D_x + 0.5 \\sigma^2 D_{xx} - r I

The drift term ``a (b - x)`` is space-dependent (linear in ``x``);
the diffusion term ``0.5 sigma^2`` is constant. The C++ build is a
two-stage operation: ``m_`` accumulates ``drift * D_x + diffusion *
D_{xx}`` (spatial, time-independent), and ``mapX_`` adds the
time-dependent ``-r I`` bias at each ``set_time(t1, t2)`` call.
"""

from __future__ import annotations

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
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


class FdmOrnsteinUhlenbeckOp:
    """1-D Ornstein-Uhlenbeck FD operator.

    # C++ parity: ``class FdmOrnsteinUhlenbeckOp : public FdmLinearOpComposite``.

    Satisfies the ``FdmLinearOpComposite`` Protocol so it can drive
    the existing schemes + backward solver.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        process: OrnsteinUhlenbeckProcess,
        rTS: YieldTermStructure,  # noqa: N803 (C++ field name preserved)
        direction: int = 0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._process: OrnsteinUhlenbeckProcess = process
        self._rTS: YieldTermStructure = rTS
        self._direction: int = direction

        # Initialize the two TripleBandLinearOps along the working direction.
        self._m: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)
        self._map_x: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)

        # Build the spatial diffusion + drift map ``m_``.
        # drift[i] = process.drift(0, x[i]) (uses the OU drift formula).
        x_locations = mesher.locations(direction)
        n = mesher.layout().size()
        drift = np.empty(n, dtype=np.float64)
        for i in range(n):
            drift[i] = process.drift_1d(0.0, float(x_locations[i]))

        # m_ = drift * D_x + 0.5 * sigma^2 * D_{xx}.
        sigma = process.volatility()
        diffusion_const = 0.5 * sigma * sigma
        diffusion_arr = np.full(n, diffusion_const, dtype=np.float64)
        scaled_dxx = SecondDerivativeOp(direction, mesher).mult(diffusion_arr)
        # axpyb: self <- a*x + y + b. With a = drift, x = D_x, y = scaled_dxx, b = None.
        self._m.axpyb(drift, FirstDerivativeOp(direction, mesher), scaled_dxx, None)

    def size(self) -> int:
        """Number of directions.

        # C++ parity: ``FdmOrnsteinUhlenbeckOp::size`` returns the
        # mesher's layout dimension count.
        """
        return len(self._mesher.layout().dim())

    def set_time(self, t1: float, t2: float) -> None:
        """Recompute ``mapX = m - r * I`` for the forward rate over ``[t1, t2]``.

        # C++ parity: ``FdmOrnsteinUhlenbeckOp::setTime``.
        """
        r = self._rTS.forward_rate(t1, t2, Compounding.Continuous).rate()
        # mapX = m + 0 * x + Array(1, -r) — i.e. m on bands, -r on diag.
        r_bias = np.array([-r], dtype=np.float64)
        self._map_x.axpyb(None, None, self._m, r_bias)

    def apply(self, r: Array) -> Array:
        """Apply ``mapX @ r``.

        # C++ parity: ``FdmOrnsteinUhlenbeckOp::apply``.
        """
        return self._map_x.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """Mixed-derivative apply — zero for the 1-D OU op.

        # C++ parity: ``FdmOrnsteinUhlenbeckOp::apply_mixed`` returns
        # an Array of zeros with the same size as ``r``.
        """
        return np.zeros_like(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Directional apply — equal to ``apply`` along the op's direction.

        # C++ parity: ``FdmOrnsteinUhlenbeckOp::apply_direction``.
        """
        if direction == self._direction:
            return self._map_x.apply(r)
        return np.zeros_like(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """Splitting solve along the op's direction.

        # C++ parity: ``FdmOrnsteinUhlenbeckOp::solve_splitting``.
        """
        if direction == self._direction:
            return self._map_x.solve_splitting(r, dt, 1.0)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner = solve along the op's direction.

        # C++ parity: ``FdmOrnsteinUhlenbeckOp::preconditioner``.
        """
        return self.solve_splitting(self._direction, r, dt)


__all__ = ["FdmOrnsteinUhlenbeckOp"]
