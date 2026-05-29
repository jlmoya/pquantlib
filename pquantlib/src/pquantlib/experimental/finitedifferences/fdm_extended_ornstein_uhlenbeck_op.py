"""FdmExtendedOrnsteinUhlenbeckOp — FD operator for the ExtOU SDE.

# C++ parity: ql/experimental/finitedifferences/fdmextendedornsteinuhlenbeckop.{hpp,cpp}
# (v1.42.1).

For the ExtOU SDE

    dx_t = a (b(t) - x_t) dt + sigma dW_t

the linear PDE for an option price ``V(t, x)`` is

    dV/dt + drift(t, x) dV/dx + 0.5 sigma^2 d2V/dx2 - r V = 0

so the spatial operator at fixed time slice ``[t1, t2]`` is

    L = mu(t, x) D_x + 0.5 sigma^2 D_xx - r I

where ``mu(t, x) = drift(0.5*(t1+t2), x)`` evaluated on the mesh and
``r`` is the forward rate over ``[t1, t2]`` from the discount curve.

The operator is built once on construction with ``FirstDerivativeOp``
+ a ``SecondDerivativeOp`` scaled by ``0.5 sigma^2``; ``setTime``
recomputes the drift vector + rate and recombines via
``TripleBandLinearOp.axpyb``.

The C++ class inherits from ``FdmLinearOpComposite``. The Python
port matches the ``FdmExtOUJumpOp`` / ``FdmKlugeExtOUOp`` composite
contract by implementing the same methods (``size``, ``set_time``,
``apply``, ``apply_mixed``, ``apply_direction``, ``solve_splitting``,
``preconditioner``).
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)
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
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@final
class FdmExtendedOrnsteinUhlenbeckOp:
    """FD operator for the ExtOU SDE.

    # C++ parity: ``class FdmExtendedOrnsteinUhlenbeckOp : public
    # FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        process: ExtendedOrnsteinUhlenbeckProcess,
        r_ts: YieldTermStructure,
        direction: int = 0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._process: ExtendedOrnsteinUhlenbeckProcess = process
        self._r_ts: YieldTermStructure = r_ts
        self._direction: int = direction

        # Pre-cache per-node x locations along the active direction.
        self._x: Array = mesher.locations(direction)

        # Build first and (sigma^2/2)-scaled second derivative operators.
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(direction, mesher)
        # 0.5 * sigma^2 broadcast over every grid node.
        sigma2_half = 0.5 * process.volatility() * process.volatility()
        size = mesher.layout().size()
        self._dxx_map: TripleBandLinearOp = SecondDerivativeOp(direction, mesher).mult(
            np.full(size, sigma2_half, dtype=np.float64)
        )
        # The actual time-dependent operator gets recomputed via set_time.
        self._map_x: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)

    # --- composite interface --------------------------------------------

    def size(self) -> int:
        """Number of directions in the operator (one per mesh axis).

        # C++ parity: ``FdmExtendedOrnsteinUhlenbeckOp::size`` returns
        # the count of *all* mesh dimensions, not just the active one.
        """
        return len(self._mesher.layout().dim())

    def set_time(self, t1: float, t2: float) -> None:
        """Recompute the operator's coefficients at the new time slice.

        # C++ parity: ``setTime`` — drift evaluated at mid-time;
        # forward rate over [t1, t2] subtracted as the ``-r * I``
        # bias.
        """
        r = self._r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        t_mid = 0.5 * (t1 + t2)

        # Drift per grid node: drift(t_mid, x_i).
        layout_size = self._mesher.layout().size()
        drift = np.empty(layout_size, dtype=np.float64)
        for iter_ in self._mesher.layout().iter():
            i = iter_.index
            drift[i] = self._process.drift_1d(t_mid, float(self._x[i]))

        # mapX = drift * D_x + dxxMap - r * I
        self._map_x.axpyb(
            drift, self._dx_map, self._dxx_map, np.array([-r], dtype=np.float64)
        )

    def apply(self, r: Array) -> Array:
        """Matrix-vector product ``L @ r``.

        # C++ parity: ``apply`` — single direction so it's just the
        # triple-band's apply.
        """
        return self._map_x.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """No mixed term in the ExtOU op — returns zeros.

        # C++ parity: ``apply_mixed`` returns ``Array(r.size(), 0.0)``.
        """
        return np.zeros(r.size, dtype=np.float64)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Apply only the ``direction``-axis part of the operator.

        # C++ parity: ``apply_direction``: matches ``direction`` ->
        # full apply; else returns zeros.
        """
        if direction == self._direction:
            return self._map_x.apply(r)
        return np.zeros(r.size, dtype=np.float64)

    def solve_splitting(self, direction: int, r: Array, a: float) -> Array:
        """Solve ``(I + a * L_dir) x = r`` along the given direction.

        # C++ parity: ``solve_splitting`` — only the active direction
        # is non-trivial; other directions short-circuit to ``r``.
        """
        if direction == self._direction:
            return self._map_x.solve_splitting(r, a, 1.0)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner = ``solve_splitting`` along the active direction.

        # C++ parity: ``preconditioner``.
        """
        return self.solve_splitting(self._direction, r, dt)


__all__ = ["FdmExtendedOrnsteinUhlenbeckOp"]
