"""FdmDupire1dOp — 1-D Dupire local-volatility FD operator.

# C++ parity: ql/experimental/finitedifferences/fdmdupire1dop.{hpp,cpp}
# (v1.42.1).

The Dupire forward PDE in spot-space ``S`` (with time reversed so a
backward solver can be used) for a call price ``C(S, t)`` under a
local-vol surface ``sigma(S, t)`` is

.. math::

    \\partial_t C = 0.5 * sigma(S, t)^2 * \\partial_{SS} C

so the spatial operator is

.. math::

    L = 0.5 * sigma(S, t)^2 * D_{SS}

For the simplified 1-D Dupire op the local-vol curve is passed as a
fixed ``Array`` at construction (i.e. the time-slice is captured at
build time); the operator does not refresh its coefficients on
``set_time``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)


class FdmDupire1dOp:
    """1-D Dupire local-volatility operator.

    # C++ parity: ``class FdmDupire1dOp : public FdmLinearOpComposite``.

    Satisfies the ``FdmLinearOpComposite`` Protocol so it can be
    used with the existing schemes + backward solver.
    """

    def __init__(self, mesher: FdmMesher, local_volatility: Array) -> None:
        self._mesher: FdmMesher = mesher
        self._local_volatility: Array = np.asarray(local_volatility, dtype=np.float64).copy()
        # L = 0.5 * sigma^2 * D_{SS}.
        dxx = SecondDerivativeOp(0, mesher)
        scale = 0.5 * self._local_volatility * self._local_volatility
        self._map_t: TripleBandLinearOp = dxx.mult(scale)

    def size(self) -> int:
        """Number of directions (always 1 for the 1-D Dupire op).

        # C++ parity: ``FdmDupire1dOp::size``.
        """
        return 1

    def set_time(self, t1: float, t2: float) -> None:
        """No-op for the constant-vol-slice Dupire op.

        # C++ parity: ``FdmDupire1dOp::setTime`` — empty body.
        """

    def apply(self, r: Array) -> Array:
        """Second-derivative apply with scaled local-vol coefficients.

        # C++ parity: ``FdmDupire1dOp::apply``.
        """
        return self._map_t.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """Mixed-derivative apply — returns ``r`` unchanged.

        # C++ parity: ``FdmDupire1dOp::apply_mixed`` — returns ``r``
        # (identity, not zero — this matches the C++ semantics where
        # mixed-derivative contribution to a 1-D op is the identity).
        """
        return np.asarray(r, dtype=np.float64).copy()

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Directional apply — direction 0 only.

        # C++ parity: ``FdmDupire1dOp::apply_direction`` — fails on
        # ``direction > 0`` in C++; the Python port returns zeros
        # for robustness when the solver iterates over directions.
        """
        if direction == 0:
            return self._map_t.apply(r)
        raise ValueError(f"FdmDupire1dOp: direction too large, got {direction}")

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """Splitting solve — direction 0 only.

        # C++ parity: ``FdmDupire1dOp::solve_splitting`` —
        # ``(I + dt * L)^{-1} r`` via the Thomas algorithm.
        """
        if direction == 0:
            return self._map_t.solve_splitting(r, dt, 1.0)
        raise ValueError(f"FdmDupire1dOp: direction too large, got {direction}")

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner = solve along direction 0.

        # C++ parity: ``FdmDupire1dOp::preconditioner``.
        """
        return self.solve_splitting(0, r, dt)


__all__ = ["FdmDupire1dOp"]
