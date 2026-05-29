"""FdmZabrOp — 2-D ZABR pricing operator.

# C++ parity: ql/experimental/finitedifferences/fdmzabrop.{hpp,cpp}
# (v1.42.1).

The ZABR (extended SABR) PDE in (forward, vol)-space follows from
the SDE

.. math::

    dF = V * F^\\beta dW_1, \\quad dV = \\nu * V^\\gamma dW_2,
    \\quad d\\langle W_1, W_2\\rangle = \\rho dt

(with classic SABR recovered at ``gamma = 1``).

The spatial operator is

.. math::

    L = 0.5 V^2 F^{2\\beta} D_{FF}
      + 0.5 \\nu^2 V^{2\\gamma} D_{VV}
      + \\nu \\rho |V|^{\\gamma+1} F^\\beta D_{FV}

with directions ``d0 = 0`` (forward) and ``d1 = 1`` (vol). The
discretization composes ``SecondDerivativeOp`` (along each direction)
scaled by the spatially-varying diffusion coefficients, plus the
mixed-derivative term via ``SecondOrderMixedDerivativeOp``.

This is the core of the **ZABR FD modes** carve-out (W2-A LocalVolatility
/ FullFd / ProjectedHedge) — together with a solver wrapper, this
operator enables `zabr_volatility(..., mode=...)` evaluation paths
beyond the closed-form short-maturity lognormal model.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.second_order_mixed_derivative_op import (
    SecondOrderMixedDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)


class FdmZabrUnderlyingPart:
    """The ``d0 = forward`` diffusion piece of the ZABR operator.

    # C++ parity: ``class FdmZabrUnderlyingPart``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        beta: float,
        nu: float,
        rho: float,
        gamma: float,
    ) -> None:
        # Vol values (mesher direction 1) + forward values (direction 0).
        forward_values = mesher.locations(0)
        volatility_values = mesher.locations(1)
        # mapT = SecondDerivativeOp(0).mult(0.5 * V^2 * F^(2*beta)).
        coef = 0.5 * volatility_values * volatility_values * np.power(forward_values, 2.0 * beta)
        self._map_t: TripleBandLinearOp = SecondDerivativeOp(0, mesher).mult(coef)

    def set_time(self, t1: float, t2: float) -> None:
        """No-op — coefficients are spatial, time-independent.

        # C++ parity: ``FdmZabrUnderlyingPart::setTime`` — empty body.
        """

    def get_map(self) -> TripleBandLinearOp:
        return self._map_t


class FdmZabrVolatilityPart:
    """The ``d1 = vol`` diffusion piece of the ZABR operator.

    # C++ parity: ``class FdmZabrVolatilityPart``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        beta: float,
        nu: float,
        rho: float,
        gamma: float,
    ) -> None:
        volatility_values = mesher.locations(1)
        # mapT = SecondDerivativeOp(1).mult(0.5 * nu^2 * V^(2*gamma)).
        coef = 0.5 * nu * nu * np.power(volatility_values, 2.0 * gamma)
        self._map_t: TripleBandLinearOp = SecondDerivativeOp(1, mesher).mult(coef)

    def set_time(self, t1: float, t2: float) -> None:
        """No-op — coefficients are spatial, time-independent."""

    def get_map(self) -> TripleBandLinearOp:
        return self._map_t


class FdmZabrOp:
    """2-D ZABR pricing operator (FdmLinearOpComposite-compatible).

    # C++ parity: ``class FdmZabrOp : public FdmLinearOpComposite``.

    Satisfies the ``FdmLinearOpComposite`` Protocol — size 2 +
    apply / apply_mixed / apply_direction / solve_splitting /
    preconditioner. ``set_time`` delegates to the two part objects
    (both of which are no-ops in C++).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        beta: float,
        nu: float,
        rho: float,
        gamma: float = 1.0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        forward_values = mesher.locations(0)
        volatility_values = mesher.locations(1)
        # Mixed-derivative coefficient:
        #   nu * rho * |V|^(gamma+1) * F^beta.
        mixed_coef = nu * rho * np.power(np.abs(volatility_values), gamma + 1.0) * np.power(forward_values, beta)
        self._dxy_map = SecondOrderMixedDerivativeOp(0, 1, mesher).mult(mixed_coef)
        self._dx_map = FdmZabrUnderlyingPart(mesher, beta, nu, rho, gamma)
        self._dy_map = FdmZabrVolatilityPart(mesher, beta, nu, rho, gamma)

    def size(self) -> int:
        """Number of directions — always 2 for the ZABR op.

        # C++ parity: ``FdmZabrOp::size`` returns 2.
        """
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        """Delegate to the two parts (both no-ops in C++)."""
        self._dx_map.set_time(t1, t2)
        self._dy_map.set_time(t1, t2)

    def apply(self, r: Array) -> Array:
        """Full apply: ``L_x + L_y + L_xy``.

        # C++ parity: ``FdmZabrOp::apply``.
        """
        return self._dy_map.get_map().apply(r) + self._dx_map.get_map().apply(r) + self._dxy_map.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """Only the mixed-derivative contribution.

        # C++ parity: ``FdmZabrOp::apply_mixed``.
        """
        return self._dxy_map.apply(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Only the directional component (0 = forward, 1 = vol).

        # C++ parity: ``FdmZabrOp::apply_direction``.
        """
        if direction == 0:
            return self._dx_map.get_map().apply(r)
        if direction == 1:
            return self._dy_map.get_map().apply(r)
        raise ValueError(f"FdmZabrOp: direction too large, got {direction}")

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """Splitting solve along ``direction``.

        # C++ parity: ``FdmZabrOp::solve_splitting``.
        """
        if direction == 0:
            return self._dx_map.get_map().solve_splitting(r, dt, 1.0)
        if direction == 1:
            return self._dy_map.get_map().solve_splitting(r, dt, 1.0)
        raise ValueError(f"FdmZabrOp: direction too large, got {direction}")

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner = solve along direction 0.

        # C++ parity: ``FdmZabrOp::preconditioner``.
        """
        return self.solve_splitting(0, r, dt)


__all__ = ["FdmZabrOp", "FdmZabrUnderlyingPart", "FdmZabrVolatilityPart"]
