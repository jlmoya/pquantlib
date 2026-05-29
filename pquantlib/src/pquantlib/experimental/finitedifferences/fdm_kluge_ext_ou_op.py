"""FdmKlugeExtOUOp — 3-D FD operator for the correlated Kluge + ExtOU process.

# C++ parity: ql/experimental/finitedifferences/fdmklugeextouop.{hpp,cpp}
# (v1.42.1).

For the three-factor system

    P_t = exp(p_t + X_t + Y_t)      G_t = exp(g_t + U_t)
    dX_t = -alpha X_t dt + sigma_x dW_t^x
    dY_t = -beta Y_t dt + J_t dN_t
    dU_t = -kappa U_t dt + sigma_u dW_t^u
    rho = corr(dW_t^x, dW_t^u)

the FD operator splits into:

* ``klugeOp`` — ``FdmExtOUJumpOp`` on (X, Y).
* ``ouOp`` — ``FdmExtendedOrnsteinUhlenbeckOp`` on U (direction 2)
  with a zero-rate term-structure (so we don't double-subtract r).
* ``corrMap`` — ``SecondOrderMixedDerivativeOp`` between X (dir 0)
  and U (dir 2), scaled by ``rho * sigma_x * sigma_u``.

The composite ``apply`` sums these. ``solve_splitting`` routes by
direction: 0,1 -> klugeOp; 2 -> ouOp.
"""

from __future__ import annotations

from typing import cast, final

import numpy as np

from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_op import (
    FdmExtOUJumpOp,
)
from pquantlib.experimental.finitedifferences.fdm_extended_ornstein_uhlenbeck_op import (
    FdmExtendedOrnsteinUhlenbeckOp,
)
from pquantlib.experimental.finitedifferences.nine_point_linear_op import (
    NinePointLinearOp,
)
from pquantlib.experimental.finitedifferences.second_order_mixed_derivative_op import (
    SecondOrderMixedDerivativeOp,
)
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


@final
class FdmKlugeExtOUOp:
    """3-D FD operator for the correlated Kluge + ExtOU process.

    # C++ parity: ``class FdmKlugeExtOUOp : public FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        kluge_ext_ou_process: KlugeExtOUProcess,
        r_ts: YieldTermStructure,
        integro_integration_order: int = 16,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._r_ts: YieldTermStructure = r_ts

        kluge = kluge_ext_ou_process.get_kluge_process()
        ext_ou = kluge_ext_ou_process.get_ext_ou_process()

        # KlugeOp (X+Y axes) — uses the full r_ts.
        self._kluge_op: FdmExtOUJumpOp = FdmExtOUJumpOp(
            mesher, kluge, r_ts, integro_integration_order
        )

        # OU op (U axis = direction 2) — feed a zero-rate TS to avoid
        # double-counting the discount factor (klugeOp already
        # subtracts r). The C++ implementation builds a fresh
        # FlatForward(referenceDate(rTS), Quote(0.0), dayCounter) for
        # this. We match this by wrapping the yield term structure.
        zero_rate_ts = _ZeroRateYieldTermStructure(r_ts)
        self._ou_op: FdmExtendedOrnsteinUhlenbeckOp = FdmExtendedOrnsteinUhlenbeckOp(
            mesher,
            ext_ou,
            cast(YieldTermStructure, zero_rate_ts),
            direction=2,
        )

        # corrMap: SecondOrderMixedDerivative on (0, 2) scaled by
        # rho * sigma_u * sigma_x where sigma_x is the *embedded*
        # ExtendedOU process's volatility (i.e., kluge -> ext-ou -> sigma).
        sigma_x = kluge.get_extended_ornstein_uhlenbeck_process().volatility()
        sigma_u = ext_ou.volatility()
        scale = kluge_ext_ou_process.rho() * sigma_x * sigma_u
        layout_size = mesher.layout().size()
        self._corr_map: NinePointLinearOp = SecondOrderMixedDerivativeOp(0, 2, mesher).mult(
            np.full(layout_size, scale, dtype=np.float64)
        )

    # --- composite interface --------------------------------------------

    def size(self) -> int:
        """Number of directions in the operator.

        # C++ parity: size() returns layout.dim().size() (= 3).
        """
        return len(self._mesher.layout().dim())

    def set_time(self, t1: float, t2: float) -> None:
        """Recompute time-dependent parts (ouOp + klugeOp).

        # C++ parity: ``setTime`` propagates to both sub-ops.
        """
        self._ou_op.set_time(t1, t2)
        self._kluge_op.set_time(t1, t2)

    def apply(self, r: Array) -> Array:
        """Full apply: ouOp + klugeOp + corrMap.

        # C++ parity: ``apply``.
        """
        return self._ou_op.apply(r) + self._kluge_op.apply(r) + self._corr_map.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """corrMap.apply + klugeOp.apply_mixed (the integro part).

        # C++ parity: ``apply_mixed`` sums the mixed-derivative term
        # and the jump-integro contribution.
        """
        return self._corr_map.apply(r) + self._kluge_op.apply_mixed(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Apply only the ``direction``-axis part.

        # C++ parity: ``apply_direction`` = ``klugeOp.apply_direction +
        # ouOp.apply_direction`` (both check if direction matches).
        """
        return self._kluge_op.apply_direction(direction, r) + self._ou_op.apply_direction(
            direction, r
        )

    def solve_splitting(self, direction: int, r: Array, a: float) -> Array:
        """Solve ``(I + a * L_dir) x = r`` along the given direction.

        # C++ parity: directions 0, 1 -> klugeOp; 2 -> ouOp; else r.
        """
        if direction in (0, 1):
            return self._kluge_op.solve_splitting(direction, r, a)
        if direction == 2:
            return self._ou_op.solve_splitting(2, r, a)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner via klugeOp solve along direction 0.

        # C++ parity: ``preconditioner`` always uses klugeOp on direction 0.
        """
        return self._kluge_op.solve_splitting(0, r, dt)


class _ZeroRateYieldTermStructure:
    """Minimal YTS wrapper that returns rate=0 for every forward rate query.

    # C++ parity: lines fdmklugeextouop.cpp:55-59 — the C++ code builds
    # ``FlatForward(rTS->referenceDate(), Handle<Quote>(SimpleQuote(0.0)),
    # rTS->dayCounter())`` to drop the discount factor in the ouOp.
    # We replicate by routing only the ``forward_rate(t1, t2, Continuous)``
    # surface used by ``FdmExtendedOrnsteinUhlenbeckOp.set_time``.
    """

    __slots__ = ("_base",)

    def __init__(self, base: YieldTermStructure) -> None:
        self._base: YieldTermStructure = base

    def forward_rate(self, t1: float, t2: float, compounding: object, *args: object, **kwargs: object) -> _ZeroInterestRate:
        return _ZeroInterestRate()


class _ZeroInterestRate:
    """Trivial helper exposing only ``.rate() -> 0.0``."""

    __slots__ = ()

    def rate(self) -> float:
        return 0.0


__all__ = ["FdmKlugeExtOUOp"]
