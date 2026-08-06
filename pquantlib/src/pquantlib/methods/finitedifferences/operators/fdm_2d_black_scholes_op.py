"""Fdm2dBlackScholesOp — two-asset Black-Scholes operator with correlation.

# C++ parity: ql/methods/finitedifferences/operators/fdm2dblackscholesop.{hpp,cpp}
# (v1.43).

Two log-spot directions, one 1-D Black-Scholes operator each, plus a
mixed second-derivative term carrying the correlation:

.. math::

    L = L_x + L_y + \\rho \\sigma_x \\sigma_y \\partial_{xy} + r I

The trailing ``+ r I`` looks odd until you notice that each of ``L_x``
and ``L_y`` already contributes its own ``-r I``; the mixed part adds
``r`` back so the total discounting is ``-r``, not ``-2r``. C++ carries
that ``+ currentForwardRate_ * x`` inside ``apply_mixed``, so the ADI
schemes see it in the explicit (mixed) stage. Ported verbatim.
"""

from __future__ import annotations

import sys
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import (
    FdmBlackScholesOp,
)
from pquantlib.methods.finitedifferences.operators.nine_point_linear_op import (
    NinePointLinearOp,
)
from pquantlib.methods.finitedifferences.operators.second_order_mixed_derivative_op import (
    SecondOrderMixedDerivativeOp,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.compounding import Compounding

# C++ parity: the default ``-Null<Real>()`` sentinel, i.e. ``-QL_MAX_REAL``.
_NEG_NULL_REAL: float = -sys.float_info.max


@final
class Fdm2dBlackScholesOp:
    """Correlated two-asset Black-Scholes operator.

    # C++ parity: ``class Fdm2dBlackScholesOp : public FdmLinearOpComposite``.

    ``local_vol`` / ``illegal_local_vol_overwrite`` are forwarded to the two
    ``FdmBlackScholesOp`` sub-operators and, independently, drive the
    correlation map: C++ scales the mixed-derivative template by the product
    of the two *local* vols per node rather than by the product of the two
    forward Black vols.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        p1: GeneralizedBlackScholesProcess,
        p2: GeneralizedBlackScholesProcess,
        correlation: float,
        maturity: float,
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = _NEG_NULL_REAL,
    ) -> None:
        """Build the operator.

        ``maturity`` is accepted for signature parity and never read —
        the C++ constructor comments the parameter out
        (fdm2dblackscholesop.cpp:38).
        """
        del maturity  # C++ parity: parameter accepted and discarded.
        self._mesher: FdmMesher = mesher
        self._p1: GeneralizedBlackScholesProcess = p1
        self._p2: GeneralizedBlackScholesProcess = p2
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        self._current_forward_rate: float = 0.0

        # C++ parity: ``localVol1_`` / ``localVol2_`` and the ``x_`` / ``y_``
        # spot grids, all empty unless ``localVol``.
        self._local_vol_1: LocalVolTermStructure | None = (
            p1.local_volatility() if local_vol else None
        )
        self._local_vol_2: LocalVolTermStructure | None = (
            p2.local_volatility() if local_vol else None
        )
        self._x: Array | None = (
            np.exp(np.asarray(mesher.locations(0), dtype=np.float64)) if local_vol else None
        )
        self._y: Array | None = (
            np.exp(np.asarray(mesher.locations(1), dtype=np.float64)) if local_vol else None
        )

        self._op_x: FdmBlackScholesOp = FdmBlackScholesOp(
            mesher, p1, p1.x0(), local_vol, illegal_local_vol_overwrite, 0
        )
        self._op_y: FdmBlackScholesOp = FdmBlackScholesOp(
            mesher, p2, p2.x0(), local_vol, illegal_local_vol_overwrite, 1
        )

        size = mesher.layout().size()
        self._corr_map_t: NinePointLinearOp = NinePointLinearOp(0, 1, mesher)
        self._corr_map_template: NinePointLinearOp = SecondOrderMixedDerivativeOp(
            0, 1, mesher
        ).mult(np.full(size, correlation, dtype=np.float64))

    def size(self) -> int:
        """# C++ parity: ``Fdm2dBlackScholesOp::size`` — always 2."""
        return 2

    def _node_vols(self, t1: float, t2: float) -> Array:
        """Per-node ``vol1 * vol2`` for the local-vol branch.

        # C++ parity: the ``localVol1_ != nullptr`` loop in ``setTime`` —
        # note it stores the vols themselves, not their squares, and applies
        # ``illegalLocalVolOverwrite_`` per asset independently.
        """
        lv1, lv2 = self._local_vol_1, self._local_vol_2
        x, y = self._x, self._y
        assert lv1 is not None
        assert lv2 is not None
        assert x is not None
        assert y is not None
        t_mid = 0.5 * (t1 + t2)
        overwrite = self._illegal_local_vol_overwrite
        size = self._mesher.layout().size()
        out: Array = np.empty(size, dtype=np.float64)
        for i in range(size):
            if overwrite < 0.0:
                v1 = lv1.local_vol_at_time(t_mid, float(x[i]), extrapolate=True)
                v2 = lv2.local_vol_at_time(t_mid, float(y[i]), extrapolate=True)
            else:
                try:
                    v1 = lv1.local_vol_at_time(t_mid, float(x[i]), extrapolate=True)
                except LibraryException:
                    v1 = overwrite
                try:
                    v2 = lv2.local_vol_at_time(t_mid, float(y[i]), extrapolate=True)
                except LibraryException:
                    v2 = overwrite
            out[i] = v1 * v2
        return out

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``Fdm2dBlackScholesOp::setTime``."""
        self._op_x.set_time(t1, t2)
        self._op_y.set_time(t1, t2)

        size = self._mesher.layout().size()
        if self._local_vol_1 is not None:
            self._corr_map_t = self._corr_map_template.mult(self._node_vols(t1, t2))
        else:
            vol1 = self._p1.black_volatility().black_forward_vol_at_time(
                t1, t2, self._p1.x0()
            )
            vol2 = self._p2.black_volatility().black_forward_vol_at_time(
                t1, t2, self._p2.x0()
            )
            self._corr_map_t = self._corr_map_template.mult(
                np.full(size, vol1 * vol2, dtype=np.float64)
            )

        self._current_forward_rate = (
            self._p1.risk_free_rate().forward_rate(t1, t2, Compounding.Continuous).rate()
        )

    def apply(self, x: Array) -> Array:
        """# C++ parity: ``Fdm2dBlackScholesOp::apply``."""
        return self._op_x.apply(x) + self._op_y.apply(x) + self.apply_mixed(x)

    def apply_mixed(self, x: Array) -> Array:
        """# C++ parity: ``Fdm2dBlackScholesOp::apply_mixed``.

        Carries the correlation stencil **plus** ``+r*x`` (see the
        module docstring for why the sign is positive).
        """
        return self._corr_map_t.apply(x) + self._current_forward_rate * x

    def apply_direction(self, direction: int, x: Array) -> Array:
        """# C++ parity: ``Fdm2dBlackScholesOp::apply_direction``."""
        if direction == 0:
            return self._op_x.apply(x)
        if direction == 1:
            return self._op_y.apply(x)
        return qassert.fail("direction is too large")

    def solve_splitting(self, direction: int, x: Array, dt: float) -> Array:
        """# C++ parity: ``Fdm2dBlackScholesOp::solve_splitting``."""
        if direction == 0:
            return self._op_x.solve_splitting(direction, x, dt)
        if direction == 1:
            return self._op_y.solve_splitting(direction, x, dt)
        return qassert.fail("direction is too large")

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: ``Fdm2dBlackScholesOp::preconditioner``."""
        return self.solve_splitting(0, r, dt)


__all__ = ["Fdm2dBlackScholesOp"]
