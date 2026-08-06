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
from pquantlib.time.compounding import Compounding

# C++ parity: the default ``-Null<Real>()`` sentinel, i.e. ``-QL_MAX_REAL``.
_NEG_NULL_REAL: float = -sys.float_info.max


@final
class Fdm2dBlackScholesOp:
    """Correlated two-asset Black-Scholes operator.

    # C++ parity: ``class Fdm2dBlackScholesOp : public FdmLinearOpComposite``.

    **Carve-out — ``local_vol=True``.** The C++ operator forwards the
    ``localVol`` / ``illegalLocalVolOverwrite`` flags into the two
    ``FdmBlackScholesOp`` sub-operators. PQuantLib's pre-existing
    ``FdmBlackScholesOp`` (``fdm_black_scholes_op.py``) documents
    ``local_vol`` as deferred and exposes no such parameter, so the
    local-vol branch cannot be wired up from here without changing that
    module. Constructing with ``local_vol=True`` therefore raises; the
    constant-vol branch (the C++ default) is complete and
    cross-validated.
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
        qassert.require(
            not local_vol,
            "Fdm2dBlackScholesOp(local_vol=True) needs the local-vol branch of "
            "FdmBlackScholesOp, which is a documented carve-out of the pquantlib "
            "port; use local_vol=False",
        )
        self._mesher: FdmMesher = mesher
        self._p1: GeneralizedBlackScholesProcess = p1
        self._p2: GeneralizedBlackScholesProcess = p2
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        self._current_forward_rate: float = 0.0

        self._op_x: FdmBlackScholesOp = FdmBlackScholesOp(mesher, p1, p1.x0(), 0)
        self._op_y: FdmBlackScholesOp = FdmBlackScholesOp(mesher, p2, p2.x0(), 1)

        size = mesher.layout().size()
        self._corr_map_t: NinePointLinearOp = NinePointLinearOp(0, 1, mesher)
        self._corr_map_template: NinePointLinearOp = SecondOrderMixedDerivativeOp(
            0, 1, mesher
        ).mult(np.full(size, correlation, dtype=np.float64))

    def size(self) -> int:
        """# C++ parity: ``Fdm2dBlackScholesOp::size`` — always 2."""
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``Fdm2dBlackScholesOp::setTime`` (constant-vol branch)."""
        self._op_x.set_time(t1, t2)
        self._op_y.set_time(t1, t2)

        vol1 = self._p1.black_volatility().black_forward_vol_at_time(
            t1, t2, self._p1.x0()
        )
        vol2 = self._p2.black_volatility().black_forward_vol_at_time(
            t1, t2, self._p2.x0()
        )
        size = self._mesher.layout().size()
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
