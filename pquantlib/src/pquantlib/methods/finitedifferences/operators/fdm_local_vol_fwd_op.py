"""FdmLocalVolFwdOp — local-volatility Fokker-Planck (forward) operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmlocalvolfwdop.{hpp,cpp}
# (v1.43).

Same forward equation as :mod:`fdm_black_scholes_fwd_op`, but the
local-vol surface is supplied directly instead of being pulled off a
``GeneralizedBlackScholesProcess``, and there is no constant-vol
fallback branch — ``localVol`` is always used.
"""

from __future__ import annotations

from typing import final

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
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
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@final
class FdmLocalVolFwdOp:
    """1-D local-volatility forward (Fokker-Planck) operator.

    # C++ parity: ``class FdmLocalVolFwdOp : public FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        spot: Quote,
        r_ts: YieldTermStructure,
        q_ts: YieldTermStructure,
        local_vol: LocalVolTermStructure,
        direction: int = 0,
    ) -> None:
        """Build the operator.

        ``spot`` is accepted for signature parity with C++ but never
        read — the C++ constructor takes ``const ext::shared_ptr<Quote>&
        spot`` and does not store it either (fdmlocalvolfwdop.cpp:30-39).
        """
        del spot  # C++ parity: parameter accepted and discarded.
        self._mesher: FdmMesher = mesher
        self._r_ts: YieldTermStructure = r_ts
        self._q_ts: YieldTermStructure = q_ts
        self._local_vol: LocalVolTermStructure = local_vol
        self._x: Array = np.exp(mesher.locations(direction))
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(direction, mesher)
        self._dxx_map: SecondDerivativeOp = SecondDerivativeOp(direction, mesher)
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)
        self._direction: int = direction

    def size(self) -> int:
        """# C++ parity: ``FdmLocalVolFwdOp::size`` — always 1."""
        return 1

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``FdmLocalVolFwdOp::setTime``."""
        r = self._r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        q = self._q_ts.forward_rate(t1, t2, Compounding.Continuous).rate()

        size = self._mesher.layout().size()
        v = np.empty(size, dtype=np.float64)
        t_mid = 0.5 * (t1 + t2)
        for i in range(size):
            lv = self._local_vol.local_vol_at_time(t_mid, float(self._x[i]), True)
            v[i] = lv * lv

        # C++: mapT_.axpyb(Array(1, 1.0), dxMap_.multR(-r + q + 0.5*v),
        #                  dxxMap_.multR(0.5*v), Array(1, 0.0));
        self._map_t.axpyb(
            np.array([1.0], dtype=np.float64),
            self._dx_map.mult_r(-r + q + 0.5 * v),
            self._dxx_map.mult_r(0.5 * v),
            np.array([0.0], dtype=np.float64),
        )

    def apply(self, u: Array) -> Array:
        """# C++ parity: ``FdmLocalVolFwdOp::apply``."""
        return self._map_t.apply(u)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: ``FdmLocalVolFwdOp::apply_mixed`` — zero."""
        return np.zeros_like(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: ``FdmLocalVolFwdOp::apply_direction``."""
        if direction == self._direction:
            return self._map_t.apply(r)
        return np.zeros_like(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmLocalVolFwdOp::solve_splitting``.

        Off-direction returns ``r`` unchanged (not zero).
        """
        if direction == self._direction:
            return self._map_t.solve_splitting(r, dt, 1.0)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmLocalVolFwdOp::preconditioner``."""
        return self.solve_splitting(self._direction, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: ``FdmLocalVolFwdOp::toMatrixDecomp``."""
        return [self._map_t.to_matrix()]


__all__ = ["FdmLocalVolFwdOp"]
