"""FdmCIROp — Heston-style operator with a CIR stochastic short rate.

# C++ parity: ql/methods/finitedifferences/operators/fdmcirop.{hpp,cpp}
# (v1.43).

Direction 0 carries ``x = log S`` and direction 1 the CIR short rate
``r``. With a *deterministic* Black volatility ``v`` (the forward
variance per unit time over the step) the operator splits into

* equity part ``FdmCIREquityPart``

  .. math::

      L_x = (r - q - \\tfrac{1}{2} v) D_x + \\tfrac{1}{2} v D_{xx}
            - \\tfrac{1}{2} r I

* rates part ``FdmCIRRatesPart``

  .. math::

      L_r = \\sigma^2 r D_{rr} + \\kappa(\\theta - r) D_r - \\tfrac{1}{2} r I

* mixed part ``FdmCIRMixedPart``: ``2 \\rho \\sigma_{CIR} \\sqrt{v} D_{xr}``

Note the rates part uses ``sigma^2 r`` (not ``0.5 sigma^2 r``) — that is
what v1.43 does; ported verbatim.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.nine_point_linear_op import (
    NinePointLinearOp,
)
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.second_order_mixed_derivative_op import (
    SecondOrderMixedDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


class FdmCIREquityPart:
    """Log-spot direction of the CIR hybrid operator.

    # C++ parity: ``class FdmCIREquityPart`` (fdmcirop.hpp:40-59,
    # fdmcirop.cpp:29-53).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        bs_process: GeneralizedBlackScholesProcess,
        strike: float,
    ) -> None:
        # C++ parity: fdmcirop.cpp:29-40.
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(0, mesher)
        self._dxx_map: SecondDerivativeOp = SecondDerivativeOp(0, mesher)
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(0, mesher)
        self._mesher: FdmMesher = mesher
        self._qTS: YieldTermStructure = bs_process.dividend_yield()
        self._strike: float = strike
        self._sigma1: BlackVolTermStructure = bs_process.black_volatility()

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmcirop.cpp:42-49."""
        q = self._qTS.forward_rate(t1, t2, Compounding.Continuous).rate()
        v = self._sigma1.black_forward_variance_at_time(t1, t2, self._strike, extrapolate=True) / (t2 - t1)
        locations = self._mesher.locations(1)
        size = self._mesher.layout().size()
        self._map_t.axpyb(
            locations - q - 0.5 * v,
            self._dx_map,
            self._dxx_map.mult(np.full(size, v / 2.0, dtype=np.float64)),
            -0.5 * locations,
        )

    def get_map(self) -> TripleBandLinearOp:
        """# C++ parity: ``FdmCIREquityPart::getMap``."""
        return self._map_t


class FdmCIRRatesPart:
    """CIR short-rate direction of the hybrid operator.

    # C++ parity: ``class FdmCIRRatesPart`` (fdmcirop.hpp:61-74,
    # fdmcirop.cpp:55-72).
    """

    def __init__(self, mesher: FdmMesher, sigma: float, kappa: float, theta: float) -> None:
        # C++ parity: fdmcirop.cpp:55-64.
        locations = mesher.locations(1)
        self._dy_map: TripleBandLinearOp = (
            SecondDerivativeOp(1, mesher)
            .mult(sigma * sigma * locations)
            .add(FirstDerivativeOp(1, mesher).mult(kappa * (theta - locations)))
        )
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(1, mesher)
        self._mesher: FdmMesher = mesher

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmcirop.cpp:66-68 — time-independent in v1.43."""
        _ = t1, t2
        self._map_t.axpyb(None, self._dy_map, self._dy_map, -0.5 * self._mesher.locations(1))

    def get_map(self) -> TripleBandLinearOp:
        """# C++ parity: ``FdmCIRRatesPart::getMap``."""
        return self._map_t


class FdmCIRMixedPart:
    """Spot / short-rate correlation piece of the CIR hybrid operator.

    # C++ parity: ``class FdmCIRMixedPart`` (fdmcirop.hpp:76-93,
    # fdmcirop.cpp:74-96).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        cir_process: CoxIngersollRossProcess,
        bs_process: GeneralizedBlackScholesProcess,
        rho: float,
        strike: float,
    ) -> None:
        # C++ parity: fdmcirop.cpp:74-86.
        size = mesher.layout().size()
        self._dy_map: NinePointLinearOp = SecondOrderMixedDerivativeOp(0, 1, mesher).mult(
            np.full(size, 2.0 * rho * cir_process.volatility(), dtype=np.float64)
        )
        self._map_t: NinePointLinearOp = NinePointLinearOp(0, 1, mesher)
        self._mesher: FdmMesher = mesher
        self._sigma1: BlackVolTermStructure = bs_process.black_volatility()
        self._strike: float = strike

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmcirop.cpp:88-92 (build then ``swap``)."""
        v = math.sqrt(
            self._sigma1.black_forward_variance_at_time(t1, t2, self._strike, extrapolate=True) / (t2 - t1)
        )
        size = self._mesher.layout().size()
        self._map_t = self._dy_map.mult(np.full(size, v, dtype=np.float64))

    def get_map(self) -> NinePointLinearOp:
        """# C++ parity: ``FdmCIRMixedPart::getMap``."""
        return self._map_t


class FdmCIROp:
    """Composite spot / CIR-short-rate operator on a 2-D grid.

    # C++ parity: ``class FdmCIROp : public FdmLinearOpComposite``
    # (fdmcirop.hpp:96-120, fdmcirop.cpp:98-173). Python satisfies the
    # ``FdmLinearOpComposite`` Protocol structurally.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        cir_process: CoxIngersollRossProcess,
        bs_process: GeneralizedBlackScholesProcess,
        rho: float,
        strike: float,
    ) -> None:
        # C++ parity: fdmcirop.cpp:98-116.
        self._dx_map: FdmCIREquityPart = FdmCIREquityPart(mesher, bs_process, strike)
        self._dy_map: FdmCIRRatesPart = FdmCIRRatesPart(
            mesher,
            cir_process.volatility(),
            cir_process.speed(),
            cir_process.level(),
        )
        self._dz_map: FdmCIRMixedPart = FdmCIRMixedPart(mesher, cir_process, bs_process, rho, strike)

    def size(self) -> int:
        """# C++ parity: fdmcirop.cpp:125-127 — always 2."""
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmcirop.cpp:119-123."""
        self._dx_map.set_time(t1, t2)
        self._dy_map.set_time(t1, t2)
        self._dz_map.set_time(t1, t2)

    def apply(self, r: Array) -> Array:
        """# C++ parity: fdmcirop.cpp:129-135."""
        dx = self._dx_map.get_map().apply(r)
        dy = self._dy_map.get_map().apply(r)
        dz = self._dz_map.get_map().apply(r)
        return dy + dx + dz

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: fdmcirop.cpp:137-145."""
        if direction == 0:
            return self._dx_map.get_map().apply(r)
        if direction == 1:
            return self._dy_map.get_map().apply(r)
        qassert.fail("direction too large")

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: fdmcirop.cpp:147-149."""
        return self._dz_map.get_map().apply(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: fdmcirop.cpp:151-161."""
        if direction == 0:
            return self._dx_map.get_map().solve_splitting(r, dt, 1.0)
        if direction == 1:
            return self._dy_map.get_map().solve_splitting(r, dt, 1.0)
        qassert.fail("direction too large")

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: fdmcirop.cpp:163-165."""
        return self.solve_splitting(1, self.solve_splitting(0, r, dt), dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: fdmcirop.cpp:167-173."""
        return [
            self._dx_map.get_map().to_matrix(),
            self._dy_map.get_map().to_matrix(),
            self._dz_map.get_map().to_matrix(),
        ]


__all__ = [
    "FdmCIREquityPart",
    "FdmCIRMixedPart",
    "FdmCIROp",
    "FdmCIRRatesPart",
]
