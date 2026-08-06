"""FdmHestonHullWhiteOp — Heston / Hull-White hybrid linear operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmhestonhullwhiteop.{hpp,cpp}
# (v1.43).

Three grid directions: ``x = log S`` (0), the Heston variance ``v`` (1)
and the Hull-White state ``x_r`` (2). The operator is

* equity part ``FdmHestonHullWhiteEquityPart``

  .. math::

      L_x = (x_r + \\varphi - \\tfrac{1}{2} v - q) D_x + \\tfrac{1}{2} v D_{xx}

  (no discount term — the whole ``-r`` sits in the Hull-White piece),

* variance part ``\\tfrac{1}{2}\\sigma^2 v D_{vv} + \\kappa(\\theta - v) D_v``,

* a ``FdmHullWhiteOp`` on direction 2 (which carries ``-(x_r + phi)``),

* two nine-point correlation pieces: Heston's ``rho sigma v D_{xv}`` and the
  equity/short-rate ``sqrt(v) sigma_r rho_{Sr} D_{xr}``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_hull_white_op import (
    FdmHullWhiteOp,
)
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
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@runtime_checkable
class _HullWhiteProcessLike(Protocol):
    """Structural stand-in for ``HullWhiteProcess``.

    # C++ parity: ql/processes/hullwhiteprocess.hpp — ``Real a() const``
    # and ``Real sigma() const``.

    ``FdmHestonHullWhiteOp`` reads nothing else from the process, so the
    dependency is expressed structurally. ``HullWhiteProcess`` itself is
    not (yet) part of ``pquantlib.processes``; the ported
    ``HullWhiteForwardProcess`` exposes the same two accessors with the
    same semantics and therefore satisfies this Protocol.
    """

    def a(self) -> float:
        """Mean-reversion speed."""
        ...

    def sigma(self) -> float:
        """Short-rate volatility."""
        ...


class FdmHestonHullWhiteEquityPart:
    """Log-spot direction of the Heston / Hull-White operator.

    # C++ parity: ``class FdmHestonHullWhiteEquityPart``
    # (fdmhestonhullwhiteop.hpp:41-60, fdmhestonhullwhiteop.cpp:32-67).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        hw_model: HullWhite,
        qTS: YieldTermStructure,  # noqa: N803 — C++ member name preserved
    ) -> None:
        # C++ parity: fdmhestonhullwhiteop.cpp:32-51. ``dxxMap_`` is built
        # from the *unmodified* ``0.5 * locations(1)`` in the member
        # initialiser list, before the boundary loop zeroes
        # ``varianceValues_``.
        self._x: Array = mesher.locations(2)
        self._variance_values: Array = 0.5 * mesher.locations(1)
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(0, mesher)
        self._dxx_map: TripleBandLinearOp = SecondDerivativeOp(0, mesher).mult(0.5 * mesher.locations(1))
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(0, mesher)
        self._hw_model: HullWhite = hw_model
        self._mesher: FdmMesher = mesher
        self._qTS: YieldTermStructure = qTS

        last = mesher.layout().dim()[0] - 1
        for iter_ in mesher.layout().iter():
            if iter_.coordinates[0] in (0, last):
                self._variance_values[iter_.index] = 0.0
        self._volatility_values: Array = np.sqrt(2.0 * self._variance_values)

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmhestonhullwhiteop.cpp:53-63."""
        dynamics = self._hw_model.dynamics()
        phi = 0.5 * (dynamics.short_rate(t1, 0.0) + dynamics.short_rate(t2, 0.0))
        q = self._qTS.forward_rate(t1, t2, Compounding.Continuous).rate()
        # C++ passes an empty ``Array()`` bias — no diagonal offset here.
        self._map_t.axpyb(
            self._x + phi - self._variance_values - q,
            self._dx_map,
            self._dxx_map,
            None,
        )

    def get_map(self) -> TripleBandLinearOp:
        """# C++ parity: ``FdmHestonHullWhiteEquityPart::getMap``."""
        return self._map_t


class FdmHestonHullWhiteOp:
    """Composite Heston / Hull-White operator on a 3-D grid.

    # C++ parity: ``class FdmHestonHullWhiteOp : public FdmLinearOpComposite``
    # (fdmhestonhullwhiteop.hpp:62-91, fdmhestonhullwhiteop.cpp:69-151).
    # Python satisfies the ``FdmLinearOpComposite`` Protocol structurally.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        heston_process: HestonProcess,
        hw_process: _HullWhiteProcessLike,
        equity_short_rate_correlation: float,
    ) -> None:
        # C++ parity: fdmhestonhullwhiteop.cpp:69-91.
        self._v0: float = heston_process.v0
        self._kappa: float = heston_process.kappa
        self._theta: float = heston_process.theta
        self._sigma: float = heston_process.sigma
        self._rho: float = heston_process.rho
        self._hw_model: HullWhite = HullWhite(
            heston_process.risk_free_rate(), hw_process.a(), hw_process.sigma()
        )

        locations1 = mesher.locations(1)
        self._heston_corr_map: NinePointLinearOp = SecondOrderMixedDerivativeOp(0, 1, mesher).mult(
            self._rho * self._sigma * locations1
        )
        self._equity_ir_corr_map: NinePointLinearOp = SecondOrderMixedDerivativeOp(0, 2, mesher).mult(
            np.sqrt(locations1) * hw_process.sigma() * equity_short_rate_correlation
        )
        self._dy_map: TripleBandLinearOp = (
            SecondDerivativeOp(1, mesher)
            .mult(0.5 * self._sigma * self._sigma * locations1)
            .add(FirstDerivativeOp(1, mesher).mult(self._kappa * (self._theta - locations1)))
        )
        self._dx_map: FdmHestonHullWhiteEquityPart = FdmHestonHullWhiteEquityPart(
            mesher, self._hw_model, heston_process.dividend_yield()
        )
        self._hull_white_op: FdmHullWhiteOp = FdmHullWhiteOp(mesher, self._hw_model, 2)

        qassert.require(
            equity_short_rate_correlation * equity_short_rate_correlation
            + heston_process.rho * heston_process.rho
            <= 1.0,
            "correlation matrix has negative eigenvalues",
        )

    def size(self) -> int:
        """# C++ parity: fdmhestonhullwhiteop.cpp:98-100 — always 3."""
        return 3

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmhestonhullwhiteop.cpp:93-96."""
        self._dx_map.set_time(t1, t2)
        self._hull_white_op.set_time(t1, t2)

    def apply(self, r: Array) -> Array:
        """# C++ parity: fdmhestonhullwhiteop.cpp:102-106."""
        return (
            self._dy_map.apply(r)
            + self._dx_map.get_map().apply(r)
            + self._hull_white_op.apply(r)
            + self._heston_corr_map.apply(r)
            + self._equity_ir_corr_map.apply(r)
        )

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: fdmhestonhullwhiteop.cpp:108-118."""
        if direction == 0:
            return self._dx_map.get_map().apply(r)
        if direction == 1:
            return self._dy_map.apply(r)
        if direction == 2:
            return self._hull_white_op.apply(r)
        qassert.fail("direction too large")

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: fdmhestonhullwhiteop.cpp:120-122."""
        return self._heston_corr_map.apply(r) + self._equity_ir_corr_map.apply(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: fdmhestonhullwhiteop.cpp:124-137."""
        if direction == 0:
            return self._dx_map.get_map().solve_splitting(r, dt, 1.0)
        if direction == 1:
            return self._dy_map.solve_splitting(r, dt, 1.0)
        if direction == 2:
            return self._hull_white_op.solve_splitting(2, r, dt)
        qassert.fail("direction too large")

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: fdmhestonhullwhiteop.cpp:139-142."""
        return self.solve_splitting(0, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: fdmhestonhullwhiteop.cpp:144-151."""
        return [
            self._dx_map.get_map().to_matrix(),
            self._dy_map.to_matrix(),
            self._hull_white_op.to_matrix_decomp()[0],
            self._heston_corr_map.to_matrix() + self._equity_ir_corr_map.to_matrix(),
        ]


__all__ = ["FdmHestonHullWhiteEquityPart", "FdmHestonHullWhiteOp"]
