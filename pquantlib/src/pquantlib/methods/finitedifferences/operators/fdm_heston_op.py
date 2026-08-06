"""FdmHestonOp — Heston linear operator (log-spot x variance grid).

# C++ parity: ql/methods/finitedifferences/operators/fdmhestonop.{hpp,cpp}
# (v1.43).

With ``x = log S`` on direction 0 and the variance ``v`` on direction 1,
the (stochastic-local-volatility) Heston PDE operator splits into

* an **equity part** ``FdmHestonEquityPart``

  .. math::

      L_x = (r - q - \\tfrac{1}{2} v L^2) D_x
            + \\tfrac{1}{2} v L^2 D_{xx} - \\tfrac{1}{2} r I

* a **variance part** ``FdmHestonVariancePart``

  .. math::

      L_v = \\tfrac{1}{2}\\sigma^2 v D_{vv} + \\kappa(\\theta - v) D_v
            - \\tfrac{1}{2} r I

* a **correlation** nine-point piece ``rho * sigma * v * D_{xv}``, scaled
  by the leverage slice ``L`` when ``apply``-ing.

``L`` is the leverage function slice (all ones when no leverage function
is supplied); the ``-r I`` discount term is split evenly between the two
directional parts so that ``apply`` sums to the full ``-r I``.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

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
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@runtime_checkable
class _FdmQuantoHelperLike(Protocol):
    """Structural stand-in for ``FdmQuantoHelper``.

    # C++ parity: ql/methods/finitedifferences/utilities/fdmquantohelper.hpp
    # — ``Array quantoAdjustment(const Array& equityVol, Time t1, Time t2)``.

    ``FdmHestonOp`` only ever calls the ``Array`` overload of
    ``quantoAdjustment`` — spelled ``quanto_adjustment_array`` in Python,
    which cannot overload on argument type. The dependency is expressed
    structurally so this module does not have to import from
    ``methods.finitedifferences.utilities`` (which imports back from
    ``operators``); ``FdmQuantoHelper`` satisfies the Protocol.
    """

    def quanto_adjustment_array(self, equity_vol: Array, t1: float, t2: float) -> Array:
        """Quanto drift adjustment for a per-node equity vol vector."""
        ...


class FdmHestonEquityPart:
    """Log-spot direction of the Heston operator.

    # C++ parity: ``class FdmHestonEquityPart`` (fdmhestonop.hpp:40-65,
    # fdmhestonop.cpp:32-101).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        rTS: YieldTermStructure,  # noqa: N803 — C++ member name preserved
        qTS: YieldTermStructure,  # noqa: N803 — C++ member name preserved
        quanto_helper: _FdmQuantoHelperLike | None = None,
        leverage_fct: LocalVolTermStructure | None = None,
    ) -> None:
        # C++ parity: fdmhestonop.cpp:32-52. Note ``dxxMap_`` is built in
        # the member-initialiser list, i.e. from the *unmodified*
        # ``0.5 * locations(1)``, before the boundary loop below zeroes
        # ``varianceValues_``.
        self._variance_values: Array = 0.5 * mesher.locations(1)
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(0, mesher)
        self._dxx_map: TripleBandLinearOp = SecondDerivativeOp(0, mesher).mult(0.5 * mesher.locations(1))
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(0, mesher)
        self._mesher: FdmMesher = mesher
        self._rTS: YieldTermStructure = rTS
        self._qTS: YieldTermStructure = qTS
        self._quanto_helper: _FdmQuantoHelperLike | None = quanto_helper
        self._leverage_fct: LocalVolTermStructure | None = leverage_fct

        # On the boundary s_min / s_max the second derivative d^2V/dS^2 is
        # zero, and by Ito's lemma the variance term in the drift vanishes.
        last = mesher.layout().dim()[0] - 1
        for iter_ in mesher.layout().iter():
            if iter_.coordinates[0] in (0, last):
                self._variance_values[iter_.index] = 0.0
        self._volatility_values: Array = np.sqrt(2.0 * self._variance_values)

        # C++ leaves ``L_`` default-constructed (size 0) until setTime.
        self._leverage: Array = np.empty(0, dtype=np.float64)

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmhestonop.cpp:54-70."""
        r = self._rTS.forward_rate(t1, t2, Compounding.Continuous).rate()
        q = self._qTS.forward_rate(t1, t2, Compounding.Continuous).rate()

        self._leverage = self._get_leverage_fct_slice(t1, t2)
        l_square = self._leverage * self._leverage

        bias = np.array([-0.5 * r], dtype=np.float64)
        if self._quanto_helper is not None:
            drift = (
                r
                - q
                - self._variance_values * l_square
                - self._quanto_helper.quanto_adjustment_array(
                    self._volatility_values * self._leverage, t1, t2
                )
            )
        else:
            drift = r - q - self._variance_values * l_square
        self._map_t.axpyb(drift, self._dx_map, self._dxx_map.mult(l_square), bias)

    def _get_leverage_fct_slice(self, t1: float, t2: float) -> Array:
        """# C++ parity: fdmhestonop.cpp:72-96.

        The C++ writes ``v[nx]`` (the *coordinate* along direction 0) while
        walking the ``coordinates()[1] == 0`` row; because ``spacing[0] == 1``
        the flat index of that row equals ``nx``, so the later
        ``v[iter.index()] = v[nx]`` copy always reads an already-populated
        slot. Ported verbatim.
        """
        layout = self._mesher.layout()
        v: Array = np.ones(layout.size(), dtype=np.float64)
        if self._leverage_fct is None:
            return v

        t = 0.5 * (t1 + t2)
        time = min(self._leverage_fct.max_time(), t)
        min_strike = self._leverage_fct.min_strike()
        max_strike = self._leverage_fct.max_strike()

        for iter_ in layout.iter():
            nx = iter_.coordinates[0]
            if iter_.coordinates[1] == 0:
                x = math.exp(self._mesher.location(iter_, 0))
                spot = min(max_strike, max(min_strike, x))
                v[nx] = max(0.01, self._leverage_fct.local_vol_at_time(time, spot, True))
            else:
                v[iter_.index] = v[nx]
        return v

    def get_map(self) -> TripleBandLinearOp:
        """# C++ parity: ``FdmHestonEquityPart::getMap``."""
        return self._map_t

    def get_L(self) -> Array:  # noqa: N802 — C++ accessor name ``getL``
        """# C++ parity: ``const Array& getL() const`` (the ``L_`` member)."""
        return self._leverage


class FdmHestonVariancePart:
    """Variance direction of the Heston operator.

    # C++ parity: ``class FdmHestonVariancePart`` (fdmhestonop.hpp:67-83,
    # fdmhestonop.cpp:103-120).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        rTS: YieldTermStructure,  # noqa: N803 — C++ member name preserved
        mixed_sigma: float,
        kappa: float,
        theta: float,
    ) -> None:
        # C++ parity: fdmhestonop.cpp:103-111.
        locations = mesher.locations(1)
        self._dy_map: TripleBandLinearOp = (
            SecondDerivativeOp(1, mesher)
            .mult(0.5 * mixed_sigma * mixed_sigma * locations)
            .add(FirstDerivativeOp(1, mesher).mult(kappa * (theta - locations)))
        )
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(1, mesher)
        self._rTS: YieldTermStructure = rTS

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmhestonop.cpp:113-116."""
        r = self._rTS.forward_rate(t1, t2, Compounding.Continuous).rate()
        self._map_t.axpyb(None, self._dy_map, self._dy_map, np.array([-0.5 * r], dtype=np.float64))

    def get_map(self) -> TripleBandLinearOp:
        """# C++ parity: ``FdmHestonVariancePart::getMap``."""
        return self._map_t


class FdmHestonOp:
    """Composite Heston operator on a 2-D (log-spot, variance) grid.

    # C++ parity: ``class FdmHestonOp : public FdmLinearOpComposite``
    # (fdmhestonop.hpp:86-112, fdmhestonop.cpp:122-194). Python satisfies
    # the ``FdmLinearOpComposite`` Protocol structurally.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        heston_process: HestonProcess,
        quanto_helper: _FdmQuantoHelperLike | None = None,
        leverage_fct: LocalVolTermStructure | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        # C++ parity: fdmhestonop.cpp:122-140.
        self._correlation_map: NinePointLinearOp = SecondOrderMixedDerivativeOp(0, 1, mesher).mult(
            heston_process.rho * heston_process.sigma * mixing_factor * mesher.locations(1)
        )
        self._dy_map: FdmHestonVariancePart = FdmHestonVariancePart(
            mesher,
            heston_process.risk_free_rate(),
            heston_process.sigma * mixing_factor,
            heston_process.kappa,
            heston_process.theta,
        )
        self._dx_map: FdmHestonEquityPart = FdmHestonEquityPart(
            mesher,
            heston_process.risk_free_rate(),
            heston_process.dividend_yield(),
            quanto_helper,
            leverage_fct,
        )

    def size(self) -> int:
        """# C++ parity: fdmhestonop.cpp:148-150 — always 2."""
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmhestonop.cpp:143-146."""
        self._dx_map.set_time(t1, t2)
        self._dy_map.set_time(t1, t2)

    def apply(self, r: Array) -> Array:
        """# C++ parity: fdmhestonop.cpp:152-155."""
        return (
            self._dy_map.get_map().apply(r)
            + self._dx_map.get_map().apply(r)
            + self._dx_map.get_L() * self._correlation_map.apply(r)
        )

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: fdmhestonop.cpp:157-165."""
        if direction == 0:
            return self._dx_map.get_map().apply(r)
        if direction == 1:
            return self._dy_map.get_map().apply(r)
        qassert.fail("direction too large")

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: fdmhestonop.cpp:167-169."""
        return self._dx_map.get_L() * self._correlation_map.apply(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: fdmhestonop.cpp:171-182."""
        if direction == 0:
            return self._dx_map.get_map().solve_splitting(r, dt, 1.0)
        if direction == 1:
            return self._dy_map.get_map().solve_splitting(r, dt, 1.0)
        qassert.fail("direction too large")

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: fdmhestonop.cpp:184-186."""
        return self.solve_splitting(1, self.solve_splitting(0, r, dt), dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: fdmhestonop.cpp:188-194."""
        return [
            self._dx_map.get_map().to_matrix(),
            self._dy_map.get_map().to_matrix(),
            self._correlation_map.to_matrix(),
        ]


__all__ = [
    "FdmHestonEquityPart",
    "FdmHestonOp",
    "FdmHestonVariancePart",
]
