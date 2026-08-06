"""FdmSabrOp — FDM operator for the SABR model (absorbing boundary at f=0).

# C++ parity: ql/methods/finitedifferences/operators/fdmsabrop.{hpp,cpp}
# (v1.43).

.. math::

    df_t     &= \\alpha_t f_t^\\beta \\, dW_t \\\\
    d\\alpha_t &= \\nu \\alpha_t \\, dZ_t \\\\
    \\rho\\,dt  &= \\langle dW_t, dZ_t \\rangle

Direction 0 is the forward ``f``; direction 1 is ``x = log alpha``
(hence the ``exp`` factors on the mesher's second axis). The
discount term ``-r`` is split evenly between the two directional
operators (``-0.5 r`` each), so ``apply`` reproduces ``-r`` in total.
"""

from __future__ import annotations

from typing import final

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
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@final
class FdmSabrOp:
    """SABR finite-difference operator.

    # C++ parity: ``class FdmSabrOp : public FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        r_ts: YieldTermStructure,
        f0: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float,
    ) -> None:
        """Build the operator.

        ``f0`` and ``alpha`` are accepted for signature parity and never
        read — the C++ constructor takes both and uses neither
        (fdmsabrop.cpp:34-49); the volatility level enters through
        ``exp(mesher->locations(1))``.
        """
        del f0, alpha  # C++ parity: parameters accepted and discarded.
        self._r_ts: YieldTermStructure = r_ts
        layout = mesher.layout()
        size = layout.size()
        loc0 = mesher.locations(0)
        loc1 = mesher.locations(1)

        self._dff_map: TripleBandLinearOp = SecondDerivativeOp(0, mesher).mult(
            0.5 * np.exp(2.0 * loc1) * np.power(loc0, 2.0 * beta)
        )
        self._dx_map: TripleBandLinearOp = FirstDerivativeOp(1, mesher).mult(
            np.full(size, -0.5 * nu * nu, dtype=np.float64)
        )
        self._dxx_map: TripleBandLinearOp = SecondDerivativeOp(1, mesher).mult(
            np.full(size, 0.5 * nu * nu, dtype=np.float64)
        )
        self._correlation_map: NinePointLinearOp = SecondOrderMixedDerivativeOp(
            0, 1, mesher
        ).mult(rho * nu * np.exp(loc1) * np.power(loc0, beta))

        self._map_f: TripleBandLinearOp = TripleBandLinearOp(0, mesher)
        self._map_a: TripleBandLinearOp = TripleBandLinearOp(1, mesher)

    def size(self) -> int:
        """# C++ parity: ``FdmSabrOp::size`` — always 2."""
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``FdmSabrOp::setTime``."""
        r = self._r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        half_r = np.array([-0.5 * r], dtype=np.float64)
        # C++: mapF_.axpyb(Array(), dffMap_, dffMap_, Array(1, -0.5*r));
        self._map_f.axpyb(None, None, self._dff_map, half_r)
        # C++: mapA_.axpyb(Array(1, 1.0), dxMap_, dxxMap_, Array(1, -0.5*r));
        self._map_a.axpyb(
            np.array([1.0], dtype=np.float64), self._dx_map, self._dxx_map, half_r
        )

    def apply(self, u: Array) -> Array:
        """# C++ parity: ``FdmSabrOp::apply``."""
        return self._map_f.apply(u) + self._map_a.apply(u) + self._correlation_map.apply(u)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: ``FdmSabrOp::apply_mixed``."""
        return self._correlation_map.apply(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: ``FdmSabrOp::apply_direction``."""
        if direction == 0:
            return self._map_f.apply(r)
        if direction == 1:
            return self._map_a.apply(r)
        return qassert.fail("direction too large")

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmSabrOp::solve_splitting``."""
        if direction == 0:
            return self._map_f.solve_splitting(r, dt, 1.0)
        if direction == 1:
            return self._map_a.solve_splitting(r, dt, 1.0)
        return qassert.fail("direction too large")

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmSabrOp::preconditioner`` — solve both directions."""
        return self.solve_splitting(1, self.solve_splitting(0, r, dt), dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: ``FdmSabrOp::toMatrixDecomp``.

        Order is ``[mapA_, mapF_, correlationMap_]`` — the alpha
        direction comes **first**, unlike the ``apply`` order.
        """
        return [
            self._map_a.to_matrix(),
            self._map_f.to_matrix(),
            self._correlation_map.to_matrix(),
        ]


__all__ = ["FdmSabrOp"]
