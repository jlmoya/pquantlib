"""FdmCEVOp — FDM operator for the CEV model (absorbing boundary at f=0).

# C++ parity: ql/methods/finitedifferences/operators/fdmcevop.{hpp,cpp}
# (v1.43).

Constant-elasticity-of-variance process

.. math::

    df_t = \\alpha f_t^\\beta \\, dW_t

whose (discounted) backward operator on the ``f`` grid is

.. math::

    L = \\tfrac{1}{2}\\alpha^2 f^{2\\beta}\\, \\partial_{ff} - r I
"""

from __future__ import annotations

from typing import final

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@final
class FdmCEVOp:
    """CEV finite-difference operator.

    # C++ parity: ``class FdmCEVOp : public FdmLinearOpComposite``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        r_ts: YieldTermStructure,
        f0: float,
        alpha: float,
        beta: float,
        direction: int,
    ) -> None:
        """Build the operator.

        ``f0`` is accepted for signature parity and never read — the C++
        constructor takes it and does not store or use it
        (fdmcevop.cpp:32-43).

        .. note::
           The second-derivative operator is built along direction **0**
           while the coefficient array and the splitting operator use
           ``direction``. That asymmetry is in the C++ source verbatim
           (``SecondDerivativeOp(0, mesher).mult(... locations(direction) ...)``
           with ``mapT_(direction, mesher)``) and is reproduced here; it
           only matters when ``direction != 0``.
        """
        del f0  # C++ parity: parameter accepted and discarded.
        self._r_ts: YieldTermStructure = r_ts
        self._direction: int = direction
        self._dxx_map: TripleBandLinearOp = SecondDerivativeOp(0, mesher).mult(
            0.5 * alpha * alpha * np.power(mesher.locations(direction), 2.0 * beta)
        )
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)

    def size(self) -> int:
        """# C++ parity: ``FdmCEVOp::size`` — always 1."""
        return 1

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``FdmCEVOp::setTime``.

        ``mapT_.axpyb(Array(), dxxMap_, dxxMap_, Array(1, -r))`` — the
        empty ``a`` skips the ``a*x`` term entirely, so the result is
        ``dxxMap_`` with ``-r`` added to the diagonal.
        """
        r = self._r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        self._map_t.axpyb(
            None, None, self._dxx_map, np.array([-r], dtype=np.float64)
        )

    def apply(self, r: Array) -> Array:
        """# C++ parity: ``FdmCEVOp::apply``."""
        return self._map_t.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: ``FdmCEVOp::apply_mixed`` — zero."""
        return np.zeros_like(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: ``FdmCEVOp::apply_direction``."""
        if direction == self._direction:
            return self._map_t.apply(r)
        return np.zeros_like(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmCEVOp::solve_splitting``.

        Unlike most forward operators, the off-direction branch returns
        **zero**, not ``r`` (fdmcevop.cpp:69-76).
        """
        if direction == self._direction:
            return self._map_t.solve_splitting(r, dt, 1.0)
        return np.zeros_like(r)

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmCEVOp::preconditioner``."""
        return self.solve_splitting(self._direction, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: ``FdmCEVOp::toMatrixDecomp``."""
        return [self._map_t.to_matrix()]


__all__ = ["FdmCEVOp"]
