"""FdmG2Op — FDM operator for the two-factor G2++ short-rate model.

# C++ parity: ql/methods/finitedifferences/operators/fdmg2op.{hpp,cpp}
# (v1.43).

G2++ writes the short rate as ``r_t = x_t + y_t + phi(t)`` with

.. math::

    dx_t = -a x_t dt + \\sigma dW^1_t, \\qquad
    dy_t = -b y_t dt + \\eta dW^2_t, \\qquad
    dW^1 dW^2 = \\rho \\, dt

so the spatial operator splits into two banded directional pieces plus a
nine-point correlation piece:

.. math::

    L_x = -a x D_x + \\tfrac{1}{2}\\sigma^2 D_{xx}, \\quad
    L_y = -b y D_y + \\tfrac{1}{2}\\eta^2 D_{yy}, \\quad
    L_{xy} = \\rho \\sigma \\eta D_{xy}

The discount bias ``-0.5 (x + y + phi)`` is added to **both** directional
diagonals at every ``set_time`` (so that ``apply`` sums to the full
``-(x + y + phi)``), with ``phi`` frozen at the step midpoint.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

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
from pquantlib.models.shortrate.twofactor.g2 import G2


class FdmG2Op:
    """FDM operator for the G2++ model on two grid directions.

    # C++ parity: ``class FdmG2Op : public FdmLinearOpComposite``
    # (fdmg2op.hpp:36-64). Python satisfies the
    # ``FdmLinearOpComposite`` Protocol structurally.
    """

    def __init__(self, mesher: FdmMesher, model: G2, direction1: int, direction2: int) -> None:
        # C++ parity: fdmg2op.cpp:34-56.
        self._direction1: int = direction1
        self._direction2: int = direction2
        self._x: Array = mesher.locations(direction1)
        self._y: Array = mesher.locations(direction2)
        size = mesher.layout().size()
        ones = np.ones(size, dtype=np.float64)

        self._dx_map: TripleBandLinearOp = (
            FirstDerivativeOp(direction1, mesher)
            .mult(-self._x * model.a())
            .add(SecondDerivativeOp(direction1, mesher).mult(0.5 * model.sigma() * model.sigma() * ones))
        )
        self._dy_map: TripleBandLinearOp = (
            FirstDerivativeOp(direction2, mesher)
            .mult(-self._y * model.b())
            .add(SecondDerivativeOp(direction2, mesher).mult(0.5 * model.eta() * model.eta() * ones))
        )
        self._corr_map: NinePointLinearOp = SecondOrderMixedDerivativeOp(direction1, direction2, mesher).mult(
            np.full(size, model.rho() * model.sigma() * model.eta(), dtype=np.float64)
        )

        self._map_x: TripleBandLinearOp = TripleBandLinearOp(direction1, mesher)
        self._map_y: TripleBandLinearOp = TripleBandLinearOp(direction2, mesher)
        self._model: G2 = model

    def size(self) -> int:
        """# C++ parity: fdmg2op.cpp:58 — always ``2U``."""
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: fdmg2op.cpp:60-71."""
        dynamics = self._model.dynamics()
        phi = 0.5 * (dynamics.short_rate(t1, 0.0, 0.0) + dynamics.short_rate(t2, 0.0, 0.0))
        hr = -0.5 * (self._x + self._y + phi)
        # C++ passes an empty ``Array()`` for ``a``: the ``a*x`` term is
        # skipped and ``dxMap_`` / ``dyMap_`` act only as the ``y`` operand.
        self._map_x.axpyb(None, self._dx_map, self._dx_map, hr)
        self._map_y.axpyb(None, self._dy_map, self._dy_map, hr)

    def apply(self, r: Array) -> Array:
        """# C++ parity: fdmg2op.cpp:73-75."""
        return self._map_x.apply(r) + self._map_y.apply(r) + self.apply_mixed(r)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: fdmg2op.cpp:77-79."""
        return self._corr_map.apply(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: fdmg2op.cpp:81-91."""
        if direction == self._direction1:
            return self._map_x.apply(r)
        if direction == self._direction2:
            return self._map_y.apply(r)
        return np.zeros_like(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: fdmg2op.cpp:93-103."""
        if direction == self._direction1:
            return self._map_x.solve_splitting(r, dt, 1.0)
        if direction == self._direction2:
            return self._map_y.solve_splitting(r, dt, 1.0)
        return np.zeros_like(r)

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: fdmg2op.cpp:105-107."""
        return self.solve_splitting(self._direction1, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: fdmg2op.cpp:109-115."""
        return [
            self._map_x.to_matrix(),
            self._map_y.to_matrix(),
            self._corr_map.to_matrix(),
        ]


__all__ = ["FdmG2Op"]
