"""FdmHullWhiteOp — FDM operator for the Hull-White short-rate model.

# C++ parity: ql/methods/finitedifferences/operators/fdmhullwhiteop.{hpp,cpp}
# (v1.43).

The Hull-White short rate ``r_t = x_t + phi(t)`` with

.. math::

    dx_t = -a x_t dt + \\sigma dW_t

gives the spatial operator, on the ``x`` axis of a (possibly
multi-dimensional) grid,

.. math::

    L = -a x \\, D_x + \\tfrac{1}{2}\\sigma^2 D_{xx} - (x + \\varphi) I

where ``phi`` is frozen at the midpoint of the step,
``phi = 0.5 * (r(t1, 0) + r(t2, 0))``, and the discount bias ``-(x + phi)``
is added to the diagonal at every ``set_time``.
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
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite


class FdmHullWhiteOp:
    """FDM operator for the Hull-White model along a single direction.

    # C++ parity: ``class FdmHullWhiteOp : public FdmLinearOpComposite``
    # (fdmhullwhiteop.hpp:35-62). Python satisfies the
    # ``FdmLinearOpComposite`` Protocol structurally instead of
    # inheriting from it.
    """

    def __init__(self, mesher: FdmMesher, model: HullWhite, direction: int) -> None:
        # C++ parity: fdmhullwhiteop.cpp:32-44.
        self._direction: int = direction
        self._x: Array = mesher.locations(direction)
        size = mesher.layout().size()
        self._dz_map: TripleBandLinearOp = (
            FirstDerivativeOp(direction, mesher)
            .mult(-self._x * model.a())
            .add(
                SecondDerivativeOp(direction, mesher).mult(
                    0.5 * model.sigma() * model.sigma() * np.ones(size, dtype=np.float64)
                )
            )
        )
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(direction, mesher)
        self._model: HullWhite = model

    def size(self) -> int:
        """# C++ parity: fdmhullwhiteop.cpp:46 — always ``1U``."""
        return 1

    def set_time(self, t1: float, t2: float) -> None:
        """Rebuild the operator over ``[t1, t2]`` (``t1 <= t2`` required).

        # C++ parity: fdmhullwhiteop.cpp:48-57.
        """
        dynamics = self._model.dynamics()
        phi = 0.5 * (dynamics.short_rate(t1, 0.0) + dynamics.short_rate(t2, 0.0))
        # C++ passes an empty ``Array()`` for ``a`` — the ``a*x`` term is
        # skipped and ``dzMap_`` is used only as the ``y`` operand.
        self._map_t.axpyb(None, self._dz_map, self._dz_map, -(self._x + phi))

    def apply(self, r: Array) -> Array:
        """# C++ parity: fdmhullwhiteop.cpp:59-61."""
        return self._map_t.apply(r)

    def apply_mixed(self, r: Array) -> Array:
        """# C++ parity: fdmhullwhiteop.cpp:63-65 — an all-zero array."""
        return np.zeros_like(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """# C++ parity: fdmhullwhiteop.cpp:67-73."""
        if direction == self._direction:
            return self._map_t.apply(r)
        return np.zeros_like(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """# C++ parity: fdmhullwhiteop.cpp:75-82.

        Note the C++ returns an all-**zero** array (not ``r``) for any
        direction other than the operator's own.
        """
        if direction == self._direction:
            return self._map_t.solve_splitting(r, dt, 1.0)
        return np.zeros_like(r)

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: fdmhullwhiteop.cpp:84-86."""
        return self.solve_splitting(self._direction, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: fdmhullwhiteop.cpp:88-90 — a single-element vector."""
        return [self._map_t.to_matrix()]


__all__ = ["FdmHullWhiteOp"]
