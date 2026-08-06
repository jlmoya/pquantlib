"""FdmWienerOp — multi-dimensional (driftless) Wiener operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmwienerop.{hpp,cpp}
# (v1.43).

For an ``n``-dimensional grid the operator is

.. math::

    L = -r I + \\sum_i \\tfrac{1}{2}\\lambda_i \\partial_{x_i x_i}

i.e. one independent (uncorrelated) diffusion per direction plus a
discount term taken from the risk-free curve. ``rTS`` may be ``None``,
in which case ``r`` stays 0 (C++ tests the shared pointer for null in
``setTime``).
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
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@final
class FdmWienerOp:
    """Uncorrelated multi-dimensional diffusion operator.

    # C++ parity: ``class FdmWienerOp : public FdmLinearOpComposite``.
    # Python composes rather than inherits — the composite surface is the
    # structural ``FdmLinearOpComposite`` Protocol.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        r_ts: YieldTermStructure | None,
        lambdas: Array,
    ) -> None:
        layout = mesher.layout()
        lambdas_arr = np.asarray(lambdas, dtype=np.float64)
        qassert.require(
            len(layout.dim()) == lambdas_arr.shape[0],
            "mesher and lambdas need to be of the same dimension",
        )

        self._r_ts: YieldTermStructure | None = r_ts
        self._r: float = 0.0
        size = layout.size()
        # C++ parity: TripleBandLinearOp(SecondDerivativeOp(i, mesher)
        # .mult(Array(size, 0.5*lambdas[i]))).
        self._ops: list[TripleBandLinearOp] = [
            SecondDerivativeOp(i, mesher).mult(
                np.full(size, 0.5 * float(lambdas_arr[i]), dtype=np.float64)
            )
            for i in range(lambdas_arr.shape[0])
        ]

    def size(self) -> int:
        """# C++ parity: ``FdmWienerOp::size`` — one direction per lambda."""
        return len(self._ops)

    def set_time(self, t1: float, t2: float) -> None:
        """# C++ parity: ``FdmWienerOp::setTime`` — no-op when ``rTS`` is null."""
        if self._r_ts is not None:
            self._r = self._r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()

    def apply(self, x: Array) -> Array:
        """# C++ parity: ``FdmWienerOp::apply`` — ``-r*x + sum_i L_i x``."""
        y: Array = -self._r * x
        for op in self._ops:
            y = y + op.apply(x)
        return y

    def apply_mixed(self, x: Array) -> Array:
        """# C++ parity: ``FdmWienerOp::apply_mixed`` — always zero."""
        return np.zeros_like(x)

    def apply_direction(self, direction: int, x: Array) -> Array:
        """# C++ parity: ``FdmWienerOp::apply_direction``."""
        return self._ops[direction].apply(x)

    def solve_splitting(self, direction: int, x: Array, dt: float) -> Array:
        """# C++ parity: ``FdmWienerOp::solve_splitting``."""
        return self._ops[direction].solve_splitting(x, dt, 1.0)

    def preconditioner(self, r: Array, dt: float) -> Array:
        """# C++ parity: ``FdmWienerOp::preconditioner`` — solve along 0."""
        return self.solve_splitting(0, r, dt)

    def to_matrix_decomp(self) -> list[csr_matrix]:
        """# C++ parity: ``FdmWienerOp::toMatrixDecomp`` — one per direction."""
        return [op.to_matrix() for op in self._ops]


__all__ = ["FdmWienerOp"]
