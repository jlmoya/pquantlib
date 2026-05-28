"""SecondDerivativeOp — central second-derivative banded operator.

# C++ parity: ql/methods/finitedifferences/operators/secondderivativeop.{hpp,cpp}
# (v1.42.1).

On a non-uniform grid with backward step ``hm`` and forward step
``hp`` at the interior, the central second-derivative stencil is::

    lower =  2 / (hm * (hm + hp))
    diag  = -2 / (hm * hp)
    upper =  2 / (hp * (hm + hp))

At the boundaries (coord = 0 or coord = dim - 1) the stencil is
*zero everywhere* — Dirichlet boundary handling is delegated to the
caller (i.e. the BSM operator builds the boundary in via the drift
+ rate terms, and the no-arbitrage tails decay).
"""

from __future__ import annotations

from typing import final

from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)


@final
class SecondDerivativeOp(TripleBandLinearOp):
    """Central second-derivative operator (zero at boundaries).

    # C++ parity: ``class SecondDerivativeOp : public TripleBandLinearOp``.
    """

    def __init__(self, direction: int, mesher: FdmMesher) -> None:
        super().__init__(direction, mesher)
        layout = mesher.layout()
        last_coord = layout.dim()[direction] - 1
        for iter_ in layout.iter():
            i = iter_.index
            coord = iter_.coordinates[direction]
            if coord in (0, last_coord):
                self._lower[i] = 0.0
                self._diag[i] = 0.0
                self._upper[i] = 0.0
            else:
                hm = mesher.dminus(iter_, direction)
                hp = mesher.dplus(iter_, direction)
                zetam1 = hm * (hm + hp)
                zeta0 = hm * hp
                zetap1 = hp * (hm + hp)
                self._lower[i] = 2.0 / zetam1
                self._diag[i] = -2.0 / zeta0
                self._upper[i] = 2.0 / zetap1


__all__ = ["SecondDerivativeOp"]
