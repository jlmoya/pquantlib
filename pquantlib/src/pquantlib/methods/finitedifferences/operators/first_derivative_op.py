"""FirstDerivativeOp — central first-derivative banded operator.

# C++ parity: ql/methods/finitedifferences/operators/firstderivativeop.{hpp,cpp}
# (v1.42.1).

On a non-uniform grid with backward step ``hm`` and forward step
``hp`` at the interior, the central first-derivative stencil is::

    lower = -hp / (hm * (hm + hp))
    diag  = (hp - hm) / (hm * hp)
    upper = hm / (hp * (hm + hp))

At the first node (coord=0 along the direction) an *upwinding* (one-sided
forward) stencil is used::

    lower = 0
    diag  = -1/hp
    upper = 1/hp

At the last node, a *downwinding* (one-sided backward) stencil::

    lower = -1/hm
    diag  = 1/hm
    upper = 0
"""

from __future__ import annotations

from typing import final

from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)


@final
class FirstDerivativeOp(TripleBandLinearOp):
    """Central first-derivative operator (with one-sided boundaries).

    # C++ parity: ``class FirstDerivativeOp : public TripleBandLinearOp``.
    """

    def __init__(self, direction: int, mesher: FdmMesher) -> None:
        super().__init__(direction, mesher)
        layout = mesher.layout()
        last_coord = layout.dim()[direction] - 1
        for iter_ in layout.iter():
            i = iter_.index
            hm = mesher.dminus(iter_, direction)
            hp = mesher.dplus(iter_, direction)
            coord = iter_.coordinates[direction]
            if coord == 0:
                # Upwinding (forward) stencil at the lower boundary.
                self._lower[i] = 0.0
                self._upper[i] = 1.0 / hp
                self._diag[i] = -self._upper[i]
            elif coord == last_coord:
                # Downwinding (backward) stencil at the upper boundary.
                self._diag[i] = 1.0 / hm
                self._lower[i] = -self._diag[i]
                self._upper[i] = 0.0
            else:
                zetam1 = hm * (hm + hp)
                zeta0 = hm * hp
                zetap1 = hp * (hm + hp)
                self._lower[i] = -hp / zetam1
                self._diag[i] = (hp - hm) / zeta0
                self._upper[i] = hm / zetap1


__all__ = ["FirstDerivativeOp"]
