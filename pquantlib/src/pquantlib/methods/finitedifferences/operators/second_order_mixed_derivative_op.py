"""SecondOrderMixedDerivativeOp — 2-D mixed-derivative operator.

# C++ parity: ql/methods/finitedifferences/operators/secondordermixedderivativeop.{hpp,cpp}
# (v1.42.1).

On a non-uniform 2-D grid with backward / forward steps ``hm_d0``,
``hp_d0`` along direction ``d0`` and ``hm_d1``, ``hp_d1`` along
direction ``d1``, the centered mixed-derivative stencil is the
tensor product of the two 1-D centered first-derivative stencils.

The constructor fills the nine coefficient arrays per the formulas
in the C++ reference; at boundaries (corners, edges) one-sided
stencils with the appropriate zero rows are used.
"""

from __future__ import annotations

from typing import final

from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.nine_point_linear_op import (
    NinePointLinearOp,
)


@final
class SecondOrderMixedDerivativeOp(NinePointLinearOp):
    """2-D mixed-derivative operator.

    # C++ parity: ``class SecondOrderMixedDerivativeOp : public NinePointLinearOp``.
    """

    def __init__(self, d0: int, d1: int, mesher: FdmMesher) -> None:  # noqa: PLR0915 (per-case stencil branches)
        super().__init__(d0, d1, mesher)
        layout = mesher.layout()
        dim = layout.dim()
        for iter_ in layout.iter():
            i = iter_.index
            hm_d0 = mesher.dminus(iter_, d0)
            hp_d0 = mesher.dplus(iter_, d0)
            hm_d1 = mesher.dminus(iter_, d1)
            hp_d1 = mesher.dplus(iter_, d1)
            zetam1 = hm_d0 * (hm_d0 + hp_d0)
            zeta0 = hm_d0 * hp_d0
            zetap1 = hp_d0 * (hm_d0 + hp_d0)
            phim1 = hm_d1 * (hm_d1 + hp_d1)
            phi0 = hm_d1 * hp_d1
            phip1 = hp_d1 * (hm_d1 + hp_d1)

            c0 = iter_.coordinates[d0]
            c1 = iter_.coordinates[d1]
            last0 = dim[d0] - 1
            last1 = dim[d1] - 1
            if c0 == 0 and c1 == 0:
                # Lower-left corner.
                self._a00[i] = self._a01[i] = self._a02[i] = self._a10[i] = self._a20[i] = 0.0
                self._a11[i] = self._a22[i] = 1.0 / (hp_d0 * hp_d1)
                self._a21[i] = self._a12[i] = -self._a11[i]
            elif c0 == last0 and c1 == 0:
                # Upper-left corner.
                self._a22[i] = self._a21[i] = self._a20[i] = self._a10[i] = self._a00[i] = 0.0
                self._a01[i] = self._a12[i] = 1.0 / (hm_d0 * hp_d1)
                self._a11[i] = self._a02[i] = -self._a01[i]
            elif c0 == 0 and c1 == last1:
                # Lower-right corner.
                self._a00[i] = self._a01[i] = self._a02[i] = self._a12[i] = self._a22[i] = 0.0
                self._a10[i] = self._a21[i] = 1.0 / (hp_d0 * hm_d1)
                self._a20[i] = self._a11[i] = -self._a10[i]
            elif c0 == last0 and c1 == last1:
                # Upper-right corner.
                self._a20[i] = self._a21[i] = self._a22[i] = self._a12[i] = self._a02[i] = 0.0
                self._a00[i] = self._a11[i] = 1.0 / (hm_d0 * hm_d1)
                self._a10[i] = self._a01[i] = -self._a00[i]
            elif c0 == 0:
                # Lower side (c0 == 0, c1 interior).
                self._a00[i] = self._a01[i] = self._a02[i] = 0.0
                self._a10[i] = hp_d1 / (hp_d0 * phim1)
                self._a20[i] = -self._a10[i]
                self._a21[i] = (hp_d1 - hm_d1) / (hp_d0 * phi0)
                self._a11[i] = -self._a21[i]
                self._a22[i] = hm_d1 / (hp_d0 * phip1)
                self._a12[i] = -self._a22[i]
            elif c0 == last0:
                # Upper side (c0 == last0, c1 interior).
                self._a20[i] = self._a21[i] = self._a22[i] = 0.0
                self._a00[i] = hp_d1 / (hm_d0 * phim1)
                self._a10[i] = -self._a00[i]
                self._a11[i] = (hp_d1 - hm_d1) / (hm_d0 * phi0)
                self._a01[i] = -self._a11[i]
                self._a12[i] = hm_d1 / (hm_d0 * phip1)
                self._a02[i] = -self._a12[i]
            elif c1 == 0:
                # Left side (c1 == 0, c0 interior).
                self._a00[i] = self._a10[i] = self._a20[i] = 0.0
                self._a01[i] = hp_d0 / (zetam1 * hp_d1)
                self._a02[i] = -self._a01[i]
                self._a12[i] = (hp_d0 - hm_d0) / (zeta0 * hp_d1)
                self._a11[i] = -self._a12[i]
                self._a22[i] = hm_d0 / (zetap1 * hp_d1)
                self._a21[i] = -self._a22[i]
            elif c1 == last1:
                # Right side (c1 == last1, c0 interior).
                self._a22[i] = self._a12[i] = self._a02[i] = 0.0
                self._a00[i] = hp_d0 / (zetam1 * hm_d1)
                self._a01[i] = -self._a00[i]
                self._a11[i] = (hp_d0 - hm_d0) / (zeta0 * hm_d1)
                self._a10[i] = -self._a11[i]
                self._a21[i] = hm_d0 / (zetap1 * hm_d1)
                self._a20[i] = -self._a21[i]
            else:
                # Interior — tensor product of central 1st-derivative stencils.
                self._a00[i] = hp_d0 * hp_d1 / (zetam1 * phim1)
                self._a10[i] = -(hp_d0 - hm_d0) * hp_d1 / (zeta0 * phim1)
                self._a20[i] = -hm_d0 * hp_d1 / (zetap1 * phim1)
                self._a01[i] = -hp_d0 * (hp_d1 - hm_d1) / (zetam1 * phi0)
                self._a11[i] = (hp_d0 - hm_d0) * (hp_d1 - hm_d1) / (zeta0 * phi0)
                self._a21[i] = hm_d0 * (hp_d1 - hm_d1) / (zetap1 * phi0)
                self._a02[i] = -hp_d0 * hm_d1 / (zetam1 * phip1)
                self._a12[i] = hm_d1 * (hp_d0 - hm_d0) / (zeta0 * phip1)
                self._a22[i] = hm_d0 * hm_d1 / (zetap1 * phip1)


__all__ = ["SecondOrderMixedDerivativeOp"]
