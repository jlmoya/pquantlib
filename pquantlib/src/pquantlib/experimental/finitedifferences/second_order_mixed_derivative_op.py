"""SecondOrderMixedDerivativeOp — second-order mixed-derivative operator.

# C++ parity: ql/methods/finitedifferences/operators/secondordermixedderivativeop.{hpp,cpp}
# (v1.42.1).

Implements the mixed partial-derivative ``d2/(dx_d0 dx_d1)`` on a
non-uniform multi-D grid via a nine-point stencil. At interior
nodes the central stencil is

    a_{ij} = (hp_d0 - h_d0)^a * (hp_d1 - h_d1)^b
             / (zeta(i, h_d0, hp_d0) * phi(j, h_d1, hp_d1))

with appropriate corner / side adjustments to handle boundaries.

Used by ``FdmKlugeExtOUOp`` to add the rho * sigma_x * sigma_u
cross-term to the splitting operator.
"""

from __future__ import annotations

from typing import final

from pquantlib.experimental.finitedifferences.nine_point_linear_op import (
    NinePointLinearOp,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher


@final
class SecondOrderMixedDerivativeOp(NinePointLinearOp):
    """Second-order mixed-derivative operator.

    # C++ parity: ``class SecondOrderMixedDerivativeOp : public NinePointLinearOp``.
    """

    def __init__(self, d0: int, d1: int, mesher: FdmMesher) -> None:  # noqa: PLR0915
        super().__init__(d0, d1, mesher)
        layout = mesher.layout()
        dim_d0 = layout.dim()[d0]
        dim_d1 = layout.dim()[d1]
        last_d0 = dim_d0 - 1
        last_d1 = dim_d1 - 1
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
            if c0 == 0 and c1 == 0:
                # Lower-left corner.
                self._a00[i] = self._a01[i] = self._a02[i] = self._a10[i] = self._a20[i] = 0.0
                # a21 = a12 = -(a11 = a22 = 1/(hp_d0 * hp_d1))
                v = 1.0 / (hp_d0 * hp_d1)
                self._a11[i] = self._a22[i] = v
                self._a21[i] = self._a12[i] = -v
            elif c0 == last_d0 and c1 == 0:
                # Upper-left corner.
                self._a22[i] = self._a21[i] = self._a20[i] = self._a10[i] = self._a00[i] = 0.0
                v = 1.0 / (hm_d0 * hp_d1)
                self._a01[i] = self._a12[i] = v
                self._a11[i] = self._a02[i] = -v
            elif c0 == 0 and c1 == last_d1:
                # Lower-right corner.
                self._a00[i] = self._a01[i] = self._a02[i] = self._a12[i] = self._a22[i] = 0.0
                v = 1.0 / (hp_d0 * hm_d1)
                self._a10[i] = self._a21[i] = v
                self._a20[i] = self._a11[i] = -v
            elif c0 == last_d0 and c1 == last_d1:
                # Upper-right corner.
                self._a20[i] = self._a21[i] = self._a22[i] = self._a12[i] = self._a02[i] = 0.0
                v = 1.0 / (hm_d0 * hm_d1)
                self._a00[i] = self._a11[i] = v
                self._a10[i] = self._a01[i] = -v
            elif c0 == 0:
                # Lower side (c0 == 0, c1 interior).
                self._a00[i] = self._a01[i] = self._a02[i] = 0.0
                a10 = hp_d1 / (hp_d0 * phim1)
                self._a10[i] = a10
                self._a20[i] = -a10
                a21 = (hp_d1 - hm_d1) / (hp_d0 * phi0)
                self._a21[i] = a21
                self._a11[i] = -a21
                a22 = hm_d1 / (hp_d0 * phip1)
                self._a22[i] = a22
                self._a12[i] = -a22
            elif c0 == last_d0:
                # Upper side (c0 == last, c1 interior).
                self._a20[i] = self._a21[i] = self._a22[i] = 0.0
                a00 = hp_d1 / (hm_d0 * phim1)
                self._a00[i] = a00
                self._a10[i] = -a00
                a11 = (hp_d1 - hm_d1) / (hm_d0 * phi0)
                self._a11[i] = a11
                self._a01[i] = -a11
                a12 = hm_d1 / (hm_d0 * phip1)
                self._a12[i] = a12
                self._a02[i] = -a12
            elif c1 == 0:
                # Left side (c1 == 0, c0 interior).
                self._a00[i] = self._a10[i] = self._a20[i] = 0.0
                a01 = hp_d0 / (zetam1 * hp_d1)
                self._a01[i] = a01
                self._a02[i] = -a01
                a12 = (hp_d0 - hm_d0) / (zeta0 * hp_d1)
                self._a12[i] = a12
                self._a11[i] = -a12
                a22 = hm_d0 / (zetap1 * hp_d1)
                self._a22[i] = a22
                self._a21[i] = -a22
            elif c1 == last_d1:
                # Right side (c1 == last, c0 interior).
                self._a22[i] = self._a12[i] = self._a02[i] = 0.0
                a00 = hp_d0 / (zetam1 * hm_d1)
                self._a00[i] = a00
                self._a01[i] = -a00
                a11 = (hp_d0 - hm_d0) / (zeta0 * hm_d1)
                self._a11[i] = a11
                self._a10[i] = -a11
                a21 = hm_d0 / (zetap1 * hm_d1)
                self._a21[i] = a21
                self._a20[i] = -a21
            else:
                # Pure interior.
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
