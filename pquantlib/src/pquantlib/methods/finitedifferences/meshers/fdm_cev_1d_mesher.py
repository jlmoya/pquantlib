"""FdmCEV1dMesher — 1-D mesher for a CEV process.

# C++ parity: ql/methods/finitedifferences/meshers/fdmcev1dmesher.{hpp,cpp}
# (v1.43).

Boundaries come straight from ``CEVRNDCalculator``: the upper bound is
``scaleFactor * invcdf(1-eps, T)``; the lower bound is 0 (or ``QL_EPSILON``
for ``beta < 0``) when the absorbing mass at zero already exceeds ``eps``,
otherwise ``invcdf(eps, T)/scaleFactor``. Inside those bounds it delegates to
``Concentrating1dMesher`` if a usable critical point was supplied, else to
``Uniform1dMesher``.
"""

from __future__ import annotations

from typing import final

from pquantlib.math.constants import QL_EPSILON
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.utilities.cev_rnd_calculator import (
    CEVRNDCalculator,
)


@final
class FdmCEV1dMesher(Fdm1dMesher):
    """1-D mesher spanning the CEV terminal distribution.

    # C++ parity: ``class FdmCEV1dMesher : public Fdm1dMesher``.
    """

    def __init__(
        self,
        size: int,
        f0: float,
        alpha: float,
        beta: float,
        maturity: float,
        eps: float = 0.0001,
        scale_factor: float = 1.5,
        c_point: tuple[float | None, float | None] = (None, None),
    ) -> None:
        super().__init__(size)

        rnd = CEVRNDCalculator(f0, alpha, beta)

        upper_bound = scale_factor * rnd.invcdf(1.0 - eps, maturity)
        mass_at_zero = rnd.mass_at_zero(maturity)

        if mass_at_zero > eps:
            lower_bound = QL_EPSILON if beta < 0.0 else 0.0
        else:
            lower_bound = rnd.invcdf(eps, maturity) / scale_factor

        helper: Fdm1dMesher
        if c_point[0] is not None and lower_bound <= c_point[0] <= upper_bound:
            helper = Concentrating1dMesher(lower_bound, upper_bound, size, c_point)
        else:
            helper = Uniform1dMesher(lower_bound, upper_bound, size)

        self._locations[:] = helper.locations()
        for i in range(size):
            self._dplus[i] = helper.dplus(i)
            self._dminus[i] = helper.dminus(i)


__all__ = ["FdmCEV1dMesher"]
