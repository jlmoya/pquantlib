"""FdmBlackScholesMultiStrikeMesher — log-spot mesher spanning several strikes.

# C++ parity: ql/methods/finitedifferences/meshers/fdmblackscholesmultistrikemesher.{hpp,cpp}
# (v1.43).

Used when one FD grid must price a whole strike ladder. The grid is built in
``x = ln(S)`` over

    Fmin = S0^2/Kmax * d,   Fmax = S0^2/Kmin * d,   d = q(T)/r(T)

widened by ``scaleFactor * N^{-1}(1-eps)`` standard deviations at the vol of
the nearest strike, and floored/capped by the ``0.8*log(0.8 S0^2/Kmax)`` /
``1.2*log(0.8 S0^2/Kmin)`` heuristics C++ applies. Note the asymmetry
(``0.8`` inside both logs, ``0.8`` vs ``1.2`` outside) is C++'s, not a typo
here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, final

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher

if TYPE_CHECKING:
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )

_INV_CUM_NORMAL = InverseCumulativeNormal()


@final
class FdmBlackScholesMultiStrikeMesher(Fdm1dMesher):
    """Log-spot mesher wide enough for a whole strike ladder.

    # C++ parity: ``class FdmBlackScholesMultiStrikeMesher : public Fdm1dMesher``.
    """

    def __init__(
        self,
        size: int,
        process: GeneralizedBlackScholesProcess,
        maturity: float,
        strikes: Sequence[float],
        eps: float = 0.0001,
        scale_factor: float = 1.5,
        c_point: tuple[float | None, float | None] = (None, None),
    ) -> None:
        super().__init__(size)

        spot = process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")

        d = process.dividend_yield().discount(maturity) / process.risk_free_rate().discount(
            maturity
        )
        min_strike = min(strikes)
        max_strike = max(strikes)

        f_min = spot * spot / max_strike * d
        f_max = spot * spot / min_strike * d
        qassert.require(f_min > 0.0, "negative forward given")

        norm_inv_eps = _INV_CUM_NORMAL(1.0 - eps)
        vol_ts = process.black_volatility()
        sigma_sqrt_t_min = vol_ts.black_vol_at_time(maturity, min_strike) * math.sqrt(maturity)
        sigma_sqrt_t_max = vol_ts.black_vol_at_time(maturity, max_strike) * math.sqrt(maturity)

        x_min = min(
            0.8 * math.log(0.8 * spot * spot / max_strike),
            math.log(f_min)
            - sigma_sqrt_t_min * norm_inv_eps * scale_factor
            - sigma_sqrt_t_min * sigma_sqrt_t_min / 2.0,
        )
        x_max = max(
            1.2 * math.log(0.8 * spot * spot / min_strike),
            math.log(f_max)
            + sigma_sqrt_t_max * norm_inv_eps * scale_factor
            - sigma_sqrt_t_max * sigma_sqrt_t_max / 2.0,
        )

        helper: Fdm1dMesher
        if c_point[0] is not None and x_min <= math.log(c_point[0]) <= x_max:
            helper = Concentrating1dMesher(
                x_min, x_max, size, (math.log(c_point[0]), c_point[1])
            )
        else:
            helper = Uniform1dMesher(x_min, x_max, size)

        self._locations[:] = helper.locations()
        for i in range(size):
            self._dplus[i] = helper.dplus(i)
            self._dminus[i] = helper.dminus(i)


__all__ = ["FdmBlackScholesMultiStrikeMesher"]
