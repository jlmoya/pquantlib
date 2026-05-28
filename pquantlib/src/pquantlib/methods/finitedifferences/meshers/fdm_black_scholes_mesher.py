"""FdmBlackScholesMesher — 1-D mesher for the BSM process in log-spot.

# C++ parity: ql/methods/finitedifferences/meshers/fdmblackscholesmesher.{hpp,cpp}
# (v1.42.1).

Builds a 1-D mesh in ``log S`` over ``[xMin, xMax]`` where the bounds
are derived from the underlying spot, the forward drift over the
intermediate time steps, and the Black-vol-scaled tail:

    xMin = log(min forward) - sigma * sqrt(T) * norminv(1-eps) * scale
    xMax = log(max forward) + sigma * sqrt(T) * norminv(1-eps) * scale

The strike enters via the at-the-money sigma lookup
``blackVol(maturity, strike)`` and (in C++) via the ``cPoint``
parameter that triggers a ``Concentrating1dMesher`` anchored at
``log(strike)``.

**Carve-out (L5-D):** The Python port supports the **Uniform1dMesher**
path only. ``Concentrating1dMesher`` is deferred to Phase 6 — until
it lands, ``cPoint`` is silently ignored and the mesher always falls
back to the uniform mesh. This converges to the analytic European
price (LOOSE 1e-4 at xGrid=200) — see ``cluster/l5d.json`` ←→
``test_fd_black_scholes_vanilla_engine``.
"""

from __future__ import annotations

import math
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdmBlackScholesMesher(Fdm1dMesher):
    """1-D log-spot mesher anchored around log(spot).

    # C++ parity: ``class FdmBlackScholesMesher : public Fdm1dMesher``.

    Python divergence: ``Concentrating1dMesher`` is deferred (Phase 6
    carve-out) so the ``c_point`` parameter is accepted but ignored.
    The uniform-mesh path matches C++'s fallback exactly.
    """

    def __init__(
        self,
        size: int,
        process: GeneralizedBlackScholesProcess,
        maturity: float,
        strike: float,
        x_min_override: float | None = None,
        x_max_override: float | None = None,
        eps: float = 0.0001,
        scale_factor: float = 1.5,
        c_point: tuple[float | None, float | None] | None = None,
        spot_adjustment: float = 0.0,
    ) -> None:
        super().__init__(size)
        spot = process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")

        # Intermediate-step forward evolution. C++:
        #   intermediateTimeSteps = max(2, int(24 * maturity)).
        intermediate_time_steps: int = max(2, int(24.0 * maturity))
        # Build intermediate time points (linear on (0, T]).
        # Dividends are not supported in the L5-D scope.
        intermediate_steps: list[tuple[float, float]] = [
            ((i + 1) * (maturity / intermediate_time_steps), 0.0) for i in range(intermediate_time_steps)
        ]
        intermediate_steps.sort()

        rts = process.risk_free_rate()
        qts = process.dividend_yield()

        last_div_time = 0.0
        fwd = spot + spot_adjustment
        mi, ma = fwd, fwd
        for div_time, div_amount in intermediate_steps:
            fwd = (
                fwd
                / rts.discount(div_time)
                * rts.discount(last_div_time)
                * qts.discount(div_time)
                / qts.discount(last_div_time)
            )
            mi = min(mi, fwd)
            ma = max(ma, fwd)
            fwd -= div_amount
            mi = min(mi, fwd)
            ma = max(ma, fwd)
            last_div_time = div_time

        # Grid boundaries from sigma * sqrt(T) * norminv(1-eps) * scale.
        norm_inv_eps = InverseCumulativeNormal()(1.0 - eps)
        vol_at_strike = process.black_volatility().black_vol_at_time(maturity, strike, extrapolate=True)
        sigma_sqrt_t = vol_at_strike * math.sqrt(maturity)
        x_min = math.log(mi) - sigma_sqrt_t * norm_inv_eps * scale_factor
        x_max = math.log(ma) + sigma_sqrt_t * norm_inv_eps * scale_factor
        if x_min_override is not None:
            x_min = x_min_override
        if x_max_override is not None:
            x_max = x_max_override

        # C++ branches to Concentrating1dMesher when c_point is given —
        # Python defers that and falls back to uniform unconditionally.
        helper = Uniform1dMesher(x_min, x_max, size)

        # Copy locations + dplus/dminus arrays from the helper.
        self._locations = np.asarray(helper.locations(), dtype=np.float64).copy()
        for i in range(size):
            self._dplus[i] = helper.dplus(i)
            self._dminus[i] = helper.dminus(i)


__all__ = ["FdmBlackScholesMesher"]
