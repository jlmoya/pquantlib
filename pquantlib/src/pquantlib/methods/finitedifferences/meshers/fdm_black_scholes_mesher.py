"""FdmBlackScholesMesher — 1-D mesher for the BSM process in log-spot.

# C++ parity: ql/methods/finitedifferences/meshers/fdmblackscholesmesher.{hpp,cpp}
# @ v1.43 (6b57206e0).

Builds a 1-D mesh in ``log S`` over ``[xMin, xMax]`` where the bounds
are derived from the underlying spot, the forward drift over the
intermediate time steps, and the Black-vol-scaled tail:

    xMin = log(min forward) - sigma * sqrt(T) * norminv(1-eps) * scale
    xMax = log(max forward) + sigma * sqrt(T) * norminv(1-eps) * scale

The strike enters via the at-the-money sigma lookup
``blackVol(maturity, strike)`` and via ``c_point``, which switches the
helper mesh to a ``Concentrating1dMesher`` anchored at ``log(c_point[0])``
— but only when that log lies inside ``[x_min, x_max]``, exactly as the
C++ guard requires.

**Absent C++ parameters.** C++ also takes a ``DividendSchedule`` and an
``FdmQuantoHelper``, which feed the forward walk and swap the dividend
curve for a ``QuantoTermStructure`` respectively. pquantlib has neither
type yet, so neither parameter exists on this signature — a caller cannot
pass one and have it ignored.
"""

from __future__ import annotations

import math
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
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

        # C++ parity: the helper is a Concentrating1dMesher iff a critical
        # point was given *and* its log lies within the grid; otherwise
        # uniform. The range guard is C++'s, not a defensive extra — a
        # critical point outside the mesh silently falls back rather than
        # raising (fdmblackscholesmesher.cpp, the `cPoint.first !=
        # Null<Real>() && log(cPoint.first) >= xMin && <= xMax` test).
        helper: Fdm1dMesher
        c_location = c_point[0] if c_point is not None else None
        if c_location is not None and x_min <= math.log(c_location) <= x_max:
            helper = Concentrating1dMesher(
                x_min, x_max, size, (math.log(c_location), c_point[1] if c_point else None)
            )
        else:
            helper = Uniform1dMesher(x_min, x_max, size)

        # Copy locations + dplus/dminus arrays from the helper.
        self._locations = np.asarray(helper.locations(), dtype=np.float64).copy()
        for i in range(size):
            self._dplus[i] = helper.dplus(i)
            self._dminus[i] = helper.dminus(i)


__all__ = ["FdmBlackScholesMesher"]
