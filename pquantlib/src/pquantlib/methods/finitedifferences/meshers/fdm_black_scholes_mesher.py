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

**Dividends and quanto.** C++ also takes a ``DividendSchedule`` and an
``FdmQuantoHelper``. The schedule adds one ``(time, amount)`` point per
dividend inside ``[0, maturity]`` to the forward walk — so it moves both
the ``mi``/``ma`` envelope *and* the grid bounds — and the quanto helper
swaps ``process->dividendYield()`` for a ``QuantoTermStructure``. Both are
ported; an earlier revision of this module omitted them and every caller
that passes a dividend schedule (``FdBlackScholesVanillaEngine``,
``FdBlackScholesBarrierEngine``, ``FdBlackScholesRebateEngine``,
``FdCIRVanillaEngine``, ``FdHestonVanillaEngine``) silently built a
different mesh than C++.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
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
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.yield_.quanto_term_structure import QuantoTermStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


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
        dividend_schedule: Sequence[Dividend] = (),
        quanto_helper: FdmQuantoHelper | None = None,
        spot_adjustment: float = 0.0,
    ) -> None:
        super().__init__(size)
        spot = process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")

        # C++ parity: the dividend points come first, then the uniform ones,
        # then the whole vector is sorted.
        intermediate_steps: list[tuple[float, float]] = []
        for div in dividend_schedule:
            t = process.time(div.date())
            if t <= maturity and t >= 0.0:
                intermediate_steps.append((process.time(div.date()), div.amount()))

        # Intermediate-step forward evolution. C++:
        #   intermediateTimeSteps = max(2, int(24 * maturity)).
        intermediate_time_steps: int = max(2, int(24.0 * maturity))
        intermediate_steps.extend(
            ((i + 1) * (maturity / intermediate_time_steps), 0.0)
            for i in range(intermediate_time_steps)
        )
        intermediate_steps.sort()

        rts = process.risk_free_rate()
        # C++ parity: with a quanto helper the dividend curve is replaced by a
        # ``QuantoTermStructure`` built from the process' own curves plus the
        # helper's foreign curve, FX vol, ATM level and correlation.
        qts: YieldTermStructure
        if quanto_helper is not None:
            qts = QuantoTermStructure(
                process.dividend_yield(),
                process.risk_free_rate(),
                quanto_helper.f_ts,
                process.black_volatility(),
                strike,
                quanto_helper.fx_vol_ts,
                quanto_helper.exch_rate_atm_level,
                quanto_helper.equity_fx_correlation,
            )
        else:
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
