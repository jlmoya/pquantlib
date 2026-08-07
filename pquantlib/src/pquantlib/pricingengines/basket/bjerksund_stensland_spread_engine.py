"""BjerksundStenslandSpreadEngine — Bjerksund-Stensland (2014) spread formula.

# C++ parity:
# ql/pricingengines/basket/bjerksundstenslandspreadengine.{hpp,cpp} (v1.43),
# ``QuantLib::BjerksundStenslandSpreadEngine``.

P. Bjerksund and G. Stensland, *Closed form spread option valuation*,
Quantitative Finance 14 (2014), 1785-1794.

Derives from :class:`SpreadBlackScholesVanillaEngine`, so it implements only
the protected ``(f1, f2, strike, type, var1, var2, df)`` hook; the base class
extracts the forwards, variances and discount factor.

The formula is a three-term normal expansion about ``b = f2 / (f2 + k)``:

    stdev = sqrt(var1 + b^2 var2 - 2 rho b sigma1 sigma2)
    d1 = (ln(f1/(f2+k)) + (var1/2 + b^2 var2/2 - b rho sigma1 sigma2)) / stdev
    d2 = (ln(f1/(f2+k)) + (-var1/2 + var2 b (b/2 - 1) + rho sigma1 sigma2)) / stdev
    d3 = (ln(f1/(f2+k)) + (-var1/2 + b^2 var2/2)) / stdev
    value = df cp (f1 Phi(cp d1) - f2 Phi(cp d2) - k Phi(cp d3))

with ``cp = +1`` for a call and ``-1`` for a put. Note that the put is *not* a
separate formula: flipping ``cp`` flips every argument of Phi as well, which is
what makes put-call parity hold to round-off (pinned by the probe).
"""

from __future__ import annotations

import math

from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.basket.spread_black_scholes_vanilla_engine import (
    SpreadBlackScholesVanillaEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class BjerksundStenslandSpreadEngine(SpreadBlackScholesVanillaEngine):
    """Closed-form spread-option engine on two futures.

    # C++ parity: ``BjerksundStenslandSpreadEngine``
    # (bjerksundstenslandspreadengine.hpp:38-48).
    """

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        correlation: float,
    ) -> None:
        # C++ parity: bjerksundstenslandspreadengine.cpp:26-31.
        super().__init__(process1, process2, correlation)

    def _calculate(
        self,
        f1: float,
        f2: float,
        strike: float,
        option_type: OptionType,
        variance1: float,
        variance2: float,
        df: float,
    ) -> float:
        """# C++ parity: ``calculate`` (bjerksundstenslandspreadengine.cpp:33-58)."""
        k = strike
        cp = 1.0 if option_type == OptionType.Call else -1.0

        a = f2 + k
        b = f2 / a

        sigma1 = math.sqrt(variance1)
        sigma2 = math.sqrt(variance2)

        stdev = math.sqrt(
            variance1 + b * b * variance2 - 2 * self._rho * b * sigma1 * sigma2
        )

        lfa = math.log(f1 / a)

        d1 = (
            lfa
            + (
                0.5 * variance1
                + 0.5 * b * b * variance2
                - b * self._rho * sigma1 * sigma2
            )
        ) / stdev
        d2 = (
            lfa
            + (
                -0.5 * variance1
                + variance2 * b * (0.5 * b - 1)
                + self._rho * sigma1 * sigma2
            )
        ) / stdev
        d3 = (lfa + (-0.5 * variance1 + 0.5 * b * b * variance2)) / stdev

        phi = CumulativeNormalDistribution()
        return df * cp * (f1 * phi(cp * d1) - f2 * phi(cp * d2) - k * phi(cp * d3))


__all__ = ["BjerksundStenslandSpreadEngine"]
