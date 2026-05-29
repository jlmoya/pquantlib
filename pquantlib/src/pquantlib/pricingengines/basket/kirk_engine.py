"""KirkEngine — Kirk 1995 spread option formula.

# C++ parity:
# ql/pricingengines/basket/kirkengine.{hpp,cpp} (v1.42.1).

Kirk 1995 closed-form approximation for spread options on two
correlated futures, popular in energy markets.

Reference: E. Kirk, "Correlation in the Energy Markets", *Managing
Energy Price Risk*, London: Risk Publications and Enron, pp.71-78.

Formula::

    f = F1 / (F2 + K)
    sigma = sqrt(var1 + var2 * (F2/(F2+K))^2 - 2*rho*sqrt(var1*var2)*F2/(F2+K))
    Spread = (F2 + K) * BlackCalculator(payoff(1.0), f, sigma, df)

where ``payoff(1.0)`` is a plain-vanilla call/put on the rescaled
underlying with strike ``1.0``.
"""

from __future__ import annotations

import math

from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.spread_black_scholes_vanilla_engine import (
    SpreadBlackScholesVanillaEngine,
)
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class KirkEngine(SpreadBlackScholesVanillaEngine):
    """Kirk 1995 closed-form for spread options on two correlated futures.

    # C++ parity: ``KirkEngine``.
    """

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        correlation: float,
    ) -> None:
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
        """Kirk's approximation.

        # C++ parity:
        # ``KirkEngine::calculate(f1, f2, strike, optionType, var1, var2, df)``.
        """
        f = f1 / (f2 + strike)
        ratio = f2 / (f2 + strike)
        v = math.sqrt(
            variance1
            + variance2 * ratio * ratio
            - 2.0 * self._rho * math.sqrt(variance1 * variance2) * ratio
        )
        # Note: BlackCalculator uses cumulative std-dev directly.
        black = BlackCalculator(
            PlainVanillaPayoff(option_type, 1.0), f, v, df
        )
        return (f2 + strike) * black.value()


__all__ = ["KirkEngine"]
