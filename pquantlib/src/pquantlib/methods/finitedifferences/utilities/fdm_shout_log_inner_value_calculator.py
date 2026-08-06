"""FdmShoutLogInnerValueCalculator — exercise value of a shout option.

# C++ parity: ql/methods/finitedifferences/utilities/fdmshoutloginnervaluecalculator.{hpp,cpp}
# @ v1.43 (6b57206e0).

Shouting at time ``t`` locks in the intrinsic value while keeping the
optionality alive to maturity, so the exercise value is::

    fwd    = s_t * q(t, T) / d(t, T)
    stdDev = blackForwardVol(t, T, s_t) * sqrt(T - t)
    npv    = blackFormula(type, s_t, fwd, stdDev, d(t, T))     # strike == s_t
    spot   = s_t - escrowedDividendAdj.dividendAdjustment(t)
    value  = max(0, npv + intrinsic(spot) * d(t, T))

Note the Black *strike* is ``s_t`` itself, not the payoff strike — the
already-locked intrinsic is added separately, discounted.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.methods.finitedifferences.utilities.escrowed_dividend_adjustment import (
    EscrowedDividendAdjustment,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmInnerValueCalculator,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)


@final
class FdmShoutLogInnerValueCalculator(FdmInnerValueCalculator):
    """Inner value of a shout option on a log-spot grid.

    # C++ parity: ``class FdmShoutLogInnerValueCalculator : public
    # FdmInnerValueCalculator``.
    """

    __slots__ = (
        "_black_volatility",
        "_direction",
        "_escrowed_dividend_adj",
        "_maturity",
        "_mesher",
        "_payoff",
    )

    def __init__(
        self,
        black_volatility: BlackVolTermStructure,
        escrowed_dividend_adj: EscrowedDividendAdjustment,
        maturity: float,
        payoff: PlainVanillaPayoff,
        mesher: FdmMesher,
        direction: int,
    ) -> None:
        self._black_volatility: BlackVolTermStructure = black_volatility
        self._escrowed_dividend_adj: EscrowedDividendAdjustment = escrowed_dividend_adj
        self._maturity: float = maturity
        self._payoff: PlainVanillaPayoff = payoff
        self._mesher: FdmMesher = mesher
        self._direction: int = direction

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Shout-exercise value at the node.

        # C++ parity: ``FdmShoutLogInnerValueCalculator::innerValue``.
        """
        s_t = math.exp(self._mesher.location(iterator, self._direction))

        qf = self._escrowed_dividend_adj.dividend_yield().discount(
            self._maturity
        ) / self._escrowed_dividend_adj.dividend_yield().discount(t)

        df = self._escrowed_dividend_adj.risk_free_rate().discount(
            self._maturity
        ) / self._escrowed_dividend_adj.risk_free_rate().discount(t)

        fwd = s_t * qf / df
        std_dev = self._black_volatility.black_forward_vol_at_time(
            t, self._maturity, s_t
        ) * math.sqrt(self._maturity - t)

        npv = black_formula(self._payoff.option_type(), s_t, fwd, std_dev, df)

        spot = s_t - self._escrowed_dividend_adj.dividend_adjustment(t)

        intrinsic = (
            spot - self._payoff.strike()
            if self._payoff.option_type() == OptionType.Call
            else self._payoff.strike() - spot
        )

        return max(0.0, npv + intrinsic * df)

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Same as :meth:`inner_value`.

        # C++ parity: ``FdmShoutLogInnerValueCalculator::avgInnerValue``.
        """
        return self.inner_value(iterator, t)


__all__ = ["FdmShoutLogInnerValueCalculator"]
