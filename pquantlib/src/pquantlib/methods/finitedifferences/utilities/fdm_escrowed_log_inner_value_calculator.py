"""FdmEscrowedLogInnerValueCalculator — payoff under the escrowed-dividend model.

# C++ parity: ql/methods/finitedifferences/utilities/fdmescrowedloginnervaluecalculator.{hpp,cpp}
# @ v1.43 (6b57206e0).

The FD variable is ``log(s_t)`` where ``s_t`` is the *escrowed* spot; the
observable spot adds back the present value of the dividends still due::

    spot = exp(location) - escrowedDividendAdj.dividendAdjustment(t)

(the adjustment is negative, so this is an addition).
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
from pquantlib.payoffs import Payoff


@final
class FdmEscrowedLogInnerValueCalculator(FdmInnerValueCalculator):
    """Inner value on a log-spot grid under the escrowed-dividend model.

    # C++ parity: ``class FdmEscrowedLogInnerValueCalculator : public
    # FdmInnerValueCalculator``.
    """

    __slots__ = ("_direction", "_escrowed_dividend_adj", "_mesher", "_payoff")

    def __init__(
        self,
        escrowed_dividend_adj: EscrowedDividendAdjustment,
        payoff: Payoff,
        mesher: FdmMesher,
        direction: int,
    ) -> None:
        self._escrowed_dividend_adj: EscrowedDividendAdjustment = escrowed_dividend_adj
        self._payoff: Payoff = payoff
        self._mesher: FdmMesher = mesher
        self._direction: int = direction

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Payoff at the dividend-adjusted spot.

        # C++ parity: ``FdmEscrowedLogInnerValueCalculator::innerValue``.
        """
        s_t = math.exp(self._mesher.location(iterator, self._direction))
        spot = s_t - self._escrowed_dividend_adj.dividend_adjustment(t)

        return self._payoff(spot)

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Same as :meth:`inner_value`.

        # C++ parity: ``FdmEscrowedLogInnerValueCalculator::avgInnerValue``.
        """
        return self.inner_value(iterator, t)


__all__ = ["FdmEscrowedLogInnerValueCalculator"]
