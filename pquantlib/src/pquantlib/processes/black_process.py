"""BlackProcess — Black 1976 process for forwards / futures.

# C++ parity: ql/processes/blackscholesprocess.{hpp,cpp} (v1.42.1) —
# ``class BlackProcess : public GeneralizedBlackScholesProcess``.

The Black 76 process applies to a forward or futures contract; the
drift is purely ``-sigma^2/2`` because the cost-of-carry of a futures
contract under the futures measure is zero. C++ achieves this by
passing the same yield curve twice (once as risk-free, once as
dividend), so ``r - q`` cancels.
"""

from __future__ import annotations

from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1DDiscretization
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class BlackProcess(GeneralizedBlackScholesProcess):
    """Black 1976 process for forwards / futures.

    # C++ parity: ``BlackProcess::BlackProcess`` — the same risk-free
    # curve is passed as both the risk-free AND the dividend curve so
    # ``r - q = 0`` in the drift.
    """

    def __init__(
        self,
        *,
        x0: Quote,
        risk_free_ts: YieldTermStructure,
        black_vol_ts: BlackVolTermStructure,
        discretization: StochasticProcess1DDiscretization | None = None,
        force_discretization: bool = False,
    ) -> None:
        super().__init__(
            x0=x0,
            dividend_ts=risk_free_ts,  # same curve → r-q cancels.
            risk_free_ts=risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
            force_discretization=force_discretization,
        )


__all__ = ["BlackProcess"]
