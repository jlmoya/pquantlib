"""BlackScholesMertonProcess — Merton 1973 extension of BSM.

# C++ parity: ql/processes/blackscholesprocess.{hpp,cpp} (v1.42.1) —
# ``class BlackScholesMertonProcess : public GeneralizedBlackScholesProcess``.

Mathematically identical to ``GeneralizedBlackScholesProcess``; the C++
class exists primarily as an API alias / type-discriminant. The Python
port preserves the type for downstream pattern-matching.
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


class BlackScholesMertonProcess(GeneralizedBlackScholesProcess):
    """Merton 1973 process — full risk-free + dividend BSM.

    # C++ parity: trivial forward to ``GeneralizedBlackScholesProcess``.
    """

    def __init__(
        self,
        *,
        x0: Quote,
        dividend_ts: YieldTermStructure,
        risk_free_ts: YieldTermStructure,
        black_vol_ts: BlackVolTermStructure,
        discretization: StochasticProcess1DDiscretization | None = None,
        force_discretization: bool = False,
    ) -> None:
        super().__init__(
            x0=x0,
            dividend_ts=dividend_ts,
            risk_free_ts=risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
            force_discretization=force_discretization,
        )


__all__ = ["BlackScholesMertonProcess"]
