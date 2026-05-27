"""BlackScholesProcess — Black-Scholes (1973) process (no dividends).

# C++ parity: ql/processes/blackscholesprocess.{hpp,cpp} (v1.42.1) —
# ``class BlackScholesProcess : public GeneralizedBlackScholesProcess``.

Trivial specialization that injects a zero-yield FlatForward in place
of the dividend curve. All other behaviour comes from the base.
"""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1DDiscretization
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class BlackScholesProcess(GeneralizedBlackScholesProcess):
    """Black-Scholes (1973) process — no dividend yield.

    # C++ parity: the C++ constructor builds an internal ``FlatForward(0,
    # NullCalendar(), 0.0, Actual365Fixed())`` for the dividend curve;
    # Python builds the equivalent (anchored at the risk-free reference
    # date for max parity with subclasses that need a date).
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
        dividend_ts = FlatForward.from_rate(
            reference_date=risk_free_ts.reference_date(),
            forward_rate=0.0,
            day_counter=Actual365Fixed(),
        )
        super().__init__(
            x0=x0,
            dividend_ts=dividend_ts,
            risk_free_ts=risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
            force_discretization=force_discretization,
        )


__all__ = ["BlackScholesProcess"]
