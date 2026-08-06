"""GarmanKohlagenProcess — Garman-Kohlhagen (1983) FX process.

# C++ parity: ql/processes/blackscholesprocess.{hpp,cpp} (v1.43) —
# ``class GarmanKohlagenProcess : public GeneralizedBlackScholesProcess``
# (blackscholesprocess.hpp:192-204, blackscholesprocess.cpp:265-273).

Describes the exchange rate ``S`` by

    d ln S(t) = (r(t) - r_f(t) - 0.5 * sigma(t, S)^2) dt + sigma dW_t

where ``r`` is the DOMESTIC risk-free rate and ``r_f`` the FOREIGN one.

The whole class is one constructor. The only thing it does — and the only
thing that can go wrong — is the argument mapping into the base:

    GeneralizedBlackScholesProcess(x0,
                                   foreignRiskFreeTS,    -> dividendTS
                                   domesticRiskFreeTS,   -> riskFreeTS
                                   blackVolTS, d, forceDiscretization)

i.e. the FOREIGN curve plays the dividend-yield role and the DOMESTIC curve
the risk-free role. Consequently ``dividend_yield()`` returns the foreign
curve and ``risk_free_rate()`` the domestic one — the inherited accessor
names keep the equity vocabulary even though the process is an FX one.
Swapping the two arguments flips the sign of the whole carry term, so the
cross-validation pins both accessors as well as the drift.

(Note the class name: C++ spells it ``GarmanKohlagenProcess``, dropping the
'h' from Garman-Kohlhagen. The Python name matches the C++ symbol exactly.)
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


class GarmanKohlagenProcess(GeneralizedBlackScholesProcess):
    """Garman-Kohlhagen (1983) exchange-rate process.

    # C++ parity: ``class GarmanKohlagenProcess : public
    # GeneralizedBlackScholesProcess``.

    # C++ parity divergence: C++ takes ``Handle<Quote>`` /
    # ``Handle<YieldTermStructure>`` / ``Handle<BlackVolTermStructure>``;
    # this port does not implement ``Handle<T>`` and threads the pointed-to
    # object directly.
    """

    def __init__(
        self,
        *,
        x0: Quote,
        foreign_risk_free_ts: YieldTermStructure,
        domestic_risk_free_ts: YieldTermStructure,
        black_vol_ts: BlackVolTermStructure,
        discretization: StochasticProcess1DDiscretization | None = None,
        force_discretization: bool = False,
    ) -> None:
        # C++ parity: blackscholesprocess.cpp:265-273 — foreign becomes the
        # dividend curve, domestic becomes the risk-free curve.
        super().__init__(
            x0=x0,
            dividend_ts=foreign_risk_free_ts,
            risk_free_ts=domestic_risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
            force_discretization=force_discretization,
        )


__all__ = ["GarmanKohlagenProcess"]
