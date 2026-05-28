"""AnalyticContinuousFloatingLookbackEngine — Conze-Viswanathan / Goldman-Sosin-Gatto.

# C++ parity:
# ql/pricingengines/lookback/analyticcontinuousfloatinglookback.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for European continuous floating-strike lookback
options under Black-Scholes dynamics. Reference: Haug "Option Pricing
Formulas" pp. 61-62 (Conze-Viswanathan 1991 / Goldman-Sosin-Gatto 1979).

The floating-strike lookback pays at maturity:
  Call:  S_T - min_{0 <= t <= T} S_t
  Put:   max_{0 <= t <= T} S_t - S_T

Both have the same closed form (with eta=+1 for Call, eta=-1 for Put):

  A(eta) = eta * [(S * div_disc * N(eta * d1)
                  - minmax * rf_disc * N(eta * (d1 - sigma*sqrt(T))))
                 + S * rf_disc / lambda * (s^(-lambda) * N(eta*(-d1 + lambda*sigma*sqrt(T)))
                                           - div_disc/rf_disc * N(eta*-d1))]

where ``s = S/minmax``, ``lambda = 2*(r-q)/sigma^2``, ``minmax`` is
the running extremum (min for Call, max for Put).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.lookback_option import (
    ContinuousFloatingLookbackOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import FloatingTypePayoff, OptionType
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticContinuousFloatingLookbackEngine(
    GenericEngine[
        ContinuousFloatingLookbackOptionArguments, OneAssetOptionResults
    ]
):
    """Conze-Viswanathan / Goldman-Sosin-Gatto closed-form engine.

    # C++ parity: ``AnalyticContinuousFloatingLookbackEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            ContinuousFloatingLookbackOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)
        self._cnd: CumulativeNormalDistribution = CumulativeNormalDistribution()

    # --- helpers (mirror C++) -------------------------------------------

    def _underlying(self) -> float:
        return self._process.x0()

    def _residual_time(self) -> float:
        ex = self._arguments.exercise
        assert ex is not None
        return self._process.time(ex.last_date())

    def _volatility(self) -> float:
        return self._process.black_volatility().black_vol_at_time(
            self._residual_time(), self._minmax(), extrapolate=True
        )

    def _std_deviation(self) -> float:
        return self._volatility() * math.sqrt(self._residual_time())

    def _risk_free_rate(self) -> float:
        return self._process.risk_free_rate().zero_rate(
            self._residual_time(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _risk_free_discount(self) -> float:
        return self._process.risk_free_rate().discount(self._residual_time())

    def _dividend_yield(self) -> float:
        return self._process.dividend_yield().zero_rate(
            self._residual_time(),
            Compounding.Continuous,
            Frequency.NoFrequency,
        ).rate()

    def _dividend_discount(self) -> float:
        return self._process.dividend_yield().discount(self._residual_time())

    def _minmax(self) -> float:
        assert self._arguments.minmax is not None
        return self._arguments.minmax

    def _term_A(self, eta: float) -> float:  # noqa: N802
        """Conze-Viswanathan ``A(eta)`` closed form."""
        vol = self._volatility()
        lam = 2.0 * (self._risk_free_rate() - self._dividend_yield()) / (vol * vol)
        s = self._underlying() / self._minmax()
        sd = self._std_deviation()
        d1 = math.log(s) / sd + 0.5 * (lam + 1.0) * sd
        n1 = self._cnd(eta * d1)
        n2 = self._cnd(eta * (d1 - sd))
        n3 = self._cnd(eta * (-d1 + lam * sd))
        n4 = self._cnd(eta * -d1)
        pow_s = s ** (-lam)
        return eta * (
            (
                self._underlying() * self._dividend_discount() * n1
                - self._minmax() * self._risk_free_discount() * n2
            )
            + (
                self._underlying()
                * self._risk_free_discount()
                * (pow_s * n3 - self._dividend_discount() * n4 / self._risk_free_discount())
                / lam
            )
        )

    # --- main entry point -----------------------------------------------

    def calculate(self) -> None:
        """Compute the floating-strike lookback NPV.

        # C++ parity: ``AnalyticContinuousFloatingLookbackEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, FloatingTypePayoff), "Non-floating payoff given"
        )
        assert isinstance(payoff, FloatingTypePayoff)
        qassert.require(
            self._process.x0() > 0.0, "negative or null underlying"
        )

        if payoff.option_type() == OptionType.Call:
            results.value = self._term_A(1.0)
        else:
            results.value = self._term_A(-1.0)


__all__ = ["AnalyticContinuousFloatingLookbackEngine"]
